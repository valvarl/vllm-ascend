import importlib
import math
from collections.abc import Sequence
from typing import Any

import vllm.envs as envs
import zmq
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackEncoder

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    backend_map,
    get_layerwise_data_plane,
    get_layerwise_protocol,
    validate_layerwise_runtime,
    validate_layerwise_topology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.coordinator import AscendStoreCoordinator
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    build_layerwise_cache_layout,
    build_layerwise_reuse_layout,
    get_layerwise_kv_cache_specs,
    get_layerwise_reuse_config,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    AscendStoreKVConnectorWorkerMetadata,
    KeyMetadata,
    LoadSpec,
    PoolKey,
    ReqMeta,
    RequestTracker,
    block_hash_to_str,
    get_block_hashes,
    get_group_block_size,
    get_group_cache_family,
    infer_cache_transfer_granularity,
    infer_cacheable_group_ids,
    infer_group_block_sizes,
    infer_group_cache_families,
    infer_tp_mismatch_info,
    is_kv_save_role,
    normalize_block_ids_by_group,
    partial_tail_key_suffix,
    uses_hybrid_kv_cache,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metrics import (
    AscendStoreKVConnectorStats,
)


class KVPoolScheduler:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        use_layerwise,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        self.vllm_config = vllm_config
        self.use_layerwise = use_layerwise
        self.kv_cache_config = kv_cache_config
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        backend_name = str(extra_config.get("backend", "mooncake"))
        self.backend_name = backend_name.lower()
        self.layerwise_protocol = get_layerwise_protocol(self.backend_name)
        self.layerwise_data_plane = get_layerwise_data_plane(self.layerwise_protocol)
        self.use_block_key_layerwise = self.use_layerwise and self.layerwise_data_plane == "block_key"
        self.use_layerwise_transfer = self.use_layerwise and self.layerwise_data_plane == "gva"
        validate_layerwise_topology(self.layerwise_protocol, vllm_config.parallel_config, self.use_layerwise)
        hf_text_config = getattr(vllm_config.model_config, "hf_text_config", None)
        hf_config = getattr(vllm_config.model_config, "hf_config", hf_text_config)
        self.hf_config = hf_text_config or hf_config
        self.compress_ratios = getattr(hf_text_config, "compress_ratios", None)
        if self.compress_ratios is None:
            self.compress_ratios = getattr(hf_config, "compress_ratios", None)
        kv_cache_groups = kv_cache_config.kv_cache_groups if kv_cache_config is not None else None
        self.use_hybrid = uses_hybrid_kv_cache(vllm_config.scheduler_config, kv_cache_groups) or (
            self.use_block_key_layerwise and kv_cache_groups is not None and len(kv_cache_groups) > 1
        )
        self.kv_cache_group_ids = (
            list(range(len(kv_cache_config.kv_cache_groups)))
            if kv_cache_config is not None and self.use_hybrid
            else [0]
        )
        self.kv_cache_group_families = infer_group_cache_families(kv_cache_groups, self.compress_ratios, self.hf_config)
        self.num_speculative_blocks_by_group = self._infer_mamba_groups()
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.consumer_is_to_load = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_load", False
        )
        self.consumer_is_to_put = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "consumer_is_to_put", False
        )
        self.load_async = vllm_config.kv_transfer_config.kv_connector_extra_config.get("load_async", False)
        kv_event_config = vllm_config.kv_events_config
        self.enable_kv_events = bool(kv_event_config and kv_event_config.enable_kv_cache_events)
        retention_interval = getattr(envs, "VLLM_PREFIX_CACHE_RETENTION_INTERVAL", None)
        self.retention_interval = retention_interval if isinstance(retention_interval, int) else None
        self.save_decode_cache = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "save_decode_cache", False
        )
        # request_id -> (vllm cached tokes, kvpool cached tokens)
        self.load_specs: dict[str, LoadSpec] = {}
        self.pcp_size = getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1)
        self.dcp_size = getattr(vllm_config.parallel_config, "decode_context_parallel_size", 1)

        speculative_config = getattr(vllm_config, "speculative_config", None)
        use_eagle_fn = getattr(speculative_config, "use_eagle", None)
        self.use_eagle = use_eagle_fn() is True if callable(use_eagle_fn) else False
        self.original_block_size = infer_group_block_sizes(vllm_config.cache_config.block_size, kv_cache_groups)
        self.cacheable_group_ids = infer_cacheable_group_ids(kv_cache_groups)
        cacheable_block_sizes = [self.original_block_size[i] for i in self.cacheable_group_ids]
        if self.use_layerwise and len(cacheable_block_sizes) != len(self.original_block_size):
            raise ValueError("AscendStore private KV state requires non-layerwise transfer")
        self.grouped_block_size = [block_size * self.dcp_size for block_size in self.original_block_size]
        requested_hash_block_size = vllm_config.cache_config.prefix_match_unit
        if not isinstance(requested_hash_block_size, int):
            requested_hash_block_size = None
        # W1-27 (hybrid-robustness-hunt W2): keep the raw prefix-match unit
        # (pre-dcp scaling). The tail-probe walk steps by the finest
        # configured boundary granularity: hash_block_size = pmu * dcp_size,
        # so any dcp > 1 layout would otherwise never probe
        # pmu-aligned-but-not-hbs-aligned boundaries (the save side accepts
        # any unaligned partial boundary; the probe walk is the only
        # reader-side discovery path for those artifacts).
        self.prefix_match_unit = requested_hash_block_size
        self.hash_block_size = (
            requested_hash_block_size if requested_hash_block_size is not None else min(cacheable_block_sizes)
        ) * self.dcp_size
        for group_id in self.cacheable_group_ids:
            group_block_size = self.grouped_block_size[group_id]
            assert group_block_size % self.hash_block_size == 0, "block_size must be divisible by hash_block_size"
        self._block_size = self.grouped_block_size[0]
        self.lcm_block_size = math.lcm(*(self.grouped_block_size[i] for i in self.cacheable_group_ids))
        self.cache_transfer_granularity = infer_cache_transfer_granularity(
            self.grouped_block_size, self.lcm_block_size, self.cacheable_group_ids
        )
        self.cache_coordinator = self._build_cache_coordinator()
        # request_id -> full_token_ids
        self._request_trackers: dict[str, RequestTracker] = {}
        self._finished_partial_tail_metas: dict[str, ReqMeta] = {}
        self._preempted_req_ids: set[str] = set()
        # Whether to discard partial chunks
        self._discard_partial_chunks = vllm_config.kv_transfer_config.get_from_extra_config(
            "discard_partial_chunks", True
        )
        self._unfinished_requests: dict[str, tuple[Request, list[list[int]]]] = {}
        self._loading_req_ids: set[str] = set()

        self._block_pool: BlockPool | None = None
        # D-MTX.7: the GDN repair-anchor scratch blocks held for the CURRENT
        # load-bearing step's lifetime (alloc at build_connector_meta, free at
        # update_connector_output). One row PER GDN STATE GROUP (group order):
        # the D16 runtime dump proved the GDN pools are shared tensors aliased
        # across each (4k, 4k+1, 4k+2) layer triple, so a single shared row
        # collapses every group's GET onto one address and only one group's
        # content survives (the D15 permutation, the F-H3/D-MTX.11 root). The
        # refcount discipline is this tracker alone - never a request table
        # entry, so the mamba align pre-copy can never see it as src/dst.
        self._holemiss_repair_scratch: list[KVCacheBlock] | None = None
        self.sending_event_id = 0
        # {event_id, flattened block_ids}
        self.sending_blocks: dict[int, list[int]] = {}
        # {event_id, completed_woke_count}
        self.sending_events: dict[int, int] = {}
        self._expected_worker_count = vllm_config.parallel_config.world_size

        use_mla = getattr(vllm_config.model_config, "use_mla", False)
        tp_mismatch_info = infer_tp_mismatch_info(
            self.kv_role,
            vllm_config.kv_transfer_config.kv_connector_extra_config,
            getattr(vllm_config.parallel_config, "tensor_parallel_size", 1),
            vllm_config.model_config.get_total_num_kv_heads(),
            use_mla if isinstance(use_mla, bool) else False,
            use_hybrid=self.use_hybrid,
        )
        self.tp_mismatch = tp_mismatch_info.enabled

        self.block_key_hybrid = self.use_block_key_layerwise and self.use_hybrid
        self.layerwise_keys = (
            self.layerwise_protocol.bind_layerwise_keys(
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
                model_name=vllm_config.model_config.model.split("/")[-1],
                use_hybrid=self.use_hybrid,
                grouped_block_size=self.grouped_block_size,
            )
            if self.use_layerwise and self.layerwise_protocol is not None
            else None
        )
        validate_layerwise_runtime(
            self.layerwise_protocol,
            use_hybrid=self.block_key_hybrid,
            has_recurrent_state=bool(self.num_speculative_blocks_by_group),
            tp_mismatch=self.use_block_key_layerwise and self.tp_mismatch,
        )
        self.layerwise_max_transfer_blocks = int(
            vllm_config.kv_transfer_config.kv_connector_extra_config.get("layerwise_max_transfer_blocks", 0)
        )
        backend = backend_map.get(self.backend_name)
        if backend is None:
            raise ValueError(f"Unsupported KV pool backend: {backend_name}")
        backend_path = backend.get("path")
        backend_class_name = backend.get("name")
        assert backend_path is not None and backend_class_name is not None
        backend_module = importlib.import_module(backend_path)
        backend_class = getattr(backend_module, backend_class_name)
        self.store_scheduler = backend_class.create_scheduler_client(vllm_config.parallel_config)

        model_config = vllm_config.model_config
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.pp_size = vllm_config.parallel_config.pipeline_parallel_size
        self.pp_rank = (vllm_config.parallel_config.rank // self.tp_size) % self.pp_size
        # Global layer offset for layerwise pool keys under PP (matches the
        # pool worker's pp_layer_offset).
        self.pp_layer_offset = 0
        try:
            start, _ = vllm_config.model_config.get_layers_start_end_indices(vllm_config.parallel_config)
            self.pp_layer_offset = start
        except AttributeError:
            self.pp_layer_offset = 0
        except Exception:
            self.pp_layer_offset = 0
        self.use_mla = False
        if hasattr(model_config, "use_mla") and isinstance(model_config.use_mla, bool) and model_config.use_mla:
            self.use_mla = True
        if self.use_mla:
            self.num_kv_head = 1
        else:
            self.num_kv_head = model_config.get_total_num_kv_heads()
        if self.num_kv_head < self.tp_size:
            self.put_step = self.tp_size // self.num_kv_head
        else:
            self.put_step = 1
        self.num_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.layerwise_offload = False
        if self.use_layerwise_transfer:
            extra_config = get_layerwise_reuse_config(vllm_config.kv_transfer_config)
            if kv_cache_config is not None and extra_config is not None:
                reuse_layout = build_layerwise_reuse_layout(
                    get_layerwise_kv_cache_specs(kv_cache_config),
                    self.num_layers,
                    extra_config,
                )
                if reuse_layout.layer_cache_specs:
                    self.num_layers = max(
                        self.num_layers,
                        max(reuse_layout.layer_cache_specs) + 1,
                    )
                self.layerwise_offload = reuse_layout.has_layer_reuse
            else:
                self.layerwise_offload = build_layerwise_cache_layout(
                    self.num_layers,
                    vllm_config.kv_transfer_config.kv_connector_extra_config,
                ).has_layer_reuse
        self.model_name = model_config.model.split("/")[-1]

        self.client: LookupKeyClient | None = None

    def _get_or_create_request_tracker(self, req_id: str) -> RequestTracker:
        tracker = self._request_trackers.get(req_id)
        if tracker is None:
            tracker = RequestTracker(
                req_id=req_id,
                token_len=0,
                allocated_block_ids=[],
            )
            self._request_trackers[req_id] = tracker
        return tracker

    def _generate_store_query_keys(
        self,
        block_hashes,
        include_layers: bool = False,
        kv_cache_group_id: int = 0,
    ) -> list[list[str]]:
        if include_layers and bool(self.use_hybrid):
            raise RuntimeError(
                "B7: the legacy per-layer store lookup enumerates "
                "split_layers(num_layers) physical layer keys, which can never match a "
                "block-key hybrid store - refusing a silent zero-hit. Use the block-key "
                "layerless hit check (use_block_key_layerwise)."
            )
        head_or_tp_ranks = self.tp_size // self.put_step
        cache_family = get_group_cache_family(self.kv_cache_group_families, kv_cache_group_id)
        keys_by_block = []
        for block_hash in block_hashes:
            block_keys: list[str] = []
            chunk_hash = block_hash if isinstance(block_hash, str) else block_hash.hex()
            pp_ranks = [self.pp_rank] if include_layers else range(self.pp_size)
            for dcp_rank in range(self.dcp_size):
                for head_or_tp_rank in range(head_or_tp_ranks):
                    for pp_rank in pp_ranks:
                        pool_key = PoolKey(
                            KeyMetadata(
                                self.model_name,
                                head_or_tp_rank,
                                dcp_rank,
                                pp_rank,
                                kv_cache_group_id=kv_cache_group_id,
                                cache_family=cache_family,
                            ),
                            chunk_hash,
                        )
                        if include_layers:
                            block_keys.extend(
                                layer_key.to_string()
                                for layer_key in pool_key.split_layers(self.num_layers, self.pp_layer_offset)
                            )
                        else:
                            block_keys.append(pool_key.to_string())
            keys_by_block.append(block_keys)
        return keys_by_block

    def _get_store_lookup_hit_tokens(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
        include_layers: bool = False,
    ) -> int:
        num_blocks = token_len // self._block_size
        # In layerwise mode, always query from block 0 because the remote
        # pool stores per-layer data that may not match local prefix cache.
        query_start_block = 0 if self.use_layerwise else min(num_computed_tokens // self._block_size, num_blocks)
        block_hashes_to_query = request.block_hashes[query_start_block:num_blocks]
        if not block_hashes_to_query:
            return 0

        query_keys_by_block = self._generate_store_query_keys(
            block_hashes_to_query,
            include_layers=include_layers,
        )
        query_keys = [key for block_keys in query_keys_by_block for key in block_keys]
        exists_states = self.store_scheduler.batch_is_exist(query_keys)
        if len(exists_states) != len(query_keys):
            raise RuntimeError(
                "KV pool exists check returned unexpected number of "
                f"states for request {request.request_id}: "
                f"expected={len(query_keys)}, actual={len(exists_states)}"
            )

        num_queried_hit_blocks = 0
        offset = 0
        for block_keys in query_keys_by_block:
            block_states = exists_states[offset : offset + len(block_keys)]
            offset += len(block_keys)
            if all(exists == 1 for exists in block_states):
                num_queried_hit_blocks += 1
                continue
            if any(exists == 0 for exists in block_states):
                break
            raise RuntimeError(f"KV pool exists check failed for request {request.request_id}: states={exists_states}")

        num_hit_blocks = query_start_block + num_queried_hit_blocks
        return num_hit_blocks * self._block_size

    def _build_cache_coordinator(self) -> AscendStoreCoordinator | None:
        """Build the hybrid cache-hit/mask coordinator (mirrors the worker)."""
        if self.kv_cache_config is None or not self.use_hybrid:
            return None
        return AscendStoreCoordinator(
            self.kv_cache_config.kv_cache_groups,
            scheduler_block_size=self.cache_transfer_granularity,
            hash_block_size=self.hash_block_size,
            group_block_sizes=self.grouped_block_size,
            group_cache_families=self.kv_cache_group_families,
            use_eagle=self.use_eagle,
            retention_interval=self.retention_interval,
        )

    def _make_layerwise_hit_check_keys(self, group_id: int, block_hash_hex: str) -> list[str]:
        """All-rank keys for scheduler-side hit check, built by the
        backend's protocol module.

        Single-group uses PR #11585 format; multi-group includes group_id.
        A block is a hit only when every PP stage has saved it, so the
        protocol helper enumerates all stages and head/TP ranks.
        """
        assert self.layerwise_keys is not None
        return self.layerwise_keys.make_hit_check_keys(group_id, block_hash_hex, self.tp_size // self.put_step)

    def _get_layerwise_hit_tokens(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
    ) -> int:
        self._get_or_create_request_tracker(request.request_id)
        if self.cache_coordinator is not None:
            return self._lookup_layerwise_with_coordinator(request, token_len)
        return self._lookup_layerwise_contiguous(request, token_len, num_computed_tokens)

    def _query_layerwise_block_hits(self, keys_by_block: list[list[str]]) -> list[bool]:
        """Probe backend readability and fold physical keys into block hits."""
        all_keys = [key for block_keys in keys_by_block for key in block_keys]
        if not all_keys:
            return []

        keys_per_block = max(len(block_keys) for block_keys in keys_by_block)
        batch_size = (
            self.layerwise_max_transfer_blocks * keys_per_block
            if self.layerwise_max_transfer_blocks > 0
            else len(all_keys)
        )
        readable: list[bool] = []
        for start in range(0, len(all_keys), batch_size):
            batch = all_keys[start : start + batch_size]
            batch_readable = self.store_scheduler.batch_is_readable(batch)
            if len(batch_readable) != len(batch) or any(type(state) is not bool for state in batch_readable):
                raise RuntimeError("Layerwise readability probe returned invalid results")
            readable.extend(batch_readable)

        block_hits = []
        offset = 0
        for block_keys in keys_by_block:
            block_hits.append(all(readable[offset : offset + len(block_keys)]))
            offset += len(block_keys)
        return block_hits

    def _lookup_layerwise_with_coordinator(
        self,
        request: "Request",
        token_len: int,
    ) -> int:
        """Reachability-aware hit check for hybrid models.

        Only the blocks the KV cache managers consider reachable (sliding-
        window / compressor-state tails) are queried for reachability-limited
        groups — those groups are stored sparsely by the layerwise save path —
        and the hit length is derived by the coordinator lookup shared with
        the non-layerwise path, so it matches the store-side mask semantics.
        """
        coordinator = self.cache_coordinator
        assert coordinator is not None

        def query_group_hits(
            group_id: int,
            group_block_hashes: Sequence[BlockHash | str],
            lookup_mask: Sequence[bool] | None,
        ) -> list[BlockHash]:
            keys_by_block: list[list[str]] = []
            allowed_hashes: list[BlockHash] = []
            for block_idx, block_hash in enumerate(group_block_hashes):
                if lookup_mask is not None and not (block_idx < len(lookup_mask) and lookup_mask[block_idx]):
                    continue
                keys_by_block.append(self._make_layerwise_hit_check_keys(group_id, block_hash_to_str(block_hash)))
                allowed_hashes.append(block_hash)
            block_hits = self._query_layerwise_block_hits(keys_by_block)
            return [block_hash for block_hash, hit in zip(allowed_hashes, block_hits) if hit]

        return coordinator.find_reachable_hit_tokens(
            request.block_hashes,
            token_len,
            query_group_hits,
            log_context=f"hit_check: req={request.request_id}",
        )

    def _lookup_layerwise_contiguous(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
    ) -> int:
        # In layerwise mode, always query from block 0 because the remote
        # pool stores per-layer data that may not match local prefix cache.
        num_hash_blocks = token_len // self.hash_block_size
        block_hashes_to_check = request.block_hashes[:num_hash_blocks]
        hits_per_group: list[int] = []

        for group_id in range(len(self.grouped_block_size)):
            effective_block_size = get_group_block_size(self.grouped_block_size, group_id)
            group_block_hashes = get_block_hashes(block_hashes_to_check, effective_block_size, self.hash_block_size)
            query_start_block = (
                0 if self.use_layerwise else min(num_computed_tokens // effective_block_size, len(group_block_hashes))
            )
            group_block_hashes = group_block_hashes[query_start_block:]
            # Generate all-rank keys for each block hash
            keys_by_block = [
                self._make_layerwise_hit_check_keys(group_id, block_hash_to_str(bh)) for bh in group_block_hashes
            ]
            if not keys_by_block:
                continue
            block_hits = self._query_layerwise_block_hits(keys_by_block)
            num_hit_blocks = 0
            for hit in block_hits:
                if hit:
                    num_hit_blocks += 1
                else:
                    break

            hits_per_group.append((query_start_block + num_hit_blocks) * effective_block_size)

        if not hits_per_group:
            logger.debug(
                "hit_check: req=%s token_len=%d no participating groups (all skipped)",
                request.request_id,
                token_len,
            )
            return 0
        hit_tokens = min(hits_per_group)
        logger.debug(
            "hit_check: req=%s token_len=%d hits_per_group=%s hit_tokens=%d",
            request.request_id,
            token_len,
            hits_per_group,
            hit_tokens,
        )
        return hit_tokens

    def _lookup_block_key_contiguous(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
    ) -> int:
        del num_computed_tokens
        num_hash_blocks = token_len // self.hash_block_size
        block_hashes = get_block_hashes(
            request.block_hashes[:num_hash_blocks],
            self._block_size,
            self.hash_block_size,
        )
        if not block_hashes:
            return 0
        keys_by_block = [
            self._make_layerwise_hit_check_keys(0, block_hash_to_str(block_hash)) for block_hash in block_hashes
        ]
        block_hits = self._query_layerwise_block_hits(keys_by_block)
        num_hit_blocks = 0
        for hit in block_hits:
            if not hit:
                break
            num_hit_blocks += 1
        logger.info(
            "Block-key layerwise hit check request=%s hit_blocks=%d/%d",
            request.request_id,
            num_hit_blocks,
            len(keys_by_block),
        )
        return num_hit_blocks * self._block_size

    def _get_block_key_layerwise_hit_tokens(
        self,
        request: "Request",
        token_len: int,
        num_computed_tokens: int,
    ) -> int:
        if self.block_key_hybrid:
            return self._lookup_layerwise_with_coordinator(request, token_len)
        return self._lookup_block_key_contiguous(request, token_len, num_computed_tokens)

    def _floor_to_cache_transfer_granularity(self, token_len: int) -> int:
        return token_len // self.cache_transfer_granularity * self.cache_transfer_granularity

    def _infer_mamba_groups(self) -> dict[int, int]:
        """Validate Mamba cache modes and return hybrid group scratch counts."""
        if self.kv_cache_config is None:
            return {}

        num_speculative_blocks_by_group: dict[int, int] = {}
        for group_id, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                kv_cache_spec = next(iter(kv_cache_spec.kv_cache_specs.values()))
            if not isinstance(kv_cache_spec, MambaSpec):
                continue
            if getattr(kv_cache_spec, "mamba_cache_mode", None) != "align":
                raise NotImplementedError(
                    "AscendStore hybrid linear-attention support currently requires mamba_cache_mode='align'."
                )
            if self.use_hybrid:
                num_speculative_blocks_by_group[group_id] = kv_cache_spec.num_speculative_blocks
        return num_speculative_blocks_by_group

    def _probe_partial_tail_hit(
        self,
        request: "Request",
        boundary_tokens: int,
        validated_tokens: int,
    ) -> int | None:
        """Probe the FA partial-tail key at a candidate boundary (T18 + §c).

        Returns the exact-end boundary on a hit, None otherwise. The key is
        the boundary sub-hash (the last complete hash block at the boundary)
        qualified by the boundary position — the same derivation the worker
        persists under, never the req-id last-block key.

        W1-20 (hybrid-robustness-hunt W2): a tail-key hit alone certifies
        only the boundary artifact. The claim extends the load into every
        full block in [validated_tokens, boundary_tokens) — blocks the
        floor-anchored full-block hit never verified. The chained-hash law
        says they exist (the producer that saved the tail saved the whole
        prefix under the same chain), but eviction breaks that implication
        silently, so the intermediate full-block keys' readability is
        verified (metadata-only) before the claim is exposed to the
        scheduler; a candidate with a missing intermediate block is
        declined and the walk continues to older boundaries instead of
        building a claim whose load fails in the engine.
        """
        hash_block_size = self.hash_block_size
        hash_index = boundary_tokens // hash_block_size - 1
        if hash_index < 0 or hash_index >= len(request.block_hashes):
            return None
        block_size = get_group_block_size(self.grouped_block_size, 0)
        if boundary_tokens % block_size == 0:
            return None
        suffix = partial_tail_key_suffix(block_hash_to_str(request.block_hashes[hash_index]), boundary_tokens)
        keys = self._make_layerwise_hit_check_keys(0, suffix)
        if not keys:
            return None
        hits = self._query_layerwise_block_hits([keys])
        if not (hits and all(hits)):
            return None
        missing = self._unreadable_intermediate_blocks(request, boundary_tokens, validated_tokens)
        if missing:
            logger.info(
                "KV pool partial tail declined req=%s boundary=%d reason=intermediate_block_missing missing=%s",
                request.request_id,
                boundary_tokens,
                missing,
            )
            return None
        return boundary_tokens

    def _unreadable_intermediate_blocks(
        self,
        request: "Request",
        boundary_tokens: int,
        validated_tokens: int,
    ) -> list[tuple[int, int]]:
        """W1-20: the (group, block) pairs in the tail claim's unverified
        span whose full-block keys are not readable.

        The span is [validated_tokens, boundary_tokens) — full blocks the
        floor-anchored hit already verified are skipped, matching the
        worker's fetch start (``vllm_cached_tokens // block_size``). The
        per-group iteration mirrors the worker's layerwise load exactly:
        hybrid groups consult the same reachability load mask the worker
        will apply at the claimed boundary (only masked-in blocks are
        fetched, so only masked-in blocks are verified); single-group text
        layouts check the FA group densely (the text fetch is unmasked).
        """
        missing: list[tuple[int, int]] = []
        spans: list[tuple[int, int, int, int]] = []
        for group_id in self.cacheable_group_ids:
            group_block_size = get_group_block_size(self.grouped_block_size, group_id)
            start_block = validated_tokens // group_block_size
            end_block = boundary_tokens // group_block_size
            if end_block > start_block:
                spans.append((group_id, group_block_size, start_block, end_block))
        if not spans:
            # The dominant live shape: the boundary within one block of the
            # validated floor — the claim adds only the tail artifact, so
            # there is nothing intermediate to verify.
            return missing
        load_masks: Sequence[Sequence[bool] | None] | None = None
        if self.block_key_hybrid:
            coordinator = self.cache_coordinator
            assert coordinator is not None, "Block-key hybrid tail probe requires a reachability coordinator"
            load_masks = coordinator.load_mask(request.block_hashes, boundary_tokens)
        for group_id, group_block_size, start_block, end_block in spans:
            group_block_hashes = get_block_hashes(request.block_hashes, group_block_size, self.hash_block_size)
            group_mask = load_masks[group_id] if load_masks is not None else None
            keys_by_block: list[list[str]] = []
            block_indices: list[int] = []
            for block_index in range(start_block, min(end_block, len(group_block_hashes))):
                if group_mask is not None and not (block_index < len(group_mask) and group_mask[block_index]):
                    continue
                keys_by_block.append(
                    self._make_layerwise_hit_check_keys(group_id, block_hash_to_str(group_block_hashes[block_index]))
                )
                block_indices.append(block_index)
            if not keys_by_block:
                continue
            hits = self._query_layerwise_block_hits(keys_by_block)
            missing.extend((group_id, block_index) for block_index, hit in zip(block_indices, hits) if not hit)
        return missing

    def _partial_tail_probe_boundaries(self, prompt_token_len: int, token_len: int) -> list[int]:
        """Tail-probe candidate boundaries, NEWEST-first (design §c).

        The prompt's exact end is always the newest candidate (the T18
        single-probe call site — text prompt ends are not pmu-aligned in
        general, and the text extension must keep working byte-identically);
        the older candidates are the step-aligned boundaries in
        (token_len, prompt_token_len]. The step is the finest configured
        boundary granularity — prefix_match_unit when it is finer than
        hash_block_size (W1-27: hash_block_size = pmu * dcp_size, so a
        dcp > 1 layout would otherwise leave pmu-aligned-but-not-hbs-aligned
        tail artifacts stranded), hash_block_size itself otherwise — so the
        walk costs at most (prompt_token_len - token_len) // step + 1 cheap
        metadata probes; the quantization bounds it, never O(prompt).
        """
        hash_block_size = self.hash_block_size
        step = hash_block_size
        prefix_match_unit = getattr(self, "prefix_match_unit", None)
        if isinstance(prefix_match_unit, int) and 0 < prefix_match_unit < hash_block_size:
            step = prefix_match_unit
        candidates: list[int] = []
        if prompt_token_len % step != 0:
            candidates.append(prompt_token_len)
        newest_step_aligned = prompt_token_len // step * step
        candidates.extend(range(newest_step_aligned, token_len, -step))
        return candidates

    def register_finished_partial_tail(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        """Queue a finish-time partial tail for the next connector step (T18).

        Accepting responsibility returns True: the caller delays the
        request's block free until the connector reports the save finished,
        which keeps the CoW'd tail blocks readable while the offload-only
        ReqMeta (consumed at the next build_connector_meta) persists them
        under the cross-request boundary sub-hash key.
        """
        if self.kv_role == "kv_consumer" or not partial_tail_offloads:
            return False
        if not self.use_layerwise or not is_kv_save_role(self.kv_role, self.consumer_is_to_put):
            return False
        tracker = self._request_trackers.get(request.request_id)
        if tracker is None or not any(block_ids):
            return False
        boundaries = {boundary for _, _, boundary in partial_tail_offloads}
        if len(boundaries) != 1:
            raise ValueError(f"Partial-tail offloads for one request must share a boundary: {sorted(boundaries)}")
        boundary_tokens = next(iter(boundaries))
        if boundary_tokens <= 0 or boundary_tokens > tracker.token_len:
            return False
        # The offered (align/boundary-state) blocks carry the shared boundary;
        # the GDN tail state composes with the existing boundary-state channel
        # (the T14 ladder) — only the FA partial block is a NEW tail artifact,
        # taken from the caller's FA block table.
        if any(block_id == 0 for _, block_id, _ in partial_tail_offloads):
            return False
        block_size = get_group_block_size(self.grouped_block_size, 0)
        if boundary_tokens % block_size == 0:
            return False
        tail_block_index = boundary_tokens // block_size
        fa_blocks = block_ids[0] if block_ids and block_ids[0] else []
        if tail_block_index >= len(fa_blocks):
            return False
        if request.request_id in self._finished_partial_tail_metas:
            return False
        self._finished_partial_tail_metas[request.request_id] = ReqMeta(
            req_id=request.request_id,
            token_len_chunk=0,
            save_end_token=boundary_tokens,
            save_start_token=boundary_tokens,
            target_token_len=boundary_tokens,
            block_ids_by_group=[list(ids) for ids in block_ids],
            block_hashes=list(request.block_hashes),
            can_save=True,
            is_last_chunk=True,
            partial_block_index=tail_block_index,
            finished_tail_boundary=boundary_tokens,
            num_prompt_tokens=tracker.num_prompt_tokens,
        )
        logger.info(
            "KV pool partial tail accepted req=%s boundary=%d tail_block=%d",
            request.request_id,
            boundary_tokens,
            tail_block_index,
        )
        return True

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        if self.kv_role == "kv_consumer" and not self.consumer_is_to_load:
            return 0, False

        prompt_token_len = len(request.prompt_token_ids)
        if self.use_block_key_layerwise:
            token_len = self._floor_to_cache_transfer_granularity(prompt_token_len)
            if token_len < self.cache_transfer_granularity:
                return 0, False
            num_external_hit_tokens = self._get_block_key_layerwise_hit_tokens(request, token_len, num_computed_tokens)
        elif self.use_layerwise_transfer:
            token_len = prompt_token_len
            num_external_hit_tokens = self._get_layerwise_hit_tokens(request, token_len, num_computed_tokens)
        else:
            if self._discard_partial_chunks:
                token_len = self._floor_to_cache_transfer_granularity(prompt_token_len)
            else:
                token_len = prompt_token_len

            if token_len < self.cache_transfer_granularity:
                return 0, False

            if self.use_layerwise:
                num_external_hit_tokens = self._get_store_lookup_hit_tokens(
                    request, token_len, num_computed_tokens, include_layers=True
                )
            else:
                if num_computed_tokens >= token_len:
                    return 0, False
                if self.client is None:
                    self.client = LookupKeyClient(self.vllm_config)
                num_external_hit_tokens = self.client.lookup(
                    token_len,
                    request.block_hashes,
                    self.kv_cache_group_ids,
                    hbm_hit_tokens=num_computed_tokens,
                )

        if num_external_hit_tokens == 0:
            return 0, False

        # T18/§c interior tail probe: when the floor truncated a partial tail
        # and the full-block hit is COMPLETE at the floor (the FA-prefix
        # anchor — the full-block check already validated every group's full
        # blocks), walk the tail candidates NEWEST-first (the step-aligned
        # interior boundaries plus the prompt's exact end), probing the FA
        # tail key at each — the FIRST hit wins (the longest claim). The GDN
        # tail state is the last full boundary's state (already covered by
        # the full-block hit): nothing is checked at the tail boundary
        # except this one FA key — never an all-groups-complete-chunks
        # anchor (the #57216 defect class). W1-20: a winning candidate also
        # verifies the intermediate full blocks the claim extends the load
        # into (see _probe_partial_tail_hit); the verification span starts
        # at the local-computed boundary when it exceeds the floor,
        # mirroring the worker's fetch start.
        tail_boundary_tokens = None
        if self.use_block_key_layerwise and prompt_token_len > token_len and num_external_hit_tokens == token_len:
            validated_tokens = max(token_len, num_computed_tokens)
            for boundary in self._partial_tail_probe_boundaries(prompt_token_len, token_len):
                probed = self._probe_partial_tail_hit(request, boundary, validated_tokens)
                if probed is None:
                    continue
                if probed == request.num_tokens and self.num_speculative_blocks_by_group:
                    # W1-24 (hybrid-robustness-hunt W2): the refusal compares
                    # a prompt-derived candidate (always <= prompt_token_len)
                    # against request.num_tokens. Those diverge legitimately
                    # on a resumed preempted request (num_tokens = prompt +
                    # generated tokens): there the refusal is structurally
                    # dead and equally unneeded — the hit can never reach
                    # num_tokens, so the -= 1 below never fires. The
                    # meeting-point pin: when the refusal DOES fire, the hit
                    # must be the prompt's exact end; anything else means the
                    # candidate grid or the comparison drifted.
                    assert probed == prompt_token_len, (
                        "hybrid full-end refusal fired off the prompt end: "
                        f"probed={probed} prompt_token_len={prompt_token_len} "
                        f"num_tokens={request.num_tokens} req={request.request_id}"
                    )
                    # The honest -1 refusal (§c, GDN layouts only): a hit at
                    # the prompt's exact end would trigger the -= 1 below, and
                    # the boundary-state tail artifact cannot serve the
                    # re-added token's composition (the structural off-by-one:
                    # the artifact holds after b-1, the -1 claim needs
                    # after b-2). Decline THIS candidate, keep walking older
                    # ones. Unconditional — no env gate; text never declines.
                    logger.info(
                        "KV pool partial tail declined req=%s boundary=%d reason=hybrid_full_end",
                        request.request_id,
                        probed,
                    )
                    continue
                tail_boundary_tokens = probed
                num_external_hit_tokens = tail_boundary_tokens
                logger.info(
                    "KV pool partial tail hit req=%s exact_end=%d (floor was %d)",
                    request.request_id,
                    tail_boundary_tokens,
                    token_len,
                )
                break

        store_skip_tokens = num_external_hit_tokens
        if self.use_eagle:
            # Keep the draft model's recomputation zone intact: the
            # generation-point hidden states must be freshly computed, and the
            # local prefix-cache path already drops its trailing block
            # (drop_eagle_block). Only trim the external hit when it reaches
            # into the prompt's final granularity block, so that (local +
            # external) never covers the last block whose KV the engine will
            # rewrite during MTP draft/verify steps. Partial hits that stop on
            # an interior block boundary carry a valid mamba state snapshot
            # at that boundary and can be loaded as-is.
            # The final granularity block starts at the lcm-aligned boundary
            # containing the last token; num_tokens - lcm_block_size equals
            # that boundary only for lcm-aligned prompts and over-trims
            # unaligned prompts whose hit stops exactly on the boundary.
            final_block_start = (request.num_tokens - 1) // self.lcm_block_size * self.lcm_block_size
            hit_reaches_final_block = num_external_hit_tokens > final_block_start
            if hit_reaches_final_block:
                num_external_hit_tokens = max(
                    num_computed_tokens,
                    num_external_hit_tokens - self.lcm_block_size,
                )
        if num_external_hit_tokens == request.num_tokens:
            num_external_hit_tokens -= 1
        if len(self.cacheable_group_ids) != len(self.original_block_size):
            # Private compressor state is rebuilt locally. Recompute the last
            # whole token page rather than resume inside a compressed block.
            num_external_hit_tokens = self._floor_to_cache_transfer_granularity(num_external_hit_tokens)

        if num_external_hit_tokens < num_computed_tokens:
            need_to_allocate = 0
        else:
            need_to_allocate = num_external_hit_tokens - num_computed_tokens

        logger.info(
            "Reqid: %s, Total tokens %d, kvpool hit tokens: %d, need to load: %d",
            request.request_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
        )

        # In layerwise mode, even when vLLM has local cached tokens, we still
        # need to load KV cache from the pool because layerwise transfer loads
        # per-layer data that may not be in HBM.
        force_layerwise_load = self.use_layerwise and store_skip_tokens > 0
        if need_to_allocate <= 0 and not force_layerwise_load:
            return 0, False

        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=num_external_hit_tokens,
            can_load=force_layerwise_load,
            kvpool_store_skip_tokens=store_skip_tokens,
            tail_boundary_tokens=tail_boundary_tokens,
        )
        logger.debug(
            "KV pool load spec created req=%s vllm_cached=%d kvpool_cached=%d "
            "need_to_allocate=%d load_async=%s use_layerwise=%s",
            request.request_id,
            num_computed_tokens,
            num_external_hit_tokens,
            need_to_allocate,
            self.load_async,
            self.use_layerwise,
        )

        return need_to_allocate, self.load_async and not self.use_layerwise

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """
        local_block_ids: list[list[int]] = [[] for _ in self.kv_cache_group_ids]
        if num_external_tokens > 0:
            local_block_ids = normalize_block_ids_by_group(blocks.get_block_ids())

        self._unfinished_requests[request.request_id] = (request, local_block_ids)
        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            logger.debug(
                "KV pool update_state_after_alloc req=%s has no load spec; num_external_tokens=%d",
                request.request_id,
                num_external_tokens,
            )
            return

        if num_external_tokens == 0:
            # No need to load anything
            load_spec = self.load_specs[request.request_id]
            self.load_specs[request.request_id].can_load = self.use_layerwise and (
                load_spec.kvpool_cached_tokens > 0 or bool(load_spec.kvpool_store_skip_tokens)
            )
            logger.debug(
                "KV pool load spec updated req=%s because num_external_tokens=0 "
                "can_load=%s vllm_cached=%d kvpool_cached=%d",
                request.request_id,
                self.load_specs[request.request_id].can_load,
                self.load_specs[request.request_id].vllm_cached_tokens,
                self.load_specs[request.request_id].kvpool_cached_tokens,
            )
            return

        assert (
            num_external_tokens > 0
            and num_external_tokens
            == self.load_specs[request.request_id].kvpool_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
        ), (
            f"Mismatch in number of tokens: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].kvpool_cached_tokens} - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens}"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True
        if self.load_async and not self.use_layerwise:
            self._loading_req_ids.add(request.request_id)
        logger.debug(
            "KV pool load spec enabled req=%s num_external_tokens=%d vllm_cached=%d kvpool_cached=%d groups=%s",
            request.request_id,
            num_external_tokens,
            self.load_specs[request.request_id].vllm_cached_tokens,
            self.load_specs[request.request_id].kvpool_cached_tokens,
            [len(blocks) for blocks in local_block_ids],
        )

    def _get_last_chunk_tokens_num(self, prompt_token_ids: list[int]) -> int:
        if self._discard_partial_chunks:
            return self._floor_to_cache_transfer_granularity(len(prompt_token_ids))
        return len(prompt_token_ids)

    def _build_req_meta(
        self,
        request_tracker: RequestTracker,
        block_hashes,
        load_spec: LoadSpec | None,
        prompt_token_ids: list[int],
        skip_save: bool | None,
    ):
        last_chunk_tokens_num = self._get_last_chunk_tokens_num(prompt_token_ids)
        return ReqMeta.from_request_tracker(
            request_tracker,
            self.cache_transfer_granularity,
            load_spec=load_spec,
            skip_save=skip_save,
            block_hashes=block_hashes,
            is_last_chunk=request_tracker.token_len >= last_chunk_tokens_num,
            discard_partial_chunks=self._discard_partial_chunks,
            original_block_size=self.original_block_size,
            kv_cache_group_families=self.kv_cache_group_families,
            save_partial_block=self.layerwise_offload,
            hash_block_size=self.hash_block_size,
            state_tail_capture_keepalive=self._state_tail_capture_keepalive(request_tracker, load_spec),
        )

    def _state_tail_capture_keepalive(self, request_tracker: RequestTracker, load_spec: LoadSpec | None) -> bool:
        """hybrid-gdn-tail-state T7 fix: keep the ReqMeta alive at the step
        the state-tail snapshot capture freezes at.

        A pure-save request's unaligned prompt-end last chunk produces no new
        full block and no load, so ``from_request_tracker`` drops its ReqMeta
        (skip_save, load_spec None) — the step's connector metadata then
        carries no request and the worker's final-layer capture finds an empty
        window (the T7 live root cause: PUT_SKIPPED no_frozen_snapshot). The
        keep-alive mirrors the load-bearing survival: the ReqMeta persists
        with ``can_save=False`` (a verified save-side no-op) exactly when the
        capture's own guard would qualify it — the step end == the prompt end
        (``target_token_len == num_prompt_tokens``), the end unaligned vs a
        GDN state group's block, and GDN state groups exist. Aligned ends,
        mid-prompt chunks, decode steps, text layouts, and load-bearing steps
        are byte-identical (the load path already survives the drop).
        """
        if load_spec is not None:
            return False
        num_prompt = request_tracker.num_prompt_tokens
        if not num_prompt or request_tracker.token_len != num_prompt:
            return False
        if not self.num_speculative_blocks_by_group:
            return False
        return any(
            num_prompt % get_group_block_size(self.grouped_block_size, group_id) != 0
            for group_id in self.num_speculative_blocks_by_group
        )

    def _process_new_request(
        self,
        request: Request,
        scheduler_output: SchedulerOutput,
        force_skip_save: bool,
    ) -> ReqMeta | None:
        load_spec = self.load_specs.pop(request.req_id, None)
        num_tokens_to_compute = request.num_computed_tokens + scheduler_output.num_scheduled_tokens[request.req_id]
        request_tuple = self._unfinished_requests.get(request.req_id)
        if request_tuple is None:
            raise ValueError(
                f"Request {request.req_id} is not in _unfinished_requests, but it is scheduled as a new request"
            )
        request_real = request_tuple[0]
        block_ids_by_group = normalize_block_ids_by_group(request.block_ids)
        previous_tracker = self._request_trackers.get(request.req_id)
        request_tracker = RequestTracker(
            req_id=request.req_id,
            token_len=num_tokens_to_compute,
            allocated_block_ids_by_group=block_ids_by_group,
            num_saved_tokens=0,
            token_ids=(request.prompt_token_ids[:num_tokens_to_compute].copy() if self.enable_kv_events else None),
            num_prompt_tokens=len(request.prompt_token_ids),
            block_gvas=(previous_tracker.block_gvas.copy() if previous_tracker else []),
            gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
            num_speculative_blocks_by_group=self.num_speculative_blocks_by_group,
            block_sizes=self.grouped_block_size,
        )
        self._request_trackers[request.req_id] = request_tracker
        return self._build_req_meta(
            request_tracker,
            request_real.block_hashes,
            load_spec,
            request.prompt_token_ids,
            force_skip_save,
        )

    def _process_preempted_cached_request(
        self,
        new_block_ids,
        req_id: str,
        i: int,
        cached_reqs,
        scheduler_output: SchedulerOutput,
        force_skip_save: bool,
    ) -> ReqMeta | None:
        new_block_ids_by_group = normalize_block_ids_by_group(new_block_ids)
        self._preempted_req_ids.discard(req_id)
        load_spec = self.load_specs.pop(req_id, None)
        request_tuple = self._unfinished_requests.get(req_id)
        if request_tuple is None:
            raise ValueError(
                f"Request {req_id} is not in _unfinished_requests, but it is scheduled as a preempted cached request"
            )
        request_real = request_tuple[0]
        num_tokens_to_compute = request_real.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
        previous_tracker = self._request_trackers.get(req_id)
        request_tracker = RequestTracker(
            req_id=req_id,
            token_len=num_tokens_to_compute,
            allocated_block_ids_by_group=new_block_ids_by_group,
            num_saved_tokens=0,
            token_ids=(request_real.prompt_token_ids[:num_tokens_to_compute].copy() if self.enable_kv_events else None),
            num_prompt_tokens=len(request_real.prompt_token_ids),
            block_gvas=(previous_tracker.block_gvas.copy() if previous_tracker else []),
            gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
            num_speculative_blocks_by_group=self.num_speculative_blocks_by_group,
            block_sizes=self.grouped_block_size,
        )
        self._request_trackers[req_id] = request_tracker
        return self._build_req_meta(
            request_tracker,
            request_real.block_hashes,
            load_spec,
            request_real.prompt_token_ids,
            force_skip_save,
        )

    def _process_running_cached_request(
        self,
        new_block_ids,
        req_id: str,
        i: int,
        cached_reqs,
        scheduler_output: SchedulerOutput,
        force_skip_save: bool,
    ) -> ReqMeta | None:
        # Reused buffers must save every step; otherwise only explicit
        # save_decode_cache keeps Decode increments.
        req_tuple = self._unfinished_requests.get(req_id)
        is_decoding = req_tuple is not None and req_tuple[0].num_computed_tokens >= req_tuple[0].num_prompt_tokens
        if is_decoding and not self.save_decode_cache and not self.layerwise_offload:
            return None
        request_tracker = self._request_trackers.get(req_id)
        if request_tracker is None:
            raise ValueError(f"Request {req_id} is not in _request_trackers, but it is scheduled to be cached")
        num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
        if req_tuple:
            request = req_tuple[0]
            # The scheduler rolls this back when speculative tokens are rejected.
            num_current_tokens = request.num_computed_tokens
            new_token_ids = request.all_token_ids[num_current_tokens : num_current_tokens + num_new_tokens]
            if request_tracker.token_ids is not None and new_token_ids:
                request_tracker.token_ids.extend(new_token_ids)
            request_tracker.token_len = num_current_tokens + num_new_tokens
        else:
            raise ValueError(f"Request {req_id} is not in _unfinished_requests, but it is scheduled to be cached")
        if new_block_ids is not None:
            request_tracker.update(new_block_ids, request.num_computed_tokens)
        load_spec = None
        if self.layerwise_offload and num_current_tokens > 0:
            load_spec = LoadSpec(
                vllm_cached_tokens=num_current_tokens,
                kvpool_cached_tokens=num_current_tokens,
                can_load=True,
            )
        return self._build_req_meta(
            request_tracker,
            request.block_hashes,
            load_spec,
            request.prompt_token_ids,
            force_skip_save,
        )

    def _process_async_load_request(
        self,
        request_id: str,
        request: Request,
        block_ids: list[list[int]],
    ) -> ReqMeta | None:
        load_spec = self.load_specs.pop(request_id, None)
        if not load_spec:
            return None
        num_tokens_to_compute = load_spec.kvpool_cached_tokens
        if (num_tokens_to_compute % self._block_size != 0) and (
            num_tokens_to_compute == len(request.prompt_token_ids) - 1
        ):
            num_tokens_to_compute = num_tokens_to_compute + 1
        previous_tracker = self._request_trackers.get(request_id)
        request_tracker = RequestTracker(
            req_id=request_id,
            token_len=num_tokens_to_compute,
            allocated_block_ids_by_group=block_ids,
            num_saved_tokens=0,
            num_prompt_tokens=len(request.prompt_token_ids),
            block_gvas=(previous_tracker.block_gvas.copy() if previous_tracker else []),
            gva_block_offset=(previous_tracker.gva_block_offset if previous_tracker else 0),
            num_speculative_blocks_by_group=self.num_speculative_blocks_by_group,
            block_sizes=self.grouped_block_size,
        )
        self._request_trackers[request_id] = request_tracker
        return ReqMeta.from_request_tracker(
            request_tracker,
            self.cache_transfer_granularity,
            load_spec=load_spec,
            skip_save=None,
            block_hashes=request.block_hashes,
            discard_partial_chunks=self._discard_partial_chunks,
            original_block_size=self.original_block_size,
            kv_cache_group_families=self.kv_cache_group_families,
            hash_block_size=self.hash_block_size,
        )

    def _step_carries_layerwise_save(self, meta: AscendConnectorMetadata) -> bool:
        """Whether the worker's layerwise save path will act on this step.

        Mirrors the worker's save gating: per-layer save tasks are built for
        every metadata request whose ``can_save`` is set. A load-only role
        never qualifies (its save specs are nullified via ``force_skip_save``
        before reaching the metadata) and decode steps carry no save spec.
        """
        return self.use_layerwise and any(request.can_save for request in meta.requests)

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` and `has_sync_kv_loads` fields (the
        latter is raised on save-bearing steps; see
        `_step_carries_layerwise_save`).
        Also, calling this function will reset the state of the connector.
        """
        force_skip_save = self.kv_role == "kv_consumer" and not self.consumer_is_to_put

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._preempted_req_ids.discard(finished_req_id)
            self._loading_req_ids.discard(finished_req_id)

        for req_id in scheduler_output.preempted_req_ids:
            self._preempted_req_ids.update(scheduler_output.preempted_req_ids)
            self._request_trackers.pop(req_id, None)
            self._unfinished_requests.pop(req_id, None)
            self._loading_req_ids.discard(req_id)

        meta = AscendConnectorMetadata(
            scheduler_output.preempted_req_ids,
            self._loading_req_ids.copy(),
        )

        for request in scheduler_output.scheduled_new_reqs:
            req_meta = self._process_new_request(request, scheduler_output, force_skip_save)
            if req_meta is not None:
                self.touch_sending_mamba_blocks(req_meta)
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        if not force_skip_save:
            for i, req_id in enumerate(cached_reqs.req_ids):
                new_block_ids = cached_reqs.new_block_ids[i]
                if (
                    not new_block_ids
                    and not self.tp_mismatch
                    and not self.layerwise_offload
                    and not self.save_decode_cache
                ):
                    continue
                if req_id in self._preempted_req_ids:
                    if not new_block_ids:
                        continue
                    req_meta = self._process_preempted_cached_request(
                        new_block_ids,
                        req_id,
                        i,
                        cached_reqs,
                        scheduler_output,
                        force_skip_save,
                    )
                else:
                    req_meta = self._process_running_cached_request(
                        new_block_ids,
                        req_id,
                        i,
                        cached_reqs,
                        scheduler_output,
                        force_skip_save,
                    )
                if req_meta is not None:
                    self.touch_sending_mamba_blocks(req_meta)
                    meta.add_request(req_meta)

        # Finish-time partial tails ride the next FORWARD-BEARING connector
        # step: their requests are finished (the caller delayed their block
        # free on our acceptance), but the per-layer save submission needs
        # this step's forward — an idle step's tasks would be dropped by the
        # next step's fresh-list swap, leaving the put session open forever.
        if scheduler_output.scheduled_new_reqs or scheduler_output.scheduled_cached_reqs.req_ids:
            for queued_tail in self._finished_partial_tail_metas.values():
                self.touch_sending_mamba_blocks(queued_tail)
                meta.add_request(queued_tail)
            self._finished_partial_tail_metas.clear()

        request_ids = [req.req_id for req in scheduler_output.scheduled_new_reqs]
        for request_id, (request, block_ids) in self._unfinished_requests.items():
            if request_id not in request_ids and request_id not in cached_reqs.req_ids:
                req_meta = self._process_async_load_request(request_id, request, block_ids)
                if req_meta is not None:
                    self.touch_sending_mamba_blocks(req_meta)
                    meta.add_request(req_meta)

        if self._step_carries_layerwise_save(meta) or any(
            # hybrid-gdn-tail-state T7 fix: the keep-alive ReqMeta drives no
            # save tasks, but the state-tail snapshot capture at the final
            # layer still needs the pre-forward start_load_kv cursor reset
            # (B19's fire-into-empty-task-state class: a post-forward reset
            # leaves current_layer stale and the capture never fires).
            request.state_tail_capture_keepalive
            for request in meta.requests
        ):
            # The layerwise save drivers fire DURING the forward (attention_v1's
            # on_kv_cache_written notify and the hybrid attention-entry gate),
            # but every per-step save prerequisite — cursor reset, fresh layer
            # task lists, PutStart registration and task building — lives in
            # the worker's start_load_kv. The model-runner mixin defers that
            # call to post-forward on steps without sync loads, leaving the
            # drivers to fire into empty task state (B19); force the
            # pre-forward start the same way the upstream hisparse connector
            # forces its per-step rebind.
            scheduler_output.has_sync_kv_loads = True
        self._holemiss_attach_repair_scratch(meta)
        return meta

    def get_sending_event_id(self):
        """
        get a unique event id for a kv store request
        """
        using_id = self.sending_event_id
        # todo: reset sending_event_id, in case infinitely increasing
        self.sending_event_id += 1
        return using_id

    def touch_sending_mamba_blocks(self, req_meta: ReqMeta):
        """
        keep the reference of all non-null mamba blocks that will send to external kv store
        """
        if not self.use_hybrid or not self.num_speculative_blocks_by_group or not req_meta.can_save:
            return
        # Layerwise transfer frees mamba blocks layer by layer on its own
        # completion path (see KVCacheStoreSendingThread); bulk-touching them
        # here would double-reference the block pool and leak blocks.
        if self.use_layerwise:
            return
        using_event_id = self.get_sending_event_id()
        req_meta.event_id = using_event_id
        current_step_sending: list[int] = []
        for group_id, num_speculative_blocks in self.num_speculative_blocks_by_group.items():
            group_block_ids = req_meta.block_ids_by_group[group_id]
            # The last num_speculative_blocks slots are speculative scratch
            # blocks that the allocator relocates in place; relocation requires
            # them exclusively owned (ref_cnt == 1), so they must never be
            # pinned here (crashes allocate_new_blocks after vllm PR #51358).
            non_spec_block_ids = (
                group_block_ids[:-num_speculative_blocks] if num_speculative_blocks > 0 else group_block_ids
            )
            current_step_sending.extend([block_id for block_id in non_spec_block_ids if block_id > 0])
        logger.debug("event: %s touch blocks: %s", using_event_id, current_step_sending)
        assert self._block_pool is not None
        self._block_pool.touch([self._block_pool.blocks[block_id] for block_id in current_step_sending])
        self.sending_events[using_event_id] = 0
        self.sending_blocks[using_event_id] = current_step_sending

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        hand the connector_output, free non-null mamba blocks and so on.
        """
        # D-MTX.7: the repair-anchor scratch outlives the worker meta check
        # below — its step is over whatever the workers reported.
        self._holemiss_release_repair_scratch()
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, AscendStoreKVConnectorWorkerMetadata) or self._block_pool is None:
            return

        for event_id, count in meta.completed_events.items():
            logger.debug("event %s update with %s", event_id, count)
            total = self.sending_events.get(event_id, -1)
            if total == -1:
                logger.warning("worker reports an invalid event: %s, count %s", event_id, count)
                continue
            total = total + count
            if total >= self._expected_worker_count:
                to_free_block_ids = self.sending_blocks.pop(event_id, [])
                self.sending_events.pop(event_id, None)
                if to_free_block_ids:
                    logger.debug("event %s free blocks: %s", event_id, to_free_block_ids)
                    self._block_pool.free_blocks([self._block_pool.blocks[block_id] for block_id in to_free_block_ids])
            else:
                self.sending_events[event_id] = total

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Allow the scheduler to free blocks after synchronous saving."""
        return False, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Allow the scheduler to free all groups after synchronous saving."""
        return False, None

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        self._block_pool = gpu_block_pool

    def _holemiss_gdn_state_group_count(self) -> int:
        """The GDN state-group count (the worker's _holemiss_state_groups)."""
        if not self.use_hybrid or self.kv_cache_config is None:
            return 0
        count = 0
        for group_spec in self.kv_cache_config.kv_cache_groups:
            spec = group_spec.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = next(iter(spec.kv_cache_specs.values()))
            if isinstance(spec, MambaSpec):
                count += 1
        return count

    def _holemiss_attach_repair_scratch(self, meta: AscendConnectorMetadata) -> None:
        """D-MTX.7 + D15/D16: arm the scratch rows for the GDN repair anchor
        on hybrid load-bearing steps -- ONE ROW PER GDN STATE GROUP.

        The run3 battery proved the GET's bytes arrive only at rows the
        pool's own allocation machinery has handed out (the drain lands at
        the working slots; the never-allocated null row never receives
        them), and the worker cannot allocate (no BlockPool reference, a
        separate process under spawn+TP). So the scheduler allocates one
        block PER STATE GROUP per load-bearing step and stamps the list on
        every load-capable request's ReqMeta; the worker's
        _holemiss_fetch_gdn_repair_anchor lands each group's rollback
        boundary state at that group's own row. The D15/D16 evidence: the
        GDN pools alias across each (4k, 4k+1, 4k+2) layer triple, so the
        pre-fix single shared row collapsed all three groups' GETs onto
        one address and every non-surviving group anchored at the wrong
        layer's state. Never a request-table entry (the mamba align
        pre-copy src/dst window), never the null row; any degrade keeps
        the hook's loud zero anchor.
        """
        if not self.use_hybrid or not self.use_layerwise:
            return
        load_capable = [req for req in meta.requests if req.load_spec is not None and req.load_spec.can_load]
        if not load_capable:
            self._holemiss_release_repair_scratch()
            return
        n_slots = self._holemiss_gdn_state_group_count()
        if n_slots <= 0:
            return
        if self._holemiss_repair_scratch is None:
            if self._block_pool is None:
                logger.warning(
                    "HOLEMISS_REPAIR_SCRATCH unavailable tp=-1 reason=no_block_pool "
                    "(the GDN repair anchor degrades to the zero state)"
                )
                return
            try:
                self._holemiss_repair_scratch = self._block_pool.get_new_blocks(n_slots)
            except ValueError as exc:
                logger.warning(
                    "HOLEMISS_REPAIR_SCRATCH alloc failed reason=%s (the GDN repair anchor degrades to the zero state)",
                    str(exc)[:80],
                )
                return
            logger.info(
                "HOLEMISS_REPAIR_SCRATCH_ALLOC blocks=%s",
                [block.block_id for block in self._holemiss_repair_scratch],
            )
        slots = [block.block_id for block in self._holemiss_repair_scratch]
        if len(slots) != n_slots:
            # The geometry is static per engine; a mismatch means the tracker
            # predates this build. Refuse to arm a wrong-count list (the
            # never-guess law): release and let the next step re-allocate.
            logger.warning(
                "HOLEMISS_REPAIR_SCRATCH count mismatch held=%d need=%d "
                "(releasing; the anchor degrades to the zero state)",
                len(slots),
                n_slots,
            )
            self._holemiss_release_repair_scratch()
            return
        for req_meta in load_capable:
            req_meta.repair_scratch_slot = slots[0]
            req_meta.repair_scratch_slots = list(slots)

    def _holemiss_release_repair_scratch(self) -> None:
        """Free the step's repair-anchor scratch rows (the never-leak discipline).

        The blocks' lifetime is exactly one load-bearing step: the anchor
        fetch + repair both run inside that step's forward, so the earliest
        scheduler-side release point is this step's update_connector_output.
        A release skipped by an aborted step merely extends the hold (the
        next load-less build releases it here); the tracker is the single
        owner, so a double release is structurally impossible.
        """
        blocks = self._holemiss_repair_scratch
        if blocks is None:
            return
        self._holemiss_repair_scratch = None
        if self._block_pool is None:
            return
        try:
            self._block_pool.free_blocks(blocks)
            logger.info(
                "HOLEMISS_REPAIR_SCRATCH_FREE blocks=%s",
                [block.block_id for block in blocks],
            )
        except (KeyError, ValueError, AssertionError) as exc:
            logger.error(
                "HOLEMISS_REPAIR_SCRATCH free failed blocks=%s error=%s",
                [block.block_id for block in blocks],
                str(exc)[:120],
            )

    def update_finished_recving(self, finished_recving: set[str] | None) -> None:
        if finished_recving:
            self._loading_req_ids.difference_update(finished_recving)

    def get_stats(self) -> AscendStoreKVConnectorStats | None:
        return None


class LookupKeyClient:
    def __init__(self, vllm_config: "VllmConfig"):
        self.encoder = MsgpackEncoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REQ,  # type: ignore[attr-defined]
            bind=False,
        )

    def lookup(
        self,
        token_len: int,
        block_hashes: list[BlockHash],
        kv_cache_group_ids: list[int] | None = None,
        hbm_hit_tokens: int = 0,
    ) -> int:
        kv_cache_group_ids = kv_cache_group_ids or [0]
        hash_strs = [h.hex() for h in block_hashes]
        hash_frames = self.encoder.encode(hash_strs)
        kv_group_frames = self.encoder.encode(kv_cache_group_ids)
        all_frames = [
            token_len.to_bytes(4, byteorder="big"),
            *kv_group_frames,
            hbm_hit_tokens.to_bytes(4, byteorder="big"),
            *hash_frames,
        ]
        self.socket.send_multipart(all_frames, copy=False)
        resp = self.socket.recv()
        result = int.from_bytes(resp, "big")
        return result

    def close(self):
        self.socket.close(linger=0)


def get_zmq_rpc_path_lookup(vllm_config: "VllmConfig") -> str:
    dp_rank = vllm_config.parallel_config.data_parallel_rank
    base_url = envs.VLLM_RPC_BASE_PATH
    # Default to 0 if not configured
    rpc_port = 0
    if vllm_config is not None:
        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        if "lookup_rpc_port" in extra_config:
            rpc_port = extra_config["lookup_rpc_port"]
        elif "mooncake_rpc_port" in extra_config:
            rpc_port = extra_config["mooncake_rpc_port"]
            logger.warning(
                "It is recommended to use the lookup_rpc_port, as the mooncake_rpc_port will be removed in the future."
            )
    logger.debug("Base URL: %s, RPC Port: %s", base_url, rpc_port)
    return f"ipc://{base_url}/lookup_rpc_port_{rpc_port}_dp_rank{dp_rank}"
