# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""T18 Stage-2: partial-tail exact-end persistence + hit (CPU E2E).

The save path (offer -> queue+delay -> the next step's meta consume -> the
boundary sub-hash persist) and the load path (the FA-prefix-anchored exact-end
probe -> the tail-key substitution) run end-to-end on the real Scheduler +
real connectors + the MemoryStore leaf: a request with a non-aligned prompt
persists its tail under the cross-request key, and a second request with the
SAME prompt hits to the exact end instead of the floor (zero re-added tail
tokens).

Offer timing note: upstream delivers the offer from ``_connector_finished``
BEFORE ``request_finished_all_groups`` (the finish window, pre-free); the
harness mirrors that order with a wrapper on the scheduler-side connector.
The live offer delivery is align-manager machinery that requires a hash
boundary strictly inside a block (the fine-grained geometry) — see the seam
map's section (b) and the Phase-2 handoff notes.
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm import SamplingParams
from vllm.config import CUDAGraphMode, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import create_model_runner_output, create_vllm_config
from tests.ut.kv_transfer.test_text_model_save_driver import (
    MemoryStore,
    aligned_tensor,
    wait_until,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_scheduler as pool_scheduler_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_layerwise import make_block_key
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
    get_layerwise_kv_cache_specs,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    block_hash_to_str,
    partial_tail_key_suffix,
)

NUM_LAYERS = 4
FULL_BLOCK = 16
HEAD_SIZE = 8
# 100 = 6 full blocks (96) + a 4-token partial tail: the non-aligned shape.
PROMPT_LEN = 6 * FULL_BLOCK + 4
FLOOR = 6 * FULL_BLOCK
TAIL_BLOCK = 6
TAIL_TOKENS = PROMPT_LEN - FLOOR


def _tail_key(config, request, boundary, head=0):
    model_name = config.model_config.model.split("/")[-1]
    hash_index = boundary // FULL_BLOCK - 1
    return make_block_key(
        model_name,
        partial_tail_key_suffix(block_hash_to_str(request.block_hashes[hash_index]), boundary),
        head,
    )


@pytest.fixture
def memory_store(monkeypatch, tmp_path):
    store = MemoryStore()
    monkeypatch.setitem(
        sys.modules,
        "mooncake.store",
        SimpleNamespace(MooncakeDistributedStore=lambda: store, ReplicateConfig=SimpleNamespace),
    )
    config_path = tmp_path / "mooncake.json"
    config_path.write_text(
        json.dumps({"protocol": "ascend", "metadata_server": "test", "master_server_address": "test"})
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0")
    monkeypatch.setenv("ASCEND_GLOBAL_RESOURCE_CONFIG", "{}")
    monkeypatch.setenv("HOLEMISS_SYNC_LAYERWISE_LOAD", "1")
    monkeypatch.setattr(pool_worker, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(pool_worker, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pool_worker, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(pool_worker, "get_decode_context_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pool_worker, "get_decode_context_model_parallel_rank", lambda: 0)
    register_all_kvcache_specs(None)
    init_none_hash(sha256)
    return store


def _build_engine():
    full = FullAttentionSpec(block_size=FULL_BLOCK, num_kv_heads=1, head_size=HEAD_SIZE, dtype=torch.float32)
    plan = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f"model.layers.{layer}.attn" for layer in range(NUM_LAYERS)], full)],
    )
    config = create_vllm_config(max_num_batched_tokens=256, block_size=FULL_BLOCK)
    config.model_config.hf_text_config.num_hidden_layers = NUM_LAYERS
    config.model_config.model_arch_config.total_num_hidden_layers = NUM_LAYERS
    config.model_config.max_model_len = 512
    config.scheduler_config.max_model_len = 512
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.cache_config.num_gpu_blocks = 128
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_connector_module_path="vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
        kv_role="kv_both",
        kv_connector_extra_config={"backend": "mooncake", "use_layerwise": True},
    )
    block_size, hash_size = resolve_kv_cache_block_sizes(plan, config)
    worker = KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, plan)
    specs = get_layerwise_kv_cache_specs(plan)
    caches = {
        name: tuple(aligned_tensor((plan.num_blocks, spec.block_size, 1, spec.head_size)) for _ in range(2))
        for name, spec in specs.items()
    }
    scheduler = Scheduler(
        config,
        plan,
        StructuredOutputManager(config),
        block_size=block_size,
        hash_block_size=hash_size,
    )
    return config, plan, worker, specs, caches, scheduler, hash_size


def _drive_request(
    worker, scheduler, req_id, tokens, max_tokens, specs, caches, offer_boundary=None, capture_first_meta=None
):
    """Run one request to completion through the real scheduler loop.

    With ``offer_boundary`` set, the finish-time tail offer is driven inside
    the finish window (the ``request_finished_all_groups`` wrapper mirrors
    ``_connector_finished``'s order: block_ids -> register_finished_partial_
    tail -> request_finished), so the accepted tail's delay keeps the blocks
    alive for the next step's persist.
    """
    request = Request(
        request_id=req_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        pooling_params=None,
        block_hasher=get_request_block_hasher(FULL_BLOCK, sha256),
    )
    scheduler.add_request(request)
    pool = worker.connector_worker
    offered = False

    original_all_groups = scheduler.connector.request_finished_all_groups
    original_finished = scheduler.connector.request_finished

    def finished_wrapper(finished_request, block_ids):
        nonlocal offered
        if (
            offer_boundary is not None
            and not offered
            and finished_request.request_id == req_id
            and finished_request.num_computed_tokens >= offer_boundary
        ):
            offered = True
            computed_ids = scheduler.kv_cache_manager.get_block_ids_for_computed_tokens(
                request_id=req_id,
                num_computed_tokens=offer_boundary,
            )
            accepted = scheduler.connector.register_finished_partial_tail(
                finished_request,
                tuple(tuple(ids) for ids in computed_ids),
                [(0, computed_ids[0][TAIL_BLOCK], offer_boundary)],
            )
            if accepted:
                # Mirror _connector_finished's `delay_free or partial_tail_delay`:
                # the accepted tail holds the offered blocks until the save
                # reports finished.
                return True, None
        if isinstance(block_ids, tuple):
            return original_all_groups(finished_request, block_ids)
        return original_finished(finished_request, block_ids)

    # Patch BOTH finish hooks: the connector takes the legacy single-group
    # path when it is not SupportsHMA (the BalanceScheduler wrapper here).
    with (
        patch.object(scheduler.connector, "request_finished_all_groups", side_effect=finished_wrapper),
        patch.object(scheduler.connector, "request_finished", side_effect=finished_wrapper),
    ):
        for _ in range(len(tokens) + max_tokens + 8):
            output = scheduler.schedule()
            meta = output.kv_connector_metadata
            if capture_first_meta is not None and not capture_first_meta:
                capture_first_meta.append(meta)
            worker.handle_preemptions(meta)
            worker.bind_connector_metadata(meta)
            scheduled = output.num_scheduled_tokens.get(req_id, 0)
            worker.start_load_kv(SimpleNamespace() if scheduled else None)
            if scheduled:
                end = request.num_computed_tokens
                start = end - scheduled
                blocks = scheduler.kv_cache_manager.get_blocks(req_id).get_block_ids()[0]
                for layer in range(NUM_LAYERS):
                    name = f"model.layers.{layer}.attn"
                    for index, bid in enumerate(blocks):
                        if not bid:
                            continue
                        block_start = index * FULL_BLOCK
                        lo = max(start, block_start) - block_start
                        hi = min(end, block_start + FULL_BLOCK) - block_start
                        if hi > lo:
                            for plane, cache in enumerate(caches[name]):
                                cache[bid][lo:hi].fill_(plane + 1)
                for layer in range(NUM_LAYERS):
                    name = f"model.layers.{layer}.attn"
                    worker.save_kv_layer(name, caches[name], None)
                wait_until(lambda: pool.kv_send_thread.request_queue.unfinished_tasks == 0, pool)
            sending, recving = worker.get_finished(output.finished_req_ids)
            result = create_model_runner_output([request] if scheduled else [])
            if scheduled:
                # One REAL generated token per scheduled step so the request
                # actually finishes (the offer fires at its _free_request);
                # an empty inner list samples nothing and stalls it.
                result.sampled_token_ids = [[777]]
            result.kv_connector_output = KVConnectorOutput(
                finished_sending=sending,
                finished_recving=recving,
                kv_connector_worker_meta=worker.build_connector_worker_meta(),
            )
            scheduler.update_from_output(output, result)
            worker.clear_connector_metadata()
            if req_id not in scheduler.requests:
                break
    assert offered == (offer_boundary is not None), "the offer must fire inside the finish window"
    assert request.num_computed_tokens >= len(tokens)
    return request


def _meta_load_spec(meta, req_id):
    for item in meta.requests:
        if item.req_id == req_id:
            return item.load_spec
    return None


class TestPartialTailSaveAndHit:
    def test_offer_queue_consume_persist_and_exact_end_hit(self, memory_store):
        """The full CPU E2E: the accepted tail -> the boundary sub-hash key ->
        the second request's exact-end hit + the tail-key load, byte-exact."""
        config, plan, worker, specs, caches, scheduler, hash_size = _build_engine()
        pool = worker.connector_worker
        worker.register_kv_caches(caches)
        tokens = list(range(PROMPT_LEN))

        request1 = _drive_request(
            worker,
            scheduler,
            "tail-prod",
            tokens,
            max_tokens=2,
            specs=specs,
            caches=caches,
            offer_boundary=PROMPT_LEN,
        )
        assert "tail-prod" in scheduler.connector.connector_scheduler._finished_partial_tail_metas

        # The queued tail rides the NEXT connector step (a filler request's):
        # its build_connector_meta consumes the tail meta and the worker
        # persists the tail key — only THEN can a later request probe it.
        filler_meta: list = []
        _drive_request(
            worker,
            scheduler,
            "filler",
            [777],
            max_tokens=1,
            specs=specs,
            caches=caches,
            capture_first_meta=filler_meta,
        )
        tail_metas = [item for item in filler_meta[0].requests if item.req_id == "tail-prod"]
        assert len(tail_metas) == 1
        assert tail_metas[0].finished_tail_boundary == PROMPT_LEN
        assert tail_metas[0].partial_block_index == TAIL_BLOCK
        assert not scheduler.connector.connector_scheduler._finished_partial_tail_metas

        scheduler.reset_prefix_cache()
        request2 = Request(
            request_id="tail-cons",
            prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_tokens=2, ignore_eos=True),
            pooling_params=None,
            block_hasher=get_request_block_hasher(FULL_BLOCK, sha256),
        )
        scheduler.add_request(request2)
        output = scheduler.schedule()
        meta = output.kv_connector_metadata
        # The exact-end hit: only ONE token is scheduled (the -1 convention on
        # the full-prompt hit) — the floor baseline would schedule TAIL_TOKENS.
        assert output.num_scheduled_tokens["tail-cons"] == 1
        load_spec = _meta_load_spec(meta, "tail-cons")
        assert load_spec is not None
        assert load_spec.tail_boundary_tokens == PROMPT_LEN
        assert load_spec.kvpool_cached_tokens == PROMPT_LEN - 1

        # Drive the step: the tail persists under the boundary sub-hash key,
        # and the consumer loads it back byte-exactly.
        worker.handle_preemptions(meta)
        worker.bind_connector_metadata(meta)
        worker.start_load_kv(SimpleNamespace())
        wait_until(lambda: pool.kv_send_thread.request_queue.unfinished_tasks == 0, pool)
        expected_key = _tail_key(config, request1, PROMPT_LEN, pool.head_or_tp_rank)
        assert expected_key in memory_store.complete, sorted(memory_store.complete)

        cons_blocks = scheduler.kv_cache_manager.get_blocks("tail-cons").get_block_ids()[0]
        tail_bid = cons_blocks[TAIL_BLOCK]
        for layer in range(NUM_LAYERS):
            name = f"model.layers.{layer}.attn"
            for plane, cache in enumerate(caches[name]):
                loaded = cache[tail_bid]
                assert torch.all(loaded[:TAIL_TOKENS] == plane + 1), (layer, plane)

    def test_absent_tail_falls_back_to_the_floor(self, memory_store):
        """The negative control: no persisted tail -> the hit stays at the
        floor (Stage-1 behavior, GT-correct, the load spec carries no tail)."""
        config, plan, worker, specs, caches, scheduler, hash_size = _build_engine()
        worker.register_kv_caches(caches)
        tokens = list(range(PROMPT_LEN))
        _drive_request(worker, scheduler, "no-tail", tokens, max_tokens=2, specs=specs, caches=caches)
        scheduler.reset_prefix_cache()
        request2 = Request(
            request_id="no-tail-cons",
            prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_tokens=2, ignore_eos=True),
            pooling_params=None,
            block_hasher=get_request_block_hasher(FULL_BLOCK, sha256),
        )
        scheduler.add_request(request2)
        output = scheduler.schedule()
        assert output.num_scheduled_tokens["no-tail-cons"] == TAIL_TOKENS, (
            "without a persisted tail the floor truncation re-adds the tail tokens"
        )
        load_spec = _meta_load_spec(output.kv_connector_metadata, "no-tail-cons")
        assert load_spec is not None
        assert load_spec.tail_boundary_tokens is None

    def test_tail_key_is_never_a_full_block_key(self, memory_store):
        """The collision-free property at hash==block: the tail key differs
        from every full-block key (the reference's bare sub-hash would not)."""
        config, plan, worker, specs, caches, scheduler, hash_size = _build_engine()
        worker.register_kv_caches(caches)
        tokens = list(range(PROMPT_LEN))
        request1 = _drive_request(worker, scheduler, "collide", tokens, max_tokens=2, specs=specs, caches=caches)
        expected_key = _tail_key(config, request1, PROMPT_LEN, worker.connector_worker.head_or_tp_rank)
        model_name = config.model_config.model.split("/")[-1]
        full_keys = {
            make_block_key(model_name, block_hash_to_str(request1.block_hashes[idx]), 0)
            for idx in range(PROMPT_LEN // FULL_BLOCK)
        }
        assert expected_key not in full_keys


class TestOfferValidation:
    def _scheduler_stub(self):
        sched = pool_scheduler_module.KVPoolScheduler.__new__(pool_scheduler_module.KVPoolScheduler)
        sched.kv_role = "kv_both"
        sched.consumer_is_to_put = False
        sched.use_layerwise = True
        sched.grouped_block_size = [FULL_BLOCK]
        sched.hash_block_size = FULL_BLOCK
        sched._request_trackers = {}
        sched._finished_partial_tail_metas = {}
        sched._request_trackers["req-T"] = SimpleNamespace(token_len=PROMPT_LEN, num_prompt_tokens=PROMPT_LEN)
        return sched

    def _request_stub(self):
        request = SimpleNamespace(request_id="req-T")
        request.block_hashes = [f"{index:064x}" for index in range(PROMPT_LEN // FULL_BLOCK + 1)]
        return request

    def test_mixed_boundaries_refuse_loud(self):
        sched = self._scheduler_stub()
        request = self._request_stub()
        with pytest.raises(ValueError, match="share a boundary"):
            sched.register_finished_partial_tail(request, ([10] * 8,), [(0, 16, PROMPT_LEN), (0, 15, 96)])

    def test_boundary_beyond_tracker_refuses(self):
        sched = self._scheduler_stub()
        request = self._request_stub()
        assert sched.register_finished_partial_tail(request, ([10] * 8,), [(0, 16, PROMPT_LEN + 16)]) is False

    def test_null_block_refuses(self):
        sched = self._scheduler_stub()
        request = self._request_stub()
        assert sched.register_finished_partial_tail(request, ([10] * 8,), [(0, 0, PROMPT_LEN)]) is False

    def test_aligned_boundary_refuses(self):
        sched = self._scheduler_stub()
        request = self._request_stub()
        assert sched.register_finished_partial_tail(request, ([10] * 8,), [(0, 6, FLOOR)]) is False

    def test_consumer_role_refuses(self):
        sched = self._scheduler_stub()
        sched.kv_role = "kv_consumer"
        request = self._request_stub()
        assert sched.register_finished_partial_tail(request, ([10] * 8,), [(0, 6, PROMPT_LEN)]) is False

    def test_hybrid_shaped_offer_is_accepted(self):
        """The real (hybrid) offers carry the boundary-state group ids — the
        consume accepts any well-formed offer (the shared boundary is the
        contract; the FA tail comes from the caller's block table)."""
        sched = self._scheduler_stub()
        request = self._request_stub()
        accepted = sched.register_finished_partial_tail(request, ([10] * 8, [20] * 8), [(1, 26, PROMPT_LEN)])
        assert accepted is True
        assert request.request_id in sched._finished_partial_tail_metas


class TestStage1NonContamination:
    def test_no_offers_leaves_the_queue_empty(self, memory_store):
        """Stage-1 non-contamination: without offers the drain is a no-op and
        the connector behaves exactly as before (the suite-wide green)."""
        config, plan, worker, specs, caches, scheduler, hash_size = _build_engine()
        worker.register_kv_caches(caches)
        _drive_request(worker, scheduler, "plain", list(range(PROMPT_LEN)), max_tokens=2, specs=specs, caches=caches)
        assert scheduler.connector.connector_scheduler._finished_partial_tail_metas == {}
        output = scheduler.schedule()
        tail_metas = [item for item in output.kv_connector_metadata.requests if item.finished_tail_boundary is not None]
        assert tail_metas == []


class Test57216Anchoring:
    def test_probe_checks_only_the_fa_tail_key(self):
        """The #57216 QA rule at source: the probe constructs exactly ONE key
        list (group 0, the FA tail key) — never an all-groups anchor at the
        tail boundary (the GDN groups have no tail-position keys)."""
        import inspect

        source = inspect.getsource(pool_scheduler_module.KVPoolScheduler._probe_partial_tail_hit)
        assert source.count("_make_layerwise_hit_check_keys(") == 1
        assert "_make_layerwise_hit_check_keys(0, suffix)" in source
        # The probe is reached ONLY behind the complete full-block hit (the
        # FA-prefix anchor), never as a standalone all-groups tail anchor.
        outer = inspect.getsource(pool_scheduler_module.KVPoolScheduler.get_num_new_matched_tokens)
        assert "num_external_hit_tokens == token_len" in outer
        assert "_probe_partial_tail_hit" in outer

    def test_stored_tail_hit_requires_only_the_fa_key(self, memory_store):
        """The behavioral discriminator: a stored tail with ONLY the FA tail
        key present (no GDN/tail-boundary group keys — a complete-chunk
        anchor would demand them and MISS) still extends the hit."""
        config, plan, worker, specs, caches, scheduler, hash_size = _build_engine()
        worker.register_kv_caches(caches)
        tokens = list(range(PROMPT_LEN))
        request1 = _drive_request(
            worker,
            scheduler,
            "qa-prod",
            tokens,
            max_tokens=2,
            specs=specs,
            caches=caches,
            offer_boundary=PROMPT_LEN,
        )
        # The tail rides the next step (a filler) before the probe.
        _drive_request(worker, scheduler, "qa-fill", [777], max_tokens=1, specs=specs, caches=caches)
        expected_key = _tail_key(config, request1, PROMPT_LEN, worker.connector_worker.head_or_tp_rank)
        assert expected_key in memory_store.complete
        # No key other than the FA tail key mentions the tail boundary.
        assert not any("_tail_" in key and key != expected_key for key in memory_store.complete)
        scheduler.reset_prefix_cache()
        request2 = Request(
            request_id="qa-cons",
            prompt_token_ids=tokens,
            sampling_params=SamplingParams(max_tokens=2, ignore_eos=True),
            pooling_params=None,
            block_hasher=get_request_block_hasher(FULL_BLOCK, sha256),
        )
        scheduler.add_request(request2)
        output = scheduler.schedule()
        assert output.num_scheduled_tokens["qa-cons"] == 1, (
            "the FA-prefix-anchored probe extends to the exact end; an "
            "all-groups-complete-chunks anchor would have missed the tail"
        )
