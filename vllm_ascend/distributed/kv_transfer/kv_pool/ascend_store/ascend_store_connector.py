import functools
import os
import threading
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import torch
import zmq
from vllm.config import VllmConfig
from vllm.distributed.kv_events import (
    KVCacheEvent,
    KVConnectorKVEvents,
    KVEventAggregator,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.forward_context import ForwardContext
from vllm.logger import logger
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata  # type: ignore
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder
from vllm.v1.worker import mamba_utils

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    get_layerwise_data_plane,
    get_layerwise_protocol,
    validate_layerwise_topology,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendStoreKVConnectorWorkerMetadata,
    is_kv_save_role,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metrics import (
    AscendStoreKVConnectorStats,
    AscendStorePromMetrics,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
    get_zmq_rpc_path_lookup,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorHandshakeMetadata

HOLEMISS_BENCH_ROOT_ENV = "HOLEMISS_BENCH_ROOT"
_HOLEMISS_LEGACY_BENCH_ROOT = "/vllm-workspace"
_HOLEMISS_HOOK_TRIO = ("hole_recovery_hook.py", "hole_recovery_config.py", "timer.py")


@functools.lru_cache(maxsize=8)
def _holemiss_missing_hook_trio(root: str) -> tuple[str, ...]:
    hook_dir = os.path.join(root, "benchmarks", "hole_recovery")
    return tuple(name for name in _HOLEMISS_HOOK_TRIO if not os.path.isfile(os.path.join(hook_dir, name)))


def holemiss_verify_bench_root() -> str:
    """Fail-fast HOLEMISS_BENCH_ROOT resolution check for the HR hook trio.

    The vllm-side HR mixin imports benchmarks.hole_recovery.{hole_recovery_hook,
    hole_recovery_config,timer} via sys.path.insert(0, HOLEMISS_BENCH_ROOT). An
    unset/incorrect root would surface as a bare ModuleNotFoundError deep inside
    arming; this guard runs as soon as HR regions are about to flow to the mixin
    and raises a documented error instead. Returns the resolved root.
    """
    root = os.environ.get(HOLEMISS_BENCH_ROOT_ENV, "")
    if not root:
        raise RuntimeError(
            f"{HOLEMISS_BENCH_ROOT_ENV} is not set but hole-recovery regions were recorded. "
            "Set it to the bench repo root containing benchmarks/hole_recovery/ "
            f"(the mixin's legacy default {_HOLEMISS_LEGACY_BENCH_ROOT!r} does not contain that tree)."
        )
    missing = _holemiss_missing_hook_trio(root)
    if missing:
        raise RuntimeError(
            f"{HOLEMISS_BENCH_ROOT_ENV}={root!r} does not resolve the hole-recovery hook trio: missing "
            f"{list(missing)} under {os.path.join(root, 'benchmarks', 'hole_recovery')!r}."
        )
    return root


class AscendStoreKVEvents(KVConnectorKVEvents):
    def __init__(self, num_workers: int) -> None:
        self._aggregator = KVEventAggregator(num_workers)

    def add_events(self, events: list[KVCacheEvent]) -> None:
        self._aggregator.add_events(events)

    def aggregate(self) -> "AscendStoreKVEvents":
        """
        Aggregate KV events and retain only common events.
        """
        common_events = self._aggregator.get_common_events()
        self._aggregator.clear_events()
        self._aggregator.add_events(common_events)
        self._aggregator.reset_workers()
        return self

    def increment_workers(self, count: int = 1) -> None:
        self._aggregator.increment_workers(count)

    def get_all_events(self) -> list[KVCacheEvent]:
        return self._aggregator.get_all_events()

    def get_number_of_workers(self) -> int:
        return self._aggregator.get_number_of_workers()

    def clear_events(self) -> None:
        self._aggregator.clear_events()
        self._aggregator.reset_workers()

    def __repr__(self) -> str:
        return f"<AscendStoreKVEvents events={self.get_all_events()}>"


def gdn_boundary_state_persistence_default(kv_cache_config: KVCacheConfig | None) -> bool:
    """Whether align-mode GDN boundary-state persistence applies to this layout.

    Upstream #17287 saves the conv/ssm boundary state of align-mode Mamba
    (GDN) cache groups through the mooncake layerwise sessions (per-group put
    sessions + post-compute ``save_kv_layer`` payload ranges). The stack
    contract makes that save the DEFAULT for the geometries it applies to --
    hybrid cache layouts carrying at least one align-mode ``MambaSpec`` group
    -- so no explicit ``use_layerwise`` opt-in is required for them. Layouts
    without hybrid GDN groups keep the historical default (off), including
    single-group text and plain-mamba models.
    """
    if kv_cache_config is None:
        return False
    groups = getattr(kv_cache_config, "kv_cache_groups", None) or []
    if len(groups) < 2:
        return False
    for group in groups:
        spec = group.kv_cache_spec
        specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        if any(isinstance(sub, MambaSpec) and getattr(sub, "mamba_cache_mode", None) == "align" for sub in specs):
            return True
    return False


class AscendStoreConnector(KVConnectorBase_V1, SupportsHMA):
    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        """
        AscendStore requires PIECEWISE CUDA graph mode when layerwise
        operations are enabled.
        """
        return extra_config.get("use_layerwise", False)

    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role

        extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.backend_name = extra_config.get("backend", "mooncake").lower()
        explicit_use_layerwise = extra_config.get("use_layerwise", None)
        if explicit_use_layerwise is None:
            # Stack contract: align-mode GDN boundary-state persistence is
            # DEFAULT-ON for hybrid mooncake layouts (upstream #17287 saves
            # the conv/ssm boundary state through the layerwise sessions), so
            # no opt-in flag is required. An explicit True/False always wins.
            self.use_layerwise = self.backend_name == "mooncake" and gdn_boundary_state_persistence_default(
                kv_cache_config
            )
            if self.use_layerwise and vllm_config.compilation_config.cudagraph_mode.has_full_cudagraphs():
                # requires_piecewise_for_cudagraph cannot see the cache
                # layout, so the default-on path can arrive here under full
                # CUDA graphs; layerwise ops cannot be captured in them.
                raise ValueError(
                    "AscendStoreConnector defaults use_layerwise=True for hybrid "
                    "align-mode (GDN) cache layouts, but layerwise operations cannot "
                    "run under full CUDA graphs. Use piecewise cudagraphs or "
                    "--enforce-eager, or set kv_connector_extra_config "
                    "{'use_layerwise': False} to keep the legacy bulk path."
                )
        else:
            self.use_layerwise = bool(explicit_use_layerwise)
        self.consumer_is_to_put = extra_config.get("consumer_is_to_put", False)
        self.layerwise_protocol = get_layerwise_protocol(self.backend_name)
        self.layerwise_data_plane = get_layerwise_data_plane(self.layerwise_protocol)
        self.use_block_key_layerwise = self.use_layerwise and self.layerwise_data_plane == "block_key"
        validate_layerwise_topology(self.layerwise_protocol, vllm_config.parallel_config, self.use_layerwise)

        connector_name = vllm_config.kv_transfer_config.kv_connector
        if connector_name == "MooncakeConnectorStoreV1":
            logger.warning(
                "It is recommended to use the AscendStoreConnector, "
                "as the MoonCakeStoreConnector will be removed in the future."
            )

        self._kv_cache_events: AscendStoreKVEvents | None = None

        self._current_step_has_real_forward = False
        self._mamba_copy_bufs = None
        # Local layer index whose per-layer save was last driven by
        # on_kv_cache_written. Backends that notify at scatter time
        # (mla_v1/sfa_v1/dsa_v1) also call save_kv_layer at layer end;
        # the trailing call must not drive the next layer's submission.
        self._on_written_saved_through: int | None = None
        # Handle to the (V2) mamba hybrid model state while its per-layer
        # align pre-copy is deferred behind this connector's layerwise loads.
        self._mamba_state: Any = None
        self.requires_mamba_state_copy_after_layer_load = self.use_layerwise

        self.connector_scheduler: KVPoolScheduler | None = None
        self.connector_worker: KVPoolWorker | None = None

        if role == KVConnectorRole.SCHEDULER:
            assert kv_cache_config is not None
            self.connector_scheduler = KVPoolScheduler(vllm_config, self.use_layerwise, kv_cache_config)
        else:
            self.connector_worker = KVPoolWorker(
                vllm_config,
                self.use_layerwise,
                kv_cache_config,
            )
            assert self.connector_worker is not None
            if not self.use_layerwise and vllm_config.parallel_config.rank == 0:
                self.lookup_server = LookupKeyServer(self.connector_worker, vllm_config)

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def set_xfer_handshake_metadata_pp_aware(
        self,
        metadata: dict[tuple[int, int], "KVConnectorHandshakeMetadata"],
    ) -> None:
        """Ignore P/D handshake metadata because AscendStore handles PP via pool keys."""
        pass

    def get_num_new_matched_tokens(self, request: "Request", num_computed_tokens: int) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def register_finished_partial_tail(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        """Queue a finish-time partial tail (T18); True delays the block free.

        Accepting the tail keeps the CoW'd blocks readable until the
        offload-only save finishes (the caller's delay-free contract); the
        tail persists under the cross-request boundary sub-hash key, never
        the req-id last-block key.
        """
        if self.connector_scheduler is None:
            return False
        return self.connector_scheduler.register_finished_partial_tail(request, block_ids, partial_tail_offloads)

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished_all_groups(request, block_ids)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """
        Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side connectors output.
        """
        if self.connector_scheduler is not None:
            self.connector_scheduler.update_connector_output(connector_output)

        # Get the KV events
        kv_cache_events = connector_output.kv_cache_events
        if not kv_cache_events or not isinstance(kv_cache_events, AscendStoreKVEvents):
            return

        if self._kv_cache_events is None:
            self._kv_cache_events = kv_cache_events
        else:
            self._kv_cache_events.add_events(kv_cache_events.get_all_events())
            self._kv_cache_events.increment_workers(kv_cache_events.get_number_of_workers())
        return

    def take_events(self) -> Iterable["KVCacheEvent"]:
        """
        Take the KV cache events from the connector.

        Yields:
            New KV cache events since the last call.
        """
        if self._kv_cache_events is not None:
            self._kv_cache_events.aggregate()
            kv_cache_events = self._kv_cache_events.get_all_events()
            yield from kv_cache_events
            self._kv_cache_events.clear_events()
            self._kv_cache_events = None

    ############################################################
    # Worker Side Methods
    ############################################################
    def set_external_slot_release_waiter(self, waiter: Callable[[int], None]) -> bool:
        """Pure forwarder: the layerwise transfer gate is evaluated by the worker.

        The connector must not derive the gate itself — the copy here was
        dropped by #14465 while this method still read it (crashing
        MultiConnector init), and restored by #15291. Gating at the
        data-plane consumer, where the flag is already derived, makes that
        class of regression structurally impossible and supersedes the
        connector-side flag entirely.
        """
        if self.connector_worker is None:
            return False
        return self.connector_worker.set_external_slot_release_waiter(waiter)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        """Fence the previous save before this step can reuse KV blocks.

        This hook is temporarily reused for deferred KV cache save
        synchronization and will be replaced by a dedicated mechanism.
        """
        assert self.connector_worker is not None
        self.connector_worker.wait_for_previous_save()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        self._mamba_copy_bufs = None
        self._on_written_saved_through = None
        metadata = self._get_connector_metadata()
        self._current_step_has_real_forward = forward_context is not None
        logger.debug(
            "KV pool connector start_load_kv metadata_requests=%d specs=%s",
            len(metadata.requests),
            [
                (
                    request.req_id,
                    None if request.load_spec is None else request.load_spec.can_load,
                    None if request.load_spec is None else request.load_spec.vllm_cached_tokens,
                    None if request.load_spec is None else request.load_spec.kvpool_cached_tokens,
                )
                for request in metadata.requests
            ],
        )
        self.connector_worker.start_load_kv(metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self.use_layerwise:
            return
        assert self.connector_worker is not None
        if not getattr(self, "_hr_hook_active", False):
            # An armed HR repair forward must not block on native layerwise load
            # events; the mamba state copy below still runs (the #17287 contract).
            self.connector_worker.wait_for_layer_load()
        if self._mamba_copy_bufs is not None:
            mamba_utils.do_mamba_copy_block_for_layer(
                self._mamba_copy_bufs,
                layer_name,
            )
        # Mamba state copy must run AFTER this layer's load finishes,
        # otherwise the copy would race with the in-flight layerwise load and
        # read half-loaded state. The model state no-ops for non-mamba layers
        # and for layers whose copy already ran.
        if self._mamba_state is not None:
            self._mamba_state.do_mamba_copy_for_layer(layer_name)

    def prepare_mamba_state_copy(self, mamba_state_or_copy_bufs) -> bool:
        """Take over the mamba align pre-copy for this step.

        The V1 model runner passes its mamba copy buffers; each layer's copy
        is then executed from :meth:`wait_for_layer_load` via
        ``mamba_utils.do_mamba_copy_block_for_layer``. The V2 model runner
        passes its mamba hybrid model state from ``preprocess_state``; each
        layer's copy is executed from :meth:`wait_for_layer_load` right after
        that layer's KV load (conv/ssm state included) completes.
        """
        if not self.requires_mamba_state_copy_after_layer_load:
            return False
        if hasattr(mamba_state_or_copy_bufs, "do_mamba_copy_for_layer"):
            self._mamba_state = mamba_state_or_copy_bufs
        else:
            mamba_utils.prepare_mamba_copy_by_layer(mamba_state_or_copy_bufs)
            self._mamba_copy_bufs = mamba_state_or_copy_bufs
        return True

    def finish_mamba_state_copy(self) -> None:
        if self._mamba_copy_bufs is not None:
            try:
                mamba_utils.finish_mamba_copy_by_layer(self._mamba_copy_bufs)
            finally:
                self._mamba_copy_bufs = None
        self._mamba_state = None

    def on_kv_cache_written(self, layer_name: str = "") -> None:
        """Drive the per-layer layerwise save from the cache-write notification.

        Plain-GQA attention (attention_v1) notifies only this hook after
        ``reshape_and_cache`` and never calls ``save_kv_layer``, so without
        this driver the block-key put sessions only register their metadata
        (PutStart) while no payload range or session commit is ever issued
        and replicas never complete. The notification maps onto the same
        per-layer submission ``save_kv_layer`` drives (upstream #17287):
        the layer's KV is already scattered when the hook fires, so the
        submission can start at once.

        Backends that ALSO call ``save_kv_layer`` at layer end
        (mla_v1/sfa_v1/dsa_v1) are deduplicated through
        ``_on_written_saved_through``. Hybrid layouts keep their
        attention-entry gate driver and are skipped here.
        """
        if not self.use_layerwise or self.connector_worker is None:
            return
        if getattr(self.connector_worker, "block_key_hybrid", False):
            # Hybrid saves are driven by the attention compute gate
            # (pool_worker wait_for_layer_load -> _submit_attention_save)
            # plus the recurrent-layer post-compute path; a second driver
            # here would double-submit every full-attention layer.
            return
        if not is_kv_save_role(self.kv_role, self.consumer_is_to_put):
            # A load-only consumer does not publish KV.
            return
        if getattr(self, "_hr_hook_active", False):
            return
        if not self.has_connector_metadata():
            return
        worker = self.connector_worker
        if layer_name:
            worker.validate_notified_save_layer(layer_name)
        worker.save_kv_layer(self._get_connector_metadata())
        self._on_written_saved_through = worker.current_layer

    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: "AttentionMetadata", **kwargs
    ) -> None:
        if not self.use_layerwise:
            return

        if not is_kv_save_role(self.kv_role, self.consumer_is_to_put):
            # A load-only consumer does not publish KV.
            return
        if getattr(self, "_hr_hook_active", False):
            return
        assert self.connector_worker is not None
        driven_through = getattr(self, "_on_written_saved_through", None)
        if driven_through is not None and self.connector_worker.current_layer == driven_through:
            # The scatter-time notification already drove this layer's
            # submission (on_kv_cache_written); consuming the marker keeps
            # the layer-end call from submitting the next layer early.
            self._on_written_saved_through = None
            return
        self._on_written_saved_through = None
        self.connector_worker.save_kv_layer(self._get_connector_metadata())

    def wait_for_save(self):
        if not is_kv_save_role(self.kv_role, self.consumer_is_to_put):
            # Don't do save if the role is kv_consumer
            return

        if self.use_layerwise:
            return

        assert self.connector_worker is not None
        self.connector_worker.wait_for_save(self._get_connector_metadata())

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        metadata = self._get_connector_metadata()
        if self._current_step_has_real_forward:
            try:
                self.connector_worker.ensure_store_initialized()
            finally:
                self._current_step_has_real_forward = False
        done_sending, done_recving = self.connector_worker.get_finished(finished_req_ids, metadata)
        return done_sending, done_recving

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return KV block IDs that failed to load on the worker."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def get_gdn_state_load_result(self, req_id: str):
        """The request's GDN state-ladder result (G14's dynamic boundary).

        Called by the vllm-side HR mixin (hasattr-gated) when arming a hybrid
        repair: the rollback boundary is the boundary whose state ACTUALLY
        loaded, not the dense floor assumption, when the ladder degraded.
        """
        worker = getattr(self, "connector_worker", None)
        if worker is None or not hasattr(worker, "get_gdn_state_load_result"):
            return None
        return worker.get_gdn_state_load_result(req_id)

    def get_hole_recovery_regions(self) -> list:
        """Holemiss HR payload passthrough; the worker store is canonical.

        Called by the vllm-side HR mixin before forward (hasattr-gated); a
        non-empty return with result<0 arms the HR hook, so fail fast here when
        the hook trio cannot resolve from HOLEMISS_BENCH_ROOT - a documented
        error instead of a bare ModuleNotFoundError inside the mixin arming site.
        """
        worker = getattr(self, "connector_worker", None)
        if worker is None or not hasattr(worker, "get_hole_recovery_regions"):
            return []
        regions = worker.get_hole_recovery_regions()
        if regions:
            holemiss_verify_bench_root()
            if os.getenv("HOLEMISS_RUNTIME_PROFILE", "perf") == "evidence":
                logger.info("HOLEMISS get_hole_recovery_regions: %d regions", len(regions))
        return regions

    def writeback_hole_recovery_kv(self) -> list:
        """tail-writeback forward: heal the repaired blocks back to the store.

        Called by the vLLM-side HR mixin teardown (duck-typed, env-gated by
        the caller); the worker executor owns the loud best-effort contract
        and returns the per-key result summary the mixin logs.
        """
        worker = getattr(self, "connector_worker", None)
        writeback = getattr(worker, "writeback_hole_recovery_regions", None)
        if not callable(writeback):
            logger.error(
                "HOLEMISS_WRITEBACK_SKIP reason=worker_lacks_writeback type=%s",
                type(worker).__name__,
            )
            return []
        return writeback()

    def get_kv_connector_kv_cache_events(self) -> AscendStoreKVEvents | None:
        """
        Get the KV connector kv cache events collected during the last interval.
        """
        assert self.connector_worker is not None
        events = self.connector_worker.get_kv_events()
        if not events:
            return None

        ascend_store_kv_events = AscendStoreKVEvents(num_workers=1)
        ascend_store_kv_events.add_events(events)
        return ascend_store_kv_events

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        assert self.connector_scheduler is not None
        self.connector_scheduler.bind_gpu_block_pool(gpu_block_pool)

    def build_connector_worker_meta(self) -> AscendStoreKVConnectorWorkerMetadata | None:
        assert self.connector_worker is not None
        return self.connector_worker.build_connector_worker_meta()

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.connector_scheduler is not None:
            return self.connector_scheduler.get_stats()
        if self.connector_worker is not None:
            return self.connector_worker.get_stats()
        return None

    @classmethod
    def build_kv_connector_stats(cls, data: dict[str, Any] | None = None) -> KVConnectorStats:
        return AscendStoreKVConnectorStats(data=data or {})

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return AscendStorePromMetrics(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )


class LookupKeyServer:
    def __init__(
        self,
        pool_worker: KVPoolWorker,
        vllm_config: "VllmConfig",
    ):
        self.decoder = MsgpackDecoder()
        self.ctx = zmq.Context()  # type: ignore[attr-defined]
        socket_path = get_zmq_rpc_path_lookup(vllm_config)
        self.socket = make_zmq_socket(
            self.ctx,
            socket_path,
            zmq.REP,  # type: ignore[attr-defined]
            bind=True,
        )

        self.pool_worker = pool_worker
        self.running = True

        def process_request():
            while self.running:
                all_frames = self.socket.recv_multipart(copy=False)
                token_len = int.from_bytes(all_frames[0], byteorder="big")
                kv_group_ids = self.decoder.decode([all_frames[1]])
                hbm_hit_tokens = int.from_bytes(all_frames[2], byteorder="big")
                hashes_str = self.decoder.decode(all_frames[3:])
                result = self.pool_worker.lookup_scheduler(
                    token_len,
                    hashes_str,
                    kv_group_ids,
                    use_layerwise=False,
                    hbm_hit_tokens=hbm_hit_tokens,
                )
                logger.debug(
                    "KV pool lookup response token_len=%d groups=%s hit_tokens=%d",
                    token_len,
                    kv_group_ids,
                    result,
                )
                response = result.to_bytes(4, "big")
                self.socket.send(response)

        self.thread = threading.Thread(target=process_request, daemon=True)
        self.thread.start()

    def close(self):
        self.socket.close(linger=0)
