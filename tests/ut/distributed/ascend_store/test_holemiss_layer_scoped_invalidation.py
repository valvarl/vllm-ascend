#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""T4 lifecycle: layer-scoped invalidation for partially-delivered blocks.

V3a (T3) made the armed key's isolated call per-range: the healthy FA layers'
ranges LAND while the armed layer's range fails, so the block is only
PARTIALLY invalid. These tests pin the invalid-block lifecycle around that
delivery: the layer-scoped record (block -> failed PHYSICAL layers) beside
the unchanged whole-block surface, the healthy layers' bytes surviving every
va-side consumer between the load failure and the HR forward (the repair's
below-CL attention reads them), the exactly-one-region property the vllm
mixin's ``injected_regions[0]`` silently relies on, the HR-forward bypass of
``_check_hybrid_load_errors`` (the load-bearing guard), and the old paths'
neutrality (legacy store / text / unarmed / healthy).
"""

import os
import threading
import unittest
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    AscendStoreConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.layerwise_keys import (
    LayerwiseKeyBuilder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_layerwise import (
    make_block_key,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    LayerBatchedLoadTask,
    LayerLoadTask,
    LayerRangeReqMeta,
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    HOLEMISS_EIO_TARGET_ENV,
    KVPoolWorker,
)

# LIVE hybrid g0 geometry (N18/N19 - never text-shaped fixtures): 16 FA
# store-local layers, 2 cache entries per layer, 1572864 bytes per entry ->
# 3145728 bytes per layer extent, 50331648 bytes per block object. The
# simulated buffers carry small payloads (the UT pins the offset arithmetic
# and the delivery bookkeeping, not the byte volume).
NUM_FA_LAYERS = 16
ENTRY_BYTES = 1572864
ENTRIES_PER_LAYER = 2
LAYER_BYTES = ENTRY_BYTES * ENTRIES_PER_LAYER
RANGE_BYTES = 256
NUM_LAYERS = 64

FA_PHYSICAL = [3 + 4 * k for k in range(NUM_FA_LAYERS)]
ARMED_STORE_LAYER = 5
ARMED_PHYSICAL = FA_PHYSICAL[ARMED_STORE_LAYER]
ARMED_BLOCK_ID = 11
HEALTHY_BLOCK_IDS = [20, 21]

MODEL_NAME = "testmodel"


def _mock_layerwise_keys() -> LayerwiseKeyBuilder:
    return LayerwiseKeyBuilder(
        make_full_key=lambda group_id, block_hash, rank, stage: make_block_key(
            MODEL_NAME, block_hash, rank, pp_rank=stage
        ),
        pp_size=1,
    )


def _hybrid_kv_cache_config():
    groups = [SimpleNamespace(layer_names=[f"model.layers.{p}.self_attn" for p in FA_PHYSICAL])]
    for gdn_group in range(3):
        groups.append(
            SimpleNamespace(
                layer_names=[
                    f"model.layers.{p}.linear_attn" for p in range(64) if p not in FA_PHYSICAL and (p % 4) == gdn_group
                ]
            )
        )
    return SimpleNamespace(kv_cache_groups=groups)


def _text_kv_cache_config():
    return SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=[f"model.layers.{p}" for p in range(NUM_LAYERS)])]
    )


def _live_hybrid_geometry():
    entry_offsets = list(range(0, NUM_FA_LAYERS * ENTRIES_PER_LAYER + 1, ENTRIES_PER_LAYER))
    block_len = [ENTRY_BYTES] * (NUM_FA_LAYERS * ENTRIES_PER_LAYER)
    return {0: entry_offsets}, {0: block_len}


def _make_request():
    block_hashes = [bytes([(i + 1) % 256] * 32) for i in range(4)]
    load_spec = LoadSpec(
        vllm_cached_tokens=0,
        kvpool_cached_tokens=3 * 1536,
        can_load=True,
        kvpool_store_skip_tokens=None,
        token_len=0,
    )
    return ReqMeta(
        req_id="req-T4",
        token_len_chunk=4 * 1536,
        block_ids_by_group=[[10, ARMED_BLOCK_ID, 12, 13]],
        block_hashes=block_hashes,
        can_save=False,
        load_spec=load_spec,
    )


def _make_worker(store, use_hybrid=True, with_geometry=True):
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.put_step = 1
    worker.head_or_tp_rank = 0
    worker.pp_rank = 0
    worker.num_layers = NUM_LAYERS
    worker.block_size = 1536
    worker.grouped_block_size = [1536]
    worker.hash_block_size = 1536
    worker.num_kv_cache_groups = 4 if use_hybrid else 1
    worker.use_hybrid = use_hybrid
    worker.use_layerwise = True
    worker.use_block_key_layerwise = True
    worker.use_layerwise_transfer = False
    worker.use_eagle = False
    worker.layerwise_offload = False
    worker.group_uses_align_state = [False] * worker.num_kv_cache_groups
    worker.hf_config = SimpleNamespace(num_hidden_layers=NUM_LAYERS)
    worker.kv_cache_config = _hybrid_kv_cache_config() if use_hybrid else _text_kv_cache_config()
    worker.layerwise_keys = _mock_layerwise_keys()
    worker._invalid_block_ids = set()
    worker._invalid_block_ids_lock = threading.Lock()
    worker._hole_recovery_regions = []
    worker._hole_recovery_lock = threading.Lock()
    worker._holemiss_armed_fault = None
    worker._holemiss_fa_layer_map = None
    worker._holemiss_invalid_block_layers = {}
    worker._holemiss_state_dispositioned_blocks = set()
    worker._holemiss_scratch_owner = None
    worker._layer_load_aborted = threading.Event()
    worker.kv_send_thread = None
    worker.kv_recv_thread = None
    worker.m_store = store
    if with_geometry:
        worker.group_layer_cache_entry_offsets, worker.group_block_len = _live_hybrid_geometry()
    else:
        worker.group_layer_cache_entry_offsets = {}
        worker.group_block_len = {}
    return worker


def _arm_disk_eio(worker, store_layer=ARMED_STORE_LAYER, logical_block=1):
    request = _make_request()
    key = worker._holemiss_derive_block_key(request, logical_block)
    worker._holemiss_armed_fault = worker._holemiss_make_armed_record(
        "disk_eio", request, key, store_layer, logical_block
    )
    return key


def _delivered_bytes(buffer_id: int, size: int) -> bytes:
    return bytes((buffer_id * 31 + i) % 251 for i in range(size))


class _PoolSimStore:
    """Store double simulating the V3a per-range scatter into pool buffers.

    ``batch_copy_get`` delivers every range of every key (the healthy batch);
    ``batch_copy_get_ranges`` delivers every range EXCEPT the EIO range index
    (the armed layer's extent stays untouched - the pre-existing page content,
    which the mixin's arming-zero owns). Every call is recorded so the tests
    can assert no re-issue/zero-write touches the armed block in the window.
    """

    def __init__(self, eio_range_index=None, has_v2=True):
        self.pool: dict[int, bytes] = {}
        self.calls: list[tuple] = []
        self.eio_range_index = eio_range_index
        self.has_v2 = has_v2

    def batch_copy_get(self, keys, all_buffers, all_sizes, all_offsets):
        self.calls.append(("copy", tuple(keys)))
        results = []
        for buffers, sizes in zip(all_buffers, all_sizes, strict=True):
            for buffer_id, size in zip(buffers, sizes, strict=True):
                self.pool[buffer_id] = _delivered_bytes(buffer_id, size)
            results.append(sum(sizes) if sizes else 0)
        return results

    def batch_copy_get_ranges(self, keys, all_buffers, all_sizes, all_offsets):
        self.calls.append(("ranges", tuple(keys)))
        rows = []
        for buffers, sizes in zip(all_buffers, all_sizes, strict=True):
            row = []
            for index, (buffer_id, size) in enumerate(zip(buffers, sizes, strict=True)):
                if self.eio_range_index is not None and index == self.eio_range_index:
                    row.append(-1102)
                else:
                    self.pool[buffer_id] = _delivered_bytes(buffer_id, size)
                    row.append(size)
            rows.append(row)
        return rows

    def batch_remove_lease(self, keys):
        self.calls.append(("lease", tuple(keys)))


def _layer_req_meta(layer_index: int, armed_key: str):
    """One FA layer's merged-call metadata: per-key one range at the layer's
    object offset (the shape LayerBatchBuilder.build_addrs produces)."""
    return LayerRangeReqMeta(
        req_ids=["req-T4"],
        layer_id=layer_index,
        block_ids=[ARMED_BLOCK_ID, *HEALTHY_BLOCK_IDS],
        keys=[armed_key, f"{MODEL_NAME}@healthy0@0", f"{MODEL_NAME}@healthy1@0"],
        all_buffers=[
            [1000 + layer_index],
            [2000 + layer_index],
            [3000 + layer_index],
        ],
        all_sizes=[[RANGE_BYTES]] * 3,
        all_offsets=[[layer_index * LAYER_BYTES]] * 3,
        load_keys=[],
    )


def _transfer_task(req_meta, layer_index):
    task = SimpleNamespace()
    task.group_id = 0
    task.layer_idx_in_group = layer_index
    task.final_group_layer = False
    task.shared_block_data = SimpleNamespace(req_ids=req_meta.req_ids, is_last_chunks=[True])
    task.block_ranges = [
        SimpleNamespace(
            request=SimpleNamespace(
                block_ids_by_group=[list(req_meta.block_ids)],
                block_ids=list(req_meta.block_ids),
            ),
            start_block=0,
            end_block=len(req_meta.keys),
            partial_block_index=None,
        )
    ]
    return task


def _batched_task(armed_key):
    metas = {layer: _layer_req_meta(layer, armed_key) for layer in range(NUM_FA_LAYERS)}
    layer_tasks = [
        LayerLoadTask(wait_for_save_layer=None, transfer_tasks=[_transfer_task(metas[layer], layer)], layer_id=layer)
        for layer in range(NUM_FA_LAYERS)
    ]
    return LayerBatchedLoadTask(layer_tasks=layer_tasks), metas


def _stub_recv_thread(worker, build_map, max_transfer_blocks=0):
    thread = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
    thread.layer_load_finished_events = [threading.Event() for _ in range(NUM_LAYERS)]
    thread._load_abort_event = threading.Event()
    # The LIVE wiring (pool_worker.py:953-954): the worker's own set + lock.
    thread._invalid_block_ids = worker._invalid_block_ids
    thread._invalid_block_ids_lock = worker._invalid_block_ids_lock
    thread._fatal_error = None
    thread.request_queue = SimpleNamespace(task_done=lambda: None)
    thread.get_event = SimpleNamespace(set=lambda: None)
    thread.m_store = worker.m_store
    thread.max_transfer_blocks = max_transfer_blocks
    thread.max_transfer_bytes = 0
    thread.worker = worker
    thread.h2d_stagger_us = 0
    thread.tp_rank = 0
    thread.tp_size = 1
    thread.group_builders = None

    class _Builder:
        def build_addrs(self, shared, layer_idx):
            return build_map[layer_idx]

    thread.layer_batch_builder = _Builder()
    finished = []
    thread.set_finished_request = finished.append
    thread.finished_requests = finished
    return thread


def _drive_new_path_fault(store=None, use_hybrid=True, armed=True, max_transfer_blocks=0):
    store = store or _PoolSimStore(eio_range_index=ARMED_STORE_LAYER)
    worker = _make_worker(store, use_hybrid=use_hybrid)
    armed_key = _arm_disk_eio(worker) if armed else "m@unarmed@0"
    if not armed:
        worker._holemiss_armed_fault = None
    task, build_map = _batched_task(armed_key)
    thread = _stub_recv_thread(worker, build_map, max_transfer_blocks=max_transfer_blocks)
    thread._handle_batched_range_layer_task(task)
    return worker, thread, store


class T4TestBase(unittest.TestCase):
    def setUp(self):
        self._saved_env = {
            name: os.environ.get(name)
            for name in (HOLEMISS_EIO_TARGET_ENV, "HOLEMISS_DISK_EIO_ENABLED", "HOLEMISS_FAIL_LAYER")
        }
        os.environ.pop(HOLEMISS_EIO_TARGET_ENV, None)
        os.environ.pop("HOLEMISS_DISK_EIO_ENABLED", None)
        os.environ.pop("HOLEMISS_FAIL_LAYER", None)

    def tearDown(self):
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class TestNewPathLayerScopedInvalidation(T4TestBase):
    def test_flow_records_layer_scope_beside_the_whole_block_surface(self):
        worker, thread, _ = _drive_new_path_fault()
        # The whole-block surface is UNCHANGED: the armed block id flows to
        # the scheduler-facing set exactly as on the old path.
        self.assertEqual(worker._invalid_block_ids, {ARMED_BLOCK_ID})
        # The layer-scoped record: only the failed PHYSICAL layer.
        self.assertEqual(worker.get_holemiss_invalid_block_layers(), {ARMED_BLOCK_ID: {ARMED_PHYSICAL}})
        # The healthy layer set derives as the FA physical layers minus the
        # failed one - 15 delivered layers stay valid for the repair.
        healthy = set(FA_PHYSICAL) - worker.get_holemiss_invalid_block_layers()[ARMED_BLOCK_ID]
        self.assertEqual(healthy, set(FA_PHYSICAL) - {ARMED_PHYSICAL})
        self.assertEqual(len(healthy), NUM_FA_LAYERS - 1)
        # The GDN ladder dispositions state groups only; the armed FA block
        # stays undispositioned (the HR-forward bypass owns its survival).
        self.assertNotIn(ARMED_BLOCK_ID, worker._holemiss_state_dispositioned_blocks)
        self.assertFalse(thread._load_abort_event.is_set())

    def test_the_region_carries_the_consumed_seam(self):
        worker, _, _ = _drive_new_path_fault()
        regions = worker.get_hole_recovery_regions()
        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertLess(region["result"], 0)
        self.assertEqual(region["physical_block_id"], ARMED_BLOCK_ID)
        self.assertEqual(region["failed_layers"], [ARMED_PHYSICAL])
        self.assertEqual(region["failed_range_indices"], [ARMED_STORE_LAYER])
        self.assertEqual(region["range_rcs"][ARMED_STORE_LAYER], -1102)
        self.assertEqual([rc for rc in region["range_rcs"] if rc < 0], [-1102])
        self.assertEqual(region["layer_start"], ARMED_PHYSICAL)
        self.assertEqual(region["layer_end"], ARMED_PHYSICAL)


class TestHealthyLayerSurvival(T4TestBase):
    """The lifecycle invariant: nothing frees/zeroes/overwrites the armed
    block's healthy-layer slots between the load failure and the HR forward."""

    def test_healthy_layers_delivered_and_failed_layer_not(self):
        worker, _, store = _drive_new_path_fault()
        for layer in range(NUM_FA_LAYERS):
            buffer_id = 1000 + layer
            if layer == ARMED_STORE_LAYER:
                self.assertNotIn(buffer_id, store.pool, "the EIO range must not land")
            else:
                self.assertEqual(
                    store.pool.get(buffer_id),
                    _delivered_bytes(buffer_id, RANGE_BYTES),
                    f"healthy layer {layer} must hold its delivered bytes",
                )

    def test_bytes_survive_the_window_mutators_to_the_repair_entry(self):
        worker, _, store = _drive_new_path_fault()
        pre_window = {
            layer: store.pool.get(1000 + layer) for layer in range(NUM_FA_LAYERS) if layer != ARMED_STORE_LAYER
        }
        self.assertEqual(len(pre_window), NUM_FA_LAYERS - 1)

        # Window mutator 1: the hybrid load-error raise would fire on the
        # undispositioned armed block - pinning that ONLY the HR-forward
        # bypass keeps the fault flow alive (connector-level pin below).
        with self.assertRaises(RuntimeError):
            worker._check_hybrid_load_errors()

        # Window mutator 2: the mixin's EXIT drain (get-and-clear) empties
        # the whole-block surface; the repair-facing records survive it.
        self.assertEqual(worker.get_block_ids_with_load_errors(), {ARMED_BLOCK_ID})
        self.assertEqual(worker._invalid_block_ids, set())
        self.assertEqual(worker.get_holemiss_invalid_block_layers(), {ARMED_BLOCK_ID: {ARMED_PHYSICAL}})
        self.assertEqual(len(worker.get_hole_recovery_regions()), 1)

        # Window mutator 3: the next step's start_load_kv clears the
        # step-scoped records (the block-id recycling law); the bytes and
        # the region list are untouched by the clear itself.
        worker.start_load_kv(SimpleNamespace(requests=[]))
        self.assertEqual(worker.get_holemiss_invalid_block_layers(), {})
        self.assertEqual(len(worker.get_hole_recovery_regions()), 1)

        # The repair entry state: every healthy layer still holds exactly
        # its delivered bytes - no free, no zero, no overwrite (the store
        # saw no re-issue or write call in the window).
        for layer, delivered in pre_window.items():
            self.assertEqual(
                store.pool.get(1000 + layer),
                delivered,
                f"healthy layer {layer} bytes must survive to the repair entry",
            )
        self.assertNotIn(1000 + ARMED_STORE_LAYER, store.pool)
        self.assertEqual(
            [call[0] for call in store.calls],
            ["copy", "ranges", "lease"],
            "exactly one healthy batch + one isolated call + the lease release; no re-issue",
        )

    def test_connector_hr_hook_bypasses_wait_for_layer_load(self):
        """The load-bearing guard: under _hr_hook_active the connector skips
        the worker's wait_for_layer_load, so _check_hybrid_load_errors never
        runs during the armed repair forward."""
        waited = []
        connector = AscendStoreConnector.__new__(AscendStoreConnector)
        connector.use_layerwise = True
        connector.connector_worker = SimpleNamespace(wait_for_layer_load=lambda: waited.append(1))
        connector._mamba_copy_bufs = None
        connector._mamba_state = None
        connector._hr_hook_active = True
        connector.wait_for_layer_load("model.layers.3.self_attn")
        self.assertEqual(waited, [], "the HR forward must not block on native load waits")

        del connector._hr_hook_active
        connector.wait_for_layer_load("model.layers.3.self_attn")
        self.assertEqual(waited, [1], "without the flag the native wait runs (the old path)")


class TestOneRegionPin(T4TestBase):
    """The vllm mixin consumes injected_regions[0] SILENTLY (mixin:119) - a
    second region per fault request would be dropped without a trace. The
    worker-side properties here make that violation impossible to miss."""

    def test_exactly_one_injected_region_per_fault_request(self):
        worker, _, _ = _drive_new_path_fault()
        regions = worker.get_hole_recovery_regions()
        injected = [r for r in regions if r.get("result", 0) < 0]
        self.assertEqual(len(injected), 1, "the mixin's injected_regions[0] must be total")

    def test_chunked_delivery_still_one_region_and_one_isolated_call(self):
        worker, _, store = _drive_new_path_fault(max_transfer_blocks=1)
        self.assertEqual(len(worker.get_hole_recovery_regions()), 1)
        isolated = [call for call in store.calls if call[0] == "ranges"]
        self.assertEqual(len(isolated), 1, "the armed key isolates exactly once across chunks")
        self.assertEqual(len(isolated[0][1]), 1, "the isolated call carries the armed key alone")
        self.assertEqual(
            worker.get_holemiss_invalid_block_layers(),
            {ARMED_BLOCK_ID: {ARMED_PHYSICAL}},
        )

    def test_text_fault_records_the_store_local_layer(self):
        """Text: store-local == physical; the record is additive and the
        whole-block surface stays byte-identical."""
        worker, thread, _ = _drive_new_path_fault(use_hybrid=False)
        self.assertEqual(worker._invalid_block_ids, {ARMED_BLOCK_ID})
        self.assertEqual(worker.get_holemiss_invalid_block_layers(), {ARMED_BLOCK_ID: {ARMED_STORE_LAYER}})
        region = worker.get_hole_recovery_regions()[0]
        self.assertEqual(region["failed_layers"], [ARMED_STORE_LAYER])


class TestOldPathNeutrality(T4TestBase):
    def test_legacy_store_keeps_whole_key_region_and_no_layer_record(self):
        """A store without the per-range API: the region keeps the pre-V3a
        whole-key shape (no seam fields) and the layer record stays EMPTY -
        the layer scoping must never trigger without real per-range rcs."""
        store = _PoolSimStore(eio_range_index=None, has_v2=False)
        store.batch_copy_get_ranges = None
        store.batch_copy_get = lambda keys, all_buffers, all_sizes, all_offsets: [
            -1102 if key.endswith("@0") and "healthy" not in key else sum(all_sizes[i]) for i, key in enumerate(keys)
        ]
        worker, thread, _ = _drive_new_path_fault(store=store)
        self.assertEqual(worker._invalid_block_ids, {ARMED_BLOCK_ID})
        self.assertEqual(worker.get_holemiss_invalid_block_layers(), {})
        region = worker.get_hole_recovery_regions()[0]
        self.assertNotIn("failed_layers", region)
        self.assertEqual(region["layer_id"], ARMED_PHYSICAL)

    def test_unarmed_failure_keeps_pure_whole_block_semantics(self):
        """A real (non-armed) whole-key load failure: no region, no layer
        record - the whole-block invalidation is the honest record for a
        whole-key loss, and the scheduler's invalid-external-block path (not
        the no-rollback gate) is its consumer."""
        store = _PoolSimStore()
        plain = store.batch_copy_get

        def failing_middle_key(keys, all_buffers, all_sizes, all_offsets):
            results = plain(keys, all_buffers, all_sizes, all_offsets)
            return [-1102 if "unarmed" in key else rc for key, rc in zip(keys, results, strict=True)]

        store.batch_copy_get = failing_middle_key
        worker, thread, _ = _drive_new_path_fault(store=store, armed=False)
        self.assertEqual(worker._invalid_block_ids, {ARMED_BLOCK_ID})
        self.assertEqual(worker.get_holemiss_invalid_block_layers(), {})
        self.assertEqual(worker.get_hole_recovery_regions(), [])

    def test_healthy_armed_run_records_nothing(self):
        store = _PoolSimStore(eio_range_index=None)
        worker, thread, _ = _drive_new_path_fault(store=store)
        self.assertEqual(worker._invalid_block_ids, set())
        self.assertEqual(worker.get_holemiss_invalid_block_layers(), {})
        self.assertEqual(worker.get_hole_recovery_regions(), [])

    def test_plain_key_failure_outside_the_arm_never_scopes(self):
        """An armed run where a HEALTHY key fails the plain batch call: the
        whole-block surface carries that block, but no region/no layer record
        exists for it (only the armed key's isolated call can scope layers)."""
        store = _PoolSimStore(eio_range_index=ARMED_STORE_LAYER)
        plain = store.batch_copy_get

        def failing_healthy(keys, all_buffers, all_sizes, all_offsets):
            results = plain(keys, all_buffers, all_sizes, all_offsets)
            return [-1102 if "healthy1" in key else rc for key, rc in zip(keys, results, strict=True)]

        store.batch_copy_get = failing_healthy
        worker, thread, _ = _drive_new_path_fault(store=store)
        self.assertEqual(worker._invalid_block_ids, {ARMED_BLOCK_ID, HEALTHY_BLOCK_IDS[1]})
        self.assertEqual(
            worker.get_holemiss_invalid_block_layers(),
            {ARMED_BLOCK_ID: {ARMED_PHYSICAL}},
            "the healthy key's whole-block loss never enters the layer record",
        )
        self.assertEqual(len(worker.get_hole_recovery_regions()), 1)


class TestSchedulerGateSourcePins(T4TestBase):
    """The scheduler-side consumers live in vllm-proper (read-only for T4):
    source pins keep their structure honest - the B12 no-rollback gate keys
    on REGIONS and its elif structure is what drops the whole-block ids in
    the fault step."""

    def test_scheduler_no_rollback_gate_precedes_invalid_block_handling(self):
        import inspect

        scheduler = __import__("vllm.v1.core.sched.scheduler", fromlist=["x"]).Scheduler
        source = inspect.getsource(scheduler.update_from_output)
        gate = source.index("if kv_connector_output and kv_connector_output.hole_recovery_regions:")
        invalid_elif = source.index("elif kv_connector_output and kv_connector_output.invalid_block_ids:")
        self.assertLess(gate, invalid_elif, "the regions gate must own the branch")
        region_branch = source[gate:invalid_elif]
        self.assertNotIn("_handle_invalid_blocks", region_branch, "regions present = log-only")
        invalid_branch = source[invalid_elif:]
        self.assertIn("_handle_invalid_blocks", invalid_branch, "the elif handles the invalid ids")

    def test_mixin_collects_invalid_ids_and_regions_into_one_output(self):
        """The whole-block ids and the regions ride the SAME
        KVConnectorOutput - the property that makes the scheduler's elif
        drop them in the fault step."""
        import inspect

        module = __import__("vllm.v1.worker.kv_connector_model_runner_mixin", fromlist=["x"])
        source = inspect.getsource(module.KVConnectorModelRunnerMixin._get_kv_connector_output)
        self.assertIn("output.invalid_block_ids = kv_connector.get_block_ids_with_load_errors()", source)
        self.assertIn("output.hole_recovery_regions = hr_regions_snapshot", source)
        self.assertIn("hr_regions_snapshot, hr_hook_installed = _holemiss_arm_hole_recovery(", source)

    def test_no_va_scheduler_copy_consumes_the_invalid_set(self):
        """The monkeypatch-copy census law: the invalid-block consumers live
        in vllm-proper update_from_output; no va schedule() copy re-copies
        that body, so no copy needs layer-scoping treatment."""
        from pathlib import Path

        import vllm_ascend

        pkg = Path(vllm_ascend.__file__).resolve().parent
        copies = [
            pkg / "core" / "recompute_scheduler.py",
            pkg / "core" / "dyntra_lb_scheduler.py",
            pkg / "core" / "scheduler_profiling_chunk.py",
            pkg / "patch" / "platform" / "patch_balance_schedule.py",
            pkg / "core" / "batch_job_aware_scheduler.py",
            pkg / "core" / "short_request_first_scheduler.py",
        ]
        for path in copies:
            with self.subTest(copy=path.name):
                self.assertTrue(path.exists(), f"missing scheduler copy: {path}")
                source = path.read_text(encoding="utf-8")
                self.assertNotIn(
                    "_handle_invalid_blocks",
                    source,
                    "a va scheduler copy started consuming invalid blocks - the "
                    "layer-scoping census must be re-run for it",
                )


if __name__ == "__main__":
    unittest.main()
