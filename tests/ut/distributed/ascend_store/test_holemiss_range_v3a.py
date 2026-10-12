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
"""CPU-only unit tests for the V3a per-range EIO target + result mapping seams.

The Mooncake C++ narrows the injected disk EIO to the armed FA layer's byte
extent inside the layerless D7 block object (HOLEMISS_EIO_TARGET_OFFSET /
HOLEMISS_EIO_TARGET_SIZE) so the armed key's healthy ranges survive the
per-range reads. These tests pin the va side: the extent derivation (the
SAME layer_cache_entry_offsets arithmetic the load plan builder uses), the
env export/clear lifecycle, the failed-range -> layer mapping seam consumed
by T4's layer-scoped invalidation, and the per-range -> per-key fold.
"""

import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import (
    BatchResultShapeError,
    fold_range_results_to_key,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.layerwise_keys import (
    LayerwiseKeyBuilder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    MooncakeBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_layerwise import (
    make_block_key,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    HOLEMISS_EIO_TARGET_ENV,
    HOLEMISS_EIO_TARGET_OFFSET_ENV,
    HOLEMISS_EIO_TARGET_SIZE_ENV,
    KVPoolWorker,
)

# LIVE hybrid g0 geometry (N18/N19 - never text-shaped fixtures): 16 FA
# store-local layers, 2 cache entries (K, V) per layer, 1572864 bytes per
# entry -> 3145728 bytes per layer, 50331648 bytes per block object.
NUM_FA_LAYERS = 16
ENTRY_BYTES = 1572864
ENTRIES_PER_LAYER = 2
LAYER_BYTES = ENTRY_BYTES * ENTRIES_PER_LAYER
OBJECT_BYTES = LAYER_BYTES * NUM_FA_LAYERS

FA_PHYSICAL = [3 + 4 * k for k in range(NUM_FA_LAYERS)]

HOLEMISS_ENVS = [
    "HOLEMISS_DISK_EIO_ENABLED",
    "HOLEMISS_EIO_LAYER",
    "HOLEMISS_EIO_LOGICAL_BLOCK",
    "HOLEMISS_FAIL_LAYER",
    "HOLEMISS_FAIL_LOGICAL_BLOCK",
    "EIO_TARGET_KEY",
    "HOLEMISS_EIO_TARGET_OFFSET",
    "HOLEMISS_EIO_TARGET_SIZE",
]

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
                    f"model.layers.{p}.linear_attn"
                    for p in range(64)
                    if p not in FA_PHYSICAL and (p % 4) == gdn_group
                ]
            )
        )
    return SimpleNamespace(kv_cache_groups=groups)


def _live_hybrid_geometry():
    """The live g0 layer_cache_entry_offsets / block_len shape."""
    entry_offsets = list(range(0, NUM_FA_LAYERS * ENTRIES_PER_LAYER + 1, ENTRIES_PER_LAYER))
    block_len = [ENTRY_BYTES] * (NUM_FA_LAYERS * ENTRIES_PER_LAYER)
    return {0: entry_offsets}, {0: block_len}


def _make_worker(
    use_hybrid: bool = True,
    with_geometry: bool = True,
    num_layers: int = 64,
):
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.put_step = 1
    worker.head_or_tp_rank = 0
    worker.pp_rank = 0
    worker.num_layers = num_layers
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
    worker.hf_config = SimpleNamespace(num_hidden_layers=64)
    worker.kv_cache_config = _hybrid_kv_cache_config() if use_hybrid else None
    worker.layerwise_keys = _mock_layerwise_keys()
    worker._invalid_block_ids = set()
    worker._invalid_block_ids_lock = threading.Lock()
    worker._hole_recovery_regions = []
    worker._hole_recovery_lock = threading.Lock()
    worker._holemiss_armed_fault = None
    worker._holemiss_fa_layer_map = None
    worker.kv_send_thread = None
    worker.kv_recv_thread = None
    worker.m_store = MagicMock()
    worker.m_store.put_end_disk.return_value = True
    if with_geometry:
        worker.group_layer_cache_entry_offsets, worker.group_block_len = _live_hybrid_geometry()
    else:
        worker.group_layer_cache_entry_offsets = {}
        worker.group_block_len = {}
    return worker


def _make_request(num_blocks: int = 4):
    block_hashes = [bytes([(i + 1) % 256] * 32) for i in range(num_blocks)]
    block_ids = [[10 + i for i in range(num_blocks)]]
    load_spec = LoadSpec(
        vllm_cached_tokens=0,
        kvpool_cached_tokens=3 * 1536,
        can_load=True,
        kvpool_store_skip_tokens=None,
        token_len=0,
    )
    return ReqMeta(
        req_id="req-V3a",
        token_len_chunk=4 * 1536,
        block_ids_by_group=block_ids,
        block_hashes=block_hashes,
        can_save=False,
        load_spec=load_spec,
    )


def _layer_ranges(layer_indices):
    """The merged-call shape: per-layer (offset, size) fragments in layer order."""
    offsets = [k * LAYER_BYTES for k in layer_indices]
    sizes = [LAYER_BYTES] * len(layer_indices)
    return offsets, sizes


class V3aTestBase(unittest.TestCase):
    def setUp(self):
        self._saved_env = {name: os.environ.get(name) for name in HOLEMISS_ENVS}
        for name in HOLEMISS_ENVS:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class TestLayerExtentDerivation(V3aTestBase):
    def test_live_hybrid_geometry_extents(self):
        worker = _make_worker()
        for k in range(NUM_FA_LAYERS):
            extent = worker._holemiss_armable_layer_extent(k)
            self.assertEqual(extent, (k * LAYER_BYTES, LAYER_BYTES))

    def test_matches_load_plan_builder_arithmetic(self):
        # The build_addrs formula: layer_object_offset =
        # sum(block_len[:base_offset]); sizes = block_len[base_offset:end].
        # The arm derivation must be the SAME numbers (the D12/EIO-TARGET
        # law: never a second arithmetic).
        worker = _make_worker()
        entry_offsets = worker.group_layer_cache_entry_offsets[0]
        block_len = worker.group_block_len[0]
        for k in range(NUM_FA_LAYERS):
            base, end = int(entry_offsets[k]), int(entry_offsets[k + 1])
            expected_offset = sum(int(x) for x in block_len[:base])
            expected_size = sum(int(x) for x in block_len[base:end])
            self.assertEqual(
                worker._holemiss_armable_layer_extent(k),
                (expected_offset, expected_size),
            )
        # The whole object is exactly covered by the 16 layer extents.
        self.assertEqual((NUM_FA_LAYERS - 1) * LAYER_BYTES + LAYER_BYTES, OBJECT_BYTES)

    def test_out_of_range_and_missing_geometry(self):
        worker = _make_worker()
        self.assertIsNone(worker._holemiss_armable_layer_extent(-1))
        self.assertIsNone(worker._holemiss_armable_layer_extent(NUM_FA_LAYERS))
        bare = _make_worker(with_geometry=False)
        self.assertIsNone(bare._holemiss_armable_layer_extent(0))


class TestArmEnvExport(V3aTestBase):
    def test_arm_exports_key_and_extent(self):
        worker = _make_worker()
        request = _make_request()
        armed_layer = 5
        worker._holemiss_arm_disk_eio(request, armed_layer, 1)
        self.assertEqual(os.environ[HOLEMISS_EIO_TARGET_ENV], worker._holemiss_armed_fault["key"])
        self.assertEqual(os.environ[HOLEMISS_EIO_TARGET_OFFSET_ENV], str(armed_layer * LAYER_BYTES))
        self.assertEqual(os.environ[HOLEMISS_EIO_TARGET_SIZE_ENV], str(LAYER_BYTES))
        self.assertEqual(worker._holemiss_armed_fault["layer_id"], armed_layer)

    def test_arm_without_geometry_degrades_loudly_to_whole_object(self):
        worker = _make_worker(with_geometry=False)
        request = _make_request()
        worker._holemiss_arm_disk_eio(request, 5, 1)
        # The key still arms (the pre-V3a whole-object window); the extent
        # envs stay unset so the C++ keeps the whole-object semantics.
        self.assertEqual(os.environ[HOLEMISS_EIO_TARGET_ENV], worker._holemiss_armed_fault["key"])
        self.assertNotIn(HOLEMISS_EIO_TARGET_OFFSET_ENV, os.environ)
        self.assertNotIn(HOLEMISS_EIO_TARGET_SIZE_ENV, os.environ)

    def test_configure_fault_clears_all_target_envs(self):
        worker = _make_worker()
        os.environ[HOLEMISS_EIO_TARGET_ENV] = "stale@key"
        os.environ[HOLEMISS_EIO_TARGET_OFFSET_ENV] = "4096"
        os.environ[HOLEMISS_EIO_TARGET_SIZE_ENV] = "4096"
        worker._holemiss_configure_fault(None)
        self.assertEqual(os.environ[HOLEMISS_EIO_TARGET_ENV], "")
        self.assertNotIn(HOLEMISS_EIO_TARGET_OFFSET_ENV, os.environ)
        self.assertNotIn(HOLEMISS_EIO_TARGET_SIZE_ENV, os.environ)


class TestRangeFailedLayersMapping(V3aTestBase):
    def test_failed_layer_maps_to_physical(self):
        worker = _make_worker()
        offsets, sizes = _layer_ranges(list(range(NUM_FA_LAYERS)))
        rcs = [LAYER_BYTES] * NUM_FA_LAYERS
        rcs[7] = -1102
        failed_layers, failed_indices = worker.holemiss_range_failed_layers(offsets, sizes, rcs)
        self.assertEqual(failed_indices, [7])
        self.assertEqual(failed_layers, [FA_PHYSICAL[7]])

    def test_multiple_failed_layers_and_spanning_range(self):
        worker = _make_worker()
        offsets, sizes = _layer_ranges([0, 1, 2])
        rcs = [-1102, LAYER_BYTES, -800]
        failed_layers, failed_indices = worker.holemiss_range_failed_layers(offsets, sizes, rcs)
        self.assertEqual(failed_indices, [0, 2])
        self.assertEqual(failed_layers, [FA_PHYSICAL[0], FA_PHYSICAL[2]])
        # A range spanning two layers attributes both (the honest superset).
        spanning_offsets = [0]
        spanning_sizes = [2 * LAYER_BYTES]
        failed_layers, failed_indices = worker.holemiss_range_failed_layers(
            spanning_offsets, spanning_sizes, [-1102]
        )
        self.assertEqual(failed_layers, FA_PHYSICAL[0:2])

    def test_healthy_only_maps_nothing(self):
        worker = _make_worker()
        offsets, sizes = _layer_ranges([0, 1, 2])
        failed_layers, failed_indices = worker.holemiss_range_failed_layers(
            offsets, sizes, [LAYER_BYTES] * 3
        )
        self.assertEqual(failed_layers, [])
        self.assertEqual(failed_indices, [])

    def test_text_passes_store_local_through(self):
        worker = _make_worker(use_hybrid=False)
        # Text: single group, store-local == physical layer ids.
        entry_offsets = list(range(0, 5))
        worker.group_layer_cache_entry_offsets = {0: entry_offsets}
        worker.group_block_len = {0: [4096] * 4}
        failed_layers, failed_indices = worker.holemiss_range_failed_layers(
            [0, 8192], [4096, 4096], [4096, -1102]
        )
        self.assertEqual(failed_layers, [2])
        self.assertEqual(failed_indices, [1])


class TestFoldRangeResults(V3aTestBase):
    def test_all_success_sums_bytes(self):
        self.assertEqual(fold_range_results_to_key([LAYER_BYTES] * 32), 32 * LAYER_BYTES)

    def test_any_failure_fails_fast_with_first_negative(self):
        self.assertEqual(fold_range_results_to_key([1, -1102, -800]), -1102)
        self.assertEqual(fold_range_results_to_key([-800, 1]), -800)


class _FakeStore:
    """A minimal store double with the V3a per-range surface."""

    def __init__(self, rows=None, has_v2=True):
        self.rows = rows or []
        self.has_v2 = has_v2
        self.calls = []

    def batch_get_into_multi_buffer_ranges_v2(self, keys, all_buffers, all_sizes, all_src_offsets):
        self.calls.append(("v2", list(keys)))
        return [list(row) for row in self.rows]

    def batch_get_into_multi_buffer_ranges(self, keys, all_buffers, all_sizes, all_src_offsets):
        self.calls.append(("v1", list(keys)))
        return [fold_range_results_to_key(row) for row in self.rows]

    def batch_copy_get(self, keys, all_buffers, all_sizes, all_src_offsets):
        self.calls.append(("copy", list(keys)))
        return [fold_range_results_to_key(row) for row in self.rows]

    def batch_copy_get_ranges(self, keys, all_buffers, all_sizes, all_src_offsets):
        self.calls.append(("ranges", list(keys)))
        return [list(row) for row in self.rows]


class TestBackendMigration(V3aTestBase):
    def _backend_with_store(self, fake_store):
        backend = MooncakeBackend.__new__(MooncakeBackend)
        backend.store = fake_store
        backend._store_initialized = True
        return backend

    def test_batch_copy_get_folds_v2_rows(self):
        fake = _FakeStore(rows=[[LAYER_BYTES, -1102, LAYER_BYTES]])
        backend = self._backend_with_store(fake)
        rcs = backend.batch_copy_get(["k"], [[1, 2, 3]], [[LAYER_BYTES] * 3], [[0, 1, 2]])
        self.assertEqual(rcs, [-1102])
        self.assertEqual(fake.calls, [("v2", ["k"])])

    def test_batch_copy_get_all_success_reports_byte_total(self):
        fake = _FakeStore(rows=[[LAYER_BYTES, LAYER_BYTES]])
        backend = self._backend_with_store(fake)
        rcs = backend.batch_copy_get(["k"], [[1, 2]], [[LAYER_BYTES] * 2], [[0, 1]])
        self.assertEqual(rcs, [2 * LAYER_BYTES])

    def test_batch_copy_get_falls_back_without_v2(self):
        fake = _FakeStore(rows=[[LAYER_BYTES, -1102]])
        fake.has_v2 = False
        fake.batch_get_into_multi_buffer_ranges_v2 = None
        backend = self._backend_with_store(fake)
        rcs = backend.batch_copy_get(["k"], [[1, 2]], [[LAYER_BYTES] * 2], [[0, 1]])
        self.assertEqual(rcs, [-1102])
        self.assertEqual(fake.calls, [("v1", ["k"])])

    def test_batch_copy_get_ranges_returns_raw_rows(self):
        fake = _FakeStore(rows=[[LAYER_BYTES, -1102, LAYER_BYTES]])
        backend = self._backend_with_store(fake)
        rows = backend.batch_copy_get_ranges(["k"], [[1, 2, 3]], [[LAYER_BYTES] * 3], [[0, 1, 2]])
        self.assertEqual(rows, [[LAYER_BYTES, -1102, LAYER_BYTES]])

    def test_batch_copy_get_ranges_fallback_replicates_key_rc(self):
        fake = _FakeStore(rows=[[-1102, -1102]])
        fake.batch_get_into_multi_buffer_ranges_v2 = None
        backend = self._backend_with_store(fake)
        rows = backend.batch_copy_get_ranges(["k"], [[1, 2]], [[LAYER_BYTES] * 2], [[0, 1]])
        self.assertEqual(rows, [[-1102, -1102]])

    def test_shape_error_on_malformed_rows(self):
        fake = _FakeStore(rows=[[LAYER_BYTES]])  # 1 row for 2 requested ranges
        backend = self._backend_with_store(fake)
        with self.assertRaises(BatchResultShapeError):
            backend.batch_copy_get_ranges(["k"], [[1, 2]], [[LAYER_BYTES] * 2], [[0, 1]])


class TestExecuteRangeTargetPerRange(V3aTestBase):
    def _armed_worker(self, fake_store):
        worker = _make_worker()
        request = _make_request()
        key = worker._holemiss_derive_block_key(request, 1)
        worker._holemiss_armed_fault = worker._holemiss_make_armed_record(
            "disk_eio", request, key, 5, 1
        )
        worker.m_store = fake_store
        return worker, key

    def test_region_carries_failed_layers_seam(self):
        rows = [[LAYER_BYTES] * NUM_FA_LAYERS]
        rows[0][5] = -1102
        fake = _FakeStore(rows=rows)
        worker, key = self._armed_worker(fake)
        offsets, sizes = _layer_ranges(list(range(NUM_FA_LAYERS)))
        results = worker.execute_holemiss_range_target(
            [key], [[1] * NUM_FA_LAYERS], [sizes], [offsets], 0
        )
        self.assertEqual(results, [-1102])
        regions = worker.get_hole_recovery_regions()
        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertEqual(region["result"], -1102)
        self.assertEqual(region["failed_layers"], [FA_PHYSICAL[5]])
        self.assertEqual(region["failed_range_indices"], [5])
        self.assertEqual(region["range_rcs"][5], -1102)
        self.assertEqual(region["layer_start"], FA_PHYSICAL[5])
        self.assertEqual(region["layer_end"], FA_PHYSICAL[5])

    def test_legacy_store_keeps_whole_key_region(self):
        fake = _FakeStore(rows=[[-1102]])
        fake.batch_get_into_multi_buffer_ranges_v2 = None
        fake.batch_copy_get_ranges = None
        worker, key = self._armed_worker(fake)
        offsets, sizes = _layer_ranges(list(range(NUM_FA_LAYERS)))
        results = worker.execute_holemiss_range_target(
            [key], [[1] * NUM_FA_LAYERS], [sizes], [offsets], 0
        )
        self.assertEqual(results, [-1102])
        region = worker.get_hole_recovery_regions()[0]
        self.assertNotIn("failed_layers", region)
        self.assertEqual(region["layer_id"], FA_PHYSICAL[5])

    def test_all_ranges_healthy_reports_fault_not_observed(self):
        rows = [[LAYER_BYTES] * NUM_FA_LAYERS]
        fake = _FakeStore(rows=rows)
        worker, key = self._armed_worker(fake)
        offsets, sizes = _layer_ranges(list(range(NUM_FA_LAYERS)))
        results = worker.execute_holemiss_range_target(
            [key], [[1] * NUM_FA_LAYERS], [sizes], [offsets], 0
        )
        self.assertEqual(results, [NUM_FA_LAYERS * LAYER_BYTES])
        self.assertEqual(worker.get_hole_recovery_regions(), [])


if __name__ == "__main__":
    unittest.main()
