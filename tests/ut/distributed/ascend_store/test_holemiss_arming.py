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
"""CPU-only unit tests for the holemiss fault-arming / HR-region seams (T5).

The mainline store is a block-key (layerless) object per (group, block); the
armed EIO target must be the backend-bound block object key, NEVER the 0.26
per-layer literal.  The real store clients on this wheel require a torch_npu
ACL context, so the backend is ALWAYS a MagicMock here.
"""

import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.layerwise_keys import (
    LayerwiseKeyBuilder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_layerwise import (
    make_block_key,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    LayerBatchedLoadTask,
    LoadSpec,
    ReqMeta,
    get_block_hashes,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    HOLEMISS_EIO_TARGET_ENV,
    HOLEMISS_SYNTHETIC_FAULT_RC,
    KVPoolWorker,
    hybrid_group0_fa_physical_layers,
)

BLOCK_SIZE = 128
NUM_LAYERS = 4
MODEL_NAME = "testmodel"

HOLEMISS_ENVS = [
    "HOLEMISS_DISK_EIO_ENABLED",
    "HOLEMISS_EIO_LAYER",
    "HOLEMISS_EIO_LOGICAL_BLOCK",
    "HOLEMISS_FAIL_LAYER",
    "HOLEMISS_FAIL_LOGICAL_BLOCK",
    "HOLEMISS_SYNC_LAYERWISE_LOAD",
    "HOLEMISS_RUNTIME_PROFILE",
    "HOLEMISS_DIAG",
    "HOLEMISS_KV_LOAD_TIMING",
    "HOLEMISS_RECOVERY_MODE",
    "EIO_TARGET_KEY",
]

FA_PHYSICAL = [3 + 4 * k for k in range(2)]


def _mock_layerwise_keys(pp_size: int = 1) -> LayerwiseKeyBuilder:
    return LayerwiseKeyBuilder(
        make_full_key=lambda group_id, block_hash, rank, stage: make_block_key(
            MODEL_NAME, block_hash, rank, pp_rank=stage
        ),
        pp_size=pp_size,
    )


def _qwen36_config():
    groups = [SimpleNamespace(layer_names=[f"model.layers.{p}.self_attn" for p in FA_PHYSICAL])]
    for gdn_group in range(3):
        groups.append(
            SimpleNamespace(
                layer_names=[
                    f"model.layers.{p}.linear_attn" for p in range(8) if p not in FA_PHYSICAL and (p % 4) == gdn_group
                ]
            )
        )
    return SimpleNamespace(kv_cache_groups=groups)


def _make_worker(num_groups: int = 1, use_hybrid: bool = False, kv_cache_config=None, num_layers: int = NUM_LAYERS):
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.put_step = 1
    worker.head_or_tp_rank = 0
    worker.pp_rank = 0
    worker.num_layers = num_layers
    worker.block_size = BLOCK_SIZE
    worker.grouped_block_size = [BLOCK_SIZE] * num_groups
    worker.hash_block_size = BLOCK_SIZE
    worker.num_kv_cache_groups = num_groups
    worker.use_hybrid = use_hybrid
    worker.use_layerwise = True
    worker.use_block_key_layerwise = True
    worker.use_layerwise_transfer = False
    worker.use_eagle = False
    worker.layerwise_offload = False
    worker.group_uses_align_state = [False] * num_groups
    worker.hf_config = SimpleNamespace(num_hidden_layers=8)
    worker.kv_cache_config = kv_cache_config
    worker.layerwise_keys = _mock_layerwise_keys()
    worker._invalid_block_ids = set()
    worker._invalid_block_ids_lock = threading.Lock()
    worker._hole_recovery_regions = []
    worker._hole_recovery_lock = threading.Lock()
    worker._holemiss_armed_fault = None
    worker._holemiss_fa_layer_map = None
    worker.kv_send_thread = None
    worker.kv_recv_thread = None
    worker.layer_load_tasks = [[] for _ in range(num_layers)]
    worker.next_layer_to_submit = 0
    worker.m_store = MagicMock()
    worker.m_store.put_end_disk.return_value = True
    worker.m_store.batch_copy_get.side_effect = lambda keys, buffers, sizes, offsets: [0] * len(keys)
    return worker


def _make_request(
    req_id: str = "req-A",
    num_blocks: int = 4,
    vllm_cached: int = 0,
    kvpool_cached: int = 3 * BLOCK_SIZE,
    block_id_base: int = 10,
):
    block_hashes = [bytes([(i + 1) % 256] * 32) for i in range(num_blocks)]
    block_ids = [[block_id_base + i for i in range(num_blocks)]]
    load_spec = LoadSpec(
        vllm_cached_tokens=vllm_cached,
        kvpool_cached_tokens=kvpool_cached,
        can_load=True,
        kvpool_store_skip_tokens=None,
        token_len=0,
    )
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=4 * BLOCK_SIZE,
        block_ids_by_group=block_ids,
        block_hashes=block_hashes,
        can_save=False,
        load_spec=load_spec,
    )


def _set_disk_eio(layer: int, logical: int) -> None:
    os.environ["HOLEMISS_DISK_EIO_ENABLED"] = "1"
    os.environ["HOLEMISS_EIO_LAYER"] = str(layer)
    os.environ["HOLEMISS_EIO_LOGICAL_BLOCK"] = str(logical)


class HolemissTestBase(unittest.TestCase):
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


class TestFaultConfigLifecycle(HolemissTestBase):
    def test_configure_fault_clears_eio_target_unconditionally(self):
        worker = _make_worker()
        request = _make_request()
        os.environ[HOLEMISS_EIO_TARGET_ENV] = "stale@target@1"
        worker._holemiss_configure_fault(request)
        self.assertEqual(os.environ[HOLEMISS_EIO_TARGET_ENV], "")
        self.assertIsNone(worker._holemiss_armed_fault)

        os.environ[HOLEMISS_EIO_TARGET_ENV] = "stale@target@2"
        worker._holemiss_configure_fault(None)
        self.assertEqual(os.environ[HOLEMISS_EIO_TARGET_ENV], "")

    def test_arming_sets_block_key_target_and_calls_put_end_disk(self):
        worker = _make_worker()
        request = _make_request()
        os.environ.update(
            {
                "HOLEMISS_DISK_EIO_ENABLED": "1",
                "HOLEMISS_EIO_LAYER": "3",
                "HOLEMISS_EIO_LOGICAL_BLOCK": "2",
            }
        )
        worker._holemiss_configure_fault(request)

        expected_key = worker._holemiss_derive_block_key(request, 2)
        self.assertEqual(os.environ[HOLEMISS_EIO_TARGET_ENV], expected_key)
        self.assertTrue(expected_key.startswith(f"{MODEL_NAME}@"))
        self.assertTrue(expected_key.endswith(f"@{request.block_hashes[2].hex()}@0"))
        worker.m_store.put_end_disk.assert_called_once_with(expected_key)

        armed = worker._holemiss_armed_fault
        self.assertIsNotNone(armed)
        self.assertEqual(armed["kind"], "disk_eio")
        self.assertEqual(armed["layer_id"], 3)
        self.assertEqual(armed["logical_block_index"], 2)
        self.assertEqual(armed["physical_block_id"], 12)
        self.assertEqual(armed["token_start"], 2 * BLOCK_SIZE)
        self.assertEqual(armed["token_end"], 3 * BLOCK_SIZE)

    def test_block_key_target_byte_matches_main_builder(self):
        worker = _make_worker()
        request = _make_request()
        _set_disk_eio(0, 1)
        worker._holemiss_configure_fault(request)

        expected = make_block_key(MODEL_NAME, request.block_hashes[1].hex(), 0)
        self.assertEqual(os.environ[HOLEMISS_EIO_TARGET_ENV], expected)

    def test_arming_without_put_end_disk_still_sets_target(self):
        worker = _make_worker()
        del worker.m_store.put_end_disk
        request = _make_request()
        _set_disk_eio(1, 1)
        worker._holemiss_configure_fault(request)
        self.assertNotEqual(os.environ[HOLEMISS_EIO_TARGET_ENV], "")

    def test_synthetic_priority_blocks_real_eio_arming(self):
        worker = _make_worker()
        request = _make_request()
        os.environ.update(
            {
                "HOLEMISS_DISK_EIO_ENABLED": "1",
                "HOLEMISS_EIO_LAYER": "3",
                "HOLEMISS_EIO_LOGICAL_BLOCK": "2",
                "HOLEMISS_FAIL_LAYER": "3",
                "HOLEMISS_FAIL_LOGICAL_BLOCK": "2",
            }
        )
        worker._holemiss_configure_fault(request)
        self.assertEqual(worker._holemiss_armed_fault["kind"], "synthetic")
        self.assertEqual(os.environ[HOLEMISS_EIO_TARGET_ENV], "")
        worker.m_store.put_end_disk.assert_not_called()

    def test_negative_fail_layer_sentinel_reaches_real_eio(self):
        worker = _make_worker()
        request = _make_request()
        os.environ.update(
            {
                "HOLEMISS_DISK_EIO_ENABLED": "1",
                "HOLEMISS_EIO_LAYER": "3",
                "HOLEMISS_EIO_LOGICAL_BLOCK": "2",
                "HOLEMISS_FAIL_LAYER": "-1",
            }
        )
        worker._holemiss_configure_fault(request)
        self.assertEqual(worker._holemiss_armed_fault["kind"], "disk_eio")

    def test_out_of_span_target_does_not_arm(self):
        worker = _make_worker()
        request = _make_request(kvpool_cached=1 * BLOCK_SIZE)
        _set_disk_eio(0, 2)
        worker._holemiss_configure_fault(request)
        self.assertIsNone(worker._holemiss_armed_fault)
        self.assertEqual(os.environ[HOLEMISS_EIO_TARGET_ENV], "")


class TestHybridFaMap(HolemissTestBase):
    def test_group0_derivation_and_none_cases(self):
        self.assertEqual(hybrid_group0_fa_physical_layers(_qwen36_config(), 8, 8), FA_PHYSICAL)
        self.assertIsNone(hybrid_group0_fa_physical_layers(None, 8, 8))
        self.assertIsNone(hybrid_group0_fa_physical_layers(SimpleNamespace(kv_cache_groups=[]), 8, 8))

    def test_worker_map_and_physical_translation(self):
        worker = _make_worker(num_groups=4, use_hybrid=True, kv_cache_config=_qwen36_config(), num_layers=8)
        self.assertEqual(worker._holemiss_hybrid_fa_layers(), FA_PHYSICAL)
        self.assertEqual(worker._holemiss_armable_layer_count(), 2)
        self.assertEqual(worker._holemiss_hybrid_physical_layer(0), 3)
        self.assertEqual(worker._holemiss_hybrid_physical_layer(1), 7)
        with self.assertRaises(RuntimeError):
            worker._holemiss_hybrid_physical_layer(2)

    def test_region_layer_id_is_physical_for_hybrid(self):
        worker = _make_worker(num_groups=4, use_hybrid=True, kv_cache_config=_qwen36_config(), num_layers=8)
        request = _make_request()
        armed = worker._holemiss_make_armed_record("synthetic", request, "k", 1, 2)
        region = worker._holemiss_make_region(armed, HOLEMISS_SYNTHETIC_FAULT_RC, 0)
        self.assertEqual(region["layer_id"], 7)
        self.assertEqual(region["layer_start"], 7)

    def test_hybrid_unavailable_map_skips_arming(self):
        worker = _make_worker(num_groups=4, use_hybrid=True, kv_cache_config=None)
        request = _make_request()
        _set_disk_eio(0, 2)
        worker._holemiss_configure_fault(request)
        self.assertIsNone(worker._holemiss_armed_fault)


class TestRegionStoreAndPurge(HolemissTestBase):
    def test_regions_delegate_and_purge_on_teardown(self):
        worker = _make_worker()
        worker.record_hole_recovery_region({"request_id": "req-A", "result": -800})
        worker.record_hole_recovery_region({"request_id": "req-B", "result": -1102})
        self.assertEqual(len(worker.get_hole_recovery_regions()), 2)

        worker._holemiss_teardown_finished({"req-A"})
        self.assertEqual([r["request_id"] for r in worker.get_hole_recovery_regions()], ["req-B"])

        worker.clear_hole_recovery_regions()
        self.assertEqual(worker.get_hole_recovery_regions(), [])

    def test_teardown_clears_armed_window_for_finished_request(self):
        worker = _make_worker()
        request = _make_request()
        _set_disk_eio(0, 2)
        worker._holemiss_configure_fault(request)
        self.assertIsNotNone(worker._holemiss_armed_fault)
        worker._holemiss_teardown_finished({"req-A"})
        self.assertIsNone(worker._holemiss_armed_fault)
        self.assertEqual(os.environ[HOLEMISS_EIO_TARGET_ENV], "")

    def test_teardown_keeps_armed_window_for_other_request(self):
        worker = _make_worker()
        request = _make_request(req_id="req-A")
        _set_disk_eio(0, 2)
        worker._holemiss_configure_fault(request)
        worker._holemiss_teardown_finished({"req-B"})
        self.assertIsNotNone(worker._holemiss_armed_fault)


class TestArmedRangeGet(HolemissTestBase):
    def _armed_worker(self):
        worker = _make_worker()
        request = _make_request()
        key = worker._holemiss_derive_block_key(request, 2)
        worker._holemiss_armed_fault = worker._holemiss_make_armed_record("synthetic", request, key, 3, 2)
        return worker, key

    def test_target_position_matches_key_and_layer(self):
        worker, key = self._armed_worker()
        self.assertEqual(worker.holemiss_range_target_position([key, "other"], 3), 0)
        self.assertIsNone(worker.holemiss_range_target_position([key, "other"], 2))
        self.assertIsNone(worker.holemiss_range_target_position(["other"], 3))

    def test_execute_emits_exactly_one_region(self):
        worker, key = self._armed_worker()
        results = worker.execute_holemiss_range_target([key, "other"], [[1], [2]], [[3], [4]], [[0], [0]], 0)
        self.assertEqual(results, [HOLEMISS_SYNTHETIC_FAULT_RC, 0])
        regions = worker.get_hole_recovery_regions()
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]["key"], key)
        self.assertEqual(regions[0]["result"], HOLEMISS_SYNTHETIC_FAULT_RC)
        self.assertEqual(regions[0]["layer_id"], 3)


class TestPmuBlockKeyDerivation(HolemissTestBase):
    """The N18/N19 closure (hybrid-matrix-pmu-10lens T2): under
    ``prefix_match_unit < block_size`` the arm target MUST be the LAST
    sub-hash of the group block — ``block_hashes[(b+1)*scale - 1]`` — the
    same ``_LazyGroupedBlockHashList`` law (metadata.py:800) the load span,
    the hit check, and the GET session use.

    The live conviction (2026-10-14, pmu10lens_hyb36_4608): the flat
    ``block_hashes[logical_block]`` read armed the 256-gran sub-hash
    ``d685ca6c...`` while the store held ``b72e19dc...`` (block 0) — a
    phantom key the consumer never reads, so the C++ EIO window never
    engaged (zero [EIO_INJECT], expected-fault-not-delivered). No green
    path had ever driven pmu + a fault row (the cl_sweep ran pmu-less;
    the gdn-tail batteries ran fault-less).
    """

    HYB_BLOCK_SIZE = 1536
    PMU = 256
    SCALE = HYB_BLOCK_SIZE // PMU  # 6
    N_BLOCKS = 3  # the 4608 lens shape: 3 full blocks = 18 sub-hashes

    def _make_hybrid_worker(self, hash_block_size: int, block_size: int = HYB_BLOCK_SIZE):
        worker = _make_worker(
            num_groups=4,
            use_hybrid=True,
            kv_cache_config=_qwen36_config(),
            num_layers=8,
        )
        worker.block_size = block_size
        worker.grouped_block_size = [block_size] * 4
        worker.hash_block_size = hash_block_size
        return worker

    def _make_pmu_request(self, num_blocks: int = N_BLOCKS):
        num_sub = num_blocks * self.SCALE
        block_hashes = [bytes([(i + 1) % 256] * 32) for i in range(num_sub)]
        block_ids = [[10 + i for i in range(num_blocks)] for _ in range(4)]
        load_spec = LoadSpec(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=num_blocks * self.HYB_BLOCK_SIZE,
            can_load=True,
            kvpool_store_skip_tokens=None,
            token_len=0,
        )
        return ReqMeta(
            req_id="req-pmu",
            token_len_chunk=num_blocks * self.HYB_BLOCK_SIZE,
            block_ids_by_group=block_ids,
            block_hashes=block_hashes,
            can_save=False,
            load_spec=load_spec,
        )

    def test_pmu_derivation_is_last_sub_hash_of_group_block(self):
        worker = self._make_hybrid_worker(self.PMU)
        request = self._make_pmu_request()
        sub = request.block_hashes
        for b in range(self.N_BLOCKS):
            key = worker._holemiss_derive_block_key(request, b)
            expected_hash = sub[(b + 1) * self.SCALE - 1]
            self.assertEqual(key, worker._make_mooncake_layerwise_key(expected_hash.hex()))
            # The REAL law the GET session uses — no derivation drift.
            self.assertEqual(
                sub[(b + 1) * self.SCALE - 1],
                get_block_hashes(sub, self.HYB_BLOCK_SIZE, self.PMU)[b],
            )
            # The discrimination that convicted the defect: the flat read
            # (the old behavior) arms the sub-hash at logical_block*256
            # tokens — a phantom key absent from the store.
            self.assertNotEqual(key, worker._make_mooncake_layerwise_key(sub[b].hex()))

    def test_pmu_arming_via_configure_fault_targets_store_key(self):
        worker = self._make_hybrid_worker(self.PMU)
        request = self._make_pmu_request()
        _set_disk_eio(0, 1)
        worker._holemiss_configure_fault(request)
        expected = worker._make_mooncake_layerwise_key(request.block_hashes[(1 + 1) * self.SCALE - 1].hex())
        self.assertEqual(os.environ[HOLEMISS_EIO_TARGET_ENV], expected)
        worker.m_store.put_end_disk.assert_called_once_with(expected)
        self.assertEqual(worker._holemiss_armed_fault["logical_block_index"], 1)

    def test_pmu_synthetic_twin_shares_the_scale_aware_derivation(self):
        worker = self._make_hybrid_worker(self.PMU)
        request = self._make_pmu_request()
        os.environ.update(
            {
                "HOLEMISS_FAIL_LAYER": "0",
                "HOLEMISS_FAIL_LOGICAL_BLOCK": "2",
            }
        )
        worker._holemiss_configure_fault(request)
        self.assertEqual(worker._holemiss_armed_fault["kind"], "synthetic")
        self.assertEqual(
            worker._holemiss_armed_fault["key"],
            worker._make_mooncake_layerwise_key(request.block_hashes[(2 + 1) * self.SCALE - 1].hex()),
        )

    def test_pmu_bounds_guard_is_group_block_granular(self):
        worker = self._make_hybrid_worker(self.PMU)
        request = self._make_pmu_request()
        # 3 group blocks: logical 3 is out of range even though the flat
        # sub-hash list has 18 entries (the old guard never tripped here).
        self.assertIsNone(worker._holemiss_derive_block_key(request, 3))
        self.assertIsNone(worker._holemiss_derive_block_key(request, -1))
        self.assertIsNone(worker._holemiss_derive_block_key(request, 17))

    def test_non_pmu_hybrid_derivation_byte_identical_flat(self):
        # The cl_sweep's green shape (no pmu): hash_block_size == block_size
        # -> get_block_hashes returns the list unchanged -> the flat read.
        worker = self._make_hybrid_worker(self.HYB_BLOCK_SIZE)
        block_hashes = [bytes([(i + 1) % 256] * 32) for i in range(self.N_BLOCKS)]
        request = ReqMeta(
            req_id="req-nopmu",
            token_len_chunk=self.N_BLOCKS * self.HYB_BLOCK_SIZE,
            block_ids_by_group=[[10 + i for i in range(self.N_BLOCKS)] for _ in range(4)],
            block_hashes=block_hashes,
            can_save=False,
            load_spec=LoadSpec(
                vllm_cached_tokens=0,
                kvpool_cached_tokens=self.N_BLOCKS * self.HYB_BLOCK_SIZE,
                can_load=True,
                kvpool_store_skip_tokens=None,
                token_len=0,
            ),
        )
        for b in range(self.N_BLOCKS):
            self.assertEqual(
                worker._holemiss_derive_block_key(request, b),
                worker._make_mooncake_layerwise_key(block_hashes[b].hex()),
            )
        self.assertIsNone(worker._holemiss_derive_block_key(request, self.N_BLOCKS))


class TestSyncDrain(HolemissTestBase):
    def test_drain_waits_all_layers_and_suppresses_native(self):
        worker = _make_worker()
        worker.kv_recv_thread = MagicMock()
        worker.layer_load_finished_events = [threading.Event() for _ in range(NUM_LAYERS)]
        for event in worker.layer_load_finished_events:
            event.set()
        worker.layer_load_tasks[0].append(object())
        worker.layer_load_tasks[2].append(object())

        worker._holemiss_sync_layerwise_drain([_make_request()])

        # The drain batches every layer's tasks into ONE queue entry: the
        # ported session read stages each key's object per CALL, so one
        # merged ranged call per key replaces the per-layer calls.
        self.assertEqual(worker.kv_recv_thread.add_request.call_count, 1)
        submitted = worker.kv_recv_thread.add_request.call_args_list[0].args[0]
        self.assertIsInstance(submitted, LayerBatchedLoadTask)
        self.assertEqual([task.layer_id for task in submitted.layer_tasks], [0, 2])
        self.assertTrue(all(task.attention_start_gate is None for task in submitted.layer_tasks))
        self.assertTrue(all(task.wait_for_save_layer is None for task in submitted.layer_tasks))
        self.assertEqual(worker.layer_load_tasks, [[] for _ in range(NUM_LAYERS)])
        self.assertEqual(worker.next_layer_to_submit, NUM_LAYERS)

    def test_drain_falls_back_for_gva(self):
        worker = _make_worker()
        worker.use_layerwise_transfer = True
        worker.kv_recv_thread = MagicMock()
        worker.layer_load_tasks[0].append(object())
        worker._holemiss_sync_layerwise_drain([_make_request()])
        worker.kv_recv_thread.add_request.assert_not_called()
        self.assertEqual(len(worker.layer_load_tasks[0]), 1)


if __name__ == "__main__":
    unittest.main()
