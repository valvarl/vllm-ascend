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
