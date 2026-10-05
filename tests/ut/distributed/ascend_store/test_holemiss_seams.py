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
"""CPU-only unit tests for the T5 holemiss peripheral seams.

Covers the recv-thread delegating region methods + the fault-aware range GET
routing, the connector HR payload passthrough with the HOLEMISS_BENCH_ROOT
fail-fast guard and the ``_hr_hook_active`` skips (which must NOT skip the
#17287 mamba state copy), and the backend ``put_end_disk`` passthrough.
"""

import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    AscendStoreConnector,
    holemiss_verify_bench_root,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    MooncakeBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
    KVTransferThread,
)

HOLEMISS_ENVS = ["HOLEMISS_BENCH_ROOT", "HOLEMISS_RUNTIME_PROFILE"]
_HOOK_TRIO = ("hole_recovery_hook.py", "hole_recovery_config.py", "timer.py")


class EnvBase(unittest.TestCase):
    def setUp(self):
        self._saved = {name: os.environ.get(name) for name in HOLEMISS_ENVS}
        for name in HOLEMISS_ENVS:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class TestRecvThreadDelegation(unittest.TestCase):
    def _thread(self, worker=None):
        thread = KVTransferThread.__new__(KVTransferThread)
        thread.worker = worker
        thread._hr_local_regions = []
        thread._hr_local_lock = threading.Lock()
        return thread

    def test_local_store_when_workerless(self):
        thread = self._thread(worker=None)
        thread.record_hole_recovery_region({"request_id": "r0"})
        self.assertEqual(thread.get_hole_recovery_regions(), [{"request_id": "r0"}])
        thread.clear_hole_recovery_regions()
        self.assertEqual(thread.get_hole_recovery_regions(), [])

    def test_delegates_to_worker_when_wired(self):
        worker = MagicMock()
        worker.get_hole_recovery_regions.return_value = [{"request_id": "w"}]
        thread = self._thread(worker=worker)
        thread.record_hole_recovery_region({"request_id": "w"})
        worker.record_hole_recovery_region.assert_called_once_with({"request_id": "w"})
        self.assertEqual(thread.get_hole_recovery_regions(), [{"request_id": "w"}])
        thread.clear_hole_recovery_regions()
        worker.clear_hole_recovery_regions.assert_called_once()
        self.assertEqual(thread._hr_local_regions, [])


class TestArmedRangeRouting(unittest.TestCase):
    def _thread(self, worker):
        thread = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
        thread.worker = worker
        thread.m_store = MagicMock()
        thread.max_transfer_blocks = 0
        thread.max_transfer_bytes = 0
        return thread

    def test_armed_target_routes_to_worker_and_returns_results(self):
        worker = MagicMock()
        worker._holemiss_armed_fault = {"key": "k"}
        worker.holemiss_range_target_position.return_value = 0
        worker.execute_holemiss_range_target.return_value = [-800]
        thread = self._thread(worker)
        results = thread._issue_range_get(["k"], [[1]], [[2]], [[0]], 0)
        self.assertEqual(results, [-800])
        worker.execute_holemiss_range_target.assert_called_once()

    def test_unarmed_uses_native_batch_path(self):
        worker = MagicMock()
        worker._holemiss_armed_fault = None
        thread = self._thread(worker)
        thread.m_store.batch_copy_get.return_value = [0]
        results = thread._issue_range_get(["k"], [[1]], [[2]], [[0]], 0)
        self.assertEqual(results, [0])
        worker.execute_holemiss_range_target.assert_not_called()

    def test_knob_set_armed_chunk_diverts_others_chunk(self):
        """F5 (D-MTX.5): with the transfer knob set the per-layer fault
        path rides the SAME chunked batches as the default arms — only the
        batch carrying the armed key diverts to the isolated executor (the
        pre-fix delegation handed the whole layer list over in ONE
        unchunked healthy-subset call, the file_storage BUFFER_OVERFLOW at
        >=2GiB/rank)."""
        worker = MagicMock()
        worker._holemiss_armed_fault = {"key": "k2"}
        worker.holemiss_range_target_position.side_effect = (
            lambda keys, layer: keys.index("k2") if "k2" in keys else None
        )
        worker.execute_holemiss_range_target.side_effect = lambda keys, buffers, sizes, offsets, target_pos: [
            -1102 if i == target_pos else 0 for i in range(len(keys))
        ]
        thread = self._thread(worker)
        thread.max_transfer_blocks = 2
        thread.m_store.batch_copy_get.side_effect = lambda keys, buffers, sizes, offsets: [0] * len(keys)
        results = thread._issue_range_get(
            ["k0", "k1", "k2", "k3"],
            [[1]] * 4,
            [[2]] * 4,
            [[0]] * 4,
            5,
        )
        self.assertEqual(results, [0, 0, -1102, 0])
        # The healthy keys before the armed batch rode ONE plain chunked
        # call of <= 2 keys; the armed batch never hit the plain path.
        self.assertEqual(
            [call.args[0] for call in thread.m_store.batch_copy_get.call_args_list],
            [["k0", "k1"]],
        )
        # The armed batch diverted whole, the armed key at chunk-local 0.
        worker.execute_holemiss_range_target.assert_called_once()
        divert_keys, divert_buffers, divert_sizes, divert_offsets, divert_pos = (
            worker.execute_holemiss_range_target.call_args.args
        )
        self.assertEqual(divert_keys, ["k2", "k3"])
        self.assertEqual(divert_pos, 0)


class TestConnectorRegions(EnvBase):
    def _connector(self, regions):
        connector = AscendStoreConnector.__new__(AscendStoreConnector)
        connector.connector_worker = MagicMock()
        connector.connector_worker.get_hole_recovery_regions.return_value = regions
        return connector

    def test_empty_regions_do_not_require_bench_root(self):
        connector = self._connector([])
        self.assertEqual(connector.get_hole_recovery_regions(), [])

    def test_regions_without_bench_root_raise_loud(self):
        connector = self._connector([{"result": -800}])
        with self.assertRaises(RuntimeError):
            connector.get_hole_recovery_regions()

    def test_regions_with_valid_bench_root_resolve(self):
        with tempfile.TemporaryDirectory() as root:
            hook_dir = Path(root) / "benchmarks" / "hole_recovery"
            hook_dir.mkdir(parents=True)
            for name in _HOOK_TRIO:
                (hook_dir / name).write_text("")
            os.environ["HOLEMISS_BENCH_ROOT"] = root
            connector = self._connector([{"result": -800}])
            self.assertEqual(len(connector.get_hole_recovery_regions()), 1)


class TestBenchRootGuard(EnvBase):
    def test_unset_root_raises(self):
        with self.assertRaises(RuntimeError):
            holemiss_verify_bench_root()

    def test_missing_trio_raises(self):
        with tempfile.TemporaryDirectory() as root:
            os.environ["HOLEMISS_BENCH_ROOT"] = root
            with self.assertRaises(RuntimeError):
                holemiss_verify_bench_root()

    def test_complete_trio_returns_root(self):
        with tempfile.TemporaryDirectory() as root:
            hook_dir = Path(root) / "benchmarks" / "hole_recovery"
            hook_dir.mkdir(parents=True)
            for name in _HOOK_TRIO:
                (hook_dir / name).write_text("")
            os.environ["HOLEMISS_BENCH_ROOT"] = root
            self.assertEqual(holemiss_verify_bench_root(), root)


class TestHrHookActiveSkips(unittest.TestCase):
    def _connector(self, hook_active, mamba_state=None):
        connector = AscendStoreConnector.__new__(AscendStoreConnector)
        connector.use_layerwise = True
        connector.connector_worker = MagicMock()
        connector._mamba_copy_bufs = None
        connector._mamba_state = mamba_state
        connector._hr_hook_active = hook_active
        return connector

    def test_hook_active_skips_worker_wait(self):
        connector = self._connector(hook_active=True)
        connector.wait_for_layer_load("layer.0")
        connector.connector_worker.wait_for_layer_load.assert_not_called()

    def test_hook_inactive_waits(self):
        connector = self._connector(hook_active=False)
        connector.wait_for_layer_load("layer.0")
        connector.connector_worker.wait_for_layer_load.assert_called_once()

    def test_hook_active_does_not_skip_mamba_state_copy(self):
        mamba_state = MagicMock()
        connector = self._connector(hook_active=True, mamba_state=mamba_state)
        connector.wait_for_layer_load("layer.0")
        connector.connector_worker.wait_for_layer_load.assert_not_called()
        mamba_state.do_mamba_copy_for_layer.assert_called_once_with("layer.0")

    def test_hook_active_skips_save(self):
        connector = self._connector(hook_active=True)
        connector.kv_role = "kv_producer"
        connector.consumer_is_to_put = False
        connector.save_kv_layer("layer.0", MagicMock(), MagicMock())
        connector.connector_worker.save_kv_layer.assert_not_called()


class TestBackendPutEndDisk(unittest.TestCase):
    def test_put_end_disk_passthrough(self):
        backend = MooncakeBackend.__new__(MooncakeBackend)
        backend.ensure_initialized = MagicMock()
        backend.store = MagicMock()
        backend.store.put_end_disk.return_value = True
        self.assertTrue(backend.put_end_disk("model@hash@0"))
        backend.store.put_end_disk.assert_called_once_with("model@hash@0")


if __name__ == "__main__":
    unittest.main()
