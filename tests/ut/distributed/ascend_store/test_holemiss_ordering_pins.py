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
"""G20 ordering source pins for the T5 holemiss load/save seams.

These are text-order pins (not behavioural): they fail LOUD if a re-copy
renames/removes an anchor or reorders the statements, which is exactly the
regression mode the ordering law protects against.
"""

import inspect
import unittest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)


def _require_order(body: str, *anchors: str) -> None:
    positions: list[int] = []
    for anchor in anchors:
        if anchor not in body:
            raise AssertionError(f"ordering anchor not found: {anchor!r}")
        positions.append(body.index(anchor))
    for earlier, later, first, second in zip(positions, positions[1:], anchors, anchors[1:]):
        if earlier >= later:
            raise AssertionError(f"ordering violated: {first!r} must precede {second!r}")


class TestStartLoadKvOrdering(unittest.TestCase):
    def _source(self) -> str:
        return inspect.getsource(KVPoolWorker.start_load_kv)

    def test_configure_then_load_then_drain(self):
        _require_order(
            self._source(),
            "_holemiss_configure_fault(",
            "process_layer_data(",
            "_holemiss_sync_layerwise_drain(",
        )

    def test_every_anchor_present(self):
        source = self._source()
        for anchor in (
            "_holemiss_configure_fault(",
            "process_layer_data(",
            "_holemiss_sync_layerwise_drain(",
        ):
            self.assertIn(anchor, source)


class TestRangeRequestOrdering(unittest.TestCase):
    def _source(self) -> str:
        return inspect.getsource(KVCacheStoreLayerRecvingThread._handle_range_request)

    def test_region_get_before_invalid_marking(self):
        _require_order(
            self._source(),
            "_issue_range_get(",
            "_mark_invalid_range_indices(",
        )

    def test_every_anchor_present(self):
        source = self._source()
        for anchor in ("_issue_range_get(", "_mark_invalid_range_indices("):
            self.assertIn(anchor, source)


class TestWorkerRegionEmissionOrdering(unittest.TestCase):
    def _source(self) -> str:
        return inspect.getsource(KVPoolWorker.execute_holemiss_range_target)

    def test_region_recorded_before_results_returned(self):
        _require_order(self._source(), "record_hole_recovery_region(", "return results")


class TestPinBite(unittest.TestCase):
    """Negative controls: the pin helper itself must fail loud on reorder."""

    def test_bites_when_drain_moved_before_process_layer_data(self):
        reordered = (
            "self._holemiss_configure_fault(request)\n"
            "self._holemiss_sync_layerwise_drain(holemiss_load_requests)\n"
            "self.process_layer_data(metadata.requests)\n"
        )
        with self.assertRaises(AssertionError):
            _require_order(
                reordered,
                "_holemiss_configure_fault(",
                "process_layer_data(",
                "_holemiss_sync_layerwise_drain(",
            )

    def test_bites_when_invalid_marked_before_region_get(self):
        reordered = "self._mark_invalid_range_indices(req_meta, failed_indices)\nresults = self._issue_range_get(k)\n"
        with self.assertRaises(AssertionError):
            _require_order(reordered, "_issue_range_get(", "_mark_invalid_range_indices(")

    def test_bites_when_anchor_removed(self):
        with self.assertRaises(AssertionError):
            _require_order("process_layer_data(metadata.requests)\n", "_holemiss_configure_fault(")


if __name__ == "__main__":
    unittest.main()
