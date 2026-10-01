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
"""CPU-only unit tests for the hybrid finish-time-tail exemption in the
reachable-store masks (B38).

The live tail E2E arm (hybrid @9216, boundary 9472 = 9216 + 256 incoming)
died in ``_compute_reachable_store_masks``: the hybrid block-key branch
demands a coordinator-aligned full-block boundary and 9472 % 1536 = 256.
The finish-time tail ReqMeta (``register_finished_partial_tail``) carries
the boundary itself, and its only new artifact is the group-0 FA partial
block, which rides the partial-block transfer path outside the full-block
masks — so the mask computation exempts it and composes at the
floor-aligned boundary instead.
"""

import unittest

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)

GRANULARITY = 1536
BOUNDARY = 9472  # 9216 cached + 256 incoming (the live arm's tail boundary)
FLOOR = 9216
TAIL_BLOCK_INDEX = 6
GROUPED_BLOCK_SIZE = [GRANULARITY, GRANULARITY]
HASH_BLOCK_SIZE = 256


class _FakeTokenDatabase:
    """Mirrors ChunkedTokenDatabase.store_mask: None without a coordinator,
    the configured masks otherwise; records every call for composition pins."""

    def __init__(self, masks=None, coordinator=None):
        self.calls = []
        self.masks = masks
        self.cache_coordinator = coordinator

    def store_mask(self, aligned_token_len, num_prompt_tokens=None):
        self.calls.append((aligned_token_len, num_prompt_tokens))
        if self.cache_coordinator is None:
            return None
        return self.masks


class TestHybridTailStoreMaskExemption(unittest.TestCase):
    def _make_worker(self, coordinator="coordinator", masks=([True] * 6, [True] * 6)):
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.block_key_hybrid = True
        worker.cache_transfer_granularity = GRANULARITY
        worker.cache_coordinator = coordinator
        worker.token_database = _FakeTokenDatabase(masks=masks, coordinator=coordinator)
        return worker

    @staticmethod
    def _make_request(**overrides):
        params = dict(
            req_id="req-tail",
            token_len_chunk=0,
            save_end_token=BOUNDARY,
            save_start_token=BOUNDARY,
            target_token_len=BOUNDARY,
            block_ids_by_group=[[10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24, 25, 26]],
            block_hashes=[f"h{i:02d}" for i in range(7 * (GRANULARITY // HASH_BLOCK_SIZE))],
            can_save=True,
            is_last_chunk=True,
            partial_block_index=TAIL_BLOCK_INDEX,
            finished_tail_boundary=BOUNDARY,
            num_prompt_tokens=9728,
        )
        params.update(overrides)
        return ReqMeta(**params)

    def test_hybrid_tail_passes_with_floor_aligned_mask(self):
        """The B38 fix: the tail ReqMeta no longer raises; the masks are
        composed at the floor-aligned boundary (9472 -> 9216), the same
        coordinator reachable-state semantics as any aligned save."""
        worker = self._make_worker()
        request = self._make_request()
        result = worker._compute_reachable_store_masks(request)
        self.assertEqual(result, ([True] * 6, [True] * 6))
        self.assertEqual(worker.token_database.calls, [(FLOOR, 9728)])

    def test_hybrid_non_tail_unaligned_save_still_raises(self):
        """The regression guard: genuine misalignment of a non-tail hybrid
        save keeps raising (the pre-B38 contract)."""
        worker = self._make_worker()
        request = self._make_request(finished_tail_boundary=None, partial_block_index=None)
        with self.assertRaises(ValueError):
            worker._compute_reachable_store_masks(request)
        self.assertEqual(worker.token_database.calls, [])

    def test_hybrid_non_tail_without_coordinator_still_raises(self):
        worker = self._make_worker(coordinator=None)
        request = self._make_request(
            save_end_token=FLOOR,
            save_start_token=FLOOR,
            target_token_len=FLOOR,
            finished_tail_boundary=None,
            partial_block_index=None,
        )
        with self.assertRaises(ValueError):
            worker._compute_reachable_store_masks(request)

    def test_exemption_keys_on_the_positive_tail_signal(self):
        """The exemption never keys on the unaligned value alone: a partial
        block index without the finish-time boundary, or a boundary that
        disagrees with the save extent, keeps the aligned-or-raise contract."""
        worker = self._make_worker()
        partial_only = self._make_request(finished_tail_boundary=None)
        with self.assertRaises(ValueError):
            worker._compute_reachable_store_masks(partial_only)

        mismatched = self._make_request(finished_tail_boundary=FLOOR)
        with self.assertRaises(ValueError):
            worker._compute_reachable_store_masks(mismatched)
        self.assertEqual(worker.token_database.calls, [])

    def test_tail_flagged_aligned_boundary_behaves_as_today(self):
        """A tail-flagged request whose boundary is aligned computes the
        same mask as the non-tail aligned path (the floor of an aligned
        extent is the extent itself)."""
        worker = self._make_worker()
        request = self._make_request(
            save_end_token=FLOOR,
            save_start_token=FLOOR,
            target_token_len=FLOOR,
            finished_tail_boundary=FLOOR,
        )
        result = worker._compute_reachable_store_masks(request)
        self.assertEqual(result, ([True] * 6, [True] * 6))
        self.assertEqual(worker.token_database.calls, [(FLOOR, 9728)])

        non_tail = self._make_request(
            save_end_token=FLOOR,
            save_start_token=FLOOR,
            target_token_len=FLOOR,
            finished_tail_boundary=None,
            partial_block_index=None,
        )
        self.assertEqual(worker._compute_reachable_store_masks(non_tail), ([True] * 6, [True] * 6))
        self.assertEqual(worker.token_database.calls, [(FLOOR, 9728), (FLOOR, 9728)])

    def test_tail_without_coordinator_degrades_to_none(self):
        """The graceful mirror of ChunkedTokenDatabase.store_mask: a
        coordinator-less tail returns None instead of raising (the tail's
        one-block scope keeps the mask moot; the non-tail hybrid path
        still demands the coordinator)."""
        worker = self._make_worker(coordinator=None)
        request = self._make_request()
        self.assertIsNone(worker._compute_reachable_store_masks(request))
        self.assertEqual(worker.token_database.calls, [(FLOOR, 9728)])

    def test_hybrid_non_saving_request_skips_the_branch(self):
        """can_save=False or a zero extent never engages the hybrid branch
        (the load-only requests keep the graceful non-hybrid shape)."""
        worker = self._make_worker()
        self.assertIsNone(worker._compute_reachable_store_masks(self._make_request(can_save=False)))
        self.assertIsNone(
            worker._compute_reachable_store_masks(
                self._make_request(can_save=None, save_end_token=0, finished_tail_boundary=None)
            )
        )
        self.assertEqual(worker.token_database.calls, [])

    def test_non_hybrid_unaligned_stays_graceful_none(self):
        """The existing non-hybrid behavior, pinned locally: an unaligned
        extent on a non-hybrid worker degrades to None (store everything)."""
        worker = self._make_worker()
        worker.block_key_hybrid = False
        request = self._make_request(finished_tail_boundary=None, partial_block_index=None)
        self.assertIsNone(worker._compute_reachable_store_masks(request))
        self.assertEqual(worker.token_database.calls, [])


class TestHybridTailSaveBatchComposition(unittest.TestCase):
    """The mask composes with the save batch: the tail request's block
    ranges are the partial tail block alone, whatever the mask says."""

    def _make_worker(self, store_masks):
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.block_key_hybrid = True
        worker.use_block_key_layerwise = True
        worker.cache_transfer_granularity = GRANULARITY
        worker.cache_coordinator = "coordinator"
        worker.token_database = _FakeTokenDatabase(masks=store_masks, coordinator="coordinator")
        worker.tp_rank = 0
        worker.dcp_size = 1
        worker.put_step = 1
        worker.num_layers = 4
        worker.grouped_block_size = list(GROUPED_BLOCK_SIZE)
        worker.hash_block_size = HASH_BLOCK_SIZE
        worker.layerwise_offload = True
        worker.group_num_layers = {0: 2, 1: 2}
        worker.layer_save_tasks = [[] for _ in range(worker.num_layers)]
        return worker

    @staticmethod
    def _tail_request():
        return ReqMeta(
            req_id="req-tail",
            token_len_chunk=0,
            save_end_token=BOUNDARY,
            save_start_token=BOUNDARY,
            target_token_len=BOUNDARY,
            block_ids_by_group=[[10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24, 25, 26]],
            block_hashes=[f"h{i:02d}" for i in range(7 * (GRANULARITY // HASH_BLOCK_SIZE))],
            can_save=True,
            is_last_chunk=True,
            partial_block_index=TAIL_BLOCK_INDEX,
            finished_tail_boundary=BOUNDARY,
            num_prompt_tokens=9728,
        )

    def test_tail_ranges_are_the_partial_block_alone(self):
        """With the floor-aligned mask in place the tail request yields ONE
        block range per group: [6, 6) plus the partial block 6 — no
        full-block range is opened, so no GDN group session either."""
        worker = self._make_worker(store_masks=([True] * 6, [True] * 6))
        request = self._tail_request()
        request.store_masks = worker._compute_reachable_store_masks(request)
        self.assertIsNotNone(request.store_masks)
        worker._process_save_for_layer_batch([request], 0, group_id=0, layer_idx_in_group=0)
        ranges = worker.layer_save_tasks[0][0].block_ranges
        self.assertEqual(
            [(r.start_block, r.end_block, r.partial_block_index) for r in ranges],
            [(TAIL_BLOCK_INDEX, TAIL_BLOCK_INDEX, TAIL_BLOCK_INDEX)],
        )

    def test_tail_block_rides_even_a_fully_masked_boundary(self):
        """The mask is moot for the tail block by construction: even a mask
        that excludes every complete block leaves the partial tail range
        intact (the tail never consults the full-block mask)."""
        worker = self._make_worker(store_masks=([False] * 6, [False] * 6))
        request = self._tail_request()
        request.store_masks = worker._compute_reachable_store_masks(request)
        self.assertEqual(request.store_masks, ([False] * 6, [False] * 6))
        worker._process_save_for_layer_batch([request], 0, group_id=1, layer_idx_in_group=0)
        ranges = worker.layer_save_tasks[0][0].block_ranges
        self.assertEqual(
            [(r.start_block, r.end_block, r.partial_block_index) for r in ranges],
            [(TAIL_BLOCK_INDEX, TAIL_BLOCK_INDEX, TAIL_BLOCK_INDEX)],
        )


if __name__ == "__main__":
    unittest.main()
