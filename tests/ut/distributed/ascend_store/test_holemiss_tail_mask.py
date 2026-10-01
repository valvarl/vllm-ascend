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

import threading
import unittest
from types import SimpleNamespace

from vllm.logger import logger as vllm_logger

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.layerwise_keys import (
    LayerwiseKeyBuilder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_layerwise import (
    _prepare_group_sessions,
    hybrid_block_key,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.session_tracker import (
    LayerwiseSessionTracker,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    LayerBatchBuilder,
)
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
MODEL = "testmodel"
LAYOUT = "deadbeefcafe"
GROUP0_OBJECT_SIZE = 3145728 * 16  # sum(group_block_len[0]): the D7 group-0 block object


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


class _SessionStore:
    """Backend-seam fake for the hybrid group put-session machinery."""

    requires_exists_before_put = True

    def __init__(self, exists=None, put_start_rc=None):
        self.exists = dict(exists or {})
        self.put_start_rc = put_start_rc
        self.put_starts: list[tuple[tuple, tuple]] = []

    def batch_is_exist(self, keys):
        return [self.exists.get(key, 0) for key in keys]

    def batch_put_start(self, keys, sizes):
        self.put_starts.append((tuple(keys), tuple(sizes)))
        if self.put_start_rc is not None:
            return [self.put_start_rc] * len(keys)
        return [0] * len(keys)


def _tail_chain_hashes(token_count: int, hbs: int):
    """The engine's chained block hashes, built the way the bench builds them
    (kvcu.hash_block_tokens over hash-block-size units from NONE_HASH) — the
    byte-mirror source for the key-equality pin."""
    import vllm.v1.core.kv_cache_utils as kvcu
    from vllm.utils.hashing import sha256

    kvcu.init_none_hash(sha256)
    parent = kvcu.NONE_HASH
    chain = []
    for start in range(0, token_count, hbs):
        parent = kvcu.hash_block_tokens(sha256, parent, list(range(start, start + hbs)), None)
        chain.append(parent)
    return chain


class TestHybridFinishedTailKeyConstruction(unittest.TestCase):
    """B39: the group-0 view of a finish-time tail request carries the hybrid
    tail key and its put session; groups 1-3 keep clearing (the T18 law)."""

    def _make_session_worker(self, store):
        def make_key(group, block_hash, head, stage):
            return hybrid_block_key(MODEL, LAYOUT, group, GROUPED_BLOCK_SIZE[group], block_hash, head)

        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.block_key_hybrid = True
        worker.cache_coordinator = "coordinator"
        worker.cache_transfer_granularity = GRANULARITY
        worker.token_database = SimpleNamespace(
            load_mask=lambda hashes, tokens: None,
            store_mask=lambda aligned_token_len, num_prompt_tokens=None: ([True] * 6, [True] * 6),
        )
        worker.num_kv_cache_groups = 2
        worker.grouped_block_size = list(GROUPED_BLOCK_SIZE)
        worker.hash_block_size = HASH_BLOCK_SIZE
        worker.head_or_tp_rank = 0
        worker.pp_rank = 0
        worker.layerwise_keys = LayerwiseKeyBuilder(make_key, 1)
        worker.kv_role = "kv_producer"
        worker.consumer_is_to_put = False
        worker.tp_rank = 0
        worker.put_step = 1
        worker.dcp_size = 1
        worker.layerwise_max_transfer_blocks = 0
        worker.group_block_len = {0: [3145728] * 16, 1: [534528] * 48}
        worker.group_num_layers = {0: 2, 1: 2}
        worker.num_layers = 4
        worker.use_block_key_layerwise = True
        worker.layerwise_offload = False
        worker.layer_save_tasks = [[] for _ in range(worker.num_layers)]
        worker._layer_load_aborted = threading.Event()
        worker._put_started_keys = set()
        worker._put_started_keys_lock = threading.Lock()
        worker._layerwise_session_tracker = LayerwiseSessionTracker()
        worker.m_store = store
        return worker

    @staticmethod
    def _tail_request(req_id="req-tail", block_hashes=None, **overrides):
        params = dict(
            req_id=req_id,
            token_len_chunk=0,
            save_end_token=BOUNDARY,
            save_start_token=BOUNDARY,
            target_token_len=BOUNDARY,
            block_ids_by_group=[[10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24, 25, 26]],
            block_hashes=block_hashes
            if block_hashes is not None
            else [f"h{i:02d}" for i in range(7 * (GRANULARITY // HASH_BLOCK_SIZE))],
            can_save=True,
            is_last_chunk=True,
            partial_block_index=TAIL_BLOCK_INDEX,
            finished_tail_boundary=BOUNDARY,
            num_prompt_tokens=9728,
        )
        params.update(overrides)
        return ReqMeta(**params)

    def test_group0_view_carries_the_tail_key_and_opens_the_put_session(self):
        store = _SessionStore()
        worker = self._make_session_worker(store)
        request = self._tail_request()
        views = _prepare_group_sessions(worker, [request])
        tail = worker._make_layerwise_finished_tail_key(request, 0)
        self.assertIsNotNone(tail)
        view0 = views[0][0]
        self.assertEqual(view0.save_last_block_key, tail[0])
        self.assertEqual(view0.partial_block_index, TAIL_BLOCK_INDEX)
        self.assertEqual(store.put_starts, [((tail[0],), (GROUP0_OBJECT_SIZE,))])

    def test_non_group0_views_stay_cleared(self):
        """The T18 law: only the FA partial block is the new tail artifact —
        the GDN views keep the cleared state and open no session. Scoped to
        group 1 ONLY (the group-0 behavior is (a)'s pin) so this test is the
        mutation's scoping guard: it must stay green when the construction
        is removed."""
        store = _SessionStore()
        worker = self._make_session_worker(store)
        request = self._tail_request()
        views = _prepare_group_sessions(worker, [request])
        self.assertIsNone(worker._make_layerwise_finished_tail_key(request, 1))
        view1 = views[1][0]
        self.assertIsNone(view1.save_last_block_key)
        self.assertIsNone(view1.partial_block_index)
        for keys, _ in store.put_starts:
            for name in keys:
                self.assertNotIn("group:1", name)

    def test_non_tail_requests_get_no_tail_key(self):
        """The positive-signal gate: a regular (non-finish-time) request with
        a partial trailing block opens no tail session — the req-id _lastblock
        key never appears on the hybrid path."""
        store = _SessionStore()
        worker = self._make_session_worker(store)
        request = self._tail_request(finished_tail_boundary=None)
        self.assertIsNone(worker._make_layerwise_finished_tail_key(request, 0))
        views = _prepare_group_sessions(worker, [request])
        self.assertIsNone(views[0][0].save_last_block_key)
        self.assertEqual(store.put_starts, [])

    def test_partial_row_lands_in_the_key_major_shared_build(self):
        """The composition proof: the group-0 view flows through the range
        derivation (layer reuse OFF — the un-cleared partial_block_index is
        the load-bearing fallback) into _build_key_major_shared's partial row
        with the tail block id and the tail key."""
        store = _SessionStore()
        worker = self._make_session_worker(store)
        request = self._tail_request()
        request.store_masks = worker._compute_reachable_store_masks(request)
        views = _prepare_group_sessions(worker, [request])
        view0 = views[0][0]
        worker._process_save_for_layer_batch([view0], 0, group_id=0, layer_idx_in_group=0)
        tasks = worker.layer_save_tasks[0]
        self.assertEqual(len(tasks), 1)
        ranges = tasks[0].block_ranges
        self.assertEqual(
            [(r.start_block, r.end_block, r.partial_block_index) for r in ranges],
            [(TAIL_BLOCK_INDEX, TAIL_BLOCK_INDEX, TAIL_BLOCK_INDEX)],
        )
        builder = LayerBatchBuilder.__new__(LayerBatchBuilder)
        builder.group_id = 0
        shared = builder._build_key_major_shared(tasks[0], is_save=True)
        self.assertIsNotNone(shared)
        self.assertEqual(shared.block_ids_arr.tolist(), [16])
        self.assertEqual(shared.block_keys, [view0.save_last_block_key])

    def test_tail_key_matches_the_bench_expected_form(self):
        """The byte-equality pin: the va key equals the bench's
        expected_partial_tail_keys construction (the boundary sub-hash = the
        chain element at boundary//hbs - 1, suffixed _tail_{boundary}, wrapped
        in the group-0 D7 hybrid key) — mirrored locally from the bench's
        derivation, never imported."""
        chain = _tail_chain_hashes(BOUNDARY, HASH_BLOCK_SIZE)
        sub_hash = chain[BOUNDARY // HASH_BLOCK_SIZE - 1].hex()
        expected = hybrid_block_key(MODEL, LAYOUT, 0, GRANULARITY, f"{sub_hash}_tail_{BOUNDARY}", 0)
        store = _SessionStore()
        worker = self._make_session_worker(store)
        request = self._tail_request(block_hashes=chain)
        tail = worker._make_layerwise_finished_tail_key(request, 0)
        self.assertEqual(tail[0], expected)
        views = _prepare_group_sessions(worker, [request])
        self.assertEqual(views[0][0].save_last_block_key, expected)
        self.assertIn("group:0", expected)
        self.assertIn(f"@{sub_hash}_tail_{BOUNDARY}@", expected)

    def test_second_same_boundary_request_does_not_reput(self):
        """The never-re-put law: the tail key is cross-request-derivable, so a
        second request sharing the boundary must not re-open the session."""
        store = _SessionStore()
        worker = self._make_session_worker(store)
        first = self._tail_request(req_id="req-A")
        second = self._tail_request(req_id="req-B")
        _prepare_group_sessions(worker, [first])
        self.assertEqual(len(store.put_starts), 1)
        views = _prepare_group_sessions(worker, [second])
        self.assertEqual(len(store.put_starts), 1)
        self.assertEqual(views[0][0].save_last_block_key, store.put_starts[0][0][0])

    def test_committed_tail_key_is_not_reput_and_drops_the_row(self):
        """The pooled twin: a tail key already committed in the store is
        filtered out — no put_start, and the view keeps the key clear so the
        partial row is skipped (no write into a committed object)."""
        worker_probe = self._make_session_worker(_SessionStore())
        key = worker_probe._make_layerwise_finished_tail_key(self._tail_request(), 0)[0]
        store = _SessionStore(exists={key: 1})
        worker = self._make_session_worker(store)
        request = self._tail_request()
        views = _prepare_group_sessions(worker, [request])
        self.assertEqual(store.put_starts, [])
        self.assertIsNone(views[0][0].save_last_block_key)
        self.assertIsNone(views[0][0].partial_block_index)

    def test_rejected_put_start_is_loud_and_drops_the_row(self):
        """The loud path: a rejected put_start warns and leaves the view's
        tail key clear (the row is skipped) — mirroring the full-block slots.
        The put_start was ATTEMPTED (one call, the group-0 size) and its -705
        result is what drops the row."""
        store = _SessionStore(put_start_rc=-705)
        worker = self._make_session_worker(store)
        request = self._tail_request()
        with self.assertLogs(vllm_logger, level="WARNING") as logs:
            views = _prepare_group_sessions(worker, [request])
        self.assertEqual(len(store.put_starts), 1)
        self.assertEqual(store.put_starts[0][1], (GROUP0_OBJECT_SIZE,))
        self.assertIsNone(views[0][0].save_last_block_key)
        self.assertIsNone(views[0][0].partial_block_index)
        self.assertTrue(any("put_start rejected" in line and "group 0" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
