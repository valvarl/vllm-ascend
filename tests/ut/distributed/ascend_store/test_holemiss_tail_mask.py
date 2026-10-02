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

import hashlib
import inspect
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
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
    KVCacheStoreLayerSendingThread,
    LayerBatchBuilder,
    LayerRangeReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
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
    """Backend-seam fake for the hybrid group put/get session machinery."""

    requires_exists_before_put = True

    def __init__(self, exists=None, put_start_rc=None):
        self.exists = dict(exists or {})
        self.put_start_rc = put_start_rc
        self.put_starts: list[tuple[tuple, tuple]] = []
        self.get_starts: list[tuple] = []

    def batch_is_exist(self, keys):
        return [self.exists.get(key, 0) for key in keys]

    def batch_put_start(self, keys, sizes):
        self.put_starts.append((tuple(keys), tuple(sizes)))
        if self.put_start_rc is not None:
            return [self.put_start_rc] * len(keys)
        return [0] * len(keys)

    def batch_get_start(self, keys):
        self.get_starts.append(tuple(keys))
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


def _make_tail_session_worker(store):
    """The shared B39/B41 fake worker: the hybrid group-session machinery on
    a __new__ stub (the real _prepare_group_sessions drives it)."""

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
    worker.use_eagle = False
    worker.layerwise_offload = False
    worker.layer_save_tasks = [[] for _ in range(worker.num_layers)]
    worker._layer_load_aborted = threading.Event()
    worker._put_started_keys = set()
    worker._put_started_keys_lock = threading.Lock()
    worker._layerwise_session_tracker = LayerwiseSessionTracker()
    worker._holemiss_failed_load_keys = {}
    worker._holemiss_tail_key_boundaries = {}
    worker._holemiss_tail_save_probed = set()
    worker._record_layerwise_invalid_blocks = lambda block_ids: None
    worker.layer_load_tasks = [[] for _ in range(worker.num_layers)]
    worker.m_store = store
    return worker


def _tail_save_request(req_id="req-tail", block_hashes=None, **overrides):
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


class TestHybridFinishedTailKeyConstruction(unittest.TestCase):
    """B39: the group-0 view of a finish-time tail request carries the hybrid
    tail key and its put session; groups 1-3 keep clearing (the T18 law)."""

    _make_session_worker = staticmethod(_make_tail_session_worker)
    _tail_request = staticmethod(_tail_save_request)

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


class TestHybridTailLoadFetch(unittest.TestCase):
    """B41: the exact-end hit's group-0 partial tail block is FETCHED — the
    load-side mirror of the B39 save key. The live W2R2B evidence: the probe
    HIT (exact_end=9472) but the LOAD census fetched blocks 0-5 only — the
    scheduler had claimed 9471 cached tokens over 256 ZEROED pool pages (the
    zero-tail hazard: the T2 sequence was the zero class, not the anchor's)."""

    _make_session_worker = staticmethod(_make_tail_session_worker)
    _tail_request = staticmethod(_tail_save_request)

    @staticmethod
    def _tail_load_request(req_id="req-load", block_hashes=None, **load_overrides):
        load_params = dict(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=BOUNDARY - 1,
            can_load=True,
            kvpool_store_skip_tokens=BOUNDARY,
            tail_boundary_tokens=BOUNDARY,
        )
        load_params.update(load_overrides)
        return ReqMeta(
            req_id=req_id,
            token_len_chunk=0,
            save_end_token=BOUNDARY,
            save_start_token=BOUNDARY,
            target_token_len=BOUNDARY,
            block_ids_by_group=[[10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24, 25, 26]],
            block_hashes=block_hashes
            if block_hashes is not None
            else [f"h{i:02d}" for i in range(7 * (GRANULARITY // HASH_BLOCK_SIZE))],
            can_save=False,
            is_last_chunk=True,
            num_prompt_tokens=BOUNDARY,
            load_spec=LoadSpec(**load_params),
        )

    def test_group0_view_fetches_the_tail_key(self):
        store = _SessionStore()
        worker = self._make_session_worker(store)
        request = self._tail_load_request()
        with self.assertLogs(vllm_logger, level="INFO") as logs:
            views = _prepare_group_sessions(worker, [request])
        view0 = views[0][0]
        tail = worker._make_layerwise_tail_load_key(request, 0)
        self.assertIsNotNone(tail)
        self.assertEqual(view0.load_block_keys[TAIL_BLOCK_INDEX], tail[0])
        # The full-block entries are unchanged: slots 0-5 carry the full keys,
        # the tail extends (never replaces) the populated range.
        self.assertEqual(len(view0.load_block_keys), 7)
        self.assertTrue(all(view0.load_block_keys[i] is not None for i in range(7)))
        # The get session opened for the tail key.
        self.assertIn(tail[0], [key for keys in store.get_starts for key in keys])
        marker_lines = [line for line in logs.output if "HOLEMISS_TAIL_KEY_LOAD" in line]
        self.assertEqual(len(marker_lines), 1)

    def test_tail_load_key_equals_the_save_key(self):
        """The mirror pin: the load fetches byte-exactly the key the B39 save
        path persisted (the same raw-chain sub-hash, the same D7 form)."""
        store = _SessionStore()
        worker = self._make_session_worker(store)
        request = self._tail_load_request()
        load_tail = worker._make_layerwise_tail_load_key(request, 0)
        save_request = self._tail_request()
        save_tail = worker._make_layerwise_finished_tail_key(save_request, 0)
        self.assertEqual(load_tail[0], save_tail[0])
        self.assertEqual(load_tail[1], save_tail[1])
        self.assertIn(f"_tail_{BOUNDARY}", load_tail[0])

    def test_group1_never_fetches_a_tail_artifact(self):
        store = _SessionStore()
        worker = self._make_session_worker(store)
        request = self._tail_load_request()
        self.assertIsNone(worker._make_layerwise_tail_load_key(request, 1))
        views = _prepare_group_sessions(worker, [request])
        view1 = views[1][0]
        self.assertEqual(len(view1.load_block_keys), 6)
        self.assertNotIn(f"_tail_{BOUNDARY}", [k for k in view1.load_block_keys if k])

    def test_no_tail_signal_leaves_the_full_block_entries_unchanged(self):
        """The regression guard: without the load spec's tail signal the get
        entries are exactly the pre-B41 full-block set."""
        store = _SessionStore()
        worker = self._make_session_worker(store)
        request = self._tail_load_request(tail_boundary_tokens=None)
        self.assertIsNone(worker._make_layerwise_tail_load_key(request, 0))
        with self.assertNoLogs(vllm_logger, level="INFO"):
            views = _prepare_group_sessions(worker, [request])
        self.assertEqual(len(views[0][0].load_block_keys), 6)
        self.assertEqual(len(views[1][0].load_block_keys), 6)

    def test_aligned_extent_fetches_no_tail(self):
        store = _SessionStore()
        worker = self._make_session_worker(store)
        request = self._tail_load_request(
            kvpool_cached_tokens=FLOOR,
            kvpool_store_skip_tokens=FLOOR,
        )
        self.assertIsNone(worker._make_layerwise_tail_load_key(request, 0))

    def test_mismatched_boundary_fetches_no_tail(self):
        """A boundary that does not extend the cached extent's own partial
        block degrades to the full-block fetch — never a claimed-but-unfetched
        tail (the correctness invariant). The mismatch must floor to a
        DIFFERENT block than the claim's partial block (BOUNDARY-256 floors
        to the same block 6 — not a mismatch)."""
        store = _SessionStore()
        worker = self._make_session_worker(store)
        request = self._tail_load_request(tail_boundary_tokens=BOUNDARY - GRANULARITY)
        self.assertIsNone(worker._make_layerwise_tail_load_key(request, 0))

    def test_claimed_equals_fetched_composition(self):
        """The populated-slot range covers the claim: the scheduler claimed
        9471 tokens (7 blocks); the loaded key set covers blocks 0-6."""
        store = _SessionStore()
        worker = self._make_session_worker(store)
        request = self._tail_load_request()
        views = _prepare_group_sessions(worker, [request])
        view0 = views[0][0]
        worker._process_load_for_layer_batch([view0], 0, group_id=0, layer_idx_in_group=0)
        tasks = worker.layer_load_tasks[0]
        self.assertEqual(len(tasks), 1)
        ranges = tasks[0].block_ranges
        self.assertEqual(
            [(r.start_block, r.end_block) for r in ranges],
            [(0, TAIL_BLOCK_INDEX + 1)],
        )
        claimed_blocks = -(-request.load_spec.kvpool_cached_tokens // GRANULARITY)
        self.assertEqual(claimed_blocks, TAIL_BLOCK_INDEX + 1)
        self.assertEqual(max(r.end_block for r in ranges), claimed_blocks)


class TestHybridTailContentProbe(unittest.TestCase):
    """B43: the tail content probes pin the zero-class arithmetic exactly,
    against the LIVE kernel-granularity cache layout.

    The W2R2C run's content probes delivered a verdict on falsified
    instruments (the fourth fake-vs-live gap): the caches are
    KERNEL-granular — the live boot log shows ``num_blocks: 510`` pages with
    ``block_shape: [128, 2, 256]`` rows — so ``cache[block_id]`` with a POOL
    block id reads kernel row ``block_id`` (page 0's content on the hybrid;
    the load-side ``tail_nonzero=True`` was measured on page 0's kernel
    block, not the tail page), and the save-side marker never emitted (the
    sending thread was constructed without a worker reference). This class
    models the TRUE live geometry — 128-token kernel rows, 12 rows per
    1536-token pool page, block 1536, hash block 256, boundary 9472 = 9216 +
    256, tail block 6, the group-0 object 50331648 = 16 FA layers x 3145728
    (32 entries x 12 rows x 128 tokens x 1024 bytes) — so the probes are
    pinned against the exact live numbers AND the page addressing.
    """

    NUM_FA_LAYERS = 16
    KV_HEADS = 4
    HEAD_DIM = 128
    KERNEL_BLOCK = 128
    BPKB = GRANULARITY // KERNEL_BLOCK  # 12 kernel rows per pool page
    TOKEN_BYTES = KV_HEADS * HEAD_DIM * 2  # bf16
    ENTRY_BLOCK_BYTES = GRANULARITY * TOKEN_BYTES  # 1572864 per entry per page
    GROUP0_OBJECT_BYTES = NUM_FA_LAYERS * 2 * ENTRY_BLOCK_BYTES  # 50331648
    NUM_BLOCKS = 12  # pool pages
    SAVE_BLOCK = 6
    TAIL_TOKENS = BOUNDARY - FLOOR  # 256

    def _fa_caches(self, populate_block=None):
        """The group-0 FA caches in the TRUE live layout: 16 attention layers
        at physical indices 3+4k, each a (K, V) pair over 128-token KERNEL
        rows (12 rows per 1536-token pool page — the dual-block-space law)."""
        caches = {}
        for layer_idx in range(self.NUM_FA_LAYERS):
            physical = 3 + 4 * layer_idx
            k = torch.zeros(
                self.NUM_BLOCKS * self.BPKB,
                self.KERNEL_BLOCK,
                self.KV_HEADS,
                self.HEAD_DIM,
                dtype=torch.bfloat16,
            )
            v = torch.zeros(
                self.NUM_BLOCKS * self.BPKB,
                self.KERNEL_BLOCK,
                self.KV_HEADS,
                self.HEAD_DIM,
                dtype=torch.bfloat16,
            )
            if populate_block is not None:
                gen = torch.Generator().manual_seed(physical)
                row0 = populate_block * self.BPKB
                rows = -(-self.TAIL_TOKENS // self.KERNEL_BLOCK)  # the attended window's rows
                k[row0 : row0 + rows] = torch.randn(
                    rows,
                    self.KERNEL_BLOCK,
                    self.KV_HEADS,
                    self.HEAD_DIM,
                    generator=gen,
                    dtype=torch.bfloat16,
                )
                v[row0 : row0 + rows] = torch.randn(
                    rows,
                    self.KERNEL_BLOCK,
                    self.KV_HEADS,
                    self.HEAD_DIM,
                    generator=gen,
                    dtype=torch.bfloat16,
                )
            caches[f"model.layers.{physical}.self_attn"] = (k, v)
        return caches

    def _probe_worker(self, caches):
        def make_key(group, block_hash, head, stage):
            return hybrid_block_key(MODEL, LAYOUT, group, GROUPED_BLOCK_SIZE[group], block_hash, head)

        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.tp_rank = 0
        worker.head_or_tp_rank = 0
        worker.pp_rank = 0
        worker.pp_size = 1
        worker.num_layers = 64
        worker.hf_config = None
        worker.use_hybrid = True
        worker.kv_cache_config = None
        worker.kv_caches = caches
        worker.num_blocks = self.NUM_BLOCKS
        worker._global_to_local_layer = {}
        worker.grouped_block_size = list(GROUPED_BLOCK_SIZE)
        worker.hash_block_size = HASH_BLOCK_SIZE
        worker.use_eagle = False
        worker.layerwise_keys = LayerwiseKeyBuilder(make_key, 1)
        worker._holemiss_tail_key_boundaries = {}
        worker._holemiss_tail_save_probed = set()
        return worker

    def _tail_key(self, worker):
        request = _tail_save_request()
        return worker._make_layerwise_finished_tail_key(request, 0)

    def test_group0_object_bytes_match_the_live_geometry(self):
        """The entry walk covers exactly the live group-0 object: 32 entries
        (16 layers x K,V) x 12 kernel rows x 128 tokens = 50331648 bytes per
        pool page; the caches carry num_pages * 12 kernel rows."""
        worker = self._probe_worker(self._fa_caches())
        entries = worker._holemiss_group0_cache_entries()
        self.assertEqual(len(entries), self.NUM_FA_LAYERS * 2)
        per_page = sum(cache[0].numel() * cache.element_size() * self.BPKB for cache in entries)
        self.assertEqual(per_page, self.GROUP0_OBJECT_BYTES)
        self.assertEqual(
            entries[0].shape,
            (self.NUM_BLOCKS * self.BPKB, self.KERNEL_BLOCK, self.KV_HEADS, self.HEAD_DIM),
        )

    def test_page_view_addresses_the_pool_page_not_one_kernel_row(self):
        """The B13 pin: the page view for pool block 6 spans kernel rows
        [72, 84) — 12 rows — while the raw block id would read kernel row 6
        (page 0's content, the W2R2C instrument bug)."""
        worker = self._probe_worker(self._fa_caches())
        cache = worker._holemiss_group0_cache_entries()[0]
        page = worker._holemiss_group0_page_view(cache, self.SAVE_BLOCK)
        self.assertEqual(page.shape[0], self.BPKB)
        self.assertEqual(page.data_ptr(), cache[self.SAVE_BLOCK * self.BPKB].data_ptr())

    def test_save_probe_emits_the_attended_window_fingerprint(self):
        """The save-side probe at the drain: pool page 6 carries real KV in
        the attended window (kernel rows [72, 74) — the first 256 tokens) and
        allocation zeros beyond — the marker carries the whole-PAGE sha (all
        12 kernel rows, independently recomputed here via the direct row
        slice), tail_tokens=256, and tail_nonzero=True; the once-per-key
        guard suppresses the 15 later layer visits."""
        worker = self._probe_worker(self._fa_caches(populate_block=self.SAVE_BLOCK))
        key, tail_index = self._tail_key(worker)
        self.assertEqual(tail_index, TAIL_BLOCK_INDEX)
        self.assertEqual(self.TAIL_TOKENS, 256)
        with self.assertLogs(vllm_logger, level="INFO") as logs:
            worker.holemiss_tail_save_content_sha(key, self.SAVE_BLOCK)
            worker.holemiss_tail_save_content_sha(key, self.SAVE_BLOCK)
        marker_lines = [line for line in logs.output if "HOLEMISS_TAIL_CONTENT_SHA" in line]
        self.assertEqual(len(marker_lines), 1)
        line = marker_lines[0]
        self.assertIn("side=save", line)
        self.assertIn(f"boundary={BOUNDARY}", line)
        self.assertIn(f"tail_tokens={self.TAIL_TOKENS}", line)
        self.assertIn(f"block={self.SAVE_BLOCK}", line)
        self.assertIn("tail_nonzero=True", line)
        row0 = self.SAVE_BLOCK * self.BPKB
        expected = hashlib.sha256()
        for cache in worker._holemiss_group0_cache_entries():
            page_rows = cache[row0 : row0 + self.BPKB]
            expected.update(page_rows.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        self.assertIn(f"sha={expected.hexdigest()}", line)
        # The page's unattended rows (beyond the window) carry allocation
        # zeros — the whole-page sha still differs from a zero page.
        zero_worker = self._probe_worker(self._fa_caches())
        zero_key, _ = self._tail_key(zero_worker)
        with self.assertLogs(vllm_logger, level="INFO") as zero_logs:
            zero_worker.holemiss_tail_save_content_sha(zero_key, self.SAVE_BLOCK)
        zero_line = [ln for ln in zero_logs.output if "HOLEMISS_TAIL_CONTENT_SHA" in ln][0]
        self.assertNotIn(f"sha={expected.hexdigest()}", zero_line)

    def test_save_probe_zero_pages_report_the_zero_class(self):
        """The discrimination's save-side pin: block 6 all zeros (the pages
        the producer's drain would read if the prefill never landed there)
        -> tail_nonzero=False, the SAVE-side seam verdict."""
        worker = self._probe_worker(self._fa_caches())
        key, _ = self._tail_key(worker)
        with self.assertLogs(vllm_logger, level="INFO") as logs:
            worker.holemiss_tail_save_content_sha(key, self.SAVE_BLOCK)
        marker_lines = [line for line in logs.output if "HOLEMISS_TAIL_CONTENT_SHA" in line]
        self.assertEqual(len(marker_lines), 1)
        self.assertIn("tail_nonzero=False", marker_lines[0])

    def test_load_probe_fingerprints_the_destination_block(self):
        """The load-side probe post-drain: the same attended-window arithmetic
        over the consumer's block-table entry 6 under the tail key the B41
        fetch opened."""
        worker = self._probe_worker(self._fa_caches(populate_block=self.SAVE_BLOCK))
        request = TestHybridTailLoadFetch._tail_load_request()
        request.block_ids_by_group = [[100, 101, 102, 103, 104, 105, self.SAVE_BLOCK], [200] * 7]
        with self.assertLogs(vllm_logger, level="INFO") as logs:
            worker._holemiss_probe_tail_content_sha(request)
        marker_lines = [line for line in logs.output if "HOLEMISS_TAIL_CONTENT_SHA" in line]
        self.assertEqual(len(marker_lines), 1)
        line = marker_lines[0]
        self.assertIn("side=load", line)
        self.assertIn(f"boundary={BOUNDARY}", line)
        self.assertIn(f"tail_tokens={self.TAIL_TOKENS}", line)
        self.assertIn(f"block={self.SAVE_BLOCK}", line)
        self.assertIn("tail_nonzero=True", line)

    def test_save_and_load_probes_agree_on_the_same_bytes(self):
        """The round-trip pair: the save-side and load-side whole-block shas
        are byte-identical over the same content (the LEG-A walk both ends
        ride)."""
        caches = self._fa_caches(populate_block=self.SAVE_BLOCK)
        save_worker = self._probe_worker(caches)
        load_worker = self._probe_worker(caches)
        key, _ = self._tail_key(save_worker)
        request = TestHybridTailLoadFetch._tail_load_request()
        request.block_ids_by_group = [[100, 101, 102, 103, 104, 105, self.SAVE_BLOCK], [200] * 7]
        with self.assertLogs(vllm_logger, level="INFO") as logs:
            save_worker.holemiss_tail_save_content_sha(key, self.SAVE_BLOCK)
            load_worker._holemiss_probe_tail_content_sha(request)
        marker_lines = [line for line in logs.output if "HOLEMISS_TAIL_CONTENT_SHA" in line]
        self.assertEqual(len(marker_lines), 2)
        shas = {line.split("sha=")[1].split()[0] for line in marker_lines}
        self.assertEqual(len(shas), 1)

    def test_send_thread_hook_probes_registered_tail_keys_once(self):
        """The send-thread hook: a LayerRangeReqMeta carrying a tail key
        fingerprints its block through the worker before the ranged read; a
        plain full-block key never triggers the worker hook."""
        probed = []

        class _HookWorker:
            def holemiss_tail_save_content_sha(self, key, block_id):
                probed.append((key, block_id))

        thread = KVCacheStoreLayerSendingThread.__new__(KVCacheStoreLayerSendingThread)
        thread.worker = _HookWorker()
        tail_key = f"model@layout@group:0@block:1536@sub_tail_{BOUNDARY}@0"
        plain_key = "model@layout@group:0@block:1536@fullhash@0"
        req_meta = LayerRangeReqMeta(
            req_ids=["req-tail"],
            layer_id=3,
            block_ids=[41, self.SAVE_BLOCK],
            keys=[plain_key, tail_key],
            all_buffers=[[0], [0]],
            all_sizes=[[1], [1]],
            all_offsets=[[0], [0]],
        )
        thread._probe_tail_save_content(req_meta)
        self.assertEqual(probed, [(tail_key, self.SAVE_BLOCK)])
        self.assertNotIn(plain_key, [key for key, _ in probed])

    def test_unregistered_tail_key_probes_nothing(self):
        """A foreign _tail_ key (no registration — e.g. a key derived by
        another worker instance) emits no marker: the probe never guesses
        the boundary."""
        worker = self._probe_worker(self._fa_caches())
        with self.assertNoLogs(vllm_logger, level="INFO"):
            worker.holemiss_tail_save_content_sha("model@x@group:0@block:1536@sub_tail_9999@0", 3)


class TestTailSaveProbeWiring(unittest.TestCase):
    """The B43 follow-up: the save-side probe's worker wiring. The W2R2C
    live run proved the gap — the side=save marker never emitted because the
    block-key sending thread was constructed WITHOUT a worker reference (the
    base class defaults worker=None, so _probe_tail_save_content silently
    no-ops); the earlier UTs attached a fake worker to a __new__ thread stub
    so the suite could not catch it. These tests exercise the REAL binding
    surfaces: the constructor accepts-and-stores the worker, and the
    construction site (KVPoolWorker._start_kv_transfer_threads) passes
    worker=self."""

    @staticmethod
    def _token_database():
        database = ChunkedTokenDatabase([KeyMetadata("model", 0, 0, 0)], [16], None)
        database.set_group_buffers(
            {0: [1000, 2000, 3000]},
            {0: [10, 20, 30]},
            {0: [100, 200, 300]},
            group_num_layers={0: 2},
            group_layer_cache_entry_offsets={0: [0, 2, 3]},
        )
        return database

    def test_real_construction_binds_the_worker(self):
        """The REAL thread class (not a __new__ stub): the constructor binds
        the passed worker, and the bound worker carries the probe end-to-end
        — a registered tail key reached through the thread's probe hook
        emits the save-side marker."""
        worker = TestHybridTailContentProbe()._probe_worker(
            TestHybridTailContentProbe()._fa_caches(populate_block=TestHybridTailContentProbe.SAVE_BLOCK)
        )
        key, _ = TestHybridTailContentProbe()._tail_key(worker)
        thread = KVCacheStoreLayerSendingThread(
            m_store=None,
            token_database=self._token_database(),
            block_size=16,
            tp_rank=0,
            tp_size=1,
            dcp_size=1,
            page_size_bytes=60,
            ready_event=threading.Event(),
            num_layers=2,
            layer_save_finished_events=[threading.Event(), threading.Event()],
            sync_save_events=[MagicMock(), MagicMock()],
            worker=worker,
        )
        self.assertIs(thread.worker, worker)
        req_meta = LayerRangeReqMeta(
            req_ids=["req-tail"],
            layer_id=3,
            block_ids=[TestHybridTailContentProbe.SAVE_BLOCK],
            keys=[key],
            all_buffers=[[0]],
            all_sizes=[[1]],
            all_offsets=[[0]],
        )
        with self.assertLogs(vllm_logger, level="INFO") as logs:
            thread._probe_tail_save_content(req_meta)
        marker_lines = [line for line in logs.output if "HOLEMISS_TAIL_CONTENT_SHA" in line]
        self.assertEqual(len(marker_lines), 1)
        self.assertIn("side=save", marker_lines[0])

    def test_construction_site_passes_worker_self(self):
        """The source pin (the inspect.getsource precedent): the block-key
        sending thread's construction inside KVPoolWorker's thread start
        passes worker=self — the exact omission the W2R2C run exposed."""
        source = inspect.getsource(KVPoolWorker._start_kv_transfer_threads)
        start = source.index("KVCacheStoreLayerSendingThread(")
        end = source.index("self.kv_send_thread.start()", start)
        construction_block = source[start:end]
        self.assertIn("worker=self", construction_block)


class TestTailMetaProbeRegistration(unittest.TestCase):
    """B43 pin: the runner-side metadata dump's worker half.

    The W2R2C2-A1 adjudication localized B43 to the compute-side FIA
    consumption (the save/load sha pair proves the bytes; the table
    construction chain is verified at source; the sequence is still the
    zero-attendance class). The sync load drain registers each tail-hit
    request's ALLOCATION — the request's own group-0 block table entry for
    the tail block, never a hardcoded page id — so the model runner can
    compare the kernel-consumed attention metadata against it at the
    forward's last pre-kernel point (the (a)-vs-(b) discrimination's
    expected-values half)."""

    def _worker(self):
        worker = TestHybridTailContentProbe()._probe_worker(TestHybridTailContentProbe()._fa_caches())
        worker._holemiss_tail_meta_records = {}
        return worker

    def test_tail_hit_request_registers_its_own_allocation(self):
        worker = self._worker()
        request = TestHybridTailLoadFetch._tail_load_request(req_id="req-meta")
        worker._holemiss_register_tail_meta_probe(request)
        self.assertEqual(
            worker._holemiss_tail_meta_records,
            {
                "req-meta": {
                    "tp": 0,
                    "boundary": BOUNDARY,
                    "tail_block": TAIL_BLOCK_INDEX,
                    "tail_page": 16,
                    "claim": BOUNDARY - 1,
                }
            },
        )

    def test_non_tail_request_registers_nothing(self):
        worker = self._worker()
        request = TestHybridTailLoadFetch._tail_load_request(tail_boundary_tokens=None)
        worker._holemiss_register_tail_meta_probe(request)
        self.assertEqual(worker._holemiss_tail_meta_records, {})

    def test_take_pops_the_records_once(self):
        worker = self._worker()
        worker._holemiss_register_tail_meta_probe(TestHybridTailLoadFetch._tail_load_request(req_id="req-meta"))
        records = worker.holemiss_take_tail_meta_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["req_id"], "req-meta")
        self.assertEqual(records[0]["tail_page"], 16)
        self.assertEqual(worker.holemiss_take_tail_meta_records(), [])
        self.assertEqual(worker._holemiss_tail_meta_records, {})

    def test_drain_registers_after_the_content_probe_post_sync(self):
        """The source pin: the registration lives in the sync drain's
        post-synchronize per-request loop, after the content probe — the
        record must reflect the post-drain allocation and precede the
        forward that consumes it."""
        source = inspect.getsource(KVPoolWorker._holemiss_sync_layerwise_drain)
        sync_idx = source.index("_holemiss_device_synchronize()")
        probe_idx = source.index("_holemiss_probe_tail_content_sha(request)")
        register_idx = source.index("_holemiss_register_tail_meta_probe(request)")
        self.assertLess(sync_idx, probe_idx)
        self.assertLess(probe_idx, register_idx)


class TestTailPostForwardProbe(unittest.TestCase):
    """B43 b2 closure: the third CONTENT_SHA side (side=post) — the
    post-forward checksum that closes the pre-forward-clobber sub-candidate.

    The W2R2C3-A1 run proved META_CORRECT (the runtime metadata exactly the
    verified derivation) with the byte-identical storage round trip and the
    zero-class sequence — (b) the kernel-side defect. The one formal gap:
    the window between the probes' reads and the kernel's read. The closure
    compares the tail page's STABLE region (the loaded attended window
    MINUS the re-added slots — reshape_and_cache legitimately rewrites the
    scheduled token's own slot, position 9471 = page-local slot 255) before
    and after the forward: unchanged ⇒ no other writer touched the bytes
    the kernel read ⇒ (b) airtight."""

    CLAIM = BOUNDARY - 1  # the scheduler's hit tokens (the live T2 claim)
    STABLE_TOKENS = CLAIM - FLOOR  # 255: the window minus the re-added slot
    BLOCK = TestHybridTailContentProbe.SAVE_BLOCK

    @staticmethod
    def _caches(populate_block=None):
        return TestHybridTailContentProbe()._fa_caches(populate_block=populate_block)

    def _worker(self, caches):
        worker = TestHybridTailContentProbe()._probe_worker(caches)
        worker._holemiss_tail_stable_pre = {}
        return worker

    def test_post_side_verdict_excludes_the_clobber_with_a_stable_page(self):
        worker = self._worker(self._caches(populate_block=self.BLOCK))
        worker.holemiss_tail_snapshot_stable_sha("req-post", self.BLOCK, BOUNDARY, self.CLAIM)
        with self.assertLogs(vllm_logger, level="INFO") as logs:
            worker.holemiss_tail_post_forward_sha("req-post", self.BLOCK, BOUNDARY, self.CLAIM)
        marker_lines = [line for line in logs.output if "HOLEMISS_TAIL_CONTENT_SHA" in line]
        self.assertEqual(len(marker_lines), 1)
        self.assertIn("side=post", marker_lines[0])
        self.assertIn("boundary=9472", marker_lines[0])
        self.assertIn("tail_tokens=256", marker_lines[0])
        self.assertIn("tail_nonzero=True", marker_lines[0])
        self.assertIn("verdict=CLOBBER_EXCLUDED", marker_lines[0])
        # The stable sha equals the snapshot (auditable from the line alone).
        stable = worker._holemiss_tail_stable_prefix_sha(self.BLOCK, self.STABLE_TOKENS)
        self.assertIn(f"stable_sha={stable}", marker_lines[0])
        self.assertIn(f"pre_stable_sha={stable}", marker_lines[0])
        # The snapshot is consumed (fire-once).
        self.assertEqual(worker._holemiss_tail_stable_pre, {})

    def test_post_side_detects_a_stable_region_clobber(self):
        caches = self._caches(populate_block=self.BLOCK)
        worker = self._worker(caches)
        block = self.BLOCK
        worker.holemiss_tail_snapshot_stable_sha("req-post", block, BOUNDARY, self.CLAIM)
        # An in-window clobber of the stable region: zero the page's first
        # kernel row (page-local slots [0, 128) — all inside the stable 255).
        for layer_caches in caches.values():
            for cache in layer_caches:
                cache[block * 12 : block * 12 + 1].zero_()
        with self.assertLogs(vllm_logger, level="INFO") as logs:
            worker.holemiss_tail_post_forward_sha("req-post", block, BOUNDARY, self.CLAIM)
        marker_lines = [line for line in logs.output if "HOLEMISS_TAIL_CONTENT_SHA" in line]
        self.assertEqual(len(marker_lines), 1)
        self.assertIn("verdict=CLOBBER_DETECTED", marker_lines[0])
        self.assertIn("tail_nonzero=True", marker_lines[0])
        # The attended window is still nonzero (only 128 of 256 tokens
        # zeroed) — the stable comparison catches what the window flag miss.
        stable_field = marker_lines[0].split("stable_sha=")[1].split()[0]
        pre_field = marker_lines[0].split("pre_stable_sha=")[1].split()[0]
        self.assertNotEqual(stable_field, pre_field)

    def test_post_side_without_a_snapshot_reports_the_honest_gap(self):
        worker = self._worker(self._caches(populate_block=self.BLOCK))
        with self.assertLogs(vllm_logger, level="INFO") as logs:
            worker.holemiss_tail_post_forward_sha("req-post", self.BLOCK, BOUNDARY, self.CLAIM)
        marker_lines = [line for line in logs.output if "HOLEMISS_TAIL_CONTENT_SHA" in line]
        self.assertEqual(len(marker_lines), 1)
        self.assertIn("verdict=NO_PRE_SNAPSHOT", marker_lines[0])
        self.assertIn("pre_stable_sha=None", marker_lines[0])

    def test_post_side_zero_pages_report_the_zero_class(self):
        worker = self._worker(self._caches())
        worker.holemiss_tail_snapshot_stable_sha("req-post", self.BLOCK, BOUNDARY, self.CLAIM)
        with self.assertLogs(vllm_logger, level="INFO") as logs:
            worker.holemiss_tail_post_forward_sha("req-post", self.BLOCK, BOUNDARY, self.CLAIM)
        marker_lines = [line for line in logs.output if "HOLEMISS_TAIL_CONTENT_SHA" in line]
        self.assertEqual(len(marker_lines), 1)
        self.assertIn("tail_nonzero=False", marker_lines[0])

    def test_the_stable_prefix_excludes_the_re_added_slot(self):
        """The one-slot correctness pin: the re-added token's write
        (page-local slot 255 = kernel row 1, offset 127 of the tail page)
        is EXCLUDED from the stable region — its mutation leaves the stable
        sha unchanged while the whole-page sha sees it."""
        caches = self._caches(populate_block=self.BLOCK)
        worker = self._worker(caches)
        block = self.BLOCK
        stable_before = worker._holemiss_tail_stable_prefix_sha(block, self.STABLE_TOKENS)
        page_before = worker._holemiss_tail_content_fingerprint(block, 256)[0]
        key_cache = caches["model.layers.3.self_attn"][0]
        key_cache[block * 12 + 1, 127, :, :] += 1.0
        stable_after = worker._holemiss_tail_stable_prefix_sha(block, self.STABLE_TOKENS)
        page_after = worker._holemiss_tail_content_fingerprint(block, 256)[0]
        self.assertEqual(stable_before, stable_after)
        self.assertNotEqual(page_before, page_after)


if __name__ == "__main__":
    unittest.main()
