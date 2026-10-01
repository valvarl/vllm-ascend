# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-key range batching (D-B23.7's companion): the sync drain's block-key
loads issue ONE ranged call per key carrying ALL its layer ranges.

The ported session read stages each key's object per CALL (no session-level
pin), so the per-layer pattern re-staged every object num_layers times (the
measured 6.27-13.01 s/rank band). These tests pin the batched shape: one
merged call with byte-identical destinations, all layer fence events firing
on completion, the exact failure-to-block mapping, and the EIO target's
single-delivery semantics.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    LOAD_LEASE_RETRY_MAX_ATTEMPTS,
    KVCacheStoreLayerRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    LayerBatchedLoadTask,
    LayerLoadTask,
    LayerRangeReqMeta,
)

KV_TIME_SLEEP = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer.time.sleep"

NUM_LAYERS = 4
NUM_KEYS = 3


def _layer_req_meta(layer_id: int, num_keys: int = NUM_KEYS, block_base: int = 100):
    """One layer's range metadata: per-key per-block segments at the layer's
    object offset (the shape LayerBatchBuilder.build_addrs produces)."""
    return LayerRangeReqMeta(
        req_ids=["req-B"],
        layer_id=layer_id,
        block_ids=[block_base + i for i in range(num_keys)],
        keys=[f"m@h{i}@0" for i in range(num_keys)],
        all_buffers=[[1000 * (i + 1) + layer_id] for i in range(num_keys)],
        all_sizes=[[256 + i] for i in range(num_keys)],
        all_offsets=[[layer_id * 1024 + i * 8] for i in range(num_keys)],
        load_keys=[],
    )


def _transfer_task(req_meta, group_id=0, layer_idx=0, final_group_layer=False):
    task = SimpleNamespace()
    task.group_id = group_id
    task.layer_idx_in_group = layer_idx
    task.final_group_layer = final_group_layer
    task.shared_block_data = SimpleNamespace(
        req_ids=req_meta.req_ids,
        is_last_chunks=[True],
    )
    # The exception fallback walks block_ranges (the conservative marking).
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


def _batched_task(req_metas_by_layer, block_ids=None):
    """A LayerBatchedLoadTask whose builders yield the given per-layer metas."""
    layer_tasks = []
    build_map = {}
    for layer_id, req_metas in req_metas_by_layer.items():
        tasks = []
        for group_id, req_meta in enumerate(req_metas):
            tasks.append(_transfer_task(req_meta, group_id=group_id, layer_idx=layer_id))
            build_map[(group_id, layer_id)] = req_meta
        layer_tasks.append(LayerLoadTask(wait_for_save_layer=None, transfer_tasks=tasks, layer_id=layer_id))
    return LayerBatchedLoadTask(layer_tasks=layer_tasks), build_map


_last_store_calls = 0


def _stub_recv_thread(build_map):
    global _last_store_calls
    _last_store_calls = 0
    thread = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
    thread.layer_load_finished_events = [__import__("threading").Event() for _ in range(NUM_LAYERS)]
    thread._load_abort_event = __import__("threading").Event()
    thread._invalid_block_ids = set()
    thread._invalid_block_ids_lock = __import__("threading").Lock()
    thread._fatal_error = None
    thread.request_queue = SimpleNamespace(task_done=lambda: None)
    thread.get_event = SimpleNamespace(set=lambda: None)
    thread.m_store = MagicMock()

    def _store_call(keys, buffers, sizes, offsets):
        global _last_store_calls
        _last_store_calls += 1
        return [sum(sizes_k) if sizes_k else 0 for sizes_k in sizes]

    thread.m_store.batch_copy_get.side_effect = _store_call
    thread.max_transfer_blocks = 0
    thread.max_transfer_bytes = 0
    thread.worker = None
    thread.h2d_stagger_us = 0
    thread.tp_rank = 0
    thread.tp_size = 1
    thread.group_builders = None

    class _Builder:
        def build_addrs(self, shared, layer_idx):
            return build_map[(0, layer_idx)]

    thread.layer_batch_builder = _Builder()

    def _set_finished(req_id):
        thread.finished_requests = getattr(thread, "finished_requests", []) + [req_id]

    thread.set_finished_request = _set_finished
    return thread


def _per_layer_metas():
    return {layer: [_layer_req_meta(layer)] for layer in range(NUM_LAYERS)}


class TestBatchedIssuance(unittest.TestCase):
    def test_one_call_per_key_all_ranges(self):
        """The merged call: ONE batch_copy_get whose per-key segment lists
        carry every layer's segments — the per-layer pattern's destinations
        byte-identical, coalesced."""
        task, build_map = _batched_task(_per_layer_metas())
        thread = _stub_recv_thread(build_map)
        thread._handle_batched_range_layer_task(task)
        self.assertEqual(thread.m_store.batch_copy_get.call_count, 1, "exactly one merged call")
        call = thread.m_store.batch_copy_get.call_args
        keys, buffers, sizes, offsets = call.args
        self.assertEqual(keys, [f"m@h{i}@0" for i in range(NUM_KEYS)])
        for key_index in range(NUM_KEYS):
            # Every layer contributed its segment, in layer order.
            self.assertEqual(
                buffers[key_index],
                [1000 * (key_index + 1) + layer for layer in range(NUM_LAYERS)],
            )
            self.assertEqual(sizes[key_index], [256 + key_index] * NUM_LAYERS)
            self.assertEqual(
                offsets[key_index],
                [layer * 1024 + key_index * 8 for layer in range(NUM_LAYERS)],
            )
        self.assertEqual(len(sizes), NUM_KEYS)

    def test_all_layer_fence_events_fire_on_completion(self):
        task, build_map = _batched_task(_per_layer_metas())
        thread = _stub_recv_thread(build_map)
        for event in thread.layer_load_finished_events:
            self.assertFalse(event.is_set())
        thread._handle_batched_range_layer_task(task)
        for layer in range(NUM_LAYERS):
            self.assertTrue(
                thread.layer_load_finished_events[layer].is_set(),
                f"layer {layer} fence must fire on the batched completion",
            )
        # The sync drain's wait-then-clear loop tolerates the all-at-once
        # setting (each wait returns immediately) — pinned by construction.
        for event in thread.layer_load_finished_events:
            event.clear()

    def test_empty_layer_list_is_a_noop(self):
        task, build_map = _batched_task({})
        thread = _stub_recv_thread(build_map)
        thread._handle_batched_range_layer_task(task)
        thread.m_store.batch_copy_get.assert_not_called()

    def test_final_group_layer_finishes_requests(self):
        metas = _per_layer_metas()
        task, build_map = _batched_task(metas)
        # Mark the LAST layer's task as the group's final layer.
        task.layer_tasks[-1].transfer_tasks[0].final_group_layer = True
        thread = _stub_recv_thread(build_map)
        thread._handle_batched_range_layer_task(task)
        self.assertEqual(thread.finished_requests, ["req-B"])


class TestBatchedFailureMapping(unittest.TestCase):
    def test_failed_keys_map_to_their_block_ids(self):
        """A negative result for a key marks exactly that key's block id on
        the invalid-block surface (the HR trigger path) — no per-layer
        scatter, no collateral marking."""
        metas = _per_layer_metas()
        task, build_map = _batched_task(metas)
        thread = _stub_recv_thread(build_map)
        thread.m_store.batch_copy_get.side_effect = lambda keys, buffers, sizes, offsets: [
            -1102 if key == "m@h1@0" else sum(sizes_k) for key, sizes_k in zip(keys, sizes)
        ]
        thread._handle_batched_range_layer_task(task)
        self.assertEqual(thread._invalid_block_ids, {101}, "only the failed key's block id")

    def test_t14_recorder_feed_untouched(self):
        """The failed-key recorder feeds from the GET-SESSION-START failures
        (pool_worker._open_layerwise_get_sessions) — upstream of any range
        call. The batched handler must not touch it; the session-opening
        path is unchanged."""
        import inspect

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import kv_transfer

        source = inspect.getsource(kv_transfer.KVCacheStoreLayerRecvingThread._handle_batched_range_layer_task)
        self.assertNotIn("_holemiss_record_failed_load_keys", source)
        self.assertNotIn("_holemiss_failed_load_keys", source)

    def test_exception_marks_all_task_blocks_and_aborts(self):
        metas = _per_layer_metas()
        task, build_map = _batched_task(metas)
        thread = _stub_recv_thread(build_map)
        thread.m_store.batch_copy_get.side_effect = RuntimeError("store read failed")
        with self.assertRaises(RuntimeError):
            thread._handle_batched_range_layer_task(task)
        self.assertTrue(thread._load_abort_event.is_set())
        # Every task's block ids are marked (the conservative fallback) and
        # every layer's fence still fires (the finally).
        self.assertEqual(thread._invalid_block_ids, {100, 101, 102})
        for event in thread.layer_load_finished_events:
            self.assertTrue(event.is_set())


class TestEioTargetSemantics(unittest.TestCase):
    def test_armed_key_isolated_once_in_the_merged_call(self):
        """The EIO target: exactly ONE isolated target read in the merged
        call (the per-layer shape deduplicated via the active-index skip —
        the count is unchanged by the batching); the healthy keys ride one
        call."""
        metas = _per_layer_metas()
        task, build_map = _batched_task(metas)
        thread = _stub_recv_thread(build_map)
        executed = {}
        worker = SimpleNamespace(
            _holemiss_armed_fault={"key": "m@h2@0", "layer_id": 2},
        )

        def target_position(keys, physical_layer_id):
            executed["position_layer"] = physical_layer_id
            return keys.index("m@h2@0") if "m@h2@0" in keys else None

        def execute_target(keys, buffers, sizes, offsets, target_pos):
            executed["target_pos"] = target_pos
            executed["keys"] = list(keys)
            return [0] * len(keys)

        worker.holemiss_range_target_position = target_position
        worker.execute_holemiss_range_target = execute_target
        thread.worker = worker
        thread._handle_batched_range_layer_task(task)
        # The position logic is called with the ARMED layer's id (the merged
        # call is layer-agnostic; the arm's layer gate must still fire).
        self.assertEqual(executed["position_layer"], 2)
        self.assertEqual(executed["target_pos"], 2)
        self.assertEqual(executed["keys"], [f"m@h{i}@0" for i in range(NUM_KEYS)])
        # The store saw NO direct call (the target executor replaced it).
        thread.m_store.batch_copy_get.assert_not_called()

    def _hybrid_worker(self, store_local_k, fa_layers):
        """The verifier probe's shape: a hybrid worker with the REAL gate and
        the REAL single-source FA map (qwen3.6 3+4k)."""
        worker = SimpleNamespace(
            _holemiss_armed_fault={"key": "m@h2@0", "layer_id": store_local_k},
            use_hybrid=True,
        )

        def physical_layer(store_layer_id):
            if not (0 <= store_layer_id < len(fa_layers)):
                raise RuntimeError("map cannot translate")
            return fa_layers[store_layer_id]

        def target_position(keys, physical_layer_id):
            armed = worker._holemiss_armed_fault
            if armed["key"] not in keys:
                return None
            if physical_layer(int(armed["layer_id"])) != physical_layer_id:
                return None
            return keys.index(armed["key"])

        worker._holemiss_hybrid_physical_layer = physical_layer
        worker.holemiss_range_target_position = target_position
        worker.execute_holemiss_range_target = lambda *args: [0] * len(args[0])
        return worker

    def _drive_hybrid_arm(self, store_local_k):
        fa_layers = [3 + 4 * k for k in range(16)]
        metas = _per_layer_metas()
        task, build_map = _batched_task(metas)
        thread = _stub_recv_thread(build_map)
        worker = self._hybrid_worker(store_local_k, fa_layers)
        inner_gate = worker.holemiss_range_target_position
        seen = {}

        def target_position(keys, physical_layer_id):
            seen["physical_layer_id"] = physical_layer_id
            return inner_gate(keys, physical_layer_id)

        worker.holemiss_range_target_position = target_position
        thread.worker = worker
        thread._handle_batched_range_layer_task(task)
        return seen

    def test_hybrid_arm_k0_isolates_via_the_physical_map(self):
        """The batching regression (F1): hybrid k=0 (physical 3) — the armed
        key MUST be isolated through the mapped physical id (the store-local
        pass made the gate return None and the armed key rode the plain
        call, losing the HR trigger on every hybrid fault arm)."""
        seen = self._drive_hybrid_arm(store_local_k=0)
        self.assertEqual(seen["physical_layer_id"], 3, "the gate must receive the PHYSICAL layer (3+4k)")
        self.assertEqual(_last_store_calls, 0, "the isolated executor replaces the plain store call")

    def test_hybrid_arm_k_gt_0_isolates_via_the_physical_map(self):
        """The pre-existing wholesale break (k>0 never live-tested): the same
        mapping isolates ANY FA store-local k through its physical layer."""
        for k in (1, 2, 15):
            with self.subTest(k=k):
                seen = self._drive_hybrid_arm(store_local_k=k)
                self.assertEqual(seen["physical_layer_id"], 3 + 4 * k)
                self.assertEqual(_last_store_calls, 0)

    def test_hybrid_gate_none_is_the_defect_signature(self):
        """The probe's discriminator: gate(keys, physical) fires; the
        store-local pass returns None — the exact F1 signature."""
        fa_layers = [3 + 4 * k for k in range(16)]
        worker = self._hybrid_worker(2, fa_layers)
        keys = [f"m@h{i}@0" for i in range(NUM_KEYS)]
        self.assertEqual(worker.holemiss_range_target_position(keys, 11), 2)
        self.assertIsNone(worker.holemiss_range_target_position(keys, 2))

    def test_text_arm_stays_byte_identical(self):
        """The text path: store-local == physical — the mapping is a no-op
        and the gate receives the armed id unchanged."""
        metas = _per_layer_metas()
        task, build_map = _batched_task(metas)
        thread = _stub_recv_thread(build_map)
        executed = {}
        worker = SimpleNamespace(
            _holemiss_armed_fault={"key": "m@h2@0", "layer_id": 2},
            use_hybrid=False,
        )

        def target_position(keys, physical_layer_id):
            executed["position_layer"] = physical_layer_id
            return keys.index("m@h2@0")

        worker.holemiss_range_target_position = target_position
        worker.execute_holemiss_range_target = lambda *args: [0] * len(args[0])
        thread.worker = worker
        thread._handle_batched_range_layer_task(task)
        self.assertEqual(executed["position_layer"], 2, "text: the armed id passes through unchanged")
        thread.m_store.batch_copy_get.assert_not_called()

    def test_unarmed_passes_through_to_the_store(self):
        metas = _per_layer_metas()
        task, build_map = _batched_task(metas)
        thread = _stub_recv_thread(build_map)
        thread.worker = SimpleNamespace(_holemiss_armed_fault=None)
        thread._handle_batched_range_layer_task(task)
        thread.m_store.batch_copy_get.assert_called_once()


class TestMemoryTierUnchanged(unittest.TestCase):
    def test_merged_call_carries_identical_destinations(self):
        """The memory tier executes the same TE scatters per segment; the
        merged call's (buffer, size, offset) triples are BYTE-IDENTICAL to
        the per-layer pattern's — coalesced, not transformed."""
        per_layer = {layer: _layer_req_meta(layer) for layer in range(NUM_LAYERS)}
        expected = []
        for layer in range(NUM_LAYERS):
            meta = per_layer[layer]
            for i in range(NUM_KEYS):
                expected.append((meta.keys[i], meta.all_buffers[i][0], meta.all_sizes[i][0], meta.all_offsets[i][0]))
        task, build_map = _batched_task({layer: [per_layer[layer]] for layer in range(NUM_LAYERS)})
        thread = _stub_recv_thread(build_map)
        thread._handle_batched_range_layer_task(task)
        call = thread.m_store.batch_copy_get.call_args
        keys, buffers, sizes, offsets = call.args
        actual = []
        for i, key in enumerate(keys):
            for buf, size, off in zip(buffers[i], sizes[i], offsets[i], strict=True):
                actual.append((key, buf, size, off))
        self.assertEqual(sorted(actual), sorted(expected))


if __name__ == "__main__":
    unittest.main()


class TestLoadLeaseRetry(unittest.TestCase):
    """B27: the bounded OBJECT_HAS_LEASE (-706) retry in the batched handler."""

    def _task(self):
        metas = _per_layer_metas()
        return _batched_task(metas)

    def _flaky_thread(self, build_map, lease_failures: dict):
        """A stub thread whose store fails specific keys with -706 for the
        first N calls (the transient restore-lease straddle), then succeeds."""
        thread = _stub_recv_thread(build_map)
        calls: list[tuple] = []

        def store_call(keys, buffers, sizes, offsets):
            calls.append(tuple(keys))
            results = []
            for key, sizes_k in zip(keys, sizes):
                remaining = lease_failures.get(key, 0)
                if remaining > 0:
                    lease_failures[key] = remaining - 1
                    results.append(-706)
                else:
                    results.append(sum(sizes_k) if sizes_k else 0)
            return results

        thread.m_store.batch_copy_get.side_effect = store_call
        thread._store_calls = calls
        return thread

    def test_transient_lease_straddle_retries_green(self):
        """UT-1: a -706 on the first call for one key -> the retry re-issues
        the failed rows only -> the load completes, no invalidation, the
        fence fires."""
        task, build_map = self._task()
        thread = self._flaky_thread(build_map, {"m@h1@0": 1})
        with patch(KV_TIME_SLEEP):
            thread._handle_batched_range_layer_task(task)
        self.assertEqual(len(thread._store_calls), 2, "the first call + one retry")
        self.assertEqual(thread._store_calls[1], ("m@h1@0",), "the retry carries the failed rows only")
        self.assertEqual(thread._invalid_block_ids, set(), "the transient class never invalidates")
        for event in thread.layer_load_finished_events:
            self.assertTrue(event.is_set())

    def test_permanent_lease_straddle_exhausts_loud(self):
        """UT-2: a permanent -706 -> the bound exhausted -> the EXISTING loud
        failure mapping (the invalid block), 1 + MAX calls total."""
        task, build_map = self._task()
        thread = self._flaky_thread(build_map, {"m@h1@0": 99})
        with patch(KV_TIME_SLEEP):
            thread._handle_batched_range_layer_task(task)
        self.assertEqual(len(thread._store_calls), 1 + LOAD_LEASE_RETRY_MAX_ATTEMPTS)
        self.assertEqual(thread._invalid_block_ids, {101}, "the exhausted key maps to its block")
        for event in thread.layer_load_finished_events:
            self.assertTrue(event.is_set(), "the fences fire even on exhaustion")

    def test_eio_negative_is_never_retried(self):
        """UT-3 (the D5 pin): the -1102 EIO-injected fault class is NEVER
        retried — one call, the failure mapping immediate."""
        task, build_map = self._task()
        thread = _stub_recv_thread(build_map)
        thread.m_store.batch_copy_get.side_effect = lambda keys, buffers, sizes, offsets: [
            -1102 if key == "m@h1@0" else sum(sizes_k) for key, sizes_k in zip(keys, sizes)
        ]
        with patch(KV_TIME_SLEEP) as slept:
            thread._handle_batched_range_layer_task(task)
        self.assertEqual(thread.m_store.batch_copy_get.call_count, 1, "the EIO class is fail-fast")
        slept.assert_not_called()
        self.assertEqual(thread._invalid_block_ids, {101})

    def test_session_lease_negative_is_never_retried(self):
        """UT-4: the -707 session-lease class (B26's mechanism) stays
        fail-fast — the retry is the -706 restore-lease class ONLY."""
        task, build_map = self._task()
        thread = _stub_recv_thread(build_map)
        thread.m_store.batch_copy_get.side_effect = lambda keys, buffers, sizes, offsets: [
            -707 if key == "m@h2@0" else sum(sizes_k) for key, sizes_k in zip(keys, sizes)
        ]
        with patch(KV_TIME_SLEEP) as slept:
            thread._handle_batched_range_layer_task(task)
        self.assertEqual(thread.m_store.batch_copy_get.call_count, 1)
        slept.assert_not_called()
        self.assertEqual(thread._invalid_block_ids, {102})

    def test_armed_target_excluded_from_the_retry(self):
        """The armed key's result stands as-is (its own delivery contract);
        a -706 on a HEALTHY key retries while the armed key's -1102 does
        not ride the plain re-issue."""
        task, build_map = self._task()
        thread = _stub_recv_thread(build_map)
        worker = SimpleNamespace(
            _holemiss_armed_fault={"key": "m@h2@0", "layer_id": 1},
            use_hybrid=False,
        )
        worker.holemiss_range_target_position = lambda keys, layer: keys.index("m@h2@0") if "m@h2@0" in keys else None

        def execute_target(keys, buffers, sizes, offsets, target_pos):
            # The isolated executor: the armed key gets the injected fault;
            # one healthy key straddled the restore lease.
            return [-706 if i == 0 else (-1102 if i == target_pos else sum(sizes[i])) for i in range(len(keys))]

        worker.execute_holemiss_range_target = execute_target
        thread.worker = worker
        store_calls: list[tuple] = []

        def store_call(keys, buffers, sizes, offsets):
            store_calls.append(tuple(keys))
            return [sum(sizes_k) for sizes_k in sizes]

        thread.m_store.batch_copy_get.side_effect = store_call
        with patch(KV_TIME_SLEEP):
            thread._handle_batched_range_layer_task(task)
        # The healthy -706 retried via exactly one plain call carrying ONLY
        # that key — the armed key never appears in a plain call.
        self.assertEqual(len(store_calls), 1)
        self.assertNotIn("m@h2@0", store_calls[0])
        self.assertEqual(store_calls[0], ("m@h0@0",))
        # The armed key's injected fault stands in the final mapping.
        self.assertEqual(thread._invalid_block_ids, {102})


class TestLoadLeaseRetryLiteralPins(unittest.TestCase):
    """F-1 follow-up: the B27 retry bounds are contract values, pinned
    LITERALLY (D-B20.10) so a bound-widening edit trips a UT instead of
    shipping silently (the verifier's M3b probe proved 4..99 currently
    pass UT-undetected)."""

    def test_max_attempts_is_the_contract_value(self):
        self.assertEqual(LOAD_LEASE_RETRY_MAX_ATTEMPTS, 3)

    def test_backoff_base_is_the_contract_value(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import kv_transfer

        self.assertEqual(kv_transfer.LOAD_LEASE_RETRY_BACKOFF_MS, 200)

    def test_retry_error_is_object_has_lease_only(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import kv_transfer

        self.assertEqual(kv_transfer.LOAD_LEASE_RETRY_ERROR, -706)

    def test_total_backoff_is_bounded_well_inside_watchdogs(self):
        """The escalating schedule 200/400/800ms sums to 1.4s — the derived
        bound (b27_retry_semantics.md); a widening breaks this pin."""
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import kv_transfer

        total_ms = sum(
            kv_transfer.LOAD_LEASE_RETRY_BACKOFF_MS * 2**attempt
            for attempt in range(kv_transfer.LOAD_LEASE_RETRY_MAX_ATTEMPTS)
        )
        self.assertEqual(total_ms, 1400)


class TestBatchedLoadLeaseRelease(unittest.TestCase):
    """B35: the completed batched load releases the session leases it took.

    The write-back's non-force remove requires the load's leases gone (the
    master's Remove checks IsLeaseExpired); the batched handler is the
    executed load path on every bench mode, so its completion point must
    mirror the legacy per-layer path's final-layer batch_remove_lease. The
    release must NOT fire on the failure paths where a lease may still be
    needed (the abort early-return and the exception path feed the retry
    machinery)."""

    def test_completion_releases_the_merged_keys_leases(self):
        task, build_map = _batched_task(_per_layer_metas())
        thread = _stub_recv_thread(build_map)
        thread._handle_batched_range_layer_task(task)
        thread.m_store.batch_remove_lease.assert_called_once_with([f"m@h{i}@0" for i in range(NUM_KEYS)])

    def test_release_covers_failed_keys(self):
        """The keys whose load FAILED are released too — the tail-writeback's
        remove targets exactly those keys (the EIO'd block), so a
        success-only release would leave the write-back blocked."""
        task, build_map = _batched_task(_per_layer_metas())
        thread = _stub_recv_thread(build_map)
        thread.m_store.batch_copy_get.side_effect = lambda keys, buffers, sizes, offsets: [
            -1102 if key == "m@h1@0" else sum(sizes_k) for key, sizes_k in zip(keys, sizes)
        ]
        thread._handle_batched_range_layer_task(task)
        self.assertEqual(thread._invalid_block_ids, {101})
        thread.m_store.batch_remove_lease.assert_called_once_with([f"m@h{i}@0" for i in range(NUM_KEYS)])

    def test_exception_path_retains_the_leases(self):
        """A load that raises never completes; its sessions must stay live
        for the retry machinery (the abort-and-reissue path)."""
        task, build_map = _batched_task(_per_layer_metas())
        thread = _stub_recv_thread(build_map)
        thread.m_store.batch_copy_get.side_effect = RuntimeError("store read failed")
        with self.assertRaises(RuntimeError):
            thread._handle_batched_range_layer_task(task)
        thread.m_store.batch_remove_lease.assert_not_called()

    def test_abort_path_retains_the_leases(self):
        """The pre-load abort early-return issued no load; nothing to
        release, and a release would drop sessions the retry still owns."""
        task, build_map = _batched_task(_per_layer_metas())
        thread = _stub_recv_thread(build_map)
        thread._load_abort_event.set()
        thread._handle_batched_range_layer_task(task)
        thread.m_store.batch_copy_get.assert_not_called()
        thread.m_store.batch_remove_lease.assert_not_called()

    def test_release_rides_the_backend_lease_api_not_the_session_tracker(self):
        """The handler calls the BACKEND lease-release API directly (the
        legacy :2541 call shape), independent of the worker-side session
        tracker's terminal release."""
        import inspect

        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import (
            kv_transfer as kv_transfer_module,
        )

        source = inspect.getsource(kv_transfer_module.KVCacheStoreLayerRecvingThread._handle_batched_range_layer_task)
        self.assertIn("self.m_store.batch_remove_lease(unique_load_keys)", source)
        self.assertNotIn("_end_layerwise_load_keys", source)
