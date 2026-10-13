# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""hybrid-gdn-tail-state T7 fix: the unaligned prompt-end last-chunk ReqMeta keep-alive.

The T7 live battery proved the drop: a pure-save request's unaligned prompt-end
last chunk builds no new full block and carries no load, so
``ReqMeta.from_request_tracker`` returns None (skip_save, load_spec None) —
the step's connector metadata then has an empty requests window and the
worker's final-layer state-tail snapshot capture never sees the qualifying
request (the live signature: ``HOLEMISS_STATE_TAIL_PUT_SKIPPED
reason=no_frozen_snapshot`` on both ranks at the drain step).

The fix keeps that ReqMeta alive (``can_save=False``, a verified save-side
no-op) exactly when the capture's own guard would qualify it. These tests
drive the REAL scheduler-side construction (``_build_req_meta`` ->
``from_request_tracker`` with a REAL ``RequestTracker`` at the live geometry)
— the class the T2 UTs missed by hand-building qualifying ReqMetas
(the fake-vs-live law N18/N19).
"""

import unittest
from types import SimpleNamespace

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_scheduler as pool_scheduler_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import LoadSpec, RequestTracker

# The hybrid proof geometry (design §0): align block 1536, pmu 256, the
# populate prompt 9472 = 6*1536 + 256 (unaligned), the floor 9216.
HYBRID_BLOCK = 1536
HYBRID_PMU = 256
FLOOR = 6 * HYBRID_BLOCK
BOUNDARY = FLOOR + HYBRID_PMU
ALIGNED_PROMPT = FLOOR
NUM_GROUPS = 4
STATE_GROUPS = (1, 2, 3)
BLOCKS_PER_GROUP = [[100 + group * 10 + index for index in range(7)] for group in range(NUM_GROUPS)]
BLOCK_HASHES = [f"{index:064x}" for index in range(BOUNDARY // HYBRID_PMU)]


def _scheduler(mamba_groups=STATE_GROUPS):
    sched = pool_scheduler_module.KVPoolScheduler.__new__(pool_scheduler_module.KVPoolScheduler)
    sched.kv_role = "kv_both"
    sched.use_hybrid = False
    sched.use_layerwise = True
    sched.use_block_key_layerwise = True
    sched.tp_mismatch = False
    sched.layerwise_offload = False
    sched.save_decode_cache = False
    sched.enable_kv_events = False
    sched._discard_partial_chunks = True
    sched.cache_transfer_granularity = HYBRID_BLOCK
    sched.hash_block_size = HYBRID_PMU
    sched.original_block_size = [HYBRID_BLOCK] * NUM_GROUPS
    sched.kv_cache_group_families = None
    sched.grouped_block_size = [HYBRID_BLOCK] * NUM_GROUPS
    sched.num_speculative_blocks_by_group = dict.fromkeys(mamba_groups, 0)
    sched.load_specs = {}
    sched._request_trackers = {}
    sched._unfinished_requests = {}
    sched._preempted_req_ids = set()
    sched._loading_req_ids = set()
    sched._finished_partial_tail_metas = {}
    return sched


def _tracker(token_len, num_prompt, num_saved):
    return RequestTracker(
        req_id="req-t7",
        token_len=token_len,
        allocated_block_ids_by_group=[list(blocks) for blocks in BLOCKS_PER_GROUP],
        num_saved_tokens=num_saved,
        num_prompt_tokens=num_prompt,
        num_speculative_blocks_by_group=dict.fromkeys(STATE_GROUPS, 0),
        block_sizes=[HYBRID_BLOCK] * NUM_GROUPS,
    )


def _build(sched, tracker, load_spec=None):
    return sched._build_req_meta(
        tracker,
        BLOCK_HASHES,
        load_spec,
        list(range(BOUNDARY)),
        False,
    )


class TestStateTailKeepalive(unittest.TestCase):
    def test_pure_save_unaligned_last_chunk_survives_the_drop(self):
        """THE T7 closure: the 9472 populate's last chunk (token_len == the
        prompt end, 9216 already saved, no load) keeps its ReqMeta — the
        capture window the live battery proved empty."""
        meta = _build(_scheduler(), _tracker(BOUNDARY, BOUNDARY, FLOOR))
        self.assertIsNotNone(meta)
        self.assertFalse(meta.can_save)
        self.assertTrue(meta.state_tail_capture_keepalive)
        self.assertEqual(meta.target_token_len, BOUNDARY)
        self.assertEqual(meta.num_prompt_tokens, BOUNDARY)
        self.assertIsNone(meta.load_spec)

    def test_text_layout_stays_dropped(self):
        """The scoping proof: no GDN state groups -> the pre-fix drop holds
        (the text path byte-identical)."""
        meta = _build(_scheduler(mamba_groups=()), _tracker(BOUNDARY, BOUNDARY, FLOOR))
        self.assertIsNone(meta)

    def test_aligned_prompt_end_needs_no_keepalive(self):
        """The aligned arms (4608/9216/18432): the last chunk SAVES its full
        block (the boundary-state channel); the keep-alive flag stays False."""
        meta = _build(_scheduler(), _tracker(ALIGNED_PROMPT, ALIGNED_PROMPT, ALIGNED_PROMPT - HYBRID_BLOCK))
        self.assertIsNotNone(meta)
        self.assertTrue(meta.can_save)
        self.assertFalse(meta.state_tail_capture_keepalive)

    def test_mid_prompt_unaligned_chunk_stays_dropped(self):
        meta = _build(_scheduler(), _tracker(FLOOR + 464, BOUNDARY, FLOOR))
        self.assertIsNone(meta)

    def test_decode_step_stays_dropped(self):
        meta = _build(_scheduler(), _tracker(BOUNDARY + 1, BOUNDARY, FLOOR))
        self.assertIsNone(meta)

    def test_load_bearing_step_does_not_carry_the_flag(self):
        """The load-bearing survival (the reference's @9728 shape): the
        ReqMeta survives through its load_spec, not the keep-alive."""
        meta = _build(
            _scheduler(),
            _tracker(BOUNDARY, BOUNDARY, FLOOR),
            load_spec=LoadSpec(
                vllm_cached_tokens=FLOOR,
                kvpool_cached_tokens=FLOOR,
                can_load=True,
            ),
        )
        self.assertIsNotNone(meta)
        self.assertFalse(meta.state_tail_capture_keepalive)
        self.assertIsNotNone(meta.load_spec)

    def test_the_keepalive_shape_qualifies_the_worker_capture(self):
        """The contract pin: the constructed ReqMeta satisfies the worker
        capture guard exactly (target == num_prompt, unaligned vs the group
        block) — the seam the T2 UTs hand-built around."""
        meta = _build(_scheduler(), _tracker(BOUNDARY, BOUNDARY, FLOOR))
        self.assertEqual(meta.target_token_len, meta.num_prompt_tokens)
        self.assertNotEqual(meta.num_prompt_tokens % HYBRID_BLOCK, 0)


class TestKeepaliveForcing(unittest.TestCase):
    """The B19 composition: the keep-alive step must force the pre-forward
    start_load_kv (has_sync_kv_loads), or the worker's current_layer stays
    stale and the capture never fires even with the ReqMeta present."""

    def _drive(self, mamba_groups=STATE_GROUPS):
        sched = _scheduler(mamba_groups)
        request = SimpleNamespace(
            req_id="req-t7",
            prompt_token_ids=list(range(BOUNDARY)),
            block_ids=[list(blocks) for blocks in BLOCKS_PER_GROUP],
            block_hashes=BLOCK_HASHES,
            num_computed_tokens=FLOOR,
            num_prompt_tokens=BOUNDARY,
            all_token_ids=list(range(BOUNDARY)),
        )
        sched._unfinished_requests[request.req_id] = (request, BLOCKS_PER_GROUP)
        sched._request_trackers[request.req_id] = _tracker(FLOOR, BOUNDARY, FLOOR)
        scheduler_output = SimpleNamespace(
            finished_req_ids=[],
            preempted_req_ids=[],
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[request.req_id], new_block_ids=[[[16], [26], [36], [46]]]),
            num_scheduled_tokens={request.req_id: HYBRID_PMU},
            has_sync_kv_loads=False,
        )
        meta = sched.build_connector_meta(scheduler_output)
        return meta, scheduler_output

    def test_keepalive_step_forces_sync_loads(self):
        meta, scheduler_output = self._drive()
        self.assertTrue(scheduler_output.has_sync_kv_loads)
        self.assertEqual(len(meta.requests), 1)
        self.assertTrue(meta.requests[0].state_tail_capture_keepalive)
        self.assertFalse(meta.requests[0].can_save)
        self.assertEqual(meta.requests[0].target_token_len, BOUNDARY)

    def test_text_step_stays_unforced(self):
        meta, scheduler_output = self._drive(mamba_groups=())
        self.assertFalse(scheduler_output.has_sync_kv_loads)
        self.assertEqual(len(meta.requests), 0)


if __name__ == "__main__":
    unittest.main()
