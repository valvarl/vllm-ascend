# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""hybrid-gdn-tail-state T4: the interior-boundary tail probe + the honest -1 refusal.

``get_num_new_matched_tokens`` walks the tail candidates NEWEST-first (the
pmu-aligned interior boundaries plus the prompt's exact end) instead of
probing only the prompt end: an extension-shaped consumer (prompt 9728 over
a 9472 artifact) claims the interior boundary, so the GDN composition is
exact (src=dst=slot, no copy). A hit at b == num_tokens on a GDN layout is
DECLINED — the boundary-state artifact cannot serve the -1's re-added token
(the structural off-by-one) — and the walk continues to older candidates.
Text/single-group layouts keep the T18 exact-end extension byte-identically
(the unaligned prompt end stays the newest candidate; the refusal's layout
gate is empty for text).

The fixtures stub only the store I/O (``_make_layerwise_hit_check_keys`` /
``_query_layerwise_block_hits``) and the full-block hit; the REAL probe
derivation (the raw-chain hash index, the aligned reject) and the REAL walk
run in every test.
"""

import unittest
from types import SimpleNamespace

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_scheduler as pool_scheduler_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    block_hash_to_str,
    partial_tail_key_suffix,
)

# The hybrid proof geometry (design §0): align block 1536, pmu 256, floor
# 9216, the populate boundary 9472, the T2' extension prompt 9728.
HYBRID_BLOCK = 1536
HYBRID_PMU = 256
HYBRID_FLOOR = 6 * HYBRID_BLOCK
TAIL_BOUNDARY = HYBRID_FLOOR + HYBRID_PMU
EXTENSION_PROMPT = TAIL_BOUNDARY + HYBRID_PMU
# The text geometry (the T18 E2E twin): block == pmu == 16, prompt 100.
TEXT_BLOCK = 16
TEXT_PROMPT = 6 * TEXT_BLOCK + 4
TEXT_FLOOR = 6 * TEXT_BLOCK
NUM_HASHES = 96


def _block_hashes(count: int = NUM_HASHES) -> list[str]:
    return [f"{index:064x}" for index in range(count)]


def _request(prompt_len: int, request_id: str = "req-T4") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        num_tokens=prompt_len,
        block_hashes=_block_hashes(),
    )


class TailProbeHarness:
    """A KVPoolScheduler stub with the REAL probe + a fake persisted-tail store.

    The full-block hit defaults to COMPLETE at the granularity floor (the
    walk's entry condition); every tail-key probe is recorded for the order
    and census pins. ``persist_tail`` derives the artifact suffix exactly the
    way the worker's save path does (the raw-chain hash at boundary//pmu-1).
    """

    def __init__(self, *, block: int, pmu: int, mamba_groups: tuple[int, ...]):
        sched = pool_scheduler_module.KVPoolScheduler.__new__(pool_scheduler_module.KVPoolScheduler)
        sched.kv_role = "kv_both"
        sched.consumer_is_to_load = True
        sched.use_block_key_layerwise = True
        sched.use_layerwise = True
        sched.use_layerwise_transfer = False
        sched.use_eagle = False
        sched.load_async = False
        sched.load_specs = {}
        num_groups = len(mamba_groups) + 1
        sched.grouped_block_size = [block] * num_groups
        sched.original_block_size = [block] * num_groups
        sched.cacheable_group_ids = list(range(num_groups))
        sched.cache_transfer_granularity = block
        sched.lcm_block_size = block
        sched.hash_block_size = pmu
        sched.num_speculative_blocks_by_group = dict.fromkeys(mamba_groups, 0)
        self.pmu = pmu
        self.persisted_suffixes: set[str] = set()
        self.probed_suffixes: list[str] = []
        self.full_block_hit: int | None = None
        sched._get_block_key_layerwise_hit_tokens = self._full_block_hit
        sched._make_layerwise_hit_check_keys = self._make_keys
        sched._query_layerwise_block_hits = self._query_hits
        self.scheduler = sched

    def _full_block_hit(self, request, token_len, num_computed_tokens):
        return token_len if self.full_block_hit is None else self.full_block_hit

    def _make_keys(self, group_id: int, block_hash_hex: str) -> list[str]:
        return [f"g{group_id}|{block_hash_hex}"]

    def _query_hits(self, keys_by_block: list[list[str]]) -> list[bool]:
        suffixes = [key.split("|", 1)[1] for key in keys_by_block[0]]
        self.probed_suffixes.extend(suffixes)
        return [suffix in self.persisted_suffixes for suffix in suffixes]

    def persist_tail(self, block_hashes: list[str], boundary: int) -> None:
        self.persisted_suffixes.add(
            partial_tail_key_suffix(block_hash_to_str(block_hashes[boundary // self.pmu - 1]), boundary)
        )

    def suffix(self, block_hashes: list[str], boundary: int) -> str:
        return partial_tail_key_suffix(block_hash_to_str(block_hashes[boundary // self.pmu - 1]), boundary)

    def get_num_new_matched_tokens(self, request) -> tuple[int, bool]:
        return self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens=0)

    def load_spec(self, request_id: str = "req-T4"):
        return self.scheduler.load_specs[request_id]


def hybrid_harness() -> TailProbeHarness:
    return TailProbeHarness(block=HYBRID_BLOCK, pmu=HYBRID_PMU, mamba_groups=(1, 2, 3))


def text_harness() -> TailProbeHarness:
    return TailProbeHarness(block=TEXT_BLOCK, pmu=TEXT_BLOCK, mamba_groups=())


class TestInteriorBoundaryWalk(unittest.TestCase):
    def test_interior_hit_at_the_populate_boundary(self):
        """T2': the 9728 consumer claims the interior 9472 artifact (the
        longest claim), not just the floor — the extension shape's exact
        GDN composition (b < num_tokens: no -1 fires)."""
        harness = hybrid_harness()
        request = _request(EXTENSION_PROMPT)
        harness.persist_tail(request.block_hashes, TAIL_BOUNDARY)
        with self.assertLogs(logger, level="INFO") as captured:
            need_to_allocate, _ = harness.get_num_new_matched_tokens(request)
        assert need_to_allocate == TAIL_BOUNDARY
        spec = harness.load_spec()
        assert spec.tail_boundary_tokens == TAIL_BOUNDARY
        assert spec.kvpool_cached_tokens == TAIL_BOUNDARY
        assert spec.kvpool_store_skip_tokens == TAIL_BOUNDARY
        assert spec.vllm_cached_tokens == 0
        assert spec.can_load is True
        hit_lines = [line for line in captured.output if "KV pool partial tail hit" in line]
        assert len(hit_lines) == 1
        assert f"exact_end={TAIL_BOUNDARY}" in hit_lines[0]
        assert f"(floor was {HYBRID_FLOOR})" in hit_lines[0]
        assert not any("KV pool partial tail declined" in line for line in captured.output)

    def test_miss_at_all_candidates_falls_back_to_the_floor(self):
        """No persisted tail anywhere: the walk probes every candidate and
        the claim stays at the floor (the T18 negative control)."""
        harness = hybrid_harness()
        request = _request(EXTENSION_PROMPT)
        need_to_allocate, _ = harness.get_num_new_matched_tokens(request)
        assert need_to_allocate == HYBRID_FLOOR
        spec = harness.load_spec()
        assert spec.tail_boundary_tokens is None
        assert spec.kvpool_cached_tokens == HYBRID_FLOOR
        assert len(harness.probed_suffixes) == 2

    def test_walk_is_newest_first_and_first_hit_wins(self):
        """The ORDER pin: artifacts at BOTH 9472 and 9728 with the prompt
        9984 — the newest hit (9728) wins and the walk STOPS (9472 is never
        probed)."""
        harness = hybrid_harness()
        prompt = TAIL_BOUNDARY + 2 * HYBRID_PMU
        request = _request(prompt)
        harness.persist_tail(request.block_hashes, TAIL_BOUNDARY)
        harness.persist_tail(request.block_hashes, TAIL_BOUNDARY + HYBRID_PMU)
        need_to_allocate, _ = harness.get_num_new_matched_tokens(request)
        assert need_to_allocate == TAIL_BOUNDARY + HYBRID_PMU
        assert harness.load_spec().tail_boundary_tokens == TAIL_BOUNDARY + HYBRID_PMU
        assert harness.probed_suffixes == [
            harness.suffix(request.block_hashes, prompt),
            harness.suffix(request.block_hashes, TAIL_BOUNDARY + HYBRID_PMU),
        ]

    def test_incomplete_full_block_hit_skips_the_walk(self):
        """The FA-prefix anchor law (#57216): the probe NEVER runs as a
        standalone all-groups anchor — an incomplete full-block hit at the
        floor means no tail probe at all."""
        harness = hybrid_harness()
        harness.full_block_hit = HYBRID_FLOOR - HYBRID_BLOCK
        request = _request(EXTENSION_PROMPT)
        harness.persist_tail(request.block_hashes, TAIL_BOUNDARY)
        need_to_allocate, _ = harness.get_num_new_matched_tokens(request)
        assert need_to_allocate == HYBRID_FLOOR - HYBRID_BLOCK
        assert harness.probed_suffixes == []
        assert harness.load_spec().tail_boundary_tokens is None


class TestHonestMinusOneRefusal(unittest.TestCase):
    def test_degenerate_prompt_is_declined_on_hybrid(self):
        """T3': the SAME 9472 prompt as the populate — the probe HITS
        _tail_9472 but b == num_tokens on a GDN layout: the extension is
        DECLINED (no hit line), the claim stays at the floor, the decline is
        logged with the boundary and the reason."""
        harness = hybrid_harness()
        request = _request(TAIL_BOUNDARY)
        harness.persist_tail(request.block_hashes, TAIL_BOUNDARY)
        with self.assertLogs(logger, level="INFO") as captured:
            need_to_allocate, _ = harness.get_num_new_matched_tokens(request)
        assert need_to_allocate == HYBRID_FLOOR
        spec = harness.load_spec()
        assert spec.tail_boundary_tokens is None
        assert spec.kvpool_cached_tokens == HYBRID_FLOOR
        declined_lines = [line for line in captured.output if "KV pool partial tail declined" in line]
        assert len(declined_lines) == 1
        assert f"boundary={TAIL_BOUNDARY}" in declined_lines[0]
        assert "reason=hybrid_full_end" in declined_lines[0]
        assert not any("KV pool partial tail hit" in line for line in captured.output)
        assert harness.probed_suffixes == [harness.suffix(request.block_hashes, TAIL_BOUNDARY)]

    def test_decline_continues_to_older_candidates(self):
        """D6: a declined full-end candidate does not forbid a valid
        interior one — the walk keeps going and accepts the next hit."""
        harness = hybrid_harness()
        prompt = TAIL_BOUNDARY + 2 * HYBRID_PMU
        request = _request(prompt)
        harness.persist_tail(request.block_hashes, prompt)
        harness.persist_tail(request.block_hashes, TAIL_BOUNDARY)
        with self.assertLogs(logger, level="INFO") as captured:
            need_to_allocate, _ = harness.get_num_new_matched_tokens(request)
        assert need_to_allocate == TAIL_BOUNDARY
        assert harness.load_spec().tail_boundary_tokens == TAIL_BOUNDARY
        assert len([line for line in captured.output if "KV pool partial tail declined" in line]) == 1
        hit_lines = [line for line in captured.output if "KV pool partial tail hit" in line]
        assert len(hit_lines) == 1 and f"exact_end={TAIL_BOUNDARY}" in hit_lines[0]


class TestTextRegressionPin(unittest.TestCase):
    def test_text_degenerate_extension_keeps_working(self):
        """The regression pin: on text (no GDN groups) the degenerate
        exact-end shape extends to the prompt end and the -1 convention
        keeps working — byte-identical to the pre-T4 behavior (the unaligned
        prompt end stays the newest walk candidate; the layout gate never
        declines)."""
        harness = text_harness()
        request = _request(TEXT_PROMPT)
        harness.persist_tail(request.block_hashes, TEXT_PROMPT)
        with self.assertLogs(logger, level="INFO") as captured:
            need_to_allocate, _ = harness.get_num_new_matched_tokens(request)
        assert need_to_allocate == TEXT_PROMPT - 1
        spec = harness.load_spec()
        assert spec.tail_boundary_tokens == TEXT_PROMPT
        assert spec.kvpool_cached_tokens == TEXT_PROMPT - 1
        assert spec.kvpool_store_skip_tokens == TEXT_PROMPT
        assert len([line for line in captured.output if "KV pool partial tail hit" in line]) == 1
        assert not any("KV pool partial tail declined" in line for line in captured.output)
        assert harness.probed_suffixes == [harness.suffix(request.block_hashes, TEXT_PROMPT)]


class TestProbeCountCensus(unittest.TestCase):
    """The cost bound: the pmu quantization caps the walk at
    (prompt - floor) // pmu + 1 probes — a pathological prompt must not walk
    O(prompt) candidates."""

    def test_bench_shapes(self):
        harness = hybrid_harness()
        harness.get_num_new_matched_tokens(_request(EXTENSION_PROMPT))
        assert len(harness.probed_suffixes) == 2, "the 9728 prompt: {9728, 9472}"
        harness = hybrid_harness()
        harness.get_num_new_matched_tokens(_request(TAIL_BOUNDARY))
        assert len(harness.probed_suffixes) == 1, "the 9472 prompt: {9472}"

    def test_pathological_prompt_is_pmu_bounded(self):
        prompt = 12 * HYBRID_BLOCK + 1147
        floor = prompt // HYBRID_BLOCK * HYBRID_BLOCK
        assert floor == 12 * HYBRID_BLOCK
        harness = hybrid_harness()
        harness.get_num_new_matched_tokens(_request(prompt))
        gap = prompt - floor
        assert len(harness.probed_suffixes) == gap // HYBRID_PMU + 1
        assert len(harness.probed_suffixes) == 5
        assert len(harness.probed_suffixes) != gap, "the walk must never be O(prompt - floor)"

    def test_text_prompt_end_is_the_single_candidate(self):
        harness = text_harness()
        harness.get_num_new_matched_tokens(_request(TEXT_PROMPT))
        assert len(harness.probed_suffixes) == 1


class TestLoadSpecPlumbing(unittest.TestCase):
    def test_interior_boundary_rides_the_existing_field(self):
        """The LoadSpec carries the interior boundary unchanged — the field
        exists, no schema change; task 5's load + ladder consume it."""
        harness = hybrid_harness()
        request = _request(EXTENSION_PROMPT, request_id="req-plumb")
        harness.persist_tail(request.block_hashes, TAIL_BOUNDARY)
        harness.get_num_new_matched_tokens(request)
        spec = harness.load_spec("req-plumb")
        assert spec.tail_boundary_tokens == TAIL_BOUNDARY
        assert spec.kvpool_cached_tokens == TAIL_BOUNDARY
        assert spec.kvpool_store_skip_tokens == TAIL_BOUNDARY
        assert spec.vllm_cached_tokens == 0

    def test_declined_walk_leaves_the_field_none(self):
        harness = hybrid_harness()
        request = _request(TAIL_BOUNDARY, request_id="req-plumb-declined")
        harness.persist_tail(request.block_hashes, TAIL_BOUNDARY)
        harness.get_num_new_matched_tokens(request)
        spec = harness.load_spec("req-plumb-declined")
        assert spec.tail_boundary_tokens is None
        assert spec.kvpool_cached_tokens == HYBRID_FLOOR
