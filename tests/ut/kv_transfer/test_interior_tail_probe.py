# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""hybrid-gdn-tail-state T4 + hybrid-robustness-hunt W2: the interior-boundary
tail probe, the honest -1 refusal, and the W1-20/24/27 hardening.

``get_num_new_matched_tokens`` walks the tail candidates NEWEST-first (the
step-aligned interior boundaries plus the prompt's exact end) instead of
probing only the prompt end: an extension-shaped consumer (prompt 9728 over
a 9472 artifact) claims the interior boundary, so the GDN composition is
exact (src=dst=slot, no copy). A hit at b == num_tokens on a GDN layout is
DECLINED — the boundary-state artifact cannot serve the -1's re-added token
(the structural off-by-one) — and the walk continues to older candidates.
Text/single-group layouts keep the T18 exact-end extension byte-identically
(the unaligned prompt end stays the newest candidate; the refusal's layout
gate is empty for text).

W2 (hybrid-robustness-hunt): W1-20 — a winning candidate also verifies the
intermediate full blocks the claim extends the load into (the chained-hash
law's eviction-breaking counterfactual); W1-24 — the refusal's meeting-point
pin; W1-27 — the walk steps by the finest configured boundary granularity.

The fixtures stub only the store I/O (``_make_layerwise_hit_check_keys`` /
``_query_layerwise_block_hits``) and the full-block hit; the REAL probe
derivation (the raw-chain hash index, the aligned reject), the REAL
intermediate-block span derivation, and the REAL walk run in every test.
"""

import unittest
from math import lcm
from types import SimpleNamespace

from vllm.logger import logger

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_scheduler as pool_scheduler_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    block_hash_to_str,
    get_block_hashes,
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
# The W1-20 heterogeneous geometry: the FA group's block (512) is FINER
# than the cache-transfer granularity (1536, set by the GDN groups' align
# blocks), hbs 256 — so a tail boundary past floor+512 extends the claim
# into FA full blocks the floor-anchored hit never verified (the
# intermediate span (floor//512, B//512)); the GDN groups' span stays
# empty (their block IS the granularity). Uniform-block layouts (the bench
# hybrid: every group 1536) have a structurally EMPTY span — the dominant
# live shape, pinned separately below.
HET_FA_BLOCK = 512
HET_GDN_BLOCK = 1536
HET_HBS = 256
HET_GRANULARITY = 1536
HET_FLOOR = 1536
HET_TAIL = 2304
HET_PROMPT = 2560
# The W1-27 dcp=2 geometry: pmu 256, hash_block_size 512 — pmu-aligned but
# not hbs-aligned interior boundaries exist only under dcp > 1. The prompt
# (4480) sits above the interior candidate (4352) with the floor at 4096.
DCP2_BLOCK = 512
DCP2_PMU = 256
DCP2_HBS = 512
DCP2_FLOOR = 8 * DCP2_BLOCK
DCP2_INTERIOR = DCP2_FLOOR + DCP2_PMU
DCP2_PROMPT = DCP2_INTERIOR + DCP2_PMU // 2


def _block_hashes(count: int = NUM_HASHES) -> list[str]:
    return [f"{index:064x}" for index in range(count)]


def _request(prompt_len: int, request_id: str = "req-T4") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        num_tokens=prompt_len,
        block_hashes=_block_hashes(),
    )


class _StubReachabilityCoordinator:
    """Duck-typed AscendStoreCoordinator for the probe's mask consult.

    Models the worker's ``_compute_reachable_load_masks`` input: per-group
    load masks at the CLAIMED boundary (None = the group has no
    reachability limits). Records every load_mask call so the tests can pin
    the claimed-boundary argument (the exact mirror of the worker's call).
    """

    def __init__(self, num_groups: int, masks: dict[int, list[bool]] | None = None):
        self._num_groups = num_groups
        self._masks = masks or {}
        self.load_mask_calls: list[tuple[list, int]] = []

    def load_mask(self, block_hashes, token_len: int):
        self.load_mask_calls.append((block_hashes, token_len))
        return tuple(self._masks.get(group) for group in range(self._num_groups))


class TailProbeHarness:
    """A KVPoolScheduler stub with the REAL probe + a fake persisted-tail store.

    The full-block hit defaults to COMPLETE at the granularity floor (the
    walk's entry condition); every key probe (tail suffixes and intermediate
    full-block hashes) is recorded for the order and census pins.
    ``persist_tail`` derives the artifact key exactly the way the worker's
    save path does (the raw-chain hash at boundary//hash_block_size-1);
    ``persist_full_block`` persists a full block under the grouped hash the
    producer's save pass would use (the _LazyGroupedBlockHashList law).
    """

    def __init__(
        self,
        *,
        block: int,
        pmu: int,
        mamba_groups: tuple[int, ...],
        hash_block_size: int | None = None,
        group_blocks: list[int] | None = None,
        load_masks: dict[int, list[bool]] | None = None,
    ):
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
        group_block_sizes = group_blocks if group_blocks is not None else [block] * num_groups
        granularity = lcm_block_size = group_block_sizes[0]
        for group_block_size in group_block_sizes[1:]:
            granularity = lcm_block_size = lcm(granularity, group_block_size)
        sched.grouped_block_size = group_block_sizes
        sched.original_block_size = list(group_block_sizes)
        sched.cacheable_group_ids = list(range(num_groups))
        sched.cache_transfer_granularity = granularity
        sched.lcm_block_size = lcm_block_size
        sched.hash_block_size = hash_block_size if hash_block_size is not None else pmu
        sched.prefix_match_unit = pmu
        sched.block_key_hybrid = bool(mamba_groups)
        sched.cache_coordinator = _StubReachabilityCoordinator(num_groups, load_masks) if mamba_groups else None
        sched.num_speculative_blocks_by_group = dict.fromkeys(mamba_groups, 0)
        self.pmu = pmu
        self.persisted_keys: set[str] = set()
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
        keys = [key for block_keys in keys_by_block for key in block_keys]
        self.probed_suffixes.extend(keys)
        return [key in self.persisted_keys for key in keys]

    @property
    def coordinator(self) -> _StubReachabilityCoordinator:
        assert self.scheduler.cache_coordinator is not None
        return self.scheduler.cache_coordinator

    def persist_tail(self, block_hashes: list[str], boundary: int) -> None:
        self.persisted_keys.add(self.tail_key(block_hashes, boundary))

    def tail_key(self, block_hashes: list[str], boundary: int) -> str:
        hash_block_size = self.scheduler.hash_block_size
        return (
            f"g0|{partial_tail_key_suffix(block_hash_to_str(block_hashes[boundary // hash_block_size - 1]), boundary)}"
        )

    def suffix(self, block_hashes: list[str], boundary: int) -> str:
        return self.tail_key(block_hashes, boundary)

    def full_block_key(self, block_hashes: list[str], group_id: int, block_index: int) -> str:
        block_size = self.scheduler.grouped_block_size[group_id]
        grouped = get_block_hashes(block_hashes, block_size, self.scheduler.hash_block_size)
        return f"g{group_id}|{block_hash_to_str(grouped[block_index])}"

    def persist_full_block(self, block_hashes: list[str], group_id: int, block_index: int) -> None:
        self.persisted_keys.add(self.full_block_key(block_hashes, group_id, block_index))

    def get_num_new_matched_tokens(self, request) -> tuple[int, bool]:
        return self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens=0)

    def load_spec(self, request_id: str = "req-T4"):
        return self.scheduler.load_specs[request_id]


def hybrid_harness() -> TailProbeHarness:
    return TailProbeHarness(block=HYBRID_BLOCK, pmu=HYBRID_PMU, mamba_groups=(1, 2, 3))


def text_harness() -> TailProbeHarness:
    return TailProbeHarness(block=TEXT_BLOCK, pmu=TEXT_BLOCK, mamba_groups=())


def hetero_harness(load_masks: dict[int, list[bool]] | None = None) -> TailProbeHarness:
    return TailProbeHarness(
        block=HET_FA_BLOCK,
        pmu=HET_HBS,
        mamba_groups=(1, 2, 3),
        group_blocks=[HET_FA_BLOCK, HET_GDN_BLOCK, HET_GDN_BLOCK, HET_GDN_BLOCK],
        load_masks=load_masks,
    )


def dcp2_harness() -> TailProbeHarness:
    return TailProbeHarness(
        block=DCP2_BLOCK,
        pmu=DCP2_PMU,
        mamba_groups=(),
        hash_block_size=DCP2_HBS,
    )


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


def _persist_producer_save_pass(harness: TailProbeHarness, request, boundary: int) -> None:
    """Persist the producer's save pass at ``boundary``: the tail artifact
    plus every full block below the boundary (all groups), under the same
    chained grouped hashes the probe derives — the store state the
    chained-hash law guarantees for a genuine tail hit."""
    harness.persist_tail(request.block_hashes, boundary)
    for group_id in harness.scheduler.cacheable_group_ids:
        block_size = harness.scheduler.grouped_block_size[group_id]
        grouped = get_block_hashes(request.block_hashes, block_size, harness.scheduler.hash_block_size)
        for index in range(boundary // block_size):
            harness.persisted_keys.add(f"g{group_id}|{block_hash_to_str(grouped[index])}")


class TestIntermediateFullBlockVerification(unittest.TestCase):
    """W1-20: a winning tail candidate verifies the intermediate full blocks
    the claim extends the load into — the chained-hash law's counterfactual
    (eviction breaks "tail hit implies the prefix blocks exist") is caught
    at probe time: the candidate is declined and the walk continues, never
    a claim whose load fails in the engine."""

    def test_deep_tail_hit_verifies_and_claims_with_all_intermediate_blocks(self):
        harness = hetero_harness()
        request = _request(HET_PROMPT)
        _persist_producer_save_pass(harness, request, HET_TAIL)
        with self.assertLogs(logger, level="INFO") as captured:
            need_to_allocate, _ = harness.get_num_new_matched_tokens(request)
        assert need_to_allocate == HET_TAIL
        assert harness.load_spec().tail_boundary_tokens == HET_TAIL
        # the FA group's intermediate block 3 (the span (floor//512, 2304//512))
        # was readability-probed; the GDN groups' spans are empty (their block
        # IS the granularity) so no GDN keys were queried
        assert harness.full_block_key(request.block_hashes, 0, 3) in harness.probed_suffixes
        for group_id in (1, 2, 3):
            assert harness.full_block_key(request.block_hashes, group_id, 0) not in harness.probed_suffixes
        assert not any("KV pool partial tail declined" in line for line in captured.output)
        # the mask consult ran exactly once, at the CLAIMED boundary — the
        # mirror of the worker's _compute_reachable_load_masks(cached_tokens=B)
        assert [tokens for _, tokens in harness.coordinator.load_mask_calls] == [HET_TAIL]

    def test_within_one_block_tail_never_probes_full_blocks_or_masks(self):
        """The dominant live shape (the bench hybrid: uniform 1536 blocks —
        the intermediate span is structurally empty): a tail hit adds ZERO
        full-block probes and ZERO coordinator consults."""
        harness = hybrid_harness()
        request = _request(EXTENSION_PROMPT)
        harness.persist_tail(request.block_hashes, TAIL_BOUNDARY)
        harness.get_num_new_matched_tokens(request)
        assert len(harness.probed_suffixes) == 2
        assert harness.probed_suffixes == [
            harness.tail_key(request.block_hashes, EXTENSION_PROMPT),
            harness.tail_key(request.block_hashes, TAIL_BOUNDARY),
        ]
        assert harness.coordinator.load_mask_calls == []

    def test_missing_fa_intermediate_block_declines_the_candidate(self):
        harness = hetero_harness()
        request = _request(HET_PROMPT)
        _persist_producer_save_pass(harness, request, HET_TAIL)
        harness.persisted_keys.discard(harness.full_block_key(request.block_hashes, 0, 3))
        with self.assertLogs(logger, level="INFO") as captured:
            need_to_allocate, _ = harness.get_num_new_matched_tokens(request)
        assert need_to_allocate == HET_FLOOR
        assert harness.load_spec().tail_boundary_tokens is None
        declined = [line for line in captured.output if "reason=intermediate_block_missing" in line]
        assert len(declined) == 1
        assert f"boundary={HET_TAIL}" in declined[0]
        assert "(0, 3)" in declined[0]
        assert not any("KV pool partial tail hit" in line for line in captured.output)

    def test_declined_candidate_falls_through_to_an_older_valid_boundary(self):
        harness = hetero_harness()
        request = _request(HET_PROMPT)
        _persist_producer_save_pass(harness, request, HET_TAIL)
        harness.persisted_keys.discard(harness.full_block_key(request.block_hashes, 0, 3))
        older = HET_FLOOR + HET_HBS  # 1792: FA-unaligned, span (3, 3) empty
        harness.persist_tail(request.block_hashes, older)
        with self.assertLogs(logger, level="INFO") as captured:
            need_to_allocate, _ = harness.get_num_new_matched_tokens(request)
        assert need_to_allocate == older
        assert harness.load_spec().tail_boundary_tokens == older
        declined = [line for line in captured.output if "reason=intermediate_block_missing" in line]
        assert len(declined) == 1 and f"boundary={HET_TAIL}" in declined[0]
        hit_lines = [line for line in captured.output if "KV pool partial tail hit" in line]
        assert len(hit_lines) == 1 and f"exact_end={older}" in hit_lines[0]

    def test_missing_gdn_intermediate_block_declines_the_candidate(self):
        # the mirrored geometry: the FA group's block IS the granularity, the
        # GDN groups' (512) is finer — a missing GDN state object declines
        # exactly like a missing FA block (the claim covers every group)
        harness = TailProbeHarness(
            block=HET_GDN_BLOCK,
            pmu=HET_HBS,
            mamba_groups=(1, 2, 3),
            group_blocks=[HET_GDN_BLOCK, HET_FA_BLOCK, HET_FA_BLOCK, HET_FA_BLOCK],
        )
        request = _request(HET_PROMPT)
        _persist_producer_save_pass(harness, request, HET_TAIL)
        harness.persisted_keys.discard(harness.full_block_key(request.block_hashes, 1, 3))
        with self.assertLogs(logger, level="INFO") as captured:
            need_to_allocate, _ = harness.get_num_new_matched_tokens(request)
        assert need_to_allocate == HET_FLOOR
        declined = [line for line in captured.output if "reason=intermediate_block_missing" in line]
        assert len(declined) == 1
        assert "(1, 3)" in declined[0]

    def test_masked_out_intermediate_block_is_not_required(self):
        # the mask mirror: a block the worker's load mask will not fetch is
        # neither probed nor required — a masked-out missing block must not
        # falsely decline a valid candidate
        mask = {0: [True, True, True, False]}
        harness = hetero_harness(load_masks=mask)
        request = _request(HET_PROMPT)
        _persist_producer_save_pass(harness, request, HET_TAIL)
        harness.persisted_keys.discard(harness.full_block_key(request.block_hashes, 0, 3))
        with self.assertLogs(logger, level="INFO") as captured:
            need_to_allocate, _ = harness.get_num_new_matched_tokens(request)
        assert need_to_allocate == HET_TAIL
        assert harness.load_spec().tail_boundary_tokens == HET_TAIL
        assert harness.full_block_key(request.block_hashes, 0, 3) not in harness.probed_suffixes
        assert not any("KV pool partial tail declined" in line for line in captured.output)

    def test_local_computed_boundary_skips_below_span(self):
        # the fetch-start mirror: blocks below max(floor, num_computed_tokens)
        # are local — the worker never fetches them, so the probe must not
        # require them (a missing below-local block does not decline)
        harness = hetero_harness()
        request = _request(HET_PROMPT)
        _persist_producer_save_pass(harness, request, HET_TAIL)
        harness.persisted_keys.discard(harness.full_block_key(request.block_hashes, 0, 3))
        local_computed = 4 * HET_FA_BLOCK  # 2048: block 3 is below the local hit
        need_to_allocate, _ = harness.scheduler.get_num_new_matched_tokens(request, local_computed)
        assert need_to_allocate == HET_TAIL - local_computed
        assert harness.load_spec().tail_boundary_tokens == HET_TAIL
        assert harness.full_block_key(request.block_hashes, 0, 3) not in harness.probed_suffixes


class TestTailClaimChainLaw(unittest.TestCase):
    """W1-20's safety argument, pinned: the intermediate full-block keys the
    probe verifies are derived from the SAME chained raw hash list as the
    boundary sub-hash — block i's identity is raw-chain element
    (i+1)*scale-1 (the _LazyGroupedBlockHashList law), every intermediate
    element strictly below the boundary element — so a producer that saved
    the tail under the boundary sub-hash necessarily saved the intermediate
    blocks under the keys this probe derives."""

    def test_grouped_hash_laws(self):
        chain = _block_hashes()
        # bs == hbs: the grouped view IS the raw chain (the text law — the
        # flat block_hashes[logical_block] read is correct only here)
        assert get_block_hashes(chain, HET_HBS, HET_HBS) is chain
        # bs > hbs: block i's identity is the LAST sub-hash of its group
        scale = HET_FA_BLOCK // HET_HBS
        grouped = get_block_hashes(chain, HET_FA_BLOCK, HET_HBS)
        assert len(grouped) == len(chain) // scale
        assert grouped[0] == chain[scale - 1]
        assert grouped[3] == chain[4 * scale - 1]

    def test_intermediate_elements_are_contained_in_the_boundary_chain(self):
        scale = HET_FA_BLOCK // HET_HBS
        boundary_hash_index = HET_TAIL // HET_HBS - 1
        for index in range(HET_FLOOR // HET_FA_BLOCK, HET_TAIL // HET_FA_BLOCK):
            assert (index + 1) * scale - 1 < boundary_hash_index
        # the boundary element is the last COMPLETE hash block at the boundary
        assert (boundary_hash_index + 1) * HET_HBS <= HET_TAIL < (boundary_hash_index + 2) * HET_HBS

    def test_producer_save_pass_satisfies_the_verification_exactly(self):
        # end-to-end: the producer's save pass at the boundary is EXACTLY the
        # key set the verification queries for the span — persisting it makes
        # the check find zero missing (the negative control — evicting one —
        # is TestIntermediateFullBlockVerification's decline pin)
        harness = hetero_harness()
        request = _request(HET_PROMPT)
        _persist_producer_save_pass(harness, request, HET_TAIL)
        missing = harness.scheduler._unreadable_intermediate_blocks(request, HET_TAIL, HET_FLOOR)
        assert missing == []


class TestRefusalMeetingPointPin(unittest.TestCase):
    """W1-24: the refusal compares a prompt-derived candidate (always
    <= prompt_token_len) against request.num_tokens. A resumed preempted
    request diverges legitimately (num_tokens = prompt + generated) — there
    the refusal must stay silent and the interior claim must stand. When the
    refusal DOES fire, the hit must be the prompt's exact end."""

    def test_resumed_preempted_request_claims_interior_boundary_without_refusal(self):
        harness = hybrid_harness()
        request = _request(EXTENSION_PROMPT)
        request.num_tokens = EXTENSION_PROMPT + 500
        harness.persist_tail(request.block_hashes, TAIL_BOUNDARY)
        with self.assertLogs(logger, level="INFO") as captured:
            need_to_allocate, _ = harness.get_num_new_matched_tokens(request)
        assert need_to_allocate == TAIL_BOUNDARY
        assert harness.load_spec().tail_boundary_tokens == TAIL_BOUNDARY
        assert not any("KV pool partial tail declined" in line for line in captured.output)

    def test_refusal_off_prompt_end_is_loud(self):
        # the drift canary: num_tokens BELOW the prompt (an impossible Request
        # today — the shape a future candidate-grid or num_tokens drift would
        # produce). The refusal firing there must raise, never silently
        # decline a wrong boundary.
        harness = hybrid_harness()
        request = _request(EXTENSION_PROMPT)
        request.num_tokens = TAIL_BOUNDARY
        harness.persist_tail(request.block_hashes, TAIL_BOUNDARY)
        with self.assertRaises(AssertionError) as raised:
            harness.get_num_new_matched_tokens(request)
        assert "prompt end" in str(raised.exception)


class TestPmuStepWalk(unittest.TestCase):
    """W1-27: under dcp > 1 the hash grid (pmu * dcp) is coarser than the
    configured pmu; the walk must step by the finest configured granularity
    or pmu-aligned-but-not-hbs-aligned tail artifacts are never probed."""

    def test_dcp2_pmu_aligned_interior_boundary_is_probed_and_hit(self):
        harness = dcp2_harness()
        request = _request(DCP2_PROMPT)
        harness.persist_tail(request.block_hashes, DCP2_INTERIOR)
        with self.assertLogs(logger, level="INFO") as captured:
            need_to_allocate, _ = harness.get_num_new_matched_tokens(request)
        assert need_to_allocate == DCP2_INTERIOR
        assert harness.load_spec().tail_boundary_tokens == DCP2_INTERIOR
        assert harness.probed_suffixes == [
            harness.tail_key(request.block_hashes, DCP2_PROMPT),
            harness.tail_key(request.block_hashes, DCP2_INTERIOR),
        ]
        assert not any("KV pool partial tail declined" in line for line in captured.output)

    def test_dcp1_walk_candidates_stay_on_the_hbs_grid(self):
        # the bench geometry (pmu == hbs): the step degenerates to hbs —
        # byte-identical candidates to the pre-W2 grid
        harness = hybrid_harness()
        assert harness.scheduler._partial_tail_probe_boundaries(EXTENSION_PROMPT, HYBRID_FLOOR) == [
            EXTENSION_PROMPT,
            TAIL_BOUNDARY,
        ]
        assert harness.scheduler._partial_tail_probe_boundaries(TAIL_BOUNDARY, HYBRID_FLOOR) == [TAIL_BOUNDARY]
        # the pathological shape stays pmu-bounded
        prompt = 12 * HYBRID_BLOCK + 1147
        floor = prompt // HYBRID_BLOCK * HYBRID_BLOCK
        candidates = harness.scheduler._partial_tail_probe_boundaries(prompt, floor)
        assert len(candidates) == (prompt - floor) // HYBRID_PMU + 1
