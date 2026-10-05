# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN state-load ladder (T14): the 4-way fallback contract, markers, glue.

The ladder module is pure (an injected boundary-attempt callable), so the
4-way semantics — present->load / older->load+recompute-span / none->
zero-recompute / load-failure->demote+loud — are proven exactly at the
contract level; the glue tests drive the real pool-worker plan builder and
ladder runner over a stubbed worker (the T5 __new__ pattern). The marker
lines are asserted byte-exact against the frozen v2 vocabulary regexes.
"""

import re
import unittest
from types import SimpleNamespace

import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import gdn_state_ladder, pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.gdn_state_ladder import (
    REASON_LOAD_FAILED,
    RUNG_LAST_SAVED,
    RUNG_OLDER_SAVED,
    RUNG_ZERO_RECOMPUTE,
    SOURCE_LAST_SAVED,
    SOURCE_OLDER_SAVED,
    StateBoundary,
    build_state_load_plan,
    run_gdn_state_load_ladder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import ReqMeta

# The frozen v2 vocabulary regexes (holemiss-mooncake marker_vocab.py,
# FROZEN at T8) — copied here so the contract test does not depend on the
# bench repo being importable; drift is caught by the T8 suite itself.
STATE_LOAD_RE = re.compile(
    r"HOLEMISS_STATE_LOAD layer=(-?\d+) group=(-?\d+) boundary=(-?\d+) "
    r"slot=(-?\d+) source=(\S+) rung=(-?\d+) bytes=(-?\d+) sha=(\S+)"
)
STATE_ABSENT_RE = re.compile(
    r"HOLEMISS_STATE_ABSENT layer=(-?\d+) group=(-?\d+) boundary=(-?\d+) "
    r"reason=(\S+) rung=(-?\d+)"
)

GROUPS = (1, 2, 3)
BLOCK = 1536
OBJECT_BYTES = 25657344


def _boundary(index: int, first_layer: int = 1) -> StateBoundary:
    return StateBoundary(
        block_index=index,
        boundary_tokens=(index + 1) * BLOCK,
        keys_by_group={group: (f"m@layout@group:{group}@block:1536@h{index}@0",) for group in GROUPS},
        blocks_by_group={group: (1000 * group + index,) for group in GROUPS},
        first_layer_by_group={group: first_layer for group in GROUPS},
    )


def _plan(save_enabled: bool = True, num_boundaries: int = 4, fa_hit: int = 4 * BLOCK):
    return build_state_load_plan(
        req_id="req-L",
        state_groups=GROUPS,
        boundaries=[_boundary(index) for index in range(num_boundaries)],
        fa_hit_tokens=fa_hit,
        save_enabled=save_enabled,
    )


class TestFourWayLadder(unittest.TestCase):
    def _run(self, plan, attempts):
        """attempts: block_index -> (attempted, loaded); records call order."""
        markers: list[str] = []
        order: list[int] = []

        def attempt(boundary):
            order.append(boundary.block_index)
            attempted, loaded = attempts[boundary.block_index]
            return gdn_state_ladder.BoundaryOutcome(
                attempted=attempted,
                loaded=loaded,
                payload_bytes=OBJECT_BYTES if loaded else 0,
                content_sha="ab" * 32 if loaded else None,
            )

        result = run_gdn_state_load_ladder(plan, attempt, markers.append)
        return result, markers, order

    def test_rung0_present_loads_newest(self):
        result, markers, order = self._run(_plan(), {3: (True, True)})
        self.assertEqual(order, [3])
        self.assertEqual(result.rung, RUNG_LAST_SAVED)
        self.assertEqual(result.source, SOURCE_LAST_SAVED)
        self.assertEqual(result.loaded_boundary_tokens, 4 * BLOCK)
        self.assertEqual(result.recompute_span, (4 * BLOCK, 4 * BLOCK))
        self.assertEqual(result.state_load_block_ids, {1000 * g + 3 for g in GROUPS})
        self.assertFalse(result.failed_block_ids)
        self.assertEqual(len(markers), len(GROUPS))
        for line in markers:
            self.assertRegex(line, STATE_LOAD_RE)
            self.assertIn("source=last_saved", line)
            self.assertIn("rung=0", line)
            self.assertIn(f"bytes={OBJECT_BYTES}", line)
            self.assertIn("sha=ab", line)

    def test_rung1_older_loads_with_recompute_span(self):
        # The T13 self-hit race shape: the newest boundary's GET fails.
        result, markers, order = self._run(_plan(), {3: (True, False), 2: (True, True)})
        self.assertEqual(order, [3, 2])
        self.assertEqual(result.rung, RUNG_OLDER_SAVED)
        self.assertEqual(result.source, SOURCE_OLDER_SAVED)
        self.assertEqual(result.loaded_boundary_tokens, 3 * BLOCK)
        self.assertEqual(result.recompute_span, (3 * BLOCK, 4 * BLOCK))
        self.assertEqual(result.demoted, ((4 * BLOCK, REASON_LOAD_FAILED),))
        absent = [line for line in markers if line.startswith("HOLEMISS_STATE_ABSENT")]
        loaded = [line for line in markers if line.startswith("HOLEMISS_STATE_LOAD")]
        self.assertEqual(len(absent), len(GROUPS))
        self.assertEqual(len(loaded), len(GROUPS))
        for line in absent:
            self.assertRegex(line, STATE_ABSENT_RE)
            self.assertIn(f"boundary={4 * BLOCK}", line)
            self.assertIn("reason=load_failed", line)
            self.assertIn("rung=1", line)
        for line in loaded:
            self.assertIn("source=older_saved", line)
            self.assertIn("rung=1", line)

    def test_rung2_none_loads_zero_recompute(self):
        result, markers, order = self._run(_plan(), {index: (True, False) for index in range(4)})
        self.assertEqual(order, [3, 2, 1, 0])
        self.assertEqual(result.rung, RUNG_ZERO_RECOMPUTE)
        self.assertIsNone(result.source)
        self.assertEqual(result.loaded_boundary_tokens, 0)
        self.assertEqual(result.recompute_span, (0, 4 * BLOCK))
        self.assertEqual(
            result.failed_block_ids,
            {1000 * g + index for g in GROUPS for index in range(4)},
        )
        loads = [line for line in markers if line.startswith("HOLEMISS_STATE_LOAD")]
        self.assertEqual(loads, [], "a phase that demotes to zero-recompute must not emit STATE_LOAD")
        absents = [line for line in markers if line.startswith("HOLEMISS_STATE_ABSENT")]
        self.assertEqual(len(absents), 4 * len(GROUPS))
        for line in absents:
            self.assertRegex(line, STATE_ABSENT_RE)
        # Only the final demotion (the oldest boundary) carries the
        # exhausted rung; the newest demotes toward the older-saved rung.
        self.assertTrue(any("rung=2" in line and f"boundary={BLOCK}" in line for line in absents))
        self.assertTrue(any("rung=1" in line and f"boundary={4 * BLOCK}" in line for line in absents))

    def test_missing_objects_demote_with_missing_reason(self):
        result, markers, _ = self._run(_plan(), {3: (False, False), 2: (True, True)})
        self.assertEqual(result.rung, RUNG_OLDER_SAVED)
        self.assertEqual(result.demoted, ((4 * BLOCK, "missing"),))

    def test_save_disabled_short_circuits_to_rung2(self):
        # G12: the save channel is off; use_layerwise=false would still bulk
        # save, so the ladder keys on the save-channel signal, never the flag.
        result, markers, order = self._run(_plan(save_enabled=False), {})
        self.assertEqual(order, [], "save-disabled never attempts a load")
        self.assertEqual(result.rung, RUNG_ZERO_RECOMPUTE)
        self.assertTrue(
            any("reason=save_disabled" in line for line in markers),
            markers,
        )

    def test_text_plan_is_none(self):
        self.assertIsNone(build_state_load_plan("req", [], [], fa_hit_tokens=0, save_enabled=True))


class TestMarkerParserContract(unittest.TestCase):
    """The emitted lines parse under the frozen v2 grammar (evidence_parse)."""

    def test_every_marker_line_matches_frozen_regexes(self):
        def _scenario_markers(attempts):
            markers: list[str] = []

            def attempt(boundary, _attempts=attempts):
                attempted, loaded = _attempts[boundary.block_index]
                return gdn_state_ladder.BoundaryOutcome(attempted=attempted, loaded=loaded, payload_bytes=OBJECT_BYTES)

            run_gdn_state_load_ladder(_plan(), attempt, markers.append)
            return markers

        scenarios = [
            {3: (True, True)},
            {3: (True, False), 2: (True, True)},
            {index: (True, False) for index in range(4)},
        ]
        for attempts in scenarios:
            with self.subTest(attempts=attempts):
                markers = _scenario_markers(attempts)
                for line in markers:
                    if line.startswith("HOLEMISS_STATE_LOAD"):
                        match = STATE_LOAD_RE.fullmatch(line)
                        self.assertIsNotNone(match, line)
                        self.assertIn(match.group(5), ("last_saved", "older_saved"))
                    else:
                        self.assertIsNotNone(STATE_ABSENT_RE.fullmatch(line), line)

    def test_state_load_emission_follows_attempt_and_no_absent_for_loaded_boundary(self):
        events: list[str] = []

        def attempt(boundary):
            events.append(f"attempt:{boundary.block_index}")
            return gdn_state_ladder.BoundaryOutcome(
                attempted=True, loaded=boundary.block_index == 2, payload_bytes=OBJECT_BYTES
            )

        markers: list[str] = []
        run_gdn_state_load_ladder(_plan(), attempt, markers.append)
        # The vocabulary's ordering law: STATE_LOAD is emitted after the
        # boundary copy (the attempt) returns, and a boundary that emitted
        # ABSENT never also emits LOAD.
        self.assertEqual(events, ["attempt:3", "attempt:2"])
        self.assertTrue(any("STATE_LOAD" in line and f"boundary={3 * BLOCK}" in line for line in markers))
        self.assertTrue(any("STATE_ABSENT" in line and f"boundary={4 * BLOCK}" in line for line in markers))
        self.assertFalse(any("STATE_ABSENT" in line and f"boundary={3 * BLOCK}" in line for line in markers))


def _stub_worker(num_state_groups: int = 3):
    """A T5-style stub worker with a real hybrid cache config (3 GDN groups)."""
    pool = pool_worker.KVPoolWorker.__new__(pool_worker.KVPoolWorker)
    pool.use_hybrid = True
    pool.use_eagle = False
    pool.tp_rank = 0
    pool.head_or_tp_rank = 0
    pool.pp_rank = 0
    pool.hash_block_size = BLOCK
    pool.grouped_block_size = [BLOCK] * (num_state_groups + 1)
    pool.group_block_len = {group: [OBJECT_BYTES] for group in range(1, num_state_groups + 1)}
    pool.hf_config = SimpleNamespace(num_hidden_layers=8)
    pool.num_layers = 8
    pool._global_to_local_layer = {physical: physical for physical in range(8)}
    pool.layerwise_keys = SimpleNamespace(
        make_full_key=lambda group, block_hash, head, pp_rank: (
            f"m@layout@group:{group}@block:1536@{block_hash}@{head}"
        )
    )
    pool._holemiss_state_load_results = {}
    pool._holemiss_state_markers = {}
    pool._holemiss_failed_load_keys = {}
    pool._holemiss_req_meta_by_id = {}
    pool._holemiss_state_dispositioned_blocks = set()
    pool._invalid_block_ids = set()
    pool._invalid_block_ids_lock = pool_worker.threading.Lock()
    pool._layer_load_aborted = pool_worker.threading.Event()
    pool._hole_recovery_regions = []
    pool._hole_recovery_lock = pool_worker.threading.Lock()
    pool._holemiss_armed_fault = None
    pool.use_layerwise = True
    pool.layer_save_tasks = []
    pool.layer_load_tasks = []
    pool.next_layer_to_submit = 0
    pool._attention_saved_layers = set()

    full = FullAttentionSpec(block_size=BLOCK, num_kv_heads=2, head_size=128, dtype=torch.float32)
    mamba = MambaSpec(
        block_size=BLOCK,
        shapes=((8,), (8,)),
        dtypes=(torch.float32, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )
    pool.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=full, layer_names=["model.layers.0.attn", "model.layers.4.attn"]),
            *(
                SimpleNamespace(
                    kv_cache_spec=mamba,
                    layer_names=[f"model.layers.{layer}.attn" for layer in (1 + group_index,)],
                )
                for group_index in range(num_state_groups)
            ),
        ]
    )
    return pool


def _hybrid_request(req_id="req-H", num_blocks=4):
    return ReqMeta(
        req_id=req_id,
        block_ids_by_group=[[10 + i for i in range(num_blocks)]]
        + [[1000 * group + i for i in range(num_blocks)] for group in (1, 2, 3)],
        block_hashes=[f"{index:064x}" for index in range(num_blocks)],
        can_save=False,
        load_spec=SimpleNamespace(
            can_load=True,
            kvpool_cached_tokens=num_blocks * BLOCK,
            kvpool_store_skip_tokens=None,
            vllm_cached_tokens=0,
        ),
    )


class TestWorkerGlue(unittest.TestCase):
    def test_plan_builder_enumerates_boundaries_newest_last(self):
        pool = _stub_worker()
        request = _hybrid_request()
        plan = pool._holemiss_build_state_ladder_plan(request)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.state_groups, GROUPS)
        self.assertEqual([b.boundary_tokens for b in plan.boundaries], [BLOCK, 2 * BLOCK, 3 * BLOCK, 4 * BLOCK])
        newest = plan.boundaries[-1]
        for group in GROUPS:
            keys = newest.keys_by_group[group]
            self.assertEqual(len(keys), 1)
            self.assertIn(f"@group:{group}@", keys[0])
            self.assertEqual(newest.blocks_by_group[group], (1000 * group + 3,))
        self.assertEqual(plan.fa_hit_tokens, 4 * BLOCK)
        self.assertTrue(plan.save_enabled)

    def test_ladder_run_rung0_and_marker_capture(self):
        pool = _stub_worker()
        request = _hybrid_request()
        result = pool._holemiss_run_gdn_state_ladder(request)
        self.assertEqual(result.rung, RUNG_LAST_SAVED)
        self.assertEqual(result.loaded_boundary_tokens, 4 * BLOCK)
        markers = pool._holemiss_state_markers["req-H"]
        self.assertEqual(len(markers), len(GROUPS))
        for line in markers:
            self.assertRegex(line, STATE_LOAD_RE)
        # G3's state_load_idx is recorded for the zeroing disjointness pin.
        self.assertEqual(result.state_load_block_ids, {1000 * g + 3 for g in GROUPS})

    def test_self_hit_race_demotes_loudly(self):
        # The T13 multi-chunk race: the newest boundary's own state GET fails.
        pool = _stub_worker()
        request = _hybrid_request()
        plan = pool._holemiss_build_state_ladder_plan(request)
        newest_keys = {key for group in GROUPS for key in plan.boundaries[-1].keys_by_group[group]}
        pool._holemiss_record_failed_load_keys("req-H", newest_keys)
        result = pool._holemiss_run_gdn_state_ladder(request)
        self.assertEqual(result.rung, RUNG_OLDER_SAVED)
        self.assertEqual(result.loaded_boundary_tokens, 3 * BLOCK)
        self.assertEqual(result.recompute_span, (3 * BLOCK, 4 * BLOCK))
        self.assertTrue(any("STATE_ABSENT" in line for line in pool._holemiss_state_markers["req-H"]))

    def test_rung2_dispositions_blocks_and_does_not_raise(self):
        pool = _stub_worker()
        request = _hybrid_request()
        plan = pool._holemiss_build_state_ladder_plan(request)
        all_keys = {key for boundary in plan.boundaries for group in GROUPS for key in boundary.keys_by_group[group]}
        pool._holemiss_record_failed_load_keys("req-H", all_keys)
        result = pool._holemiss_run_gdn_state_ladder(request)
        self.assertEqual(result.rung, RUNG_ZERO_RECOMPUTE)
        # The failed state blocks stay on the invalid-block surface ...
        self.assertTrue(pool._invalid_block_ids)
        self.assertEqual(pool._invalid_block_ids, set(result.failed_block_ids))
        # ... but are dispositioned, so the hybrid load-error check does not
        # re-raise (the whole-request recompute is the rung-2 semantics).
        pool._check_hybrid_load_errors()
        # An UNdispositioned (FA) failure still raises.
        pool._invalid_block_ids.add(999)
        with self.assertRaises(RuntimeError):
            pool._check_hybrid_load_errors()

    def test_save_disabled_env_yields_rung2(self):
        pool = _stub_worker()
        request = _hybrid_request()
        import os
        import unittest.mock

        with unittest.mock.patch.dict(os.environ, {"HOLEMISS_STATE_SAVE_DISABLED": "1"}):
            result = pool._holemiss_run_gdn_state_ladder(request)
        self.assertEqual(result.rung, RUNG_ZERO_RECOMPUTE)
        self.assertTrue(any("reason=save_disabled" in line for line in pool._holemiss_state_markers["req-H"]))

    def test_text_layout_is_structurally_inert(self):
        pool = _stub_worker()
        pool.use_hybrid = False
        self.assertIsNone(pool._holemiss_build_state_ladder_plan(_hybrid_request()))

    def test_connector_accessor_exposes_result(self):
        pool = _stub_worker()
        request = _hybrid_request()
        result = pool._holemiss_run_gdn_state_ladder(request)
        connector = AscendStoreConnector.__new__(AscendStoreConnector)
        connector.connector_worker = pool
        self.assertIs(connector.get_gdn_state_load_result("req-H"), result)
        self.assertIsNone(connector.get_gdn_state_load_result("req-other"))

    def test_rung1_recovery_survives_the_hybrid_load_error_check(self):
        """V-T14.1's live check: a rung-1 demote (the T13 self-hit race) must
        NOT hit the engine-fatal raise — the recompute span regenerates the
        demoted boundary's state; an FA failure added afterwards still raises.
        """
        pool = _stub_worker()
        request = _hybrid_request()
        plan = pool._holemiss_build_state_ladder_plan(request)
        newest_keys = {key for group in GROUPS for key in plan.boundaries[-1].keys_by_group[group]}
        pool._holemiss_record_failed_load_keys("req-H", newest_keys)
        # The get-session failure path also put the failed boundary's block
        # on the invalid-block surface (pool_worker :2378-2379) — reproduce
        # the recorder's live side effect.
        pool._invalid_block_ids.update(plan.boundaries[-1].blocks_by_group[GROUPS[0]])
        result = pool._holemiss_run_gdn_state_ladder(request)
        self.assertEqual(result.rung, RUNG_OLDER_SAVED)
        self.assertTrue(result.failed_block_ids, "rung 1 must carry the demoted boundary's failed blocks")
        pool._check_hybrid_load_errors()
        # An FA-group failure is never dispositioned by the ladder and keeps
        # the hard raise (the HR trigger is sacred).
        pool._invalid_block_ids.add(999)
        with self.assertRaises(RuntimeError):
            pool._check_hybrid_load_errors()

    def test_start_load_kv_clears_stale_dispositions(self):
        """V-T14.2: a disposition must not outlive its step — a recycled
        block id would silently mask a future FA failure's raise.
        """
        pool = _stub_worker()
        pool._holemiss_state_dispositioned_blocks.update({1234})
        pool.start_load_kv(SimpleNamespace(requests=[]))
        self.assertEqual(pool._holemiss_state_dispositioned_blocks, set())

    def test_teardown_finished_purges_per_request_records(self):
        pool = _stub_worker()
        request = _hybrid_request()
        pool._holemiss_run_gdn_state_ladder(request)
        pool._holemiss_record_failed_load_keys("req-H", {"some-key"})
        self.assertIn("req-H", pool._holemiss_state_load_results)
        self.assertIn("req-H", pool._holemiss_state_markers)
        self.assertIn("req-H", pool._holemiss_failed_load_keys)
        self.assertIn("req-H", pool._holemiss_req_meta_by_id)
        pool._holemiss_teardown_finished({"req-H"})
        self.assertNotIn("req-H", pool._holemiss_state_load_results)
        self.assertNotIn("req-H", pool._holemiss_state_markers)
        self.assertNotIn("req-H", pool._holemiss_failed_load_keys)
        self.assertNotIn("req-H", pool._holemiss_req_meta_by_id)

    def test_retry_success_discards_the_failed_key_record(self):
        """V-T14.4: the session tracker retries failed get-starts; a retry
        that succeeds must leave the failed-key record or the final ladder
        demotes a boundary whose retry actually loaded (a false STATE_ABSENT).
        """
        pool = _stub_worker()
        request = _hybrid_request()
        request.load_block_keys = []
        request.load_last_block_key = None
        request.load_keys = []
        pool._holemiss_record_failed_load_keys("req-H", {"m@layout@group:1@block:1536@h@0"})
        pool._layerwise_session_tracker = SimpleNamespace(record_get_result=lambda *args, **kwargs: None)
        key = "m@layout@group:1@block:1536@h@0"

        def fake_start(keys):
            return [0] * len(keys)

        with unittest.mock.patch.object(pool, "_start_layerwise_get_keys", side_effect=fake_start):
            pool._open_layerwise_get_sessions([(request, key, 42, 0)])
        self.assertEqual(pool._holemiss_failed_load_keys.get("req-H", set()), set())
        self.assertIn(key, request.load_keys)


class TestG3ZeroingPin(unittest.TestCase):
    """zero_idx vs state_load_idx: the zeroer never records mamba blocks."""

    def test_zeroer_records_attention_spec_blocks_only(self):
        import inspect

        source = inspect.getsource(__import__("vllm.v1.core.single_type_kv_cache_manager", fromlist=["x"]))
        anchor = "_record_new_block_ids = ("
        start = source.index(anchor)
        window = source[start : start + 300]
        self.assertIn("needs_kv_cache_zeroing", window)
        self.assertIn("isinstance(kv_cache_spec, AttentionSpec)", window)

    def test_loaded_state_blocks_are_not_zero_candidates(self):
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager  # noqa: F401

        pool = _stub_worker()
        request = _hybrid_request()
        result = pool._holemiss_run_gdn_state_ladder(request)
        # state_load_idx lives in the mamba groups' block space; the zeroer's
        # candidates come from AttentionSpec managers only (the pin above) —
        # disjoint by construction. The ladder's record is what T15's content
        # discriminator asserts against.
        self.assertTrue(result.state_load_block_ids)
        self.assertTrue(all(block_id >= 1000 for block_id in result.state_load_block_ids))


class TestG3HarnessDisjointness(unittest.TestCase):
    """V-T14.3 (seam map G3 deliverable (iii)): on a REAL hybrid schedule,
    the zeroer's candidate set and the ladder's state_load_idx are disjoint.
    """

    def test_real_schedule_zero_candidates_disjoint_from_state_load_idx(self):
        import torch
        from vllm import SamplingParams
        from vllm.config import CUDAGraphMode
        from vllm.utils.hashing import sha256
        from vllm.v1.core.kv_cache_utils import (
            get_request_block_hasher,
            init_none_hash,
            resolve_kv_cache_block_sizes,
        )
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
        from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
        from vllm.v1.request import Request
        from vllm.v1.structured_output import StructuredOutputManager

        from tests.ut.kv_offload.utils import create_vllm_config

        full = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32)
        mamba = MambaSpec(
            block_size=32,
            shapes=((8,), (8,)),
            dtypes=(torch.float32, torch.float32),
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            mamba_cache_mode="align",
        )
        plan = KVCacheConfig(
            num_blocks=64,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["model.layers.0.attn", "model.layers.2.attn"], full),
                KVCacheGroupSpec(["model.layers.1.attn", "model.layers.3.attn"], mamba),
            ],
        )
        self.assertTrue(plan.needs_kv_cache_zeroing, "the mamba group must force zeroing on")
        config = create_vllm_config(max_num_batched_tokens=256, block_size=16)
        config.model_config.hf_text_config.num_hidden_layers = 4
        config.model_config.model_arch_config.total_num_hidden_layers = 4
        config.model_config.max_model_len = 512
        config.scheduler_config.max_model_len = 512
        config.scheduler_config.disable_hybrid_kv_cache_manager = False
        config.cache_config.mamba_cache_mode = "align"
        config.cache_config.num_gpu_blocks = 64
        config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        # The disjointness is manager-side (allocation + new-block recording)
        # plus the ladder's record; no connector is needed.
        config.kv_transfer_config = None
        block_size, hash_size = resolve_kv_cache_block_sizes(plan, config)
        register_all_kvcache_specs(None)
        init_none_hash(sha256)
        scheduler = Scheduler(
            config,
            plan,
            StructuredOutputManager(config),
            block_size=block_size,
            hash_block_size=hash_size,
        )
        request = Request(
            request_id="req-G3",
            prompt_token_ids=list(range(128)),
            sampling_params=SamplingParams(max_tokens=2),
            pooling_params=None,
            block_hasher=get_request_block_hasher(hash_size, sha256),
        )
        scheduler.add_request(request)
        output = scheduler.schedule()
        zero_candidates = set(output.new_block_ids_to_zero or [])
        self.assertTrue(zero_candidates, "the FA manager must record new blocks under hybrid zeroing")

        # The ladder's state_load_idx for the same request: run the real
        # ladder over a ReqMeta built from the scheduler's own allocation.
        group_blocks = scheduler.kv_cache_manager.get_blocks("req-G3").get_block_ids()
        pool = _stub_worker(num_state_groups=1)
        pool.grouped_block_size = [16, 32]
        pool.hash_block_size = hash_size
        pool.group_block_len = {1: [OBJECT_BYTES]}
        pool.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=full, layer_names=["model.layers.0.attn"]),
                SimpleNamespace(kv_cache_spec=mamba, layer_names=["model.layers.1.attn"]),
            ]
        )
        ladder_request = ReqMeta(
            req_id="req-G3",
            block_ids_by_group=[list(group_blocks[0]), list(group_blocks[1])],
            block_hashes=[
                maybe_hash.hexdigest() if hasattr(maybe_hash, "hexdigest") else str(maybe_hash)
                for maybe_hash in request.block_hashes
            ],
            can_save=False,
            load_spec=SimpleNamespace(
                can_load=True,
                kvpool_cached_tokens=4 * 32,
                kvpool_store_skip_tokens=None,
                vllm_cached_tokens=0,
            ),
        )
        result = pool._holemiss_run_gdn_state_ladder(ladder_request)
        self.assertEqual(result.rung, RUNG_LAST_SAVED)
        self.assertTrue(result.state_load_block_ids)
        self.assertEqual(zero_candidates & set(result.state_load_block_ids), set())


class _FakeAnchorStore:
    """Records the worker-issued session-start / batch_copy_get / lease calls."""

    def __init__(self, results):
        self.get_calls = []
        self.lease_releases = []
        self.session_starts = []
        self._results = list(results)

    def batch_get_start(self, keys):
        self.session_starts.append(list(keys))
        return [0] * len(keys)

    def batch_copy_get(self, keys, buffers, sizes, offsets):
        self.get_calls.append((list(keys), buffers, sizes, offsets))
        return list(self._results)

    def batch_remove_lease(self, keys):
        self.lease_releases.append(list(keys))


def _anchor_stub_worker():
    """A stub worker whose group builders run the REAL LayerBatchBuilder math."""
    pool = _stub_worker()
    pool.use_block_key_layerwise = True
    pool.num_kv_cache_groups = 4
    pool.group_num_layers = {0: 2, 1: 1, 2: 1, 3: 1}
    pool.group_block_len = {
        0: [OBJECT_BYTES],
        1: [1600, 2400],
        2: [1600, 2400],
        3: [1600, 2400],
    }
    pool.page_size_bytes = 4096
    pool.pp_size = 1
    pool.dcp_size = 1
    pool.dcp_rank = 0
    pool.put_step = 1
    pool.layerwise_key_layer_offset = 0
    pool.layerwise_max_transfer_blocks = 0
    pool._holemiss_req_meta_by_id = {}
    token_database = pool_worker.ChunkedTokenDatabase(metadata=[], block_size=[BLOCK] * 4, partitions=None)
    token_database.set_group_buffers(
        group_kv_caches_base_addr={group: [100000 + 1000 * group, 200000 + 1000 * group] for group in (0, 1, 2, 3)},
        group_block_len={group: [1600, 2400] for group in (0, 1, 2, 3)},
        group_block_stride={group: [8, 12] for group in (0, 1, 2, 3)},
        group_layer_cache_entry_offsets={group: [0, 2] for group in (0, 1, 2, 3)},
    )
    pool.token_database = token_database
    return pool


class TestGdnRepairAnchorFetch(unittest.TestCase):
    """D-MTX.4 Option 1: the rollback-boundary state GET into the repair slot.

    Pins the boundary-key derivation at floor(repair_start/1536) (an OLDER
    rung of the ladder plan's own key space), the landing into the request's
    OWN boundary slot (never a foreign slot), the per-layer range merge, the
    loud kind=zero fallback on ANY degrade, and the connector passthrough.
    """

    def _prime(self, pool, request):
        pool._holemiss_run_gdn_state_ladder(request)

    def test_boundary_key_derivation_and_landing(self):
        pool = _anchor_stub_worker()
        request = _hybrid_request()
        self._prime(pool, request)
        store = _FakeAnchorStore(results=[0, 0, 0])
        pool.m_store = store

        mapping = pool._holemiss_fetch_gdn_repair_anchor("req-H", 2 * BLOCK)

        # block_index = boundary // 1536 - 1 = 1: the OLDER rung's key per
        # group, landing in the request's own boundary slot ids[1].
        self.assertEqual(mapping, {1: 1001, 2: 2001, 3: 3001})
        self.assertEqual(len(store.get_calls), 1)
        keys, buffers, sizes, offsets = store.get_calls[0]
        self.assertEqual(
            keys,
            [f"m@layout@group:{group}@block:1536@{1:064x}@0" for group in GROUPS],
        )
        for group, key in zip(GROUPS, keys, strict=True):
            index = keys.index(key)
            slot = 1000 * group + 1
            self.assertEqual(
                buffers[index],
                [100000 + 1000 * group + slot * 8, 200000 + 1000 * group + slot * 12],
            )
            self.assertEqual(sizes[index], [1600, 2400])
            self.assertEqual(offsets[index], [0, 1600])
        self.assertEqual(store.lease_releases, [keys])
        markers = pool._holemiss_state_markers["req-H"]
        self.assertTrue(
            any(
                "HOLEMISS_GDN_REPAIR_ANCHOR" in line
                and "kind=boundary" in line
                and "boundary=3072" in line
                and "slots=g1:1001+g2:2001+g3:3001" in line
                for line in markers
            ),
            markers,
        )

    def test_get_failure_degrades_loudly_to_zero(self):
        pool = _anchor_stub_worker()
        self._prime(pool, _hybrid_request())
        store = _FakeAnchorStore(results=[0, -1102, 0])
        pool.m_store = store

        mapping = pool._holemiss_fetch_gdn_repair_anchor("req-H", 2 * BLOCK)

        self.assertIsNone(mapping)
        markers = pool._holemiss_state_markers["req-H"]
        self.assertTrue(
            any("kind=zero fallback" in line and "reason=get_failed rc=-1102" in line for line in markers),
            markers,
        )
        # All-or-nothing: the lease release still ran (B35), but no partial
        # anchor escapes — a mixed boundary/zero anchor across GDN layers
        # would be a silent wrong-state.
        self.assertEqual(store.lease_releases and len(store.lease_releases[0]), 3)

    def test_missing_req_meta_degrades_loudly(self):
        pool = _anchor_stub_worker()
        store = _FakeAnchorStore(results=[0])
        pool.m_store = store

        self.assertIsNone(pool._holemiss_fetch_gdn_repair_anchor("req-H", 2 * BLOCK))
        markers = pool._holemiss_state_markers["req-H"]
        self.assertTrue(any("reason=no_req_meta" in line for line in markers), markers)
        self.assertEqual(store.get_calls, [])

    def test_zero_boundary_is_the_exact_zero_anchor(self):
        pool = _anchor_stub_worker()
        self._prime(pool, _hybrid_request())
        store = _FakeAnchorStore(results=[0])
        pool.m_store = store

        self.assertIsNone(pool._holemiss_fetch_gdn_repair_anchor("req-H", 0))
        markers = pool._holemiss_state_markers["req-H"]
        # state@0 IS the zero state: no fallback token on this line.
        self.assertTrue(
            any("kind=zero" in line and "reason=repair_start_zero" in line for line in markers),
            markers,
        )
        self.assertFalse(any("fallback" in line for line in markers), markers)
        self.assertEqual(store.get_calls, [])

    def test_unaligned_boundary_refuses_never_guesses(self):
        pool = _anchor_stub_worker()
        self._prime(pool, _hybrid_request())
        store = _FakeAnchorStore(results=[0])
        pool.m_store = store

        self.assertIsNone(pool._holemiss_fetch_gdn_repair_anchor("req-H", BLOCK + 1))
        markers = pool._holemiss_state_markers["req-H"]
        self.assertTrue(any("reason=unaligned_boundary" in line for line in markers), markers)
        self.assertEqual(store.get_calls, [])

    def test_boundary_out_of_range_degrades_loudly(self):
        pool = _anchor_stub_worker()
        request = _hybrid_request(num_blocks=1)
        self._prime(pool, request)
        store = _FakeAnchorStore(results=[0])
        pool.m_store = store

        # boundary 2*1536 needs block_index 1; the request only has block 0.
        self.assertIsNone(pool._holemiss_fetch_gdn_repair_anchor("req-H", 2 * BLOCK))
        markers = pool._holemiss_state_markers["req-H"]
        self.assertTrue(any("reason=boundary_out_of_range" in line for line in markers), markers)
        self.assertEqual(store.get_calls, [])

    def test_req_meta_recorded_by_ladder_and_purged_at_finish(self):
        pool = _anchor_stub_worker()
        request = _hybrid_request()
        self.assertNotIn("req-H", pool._holemiss_req_meta_by_id)
        self._prime(pool, request)
        self.assertIs(pool._holemiss_req_meta_by_id["req-H"], request)
        pool._holemiss_teardown_finished({"req-H"})
        self.assertNotIn("req-H", pool._holemiss_req_meta_by_id)

    def test_connector_passthrough_and_no_worker_loud(self):
        pool = _anchor_stub_worker()
        self._prime(pool, _hybrid_request())
        pool.m_store = _FakeAnchorStore(results=[0, 0, 0])
        connector = AscendStoreConnector.__new__(AscendStoreConnector)
        connector.connector_worker = pool
        self.assertEqual(
            connector.fetch_gdn_repair_anchor("req-H", 2 * BLOCK),
            {1: 1001, 2: 2001, 3: 3001},
        )
        connector.connector_worker = None
        with self.assertLogs("vllm", level="INFO") as logs:
            self.assertIsNone(connector.fetch_gdn_repair_anchor("req-H", 2 * BLOCK))
        self.assertTrue(
            any("kind=zero fallback" in line and "reason=no_worker_api" in line for line in logs.output),
            logs.output,
        )


if __name__ == "__main__":
    unittest.main()
