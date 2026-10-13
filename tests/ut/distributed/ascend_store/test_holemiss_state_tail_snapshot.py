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
"""hybrid-gdn-tail-state T2: the GDN tail-state snapshot freeze.

The populate request's own 1-token decode advances the tail mamba slot in
place past the boundary state before the finish-time drain reads it, so the
state-tail artifact must be a D2D snapshot FROZEN at the last prefill step's
final layer. These tests pin the freeze semantics (mutate the live slot
post-capture -> the snapshot bytes stay frozen), the byte-equality law (the
clones concatenate to exactly the ranged read's build_addrs extents), the
lifecycle (the double-capture fence, the take-consume, the teardown purge),
the trigger conditions (the prompt-end/unaligned/state-groups gates, the
non-finishing-step negative control, the text inertness, the activation
gating), the loud degrade, and both trigger-site wirings.

The fixtures model the LIVE geometry (the N18/N19 anti-fake law): 16 FA
layers at physical 3+4k with kernel-gran caches (12 rows per pool page), 48
GDN layers in 3 round-robin groups of 16, the D16 conv+ssm tensor aliasing
(one SHARED (conv, ssm) pair per {4k, 4k+1, 4k+2} family — the groups alias
ACROSS, never within), the mixed bf16-conv/fp32-ssm tuple, MRV2 page padding
on the ssm tensor, the 1536 block space, and the group metadata derived by
the REAL _infer_cache_group_metadata (never hand-mirrored).
"""

import ctypes
import hashlib
import inspect
import threading
import unittest
from contextlib import contextmanager
from math import prod
from types import SimpleNamespace

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)

NUM_LAYERS = 64
BLOCK = 1536
# The proof geometry: the populate prompt 9472 = 6*1536 + 256 (unaligned vs
# the group block, aligned vs the pmu 256); the tail state lives in the
# request's block 6 -> mamba slot block_ids_by_group[g][6].
BOUNDARY = 9472
TAIL_SLOT_INDEX = 6
ALIGNED_BOUNDARY = 6 * BLOCK
MID_PROMPT_UNALIGNED_END = 5 * BLOCK + 256
# The pool row space: 32 slots; the three state groups allocate DISTINCT
# rows of the SHARED D16 family tensors (the per-group block tables).
NUM_SLOTS = 32

FA_PHYSICAL = [3 + 4 * k for k in range(16)]
GDN_PHYSICAL = [p for p in range(NUM_LAYERS) if p not in FA_PHYSICAL]
STATE_GROUPS = (1, 2, 3)
GROUP_TAIL_ROWS = {1: 7, 2: 14, 3: 22}

# LIVE per-rank GDN state geometry (Qwen3.6-27B @ TP=2, re-derived from the
# model config + MambaStateShapeCalculator.gated_delta_net_state_shape):
# conv_dim = 128*16*2 + 128*48 = 10240 -> (5120, 3) bf16 per rank; ssm
# (24, 128, 128) fp32 per rank; 1603584 B/layer; 16 layers per group ->
# 25657344 B/group (the T13-measured GDN state object size). The fixtures
# below scale the HEAD COUNTS (the live volume is ~13 GB of device memory)
# while keeping every structural law; test_live_object_arithmetic pins the
# unscaled byte arithmetic.
LIVE_CONV_SHAPE = (5120, 3)
LIVE_SSM_SHAPE = (24, 128, 128)
GDN_OBJECT_BYTES_LIVE = 25657344

CONV_SHAPE = (640, 3)
SSM_SHAPE = (3, 128, 128)
CONV_DTYPE = torch.bfloat16
SSM_DTYPE = torch.float32
# MRV2 page padding: the ssm slots are views into one block-major buffer
# whose per-slot span exceeds the state (the live binding law; the padding
# must never leak into the artifact).
SSM_SLOT_PAD_ELEMS = prod(SSM_SHAPE) + 512

FA_KERNEL_ROWS_PER_PAGE = 12
FA_KERNEL_SHAPE = (128, 2, 8)

HOLEMISS_ENVS = ("HOLEMISS_RUNTIME_PROFILE", "HOLEMISS_DIAG")

SNAPSHOT_RE_FIELDS = (
    r"HOLEMISS_STATE_TAIL_SNAPSHOT tp=(\d+) req=(\S+) group=(\d+) "
    r"boundary=(\d+) slot=(\d+) bytes=(\d+) sha=([0-9a-f]{64})"
)


@contextmanager
def _profile(profile: str | None, diag: str | None = None):
    import os

    saved = {name: os.environ.get(name) for name in HOLEMISS_ENVS}
    try:
        if profile is None:
            os.environ.pop("HOLEMISS_RUNTIME_PROFILE", None)
        else:
            os.environ["HOLEMISS_RUNTIME_PROFILE"] = profile
        if diag is None:
            os.environ.pop("HOLEMISS_DIAG", None)
        else:
            os.environ["HOLEMISS_DIAG"] = diag
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _conv_cache(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn((NUM_SLOTS, *CONV_SHAPE), generator=gen, dtype=CONV_DTYPE)


def _ssm_cache(seed: int) -> torch.Tensor:
    """The padded block-major ssm binding (the MRV2 page-padding law)."""
    flat = torch.randn(NUM_SLOTS * SSM_SLOT_PAD_ELEMS, generator=torch.Generator().manual_seed(seed), dtype=SSM_DTYPE)
    inner = tuple(SSM_SHAPE)
    strides = (SSM_SLOT_PAD_ELEMS,) + tuple(prod(inner[dim + 1 :]) for dim in range(len(inner)))
    return flat.as_strided((NUM_SLOTS, *inner), strides)


def _fa_cache(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(
        (NUM_SLOTS * FA_KERNEL_ROWS_PER_PAGE, *FA_KERNEL_SHAPE),
        generator=gen,
        dtype=torch.bfloat16,
    )


class _Geometry:
    """The live hybrid layout: the D16 shared families + the FA group."""

    def __init__(self):
        self.kv_caches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.fa_caches = (_fa_cache(71), _fa_cache(72))
        for phys in FA_PHYSICAL:
            self.kv_caches[f"model.layers.{phys}.self_attn"] = self.fa_caches
        # D16: family k = {4k, 4k+1, 4k+2} shares ONE (conv, ssm) pair; the
        # three state groups are round-robin over the families (group 1 =
        # {4k}, group 2 = {4k+1}, group 3 = {4k+2}) so the sharing is
        # across groups and never within one.
        self.family_caches = [(_conv_cache(100 + k), _ssm_cache(200 + k)) for k in range(16)]
        for phys in GDN_PHYSICAL:
            self.kv_caches[f"model.layers.{phys}.linear_attn"] = self.family_caches[phys // 4]

    def group_layer_names(self, group_id: int) -> list[str]:
        if group_id == 0:
            return [f"model.layers.{p}.self_attn" for p in sorted(FA_PHYSICAL)]
        return [f"model.layers.{p}.linear_attn" for p in sorted(GDN_PHYSICAL) if p % 4 == group_id - 1]

    def group_spec(self, group_id: int):
        if group_id == 0:
            return FullAttentionSpec(block_size=BLOCK, num_kv_heads=2, head_size=8, dtype=torch.bfloat16)
        return MambaSpec(
            block_size=BLOCK,
            shapes=(CONV_SHAPE, SSM_SHAPE),
            dtypes=(CONV_DTYPE, SSM_DTYPE),
            mamba_cache_mode="align",
        )


def _make_worker(kv_role="kv_producer", consumer_is_to_put=False, use_hybrid=True):
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.tp_rank = 0
    worker.num_layers = NUM_LAYERS
    worker.hf_config = SimpleNamespace(num_hidden_layers=NUM_LAYERS)
    worker.use_hybrid = use_hybrid
    worker.use_layerwise = True
    worker.use_block_key_layerwise = True
    worker.block_key_hybrid = use_hybrid
    worker.kv_role = kv_role
    worker.consumer_is_to_put = consumer_is_to_put
    worker.grouped_block_size = [BLOCK, BLOCK, BLOCK, BLOCK] if use_hybrid else [BLOCK]
    worker.hash_block_size = 256
    worker.num_blocks = NUM_SLOTS
    worker._global_to_local_layer = {}
    worker._holemiss_state_tail_snapshots = {}
    worker._holemiss_state_markers = {}
    worker._holemiss_state_load_results = {}
    worker._holemiss_failed_load_keys = {}
    worker._holemiss_req_meta_by_id = {}
    worker._holemiss_scratch_owner = None
    worker._holemiss_step_requests = []
    geometry = _Geometry()
    worker.kv_caches = geometry.kv_caches
    groups = [
        SimpleNamespace(layer_names=geometry.group_layer_names(g), kv_cache_spec=geometry.group_spec(g))
        for g in range(4 if use_hybrid else 1)
    ]
    worker.kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    worker.group_kv_caches_base_addr = {}
    worker.group_block_len = {}
    worker.group_block_stride = {}
    worker.group_layer_cache_entry_offsets = {}
    worker.group_num_layers = {}
    # Derive the addressing metadata through the REAL walk (never a
    # hand-mirrored copy — the N18 law).
    for group_id, group in enumerate(worker.kv_cache_config.kv_cache_groups):
        worker._infer_cache_group_metadata(group_id, group.layer_names)
    return worker, geometry


def _make_request(
    req_id="req-A",
    num_prompt=BOUNDARY,
    target=None,
    can_save=True,
    block_tables=None,
):
    if block_tables is None:
        block_tables = {
            0: [9, 10, 11, 12, 13, 14, 15],
            1: [1, 2, 3, 4, 5, 6, GROUP_TAIL_ROWS[1]],
            2: [8, 9, 10, 11, 12, 13, GROUP_TAIL_ROWS[2]],
            3: [16, 17, 18, 19, 20, 21, GROUP_TAIL_ROWS[3]],
        }
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=num_prompt,
        block_ids_by_group=[block_tables[g] for g in sorted(block_tables)],
        block_hashes=[bytes([(i + 1) % 256] * 32) for i in range((num_prompt + 255) // 256)],
        can_save=can_save,
        num_prompt_tokens=num_prompt,
        target_token_len=num_prompt if target is None else target,
    )


def _tensors_bytes(tensors) -> bytes:
    digest = bytearray()
    for tensor in tensors:
        digest.extend(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return bytes(digest)


def _group_expected_bytes(worker, group_id, pool_slot) -> bytes:
    """The build_addrs extents read from RAW memory — independent of the clone.

    build_addrs computes (layer_base_addrs + block_id * layer_block_stride,
    layer_block_len) per entry; this reads exactly those linear byte ranges
    with ctypes from the registered tensor storage.
    """
    payload = bytearray()
    for base, length, stride in zip(
        worker.group_kv_caches_base_addr[group_id],
        worker.group_block_len[group_id],
        worker.group_block_stride[group_id],
    ):
        start = base + pool_slot * stride
        payload.extend((ctypes.c_char * length).from_address(start))
    return bytes(payload)


class _StubEvent:
    def record(self):
        pass


class _StubSendThread:
    def __init__(self):
        self.request_queue = SimpleNamespace(all_tasks_done=threading.Condition(), unfinished_tasks=0)

    def raise_if_failed(self):
        pass


class TestLiveGeometryArithmetic(unittest.TestCase):
    def test_live_object_arithmetic(self):
        """The unscaled live shapes reproduce the T13 object size exactly."""
        conv = prod(LIVE_CONV_SHAPE) * 2
        ssm = prod(LIVE_SSM_SHAPE) * 4
        self.assertEqual((conv + ssm) * 16, GDN_OBJECT_BYTES_LIVE)

    def test_fixture_group_metadata_matches_the_live_laws(self):
        worker, _ = _make_worker()
        self.assertEqual(worker._holemiss_state_groups(), STATE_GROUPS)
        for group_id in STATE_GROUPS:
            self.assertEqual(worker.group_num_layers[group_id], 16)
            self.assertEqual(len(worker.group_block_len[group_id]), 32)
            self.assertEqual(sum(worker.group_block_len[group_id]), 16 * (prod(CONV_SHAPE) * 2 + prod(SSM_SHAPE) * 4))
        # The D16 aliasing: the three groups' entry addresses are the SAME
        # 16 tensor pairs (round-robin), each group 32 entries.
        self.assertEqual(
            [tuple(worker.group_kv_caches_base_addr[g]) for g in STATE_GROUPS],
            [tuple(worker.group_kv_caches_base_addr[g]) for g in STATE_GROUPS],
        )
        self.assertEqual(worker.group_kv_caches_base_addr[1], worker.group_kv_caches_base_addr[2])
        # The GDN caches are slot-gran (scale 1: one row per pool block) —
        # never the FA group's 12-row kernel pages.
        conv = worker.kv_caches["model.layers.0.linear_attn"][0]
        self.assertEqual(conv.shape[0], NUM_SLOTS)
        fa = worker.kv_caches["model.layers.3.self_attn"][0]
        self.assertEqual(fa.shape[0], NUM_SLOTS * FA_KERNEL_ROWS_PER_PAGE)
        # The MRV2 padding: the ssm slot stride exceeds the state bytes.
        ssm = worker.kv_caches["model.layers.0.linear_attn"][1]
        self.assertGreater(ssm.stride(0) * ssm.element_size(), prod(SSM_SHAPE) * 4)


class TestFreezeSemantics(unittest.TestCase):
    def test_capture_freezes_the_tail_slot_bytes(self):
        with _profile("evidence"):
            worker, _ = _make_worker()
            request = _make_request()
            worker._holemiss_capture_state_tail_snapshots([request])
            entry = worker._holemiss_state_tail_snapshots[("req-A", BOUNDARY)]
            frozen = {group: _tensors_bytes(tensors) for group, tensors in entry.items()}
            self.assertTrue(all(frozen.values()))
            # The decode advance: every family's tail rows (all three
            # groups' block-6 slots of the SHARED tensors) are rewritten in
            # place AFTER the capture — the snapshots must stay frozen.
            for phys in GDN_PHYSICAL:
                conv, ssm = worker.kv_caches[f"model.layers.{phys}.linear_attn"]
                for row in GROUP_TAIL_ROWS.values():
                    conv[row] = torch.full_like(conv[row], 7.0)
                    ssm[row] = torch.full_like(ssm[row], 9.0)
            entry_after = worker._holemiss_state_tail_snapshots[("req-A", BOUNDARY)]
            for group, tensors in entry_after.items():
                self.assertEqual(_tensors_bytes(tensors), frozen[group])

    def test_snapshot_bytes_equal_the_build_addrs_extents(self):
        worker, _ = _make_worker()
        request = _make_request()
        worker._holemiss_capture_state_tail_snapshots([request])
        entry = worker._holemiss_state_tail_snapshots[("req-A", BOUNDARY)]
        for group_id, tensors in entry.items():
            pool_slot = request.block_ids_by_group[group_id][TAIL_SLOT_INDEX]
            self.assertEqual(_tensors_bytes(tensors), _group_expected_bytes(worker, group_id, pool_slot))

    def test_padding_never_leaks_into_the_artifact(self):
        worker, geometry = _make_worker()
        request = _make_request()
        worker._holemiss_capture_state_tail_snapshots([request])
        entry = worker._holemiss_state_tail_snapshots[("req-A", BOUNDARY)]
        conv_len = prod(CONV_SHAPE) * 2
        ssm_len = prod(SSM_SHAPE) * 4
        for group_id, tensors in entry.items():
            self.assertEqual(tensors[0].numel() * tensors[0].element_size(), conv_len)
            self.assertEqual(tensors[1].numel() * tensors[1].element_size(), ssm_len)
            total = sum(t.numel() * t.element_size() for t in tensors)
            self.assertEqual(total, 16 * (conv_len + ssm_len))

    def test_three_groups_capture_their_own_rows(self):
        """The D16 pin: the groups share tensors but hold DISTINCT rows."""
        worker, geometry = _make_worker()
        request = _make_request()
        for family, (conv, ssm) in enumerate(geometry.family_caches):
            for group_id in STATE_GROUPS:
                row = request.block_ids_by_group[group_id][TAIL_SLOT_INDEX]
                conv[row] = torch.full_like(conv[row], float(family * 10 + group_id))
                ssm[row] = torch.full_like(ssm[row], float(family * 10 + group_id))
        worker._holemiss_capture_state_tail_snapshots([request])
        entry = worker._holemiss_state_tail_snapshots[("req-A", BOUNDARY)]
        digests = {group: hashlib.sha256(_tensors_bytes(tensors)).hexdigest() for group, tensors in entry.items()}
        self.assertEqual(len(set(digests.values())), len(STATE_GROUPS))
        for group_id, tensors in entry.items():
            pool_slot = request.block_ids_by_group[group_id][TAIL_SLOT_INDEX]
            self.assertEqual(_tensors_bytes(tensors), _group_expected_bytes(worker, group_id, pool_slot))


class TestLifecycle(unittest.TestCase):
    def test_double_capture_fence_keeps_the_first_freeze(self):
        with _profile("evidence"):
            worker, _ = _make_worker()
            request = _make_request()
            worker._holemiss_capture_state_tail_snapshots([request])
            first = {
                group: _tensors_bytes(tensors)
                for group, tensors in worker._holemiss_state_tail_snapshots[("req-A", BOUNDARY)].items()
            }
            conv, ssm = worker.kv_caches["model.layers.24.linear_attn"]
            for row in GROUP_TAIL_ROWS.values():
                conv[row] = torch.full_like(conv[row], 3.0)
                ssm[row] = torch.full_like(ssm[row], 3.0)
            worker._holemiss_capture_state_tail_snapshots([request])
            self.assertEqual(list(worker._holemiss_state_tail_snapshots), [("req-A", BOUNDARY)])
            for group, tensors in worker._holemiss_state_tail_snapshots[("req-A", BOUNDARY)].items():
                self.assertEqual(_tensors_bytes(tensors), first[group])

    def test_take_pops_the_registry_and_returns_the_frozen_bytes(self):
        worker, _ = _make_worker()
        request = _make_request()
        worker._holemiss_capture_state_tail_snapshots([request])
        frozen = {
            group: _tensors_bytes(tensors)
            for group, tensors in worker._holemiss_state_tail_snapshots[("req-A", BOUNDARY)].items()
        }
        taken = worker._holemiss_take_state_tail_snapshots("req-A", BOUNDARY)
        self.assertIsNotNone(taken)
        self.assertEqual(set(taken), set(STATE_GROUPS))
        for group, tensors in taken.items():
            self.assertEqual(_tensors_bytes(tensors), frozen[group])
        self.assertNotIn(("req-A", BOUNDARY), worker._holemiss_state_tail_snapshots)
        self.assertIsNone(worker._holemiss_take_state_tail_snapshots("req-A", BOUNDARY))

    def test_teardown_purge_frees_every_boundary_of_the_request(self):
        worker, _ = _make_worker()
        worker._hole_recovery_regions = []
        worker._hole_recovery_lock = threading.Lock()
        worker._holemiss_armed_fault = None
        request = _make_request()
        worker._holemiss_capture_state_tail_snapshots([request])
        # A second boundary entry for the same request (the purge is by
        # req_id, not by boundary).
        worker._holemiss_state_tail_snapshots[("req-A", ALIGNED_BOUNDARY)] = {}
        other = _make_request(req_id="req-B", num_prompt=BOUNDARY + 256)
        worker._holemiss_capture_state_tail_snapshots([other])
        worker._holemiss_teardown_finished({"req-A"})
        self.assertNotIn(("req-A", BOUNDARY), worker._holemiss_state_tail_snapshots)
        self.assertNotIn(("req-A", ALIGNED_BOUNDARY), worker._holemiss_state_tail_snapshots)
        self.assertIn(("req-B", BOUNDARY + 256), worker._holemiss_state_tail_snapshots)


class TestTriggerConditions(unittest.TestCase):
    def test_non_finishing_step_negative_control(self):
        worker, _ = _make_worker()
        mid_prompt = _make_request(target=MID_PROMPT_UNALIGNED_END)
        decode_step = _make_request(target=BOUNDARY + 1)
        worker._holemiss_capture_state_tail_snapshots([mid_prompt, decode_step])
        self.assertEqual(worker._holemiss_state_tail_snapshots, {})

    def test_missing_num_prompt_tokens_never_qualifies(self):
        worker, _ = _make_worker()
        request = _make_request()
        request.num_prompt_tokens = None
        worker._holemiss_capture_state_tail_snapshots([request])
        self.assertEqual(worker._holemiss_state_tail_snapshots, {})

    def test_aligned_prompt_end_captures_nothing(self):
        worker, _ = _make_worker()
        worker._holemiss_capture_state_tail_snapshots([_make_request(num_prompt=ALIGNED_BOUNDARY)])
        self.assertEqual(worker._holemiss_state_tail_snapshots, {})

    def test_can_save_false_still_captures(self):
        """Section b.2: can_save is NOT part of the condition."""
        worker, _ = _make_worker()
        worker._holemiss_capture_state_tail_snapshots([_make_request(can_save=False)])
        self.assertIn(("req-A", BOUNDARY), worker._holemiss_state_tail_snapshots)

    def test_text_single_group_layout_is_inert(self):
        worker, _ = _make_worker(use_hybrid=False)
        worker._holemiss_capture_state_tail_snapshots([_make_request()])
        self.assertEqual(worker._holemiss_state_tail_snapshots, {})
        self.assertEqual(worker._holemiss_state_markers, {})


class TestActivationGating(unittest.TestCase):
    def test_plain_consumer_captures_nothing(self):
        with _profile("perf"):
            worker, _ = _make_worker(kv_role="kv_consumer", consumer_is_to_put=False)
            worker._holemiss_capture_state_tail_snapshots([_make_request()])
            self.assertEqual(worker._holemiss_state_tail_snapshots, {})

    def test_producer_plain_profile_captures_without_the_marker(self):
        with _profile("perf"):
            worker, _ = _make_worker(kv_role="kv_producer")
            worker._holemiss_capture_state_tail_snapshots([_make_request()])
            self.assertIn(("req-A", BOUNDARY), worker._holemiss_state_tail_snapshots)
            self.assertEqual(worker._holemiss_state_markers, {})

    def test_evidence_profile_emits_the_marker_verbatim(self):
        with _profile("evidence"):
            worker, _ = _make_worker(kv_role="kv_producer")
            request = _make_request()
            worker._holemiss_capture_state_tail_snapshots([request])
            lines = worker._holemiss_state_markers["req-A"]
            self.assertEqual(len(lines), len(STATE_GROUPS))
            group_bytes = 16 * (prod(CONV_SHAPE) * 2 + prod(SSM_SHAPE) * 4)
            for group_id, line in zip(STATE_GROUPS, lines):
                self.assertRegex(line, SNAPSHOT_RE_FIELDS)
                self.assertIn(f"tp=0 req=req-A group={group_id} boundary={BOUNDARY}", line)
                self.assertIn(f"slot={request.block_ids_by_group[group_id][TAIL_SLOT_INDEX]}", line)
                self.assertIn(f"bytes={group_bytes}", line)
            entry = worker._holemiss_state_tail_snapshots[("req-A", BOUNDARY)]
            for group_id, line in zip(STATE_GROUPS, lines):
                expected_sha = hashlib.sha256(_tensors_bytes(entry[group_id])).hexdigest()
                self.assertIn(f"sha={expected_sha}", line)

    def test_diag_flag_also_emits_the_marker(self):
        with _profile("perf", diag="1"):
            worker, _ = _make_worker(kv_role="kv_producer")
            worker._holemiss_capture_state_tail_snapshots([_make_request()])
            self.assertEqual(len(worker._holemiss_state_markers.get("req-A", [])), len(STATE_GROUPS))

    def test_evidence_consumer_probe_leg_captures(self):
        """Section b.5's probe leg: the T2' consumer under evidence."""
        with _profile("evidence"):
            worker, _ = _make_worker(kv_role="kv_consumer", consumer_is_to_put=False)
            worker._holemiss_capture_state_tail_snapshots([_make_request()])
            self.assertIn(("req-A", BOUNDARY), worker._holemiss_state_tail_snapshots)


class TestLoudDegrade(unittest.TestCase):
    def test_block_table_shortfall_fails_loudly_without_raising(self):
        worker, _ = _make_worker()
        tables = {0: [9, 10, 11, 12, 13, 14, 15], 1: [1] * 7, 2: [8] * 6, 3: [16] * 7}
        request = _make_request(block_tables=tables)
        from vllm.logger import logger as vllm_logger

        with self.assertLogs(vllm_logger, level="WARNING") as logs:
            worker._holemiss_capture_state_tail_snapshots([request])
        self.assertEqual(worker._holemiss_state_tail_snapshots, {})
        failed = [line for line in logs.output if "HOLEMISS_STATE_TAIL_SNAPSHOT_FAILED" in line]
        self.assertEqual(len(failed), 1)
        self.assertIn("tp=0 req=req-A group=2", failed[0])
        self.assertIn(f"boundary={BOUNDARY}", failed[0])
        self.assertIn("reason=IndexError:", failed[0])

    def test_missing_cache_entry_fails_loudly_without_raising(self):
        worker, _ = _make_worker()
        del worker.kv_caches["model.layers.4.linear_attn"]
        request = _make_request()
        from vllm.logger import logger as vllm_logger

        with self.assertLogs(vllm_logger, level="WARNING") as logs:
            worker._holemiss_capture_state_tail_snapshots([request])
        self.assertEqual(worker._holemiss_state_tail_snapshots, {})
        self.assertTrue(any("HOLEMISS_STATE_TAIL_SNAPSHOT_FAILED" in line for line in logs.output))

    def test_out_of_range_slot_fails_loudly(self):
        worker, _ = _make_worker()
        tables = {0: [9, 10, 11, 12, 13, 14, 15], 1: [1] * 7, 2: [8] * 7, 3: [16, 17, 18, 19, 20, 21, NUM_SLOTS + 99]}
        request = _make_request(block_tables=tables)
        from vllm.logger import logger as vllm_logger

        with self.assertLogs(vllm_logger, level="WARNING") as logs:
            worker._holemiss_capture_state_tail_snapshots([request])
        self.assertEqual(worker._holemiss_state_tail_snapshots, {})
        self.assertTrue(any("HOLEMISS_STATE_TAIL_SNAPSHOT_FAILED" in line for line in logs.output))


class TestTriggerSiteWiring(unittest.TestCase):
    """The behavioral pins: the capture fires at the FINAL layer, once."""

    def _step_worker(self, kv_role):
        worker, _ = _make_worker(kv_role=kv_role)
        worker.pp_size = 1
        worker.current_layer = 0
        worker.sync_save_events = [_StubEvent() for _ in range(NUM_LAYERS)]
        worker.layer_save_finished_events = [threading.Event() for _ in range(NUM_LAYERS)]
        worker.layer_save_tasks = [[] for _ in range(NUM_LAYERS)]
        worker._attention_saved_layers = set()
        send_thread = _StubSendThread()
        worker.kv_send_thread = send_thread
        worker.layerwise_protocol = SimpleNamespace(
            fence_drains_recv=lambda: False,
            send_fence_backlog=lambda: 0,
        )
        worker.prefetch_layer_map = {}
        worker._recurrent_layers = set()
        return worker

    def test_save_kv_layer_final_layer_branch_fires_the_capture_once(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
            AscendConnectorMetadata,
        )

        with _profile("evidence"):
            worker = self._step_worker("kv_producer")
            metadata = AscendConnectorMetadata(preempted_req_ids=set())
            metadata.add_request(_make_request())
            for _ in range(NUM_LAYERS):
                worker.save_kv_layer(metadata)
            self.assertEqual(worker.current_layer, NUM_LAYERS)
            self.assertIn(("req-A", BOUNDARY), worker._holemiss_state_tail_snapshots)
            self.assertEqual(len(worker._holemiss_state_markers["req-A"]), len(STATE_GROUPS))

    def test_save_kv_layer_early_layers_capture_nothing(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
            AscendConnectorMetadata,
        )

        with _profile("evidence"):
            worker = self._step_worker("kv_producer")
            metadata = AscendConnectorMetadata(preempted_req_ids=set())
            metadata.add_request(_make_request())
            for _ in range(NUM_LAYERS - 1):
                worker.save_kv_layer(metadata)
            self.assertEqual(worker._holemiss_state_tail_snapshots, {})
            self.assertEqual(worker._holemiss_state_markers, {})

    def _load_leg_worker(self, kv_role):
        worker = self._step_worker(kv_role)
        if kv_role == "kv_consumer":
            worker.consumer_is_to_put = False
        worker.layer_load_finished_events = [threading.Event() for _ in range(NUM_LAYERS)]
        worker.layer_load_tasks = [[] for _ in range(NUM_LAYERS)]
        worker.num_prefetch_layers = 1
        worker.next_layer_to_submit = NUM_LAYERS + 1
        worker.external_slot_release_waiter = None
        worker._invalid_block_ids = set()
        worker._invalid_block_ids_lock = threading.Lock()
        worker._holemiss_state_dispositioned_blocks = set()
        worker._layer_load_aborted = threading.Event()
        worker._current_layerwise_request_ids = set()
        worker._current_layerwise_last_chunk_req_ids = set()
        worker.kv_recv_thread = SimpleNamespace(
            raise_if_failed=lambda: None,
            set_finished_request=lambda req_id: None,
        )
        worker._finish_current_layerwise_load_sessions = lambda: None
        worker.current_layer = NUM_LAYERS - 1
        worker._holemiss_step_requests = [_make_request()]
        return worker

    def test_wait_for_layer_load_consumer_leg_fires_the_capture(self):
        with _profile("evidence"):
            worker = self._load_leg_worker("kv_consumer")
            worker.wait_for_layer_load()
            self.assertIn(("req-A", BOUNDARY), worker._holemiss_state_tail_snapshots)
            self.assertEqual(len(worker._holemiss_state_markers["req-A"]), len(STATE_GROUPS))

    def test_wait_for_layer_load_producer_leg_does_not_double_capture(self):
        """The producer's site is save_kv_layer (design b.1): the load-side
        leg must stay silent for save roles even at the final layer."""
        with _profile("evidence"):
            worker = self._load_leg_worker("kv_producer")
            worker.wait_for_layer_load()
            self.assertEqual(worker._holemiss_state_tail_snapshots, {})


class TestSourcePins(unittest.TestCase):
    """G20-style text pins: the call sites are load-bearing (design b.1)."""

    def test_capture_precedes_the_final_layer_save_wait(self):
        source = inspect.getsource(KVPoolWorker.save_kv_layer)
        self.assertIn("_holemiss_capture_state_tail_snapshots(", source)
        branch = source[source.index("num_local - 1") :]
        self.assertLess(
            branch.index("_holemiss_capture_state_tail_snapshots("),
            branch.index("_wait_for_final_layer_save("),
        )

    def test_step_requests_stash_precedes_the_empty_metadata_return(self):
        source = inspect.getsource(KVPoolWorker.start_load_kv)
        self.assertIn("_holemiss_step_requests = metadata.requests", source)
        self.assertLess(
            source.index("_holemiss_step_requests = metadata.requests"),
            source.index("len(metadata.requests) == 0"),
        )

    def test_consumer_leg_lives_in_wait_for_layer_load(self):
        source = inspect.getsource(KVPoolWorker.wait_for_layer_load)
        self.assertIn("_holemiss_capture_state_tail_snapshots(", source)
        self.assertIn("not is_kv_save_role(", source)

    def test_teardown_purge_walks_the_snapshot_registry(self):
        source = inspect.getsource(KVPoolWorker._holemiss_teardown_finished)
        self.assertIn("_holemiss_state_tail_snapshots", source)


if __name__ == "__main__":
    unittest.main()
