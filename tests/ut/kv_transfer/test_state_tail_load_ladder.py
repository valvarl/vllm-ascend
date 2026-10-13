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
"""hybrid-gdn-tail-state T5: the state-tail load + the ladder tail rung.

The consumer-side mirror of the T3 save surface: the GDN groups 1-3 fetch
their state-tail artifact under the B41-mirror key (the SAME raw-chain
sub-hash + ``partial_tail_key_suffix`` — the mirror byte-equality, two
signal sources) into block ``boundary // block_size`` via the existing
group-view GET channel, and the ladder plan gains the interior tail
boundary as the NEWEST rung-0 candidate: a hit loads at rung 0 with
``source=tail`` and the per-group loaded-slot digest filling the STATE_LOAD
``sha`` field (LEG-1's consumer side); a miss demotes LOUDLY to the floor
rung with the ABSENT marker for the tail boundary (the marker mutex law).
The claimed==fetched invariant extends — the claim 9472 is covered by the
six full-block keys plus the state-tail key per group.

The fixtures model the LIVE geometry (the N18/N19 anti-fake law — the T2/T3
fixture line): 16 FA layers at physical 3+4k with kernel-gran caches, 48
GDN layers in 3 round-robin groups over the 16 shared D16 conv+ssm
families, the mixed bf16/fp32 tuple, MRV2 page padding, the 1536/256 block
space, and the group metadata derived by the REAL
``_infer_cache_group_metadata``.
"""

import hashlib
import inspect
import re
import threading
import unittest
from math import prod
from types import SimpleNamespace

import torch
from vllm.logger import logger as vllm_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

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
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.gdn_state_ladder import (
    RUNG_LAST_SAVED,
    RUNG_OLDER_SAVED,
    SOURCE_LAST_SAVED,
    SOURCE_OLDER_SAVED,
    SOURCE_TAIL,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    LoadSpec,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)

# The frozen v2 vocabulary regexes (holemiss-mooncake marker_vocab.py) —
# copied per the ladder-UT precedent so the contract test does not depend
# on the bench repo being importable.
STATE_LOAD_RE = re.compile(
    r"HOLEMISS_STATE_LOAD layer=(-?\d+) group=(-?\d+) boundary=(-?\d+) "
    r"slot=(-?\d+) source=(\S+) rung=(-?\d+) bytes=(-?\d+) sha=(\S+)"
)
STATE_ABSENT_RE = re.compile(
    r"HOLEMISS_STATE_ABSENT layer=(-?\d+) group=(-?\d+) boundary=(-?\d+) "
    r"reason=(\S+) rung=(-?\d+)"
)

NUM_LAYERS = 64
BLOCK = 1536
BOUNDARY = 9472  # 6*1536 + 256: unaligned vs the group block, pmu-aligned
FLOOR = 6 * BLOCK
TAIL_SLOT_INDEX = 6
NUM_SLOTS = 32
HASH_BLOCK_SIZE = 256
EXTENSION_PROMPT = BOUNDARY + HASH_BLOCK_SIZE

FA_PHYSICAL = [3 + 4 * k for k in range(16)]
GDN_PHYSICAL = [p for p in range(NUM_LAYERS) if p not in FA_PHYSICAL]
STATE_GROUPS = (1, 2, 3)
GROUP_TAIL_ROWS = {1: 7, 2: 14, 3: 22}

CONV_SHAPE = (640, 3)
SSM_SHAPE = (3, 128, 128)
CONV_DTYPE = torch.bfloat16
SSM_DTYPE = torch.float32
SSM_SLOT_PAD_ELEMS = prod(SSM_SHAPE) + 512

FA_KERNEL_ROWS_PER_PAGE = 12
FA_KERNEL_SHAPE = (128, 2, 8)

MODEL = "Qwen3.6-27B"
# The MEASURED hybrid layout namespace (T8 --hybrid-layout constant).
LAYOUT = "b5207e0c690b2031a57eff714ad1bd7ac1eb0fdde0f4461e0e67e7ff1247539d"

GROUP_OBJECT_BYTES = {group: 16 * (prod(CONV_SHAPE) * 2 + prod(SSM_SHAPE) * 4) for group in STATE_GROUPS}
BOUNDARY_PAYLOAD_BYTES = sum(GROUP_OBJECT_BYTES.values())

# The live consumer's block tables: the FA group's pages and the state
# groups' rows (the D15 law — distinct tail rows per group; the D16
# families alias ACROSS groups, distinct rows WITHIN).
BLOCK_TABLES = [
    [9, 10, 11, 12, 13, 14, 15],
    [1, 2, 3, 4, 5, 6, GROUP_TAIL_ROWS[1]],
    [8, 9, 10, 11, 12, 13, GROUP_TAIL_ROWS[2]],
    [16, 17, 18, 19, 20, 21, GROUP_TAIL_ROWS[3]],
]


def _conv_cache(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn((NUM_SLOTS, *CONV_SHAPE), generator=gen, dtype=CONV_DTYPE)


def _ssm_cache(seed: int) -> torch.Tensor:
    flat = torch.randn(NUM_SLOTS * SSM_SLOT_PAD_ELEMS, generator=torch.Generator().manual_seed(seed), dtype=SSM_DTYPE)
    inner = tuple(SSM_SHAPE)
    strides = (SSM_SLOT_PAD_ELEMS,) + tuple(prod(inner[dim + 1 :]) for dim in range(len(inner)))
    return flat.as_strided((NUM_SLOTS, *inner), strides)


def _fa_cache(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn((NUM_SLOTS * FA_KERNEL_ROWS_PER_PAGE, *FA_KERNEL_SHAPE), generator=gen, dtype=torch.bfloat16)


class _Geometry:
    """The live hybrid layout: the D16 shared families + the FA group."""

    def __init__(self):
        self.kv_caches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.fa_caches = (_fa_cache(71), _fa_cache(72))
        for phys in FA_PHYSICAL:
            self.kv_caches[f"model.layers.{phys}.self_attn"] = self.fa_caches
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


class _SessionStore:
    """Backend-seam fake for the hybrid group get session machinery."""

    def __init__(self, get_start_rc=None):
        self.get_start_rc = get_start_rc
        self.get_starts: list[tuple] = []

    def batch_get_start(self, keys):
        self.get_starts.append(tuple(keys))
        if self.get_start_rc is not None:
            return [self.get_start_rc] * len(keys)
        return [0] * len(keys)


def _make_key(group, block_hash, head, stage):
    return hybrid_block_key(MODEL, LAYOUT, group, BLOCK, block_hash, head)


def _make_load_worker(store, use_hybrid=True, kv_role="kv_consumer"):
    """The T2/T3 live-geometry worker extended with the ladder fields."""
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.tp_rank = 0
    worker.num_layers = NUM_LAYERS
    worker.hf_config = SimpleNamespace(num_hidden_layers=NUM_LAYERS)
    worker.use_hybrid = use_hybrid
    worker.use_layerwise = True
    worker.use_block_key_layerwise = True
    worker.block_key_hybrid = use_hybrid
    worker.kv_role = kv_role
    worker.consumer_is_to_put = False
    num_groups = 4 if use_hybrid else 1
    worker.grouped_block_size = [BLOCK] * num_groups
    worker.hash_block_size = HASH_BLOCK_SIZE
    worker.num_blocks = NUM_SLOTS
    worker._global_to_local_layer = {}
    geometry = _Geometry()
    worker.kv_caches = geometry.kv_caches
    groups = [
        SimpleNamespace(layer_names=geometry.group_layer_names(g), kv_cache_spec=geometry.group_spec(g))
        for g in range(num_groups)
    ]
    worker.kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    worker.group_kv_caches_base_addr = {}
    worker.group_block_len = {}
    worker.group_block_stride = {}
    worker.group_layer_cache_entry_offsets = {}
    worker.group_num_layers = {}
    for group_id, group in enumerate(worker.kv_cache_config.kv_cache_groups):
        worker._infer_cache_group_metadata(group_id, group.layer_names)
    worker.cache_coordinator = "coordinator"
    worker.cache_transfer_granularity = BLOCK
    worker.token_database = SimpleNamespace(
        load_mask=lambda hashes, tokens: None,
        store_mask=lambda aligned_token_len, num_prompt_tokens=None: tuple(
            [True] * (aligned_token_len // BLOCK) for _ in range(num_groups)
        ),
    )
    worker.num_kv_cache_groups = num_groups
    worker.head_or_tp_rank = 0
    worker.pp_rank = 0
    worker.layerwise_keys = LayerwiseKeyBuilder(_make_key, 1)
    worker.put_step = 1
    worker.dcp_size = 1
    worker.layerwise_max_transfer_blocks = 0
    worker.use_eagle = False
    worker.layerwise_offload = False
    worker.layer_save_tasks = [[] for _ in range(NUM_LAYERS)]
    worker.layer_load_tasks = [[] for _ in range(NUM_LAYERS)]
    worker._layer_load_aborted = threading.Event()
    worker._put_started_keys = set()
    worker._put_started_keys_lock = threading.Lock()
    worker._layerwise_session_tracker = LayerwiseSessionTracker()
    worker._holemiss_failed_load_keys = {}
    worker._holemiss_tail_key_boundaries = {}
    worker._holemiss_tail_save_probed = set()
    worker._record_layerwise_invalid_blocks = lambda block_ids: None
    worker.m_store = store
    # The T2/T3/T5 registries (__new__ bypasses _init_state_vars).
    worker._holemiss_state_tail_snapshots = {}
    worker._holemiss_state_markers = {}
    worker._holemiss_state_load_results = {}
    worker._holemiss_req_meta_by_id = {}
    worker._holemiss_scratch_owner = None
    worker._holemiss_state_tail_sources = {}
    worker._holemiss_state_tail_keys = set()
    worker._holemiss_state_tail_skip_warned = set()
    worker._holemiss_state_dispositioned_blocks = set()
    worker._invalid_block_ids = set()
    worker._invalid_block_ids_lock = threading.Lock()
    worker._hole_recovery_regions = []
    worker._hole_recovery_lock = threading.Lock()
    worker._holemiss_armed_fault = None
    return worker, geometry


def _chain_hashes(token_count: int, hbs: int):
    """The engine's chained block hashes, the way the bench builds them."""
    import vllm.v1.core.kv_cache_utils as kvcu
    from vllm.utils.hashing import sha256

    token_ids = [(index * 7 + 3) % 30000 for index in range(token_count)]
    kvcu.init_none_hash(sha256)
    parent = kvcu.NONE_HASH
    chain = []
    for start in range(0, token_count, hbs):
        parent = kvcu.hash_block_tokens(sha256, parent, token_ids[start : start + hbs], None)
        chain.append(parent)
    return chain


def _load_request(
    req_id="req-load",
    boundary=BOUNDARY,
    tail_boundary=BOUNDARY,
    chain=None,
    block_tables=None,
    **overrides,
):
    """The T2' extension-shape consumer ReqMeta (claim == the interior hit)."""
    chain = chain if chain is not None else _chain_hashes(EXTENSION_PROMPT, HASH_BLOCK_SIZE)
    params = dict(
        req_id=req_id,
        token_len_chunk=0,
        save_start_token=0,
        save_end_token=0,
        target_token_len=EXTENSION_PROMPT,
        block_ids_by_group=[list(ids) for ids in (block_tables or BLOCK_TABLES)],
        block_hashes=chain,
        can_save=False,
        num_prompt_tokens=EXTENSION_PROMPT,
        load_spec=LoadSpec(
            vllm_cached_tokens=0,
            kvpool_cached_tokens=boundary,
            can_load=True,
            kvpool_store_skip_tokens=boundary,
            tail_boundary_tokens=tail_boundary,
        ),
    )
    params.update(overrides)
    return ReqMeta(**params)


def _capture_request(req_id="req-cap", boundary=BOUNDARY):
    """The last-prefill-step ReqMeta the T2 capture fires on (the producer)."""
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=boundary,
        block_ids_by_group=[list(ids) for ids in BLOCK_TABLES],
        block_hashes=_chain_hashes(boundary, HASH_BLOCK_SIZE),
        can_save=True,
        num_prompt_tokens=boundary,
        target_token_len=boundary,
    )


def _fill_tail_rows(worker, geometry):
    """Write per-group-distinct artifact content into the groups' tail rows
    (the state the completed drain leaves behind — the loaded artifact)."""
    for group, row in GROUP_TAIL_ROWS.items():
        for family, (conv, ssm) in enumerate(geometry.family_caches):
            conv[row] = torch.full_like(conv[row], float(group) + family / 100.0)
            ssm[row] = torch.full_like(ssm[row], float(group) * 2 + family / 100.0)


def _slot_digest(worker, group_id: int, row: int) -> str:
    """The producer-side byte walk (the SNAPSHOT marker's digest order)."""
    digest = hashlib.sha256()
    for _layer_name, cache in worker._holemiss_state_tail_entries(group_id):
        digest.update(cache[row].contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def _markers(worker, req_id="req-load"):
    return worker._holemiss_state_markers.get(req_id, [])


class TestStateTailLoadKey(unittest.TestCase):
    """T5 (a): the groups 1-3 state-tail load keys — the B41/T3 mirror."""

    def test_load_key_mirrors_the_save_key_byte_exactly(self):
        """The mirror byte-equality: the consumer's load key == the
        producer's T3 save key == the bench's expected_partial_tail_keys
        derivation, on a fresh chained-hash fixture (the two signal sources
        — load_spec.tail_boundary_tokens vs finished_tail_boundary — derive
        the SAME D7 object)."""
        chain = _chain_hashes(EXTENSION_PROMPT, HASH_BLOCK_SIZE)
        suffix = f"{chain[BOUNDARY // HASH_BLOCK_SIZE - 1].hex()}_tail_{BOUNDARY}"
        store = _SessionStore()
        worker, _ = _make_load_worker(store, kv_role="kv_producer")
        worker._holemiss_capture_state_tail_snapshots([_capture_request()])
        save_request = ReqMeta(
            req_id="req-cap",
            token_len_chunk=0,
            save_start_token=BOUNDARY,
            save_end_token=BOUNDARY,
            target_token_len=BOUNDARY,
            block_ids_by_group=[list(ids) for ids in BLOCK_TABLES],
            block_hashes=chain[: (BOUNDARY + HASH_BLOCK_SIZE - 1) // HASH_BLOCK_SIZE],
            can_save=True,
            is_last_chunk=True,
            partial_block_index=TAIL_SLOT_INDEX,
            finished_tail_boundary=BOUNDARY,
            num_prompt_tokens=BOUNDARY,
        )
        load_request = _load_request(chain=chain)
        for group in range(4):
            expected = hybrid_block_key(MODEL, LAYOUT, group, BLOCK, suffix, 0)
            save_key = worker._make_layerwise_finished_tail_key(save_request, group)
            load_key = worker._make_layerwise_tail_load_key(load_request, group)
            self.assertIsNotNone(save_key, f"group {group} save key")
            self.assertIsNotNone(load_key, f"group {group} load key")
            self.assertEqual(load_key[0], expected)
            self.assertEqual(save_key[0], load_key[0])
            self.assertEqual(load_key[1], TAIL_SLOT_INDEX)

    def test_group0_load_key_unchanged_by_the_state_extension(self):
        """The group-0 unchanged pin: the FA tail key derivation never
        consults the state machinery — byte-identical to the pre-T5 B41
        form, in group 0's own namespace."""
        chain = _chain_hashes(EXTENSION_PROMPT, HASH_BLOCK_SIZE)
        suffix = f"{chain[BOUNDARY // HASH_BLOCK_SIZE - 1].hex()}_tail_{BOUNDARY}"
        store = _SessionStore()
        worker, _ = _make_load_worker(store)
        request = _load_request(chain=chain)
        key = worker._make_layerwise_tail_load_key(request, 0)
        self.assertIsNotNone(key)
        self.assertEqual(key[0], hybrid_block_key(MODEL, LAYOUT, 0, BLOCK, suffix, 0))
        self.assertNotIn("state", key[0])
        for group in STATE_GROUPS:
            state_key = worker._make_layerwise_tail_load_key(request, group)
            self.assertNotEqual(key[0], state_key[0])
            self.assertIn(f"group:{group}@", state_key[0])

    def test_no_tail_signal_derives_no_key(self):
        """The degenerate-shape no-signal case: the T4 refusal leaves the
        field None — no state-tail fetch for ANY group (the worker never
        learns about the declined artifact)."""
        store = _SessionStore()
        worker, _ = _make_load_worker(store)
        request = _load_request(tail_boundary=None, boundary=FLOOR)
        for group in range(4):
            self.assertIsNone(worker._make_layerwise_tail_load_key(request, group))

    def test_aligned_extent_derives_no_key(self):
        store = _SessionStore()
        worker, _ = _make_load_worker(store)
        request = _load_request(boundary=FLOOR)
        # cached 9216 aligns exactly with 6 full blocks: no partial extent.
        for group in range(4):
            self.assertIsNone(worker._make_layerwise_tail_load_key(request, group))

    def test_mismatched_boundary_derives_no_key(self):
        store = _SessionStore()
        worker, _ = _make_load_worker(store)
        # boundary 11008 lands in block 7; the cached extent's partial block
        # is 6 — the claimed extent and the artifact's block disagree.
        request = _load_request(boundary=BOUNDARY, tail_boundary=7 * BLOCK + HASH_BLOCK_SIZE)
        for group in range(4):
            self.assertIsNone(worker._make_layerwise_tail_load_key(request, group))

    def test_block_table_shortfall_derives_no_key(self):
        store = _SessionStore()
        worker, _ = _make_load_worker(store)
        tables = [list(ids) for ids in BLOCK_TABLES]
        tables[1] = tables[1][:TAIL_SLOT_INDEX]  # group 1 covers only 6 blocks
        request = _load_request(block_tables=tables)
        self.assertIsNone(worker._make_layerwise_tail_load_key(request, 1))
        for group in (0, 2, 3):
            self.assertIsNotNone(worker._make_layerwise_tail_load_key(request, group))

    def test_non_mamba_hybrid_group_never_fetches_a_state_tail(self):
        store = _SessionStore()
        worker, _ = _make_load_worker(store)
        worker.kv_cache_config.kv_cache_groups[1].kv_cache_spec = FullAttentionSpec(
            block_size=BLOCK, num_kv_heads=2, head_size=8, dtype=torch.bfloat16
        )
        request = _load_request()
        self.assertIsNone(worker._make_layerwise_tail_load_key(request, 1))

    def test_text_layout_is_inert(self):
        """Single-group (text) layouts keep the T18 exact-end extension: the
        group-0 key derives as before and no state-group surface exists."""
        store = _SessionStore()
        worker, _ = _make_load_worker(store, use_hybrid=False)
        request = _load_request()
        request.block_ids_by_group = [list(BLOCK_TABLES[0])]
        self.assertEqual(worker._holemiss_state_tail_save_groups(), ())
        self.assertIsNotNone(worker._make_layerwise_tail_load_key(request, 0))
        self.assertIsNone(worker._make_layerwise_tail_load_key(request, 1))


class TestFetchComposition(unittest.TestCase):
    """T5 (b): the state-tail fetch rides the group-view GET channel."""

    def test_views_fetch_full_blocks_plus_the_state_tail_per_group(self):
        """The claimed==fetched composition: the claim 9472 is covered by
        the six full-block keys plus the tail key at index 6 for EVERY
        group — the GET session opens for all 28 keys (4 groups x 7)."""
        chain = _chain_hashes(EXTENSION_PROMPT, HASH_BLOCK_SIZE)
        suffix = f"{chain[BOUNDARY // HASH_BLOCK_SIZE - 1].hex()}_tail_{BOUNDARY}"
        store = _SessionStore()
        worker, _ = _make_load_worker(store)
        request = _load_request(chain=chain)
        with self.assertLogs(vllm_logger, level="INFO"):
            views = _prepare_group_sessions(worker, [request])
        for group in range(4):
            view = views[group][0]
            self.assertEqual(len(view.load_block_keys), TAIL_SLOT_INDEX + 1)
            self.assertTrue(all(key is not None for key in view.load_block_keys))
            for index in range(TAIL_SLOT_INDEX):
                grouped_hash = chain[(index + 1) * (BLOCK // HASH_BLOCK_SIZE) - 1].hex()
                self.assertEqual(
                    view.load_block_keys[index],
                    hybrid_block_key(MODEL, LAYOUT, group, BLOCK, grouped_hash, 0),
                )
            self.assertEqual(
                view.load_block_keys[TAIL_SLOT_INDEX],
                hybrid_block_key(MODEL, LAYOUT, group, BLOCK, suffix, 0),
            )
        started = [key for keys in store.get_starts for key in keys]
        self.assertEqual(len(started), 4 * (TAIL_SLOT_INDEX + 1))
        for group in range(4):
            self.assertIn(hybrid_block_key(MODEL, LAYOUT, group, BLOCK, suffix, 0), started)
        # The fetched token extent covers the claim: [0, floor) full blocks
        # plus the tail artifact's block (the state slot the kernel reads).
        self.assertEqual(TAIL_SLOT_INDEX * BLOCK + BLOCK, (BOUNDARY // BLOCK + 1) * BLOCK)

    def test_tail_key_load_marker_fires_for_group0_only(self):
        """The census law (design section h): HOLEMISS_TAIL_KEY_LOAD is the
        FA-tail family — the bench gates tail_key_loaded == tp exactly; the
        state groups' census point is the STATE_LOAD source=tail marker."""
        store = _SessionStore()
        worker, _ = _make_load_worker(store)
        request = _load_request()
        with self.assertLogs(vllm_logger, level="INFO") as logs:
            _prepare_group_sessions(worker, [request])
        tail_lines = [line for line in logs.output if "HOLEMISS_TAIL_KEY_LOAD" in line]
        self.assertEqual(len(tail_lines), 1, logs.output)
        self.assertIn("tail_block=6", tail_lines[0])
        self.assertIn(f"boundary={BOUNDARY}", tail_lines[0])
        self.assertIn("group:0@", tail_lines[0].split("key=")[1])

    def test_failed_get_start_clears_the_state_tail_row(self):
        """A rejected get-start clears the view's tail row (the failed-key
        record feeds the ladder's demotion) — the existing group-0
        semantics extended to the state groups."""
        store = _SessionStore(get_start_rc=-1)
        worker, _ = _make_load_worker(store)
        request = _load_request()
        views = _prepare_group_sessions(worker, [request])
        for group in range(4):
            self.assertIsNone(views[group][0].load_block_keys[TAIL_SLOT_INDEX])
        failed = worker._holemiss_failed_load_keys["req-load"]
        for group in STATE_GROUPS:
            tail = worker._make_layerwise_tail_load_key(_load_request(), group)
            self.assertIn(tail[0], failed)


class TestLadderTailRung(unittest.TestCase):
    """T5 (c): the interior tail boundary as the newest rung-0 candidate."""

    def test_plan_builder_appends_the_tail_boundary_as_the_newest_rung(self):
        store = _SessionStore()
        worker, _ = _make_load_worker(store)
        request = _load_request()
        plan = worker._holemiss_build_state_ladder_plan(request)
        self.assertIsNotNone(plan)
        self.assertEqual(
            [b.boundary_tokens for b in plan.boundaries],
            [(index + 1) * BLOCK for index in range(6)] + [BOUNDARY],
        )
        self.assertEqual(plan.tail_boundary_tokens, BOUNDARY)
        self.assertEqual(plan.fa_hit_tokens, BOUNDARY)
        tail = plan.boundaries[-1]
        self.assertEqual(tail.block_index, TAIL_SLOT_INDEX)
        for group in STATE_GROUPS:
            load_key = worker._make_layerwise_tail_load_key(request, group)
            self.assertEqual(tail.keys_by_group[group], (load_key[0],))
            self.assertEqual(tail.blocks_by_group[group], (BLOCK_TABLES[group][TAIL_SLOT_INDEX],))
            self.assertIn(f"group:{group}@", tail.keys_by_group[group][0])
        # The full-block boundaries are unchanged: the newest full rung is
        # still the floor, with its own grouped-hash keys.
        newest_full = plan.boundaries[-2]
        self.assertEqual(newest_full.boundary_tokens, FLOOR)
        for group in STATE_GROUPS:
            grouped_hash = request.block_hashes[(5 + 1) * (BLOCK // HASH_BLOCK_SIZE) - 1]
            self.assertEqual(
                newest_full.keys_by_group[group],
                (hybrid_block_key(MODEL, LAYOUT, group, BLOCK, grouped_hash.hex(), 0),),
            )

    def test_plan_builder_without_the_tail_signal_stays_full_blocks_only(self):
        """The degenerate-shape no-signal pin: the T4 refusal means NO tail
        signal — the ladder sees nothing, the plan stays full-blocks-only
        (rung 0 = the floor 9216)."""
        store = _SessionStore()
        worker, _ = _make_load_worker(store)
        request = _load_request(tail_boundary=None, boundary=FLOOR)
        plan = worker._holemiss_build_state_ladder_plan(request)
        self.assertEqual([b.boundary_tokens for b in plan.boundaries], [(index + 1) * BLOCK for index in range(6)])
        self.assertIsNone(plan.tail_boundary_tokens)
        result = worker._holemiss_run_gdn_state_ladder(request)
        self.assertEqual(result.rung, RUNG_LAST_SAVED)
        self.assertEqual(result.source, SOURCE_LAST_SAVED)
        self.assertEqual(result.loaded_boundary_tokens, FLOOR)
        self.assertFalse(any("source=tail" in line for line in _markers(worker)))

    def test_block_table_shortfall_drops_the_tail_rung_entirely(self):
        """All-or-nothing: one state group's shortfall suppresses the tail
        rung (a mixed rung would load one group's slot 6 while another
        demotes — the ladder's boundary granularity)."""
        store = _SessionStore()
        worker, _ = _make_load_worker(store)
        tables = [list(ids) for ids in BLOCK_TABLES]
        tables[2] = tables[2][:TAIL_SLOT_INDEX]
        request = _load_request(block_tables=tables)
        plan = worker._holemiss_build_state_ladder_plan(request)
        self.assertEqual([b.boundary_tokens for b in plan.boundaries], [(index + 1) * BLOCK for index in range(6)])
        self.assertIsNone(plan.tail_boundary_tokens)

    def test_tail_rung_hit_loads_at_rung0_with_source_tail(self):
        store = _SessionStore()
        worker, geometry = _make_load_worker(store)
        _fill_tail_rows(worker, geometry)
        request = _load_request()
        result = worker._holemiss_run_gdn_state_ladder(request)
        self.assertEqual(result.rung, RUNG_LAST_SAVED)
        self.assertEqual(result.source, SOURCE_TAIL)
        self.assertEqual(result.loaded_boundary_tokens, BOUNDARY)
        self.assertEqual(result.recompute_span, (BOUNDARY, BOUNDARY))
        self.assertEqual(result.state_load_block_ids, set(GROUP_TAIL_ROWS.values()))
        self.assertFalse(result.failed_block_ids)
        loads = [line for line in _markers(worker) if line.startswith("HOLEMISS_STATE_LOAD")]
        self.assertEqual(len(loads), len(STATE_GROUPS))
        for line in loads:
            match = STATE_LOAD_RE.fullmatch(line)
            self.assertIsNotNone(match, line)
            self.assertEqual(match.group(5), "tail")
            self.assertEqual(int(match.group(3)), BOUNDARY)
            self.assertEqual(int(match.group(6)), 0)
            self.assertEqual(int(match.group(7)), BOUNDARY_PAYLOAD_BYTES)
        self.assertFalse([line for line in _markers(worker) if line.startswith("HOLEMISS_STATE_ABSENT")])

    def test_tail_rung_sha_is_the_loaded_slot_digest_per_group(self):
        """The STATE_LOAD sha fill (design section h.4): the per-group
        digest of the loaded slot, byte-walked in the SAME order as the
        producer's SNAPSHOT marker — LEG-1's consumer side. The roundtrip:
        the T2 freeze over the same rows yields the SAME digest."""
        store = _SessionStore()
        worker, geometry = _make_load_worker(store)
        _fill_tail_rows(worker, geometry)
        # The producer leg: the T2 freeze reads the same tail rows.
        producer = _make_load_worker(_SessionStore(), kv_role="kv_producer")[0]
        producer.kv_caches = worker.kv_caches
        producer._holemiss_capture_state_tail_snapshots([_capture_request()])
        frozen = producer._holemiss_state_tail_snapshots[("req-cap", BOUNDARY)]
        request = _load_request()
        worker._holemiss_run_gdn_state_ladder(request)
        loads = [line for line in _markers(worker) if line.startswith("HOLEMISS_STATE_LOAD")]
        shas = {}
        for line in loads:
            match = STATE_LOAD_RE.fullmatch(line)
            shas[int(match.group(2))] = match.group(8)
        self.assertEqual(sorted(shas), sorted(STATE_GROUPS))
        self.assertEqual(len(set(shas.values())), len(STATE_GROUPS), "the digests must be per-group")
        for group in STATE_GROUPS:
            expected = _slot_digest(worker, group, GROUP_TAIL_ROWS[group])
            self.assertEqual(shas[group], expected)
            # LEG-1: the producer's frozen bytes hash to the same value.
            producer_digest = hashlib.sha256()
            for tensor in frozen[group]:
                producer_digest.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
            self.assertEqual(shas[group], producer_digest.hexdigest())

    def test_sha_failure_degrades_to_none_never_raises(self):
        """A broken cache walk logs loudly and leaves sha=none — an honest
        gap, never a guessed digest, never a raise into the ladder."""
        store = _SessionStore()
        worker, _ = _make_load_worker(store)
        request = _load_request()
        original = worker._holemiss_state_tail_entries
        worker._holemiss_state_tail_entries = lambda group_id: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            with self.assertLogs(vllm_logger, level="ERROR") as logs:
                result = worker._holemiss_run_gdn_state_ladder(request)
        finally:
            worker._holemiss_state_tail_entries = original
        self.assertEqual(result.source, SOURCE_TAIL)
        loads = [line for line in _markers(worker) if line.startswith("HOLEMISS_STATE_LOAD")]
        self.assertEqual(len(loads), len(STATE_GROUPS))
        for line in loads:
            self.assertIn("sha=none", line)
        self.assertTrue(any("HOLEMISS state tail load sha failed" in line for line in logs.output))

    def test_tail_rung_miss_demotes_loudly_to_the_floor(self):
        """The loud demotion: a failed state-tail GET demotes the ladder to
        the 9216 rung — the ABSENT marker for the tail boundary, the
        STATE_LOAD at the floor, the recompute span recorded, and the
        failed blocks dispositioned (the existing ladder semantics)."""
        store = _SessionStore()
        worker, geometry = _make_load_worker(store)
        _fill_tail_rows(worker, geometry)
        request = _load_request()
        plan = worker._holemiss_build_state_ladder_plan(request)
        tail_keys = {key for group in STATE_GROUPS for key in plan.boundaries[-1].keys_by_group[group]}
        worker._holemiss_record_failed_load_keys("req-load", tail_keys)
        result = worker._holemiss_run_gdn_state_ladder(request)
        self.assertEqual(result.rung, RUNG_OLDER_SAVED)
        self.assertEqual(result.source, SOURCE_OLDER_SAVED)
        self.assertEqual(result.loaded_boundary_tokens, FLOOR)
        self.assertEqual(result.recompute_span, (FLOOR, BOUNDARY))
        self.assertEqual(result.failed_block_ids, set(GROUP_TAIL_ROWS.values()))
        self.assertEqual(worker._holemiss_state_dispositioned_blocks, set(GROUP_TAIL_ROWS.values()))
        self.assertEqual(worker._invalid_block_ids, set(GROUP_TAIL_ROWS.values()))
        markers = _markers(worker)
        absent = [
            line for line in markers if line.startswith("HOLEMISS_STATE_ABSENT") and f"boundary={BOUNDARY}" in line
        ]
        self.assertEqual(len(absent), len(STATE_GROUPS))
        for line in absent:
            match = STATE_ABSENT_RE.fullmatch(line)
            self.assertIsNotNone(match, line)
            self.assertEqual(match.group(4), "load_failed")
            self.assertEqual(int(match.group(5)), RUNG_OLDER_SAVED)
        loaded = [line for line in markers if line.startswith("HOLEMISS_STATE_LOAD") and f"boundary={FLOOR}" in line]
        self.assertEqual(len(loaded), len(STATE_GROUPS))
        for line in loaded:
            self.assertIn("source=older_saved", line)
            # The full-block rungs keep sha=none — byte-identical today.
            self.assertIn("sha=none", line)

    def test_marker_mutex_law_respected_per_boundary(self):
        """A boundary emits STATE_LOAD xor STATE_ABSENT — the tail rung's
        demotion path must respect it (both the hit and the miss shapes)."""
        for failed_tail in (False, True):
            with self.subTest(failed_tail=failed_tail):
                store = _SessionStore()
                worker, _ = _make_load_worker(store)
                request = _load_request()
                if failed_tail:
                    plan = worker._holemiss_build_state_ladder_plan(request)
                    tail_keys = {key for group in STATE_GROUPS for key in plan.boundaries[-1].keys_by_group[group]}
                    worker._holemiss_record_failed_load_keys("req-load", tail_keys)
                worker._holemiss_run_gdn_state_ladder(request)
                markers = _markers(worker)
                for boundary_tokens in (BOUNDARY, FLOOR):
                    loads = [
                        line
                        for line in markers
                        if line.startswith("HOLEMISS_STATE_LOAD") and f"boundary={boundary_tokens}" in line
                    ]
                    absents = [
                        line
                        for line in markers
                        if line.startswith("HOLEMISS_STATE_ABSENT") and f"boundary={boundary_tokens}" in line
                    ]
                    # The mutex: a boundary that emits one never emits the
                    # other (a never-walked boundary — the hit case stops at
                    # rung 0 before the floor — emits neither, which is
                    # legal: the law constrains EMITTED markers only).
                    self.assertFalse(loads and absents, (boundary_tokens, loads, absents))
                    self.assertFalse(len(loads) > len(STATE_GROUPS), (boundary_tokens, loads))
                    self.assertFalse(len(absents) > len(STATE_GROUPS), (boundary_tokens, absents))

    def test_text_layout_plan_stays_structurally_inert(self):
        store = _SessionStore()
        worker, _ = _make_load_worker(store, use_hybrid=False)
        request = _load_request()
        request.block_ids_by_group = [list(BLOCK_TABLES[0])]
        self.assertIsNone(worker._holemiss_build_state_ladder_plan(request))


class TestCompositionArithmetic(unittest.TestCase):
    """design section a/d: src=dst=6 at both shapes — the kernel reads the
    loaded artifact in place (no copy); the UT pins the arithmetic against
    the LIVE preprocess_mamba formulas (S11)."""

    def test_the_live_formulas_are_pinned_at_source(self):
        from vllm_ascend.patch.worker import patch_mamba_utils

        source = inspect.getsource(patch_mamba_utils.preprocess_mamba)
        self.assertIn("prev_state_idx = (req_state.num_computed_tokens - 1) // block_size", source)
        self.assertIn("cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size)", source)
        self.assertIn("curr_state_idx = num_blocks - 1 - num_speculative_blocks", source)
        self.assertIn("if prev_state_idx != -1 and prev_state_idx != curr_state_idx:", source)

    def test_extension_shape_composes_exactly(self):
        """The T2' shape: the claim 9472 (no -1: the hit is interior), the
        true incoming 256 — the kernel reads slot 6 = the loaded artifact
        and advances it in place."""
        num_computed, scheduled, num_speculative_blocks = BOUNDARY, HASH_BLOCK_SIZE, 0
        prev_state_idx = (num_computed - 1) // BLOCK
        num_blocks = cdiv(num_computed + scheduled, BLOCK) + num_speculative_blocks
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        self.assertEqual((prev_state_idx, curr_state_idx), (TAIL_SLOT_INDEX, TAIL_SLOT_INDEX))

    def test_degenerate_shape_needs_the_state_after_9470(self):
        """The -1 shape (today's T2, the pre-T4 behavior): the kernel reads
        slot 6 needing the state after token 9470, while the boundary-state
        artifact holds after-9471 — the structural off-by-one behind the
        honest refusal (T4); the exact composition lives at the extension
        shape only."""
        num_computed, scheduled, num_speculative_blocks = BOUNDARY - 1, 1, 0
        prev_state_idx = (num_computed - 1) // BLOCK
        num_blocks = cdiv(num_computed + scheduled, BLOCK) + num_speculative_blocks
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        self.assertEqual((prev_state_idx, curr_state_idx), (TAIL_SLOT_INDEX, TAIL_SLOT_INDEX))


if __name__ == "__main__":
    unittest.main()
