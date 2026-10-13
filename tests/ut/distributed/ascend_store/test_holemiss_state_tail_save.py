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
"""hybrid-gdn-tail-state T3: the state-tail save surface.

The GDN groups 1-3 persist their tail boundary state under the B39-mirror
key (the SAME raw-chain sub-hash + ``partial_tail_key_suffix``, the group's
own D7 namespace and object size), and the put's ranged read addresses the
T2 FROZEN SNAPSHOT — never the live slot, which the populate's own decode
advanced past the boundary state before the finish-time drain runs. These
tests pin the byte-equality of all four groups' keys against the bench's
``expected_partial_tail_keys`` derivation (the cross-repo mirror), the
group-0 unchanged law, the session integration (the never-re-put filter,
the rejected-put loudness, the group's own object size), the
snapshot-consumed proof (mutate the live slot post-freeze -> the put bytes
stay the frozen content), the missing-snapshot loud skip, and the negative
controls (aligned ends, text layouts, non-finishing requests, non-Mamba
groups).

The fixtures model the LIVE geometry (the N18/N19 anti-fake law): the T2
fixture's 4-group layout (16 FA layers at physical 3+4k with kernel-gran
caches, 48 GDN layers in 3 round-robin groups of 16, the D16 conv+ssm
aliasing, the mixed bf16/fp32 tuple, MRV2 page padding, the 1536/256 block
space) extended with the B39 session machinery, and the group metadata
derived by the REAL ``_infer_cache_group_metadata``.
"""

import ctypes
import inspect
import threading
import unittest
from collections import defaultdict
from math import prod
from types import SimpleNamespace

import torch
from vllm.logger import logger as vllm_logger
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

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
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)

NUM_LAYERS = 64
BLOCK = 1536
BOUNDARY = 9472  # 6*1536 + 256: unaligned vs the group block, pmu-aligned
FLOOR = 6 * BLOCK
TAIL_SLOT_INDEX = 6
NUM_SLOTS = 32
HASH_BLOCK_SIZE = 256

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
    """Backend-seam fake for the hybrid group put/get session machinery."""

    requires_exists_before_put = True

    def __init__(self, exists=None, put_start_rc=None):
        self.exists = dict(exists or {})
        self.put_start_rc = put_start_rc
        self.put_starts: list[tuple[tuple, tuple]] = []
        self.get_starts: list[tuple] = []
        self.copy_puts: list[tuple] = []
        self.commits: list[tuple] = []
        self.revokes: list[tuple] = []

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

    def batch_copy_put(self, keys, buffers, sizes, offsets):
        # Read eagerly: the source addresses hold the artifact only while
        # the frozen sources are alive (the put-complete free drops them).
        payloads = []
        for row_buffers, row_sizes in zip(buffers, sizes, strict=True):
            chunk = bytearray()
            for address, size in zip(row_buffers, row_sizes, strict=True):
                chunk.extend((ctypes.c_char * size).from_address(address))
            payloads.append(bytes(chunk))
        self.copy_puts.append((tuple(keys), [list(b) for b in buffers], [list(s) for s in sizes], payloads))
        return [0] * len(keys)

    def batch_commit(self, keys):
        self.commits.append(tuple(keys))
        return [0] * len(keys)

    def batch_revoke(self, keys):
        self.revokes.append(tuple(keys))
        return [0] * len(keys)


def _make_key(group, block_hash, head, stage):
    return hybrid_block_key(MODEL, LAYOUT, group, BLOCK, block_hash, head)


def _make_session_worker(store, kv_role="kv_producer", use_hybrid=True):
    """The T2 live-geometry worker extended with the B39 session machinery."""
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
    # The session-plane fields (the _make_tail_session_worker pattern).
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
    # The T2/T3 registries (__new__ bypasses _init_state_vars).
    worker._holemiss_state_tail_snapshots = {}
    worker._holemiss_state_tail_snapshot_pending_free = {}
    worker._holemiss_state_tail_seen_reqs = set()
    worker._holemiss_state_markers = {}
    worker._holemiss_state_load_results = {}
    worker._holemiss_req_meta_by_id = {}
    worker._holemiss_scratch_owner = None
    worker._holemiss_state_tail_sources = {}
    worker._holemiss_state_tail_keys = set()
    worker._holemiss_state_tail_skip_warned = set()
    worker._hole_recovery_regions = []
    worker._hole_recovery_lock = threading.Lock()
    worker._holemiss_armed_fault = None
    return worker, geometry


BLOCK_TABLES = [
    [9, 10, 11, 12, 13, 14, 15],
    [1, 2, 3, 4, 5, 6, GROUP_TAIL_ROWS[1]],
    [8, 9, 10, 11, 12, 13, GROUP_TAIL_ROWS[2]],
    [16, 17, 18, 19, 20, 21, GROUP_TAIL_ROWS[3]],
]


def _tail_save_request(req_id="req-tail", block_hashes=None, boundary=BOUNDARY, **overrides):
    params = dict(
        req_id=req_id,
        token_len_chunk=0,
        save_end_token=boundary,
        save_start_token=boundary,
        target_token_len=boundary,
        block_ids_by_group=[list(ids) for ids in BLOCK_TABLES],
        block_hashes=block_hashes
        if block_hashes is not None
        else [bytes([(index + 1) % 256] * 32) for index in range((boundary + HASH_BLOCK_SIZE - 1) // HASH_BLOCK_SIZE)],
        can_save=True,
        is_last_chunk=True,
        partial_block_index=TAIL_SLOT_INDEX,
        finished_tail_boundary=boundary,
        num_prompt_tokens=boundary,
    )
    params.update(overrides)
    return ReqMeta(**params)


def _fixture_hashes(boundary: int) -> list[bytes]:
    count = (boundary + HASH_BLOCK_SIZE - 1) // HASH_BLOCK_SIZE
    return [bytes([(index + 1) % 256] * 32) for index in range(count)]


def _capture_request(req_id="req-tail", boundary=BOUNDARY):
    """The last-prefill-step ReqMeta the T2 capture fires on."""
    return ReqMeta(
        req_id=req_id,
        token_len_chunk=boundary,
        block_ids_by_group=[list(ids) for ids in BLOCK_TABLES],
        block_hashes=_fixture_hashes(boundary),
        can_save=True,
        num_prompt_tokens=boundary,
        target_token_len=boundary,
    )


def _capture(worker, request):
    worker._holemiss_capture_state_tail_snapshots([request])


def _tensors_bytes(tensors) -> bytes:
    digest = bytearray()
    for tensor in tensors:
        digest.extend(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return bytes(digest)


def _chain_hashes(token_count: int, hbs: int, token_ids=None):
    """The engine's chained block hashes, the way the bench builds them."""
    import vllm.v1.core.kv_cache_utils as kvcu
    from vllm.utils.hashing import sha256

    if token_ids is None:
        token_ids = [(index * 7 + 3) % 30000 for index in range(token_count)]
    kvcu.init_none_hash(sha256)
    parent = kvcu.NONE_HASH
    chain = []
    for start in range(0, token_count, hbs):
        parent = kvcu.hash_block_tokens(sha256, parent, token_ids[start : start + hbs], None)
        chain.append(parent)
    return chain, token_ids


def _builder_for(worker, group_id: int) -> LayerBatchBuilder:
    database = SimpleNamespace(
        group_block_len=worker.group_block_len,
        group_kv_caches_base_addr=worker.group_kv_caches_base_addr,
        group_block_stride=worker.group_block_stride,
        group_layer_cache_entry_offsets=worker.group_layer_cache_entry_offsets,
    )
    return LayerBatchBuilder(
        database, page_size_bytes=4096, num_layers=worker.group_num_layers[group_id], group_id=group_id
    )


def _sending_thread(worker) -> KVCacheStoreLayerSendingThread:
    thread = KVCacheStoreLayerSendingThread.__new__(KVCacheStoreLayerSendingThread)
    thread.worker = worker
    return thread


def _drive_state_group_put(worker, store, group_id=1, physical_layer=0):
    """Drive one state group's partial-tail save through the REAL send-thread
    task path (build_addrs -> the frozen-source substitution -> the ranged put
    -> the commit -> the put-complete free). Returns the group's view."""
    views = _prepare_group_sessions(worker, [_tail_save_request()])
    view = views[group_id][0]
    worker._process_save_for_layer_batch([view], physical_layer, group_id=group_id, layer_idx_in_group=0)
    tasks = worker.layer_save_tasks[physical_layer]
    assert tasks, "the state group's partial-tail save task must exist"
    task = tasks[0]
    builder = _builder_for(worker, group_id)
    task.shared_block_data = builder.build_shared(task, is_save=True)
    thread = KVCacheStoreLayerSendingThread.__new__(KVCacheStoreLayerSendingThread)
    thread.worker = worker
    thread.m_store = store
    thread.final_layer_id = physical_layer
    thread.max_transfer_blocks = 0
    thread.max_transfer_bytes = 0
    thread.layer_save_finished_events = [threading.Event() for _ in range(NUM_LAYERS)]
    thread.sync_save_events = [SimpleNamespace(synchronize=lambda: None) for _ in range(NUM_LAYERS)]
    thread._put_started_keys = worker._put_started_keys
    thread._put_started_keys_lock = worker._put_started_keys_lock
    thread._session_tracker = worker._layerwise_session_tracker
    thread._active_put_keys = None
    thread._group_put_keys = {}
    thread.group_builders = None
    thread.layer_batch_builder = builder
    thread.request_queue = SimpleNamespace(task_done=lambda: None)
    thread.done_task_lock = threading.Lock()
    thread.stored_requests = defaultdict(int)
    thread.finished_requests = set()
    thread.kv_event_lock = threading.Lock()
    thread.kv_events = []
    thread.add_stored_request(view.req_id)
    thread._handle_range_layer_tasks([task])
    return view


class TestStateTailKeyConstruction(unittest.TestCase):
    """T3 (a): the groups 1-3 state-tail keys — the B39 mirror."""

    def test_state_tail_keys_match_the_bench_expected_form_all_four_groups(self):
        """The byte-equality pin: all four groups' keys equal the bench's
        expected_partial_tail_keys derivation (the chained sub-hash at
        boundary//hbs - 1, suffixed _tail_{boundary}, wrapped in each group's
        D7 hybrid key) on a fresh seeded fixture."""
        chain, _ = _chain_hashes(BOUNDARY, HASH_BLOCK_SIZE)
        sub_hash = chain[BOUNDARY // HASH_BLOCK_SIZE - 1].hex()
        suffix = f"{sub_hash}_tail_{BOUNDARY}"
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request())
        request = _tail_save_request(block_hashes=chain)
        for group in range(4):
            expected = hybrid_block_key(MODEL, LAYOUT, group, BLOCK, suffix, 0)
            tail = worker._make_layerwise_finished_tail_key(request, group)
            self.assertIsNotNone(tail, f"group {group} must derive a tail key")
            self.assertEqual(tail[0], expected)
            self.assertEqual(tail[1], TAIL_SLOT_INDEX)
            self.assertIn(f"group:{group}@", expected)
            self.assertIn(f"@{suffix}@", expected)

    def test_group0_key_unchanged_with_snapshots_present(self):
        """The group-0 unchanged pin: the FA tail key derivation never
        consults the snapshot machinery — byte-identical to the pre-T3 form,
        and never registered as a state key."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request())
        request = _tail_save_request()
        key = worker._make_layerwise_finished_tail_key(request, 0)
        self.assertIsNotNone(key)
        self.assertNotIn(key[0], worker._holemiss_state_tail_keys)
        self.assertEqual(worker._holemiss_state_tail_sources, {})
        # The FA key form: group 0's own namespace, the boundary sub-hash.
        sub = request.block_hashes[BOUNDARY // HASH_BLOCK_SIZE - 1]
        self.assertEqual(
            key[0],
            hybrid_block_key(MODEL, LAYOUT, 0, BLOCK, f"{sub.hex()}_tail_{BOUNDARY}", 0),
        )

    def test_state_groups_open_put_sessions_with_group_object_sizes(self):
        """The session integration: each state group's view carries its tail
        key and the put session opens with the GROUP's own object size
        sum(group_block_len[g]) — the B39 pattern generalized."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request())
        request = _tail_save_request()
        views = _prepare_group_sessions(worker, [request])
        started = {keys: sizes for keys, sizes in store.put_starts}
        for group in range(4):
            view = views[group][0]
            tail = worker._make_layerwise_finished_tail_key(request, group)
            self.assertIsNotNone(tail)
            self.assertEqual(view.save_last_block_key, tail[0])
            self.assertEqual(view.partial_block_index, TAIL_SLOT_INDEX)
            self.assertIn((tail[0],), list(started))
        for group in STATE_GROUPS:
            tail = worker._make_layerwise_finished_tail_key(request, group)
            self.assertEqual(started[(tail[0],)], (GROUP_OBJECT_BYTES[group],))

    def test_second_same_boundary_request_does_not_reput(self):
        """The never-re-put law: the state-tail keys are cross-request
        derivable, so a second request sharing the boundary (with its OWN
        capture — the live flow) must not re-open the sessions."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request(req_id="req-A"))
        first = _tail_save_request(req_id="req-A")
        second = _tail_save_request(req_id="req-B")
        views_a = _prepare_group_sessions(worker, [first])
        self.assertEqual(len(store.put_starts), 4)
        _capture(worker, _capture_request(req_id="req-B"))
        views_b = _prepare_group_sessions(worker, [second])
        self.assertEqual(len(store.put_starts), 4)
        for group in range(4):
            self.assertEqual(views_b[group][0].save_last_block_key, views_a[group][0].save_last_block_key)

    def test_pooled_state_tail_key_is_not_reput_and_drops_the_row(self):
        """The pool-side twin: a tail key already readable in the store is
        filtered out — no put_start, and the view keeps the key clear so
        the partial row is skipped (all four groups' keys pooled)."""
        probe_store = _SessionStore()
        probe_worker, _ = _make_session_worker(probe_store)
        _capture(probe_worker, _capture_request())
        probe_request = _tail_save_request()
        keys = [probe_worker._make_layerwise_finished_tail_key(probe_request, group)[0] for group in range(4)]
        store = _SessionStore(exists={key: 1 for key in keys})
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request())
        request = _tail_save_request()
        views = _prepare_group_sessions(worker, [request])
        self.assertEqual(store.put_starts, [])
        for group in range(4):
            self.assertIsNone(views[group][0].save_last_block_key)
            self.assertIsNone(views[group][0].partial_block_index)

    def test_rejected_put_start_is_loud_and_drops_the_state_rows(self):
        """The loud path: a rejected put_start warns and leaves the state
        groups' tail keys clear — the B39 pattern verbatim, generalized."""
        store = _SessionStore(put_start_rc=-705)
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request())
        request = _tail_save_request()
        with self.assertLogs(vllm_logger, level="WARNING") as logs:
            views = _prepare_group_sessions(worker, [request])
        self.assertEqual(len(store.put_starts), 4)
        for group in STATE_GROUPS:
            view = views[group][0]
            self.assertIsNone(view.save_last_block_key)
            self.assertIsNone(view.partial_block_index)
        self.assertTrue(
            any(
                "put_start rejected" in line and f"group {group}" in line
                for line in logs.output
                for group in STATE_GROUPS
            )
        )

    def test_missing_snapshot_skips_state_keys_loudly(self):
        """The design section b.7 contract: no frozen snapshot -> the
        state-tail keys are skipped with ONE loud warning per
        (req_id, boundary); the group-0 FA key is unaffected (it never reads
        the snapshot)."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        request = _tail_save_request()  # no capture ever fired
        with self.assertLogs(vllm_logger, level="WARNING") as logs:
            views = _prepare_group_sessions(worker, [request])
        skip_lines = [line for line in logs.output if "HOLEMISS_STATE_TAIL_PUT_SKIPPED" in line]
        self.assertEqual(len(skip_lines), 1)
        self.assertIn(f"boundary={BOUNDARY}", skip_lines[0])
        self.assertIn("reason=no_frozen_snapshot", skip_lines[0])
        for group in STATE_GROUPS:
            self.assertIsNone(worker._make_layerwise_finished_tail_key(request, group))
            self.assertIsNone(views[group][0].save_last_block_key)
            self.assertIsNone(views[group][0].partial_block_index)
        fa_key = worker._make_layerwise_finished_tail_key(request, 0)[0]
        self.assertEqual(store.put_starts, [((fa_key,), (sum(worker.group_block_len[0]),))])
        # The dedup: the second derivation does not re-warn.
        with self.assertNoLogs(vllm_logger, level="WARNING"):
            worker._make_layerwise_finished_tail_key(request, 1)

    def test_degraded_capture_skips_loudly_too(self):
        """The T2 loud-degrade end of the contract: a capture that FAILED
        leaves no snapshot, so the skip fires (the FAILED line was already
        emitted by the capture side)."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        request = _capture_request()
        request.block_ids_by_group[2] = [1, 2, 3, 4, 5, 6, NUM_SLOTS + 5]  # out-of-range row
        with self.assertLogs(vllm_logger, level="WARNING") as logs:
            _capture(worker, request)
        self.assertTrue(any("HOLEMISS_STATE_TAIL_SNAPSHOT_FAILED" in line for line in logs.output))
        tail_request = _tail_save_request()
        with self.assertLogs(vllm_logger, level="WARNING") as logs:
            views = _prepare_group_sessions(worker, [tail_request])
        self.assertTrue(any("HOLEMISS_STATE_TAIL_PUT_SKIPPED" in line for line in logs.output))
        for group in STATE_GROUPS:
            self.assertIsNone(views[group][0].save_last_block_key)


class TestStateTailNegativeControls(unittest.TestCase):
    def test_aligned_end_derives_no_state_key(self):
        """An aligned boundary belongs to the aligned channel: no state-tail
        key and NO skip warning (nothing is owed — silent)."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request(boundary=FLOOR))
        self.assertEqual(worker._holemiss_state_tail_snapshots, {})
        request = _tail_save_request(boundary=FLOOR, block_hashes=_fixture_hashes(FLOOR))
        with self.assertNoLogs(vllm_logger, level="WARNING"):
            for group in STATE_GROUPS:
                self.assertIsNone(worker._make_layerwise_finished_tail_key(request, group))

    def test_text_layout_is_inert(self):
        """A single-group (text) layout never reaches the state branch: the
        group-0 key derives as before and the snapshot machinery is never
        consulted."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store, use_hybrid=False)
        request = _tail_save_request()
        request.block_ids_by_group = [list(BLOCK_TABLES[0])]
        with self.assertNoLogs(vllm_logger, level="WARNING"):
            key = worker._make_layerwise_finished_tail_key(request, 0)
        self.assertIsNotNone(key)
        self.assertEqual(worker._holemiss_state_tail_keys, set())
        self.assertEqual(worker._holemiss_state_tail_sources, {})

    def test_non_finishing_request_derives_no_key(self):
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request())
        request = _tail_save_request(finished_tail_boundary=None)
        with self.assertNoLogs(vllm_logger, level="WARNING"):
            for group in range(4):
                self.assertIsNone(worker._make_layerwise_finished_tail_key(request, group))

    def test_non_mamba_group_stays_silent(self):
        """A hybrid group that is not a MambaSpec state group (SWA/compressor
        state is not a portable snapshot) stays ineligible — silently."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        worker.kv_cache_config.kv_cache_groups[1].kv_cache_spec = FullAttentionSpec(
            block_size=BLOCK, num_kv_heads=2, head_size=8, dtype=torch.bfloat16
        )
        _capture(worker, _capture_request())
        request = _tail_save_request()
        with self.assertNoLogs(vllm_logger, level="WARNING"):
            self.assertIsNone(worker._make_layerwise_finished_tail_key(request, 1))


class TestSnapshotConsumedPut(unittest.TestCase):
    """T3 (b): the send thread's ranged read addresses the FROZEN snapshot."""

    def test_put_reads_the_frozen_snapshot_not_the_live_slot(self):
        """The core proof: mutate the live tail rows post-freeze (the decode
        advance), drive the REAL send-thread task path, and read the bytes at
        the buffers handed to batch_copy_put — they must be the FROZEN
        content, never the mutated live slot."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request())
        frozen = {
            group: _tensors_bytes(tensors)
            for group, tensors in worker._holemiss_state_tail_snapshots[("req-tail", BOUNDARY)].items()
        }
        live_addresses = [
            base + BLOCK_TABLES[1][TAIL_SLOT_INDEX] * stride
            for base, stride in zip(worker.group_kv_caches_base_addr[1], worker.group_block_stride[1])
        ]
        # The decode advance: every family's tail rows are rewritten in place
        # AFTER the freeze.
        for phys in GDN_PHYSICAL:
            conv, ssm = worker.kv_caches[f"model.layers.{phys}.linear_attn"]
            for row in GROUP_TAIL_ROWS.values():
                conv[row] = torch.full_like(conv[row], 7.0)
                ssm[row] = torch.full_like(ssm[row], 9.0)
        view = _drive_state_group_put(worker, store, group_id=1, physical_layer=0)
        tail_key = view.save_last_block_key
        self.assertIsNotNone(tail_key)
        self.assertEqual([keys for keys, _, _, _ in store.copy_puts], [(tail_key,)])
        self.assertEqual(store.commits, [(tail_key,)])
        buffers = store.copy_puts[0][1][0]
        self.assertNotEqual(buffers, live_addresses, "the put must not address the live slot")
        # The bytes the put actually transferred, read at transfer time.
        payload = store.copy_puts[0][3][0]
        # Layer 0's slice is the object-layout front: the frozen entry list
        # concatenates in the same walk build_addrs addresses.
        self.assertEqual(payload, frozen[1][: sum(store.copy_puts[0][2][0])])
        self.assertNotEqual(payload, _live_slot_bytes(worker, 1)[: len(payload)])

    def test_sources_survive_the_per_row_refcount_zero_crossing(self):
        """T7 fix-3, THE s1e closure: the stored-request count OSCILLATES per
        layer (one add per layer submission, one dec per row), so
        try_finish_and_delete_stored_request fires at EVERY row — group 1's
        FIRST row completion must NOT free the frozen sources (the live run
        freed them there; groups 2/3's rows then found no tensors and the 6
        state keys never persisted). The free fires at the teardown flush."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request())
        frozen = {
            group: _tensors_bytes(tensors)
            for group, tensors in worker._holemiss_state_tail_snapshots[("req-tail", BOUNDARY)].items()
        }
        view1 = _drive_state_group_put(worker, store, group_id=1, physical_layer=0)
        # The zero-crossing fired at group 1's row; the sources must survive.
        self.assertIn(("req-tail", BOUNDARY), worker._holemiss_state_tail_sources)
        view2 = _drive_state_group_put(worker, store, group_id=2, physical_layer=1)
        self.assertEqual(
            [keys for keys, _, _, _ in store.copy_puts],
            [(view1.save_last_block_key,), (view2.save_last_block_key,)],
        )
        # Group 2's row rode the FROZEN tensors (layer 0's slice = the
        # object-layout front of the group's frozen entry list).
        payload = store.copy_puts[1][3][0]
        self.assertEqual(payload, frozen[2][: sum(store.copy_puts[1][2][0])])
        self.assertIn(("req-tail", BOUNDARY), worker._holemiss_state_tail_sources)
        self.assertEqual(worker._holemiss_state_tail_snapshots, {})
        # The free's true owner: the drain-signal teardown flush.
        worker._holemiss_teardown_finished({"req-tail"})
        self.assertIn(("req-tail", BOUNDARY), worker._holemiss_state_tail_sources)
        worker._holemiss_record_state_tail_drain([_tail_save_request()])
        worker._holemiss_teardown_finished(set())
        self.assertEqual(worker._holemiss_state_tail_sources, {})

    def test_substitution_slices_the_layer_entries(self):
        """The per-layer correspondence: layer L's buffers are the snapshot
        entries [offsets[L]:offsets[L+1]] of the group's object-layout walk
        (the same walk the builder's entry arrays derive from)."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request())
        views = _prepare_group_sessions(worker, [_tail_save_request()])
        sources = worker._holemiss_state_tail_sources[("req-tail", BOUNDARY)]
        self.assertEqual(set(sources), set(STATE_GROUPS))
        thread = _sending_thread(worker)
        for group in STATE_GROUPS:
            for layer in range(worker.group_num_layers[group]):
                view = views[group][0]
                worker.layer_save_tasks = [[] for _ in range(NUM_LAYERS)]
                worker._process_save_for_layer_batch(
                    [view], GDN_PHYSICAL[layer], group_id=group, layer_idx_in_group=layer
                )
                task = worker.layer_save_tasks[GDN_PHYSICAL[layer]][0]
                builder = _builder_for(worker, group)
                shared = builder.build_shared(task, is_save=True)
                req_meta = builder.build_addrs(shared, task.layer_idx_in_group)
                bad = thread._holemiss_state_tail_source_rows(req_meta, task)
                self.assertEqual(bad, [])
                offsets = worker.group_layer_cache_entry_offsets[group]
                expected_tensors = sources[group][offsets[layer] : offsets[layer + 1]]
                self.assertEqual(
                    req_meta.all_buffers[0],
                    [int(tensor.data_ptr()) for tensor in expected_tensors],
                )
                self.assertEqual(
                    req_meta.all_sizes[0],
                    [tensor.numel() * tensor.element_size() for tensor in expected_tensors],
                )

    def test_tampered_source_drops_the_row_not_the_live_slot(self):
        """The defensive seam: a source that cannot back the row (a wrong
        entry count — the ordering law broke) is dropped from the req_meta
        and reported — NEVER written from the live slot."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request())
        views = _prepare_group_sessions(worker, [_tail_save_request()])
        tail_key = views[1][0].save_last_block_key
        # Tamper: the group's entry list loses its tail (the count no longer
        # covers layer 0's slice — the ordering law broke).
        worker._holemiss_state_tail_sources[("req-tail", BOUNDARY)][1] = worker._holemiss_state_tail_sources[
            ("req-tail", BOUNDARY)
        ][1][:1]
        worker.layer_save_tasks = [[] for _ in range(NUM_LAYERS)]
        worker._process_save_for_layer_batch([views[1][0]], 0, group_id=1, layer_idx_in_group=0)
        task = worker.layer_save_tasks[0][0]
        builder = _builder_for(worker, 1)
        task.shared_block_data = builder.build_shared(task, is_save=True)
        req_meta = builder.build_addrs(task.shared_block_data, task.layer_idx_in_group)
        self.assertEqual(req_meta.keys, [tail_key])
        thread = _sending_thread(worker)
        with self.assertLogs(vllm_logger, level="ERROR") as logs:
            bad = thread._holemiss_state_tail_source_rows(req_meta, task)
        self.assertEqual(bad, [tail_key])
        # The row is GONE from every parallel list — the ranged put can never
        # see the live-slot buffers for this key.
        self.assertEqual(req_meta.keys, [])
        self.assertEqual(req_meta.all_buffers, [])
        self.assertEqual(req_meta.all_sizes, [])
        self.assertEqual(req_meta.all_offsets, [])
        self.assertEqual(req_meta.block_ids, [])
        self.assertTrue(any("state-tail frozen source missing" in line for line in logs.output))

    def test_fa_page_probe_skips_state_keys(self):
        """The N18 defense: the B43 FA-page content probe must not fingerprint
        a state key (its group-0 page view would read the wrong cache at the
        state's block id)."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request())
        request = _tail_save_request()
        tail = worker._make_layerwise_finished_tail_key(request, 1)
        worker._holemiss_tail_save_probed = set()
        with self.assertNoLogs(vllm_logger, level="INFO"):
            worker.holemiss_tail_save_content_sha(tail[0], BLOCK_TABLES[1][TAIL_SLOT_INDEX])


class TestStateTailMasks(unittest.TestCase):
    """T3 (d): the design-section-f verification — ZERO mask-logic change."""

    def test_state_group_masks_compose_at_the_floor_partial_row_outside(self):
        """The request-level B38 exemption already covers every group: the
        masks compose at the floor (9216) for groups 1-3 too, and the partial
        row rides OUTSIDE the masks (the at-or-beyond-length law) — the
        state group's save task is the partial block alone."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request())
        calls = []
        worker.token_database = SimpleNamespace(
            load_mask=lambda hashes, tokens: None,
            store_mask=lambda aligned_token_len, num_prompt_tokens=None: (
                calls.append((aligned_token_len, num_prompt_tokens)),
                tuple([True] * (aligned_token_len // BLOCK) for _ in range(4)),
            )[1],
        )
        request = _tail_save_request()
        request.store_masks = worker._compute_reachable_store_masks(request)
        self.assertEqual(calls, [(FLOOR, BOUNDARY)])
        for group in range(4):
            mask = request.store_masks[group]
            self.assertEqual(len(mask), FLOOR // BLOCK)
        views = _prepare_group_sessions(worker, [request])
        for group in STATE_GROUPS:
            worker.layer_save_tasks = [[] for _ in range(NUM_LAYERS)]
            worker._process_save_for_layer_batch([views[group][0]], 0, group_id=group, layer_idx_in_group=0)
            tasks = worker.layer_save_tasks[0]
            self.assertEqual(len(tasks), 1)
            ranges = tasks[0].block_ranges
            self.assertEqual(
                [(r.start_block, r.end_block, r.partial_block_index) for r in ranges],
                [(TAIL_SLOT_INDEX, TAIL_SLOT_INDEX, TAIL_SLOT_INDEX)],
            )

    def test_non_tail_unaligned_raise_still_fires(self):
        """The B38 regression guard: a non-tail unaligned hybrid save still
        raises (the exemption keys on the positive signal only)."""
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        request = _tail_save_request(finished_tail_boundary=None)
        with self.assertRaises(ValueError):
            worker._compute_reachable_store_masks(request)


class TestSourcePins(unittest.TestCase):
    """The wiring pins (the inspect.getsource precedent)."""

    def test_substitution_precedes_the_probe_and_the_range_request(self):
        source = inspect.getsource(KVCacheStoreLayerSendingThread._handle_range_layer_tasks)
        substitution = source.index("_holemiss_state_tail_source_rows")
        probe = source.index("_probe_tail_save_content")
        range_request = source.index("_handle_range_request")
        self.assertLess(substitution, probe)
        self.assertLess(probe, range_request)

    def test_the_row_loop_never_frees_the_frozen_sources(self):
        """T7 fix-3 pin: the per-layer refcount zero-crossing is NOT the
        request's put completion (the stored count oscillates per layer;
        s1e freed the sources at group 1's first row and the remaining 47
        state rows dropped). The free lives in the teardown flush only."""
        source = inspect.getsource(KVCacheStoreLayerSendingThread._handle_range_layer_tasks)
        self.assertNotIn("_holemiss_free_state_tail_sources", source)
        self.assertIn("try_finish_and_delete_stored_request", source)

    def test_take_is_consumed_by_the_key_construction(self):
        source = inspect.getsource(KVPoolWorker._make_layerwise_state_tail_key)
        self.assertIn("_holemiss_take_state_tail_snapshots", source)

    def test_teardown_purge_walks_the_sources(self):
        source = inspect.getsource(KVPoolWorker._holemiss_teardown_finished)
        self.assertIn("_holemiss_state_tail_sources", source)

    def test_teardown_purge_frees_the_stashed_sources(self):
        store = _SessionStore()
        worker, _ = _make_session_worker(store)
        _capture(worker, _capture_request())
        tail_request = _tail_save_request()
        _prepare_group_sessions(worker, [tail_request])
        self.assertIn(("req-tail", BOUNDARY), worker._holemiss_state_tail_sources)
        # T7 fix-2: the sources free rides the snapshot deferral — it fires
        # at the drain-signal teardown (this step carried the tail ReqMeta;
        # the put-complete free already took the primary path).
        worker._holemiss_teardown_finished({"req-tail"})
        self.assertIn(("req-tail", BOUNDARY), worker._holemiss_state_tail_sources)
        worker._holemiss_record_state_tail_drain([tail_request])
        worker._holemiss_teardown_finished(set())
        self.assertEqual(worker._holemiss_state_tail_sources, {})


def _live_slot_bytes(worker, group_id: int) -> bytes:
    payload = bytearray()
    for base, length, stride in zip(
        worker.group_kv_caches_base_addr[group_id],
        worker.group_block_len[group_id],
        worker.group_block_stride[group_id],
    ):
        start = base + BLOCK_TABLES[group_id][TAIL_SLOT_INDEX] * stride
        payload.extend((ctypes.c_char * length).from_address(start))
    return bytes(payload)


if __name__ == "__main__":
    unittest.main()
