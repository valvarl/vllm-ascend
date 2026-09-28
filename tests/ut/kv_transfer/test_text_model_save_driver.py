# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Text-model per-layer save driver: AscendStoreConnector.on_kv_cache_written.

Plain-GQA attention (attention_v1) notifies only ``on_kv_cache_written`` after
``reshape_and_cache``; it never calls ``save_kv_layer``. These tests drive the
notification through the real dispatch helper attention_v1 uses (never the
worker save entry directly) and assert the block-key payload path fires: one
layerless object per block, one contiguous range per layer at that layer's
object offset, real bytes copied, and a single commit at the final layer.
"""

import ctypes
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.config import CUDAGraphMode, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash, resolve_kv_cache_block_sizes
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

import vllm_ascend.attention.utils as attention_utils
from tests.ut.kv_offload.utils import create_model_runner_output, create_vllm_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_scheduler, pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import AscendStoreConnector
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (
    attention_transfer_window,
    reset_attention_compute_start_gate,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_layerwise import make_block_key
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import get_layerwise_kv_cache_specs
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import AscendConnectorMetadata, block_hash_to_str

NUM_LAYERS = 4
FULL_BLOCK = 16
NUM_TOKENS = 128
HEAD_SIZE = 8

# Stand-in for the process-global KV transfer group the dispatch helper reads.
_CONNECTED_CONNECTOR = {"connector": None}


class MemoryStore:
    """Mooncake client leaf; copy real bytes and record put/commit order."""

    def __init__(self):
        self.objects = {}
        self.complete = set()
        self.open_reads = set()
        self.open_puts = set()
        self.regions = []
        self.copies = []
        self.events = []
        self.lock = threading.Lock()

    def setup(self, **kwargs):
        return 0

    def register_buffer(self, address, size):
        self.regions.append((address, size))
        return 0

    def batch_is_exist(self, keys):
        return [int(key in self.complete) for key in keys]

    def batch_put_session_start(self, keys, sizes, config):
        for key, size in zip(keys, sizes, strict=True):
            assert key not in self.objects
            self.objects[key] = np.zeros(size, dtype=np.uint8)
        self.open_puts.update(keys)
        return [0] * len(keys)

    def batch_put_session_end(self, keys):
        with self.lock:
            self.events.append(("commit", tuple(keys)))
        assert set(keys) <= self.open_puts
        self.open_puts.difference_update(keys)
        self.complete.update(keys)
        return [0] * len(keys)

    def batch_put_session_revoke(self, keys):
        for key in keys:
            self.objects.pop(key, None)
        self.open_puts.difference_update(keys)
        self.complete.difference_update(keys)
        return [0] * len(keys)

    def batch_get_session_start(self, keys):
        assert set(keys) <= self.complete
        self.open_reads.update(keys)
        return [0] * len(keys)

    def batch_get_session_end(self, keys):
        self.open_reads.difference_update(keys)
        return 0

    def batch_put_from_multi_buffer_ranges(self, keys, buffers, sizes, offsets):
        with self.lock:
            self.events.append(
                ("put", tuple(keys), tuple(tuple(row) for row in sizes), tuple(tuple(row) for row in offsets))
            )
        return self.copy_ranges(keys, buffers, sizes, offsets, saving=True)

    def batch_get_into_multi_buffer_ranges(self, keys, buffers, sizes, offsets):
        return self.copy_ranges(keys, buffers, sizes, offsets, saving=False)

    def copy_ranges(self, keys, buffers, sizes, offsets, *, saving):
        with self.lock:
            for key, addresses, lengths, starts in zip(keys, buffers, sizes, offsets, strict=True):
                assert key in (self.open_puts if saving else self.open_reads)
                for address, size, offset in zip(addresses, lengths, starts, strict=True):
                    assert any(start <= address < address + size <= start + length for start, length in self.regions)
                    assert 0 <= offset < offset + size <= self.objects[key].size
                    remote = self.objects[key].ctypes.data + offset
                    source, dest = (address, remote) if saving else (remote, address)
                    ctypes.memmove(dest, source, size)
                    self.copies.append((saving, address, size, key))
        return [sum(row) for row in sizes]


def aligned_tensor(shape):
    alignment = 2 * 1024 * 1024
    size = int(np.prod(shape)) * 4
    backing = np.zeros(size + alignment, dtype=np.uint8)
    offset = -backing.ctypes.data % alignment
    return torch.from_numpy(backing[offset : offset + size]).view(torch.float32).view(shape)


def payload(tokens, end, layer, plane):
    # Independent of Store keys, masks, hashes and physical allocation.
    return (sum(tokens[:end]) + layer * 13 + plane * 7) % 97 + 1


def wait_until(predicate, pool):
    deadline = time.monotonic() + 5
    while not predicate():
        pool.kv_send_thread.raise_if_failed()
        pool.kv_recv_thread.raise_if_failed()
        assert time.monotonic() < deadline, "Transfer did not complete"
        time.sleep(0.001)


@pytest.fixture
def memory_store(monkeypatch, tmp_path):
    store = MemoryStore()
    monkeypatch.setitem(
        sys.modules,
        "mooncake.store",
        SimpleNamespace(MooncakeDistributedStore=lambda: store, ReplicateConfig=SimpleNamespace),
    )
    config_path = tmp_path / "mooncake.json"
    config_path.write_text(
        json.dumps({"protocol": "ascend", "metadata_server": "test", "master_server_address": "test"})
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ASCEND_ENABLE_USE_FABRIC_MEM", "0")
    monkeypatch.setenv("ASCEND_GLOBAL_RESOURCE_CONFIG", "{}")
    monkeypatch.setattr(pool_worker, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(pool_worker, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pool_worker, "get_pcp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(pool_worker, "get_decode_context_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pool_worker, "get_decode_context_model_parallel_rank", lambda: 0)
    # attention_v1 reaches the connector through these module-level hooks.
    monkeypatch.setattr(attention_utils, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(attention_utils, "is_v1_kv_transfer_group", lambda: True)
    monkeypatch.setattr(attention_utils, "get_kv_transfer_group", lambda: _CONNECTED_CONNECTOR["connector"])
    _CONNECTED_CONNECTOR["connector"] = None
    # A stale gate from a previous test would fire foreign callbacks inside
    # attention_transfer_window; start from a clean one.
    reset_attention_compute_start_gate()
    register_all_kvcache_specs(None)
    init_none_hash(sha256)
    return store


def _text_plan():
    full = FullAttentionSpec(block_size=FULL_BLOCK, num_kv_heads=1, head_size=HEAD_SIZE, dtype=torch.float32)
    return KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([f"model.layers.{layer}.attn" for layer in range(NUM_LAYERS)], full)],
    )


def _text_config():
    config = create_vllm_config(max_num_batched_tokens=256, block_size=FULL_BLOCK)
    config.model_config.hf_text_config.num_hidden_layers = NUM_LAYERS
    config.model_config.model_arch_config.total_num_hidden_layers = NUM_LAYERS
    config.model_config.max_model_len = 512
    config.scheduler_config.max_model_len = 512
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.cache_config.num_gpu_blocks = 128
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_connector_module_path="vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
        kv_role="kv_both",
        kv_connector_extra_config={"backend": "mooncake", "use_layerwise": True},
    )
    return config


def _build_engine():
    plan = _text_plan()
    config = _text_config()
    block_size, hash_size = resolve_kv_cache_block_sizes(plan, config)
    worker = KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, plan)
    specs = get_layerwise_kv_cache_specs(plan)
    caches = {
        name: tuple(aligned_tensor((plan.num_blocks, spec.block_size, 1, spec.head_size)) for _ in range(2))
        for name, spec in specs.items()
    }
    scheduler = Scheduler(
        config,
        plan,
        StructuredOutputManager(config),
        block_size=block_size,
        hash_block_size=hash_size,
    )
    return config, plan, worker, specs, caches, scheduler, hash_size


def _layer_geometry(pool):
    entry_offsets = pool.group_layer_cache_entry_offsets[0]
    block_lens = pool.group_block_len[0]
    offsets = [sum(block_lens[: entry_offsets[layer]]) for layer in range(NUM_LAYERS)]
    sizes = [sum(block_lens[entry_offsets[layer] : entry_offsets[layer + 1]]) for layer in range(NUM_LAYERS)]
    return offsets, sizes


def _layer_segments(pool):
    """Per layer, the (size, offset) object segments its payload batch writes
    (one segment per cache plane; K and V live in separate device buffers)."""
    entry_offsets = pool.group_layer_cache_entry_offsets[0]
    block_lens = pool.group_block_len[0]
    return [
        [
            (block_lens[entry], sum(block_lens[:entry]))
            for entry in range(entry_offsets[layer], entry_offsets[layer + 1])
        ]
        for layer in range(NUM_LAYERS)
    ]


def _expected_keys(config, request, pool):
    model_name = config.model_config.model.split("/")[-1]
    return {
        make_block_key(model_name, block_hash_to_str(request.block_hashes[index]), pool.head_or_tp_rank)
        for index in range(NUM_TOKENS // FULL_BLOCK)
    }


def _drive_request(worker, scheduler, req_id, tokens, max_tokens, specs, caches, *, named_layers, flags_out=None):
    """Run one request the way attention_v1 does: per layer, scatter the KV
    bytes, notify ``on_kv_cache_written`` through the real dispatch helper,
    and run the attention op inside the transfer window. ``named_layers``
    additionally drives the mla_v1/sfa_v1/dsa_v1 layer-end save call, which
    the connector must deduplicate against the notification. ``flags_out``
    records ``(scheduled_tokens, has_sync_kv_loads)`` for every step,
    including the terminal empty schedule."""
    request = Request(
        request_id=req_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        pooling_params=None,
        block_hasher=get_request_block_hasher(FULL_BLOCK, sha256),
    )
    scheduler.add_request(request)
    pool = worker.connector_worker
    _CONNECTED_CONNECTOR["connector"] = worker
    prefill_blocks = None
    for _ in range(NUM_TOKENS + max_tokens + 8):
        output = scheduler.schedule()
        meta = output.kv_connector_metadata
        worker.handle_preemptions(meta)
        worker.bind_connector_metadata(meta)
        scheduled = output.num_scheduled_tokens.get(req_id, 0)
        if flags_out is not None:
            flags_out.append((scheduled, output.has_sync_kv_loads))
        worker.start_load_kv(SimpleNamespace() if scheduled else None)
        if scheduled:
            end = request.num_computed_tokens
            start = end - scheduled
            blocks = scheduler.kv_cache_manager.get_blocks(req_id).get_block_ids()[0]
            if scheduled == len(tokens):
                prefill_blocks = list(blocks)
            for layer in range(NUM_LAYERS):
                name = f"model.layers.{layer}.attn"
                spec = specs[name]
                # reshape_and_cache: fill every block this step completes.
                for index, bid in enumerate(blocks):
                    boundary = (index + 1) * spec.block_size
                    if bid and start < boundary <= end:
                        for plane, cache in enumerate(caches[name]):
                            cache[bid].fill_(payload(request.all_token_ids, boundary, layer, plane))
                if named_layers:
                    attention_utils.notify_kv_cache_written(name)
                else:
                    attention_utils.notify_kv_cache_written()
                with attention_transfer_window():
                    pass
                if named_layers:
                    worker.save_kv_layer(name, caches[name], None)
            wait_until(lambda: pool.kv_send_thread.request_queue.unfinished_tasks == 0, pool)
        sending, recving = worker.get_finished(output.finished_req_ids)
        result = create_model_runner_output([request] if scheduled else [])
        if scheduled and request.num_computed_tokens < len(tokens):
            result.sampled_token_ids = [[]]
        result.kv_connector_output = KVConnectorOutput(
            finished_sending=sending,
            finished_recving=recving,
            kv_connector_worker_meta=worker.build_connector_worker_meta(),
        )
        scheduler.update_from_output(output, result)
        worker.clear_connector_metadata()
        if req_id not in scheduler.requests:
            terminal = scheduler.schedule()
            if flags_out is not None:
                flags_out.append((0, terminal.has_sync_kv_loads))
            worker.bind_connector_metadata(terminal.kv_connector_metadata)
            worker.get_finished(terminal.finished_req_ids)
            worker.clear_connector_metadata()
            break
    else:
        pytest.fail("Request did not finish within the step budget")
    assert request.num_computed_tokens >= len(tokens)
    return request, prefill_blocks


def _assert_block_key_save(store, config, request, pool, caches, prefill_blocks, expected_keys):
    assert expected_keys <= set(store.complete)
    offsets, sizes = _layer_geometry(pool)
    object_size = sum(pool.group_block_len[0])
    model_name = config.model_config.model.split("/")[-1]
    for index, bid in enumerate(prefill_blocks):
        key = make_block_key(model_name, block_hash_to_str(request.block_hashes[index]), pool.head_or_tp_rank)
        obj = store.objects[key]
        assert obj.size == object_size, "one layerless object per block (D7)"
        for layer in range(NUM_LAYERS):
            name = f"model.layers.{layer}.attn"
            expected = b"".join(plane[bid].numpy().tobytes() for plane in caches[name])
            assert obj[offsets[layer] : offsets[layer] + sizes[layer]].tobytes() == expected, (
                f"layer {layer} payload landed at the wrong object offset for block {index}"
            )
    for key in expected_keys:
        key_copies = [copy for copy in store.copies if copy[3] == key]
        expected_segments = sum(len(layer_segments) for layer_segments in _layer_segments(pool))
        assert len(key_copies) == expected_segments, "one range segment per layer and plane per block object"


def _assert_commit_at_final_layer(store, pool, expected_keys, events_before):
    step_events = store.events[events_before:]
    puts = [event for event in step_events if event[0] == "put"]
    commits = [event for event in step_events if event[0] == "commit"]
    assert len(puts) == NUM_LAYERS, f"expected one payload batch per layer, got {len(puts)}"
    assert len(commits) == 1, "commit must be issued exactly once per step"
    assert set(commits[0][1]) == expected_keys
    assert step_events[-1][0] == "commit", "commit must follow the final layer's payload batch"
    # The send queue is FIFO: the i-th payload batch is layer i's, writing
    # that layer's (size, offset) segments into every block object.
    for event, layer_segments in zip(puts, _layer_segments(pool), strict=True):
        _, keys, sizes, offsets = event
        assert set(keys) == expected_keys
        for row_sizes, row_offsets in zip(sizes, offsets):
            assert list(zip(row_sizes, row_offsets)) == layer_segments
    return puts


@pytest.fixture
def text_engine(memory_store):
    config, plan, worker, specs, caches, scheduler, hash_size = _build_engine()
    pool = worker.connector_worker
    worker.register_kv_caches(caches)
    assert worker.use_layerwise is True
    assert pool.use_block_key_layerwise is True
    assert pool.block_key_hybrid is False
    assert hash_size == FULL_BLOCK
    yield config, plan, worker, specs, caches, scheduler, pool


def test_attention_v1_notification_drives_block_key_save(text_engine, memory_store):
    """The exact attention_v1 shape: nameless notify, no layer-end save call.

    Without the connector driver this is the B18 signature: sessions open
    (PutStart metadata) and no payload or commit is ever issued."""
    config, plan, worker, specs, caches, scheduler, pool = text_engine
    events_before = len(memory_store.events)

    request, prefill_blocks = _drive_request(
        worker,
        scheduler,
        "text-notify",
        list(range(NUM_TOKENS)),
        max_tokens=2,
        specs=specs,
        caches=caches,
        named_layers=False,
    )
    assert prefill_blocks is not None and all(prefill_blocks)
    expected_keys = _expected_keys(config, request, pool)
    assert set(memory_store.complete) == expected_keys
    _assert_block_key_save(memory_store, config, request, pool, caches, prefill_blocks, expected_keys)
    _assert_commit_at_final_layer(memory_store, pool, expected_keys, events_before)

    assert memory_store.open_puts == set() and memory_store.open_reads == set()
    assert not pool._put_started_keys
    assert not pool.get_block_ids_with_load_errors()


def test_named_notification_with_layer_end_save_deduplicates(text_engine, memory_store):
    """The mla_v1/sfa_v1/dsa_v1 shape: name-bearing notify at scatter time
    plus save_kv_layer at layer end must drive exactly one submission."""
    config, plan, worker, specs, caches, scheduler, pool = text_engine
    request, prefill_blocks = _drive_request(
        worker,
        scheduler,
        "mla-shape",
        list(range(NUM_TOKENS)),
        max_tokens=2,
        specs=specs,
        caches=caches,
        named_layers=True,
    )
    events_before = 0
    expected_keys = _expected_keys(config, request, pool)
    assert set(memory_store.complete) == expected_keys
    _assert_block_key_save(memory_store, config, request, pool, caches, prefill_blocks, expected_keys)
    _assert_commit_at_final_layer(memory_store, pool, expected_keys, events_before)
    assert memory_store.open_puts == set() and memory_store.open_reads == set()


def test_second_request_saves_fresh_prefix(text_engine, memory_store):
    """Both notification styles across two requests and a prefix-cache reset:
    fresh hashes get fresh layerless objects; decode steps save nothing."""
    config, plan, worker, specs, caches, scheduler, pool = text_engine
    _drive_request(
        worker,
        scheduler,
        "first",
        list(range(NUM_TOKENS)),
        max_tokens=2,
        specs=specs,
        caches=caches,
        named_layers=False,
    )
    first_keys = set(memory_store.complete)
    assert first_keys
    assert scheduler.reset_prefix_cache()
    for values in caches.values():
        for cache in values:
            cache.fill_(-1)
    events_before = len(memory_store.events)

    request, prefill_blocks = _drive_request(
        worker,
        scheduler,
        "second",
        list(range(200, 200 + NUM_TOKENS)),
        max_tokens=2,
        specs=specs,
        caches=caches,
        named_layers=True,
    )
    expected_keys = _expected_keys(config, request, pool)
    assert not (expected_keys & first_keys), "fresh prefix must produce fresh block keys"
    assert set(memory_store.complete) == first_keys | expected_keys
    _assert_block_key_save(memory_store, config, request, pool, caches, prefill_blocks, expected_keys)
    _assert_commit_at_final_layer(memory_store, pool, expected_keys, events_before)
    assert memory_store.open_puts == set() and memory_store.open_reads == set()


def test_save_bearing_step_forces_sync_loads_flag(text_engine, memory_store):
    """B19 ordering: a save-bearing step with no connector loads must raise
    has_sync_kv_loads so the model-runner mixin runs start_load_kv PRE-forward
    (the scheduler itself only raises the flag for external computed tokens,
    i.e. sync loads; a save-only populate step has none)."""
    config, plan, worker, specs, caches, scheduler, pool = text_engine
    request = Request(
        request_id="sync-flag",
        prompt_token_ids=list(range(NUM_TOKENS)),
        sampling_params=SamplingParams(max_tokens=2),
        pooling_params=None,
        block_hasher=get_request_block_hasher(FULL_BLOCK, sha256),
    )
    scheduler.add_request(request)
    output = scheduler.schedule()
    meta = output.kv_connector_metadata

    # Attribution: the step is save-bearing and carries no load spec, so the
    # scheduler's own external-token condition cannot be the flag's source.
    assert any(item.can_save for item in meta.requests)
    assert not any(item.load_spec is not None and item.load_spec.can_load for item in meta.requests)
    assert output.has_sync_kv_loads is True


def test_no_save_work_step_keeps_deferred_flag(text_engine, memory_store):
    """Steps without connector save work keep the deferred start: decode steps
    carry no save spec (D-T12.6) and the terminal empty schedule has no
    metadata requests at all — none of them may force the sync path."""
    config, plan, worker, specs, caches, scheduler, pool = text_engine
    flags = []
    _drive_request(
        worker,
        scheduler,
        "deferred-flag",
        list(range(NUM_TOKENS)),
        max_tokens=2,
        specs=specs,
        caches=caches,
        named_layers=False,
        flags_out=flags,
    )
    scheduled_flags = [flag for scheduled, flag in flags if scheduled]
    assert scheduled_flags[0] is True, "the populate step is save-bearing and must force the sync start"
    assert all(flag is False for flag in scheduled_flags[1:]), "decode steps must stay deferred"
    assert flags[-1] == (0, False), "the terminal empty schedule must stay deferred"


def test_start_load_kv_gives_each_step_fresh_task_lists(text_engine):
    """The worker swaps in fresh per-layer task lists on every start_load_kv,
    so a during-forward save driver can never observe a previous step's tasks
    once the step ordering is pre-forward (the b19 cross-step hazard)."""
    config, plan, worker, specs, caches, scheduler, pool = text_engine
    empty_meta = AscendConnectorMetadata(set(), set())
    first = pool.layer_save_tasks
    pool.start_load_kv(empty_meta)
    second = pool.layer_save_tasks
    assert second is not first
    pool.start_load_kv(empty_meta)
    assert pool.layer_save_tasks is not second
    assert pool.layer_save_tasks == [[] for _ in range(pool.num_layers)]


def test_sync_loads_forcing_pinned_in_build_connector_meta():
    """Composition pin: the flag write stays inside build_connector_meta and
    stays guarded by the save-bearing predicate (no relocation/removal drift,
    no unconditional forcing)."""
    source = Path(pool_scheduler.__file__).read_text()
    fn_start = source.index("    def build_connector_meta(self, scheduler_output: SchedulerOutput)")
    fn_end = source.index("    def get_sending_event_id(", fn_start)
    body = source[fn_start:fn_end]
    guard = body.index("if self._step_carries_layerwise_save(meta):")
    forcing = body.index("scheduler_output.has_sync_kv_loads = True")
    assert 0 < guard < forcing < body.rindex("return meta")
    helper_start = source.index("    def _step_carries_layerwise_save(")
    helper = source[helper_start : source.index("\n    def ", helper_start + len("    def _step"))]
    assert "self.use_layerwise" in helper, "the predicate must stay scoped to the layerwise path"
    assert "can_save" in helper.strip().splitlines()[-1], "the predicate must mirror the worker's save gating"


class _StubWorker:
    def __init__(self, num_layers=NUM_LAYERS, hybrid=False):
        self.current_layer = 0
        self.num_layers = num_layers
        self.block_key_hybrid = hybrid
        self.save_calls = []
        self.validated = []
        self.load_calls = []

    def save_kv_layer(self, metadata):
        self.save_calls.append(self.current_layer)
        if self.current_layer < self.num_layers:
            self.current_layer += 1

    def validate_notified_save_layer(self, layer_name):
        self.validated.append(layer_name)

    def start_load_kv(self, metadata):
        self.load_calls.append(metadata)


def _stub_connector(worker, kv_role="kv_both", use_layerwise=True, metadata=...):
    connector = AscendStoreConnector.__new__(AscendStoreConnector)
    connector.use_layerwise = use_layerwise
    connector.kv_role = kv_role
    connector.consumer_is_to_put = False
    connector.connector_worker = worker
    connector._connector_metadata = SimpleNamespace(requests=[]) if metadata is ... else metadata
    return connector


def test_notification_gates():
    worker = _StubWorker()
    _stub_connector(worker, kv_role="kv_consumer").on_kv_cache_written()
    assert worker.save_calls == []
    connector = _stub_connector(worker)
    connector._hr_hook_active = True
    connector.on_kv_cache_written()
    assert worker.save_calls == []
    _stub_connector(worker, use_layerwise=False).on_kv_cache_written()
    assert worker.save_calls == []
    _stub_connector(worker, metadata=None).on_kv_cache_written()
    assert worker.save_calls == []
    assert worker.validated == []


def test_hybrid_layouts_keep_their_own_driver():
    """The attention-entry gate (pool_worker) drives hybrid saves; the
    notification hook must not double-fire there."""
    worker = _StubWorker(hybrid=True)
    _stub_connector(worker).on_kv_cache_written()
    _stub_connector(worker).on_kv_cache_written("model.layers.0.attn")
    assert worker.save_calls == []
    assert worker.validated == []


def test_notification_dedups_layer_end_save():
    worker = _StubWorker()
    connector = _stub_connector(worker)

    # attention_v1 shape: notification only.
    for _ in range(NUM_LAYERS):
        connector.on_kv_cache_written()
    assert worker.save_calls == [0, 1, 2, 3]

    # mla/sfa/dsa shape: notification then layer-end save, one drive total.
    worker = _StubWorker()
    connector = _stub_connector(worker)
    for layer in range(NUM_LAYERS):
        connector.on_kv_cache_written(f"model.layers.{layer}.attn")
        connector.save_kv_layer(f"model.layers.{layer}.attn", None, None)
    assert worker.save_calls == [0, 1, 2, 3]
    assert worker.validated == [f"model.layers.{layer}.attn" for layer in range(NUM_LAYERS)]


def test_layer_end_save_without_notification_still_drives():
    worker = _StubWorker()
    connector = _stub_connector(worker)
    connector.save_kv_layer("model.layers.0.attn", None, None)
    assert worker.save_calls == [0]


def test_start_load_kv_resets_dedup_marker():
    worker = _StubWorker()
    connector = _stub_connector(worker)
    connector.on_kv_cache_written()
    assert worker.save_calls == [0]
    connector.start_load_kv(SimpleNamespace())
    connector.save_kv_layer("model.layers.0.attn", None, None)
    assert worker.save_calls == [0, 1]


def test_nameless_notification_skips_mapping_validation():
    worker = _StubWorker()
    connector = _stub_connector(worker)
    connector.on_kv_cache_written()
    assert worker.validated == []
    assert worker.save_calls == [0]


def _validation_pool(current_layer, num_layers=NUM_LAYERS):
    pool = pool_worker.KVPoolWorker.__new__(pool_worker.KVPoolWorker)
    pool.current_layer = current_layer
    pool.num_layers = num_layers
    pool.hf_config = SimpleNamespace(num_hidden_layers=num_layers)
    pool._global_to_local_layer = {global_id: local for local, global_id in enumerate(range(num_layers))}
    return pool


def test_validate_notified_save_layer_accepts_current_layer():
    pool = _validation_pool(2)
    pool.validate_notified_save_layer("model.layers.2.self_attn.attn")
    pool.validate_notified_save_layer("model.layers.2.attn.indexer.k_cache")


def test_validate_notified_save_layer_refuses_wrong_layer():
    pool = _validation_pool(2)
    with pytest.raises(RuntimeError, match="maps to layer 3"):
        pool.validate_notified_save_layer("model.layers.3.self_attn.attn")


def test_validate_notified_save_layer_past_end_is_noop():
    pool = _validation_pool(NUM_LAYERS)
    pool.validate_notified_save_layer("model.layers.0.self_attn.attn")
