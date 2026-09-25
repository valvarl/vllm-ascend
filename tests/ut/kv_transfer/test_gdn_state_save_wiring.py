# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN boundary-state save wiring: upstream #17287 adopt + default-on gaps.

The upstream layerwise save path (per-group put sessions + post-compute
``save_kv_layer`` payload ranges for align-mode Mamba groups) is adopted
as-is; the only new behavior under test is the default-on resolution of
``use_layerwise`` for hybrid align-mode mooncake layouts, plus the D7
key/EIO contract pins for the GDN state objects (groups 1..N).
"""

import ctypes
import json
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.config import CUDAGraphMode, KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash, resolve_kv_cache_block_sizes
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import create_model_runner_output, create_vllm_config
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import (
    ascend_store_connector as connector_module,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_scheduler as scheduler_module
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_worker
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    gdn_boundary_state_persistence_default,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.attention_fence import (
    attention_transfer_window,
    record_attention_compute_start,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_layerwise import (
    bind_layerwise_keys,
    hybrid_block_key,
    hybrid_layout_id,
    validate_runtime,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import get_layerwise_kv_cache_specs
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import block_hash_to_str
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import KVPoolScheduler

NUM_LAYERS = 4
FULL_BLOCK = 16
MAMBA_BLOCK = 32
NUM_TOKENS = 128


class MemoryStore:
    """Mooncake client leaf; copy real bytes through session/range APIs."""

    def __init__(self):
        self.objects = {}
        self.complete = set()
        self.open_reads = set()
        self.open_puts = set()
        self.regions = []
        self.copies = []
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


def wait_until(predicate, pool):
    deadline = time.monotonic() + 5
    while not predicate():
        pool.kv_send_thread.raise_if_failed()
        pool.kv_recv_thread.raise_if_failed()
        assert time.monotonic() < deadline, "Transfer did not complete"
        time.sleep(0.001)


def _spec_pair(mamba_cache_mode="align", recurrent_type=MambaAttentionBackendEnum.GDN_ATTN):
    full = FullAttentionSpec(block_size=FULL_BLOCK, num_kv_heads=1, head_size=8, dtype=torch.float32)
    recurrent = MambaSpec(
        block_size=MAMBA_BLOCK,
        shapes=((8,), (8,)),
        dtypes=(torch.float32, torch.float32),
        mamba_type=recurrent_type,
        mamba_cache_mode=mamba_cache_mode,
    )
    return full, recurrent


def _hybrid_plan(mamba_cache_mode="align"):
    full, recurrent = _spec_pair(mamba_cache_mode)
    return KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.0.attn", "model.layers.2.attn"], full),
            KVCacheGroupSpec(["model.layers.1.attn", "model.layers.3.attn"], recurrent),
        ],
    )


def _worker_plan(plan):
    return KVCacheConfig(
        num_blocks=plan.num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                group.layer_names,
                UniformTypeKVCacheSpecs(
                    block_size=group.kv_cache_spec.block_size,
                    kv_cache_specs={name: group.kv_cache_spec for name in group.layer_names},
                ),
            )
            for group in plan.kv_cache_groups
        ],
    )


def _hybrid_config(extra_config):
    config = create_vllm_config(max_num_batched_tokens=FULL_BLOCK * 2, block_size=FULL_BLOCK)
    config.model_config.hf_text_config.num_hidden_layers = NUM_LAYERS
    config.model_config.model_arch_config.total_num_hidden_layers = NUM_LAYERS
    config.model_config.max_model_len = 512
    config.scheduler_config.max_model_len = 512
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.num_gpu_blocks = 128
    config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="AscendStoreConnector",
        kv_connector_module_path="vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )
    return config


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
    monkeypatch.setattr(
        connector_module, "LookupKeyServer", lambda *args, **kwargs: SimpleNamespace(close=lambda: None)
    )
    monkeypatch.setattr(
        scheduler_module,
        "LookupKeyClient",
        lambda vllm_config: SimpleNamespace(lookup=lambda *args, **kwargs: 0, close=lambda: None),
    )
    register_all_kvcache_specs(None)
    init_none_hash(sha256)
    return store


def _drive_request(worker, scheduler, req_id, tokens, max_tokens, specs, caches):
    request = Request(
        request_id=req_id,
        prompt_token_ids=tokens,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        pooling_params=None,
        block_hasher=get_request_block_hasher(FULL_BLOCK, sha256),
    )
    scheduler.add_request(request)
    pool = worker.connector_worker
    layerwise = worker.use_layerwise
    for _ in range(NUM_TOKENS + max_tokens + 8):
        output = scheduler.schedule()
        meta = output.kv_connector_metadata
        worker.handle_preemptions(meta)
        worker.bind_connector_metadata(meta)
        scheduled = output.num_scheduled_tokens.get(req_id, 0)
        worker.start_load_kv(SimpleNamespace() if scheduled else None)
        if scheduled:
            for layer in range(NUM_LAYERS):
                name = f"model.layers.{layer}.attn"
                spec = specs[name]
                worker.wait_for_layer_load(name)
                if layerwise:
                    if isinstance(spec, MambaSpec):
                        record_attention_compute_start()
                        if layer in pool._attention_saved_layers:
                            wait_until(pool.layer_save_finished_events[layer].is_set, pool)
                    else:
                        with attention_transfer_window():
                            assert layer in pool._attention_saved_layers
                worker.save_kv_layer(name, caches[name], None)
            if layerwise:
                assert pool.kv_send_thread.request_queue.unfinished_tasks == 0
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
            worker.bind_connector_metadata(terminal.kv_connector_metadata)
            worker.get_finished(terminal.finished_req_ids)
            worker.clear_connector_metadata()
            break
    else:
        pytest.fail("Request did not finish within the step budget")
    assert request.num_computed_tokens >= len(tokens)
    return request


def _build_engine(extra_config, plan=None):
    plan = plan or _hybrid_plan()
    config = _hybrid_config(extra_config)
    block_size, hash_size = resolve_kv_cache_block_sizes(plan, config)
    worker = KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, _worker_plan(plan))
    specs = get_layerwise_kv_cache_specs(_worker_plan(plan))
    caches = {
        name: tuple(aligned_tensor((plan.num_blocks, *shape)) for shape in spec.shapes)
        if isinstance(spec, MambaSpec)
        else tuple(aligned_tensor((plan.num_blocks, spec.block_size, 1, spec.head_size)) for _ in range(2))
        for name, spec in specs.items()
    }
    scheduler = Scheduler(
        config,
        plan,
        StructuredOutputManager(config),
        block_size=block_size,
        hash_block_size=hash_size,
    )
    return config, plan, worker, specs, caches, scheduler


def _state_keys(store, group):
    marker = f"@group:{group}@"
    return sorted(key for key in store.complete if marker in key)


def test_gdn_boundary_state_persistence_default_matrix():
    full, align_mamba = _spec_pair("align")
    _, non_align_mamba = _spec_pair("none")
    swa = SlidingWindowSpec(
        block_size=MAMBA_BLOCK, num_kv_heads=1, head_size=8, dtype=torch.float32, sliding_window=MAMBA_BLOCK
    )
    single_group_text = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["model.layers.0.attn"], full)],
    )
    single_group_mamba = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["model.layers.0.attn"], align_mamba)],
    )
    hybrid_align = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.0.attn"], full),
            KVCacheGroupSpec(["model.layers.1.attn"], align_mamba),
        ],
    )
    hybrid_wrapped = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.0.attn"], full),
            KVCacheGroupSpec(
                ["model.layers.1.attn"],
                UniformTypeKVCacheSpecs(
                    block_size=align_mamba.block_size, kv_cache_specs={"model.layers.1.attn": align_mamba}
                ),
            ),
        ],
    )
    hybrid_non_align = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.0.attn"], full),
            KVCacheGroupSpec(["model.layers.1.attn"], non_align_mamba),
        ],
    )
    hybrid_swa = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["model.layers.0.attn"], full),
            KVCacheGroupSpec(["model.layers.1.attn"], swa),
        ],
    )
    assert gdn_boundary_state_persistence_default(None) is False
    assert gdn_boundary_state_persistence_default(single_group_text) is False
    assert gdn_boundary_state_persistence_default(single_group_mamba) is False
    assert gdn_boundary_state_persistence_default(hybrid_align) is True
    assert gdn_boundary_state_persistence_default(hybrid_wrapped) is True
    assert gdn_boundary_state_persistence_default(hybrid_non_align) is False
    assert gdn_boundary_state_persistence_default(hybrid_swa) is False


def test_validate_runtime_accepts_recurrent_state():
    validate_runtime(use_hybrid=True, has_recurrent_state=True, tp_mismatch=False)
    with pytest.raises(ValueError, match="TP mismatch"):
        validate_runtime(use_hybrid=True, has_recurrent_state=True, tp_mismatch=True)


def test_infer_mamba_groups_rejects_non_align():
    non_align_ctx = SimpleNamespace(kv_cache_config=_hybrid_plan("none"), use_hybrid=True)
    with pytest.raises(NotImplementedError, match="mamba_cache_mode='align'"):
        KVPoolScheduler._infer_mamba_groups(non_align_ctx)
    align_ctx = SimpleNamespace(kv_cache_config=_hybrid_plan("align"), use_hybrid=True)
    assert KVPoolScheduler._infer_mamba_groups(align_ctx) == {1: 0}


def test_gdn_state_save_default_on_wiring(memory_store):
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import MooncakeBackend
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
        KVCacheStoreLayerRecvingThread,
        KVCacheStoreLayerSendingThread,
    )

    config, plan, worker, specs, caches, scheduler = _build_engine({"backend": "mooncake"})
    pool = worker.connector_worker
    worker.register_kv_caches(caches)

    assert worker.use_layerwise is True
    assert isinstance(pool.m_store, MooncakeBackend)
    assert isinstance(pool.kv_send_thread, KVCacheStoreLayerSendingThread)
    assert isinstance(pool.kv_recv_thread, KVCacheStoreLayerRecvingThread)
    assert pool.block_key_hybrid
    assert pool.grouped_block_size == [FULL_BLOCK, MAMBA_BLOCK]

    request = _drive_request(
        worker, scheduler, "default-on", list(range(NUM_TOKENS)), max_tokens=MAMBA_BLOCK + 8, specs=specs, caches=caches
    )

    fa_keys = _state_keys(memory_store, 0)
    state_keys = _state_keys(memory_store, 1)
    assert len(fa_keys) == NUM_TOKENS // FULL_BLOCK
    assert len(state_keys) == NUM_TOKENS // MAMBA_BLOCK
    assert memory_store.open_puts == set() and memory_store.open_reads == set()

    layout = hybrid_layout_id(_worker_plan(plan), 1)
    model_name = config.model_config.model.split("/")[-1]
    scale = MAMBA_BLOCK // FULL_BLOCK
    expected_state_keys = {
        hybrid_block_key(
            model_name,
            layout,
            1,
            MAMBA_BLOCK,
            block_hash_to_str(request.block_hashes[(index + 1) * scale - 1]),
            pool.head_or_tp_rank,
        )
        for index in range(NUM_TOKENS // MAMBA_BLOCK)
    }
    assert set(state_keys) == expected_state_keys

    decode_crossed = request.num_computed_tokens - NUM_TOKENS >= MAMBA_BLOCK
    assert decode_crossed, "decode phase did not cross an align boundary"
    assert len(request.block_hashes) >= (NUM_TOKENS + MAMBA_BLOCK) // FULL_BLOCK
    assert len(_state_keys(memory_store, 1)) == NUM_TOKENS // MAMBA_BLOCK


def test_gdn_state_save_wiring_off_no_state_keys(memory_store):
    config, plan, worker, specs, caches, scheduler = _build_engine({"backend": "mooncake", "use_layerwise": False})
    pool = worker.connector_worker
    worker.register_kv_caches(caches)

    assert worker.use_layerwise is False
    assert worker.use_block_key_layerwise is False
    assert pool.use_layerwise is False
    assert pool.block_key_hybrid is False

    _drive_request(worker, scheduler, "wiring-off", list(range(NUM_TOKENS)), max_tokens=2, specs=specs, caches=caches)

    assert memory_store.objects == {}
    assert memory_store.complete == set()
    assert memory_store.copies == []


def test_default_on_refuses_full_cudagraphs(memory_store):
    config, plan = _hybrid_config({"backend": "mooncake"}), _hybrid_plan()
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    with pytest.raises(ValueError, match="use_layerwise"):
        KVConnectorFactory.create_connector(config, KVConnectorRole.WORKER, _worker_plan(plan))

    explicit = _hybrid_config({"backend": "mooncake", "use_layerwise": True})
    explicit.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    created = KVConnectorFactory.create_connector(explicit, KVConnectorRole.WORKER, _worker_plan(plan))
    assert created.use_layerwise is True


def test_state_keys_never_eio_armable():
    plan = _hybrid_plan()
    builder = bind_layerwise_keys(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_size=1, tensor_parallel_size=1, rank=0)
        ),
        kv_cache_config=plan,
        model_name="m",
        use_hybrid=True,
        grouped_block_size=[FULL_BLOCK, MAMBA_BLOCK],
    )
    pool = pool_worker.KVPoolWorker.__new__(pool_worker.KVPoolWorker)
    pool.use_hybrid = True
    pool.use_layerwise = True
    pool.use_block_key_layerwise = True
    pool.layerwise_offload = False
    pool.num_kv_cache_groups = 2
    pool.grouped_block_size = [FULL_BLOCK, MAMBA_BLOCK]
    pool.hash_block_size = FULL_BLOCK
    pool.head_or_tp_rank = 0
    pool.pp_rank = 0
    pool.layerwise_keys = builder
    request = SimpleNamespace(
        req_id="eio-contract",
        block_hashes=[f"{index:064x}" for index in range(NUM_TOKENS // FULL_BLOCK)],
        block_ids_by_group=[[1, 2, 3, 4], [5, 6, 7, 8]],
    )

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
        AscendStoreConnector,
    )

    assert AscendStoreConnector.requires_piecewise_for_cudagraph({}) is False
    assert AscendStoreConnector.requires_piecewise_for_cudagraph({"use_layerwise": True}) is True

    for logical_block in range(len(request.block_hashes)):
        target = pool._holemiss_derive_block_key(request, logical_block)
        assert target is not None
        assert target == builder.make_full_key(0, request.block_hashes[logical_block], 0, 0)
        assert "@group:0@" in target
        for state_group in (1, 2, 3):
            assert f"@group:{state_group}@" not in target

    state_key = pool._make_layerwise_full_key(1, request.block_hashes[0])
    assert "@group:1@" in state_key
    armed = {pool._holemiss_derive_block_key(request, i) for i in range(len(request.block_hashes))}
    assert state_key not in armed
    assert not any("@group:1@" in key or "@group:2@" in key or "@group:3@" in key for key in armed)
