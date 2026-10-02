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
"""CPU-only unit tests for the holemiss tail-writeback executor (i-b).

The write-back heals an HR-repaired block back into the store: non-force
remove (S4) then a fresh put session whose ONE batch_copy_put carries every
group-0 layer range (the per-call staging law).  The real store clients
require a torch_npu ACL context, so the backend seam is a byte-copying fake
with the store's S1/S4/lease semantics and the worker is a __new__ stub
carrying the real builder inputs.
"""

import ctypes
import hashlib
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.logger import logger as vllm_logger

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.ascend_store_connector import (
    AscendStoreConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.base import (
    BatchResultShapeError,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_backend import (
    MooncakeBackend,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.kv_transfer import (
    KVCacheStoreLayerRecvingThread,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)

BLOCK_SIZE = 128
NUM_LAYERS = 4
HEADS = 4
HEAD_DIM = 8
NUM_BLOCKS = 32
BLOCK_ID = 7
ENTRY_BYTES = BLOCK_SIZE * HEADS * HEAD_DIM * 4
OBJECT_SIZE = NUM_LAYERS * 2 * ENTRY_BYTES
REGION_KEY = "testmodel@abcd1234@0"
# The master's store-side handle-selection failure as the pybind surfaces it
# (mooncake-store include/types.h: NO_AVAILABLE_HANDLE = -200; the emitting
# site for this class is client_service.cpp's HasExpectedReplicaAllocation
# check inside StartBatchUpsert — the store namespace, never the TE one).
STORE_NO_AVAILABLE_HANDLE_RC = -200
# REPLICA_IS_NOT_READY (types.h): what a reader observes on a key whose only
# replica is stuck PROCESSING — the degraded state the removed upsert could
# leave behind (Case B marks the LOCAL_DISK survivor COMPLETE->PROCESSING and
# the failure path's UpsertRevoke(ALL) -> PutRevoke covers memory/NOF/DFS
# only, so nothing ever marks it back).
STORE_REPLICA_IS_NOT_READY_RC = -703

HOLEMISS_ENVS = [
    "HOLEMISS_KV_WRITEBACK",
    "HOLEMISS_RUNTIME_PROFILE",
]


def _layer_tensor(seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_BLOCKS, BLOCK_SIZE, HEADS, HEAD_DIM, generator=gen)


class WritebackStore:
    """Backend-seam fake with the store's S1/S4/lease semantics."""

    def __init__(self, regions):
        self.regions = list(regions)
        self.objects: dict[str, bytearray] = {}
        self.complete: set[str] = set()
        self.open_puts: set[str] = set()
        self.lease_failures: dict[str, int] = {}
        self.stale_replica_failures: dict[str, int] = {}
        self.copy_put_failures: dict[str, int] = {}
        self.calls: list[tuple] = []
        self.upserts: list[tuple] = []
        self.lock = threading.Lock()
        self.leased: set[str] = set()
        self.local_disk_complete: set[str] = set()
        self.local_disk_processing: set[str] = set()

    def _read(self, address: int, size: int) -> bytes:
        for start, length in self.regions:
            if start <= address and address + size <= start + length:
                return (ctypes.c_char * size).from_address(address).raw
        raise AssertionError(f"address 0x{address:x}+{size} outside registered regions")

    def batch_copy_get(self, keys, all_buffers, all_sizes, all_offsets):
        """The load-side ranged read (B35's lease model): the session get
        takes a lease on every key it reads — the master grants the read
        lease at query time (GetReplicaList/ExistKey). A key whose only
        replica is stuck PROCESSING reads REPLICA_IS_NOT_READY (the state
        the removed upsert's Case-B hazard could leave behind)."""
        self.calls.append(("batch_copy_get", tuple(keys)))
        self.leased.update(keys)
        results = []
        for key in keys:
            if key in self.local_disk_processing:
                results.append(STORE_REPLICA_IS_NOT_READY_RC)
            else:
                results.append(0)
        return results

    def batch_remove_lease(self, keys):
        """The load-end lease release (B35): the completed load drops the
        session leases it took; the master's non-force Remove then passes
        its IsLeaseExpired check."""
        self.calls.append(("batch_remove_lease", tuple(keys)))
        self.leased.difference_update(keys)
        return 0

    def batch_remove(self, keys):
        self.calls.append(("batch_remove", tuple(keys)))
        results = []
        for key in keys:
            if key in self.leased:
                results.append(-706)
                continue
            if self.lease_failures.get(key, 0) > 0:
                self.lease_failures[key] -= 1
                results.append(-706)
                continue
            if self.stale_replica_failures.get(key, 0) > 0:
                self.stale_replica_failures[key] -= 1
                results.append(-703)
                continue
            self.objects.pop(key, None)
            self.complete.discard(key)
            results.append(0)
        return results

    def batch_upsert_from(self, keys, buffer_ptrs, sizes, config=None):
        """The Case-B-faithful upsert seam — the master's dispatch, modeled
        from the Mooncake source at eeb1ce39 (closing the N11 gap: the
        pre-B42 fake modeled the takeover as an unconditional replace, which
        is exactly what the master refuses to do).

        The W2R2C topology: the armed key carries a COMPLETE adopted
        LOCAL_DISK bucket replica + a stale non-complete replica. The
        master's UpsertStart preempts the stale replica immediately, but the
        COMPLETE LOCAL_DISK survivor forces Case B (master_service.cpp
        UpsertStart, the same-size in-place branch): it marks the survivor
        COMPLETE->PROCESSING, takes over metadata.client_id, and returns the
        existing LOCAL_DISK-only descriptors. The client's allocation check
        (client_service.cpp HasExpectedReplicaAllocation, reached via
        RecordAllocatedReplicas which counts only memory/NOF/DFS) then fails
        NO_AVAILABLE_HANDLE before any transfer — every ReplicateConfig
        branch demands a nonzero memory/NOF/DFS allocation — and the failure
        path's UpsertRevoke(ALL) -> PutRevoke covers only memory/NOF/DFS,
        leaving the LOCAL_DISK replica stuck PROCESSING (the key unreadable
        until an UpsertEnd that never comes).
        """
        self.calls.append(("batch_upsert_from", tuple(keys)))
        results = []
        for key in keys:
            if key in self.local_disk_complete:
                self.local_disk_processing.add(key)
                results.append(STORE_NO_AVAILABLE_HANDLE_RC)
            else:
                self.upserts.append((tuple(keys), tuple(sizes)))
                results.append(0)
        return results

    def batch_put_start(self, keys, sizes):
        self.calls.append(("batch_put_start", tuple(keys), tuple(sizes)))
        results = []
        for key, size in zip(keys, sizes, strict=True):
            if key in self.complete or key in self.objects:
                results.append(-705)
            else:
                self.objects[key] = bytearray(size)
                self.open_puts.add(key)
                results.append(0)
        return results

    def batch_copy_put(self, keys, all_buffers, all_sizes, all_offsets):
        """The write-side ranged transfer with the store's REAL return
        convention (B44 — the N18 fifth occurrence): each key's result is
        the TRANSFERRED BYTE COUNT on success (the real
        batch_put_from_multi_buffer_ranges -> BatchTransferWriteRanges
        returns `transferred` per key) and a NEGATIVE ErrorCode on
        failure. The pre-B44 fake returned 0 on success, which never
        exercised the byte-count shape the production check must accept."""
        self.calls.append(
            (
                "batch_copy_put",
                tuple(keys),
                tuple(tuple(b) for b in all_buffers),
                tuple(tuple(s) for s in all_sizes),
                tuple(tuple(o) for o in all_offsets),
            )
        )
        results = []
        for key, buffers, sizes, offsets in zip(keys, all_buffers, all_sizes, all_offsets, strict=True):
            if key not in self.open_puts:
                results.append(-1)
                continue
            remaining = self.copy_put_failures.get(key, 0)
            if remaining > 0:
                self.copy_put_failures[key] = remaining - 1
                results.append(-800)
                continue
            obj = self.objects[key]
            for address, size, offset in zip(buffers, sizes, offsets, strict=True):
                obj[offset : offset + size] = self._read(address, size)
            results.append(sum(sizes))
        return results

    def batch_commit(self, keys):
        self.calls.append(("batch_commit", tuple(keys)))
        results = []
        for key in keys:
            if key not in self.open_puts:
                results.append(-1)
                continue
            self.open_puts.discard(key)
            self.complete.add(key)
            results.append(0)
        return results

    def batch_revoke(self, keys):
        self.calls.append(("batch_revoke", tuple(keys)))
        for key in keys:
            self.objects.pop(key, None)
            self.open_puts.discard(key)
            self.complete.discard(key)
        return [0] * len(keys)

    def call_names(self):
        return [name for name, *_ in self.calls]


def _text_caches():
    return {
        f"model.layers.{layer}.self_attn": (_layer_tensor(layer), _layer_tensor(layer + 100))
        for layer in range(NUM_LAYERS)
    }


def _ordered_entries(kv_caches):
    entries = []
    for layer in range(NUM_LAYERS):
        entries.extend(kv_caches[f"model.layers.{layer}.self_attn"])
    return entries


def _expected_object_bytes(kv_caches, block_id):
    payload = bytearray()
    for cache in _ordered_entries(kv_caches):
        payload.extend(cache[block_id].contiguous().view(torch.uint8).numpy().tobytes())
    return bytes(payload)


class WritebackTestBase(unittest.TestCase):
    def setUp(self):
        self._saved_env = {name: os.environ.get(name) for name in HOLEMISS_ENVS}
        for name in HOLEMISS_ENVS:
            os.environ.pop(name, None)

    def tearDown(self):
        for name, value in self._saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def _make_worker(self, kv_caches=None, num_groups=1, use_hybrid=False):
        kv_caches = kv_caches if kv_caches is not None else _text_caches()
        entries = _ordered_entries(kv_caches)
        regions = [(cache.data_ptr(), cache.numel() * cache.element_size()) for cache in entries]
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.tp_rank = 0
        worker.num_layers = NUM_LAYERS
        worker.hf_config = None
        worker.use_hybrid = use_hybrid
        worker.use_layerwise = True
        worker.use_block_key_layerwise = True
        worker.use_layerwise_transfer = False
        worker.num_kv_cache_groups = num_groups
        worker.kv_cache_config = None
        worker.kv_recv_thread = None
        worker.kv_send_thread = None
        worker.kv_caches = kv_caches
        worker._global_to_local_layer = {}
        worker.group_num_layers = {0: NUM_LAYERS}
        worker.group_block_len = {0: [ENTRY_BYTES] * len(entries)}
        worker.group_kv_caches_base_addr = {0: [cache.data_ptr() for cache in entries]}
        worker.group_block_stride = {0: [ENTRY_BYTES] * len(entries)}
        worker.group_layer_cache_entry_offsets = {0: list(range(0, len(entries) + 1, 2))}
        worker.token_database = SimpleNamespace(
            group_block_len=worker.group_block_len,
            group_kv_caches_base_addr=worker.group_kv_caches_base_addr,
            group_block_stride=worker.group_block_stride,
            group_layer_cache_entry_offsets=worker.group_layer_cache_entry_offsets,
        )
        worker._hole_recovery_regions = []
        worker._hole_recovery_lock = threading.Lock()
        worker._holemiss_writeback_keys = {}
        worker.m_store = WritebackStore(regions)
        return worker

    def _region(self, key=REGION_KEY, result=-1102, block_id=BLOCK_ID):
        return {
            "request_id": "req-A",
            "key": key,
            "result": result,
            "result_index": 0,
            "logical_block_index": 2,
            "token_start": 2 * BLOCK_SIZE,
            "token_end": 3 * BLOCK_SIZE,
            "physical_block_id": block_id,
            "layer_id": 1,
            "layer_start": 1,
            "layer_end": 1,
            "tp_rank": 0,
            "pool_page_row": list(range(16)),
            "manager_block_size": BLOCK_SIZE,
        }

    def _no_sleep(self):
        slept: list[float] = []

        def _fake_sleep(seconds):
            slept.append(seconds * 1000)

        patcher = patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.time.sleep",
            side_effect=_fake_sleep,
        )
        return patcher, slept

    def _fake_clock(self):
        """A faked sleep that also advances a faked monotonic clock.

        B37's remove wait is budget-driven on time.monotonic: with only the
        sleep faked the clock never advances and a never-succeeding remove
        would poll forever, so the budget-exhaustion tests drive both.
        """
        clock = [1000.0]
        slept: list[float] = []

        def _fake_sleep(seconds):
            slept.append(seconds * 1000)
            clock[0] += seconds

        def _fake_monotonic():
            return clock[0]

        sleep_patcher = patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.time.sleep",
            side_effect=_fake_sleep,
        )
        monotonic_patcher = patch(
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker.time.monotonic",
            side_effect=_fake_monotonic,
        )
        return sleep_patcher, monotonic_patcher, slept


class TestBackendBatchRemoveShape(unittest.TestCase):
    """A2: the pybind force slot receives the literal False."""

    class _PyStore:
        def __init__(self):
            self.calls = []

        def batch_remove(self, keys, force=False):
            self.calls.append((list(keys), force))
            return [0] * len(keys)

    def _backend(self, store):
        backend = MooncakeBackend.__new__(MooncakeBackend)
        backend._store_initialized = True
        backend._store_init_lock = threading.Lock()
        backend.store = store
        return backend

    def test_batch_remove_passes_literal_false(self):
        store = self._PyStore()
        results = self._backend(store).batch_remove(["k1", "k2"])
        self.assertEqual(results, [0, 0])
        self.assertEqual(store.calls, [(["k1", "k2"], False)])
        self.assertIsNot(store.calls[0][1], True)

    def test_batch_remove_never_coerces_keys_into_force_slot(self):
        store = self._PyStore()
        self._backend(store).batch_remove(["k1"])
        keys, force = store.calls[0]
        self.assertEqual(keys, ["k1"])
        self.assertIs(force, False)

    def test_batch_remove_shape_error_propagates(self):
        class _BadStore:
            def batch_remove(self, keys, force=False):
                return [0, 0, 0]

        with self.assertRaises(BatchResultShapeError):
            self._backend(_BadStore()).batch_remove(["k1"])

    def test_batch_remove_lease_delegates_to_get_session_end(self):
        """B35: the mooncake client has no lease-named API — the backend's
        lease release maps onto the GET session end (real_client.cpp
        batch_get_session_end drops the client-held sessions)."""

        class _LeaseStore:
            def __init__(self):
                self.calls = []

            def batch_get_session_end(self, keys):
                self.calls.append(list(keys))
                return 0

        store = _LeaseStore()
        rc = self._backend(store).batch_remove_lease(["k1", "k2"])
        self.assertEqual(rc, 0)
        self.assertEqual(store.calls, [["k1", "k2"]])


class TestConnectorWritebackForward(unittest.TestCase):
    def _connector(self, worker):
        connector = AscendStoreConnector.__new__(AscendStoreConnector)
        connector.connector_worker = worker
        return connector

    def test_forward_delegates_to_worker(self):
        worker = SimpleNamespace(writeback_hole_recovery_regions=lambda: [{"key": "k", "result": 0}])
        self.assertEqual(
            self._connector(worker).writeback_hole_recovery_kv(),
            [{"key": "k", "result": 0}],
        )

    def test_worker_without_method_is_loud_and_empty(self):
        with self.assertLogs(vllm_logger, level="ERROR") as logs:
            results = self._connector(SimpleNamespace()).writeback_hole_recovery_kv()
        self.assertEqual(results, [])
        self.assertTrue(any("worker_lacks_writeback" in line for line in logs.output))

    def test_worker_none_is_loud_and_empty(self):
        with self.assertLogs(vllm_logger, level="ERROR") as logs:
            results = self._connector(None).writeback_hole_recovery_kv()
        self.assertEqual(results, [])
        self.assertTrue(any("worker_lacks_writeback" in line for line in logs.output))


class TestWritebackExecutor(WritebackTestBase):
    def test_happy_path_call_order_and_single_copy_put(self):
        worker = self._make_worker()
        worker.record_hole_recovery_region(self._region())
        patcher, _ = self._no_sleep()
        with patcher:
            results = worker.writeback_hole_recovery_regions()
        store = worker.m_store
        self.assertEqual(
            store.call_names(),
            ["batch_remove", "batch_put_start", "batch_copy_put", "batch_commit"],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["key"], REGION_KEY)
        self.assertEqual(results[0]["result"], 0)
        self.assertEqual(results[0]["stage"], "commit")
        self.assertEqual(len(results[0]["sha"]), 64)
        self.assertEqual(store.calls[1][1], (REGION_KEY,))
        self.assertEqual(store.calls[1][2], (OBJECT_SIZE,))
        copy_put = store.calls[2]
        self.assertEqual(copy_put[1], (REGION_KEY,))
        self.assertEqual(len(copy_put[2][0]), NUM_LAYERS * 2)
        self.assertTrue(REGION_KEY in store.complete)

    def test_copy_put_triples_match_build_addrs_math(self):
        worker = self._make_worker()
        worker.record_hole_recovery_region(self._region())
        patcher, _ = self._no_sleep()
        with patcher:
            worker.writeback_hole_recovery_regions()
        _, _, buffers, sizes, offsets = worker.m_store.calls[2]
        segments, seg_sizes, seg_offsets = buffers[0], sizes[0], offsets[0]
        entries = _ordered_entries(worker.kv_caches)
        expected_buffers = []
        expected_sizes = []
        expected_offsets = []
        for layer in range(NUM_LAYERS):
            k_cache = entries[layer * 2]
            v_cache = entries[layer * 2 + 1]
            expected_buffers.extend(
                [
                    k_cache.data_ptr() + BLOCK_ID * ENTRY_BYTES,
                    v_cache.data_ptr() + BLOCK_ID * ENTRY_BYTES,
                ]
            )
            expected_sizes.extend([ENTRY_BYTES, ENTRY_BYTES])
            base = layer * 2 * ENTRY_BYTES
            expected_offsets.extend([base, base + ENTRY_BYTES])
        self.assertEqual(list(segments), expected_buffers)
        self.assertEqual(list(seg_sizes), expected_sizes)
        self.assertEqual(list(seg_offsets), expected_offsets)

    def test_object_bytes_equal_healed_block_walk(self):
        worker = self._make_worker()
        worker.record_hole_recovery_region(self._region())
        patcher, _ = self._no_sleep()
        with patcher:
            worker.writeback_hole_recovery_regions()
        expected = _expected_object_bytes(worker.kv_caches, BLOCK_ID)
        self.assertEqual(bytes(worker.m_store.objects[REGION_KEY]), expected)
        self.assertEqual(len(expected), OBJECT_SIZE)

    def test_remove_precedes_put_on_existing_object(self):
        worker = self._make_worker()
        worker.m_store.objects[REGION_KEY] = bytearray(OBJECT_SIZE)
        worker.m_store.complete.add(REGION_KEY)
        worker.record_hole_recovery_region(self._region())
        patcher, _ = self._no_sleep()
        with patcher:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(results[0]["result"], 0)
        names = worker.m_store.call_names()
        self.assertLess(names.index("batch_remove"), names.index("batch_put_start"))

    def test_lease_straddle_retries_then_succeeds(self):
        """B37: an aging lease (the master read lease, ~15-45s old at the
        teardown) is polled at the civil interval until it expires."""
        worker = self._make_worker()
        worker.m_store.lease_failures[REGION_KEY] = 2
        worker.record_hole_recovery_region(self._region())
        patcher, slept = self._no_sleep()
        with patcher, self.assertLogs(vllm_logger, level="INFO") as logs:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(results[0]["result"], 0)
        self.assertEqual(worker.m_store.call_names().count("batch_remove"), 3)
        self.assertEqual([round(ms) for ms in slept], [5000, 5000])
        retry_lines = [line for line in logs.output if "HOLEMISS_WRITEBACK_REMOVE_RETRY" in line]
        self.assertEqual(len(retry_lines), 2)
        self.assertIn("attempt=1", retry_lines[0])
        self.assertIn("reason=-706", retry_lines[0])

    def test_remove_waits_out_master_lease_live_model(self):
        """The B37 live timing model: the lease from the R1 hit checks is
        ~45s old at the first attempt and dies 60s after its grant — the
        wait bridges to expiry well inside the 120s budget, then the fresh
        put session lands (the round-trip census's put=commit=tp shape)."""
        worker = self._make_worker()
        worker.m_store.lease_failures[REGION_KEY] = 9
        worker.record_hole_recovery_region(self._region())
        sleep_patcher, monotonic_patcher, slept = self._fake_clock()
        with sleep_patcher, monotonic_patcher, self.assertLogs(vllm_logger, level="INFO") as logs:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(results[0]["result"], 0)
        self.assertEqual(results[0]["stage"], "commit")
        self.assertEqual(worker.m_store.call_names().count("batch_remove"), 10)
        self.assertEqual([round(ms) for ms in slept], [5000] * 9)
        retry_lines = [line for line in logs.output if "HOLEMISS_WRITEBACK_REMOVE_RETRY" in line]
        self.assertEqual(len(retry_lines), 9)
        self.assertIn("elapsed_ms=40000", retry_lines[-1])
        self.assertTrue(any("HOLEMISS_WRITEBACK_PUT" in line for line in logs.output))

    def test_lease_exhaustion_fails_loud_without_put(self):
        """A lease that never ages within the budget exhausts the wait and
        fails LOUD at stage=remove — the unchanged best-effort contract."""
        worker = self._make_worker()
        worker.m_store.lease_failures[REGION_KEY] = 99
        worker.record_hole_recovery_region(self._region())
        sleep_patcher, monotonic_patcher, slept = self._fake_clock()
        with sleep_patcher, monotonic_patcher, self.assertLogs(vllm_logger, level="INFO") as logs:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(results, [{"key": REGION_KEY, "result": -706, "stage": "remove"}])
        self.assertEqual(worker.m_store.call_names().count("batch_remove"), 25)
        self.assertEqual([round(ms) for ms in slept], [5000] * 24)
        retry_lines = [line for line in logs.output if "HOLEMISS_WRITEBACK_REMOVE_RETRY" in line]
        self.assertEqual(len(retry_lines), 24)
        self.assertNotIn("batch_put_start", worker.m_store.call_names())
        self.assertTrue(any("HOLEMISS_WRITEBACK_FAILED" in line and "stage=remove" in line for line in logs.output))

    def test_remove_first_attempt_success_zero_sleep(self):
        """Zero added latency when the first attempt succeeds (the no-lease
        teardown and the B35-released shape)."""
        worker = self._make_worker()
        worker.record_hole_recovery_region(self._region())
        patcher, slept = self._no_sleep()
        with patcher, self.assertLogs(vllm_logger, level="INFO") as logs:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(results[0]["result"], 0)
        self.assertEqual(worker.m_store.call_names().count("batch_remove"), 1)
        self.assertEqual(slept, [])
        self.assertEqual([line for line in logs.output if "HOLEMISS_WRITEBACK_REMOVE_RETRY" in line], [])

    def test_remove_non_706_returns_immediately(self):
        """The B27 -706-only law carried into the wait: any other negative
        returns on the first attempt with no retry and no sleep."""
        worker = self._make_worker()
        worker.record_hole_recovery_region(self._region())
        worker.m_store.batch_remove = lambda keys: [-707] * len(keys)
        patcher, slept = self._no_sleep()
        with patcher, self.assertLogs(vllm_logger, level="INFO") as logs:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(results, [{"key": REGION_KEY, "result": -707, "stage": "remove"}])
        self.assertNotIn("batch_put_start", worker.m_store.call_names())
        self.assertEqual(slept, [])
        self.assertEqual([line for line in logs.output if "HOLEMISS_WRITEBACK_REMOVE_RETRY" in line], [])
        self.assertTrue(any("HOLEMISS_WRITEBACK_FAILED" in line and "reason=-707" in line for line in logs.output))

    def test_remove_703_honest_degradation_never_upserts(self):
        """B42 (the honest degradation): the remove fails -703 through the
        B37 wait (which only loops on -706), the recovery ladder revokes and
        retries the remove ONCE, and — still blocked — returns the LOUD
        FAILED. THE SAFETY INVARIANT: on the live topology (the armed key
        carrying a COMPLETE adopted LOCAL_DISK bucket replica — modeled by
        local_disk_complete) the store client NEVER receives an upsert call:
        the master's UpsertStart would force Case B, the client-side
        allocation check would fail NO_AVAILABLE_HANDLE before any transfer,
        and the failure path would leave the LOCAL_DISK replica stuck
        PROCESSING — degrading a readable key to unreadable. A loud FAILED is
        recoverable; an unreadable key is not."""
        worker = self._make_worker()
        worker.m_store.stale_replica_failures[REGION_KEY] = 99
        worker.m_store.local_disk_complete = {REGION_KEY}
        worker.record_hole_recovery_region(self._region())
        patcher, _ = self._no_sleep()
        with patcher, self.assertLogs(vllm_logger, level="INFO") as logs:
            results = worker.writeback_hole_recovery_regions()
        store = worker.m_store
        # The ladder: remove (-703) -> revoke -> remove (-703) -> FAILED.
        self.assertEqual(results, [{"key": REGION_KEY, "result": -703, "stage": "remove"}])
        self.assertEqual(store.call_names().count("batch_remove"), 2)
        self.assertEqual(store.call_names().count("batch_revoke"), 1)
        self.assertNotIn("batch_upsert_from", store.call_names())
        self.assertNotIn("batch_put_start", store.call_names())
        self.assertNotIn("batch_copy_put", store.call_names())
        self.assertTrue(any("HOLEMISS_WRITEBACK_REMOVE_STALE" in line for line in logs.output))
        self.assertTrue(any("HOLEMISS_WRITEBACK_FAILED" in line and "reason=-703" in line for line in logs.output))
        # The store is NOT degraded: the key's LOCAL_DISK replica was never
        # marked PROCESSING and the key stays readable.
        self.assertEqual(store.local_disk_processing, set())
        self.assertEqual(store.batch_copy_get([REGION_KEY], [], [], []), [0])

    def test_upsert_on_local_disk_topology_cannot_land_and_degrades_the_store(self):
        """The Case-B hazard, modeled from the master source (N11 closed):
        IF any code path drove an upsert on the W2R2C topology (a COMPLETE
        adopted LOCAL_DISK replica + a stale non-complete one), the master's
        UpsertStart dispatch would mark the LOCAL_DISK survivor
        COMPLETE->PROCESSING and return a LOCAL_DISK-only descriptor set; the
        client's HasExpectedReplicaAllocation check counts only
        memory/NOF/DFS (client_service.cpp RecordAllocatedReplicas) and
        demands a nonzero memory allocation for every nof=0/dfs=0 config
        (regardless of replica_num), so the call resolves NO_AVAILABLE_HANDLE
        before any transfer — and the failure path's UpsertRevoke(ALL) ->
        PutRevoke covers only memory/NOF/DFS, leaving the survivor stuck
        PROCESSING. This test pins WHY the ladder must never issue the call;
        M-B42 (re-introducing the upsert into the ladder) makes the
        honest-degradation test above fail on exactly this model."""
        worker = self._make_worker()
        store = worker.m_store
        store.local_disk_complete = {REGION_KEY}
        results = store.batch_upsert_from([REGION_KEY], [], [])
        self.assertEqual(results, [STORE_NO_AVAILABLE_HANDLE_RC])
        self.assertEqual(store.local_disk_processing, {REGION_KEY})
        self.assertEqual(store.batch_copy_get([REGION_KEY], [], [], []), [STORE_REPLICA_IS_NOT_READY_RC])

    def test_remove_703_retry_recovery_reopens_the_normal_chain(self):
        """B42: when the revoke clears a writer-owned stale session and the
        single re-attempt succeeds, the recovery returns None and the
        standard put chain proceeds (put_start -> copy_put -> commit)."""
        worker = self._make_worker()
        worker.m_store.stale_replica_failures[REGION_KEY] = 1
        worker.record_hole_recovery_region(self._region())
        patcher, _ = self._no_sleep()
        with patcher, self.assertLogs(vllm_logger, level="INFO") as logs:
            results = worker.writeback_hole_recovery_regions()
        store = worker.m_store
        self.assertEqual(results[0]["result"], 0)
        self.assertEqual(results[0]["stage"], "commit")
        self.assertEqual(store.call_names().count("batch_remove"), 2)
        self.assertEqual(store.call_names().count("batch_revoke"), 1)
        self.assertNotIn("batch_upsert_from", store.call_names())
        self.assertEqual(store.call_names().count("batch_put_start"), 1)
        self.assertEqual(store.call_names().count("batch_commit"), 1)
        self.assertTrue(any("HOLEMISS_WRITEBACK_REMOVE_STALE" in line for line in logs.output))

    def test_transient_copy_put_failure_retries_once(self):
        worker = self._make_worker()
        worker.m_store.copy_put_failures[REGION_KEY] = 1
        worker.record_hole_recovery_region(self._region())
        patcher, slept = self._no_sleep()
        with patcher:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(results[0]["result"], 0)
        self.assertEqual(worker.m_store.call_names().count("batch_copy_put"), 2)
        self.assertEqual([round(ms) for ms in slept], [100])

    def test_copy_put_exhaustion_revokes_and_fails_loud(self):
        worker = self._make_worker()
        worker.m_store.copy_put_failures[REGION_KEY] = 99
        worker.record_hole_recovery_region(self._region())
        patcher, slept = self._no_sleep()
        with patcher, self.assertLogs(vllm_logger, level="INFO") as logs:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(results, [{"key": REGION_KEY, "result": -800, "stage": "copy_put"}])
        names = worker.m_store.call_names()
        self.assertEqual(names.count("batch_copy_put"), 4)
        self.assertEqual(names.count("batch_revoke"), 1)
        self.assertEqual([round(ms) for ms in slept], [100, 200, 400])
        self.assertTrue(any("HOLEMISS_WRITEBACK_FAILED" in line and "stage=copy_put" in line for line in logs.output))

    def test_fake_copy_put_returns_transferred_bytes(self):
        """The live-geometry return law (B44, the N18 fifth occurrence):
        the fake models the real store's batch_copy_put convention — the
        TRANSFERRED BYTE COUNT on success (negative on error) — so every
        writeback test exercises the real return shape the production
        success check must accept."""
        worker = self._make_worker()
        store = worker.m_store
        store.batch_put_start([REGION_KEY], [OBJECT_SIZE])
        base = worker.kv_caches["model.layers.0.self_attn"][0].data_ptr()
        self.assertEqual(
            store.batch_copy_put([REGION_KEY], [[base]], [[ENTRY_BYTES]], [[0]]),
            [ENTRY_BYTES],
        )
        self.assertEqual(
            store.batch_copy_put(
                [REGION_KEY],
                [[base, base + ENTRY_BYTES]],
                [[ENTRY_BYTES, ENTRY_BYTES]],
                [[0, ENTRY_BYTES]],
            ),
            [2 * ENTRY_BYTES],
        )

    def test_copy_put_byte_count_success_is_not_a_failure(self):
        """B44: the real batch_copy_put returns the TRANSFERRED BYTE
        COUNT on success (negative on error), so the write-back's outer
        check must be `write_rc < 0` — a positive count means the
        transfer COMPLETED. The live W2R2D signature: reason=50331648
        (hybrid g0) / 16777216 (text) were complete transfers misread as
        failures. A positive count -> the PUT marker fires, NO FAILED, no
        revoke, the commit lands. The mutation M-B44 (reverting the outer
        check to `write_rc != 0`) makes this test fail."""
        worker = self._make_worker()
        worker.record_hole_recovery_region(self._region())
        patcher, _ = self._no_sleep()
        with patcher, self.assertLogs(vllm_logger, level="INFO") as logs:
            results = worker.writeback_hole_recovery_regions()
        store = worker.m_store
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["result"], 0)
        self.assertEqual(results[0]["stage"], "commit")
        self.assertNotIn("batch_revoke", store.call_names())
        self.assertNotIn("HOLEMISS_WRITEBACK_FAILED", "\n".join(logs.output))
        self.assertTrue(any("HOLEMISS_WRITEBACK_PUT" in line for line in logs.output))
        self.assertIn(REGION_KEY, store.complete)

    def test_put_start_failure_fails_loud_without_copy(self):
        worker = self._make_worker()
        worker.record_hole_recovery_region(self._region())
        worker.m_store.batch_put_start = lambda keys, sizes: [-705] * len(keys)
        patcher, _ = self._no_sleep()
        with patcher, self.assertLogs(vllm_logger, level="INFO") as logs:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(results, [{"key": REGION_KEY, "result": -705, "stage": "put_start"}])
        self.assertNotIn("batch_copy_put", worker.m_store.call_names())
        self.assertNotIn("batch_revoke", worker.m_store.call_names())
        self.assertTrue(any("HOLEMISS_WRITEBACK_FAILED" in line and "stage=put_start" in line for line in logs.output))

    def test_census_markers_and_sha_tracking(self):
        worker = self._make_worker()
        worker.record_hole_recovery_region(self._region())
        patcher, _ = self._no_sleep()
        with patcher, self.assertLogs(vllm_logger, level="INFO") as logs:
            worker.writeback_hole_recovery_regions()
        expected_sha = hashlib.sha256(_expected_object_bytes(worker.kv_caches, BLOCK_ID)).hexdigest()
        self.assertEqual(worker._holemiss_writeback_keys[REGION_KEY], expected_sha)
        put_lines = [line for line in logs.output if "HOLEMISS_WRITEBACK_PUT" in line]
        commit_lines = [line for line in logs.output if "HOLEMISS_WRITEBACK_COMMIT" in line]
        self.assertEqual(len(put_lines), 1)
        self.assertEqual(len(commit_lines), 1)
        self.assertIn(f"sha={expected_sha}", put_lines[0])
        self.assertIn(f"tokens={2 * BLOCK_SIZE}-{3 * BLOCK_SIZE}", put_lines[0])
        self.assertIn(f"block={BLOCK_ID}", put_lines[0])
        self.assertIn(f"layers={NUM_LAYERS}", put_lines[0])
        self.assertIn(f"bytes={OBJECT_SIZE}", put_lines[0])
        self.assertIn("result=0", commit_lines[0])

    def test_dedupe_two_regions_one_key(self):
        worker = self._make_worker()
        worker.record_hole_recovery_region(self._region())
        second = self._region()
        second["result_index"] = 1
        second["layer_id"] = 2
        worker.record_hole_recovery_region(second)
        patcher, _ = self._no_sleep()
        with patcher:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(len(results), 1)
        self.assertEqual(worker.m_store.call_names().count("batch_remove"), 1)
        self.assertEqual(worker.m_store.call_names().count("batch_put_start"), 1)

    def test_non_negative_result_regions_skipped(self):
        worker = self._make_worker()
        worker.record_hole_recovery_region(self._region(result=0))
        results = worker.writeback_hole_recovery_regions()
        self.assertEqual(results, [])
        self.assertEqual(worker.m_store.calls, [])

    def test_no_regions_no_calls(self):
        worker = self._make_worker()
        self.assertEqual(worker.writeback_hole_recovery_regions(), [])
        self.assertEqual(worker.m_store.calls, [])

    def test_not_block_key_layerwise_skips_loud(self):
        worker = self._make_worker()
        worker.use_block_key_layerwise = False
        worker.record_hole_recovery_region(self._region())
        with self.assertLogs(vllm_logger, level="ERROR") as logs:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(results, [])
        self.assertTrue(any("not_block_key_layerwise" in line for line in logs.output))

    def test_store_exception_is_loud_best_effort(self):
        worker = self._make_worker()
        worker.record_hole_recovery_region(self._region())

        def _boom(keys):
            raise RuntimeError("store exploded")

        worker.m_store.batch_remove = _boom
        with self.assertLogs(vllm_logger, level="INFO") as logs:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(results, [{"key": REGION_KEY, "result": -1, "stage": "exception"}])
        self.assertTrue(any("stage=exception" in line for line in logs.output))

    def test_repeated_writeback_is_content_idempotent(self):
        worker = self._make_worker()
        patcher, _ = self._no_sleep()
        with patcher:
            worker.record_hole_recovery_region(self._region())
            first = worker.writeback_hole_recovery_regions()
            worker.clear_hole_recovery_regions()
            worker.record_hole_recovery_region(self._region())
            second = worker.writeback_hole_recovery_regions()
        self.assertEqual(first[0]["result"], 0)
        self.assertEqual(second[0]["result"], 0)
        self.assertEqual(first[0]["sha"], second[0]["sha"])
        self.assertEqual(worker.m_store.call_names().count("batch_remove"), 2)
        self.assertTrue(REGION_KEY in worker.m_store.complete)


class TestWritebackHybridScope(WritebackTestBase):
    """D5: the heal writes FA-group-0 objects only."""

    def _make_hybrid_worker(self):
        fa_caches = {f"model.layers.{p}.self_attn": (_layer_tensor(p), _layer_tensor(p + 100)) for p in (3, 7)}
        gdn_caches = {f"model.layers.{p}.linear_attn": (_layer_tensor(p + 50),) for p in (0, 1)}
        kv_caches = {**fa_caches, **gdn_caches}
        entries = [c for p in (3, 7) for c in fa_caches[f"model.layers.{p}.self_attn"]]
        regions = [(cache.data_ptr(), cache.numel() * cache.element_size()) for cache in entries]
        worker = KVPoolWorker.__new__(KVPoolWorker)
        worker.tp_rank = 0
        worker.num_layers = 4
        worker.hf_config = None
        worker.use_hybrid = True
        worker.use_layerwise = True
        worker.use_block_key_layerwise = True
        worker.use_layerwise_transfer = False
        worker.num_kv_cache_groups = 2
        worker.kv_caches = kv_caches
        worker._global_to_local_layer = {}
        worker.kv_recv_thread = None
        worker.kv_send_thread = None
        worker.group_num_layers = {0: 2, 1: 2}
        worker.group_block_len = {0: [ENTRY_BYTES] * 4, 1: [ENTRY_BYTES] * 2}
        gdn_entries = [gdn_caches[f"model.layers.{p}.linear_attn"][0] for p in (0, 1)]
        worker.group_kv_caches_base_addr = {
            0: [cache.data_ptr() for cache in entries],
            1: [cache.data_ptr() for cache in gdn_entries],
        }
        worker.group_block_stride = {0: [ENTRY_BYTES] * 4, 1: [ENTRY_BYTES] * 2}
        worker.group_layer_cache_entry_offsets = {0: [0, 2, 4], 1: [0, 1, 2]}
        worker.token_database = SimpleNamespace(
            group_block_len=worker.group_block_len,
            group_kv_caches_base_addr=worker.group_kv_caches_base_addr,
            group_block_stride=worker.group_block_stride,
            group_layer_cache_entry_offsets=worker.group_layer_cache_entry_offsets,
        )
        worker._hole_recovery_regions = []
        worker._hole_recovery_lock = threading.Lock()
        worker._holemiss_writeback_keys = {}
        worker.m_store = WritebackStore(regions)
        fa_names = [f"model.layers.{p}.self_attn" for p in (3, 7)]
        gdn_names = [f"model.layers.{p}.linear_attn" for p in (0, 1)]
        worker.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(layer_names=fa_names),
                SimpleNamespace(layer_names=gdn_names),
            ]
        )
        return worker, entries

    def test_ranges_cover_group0_layers_only(self):
        worker, entries = self._make_hybrid_worker()
        worker.record_hole_recovery_region(self._region())
        patcher, _ = self._no_sleep()
        with patcher:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(results[0]["result"], 0)
        copy_put = worker.m_store.calls[2]
        self.assertEqual(len(copy_put[2][0]), 4)
        self.assertEqual(worker.m_store.calls[1][2], (4 * ENTRY_BYTES,))

    def test_group0_entries_exclude_gdn_caches(self):
        worker, entries = self._make_hybrid_worker()
        selected = worker._holemiss_group0_cache_entries()
        self.assertEqual(len(selected), 4)
        for cache in selected:
            self.assertIn(id(cache), {id(c) for c in entries})

    def test_sha_walk_matches_group0_object_bytes(self):
        worker, entries = self._make_hybrid_worker()
        worker.record_hole_recovery_region(self._region())
        patcher, _ = self._no_sleep()
        with patcher:
            worker.writeback_hole_recovery_regions()
        expected = bytearray()
        for cache in entries:
            expected.extend(cache[BLOCK_ID].contiguous().view(torch.uint8).numpy().tobytes())
        self.assertEqual(
            worker._holemiss_writeback_keys[REGION_KEY],
            hashlib.sha256(bytes(expected)).hexdigest(),
        )
        self.assertEqual(bytes(worker.m_store.objects[REGION_KEY]), bytes(expected))


class TestWritebackLoadVerify(WritebackTestBase):
    def _recv_thread(self, worker):
        thread = KVCacheStoreLayerRecvingThread.__new__(KVCacheStoreLayerRecvingThread)
        thread.worker = worker
        return thread

    def _heal(self, worker):
        worker.record_hole_recovery_region(self._region())
        patcher, _ = self._no_sleep()
        with patcher:
            worker.writeback_hole_recovery_regions()

    def test_verified_match_true_after_clean_reload(self):
        import os

        os.environ["HOLEMISS_KV_WRITEBACK"] = "1"
        worker = self._make_worker()
        self._heal(worker)
        with self.assertLogs(vllm_logger, level="INFO") as logs:
            self._recv_thread(worker)._holemiss_verify_writeback_loads([REGION_KEY], [BLOCK_ID], ["req-B"], [0])
        verified = [line for line in logs.output if "HOLEMISS_WRITEBACK_VERIFIED" in line]
        self.assertEqual(len(verified), 1)
        self.assertIn("match=True", verified[0])
        self.assertIn("req=req-B", verified[0])

    def test_verified_match_false_after_mutation(self):
        import os

        os.environ["HOLEMISS_KV_WRITEBACK"] = "1"
        worker = self._make_worker()
        self._heal(worker)
        k_cache = worker.kv_caches["model.layers.0.self_attn"][0]
        k_cache[BLOCK_ID] += 1.0
        with self.assertLogs(vllm_logger, level="INFO") as logs:
            self._recv_thread(worker)._holemiss_verify_writeback_loads([REGION_KEY], [BLOCK_ID], ["req-B"], [0])
        verified = [line for line in logs.output if "HOLEMISS_WRITEBACK_VERIFIED" in line]
        self.assertEqual(len(verified), 1)
        self.assertIn("match=False", verified[0])

    def test_env_off_emits_nothing(self):
        worker = self._make_worker()
        self._heal(worker)
        thread = self._recv_thread(worker)
        with self.assertNoLogs(vllm_logger, level="INFO"):
            thread._holemiss_verify_writeback_loads([REGION_KEY], [BLOCK_ID], ["req-B"], [0])

    def test_untracked_keys_and_failed_loads_skipped(self):
        import os

        os.environ["HOLEMISS_KV_WRITEBACK"] = "1"
        worker = self._make_worker()
        self._heal(worker)
        thread = self._recv_thread(worker)
        with self.assertNoLogs(vllm_logger, level="INFO"):
            thread._holemiss_verify_writeback_loads(["other@key@0"], [BLOCK_ID], ["req-B"], [0])
            thread._holemiss_verify_writeback_loads([REGION_KEY], [BLOCK_ID], ["req-B"], [-1102])


class TestWritebackLiteralConstants(unittest.TestCase):
    """D-B20.10/F-1 class: the retry bounds are pinned literally."""

    def test_remove_lease_retry_constants(self):
        """The F-1 load-path pins stay literal and untouched by B37 (they
        are pinned at their home module, kv_transfer, where the load path
        reads them; pool_worker no longer re-exports the two bounds)."""
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import (
            kv_transfer,
            pool_worker,
        )

        self.assertEqual(kv_transfer.LOAD_LEASE_RETRY_ERROR, -706)
        self.assertEqual(kv_transfer.LOAD_LEASE_RETRY_MAX_ATTEMPTS, 3)
        self.assertEqual(kv_transfer.LOAD_LEASE_RETRY_BACKOFF_MS, 200)
        self.assertEqual(pool_worker.LOAD_LEASE_RETRY_ERROR, -706)

    def test_writeback_remove_wait_constants(self):
        """B37's OWN bound set (the bound-widening tripwire): the budget
        carries >= 2x the 60000ms master lease TTL (default_kv_lease_ttl,
        B28 — the calibration law) and the interval is the civil poll
        cadence; widening or narrowing either trips this pin."""
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import (
            pool_worker,
        )

        self.assertEqual(pool_worker.WRITEBACK_REMOVE_LEASE_WAIT_BUDGET_MS, 120_000)
        self.assertEqual(pool_worker.WRITEBACK_REMOVE_RETRY_INTERVAL_MS, 5_000)
        self.assertGreaterEqual(pool_worker.WRITEBACK_REMOVE_LEASE_WAIT_BUDGET_MS, 2 * 60_000)

    def test_write_retry_constants(self):
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import (
            pool_worker,
        )

        self.assertEqual(pool_worker.RANGE_WRITE_MAX_RETRIES, 3)
        self.assertEqual(pool_worker.RANGE_WRITE_RETRY_BACKOFF_MS, 100)

    def test_env_registry_carries_writeback_gate(self):
        from vllm_ascend import envs

        self.assertIn("HOLEMISS_KV_WRITEBACK", envs.env_variables)


class TestBatchedLoadLeaseReleaseUnblocksWriteback(WritebackTestBase):
    """B35's S1-class pin: the chain from the batched load's completion-time
    lease release to the write-back's non-force remove passing.

    The live B35 signature was stage=remove reason=-706 on both ranks and
    both geometries: the batched load leases the keys and never releases
    them, so the master's IsLeaseExpired check refuses the remove. These
    tests pin (a) the load-end release empties the lease surface, (b) the
    write-back then removes+puts+commits, and (c) the unreleased twin fails
    exactly at stage=remove — removing the handler's release line breaks
    (a) and therefore (b)."""

    def _drive_batched_load(self, store) -> None:
        from tests.ut.kv_transfer.test_range_load_batching import (
            _batched_task,
            _stub_recv_thread,
        )
        from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
            LayerRangeReqMeta,
        )

        def _meta(layer_id: int) -> LayerRangeReqMeta:
            return LayerRangeReqMeta(
                req_ids=["req-B"],
                layer_id=layer_id,
                block_ids=[BLOCK_ID],
                keys=[REGION_KEY],
                all_buffers=[[1000 + layer_id]],
                all_sizes=[[256]],
                all_offsets=[[layer_id * 1024]],
                load_keys=[],
            )

        task, build_map = _batched_task({layer: [_meta(layer)] for layer in range(NUM_LAYERS)})
        thread = _stub_recv_thread(build_map)
        thread.m_store = store
        thread._handle_batched_range_layer_task(task)

    def test_load_end_release_lets_the_writeback_remove_pass(self):
        worker = self._make_worker()
        store = worker.m_store
        self._drive_batched_load(store)
        self.assertEqual(store.leased, set(), "the load-end release must drop every lease it took")
        worker.record_hole_recovery_region(self._region())
        patcher, _ = self._no_sleep()
        with patcher:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["result"], 0)
        self.assertEqual(results[0]["stage"], "commit")
        self.assertIn(REGION_KEY, store.complete)
        self.assertIn("batch_remove_lease", store.call_names())

    def test_unreleased_lease_blocks_the_writeback_remove(self):
        """The B35 live-signature twin under the B37 wait: a lease that
        never ages (the pathological case — the master lease always ages
        out live, TTL 60s) exhausts the budget and still fails the remove
        with -706 at stage=remove — the loud, GT-safe degradation (the
        original object stays; no put is attempted)."""
        worker = self._make_worker()
        store = worker.m_store
        store.leased.add(REGION_KEY)
        worker.record_hole_recovery_region(self._region())
        sleep_patcher, monotonic_patcher, slept = self._fake_clock()
        with sleep_patcher, monotonic_patcher, self.assertLogs(vllm_logger, level="INFO") as logs:
            results = worker.writeback_hole_recovery_regions()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["result"], -706)
        self.assertEqual(results[0]["stage"], "remove")
        self.assertNotIn("batch_put_start", store.call_names())
        retry_lines = [line for line in logs.output if "HOLEMISS_WRITEBACK_REMOVE_RETRY" in line]
        self.assertEqual(len(retry_lines), 24)
        self.assertEqual([round(ms) for ms in slept], [5000] * 24)


if __name__ == "__main__":
    unittest.main()
