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
"""W1-15 / R1 Mode 2: the step-end put-lifecycle loud-check.

R1 proved the FA-tail commit chain structurally completes under
instrumentation (the Heisenbug flip: all 16 group-0 layers' rows + the
layer-63 commit rc=0) while the uninstrumented battery loses it to a
timing race — the still-open put-start expires silently at the master 60 s
later (``Offloading task expired``, master_service.cpp:9392) with no
in-engine signal. The check at ``get_finished`` converts that silent loss
into an immediate loud failure: the save fence
(``_wait_for_final_layer_save``) fires only after every queued put of the
step, and the fatal path revokes the whole open set, so ANY still-open
put-start at a step boundary is a broken submission chain.

The fixtures drive the REAL paths (the N18/N19 law): ``get_finished``'s
full body on a stubbed worker (the real teardown + the real check) and —
for the orphan shape — the REAL ``_prepare_layerwise_put_session``
put-start registration followed by the submission break (the B34/H7
class: the tasks never ran). The healthy-closure halves are guarded by the
pool-level batteries that call ``get_finished`` at every step boundary
(test_ascend_store_layerwise_lifecycle / test_partial_tail_persistence /
test_gdn_state_save_wiring / test_text_model_save_driver /
test_request_lifecycle) — any false-positive shape would fail them.
"""

import inspect
import threading
import unittest
from types import SimpleNamespace

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.metadata import (
    AscendConnectorMetadata,
    ReqMeta,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker


class _PutStartStore:
    """The m_store leaf for the real prep drive: put-start succeeds."""

    requires_exists_before_put = False

    def __init__(self):
        self.put_started: list[str] = []

    def batch_put_start(self, keys, sizes):
        self.put_started.extend(keys)
        return [0] * len(keys)


def _make_worker(kv_role="kv_producer"):
    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.tp_rank = 0
    worker.kv_role = kv_role
    worker.consumer_is_to_put = False
    worker.use_block_key_layerwise = True
    worker._put_started_keys = set()
    worker._put_started_keys_lock = threading.Lock()
    # get_finished's teardown + release path.
    worker._hole_recovery_regions = []
    worker._hole_recovery_lock = threading.Lock()
    worker._holemiss_armed_fault = None
    worker._holemiss_state_tail_snapshots = {}
    worker._holemiss_state_tail_snapshot_pending_free = {}
    worker._holemiss_state_tail_seen_reqs = set()
    worker._holemiss_state_tail_sources = {}
    worker._holemiss_state_tail_capture_failed = {}
    worker._holemiss_tail_key_boundaries = {}
    worker._holemiss_tail_save_probed = set()
    worker._holemiss_state_tail_keys = set()
    worker._holemiss_state_markers = {}
    worker._holemiss_state_load_results = {}
    worker._holemiss_failed_load_keys = {}
    worker._holemiss_req_meta_by_id = {}
    worker._holemiss_state_tail_skip_warned = set()
    worker._holemiss_scratch_owner = None
    worker._load_session_lock = threading.Lock()
    worker._layerwise_session_tracker = SimpleNamespace(
        release_terminal=lambda req_ids: [],
        register_put_keys=lambda *args, **kwargs: None,
    )
    worker.layerwise_max_transfer_blocks = 0
    worker.kv_send_thread = None
    worker.kv_recv_thread = None
    return worker


def _metadata():
    return AscendConnectorMetadata(set(), set())


class TestStepPutLifecycleLoudCheck(unittest.TestCase):
    def test_healthy_boundary_is_silent(self):
        """Every key closed within the step (the fence's guarantee): the
        boundary check must be invisible — get_finished returns its tuple
        and never raises."""
        worker = _make_worker()
        done_sending, done_recving = worker.get_finished(set(), _metadata())
        self.assertEqual(done_sending, set())
        self.assertEqual(done_recving, set())

    def test_orphaned_put_start_raises_at_the_boundary(self):
        """The silent-loss signature: a put-start that neither committed
        nor revoked by the step's drain raises immediately (the marker
        carries the open keys), not at the master's 60 s expiry."""
        worker = _make_worker()
        worker._put_started_keys.add("stub-model@abc@0")
        worker._put_started_keys.add("stub-model@def@0")
        with self.assertRaises(RuntimeError) as raised:
            worker.get_finished({"req-A"}, _metadata())
        message = str(raised.exception)
        self.assertIn("HOLEMISS_PUT_LIFECYCLE_BROKEN", message)
        self.assertIn("2 put-started key(s)", message)
        self.assertIn("stub-model@abc@0", message)

    def test_real_prep_orphan_raises(self):
        """The REAL put-start path (the B34/H7 submission-break class): the
        prep opened the sessions, the task chain never ran — the boundary
        check fires on exactly the keys the prep started."""
        worker = _make_worker()
        worker.put_step = 1
        worker.block_size = 16
        worker.hash_block_size = 16
        worker.group_block_len = {0: [4096]}
        worker.page_size_bytes = 4096
        worker.layerwise_keys = SimpleNamespace(
            make_full_key=lambda group, block_hash, rank, pp: f"stub-model@{block_hash}@{rank}"
        )
        worker.head_or_tp_rank = 0
        worker.pp_rank = 0
        worker.m_store = _PutStartStore()
        request = ReqMeta(
            req_id="req-A",
            token_len_chunk=32,
            save_start_token=0,
            save_end_token=32,
            block_ids_by_group=[[7, 8]],
            block_hashes=[bytes([1] * 32), bytes([2] * 32)],
            can_save=True,
        )
        worker._prepare_layerwise_put_session(request)
        self.assertEqual(len(worker._put_started_keys), 2)
        with self.assertRaises(RuntimeError) as raised:
            worker.get_finished(set(), _metadata())
        self.assertIn("HOLEMISS_PUT_LIFECYCLE_BROKEN", str(raised.exception))

    def test_check_is_scoped_to_the_block_key_plane(self):
        """The gate pin: only the block-key layerwise plane put-starts, so
        only its boundaries carry the check."""
        worker = _make_worker()
        worker.use_block_key_layerwise = False
        worker._put_started_keys.add("stub-model@abc@0")
        done_sending, done_recving = worker.get_finished(set(), _metadata())
        self.assertEqual(done_sending, set())

    def test_get_finished_calls_the_check_inside_the_layerwise_gate(self):
        """G20-style source pin: the check rides get_finished's
        block-key-layerwise boundary, before the terminal release."""
        source = inspect.getsource(KVPoolWorker.get_finished)
        self.assertIn("_holemiss_check_step_put_lifecycle(", source)
        gate = source.index("use_block_key_layerwise")
        check = source.index("_holemiss_check_step_put_lifecycle(")
        release = source.index("_release_layerwise_requests_terminal(")
        self.assertLess(gate, check)
        self.assertLess(check, release)


if __name__ == "__main__":
    unittest.main()
