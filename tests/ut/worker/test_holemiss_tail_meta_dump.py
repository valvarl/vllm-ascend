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
"""CPU-only unit tests for the B43 tail metadata dump probe (the runner half).

The W2R2C2-A1 adjudication localized B43 to the compute-side FIA
consumption: the tail page's bytes are proven real end-to-end (the
save/load whole-page sha pair matches exactly), the kernel-table
construction chain is verified at source, yet the T2 sequence is
bit-identical to the zero-attendance class across four runs. The residual
distinction this probe measures at the T2 forward's last pre-kernel point:
(a) a va-side metadata-refresh bug — the runtime tensors the kernel
consumes (the FA block-table entries for the tail positions /
``actual_seq_lengths_kv``) do not reflect the verified derivation; vs
(b) a kernel-side defect — the runtime metadata is correct but the FIA
kernel reads zeros anyway (the owner escalation evidence).

These tests construct the REAL surfaces: a hybrid BlockTable (1536-token
pool pages expanded to 12 kernel rows each — the live dual-block-space
geometry, ``kernel_bs=128 slot_bs=1536 bpkb=12``) whose ``add_row`` runs
the real ``_convert_physical_to_logical_blocks`` expansion, and a real
``AscendMetadata`` carrying the committed table tensor plus the
``seq_lens_list`` — the same fields ``_get_fia_params`` feeds the FIA
kernel verbatim.
"""

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.distributed.parallel_state import GroupCoordinator

from vllm_ascend.attention.attention_v1 import AscendAttentionState, AscendMetadata
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.worker.model_runner_v1 import logger as runner_logger

GRANULARITY = 1536  # the pool page size (the live "Setting attention block size")
KERNEL_BLOCK = 128  # the kernel-granularity row (the live block_shape[0])
BPKB = GRANULARITY // KERNEL_BLOCK  # 12 kernel rows per pool page
BOUNDARY = 9472  # 9216 cached + 256 incoming (the live arm's tail boundary)
FLOOR = 9216
TAIL_BLOCK_INDEX = 6
TAIL_PAGE = 7  # the consumer's destination page (the live run's fa_blocks[6])
FA_PAGES = [10, 11, 12, 13, 14, 15, TAIL_PAGE]
NUM_POOL_PAGES = 12
KV_HEADS = 4
HEAD_DIM = 128
REQ_ID = "0-acc011f8"
LAYER_NAME = "model.layers.3.self_attn"


def _hybrid_block_table():
    """A REAL hybrid BlockTable: 1536-token pages expanded to 12 kernel-row
    ids per page (the live dual-block-space geometry). ``add_row`` runs the
    real ``_convert_physical_to_logical_blocks`` expansion, so the committed
    tensor carries exactly the values the live table construction produces."""
    with patch("vllm_ascend.worker.block_table.get_dcp_group") as mock_get_dcp_group:
        mock_dcp_group = MagicMock(spec=GroupCoordinator)
        mock_dcp_group.world_size = 1
        mock_dcp_group.rank_in_group = 0
        mock_get_dcp_group.return_value = mock_dcp_group

        from vllm_ascend.worker.block_table import BlockTable

        table = BlockTable(
            block_size=GRANULARITY,
            max_num_reqs=4,
            max_num_blocks_per_req=16,
            max_num_batched_tokens=512,
            pin_memory=False,
            device=torch.device("cpu"),
            kernel_sizes=[KERNEL_BLOCK],
            kv_cache_group=None,
        )
    table.add_row(FA_PAGES, 0)
    table.commit_block_table(1)
    return table


def _runner(block_table, req_id_to_index=None):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.input_batch = SimpleNamespace(
        req_id_to_index=req_id_to_index if req_id_to_index is not None else {REQ_ID: 0},
        block_table={0: block_table},
    )
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=[LAYER_NAME])])
    return runner


def _attn_metadata(block_table, seq_kv=BOUNDARY):
    return AscendMetadata(
        block_tables=block_table.get_device_tensor(),
        seq_lens=torch.tensor([seq_kv], dtype=torch.int32),
        seq_lens_list=[seq_kv],
        attn_state=AscendAttentionState.DecodeOnly,
    )


def _attn_metadata_dict(block_table, seq_kv=BOUNDARY):
    """The per-layer metadata dict the forward context carries (all group-0
    FA layers share one AscendMetadata object)."""
    return {LAYER_NAME: _attn_metadata(block_table, seq_kv=seq_kv)}


def _record(**overrides):
    record = {
        "req_id": REQ_ID,
        "tp": 0,
        "boundary": BOUNDARY,
        "tail_block": TAIL_BLOCK_INDEX,
        "tail_page": TAIL_PAGE,
        "claim": BOUNDARY - 1,
    }
    record.update(overrides)
    return record


def _run_probe(runner, attn_metadata, records):
    """Drive the REAL duck-typing chain: the runner's take method reaches
    the connector's connector_worker and pops the worker's records."""
    take = _PoppingTake(records)
    connector = SimpleNamespace(connector_worker=SimpleNamespace(holemiss_take_tail_meta_records=take))
    with (
        patch(
            "vllm_ascend.worker.model_runner_v1.has_kv_transfer_group",
            return_value=True,
        ),
        patch(
            "vllm_ascend.worker.model_runner_v1.get_kv_transfer_group",
            return_value=connector,
        ),
    ):
        return runner._holemiss_probe_tail_attention_metadata(attn_metadata)


class _PoppingTake:
    """The worker take stub: pops like the real fire-once method."""

    def __init__(self, records):
        self.pending = list(records)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        records = self.pending
        self.pending = []
        return records


class TestHybridBlockTableExpansion(unittest.TestCase):
    """The expected-values derivation pinned against the REAL expansion —
    computed independently in the test (never through the probe's helper),
    so a mutation in either direction breaks the equality."""

    def test_the_expansion_matches_the_verified_derivation(self):
        table = _hybrid_block_table()
        # The live geometry (the BLOCKSPACE census): 128-token kernel rows,
        # 1536-token pool pages, 12 rows per page.
        self.assertEqual(table.block_size, KERNEL_BLOCK)
        self.assertEqual(table.physical_block_size, GRANULARITY)
        self.assertEqual(table.blocks_per_phys_block, BPKB)
        row = table.get_numpy_array()[0]
        # Positions [9216, 9472) are table entries [72, 74); the values are
        # the tail page's first kernel rows: 7 * 12 = 84, 85.
        self.assertEqual(row[72:74].tolist(), [TAIL_PAGE * BPKB, TAIL_PAGE * BPKB + 1])
        # The entry just before the tail page belongs to the previous page.
        self.assertEqual(row[71], FA_PAGES[5] * BPKB + BPKB - 1)
        # The committed tensor equals the CPU row (the commit contract the
        # probe's three-way comparison relies on).
        self.assertEqual(
            table.get_device_tensor()[0, 72:74].tolist(),
            row[72:74].tolist(),
        )


class TestTailMetadataDumpProbe(unittest.TestCase):
    def _assert_marker(self, logs, expected_substrings):
        marker_lines = [line for line in logs.output if "HOLEMISS_TAIL_META_DUMP" in line]
        self.assertEqual(len(marker_lines), 1)
        for substring in expected_substrings:
            self.assertIn(substring, marker_lines[0])
        return marker_lines[0]

    def test_dump_matches_the_derived_expectation(self):
        """The happy path: the kernel-consumed tensors reflect the verified
        derivation — verdict META_CORRECT (the (b) kernel-side branch)."""
        table = _hybrid_block_table()
        runner = _runner(table)
        attn_metadata = _attn_metadata_dict(table)
        with self.assertLogs(runner_logger, level="INFO") as logs:
            _run_probe(runner, attn_metadata, [_record()])
        self._assert_marker(
            logs,
            [
                "rows=72-73",
                "values=[84, 85]",
                "seq_kv=9472",
                "expected_values=[84, 85]",
                "expected_kv=9472",
                "verdict=META_CORRECT",
                "cpu_values=[84, 85]",
                "tail_page=7",
                "tail_block=6",
                "kernel_bs=128",
                "bpkb=12",
                "state=DecodeOnly",
            ],
        )

    def test_stale_runtime_table_reports_the_va_branch(self):
        """The (a) class: the metadata's table tensor carries wrong entries
        for the tail positions (the stale/not-refreshed runtime tensor)
        while the CPU-side row is correct — the mismatch localizes to the
        commit/runtime layer, not the allocation."""
        table = _hybrid_block_table()
        stale = table.get_device_tensor().clone()
        stale[0, 72:74] = 0
        attn_metadata = _attn_metadata_dict(table)
        attn_metadata[LAYER_NAME].block_tables = stale
        runner = _runner(table)
        with self.assertLogs(runner_logger, level="INFO") as logs:
            _run_probe(runner, attn_metadata, [_record()])
        self._assert_marker(
            logs,
            [
                "values=[0, 0]",
                "verdict=TABLE_MISMATCH",
                "cpu_values=[84, 85]",
                "seq_kv=9472",
            ],
        )

    def test_truncated_seq_len_reports_the_va_branch(self):
        """The (a) class, seq-len variant: an effective KV length stopping
        at the 9216 floor — the content-invariance sharpening's candidate."""
        table = _hybrid_block_table()
        runner = _runner(table)
        attn_metadata = _attn_metadata_dict(table, seq_kv=FLOOR)
        with self.assertLogs(runner_logger, level="INFO") as logs:
            _run_probe(runner, attn_metadata, [_record()])
        self._assert_marker(
            logs,
            [
                "values=[84, 85]",
                "seq_kv=9216",
                "expected_kv=9472",
                "verdict=SEQ_LEN_MISMATCH",
            ],
        )

    def test_both_mismatched_reports_the_combined_branch(self):
        table = _hybrid_block_table()
        stale = table.get_device_tensor().clone()
        stale[0, 72:74] = 0
        attn_metadata = _attn_metadata_dict(table, seq_kv=FLOOR)
        attn_metadata[LAYER_NAME].block_tables = stale
        runner = _runner(table)
        with self.assertLogs(runner_logger, level="INFO") as logs:
            _run_probe(runner, attn_metadata, [_record()])
        self._assert_marker(
            logs,
            ["verdict=TABLE_AND_SEQ_MISMATCH", "values=[0, 0]", "seq_kv=9216"],
        )

    def test_no_records_is_a_noop(self):
        """The positive-signal discipline: without a tail-hit registration
        the probe never emits (no bench registration, no noise)."""
        table = _hybrid_block_table()
        runner = _runner(table)
        with self.assertNoLogs(runner_logger, level="INFO"):
            _run_probe(runner, _attn_metadata_dict(table), [])

    def test_request_left_batch_reports_the_drop(self):
        table = _hybrid_block_table()
        runner = _runner(table, req_id_to_index={})
        with self.assertLogs(runner_logger, level="INFO") as logs:
            _run_probe(runner, _attn_metadata_dict(table), [_record()])
        self._assert_marker(
            logs,
            ["verdict=REQ_LEFT_BATCH", f"req={REQ_ID}", "boundary=9472", "tail_page=7"],
        )

    def test_call_site_runs_the_probe_before_the_model_forward(self):
        """The source pin: inside execute_model's forward-context/connector
        block — after the connector context entry (the sync load drain) and
        before _model_forward, the last pre-kernel point."""
        source = inspect.getsource(NPUModelRunner.execute_model)
        connector_idx = source.index("self.maybe_get_kv_connector_output(")
        probe_idx = source.index("self._holemiss_probe_tail_attention_metadata(attn_metadata)")
        forward_idx = source.index("self._model_forward(")
        self.assertLess(connector_idx, probe_idx)
        self.assertLess(probe_idx, forward_idx)


def _fa_caches(populate_page=None):
    """The group-0 FA caches in the live layout: attention layers at
    physical 3+4k, each a (K, V) pair over 128-token KERNEL rows (12 rows
    per 1536-token pool page)."""
    caches = {}
    for layer_idx in range(2):
        physical = 3 + 4 * layer_idx
        shape = (NUM_POOL_PAGES * BPKB, KERNEL_BLOCK, KV_HEADS, HEAD_DIM)
        k = torch.zeros(shape, dtype=torch.bfloat16)
        v = torch.zeros(shape, dtype=torch.bfloat16)
        if populate_page is not None:
            gen = torch.Generator().manual_seed(physical)
            row0 = populate_page * BPKB
            k[row0 : row0 + 2] = torch.randn(2, KERNEL_BLOCK, KV_HEADS, HEAD_DIM, generator=gen, dtype=torch.bfloat16)
            v[row0 : row0 + 2] = torch.randn(2, KERNEL_BLOCK, KV_HEADS, HEAD_DIM, generator=gen, dtype=torch.bfloat16)
        caches[f"model.layers.{physical}.self_attn"] = (k, v)
    return caches


def _store_worker(caches):
    """A REAL KVPoolWorker (the __new__ harness pattern) with just the
    surface the page walk + the b2 probes need."""
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import KVPoolWorker

    worker = KVPoolWorker.__new__(KVPoolWorker)
    worker.tp_rank = 0
    worker.use_hybrid = True
    worker.hf_config = None
    worker.num_layers = 64
    worker.kv_cache_config = None
    worker.kv_caches = caches
    worker.num_blocks = NUM_POOL_PAGES
    worker._global_to_local_layer = {}
    worker.grouped_block_size = [GRANULARITY, GRANULARITY]
    worker._holemiss_tail_meta_records = {}
    worker._holemiss_tail_stable_pre = {}
    return worker


class TestTailPostForwardChecksum(unittest.TestCase):
    """The b2 closure's runner half: the post-forward checksum rides the
    records the pre-probe consumed (register -> META_DUMP -> forward ->
    post-checksum), through the same duck-typed worker chain."""

    def _patched(self, worker):
        return (
            patch(
                "vllm_ascend.worker.model_runner_v1.has_kv_transfer_group",
                return_value=True,
            ),
            patch(
                "vllm_ascend.worker.model_runner_v1.get_kv_transfer_group",
                return_value=SimpleNamespace(connector_worker=worker),
            ),
        )

    def test_pre_probe_returns_the_consumed_records(self):
        table = _hybrid_block_table()
        runner = _runner(table)
        records = _run_probe(runner, _attn_metadata_dict(table), [_record()])
        self.assertEqual(records, [_record()])

    def test_post_checksum_emits_per_record_through_the_duck_chain(self):
        worker = _store_worker(_fa_caches(populate_page=TAIL_PAGE))
        table = _hybrid_block_table()
        runner = _runner(table)
        has_patch, get_patch = self._patched(worker)
        with self.assertLogs(runner_logger, level="INFO") as logs, has_patch, get_patch:
            runner._holemiss_probe_tail_page_checksum([_record()])
        marker_lines = [line for line in logs.output if "side=post" in line]
        self.assertEqual(len(marker_lines), 1)
        self.assertIn("verdict=NO_PRE_SNAPSHOT", marker_lines[0])
        self.assertIn(f"block={TAIL_PAGE}", marker_lines[0])
        self.assertIn("boundary=9472", marker_lines[0])
        self.assertIn("tail_nonzero=True", marker_lines[0])

    def test_post_checksum_is_a_noop_without_records(self):
        table = _hybrid_block_table()
        runner = _runner(table)
        with self.assertNoLogs(runner_logger, level="INFO"):
            runner._holemiss_probe_tail_page_checksum([])

    def test_post_checksum_tolerates_a_worker_without_the_method(self):
        table = _hybrid_block_table()
        runner = _runner(table)
        has_patch, get_patch = self._patched(SimpleNamespace())
        with self.assertNoLogs(runner_logger, level="INFO"), has_patch, get_patch:
            runner._holemiss_probe_tail_page_checksum([_record()])

    def test_post_checksum_skips_requests_that_left_the_batch(self):
        worker = _store_worker(_fa_caches(populate_page=TAIL_PAGE))
        table = _hybrid_block_table()
        runner = _runner(table, req_id_to_index={})
        has_patch, get_patch = self._patched(worker)
        with self.assertNoLogs(runner_logger, level="INFO"), has_patch, get_patch:
            runner._holemiss_probe_tail_page_checksum([_record()])

    def test_the_lifecycle_take_snapshot_dump_then_post_checksum(self):
        """The full chain on REAL classes: the worker's registration dict ->
        the runner's pre probe (the take, the META_DUMP, the stable
        snapshot) -> the forward boundary -> the post checksum with the
        CLOBBER_EXCLUDED verdict; both record stores consumed fire-once."""
        worker = _store_worker(_fa_caches(populate_page=TAIL_PAGE))
        worker._holemiss_tail_meta_records[REQ_ID] = _record()
        table = _hybrid_block_table()
        runner = _runner(table)
        has_patch, get_patch = self._patched(worker)
        with self.assertLogs(runner_logger, level="INFO") as logs, has_patch, get_patch:
            records = runner._holemiss_probe_tail_attention_metadata(_attn_metadata_dict(table))
            runner._holemiss_probe_tail_page_checksum(records)
        self.assertEqual(records, [_record()])
        meta_lines = [line for line in logs.output if "HOLEMISS_TAIL_META_DUMP" in line]
        post_lines = [line for line in logs.output if "side=post" in line]
        self.assertEqual(len(meta_lines), 1)
        self.assertEqual(len(post_lines), 1)
        self.assertIn("verdict=META_CORRECT", meta_lines[0])
        self.assertIn("verdict=CLOBBER_EXCLUDED", post_lines[0])
        self.assertIn("tail_nonzero=True", post_lines[0])
        stable = post_lines[0].split("stable_sha=")[1].split()[0]
        self.assertIn(f"pre_stable_sha={stable}", post_lines[0])
        self.assertEqual(worker._holemiss_tail_meta_records, {})
        self.assertEqual(worker._holemiss_tail_stable_pre, {})

    def test_call_site_runs_the_post_checksum_after_the_model_forward(self):
        """The source pin: the post checksum fires after _model_forward
        (and inside the forward-context block, before its exit)."""
        source = inspect.getsource(NPUModelRunner.execute_model)
        probe_idx = source.index("self._holemiss_probe_tail_attention_metadata(attn_metadata)")
        forward_idx = source.index("self._model_forward(")
        post_idx = source.index("self._holemiss_probe_tail_page_checksum(tail_meta_records)")
        block_end_idx = source.index("self.kvpp.complete_forward()")
        self.assertLess(probe_idx, forward_idx)
        self.assertLess(forward_idx, post_idx)
        self.assertLess(post_idx, block_end_idx)


if __name__ == "__main__":
    unittest.main()
