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
"""B7 consumer-lookup contract pins on the mainline block-key store.

The producer set (``make_full_key`` over every saving stage x head/TP rank)
must equal the consumer query set (``make_hit_check_keys``), and the legacy
per-layer lookup path must fail LOUD under hybrid rather than silently
enumerating physical layer keys that can never match a block-key store.
CPU-only; the store backend is always a MagicMock.
"""

import inspect
import unittest
from unittest.mock import MagicMock

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.layerwise_keys import (
    LayerwiseKeyBuilder,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend.mooncake_layerwise import (
    make_block_key,
)
from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_scheduler import (
    KVPoolScheduler,
)

MODEL_NAME = "testmodel"
BLOCK_HASH = "5942662ba27e38a478793c541c01708766667a15c12c86a3306c88ff66301a5b"


def _builder(pp_size: int = 1, namespace: str = "") -> LayerwiseKeyBuilder:
    return LayerwiseKeyBuilder(
        make_full_key=lambda group_id, block_hash, rank, stage: make_block_key(
            MODEL_NAME, block_hash, rank, namespace=namespace, pp_rank=stage
        ),
        pp_size=pp_size,
    )


def _scheduler(use_hybrid: bool, tp_size: int = 2, put_step: int = 1, pp_size: int = 1, namespace: str = ""):
    scheduler = KVPoolScheduler.__new__(KVPoolScheduler)
    scheduler.use_hybrid = use_hybrid
    scheduler.use_layerwise = True
    scheduler.layerwise_keys = _builder(pp_size, namespace)
    scheduler.tp_size = tp_size
    scheduler.put_step = put_step
    scheduler.dcp_size = 1
    scheduler.pp_size = pp_size
    scheduler.pp_rank = 0
    scheduler.pp_layer_offset = 0
    scheduler.model_name = MODEL_NAME
    scheduler.num_layers = 4
    scheduler._block_size = 128
    scheduler.kv_cache_group_families = ["default"]
    scheduler.store_scheduler = MagicMock()
    return scheduler


class TestProducerConsumerSetEquivalence(unittest.TestCase):
    def test_consumer_query_set_equals_producer_key_set(self):
        scheduler = _scheduler(use_hybrid=False, tp_size=2)
        producer = {
            make_block_key(MODEL_NAME, BLOCK_HASH, rank, pp_rank=stage)
            for stage in range(1)
            for rank in range(2)
        }
        consumer = set(scheduler._make_layerwise_hit_check_keys(0, BLOCK_HASH))
        self.assertEqual(consumer, producer)

    def test_multi_stage_consumer_enumerates_every_saver(self):
        scheduler = _scheduler(use_hybrid=False, tp_size=2, pp_size=3, namespace="ppn")
        consumer = set(scheduler._make_layerwise_hit_check_keys(0, BLOCK_HASH))
        expected = {
            make_block_key(MODEL_NAME, BLOCK_HASH, rank, namespace="ppn", pp_rank=stage)
            for stage in range(3)
            for rank in range(2)
        }
        self.assertEqual(consumer, expected)
        self.assertEqual(len(consumer), 6)

    def test_put_step_dedupes_consumer_ranks(self):
        scheduler = _scheduler(use_hybrid=False, tp_size=4, put_step=2)
        consumer = set(scheduler._make_layerwise_hit_check_keys(0, BLOCK_HASH))
        self.assertEqual(len(consumer), 2)


class TestB7LoudGuard(unittest.TestCase):
    def test_legacy_lookup_under_hybrid_raises_loud(self):
        scheduler = _scheduler(use_hybrid=True)
        with self.assertRaises(RuntimeError):
            scheduler._generate_store_query_keys([BLOCK_HASH], include_layers=True)

    def test_text_lookup_still_enumerates_layers(self):
        scheduler = _scheduler(use_hybrid=False)
        keys_by_block = scheduler._generate_store_query_keys([BLOCK_HASH], include_layers=True)
        self.assertEqual(len(keys_by_block), 1)
        self.assertEqual(len(keys_by_block[0]), 8)
        for key in keys_by_block[0]:
            self.assertIn("@layer_id:", key)

    def test_hybrid_blockless_lookup_does_not_raise(self):
        scheduler = _scheduler(use_hybrid=True)
        keys_by_block = scheduler._generate_store_query_keys([BLOCK_HASH], include_layers=False)
        self.assertEqual(len(keys_by_block), 1)
        self.assertEqual(len(keys_by_block[0]), 2)
        for key in keys_by_block[0]:
            self.assertIn("@cache_role:kv", key)


class TestB7SourcePin(unittest.TestCase):
    def test_guard_precedes_enumeration(self):
        source = inspect.getsource(KVPoolScheduler._generate_store_query_keys)
        self.assertIn("include_layers and bool(self.use_hybrid)", source)
        self.assertLess(source.index("include_layers and bool(self.use_hybrid)"), source.index("head_or_tp_ranks"))


if __name__ == "__main__":
    unittest.main()
