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
"""T6 scheduler-copy coverage pins (B5 law) + holemiss env registry completeness.

A scheduler fix in one copy is bypassed by whichever copy is live, so the B15
zero-skip call must land in EVERY copy of ``schedule()`` and every copy that
re-copies an upstream body is pinned here.  These are source pins: they fail
LOUD on a re-copy, which is the exact regression mode the B5 law protects.
"""

import ast
import unittest
from pathlib import Path

import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
from vllm_ascend import envs

_PKG = Path(envs.__file__).resolve().parent

RECOMPUTE = _PKG / "core" / "recompute_scheduler.py"
DYNTRA = _PKG / "core" / "dyntra_lb_scheduler.py"
PROFILING = _PKG / "core" / "scheduler_profiling_chunk.py"
BALANCE = _PKG / "patch" / "platform" / "patch_balance_schedule.py"
BATCH_JOB = _PKG / "core" / "batch_job_aware_scheduler.py"
SHORT_FIRST = _PKG / "core" / "short_request_first_scheduler.py"
PP_MTP = _PKG / "patch" / "platform" / "patch_pp_mtp.py"
PROFILING_PATCH = _PKG / "patch" / "platform" / "patch_profiling_chunk.py"
MODEL_RUNNER = _PKG / "worker" / "model_runner_v1.py"

POPULATION = "self._skip_zeroing_for_connector_load("
POP = "request_queue.pop_request()"
ASYNC = "if load_kv_async:"
CONSUMPTION = "self._get_new_block_ids_to_zero()"
INLINE_TAKE = "self.kv_cache_manager.take_new_block_ids()"
INLINE_SKIP = "self._skip_zero_block_ids.update("


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _require_population_between_pop_and_async(src: str) -> None:
    call = src.index(POPULATION)
    pop = src.rindex(POP, 0, call)
    if pop >= call:
        raise AssertionError("zero-skip population call must follow pop_request()")
    gap = src[pop + len(POP) : call]
    if len(gap) > 120:
        raise AssertionError("zero-skip population call is not adjacent to pop_request()")
    async_index = src.find(ASYNC, call)
    if async_index < 0 or async_index < call:
        raise AssertionError("zero-skip population call must precede the async branch")


class TestZeroSkipPopulation(unittest.TestCase):
    def test_recompute_population_and_consumption(self):
        src = _src(RECOMPUTE)
        _require_population_between_pop_and_async(src)
        self.assertIn(CONSUMPTION, src)
        self.assertNotIn(INLINE_SKIP, src)

    def test_dyntra_population_and_consumption(self):
        src = _src(DYNTRA)
        _require_population_between_pop_and_async(src)
        self.assertIn(CONSUMPTION, src)
        self.assertNotIn(INLINE_SKIP, src)

    def test_profiling_chunk_population_and_consumption(self):
        src = _src(PROFILING)
        _require_population_between_pop_and_async(src)
        self.assertIn(CONSUMPTION, src)
        self.assertNotIn(INLINE_TAKE, src)

    def test_balance_population_and_consumption(self):
        src = _src(BALANCE)
        _require_population_between_pop_and_async(src)
        self.assertIn(CONSUMPTION, src)
        self.assertNotIn(INLINE_TAKE, src)


class TestPopulationPinBites(unittest.TestCase):
    def test_bites_when_call_missing(self):
        with self.assertRaises(ValueError):
            _require_population_between_pop_and_async(f"{POP}\n")

    def test_bites_when_call_moved_after_async_branch(self):
        mutated = f"{POP}\n{ASYNC}\npass\n{POPULATION}\n"
        with self.assertRaises(AssertionError):
            _require_population_between_pop_and_async(mutated)

    def test_bites_when_call_far_from_pop(self):
        filler = "x = 1\n" * 60
        mutated = f"{POP}\n{filler}{POPULATION}\n"
        with self.assertRaises(AssertionError):
            _require_population_between_pop_and_async(mutated)


class TestDelegatingCopies(unittest.TestCase):
    def test_recompute_update_from_output_delegates_to_vllm_proper(self):
        src = _src(RECOMPUTE)
        self.assertIn("super().update_from_output(", src)
        self.assertNotIn("invalid_block_ids", src)

    def test_batch_job_aware_schedule_delegates(self):
        src = _src(BATCH_JOB)
        self.assertIn("return super().schedule(", src)
        self.assertNotIn(INLINE_TAKE, src)

    def test_pp_mtp_update_from_output_wrapper_delegates(self):
        src = _src(PP_MTP)
        self.assertIn("original_update_from_output(", src)
        self.assertNotIn("invalid_block_ids", src)

    def test_profiling_chunk_wrapper_delegates(self):
        src = _src(PROFILING_PATCH)
        self.assertIn("_original_update_from_output(", src)
        self.assertIn("_original_schedule(", src)
        self.assertNotIn("invalid_block_ids", src)

    def test_short_request_first_defines_no_scheduler_override(self):
        src = _src(SHORT_FIRST)
        self.assertNotIn("def schedule(", src)
        self.assertNotIn("def update_from_output(", src)


class TestModelRunnerSeam(unittest.TestCase):
    def test_model_runner_passes_itself_for_hr_arming(self):
        src = _src(MODEL_RUNNER)
        start = src.index("self.maybe_get_kv_connector_output(")
        end = src.index(") as kv_connector_output", start)
        self.assertIn("model_runner=self", src[start:end])


EXPECTED_REGISTRY = {
    "HOLEMISS_RUNTIME_PROFILE",
    "HOLEMISS_DIAG",
    "HOLEMISS_SYNC_LAYERWISE_LOAD",
    "HOLEMISS_KV_LOAD_TIMING",
    "HOLEMISS_DISK_EIO_ENABLED",
    "HOLEMISS_FAIL_LAYER",
    "HOLEMISS_FAIL_LOGICAL_BLOCK",
    "HOLEMISS_EIO_LAYER",
    "HOLEMISS_EIO_LOGICAL_BLOCK",
    # W1-10/W1-11 (hybrid-robustness-hunt): the multi-target list and the
    # GDN-state opt-in gate joined the fault-family registry.
    "HOLEMISS_EIO_LAYERS",
    "HOLEMISS_GDN_EIO_ENABLED",
    "HOLEMISS_EIO_GROUP",
    "HOLEMISS_RECOVERY_MODE",
    "HOLEMISS_BENCH_ROOT",
    "HOLEMISS_ALLOW_HYBRID_RECOMPUTE_POLICY",
    "EIO_TARGET_KEY",
}


def _module_string_constants(tree: ast.AST) -> dict[str, str]:
    consts: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    consts[target.id] = node.value.value
    return consts


def _env_read_names(tree: ast.AST) -> set[str]:
    consts = _module_string_constants(tree)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("getenv", "get"):
            owner = node.func.value
            is_os = isinstance(owner, ast.Name) and owner.id == "os"
            is_environ = isinstance(owner, ast.Attribute) and owner.attr == "environ"
            if not (is_os or is_environ) or not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                names.add(arg.value)
            elif isinstance(arg, ast.Name) and arg.id in consts:
                names.add(consts[arg.id])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in (
            "_hm_env_int",
            "_hm_env_int_list",
        ):
            if not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                names.add(arg.value)
            elif isinstance(arg, ast.Name) and arg.id in consts:
                # W1-10/W1-11 hardening: a module-level CONSTANT env name
                # (e.g. HOLEMISS_EIO_LAYERS_ENV) must resolve through the
                # same registry law as a literal - the constant indirection
                # is not an exemption.
                names.add(consts[arg.id])
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "environ":
            index = node.slice
            if isinstance(index, ast.Constant) and isinstance(index.value, str):
                names.add(index.value)
            elif isinstance(index, ast.Name) and index.id in consts:
                names.add(consts[index.id])
    return names


class TestEnvRegistryCompleteness(unittest.TestCase):
    def test_every_holemiss_read_is_registered(self):
        used: set[str] = set()
        for path in _PKG.rglob("*.py"):
            used |= {n for n in _env_read_names(ast.parse(_src(path))) if n.startswith("HOLEMISS_")}
        registered = {name for name in envs.env_variables if name.startswith("HOLEMISS_")}
        self.assertEqual(sorted(used - registered), [])

    def test_expected_registry_present(self):
        missing = EXPECTED_REGISTRY - set(envs.env_variables)
        self.assertEqual(sorted(missing), [])

    def test_registry_entry_is_live_getenv(self):
        import os

        os.environ["HOLEMISS_RECOVERY_MODE"] = "hr_gap"
        try:
            self.assertEqual(envs.HOLEMISS_RECOVERY_MODE, "hr_gap")
        finally:
            os.environ.pop("HOLEMISS_RECOVERY_MODE", None)
        self.assertEqual(envs.HOLEMISS_RECOVERY_MODE, "hr_sequential")


if __name__ == "__main__":
    unittest.main()
