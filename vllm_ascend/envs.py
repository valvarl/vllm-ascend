#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# This file is mainly Adapted from vllm-project/vllm/vllm/envs.py
# Copyright 2023 The vLLM team.
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
#

import os
from collections.abc import Callable
from typing import Any

# The begin-* and end* here are used by the documentation generator
# to extract the used env vars.

# begin-env-vars-definition


def _strict_binary_env(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default)
    if value not in {"0", "1"}:
        raise ValueError(f"{name} must be either '0' or '1', got {value!r}")
    return value == "1"


env_variables: dict[str, Callable[[], Any]] = {
    # max compile thread number for package building. Usually, it is set to
    # the number of CPU cores. If not set, the default value is None, which
    # means all number of CPU cores will be used.
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # The build type of the package. It can be one of the following values:
    # Release, Debug, RelWithDebugInfo. If not set, the default value is Release.
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # Whether to compile custom kernels. If not set, the default value is True.
    # If set to False, the custom kernels will not be compiled.
    # This configuration option should only be set to False when running UT
    # scenarios in an environment without an NPU. Do not set it to False in
    # other scenarios.
    "COMPILE_CUSTOM_KERNELS": lambda: bool(int(os.getenv("COMPILE_CUSTOM_KERNELS", "1"))),
    # The CXX compiler used for compiling the package. If not set, the default
    # value is None, which means the system default CXX compiler will be used.
    "CXX_COMPILER": lambda: os.getenv("CXX_COMPILER", None),
    # The C compiler used for compiling the package. If not set, the default
    # value is None, which means the system default C compiler will be used.
    "C_COMPILER": lambda: os.getenv("C_COMPILER", None),
    # The version of the Ascend chip. It's used for package building.
    # If not set, we will query chip info through `npu-smi`.
    # Please make sure that the version is correct.
    "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
    # If set, vllm-ascend will print verbose logs during compilation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # The home path for CANN toolkit. If not set, the default value is
    # /usr/local/Ascend/ascend-toolkit/latest
    "ASCEND_HOME_PATH": lambda: os.getenv("ASCEND_HOME_PATH", None),
    # The path for HCCL library, it's used by pyhccl communicator backend. If
    # not set, the default value is libhccl.so.
    "HCCL_SO_PATH": lambda: os.getenv("HCCL_SO_PATH", None),
    # The version of vllm is installed. This value is used for developers who
    # installed vllm from source locally. In this case, the version of vllm is
    # usually changed. For example, if the version of vllm is "0.9.0", but when
    # it's installed from source, the version of vllm is usually set to "0.9.1".
    # In this case, developers need to set this value to "0.9.0" to make sure
    # that the correct package is installed.
    "VLLM_VERSION": lambda: os.getenv("VLLM_VERSION", None),
    # Whether to anbale dynamic EPLB
    "DYNAMIC_EPLB": lambda: os.getenv("DYNAMIC_EPLB", "false").lower(),
    # Control the aclrtMemcpyBatchAsync compile path for KV cache offloading.
    # "1": force enable, "0": force disable, None: auto-detect from CANN headers.
    "VLLM_ASCEND_ENABLE_BATCH_MEMCPY": lambda: os.getenv("VLLM_ASCEND_ENABLE_BATCH_MEMCPY", None),
    # Emit per-layer KVPool ranged transfer audit events. Default: 0 (disabled).
    # Valid values: 0 or 1. This configuration is not sensitive.
    "VLLM_ASCEND_KVPOOL_RANGE_DEBUG": lambda: _strict_binary_env("VLLM_ASCEND_KVPOOL_RANGE_DEBUG"),
    # Override the Unified Buffer (UB) size in KB for Triton kernel tile sizing.
    # 0 (default): auto-detect from device properties, falling back to 192 KB
    # (safe for Ascend 910B/A3). Set to a positive value to override when
    # auto-detection is unavailable or for debugging UB overflow issues.
    "VLLM_ASCEND_ROPE_UB_SIZE_KB": lambda: int(os.getenv("VLLM_ASCEND_ROPE_UB_SIZE_KB") or 0),
    # ------------------------------------------------------------------
    # Holemiss hole-recovery / fault-injection contract.
    #
    # These names are shared verbatim with the Mooncake C++ disk-EIO gate and
    # the vLLM HR mixin, so every read stays a live os.getenv: a value changed
    # between two GET windows must be observed immediately. They are declared
    # here as the single registry; the HP test
    # tests/ut/distributed/ascend_store/test_holemiss_scheduler_copies.py pins
    # that every HOLEMISS_* literal used in vllm_ascend has an entry below.
    # ------------------------------------------------------------------
    # Holemiss runtime diagnostics profile. Valid values: "perf", "evidence",
    # "debug". Default: "perf" (minimal diagnostics).
    "HOLEMISS_RUNTIME_PROFILE": lambda: os.getenv("HOLEMISS_RUNTIME_PROFILE", "perf"),
    # Emit the per-layer HOLEMISS diagnostic traces. Default: 0 (disabled).
    "HOLEMISS_DIAG": lambda: os.getenv("HOLEMISS_DIAG", "0") == "1",
    # Force a serialized pre-forward drain of the block-key session load.
    # Default: 0 (native interleaved pipeline).
    "HOLEMISS_SYNC_LAYERWISE_LOAD": lambda: os.getenv("HOLEMISS_SYNC_LAYERWISE_LOAD", "0") == "1",
    # Emit the HOLEMISS_KV_LOAD_DURATION timing channel. Default: 0 (disabled).
    "HOLEMISS_KV_LOAD_TIMING": lambda: os.getenv("HOLEMISS_KV_LOAD_TIMING", "0") == "1",
    # G12 rehearsal lever: disable the GDN state-save channel so the T15
    # missing-state arm exercises the ladder's zero-recompute rung. This gates
    # the SAVE capability signal the ladder sees, NOT use_layerwise (the bulk
    # whole-object path still saves when use_layerwise=false — cfgoff is not
    # save-off). Default: 0 (save enabled).
    "HOLEMISS_STATE_SAVE_DISABLED": lambda: os.getenv("HOLEMISS_STATE_SAVE_DISABLED", "0") == "1",
    # B32: opt-in bypass of the hybrid+recompute kv-load-failure-policy gate
    # (platform.py _validate_kv_load_failure_policy). Default 0 = the gate
    # INTACT (the upstream behavior). The fork's T14 state-load ladder
    # supports the rung-2 zero-recompute path for hybrid models by design;
    # the T15 (d) missing-state rehearsal arm opts in. Values "0"/"1".
    "HOLEMISS_ALLOW_HYBRID_RECOMPUTE_POLICY": lambda: os.getenv("HOLEMISS_ALLOW_HYBRID_RECOMPUTE_POLICY", "0") == "1",
    # Master switch for the real disk-EIO injection family. Default: 0 (off).
    "HOLEMISS_DISK_EIO_ENABLED": lambda: os.getenv("HOLEMISS_DISK_EIO_ENABLED", "0") == "1",
    # Synthetic fault-injector layer id. A non-negative digit string enables
    # the synthetic family and takes priority over the disk-EIO family.
    # Raw string: the arming priority test is `.isdigit()` on the value.
    # Default: "-1" (disabled).
    "HOLEMISS_FAIL_LAYER": lambda: os.getenv("HOLEMISS_FAIL_LAYER", "-1"),
    # Synthetic fault-injector logical block id. Default: "-1".
    "HOLEMISS_FAIL_LOGICAL_BLOCK": lambda: os.getenv("HOLEMISS_FAIL_LOGICAL_BLOCK", "-1"),
    # Disk-EIO target store-local (FA-group) layer id. Default: -1.
    "HOLEMISS_EIO_LAYER": lambda: int(os.getenv("HOLEMISS_EIO_LAYER", "-1")),
    # Disk-EIO target logical block id. Default: 2.
    "HOLEMISS_EIO_LOGICAL_BLOCK": lambda: int(os.getenv("HOLEMISS_EIO_LOGICAL_BLOCK", "2")),
    # HR execution mode. Valid values: "hr_sequential", "hr_gap".
    # Default: "hr_sequential".
    "HOLEMISS_RECOVERY_MODE": lambda: os.getenv("HOLEMISS_RECOVERY_MODE", "hr_sequential"),
    # Bench tree root that resolves the hole-recovery hook trio
    # (hole_recovery_hook, hole_recovery_config, hole_recovery_timer).
    # Default: "" (unset; a fault-mode run fails fast without it).
    "HOLEMISS_BENCH_ROOT": lambda: os.getenv("HOLEMISS_BENCH_ROOT", ""),
    # Full Mooncake block-object key armed for disk-EIO injection. An empty
    # value means the injector is off (re-read live per call). Default: "".
    "EIO_TARGET_KEY": lambda: os.getenv("EIO_TARGET_KEY", ""),
}

# end-env-vars-definition


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
