# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.observability import (
    KV_TRANSFER_OBSERVABILITY_API_VERSION,
)
from vllm.plugins.extension_capabilities import (
    EXTENSION_CAPABILITIES_SCHEMA,
    HOST_EXTENSION_API_VERSION,
    get_extension_capabilities,
)
from vllm.plugins.request_processing import REQUEST_PROCESSING_HOOK_API_VERSION
from vllm.v1.core.kv_materialization import (
    KV_MATERIALIZATION_RUNTIME_CONTROL_API_VERSION,
)
from vllm.v1.core.sched.batch_admission import (
    BATCH_ADMISSION_POLICY_API_VERSION,
)
from vllm.v1.core.sched.preemption import PREEMPTION_POLICY_API_VERSION
from vllm.v1.events import (
    ASYNC_OUTPUT_EVENTS_API_VERSION,
    REQUEST_LIFECYCLE_EVENTS_API_VERSION,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_extension_capability_snapshot_matches_owned_contracts() -> None:
    snapshot = get_extension_capabilities()

    assert snapshot == {
        "schema_version": EXTENSION_CAPABILITIES_SCHEMA,
        "host_api_version": HOST_EXTENSION_API_VERSION,
        "protocols": {
            "vllm.preemption-policy": PREEMPTION_POLICY_API_VERSION,
            "vllm.batch-admission-policy": BATCH_ADMISSION_POLICY_API_VERSION,
            "vllm.request-processing-hook": REQUEST_PROCESSING_HOOK_API_VERSION,
            "vllm.request-lifecycle-events": (REQUEST_LIFECYCLE_EVENTS_API_VERSION),
            "vllm.async-output-events": ASYNC_OUTPUT_EVENTS_API_VERSION,
            "vllm.kv-materialization-runtime-control": (
                KV_MATERIALIZATION_RUNTIME_CONTROL_API_VERSION
            ),
            "vllm.kv-transfer.observer": KV_TRANSFER_OBSERVABILITY_API_VERSION,
        },
    }


def test_extension_capability_protocols_are_immutable() -> None:
    protocols = get_extension_capabilities()["protocols"]

    try:
        protocols["vllm.example"] = "1.0"  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("host capability protocols must be immutable")
