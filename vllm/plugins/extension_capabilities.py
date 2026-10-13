# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stable discovery surface for versioned extension capabilities.

Extension managers should consume this snapshot instead of importing each
implementation module to discover whether a contract exists.  The returned
object intentionally contains data only, so discovery does not activate a
plugin or grant it lifecycle ownership.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, TypedDict

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.correlated_observability import (  # noqa: E501
    KV_TRANSFER_CORRELATED_OBSERVABILITY_API_VERSION,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.observability import (
    KV_TRANSFER_OBSERVABILITY_API_VERSION,
)
from vllm.v1.events import (
    ASYNC_OUTPUT_EVENTS_API_VERSION,
    REQUEST_LIFECYCLE_EVENTS_API_VERSION,
)

EXTENSION_CAPABILITIES_SCHEMA: Final = "vllm.extension-capabilities/v1"
HOST_EXTENSION_API_VERSION: Final = "1.0"


class ExtensionCapabilitySnapshot(TypedDict):
    schema_version: str
    host_api_version: str
    protocols: Mapping[str, str]


_PROTOCOLS: Final = MappingProxyType(
    {
        "vllm.preemption-policy": "1.0",
        "vllm.batch-admission-policy": "1.1",
        "vllm.request-processing-hook": "1.0",
        "vllm.request-lifecycle-events": REQUEST_LIFECYCLE_EVENTS_API_VERSION,
        "vllm.async-output-events": ASYNC_OUTPUT_EVENTS_API_VERSION,
        "vllm.kv-materialization-runtime-control": "1.0",
        "vllm.kv-transfer.observer": KV_TRANSFER_OBSERVABILITY_API_VERSION,
        "vllm.kv-transfer.observer.v2": (
            KV_TRANSFER_CORRELATED_OBSERVABILITY_API_VERSION
        ),
    }
)


def get_extension_capabilities() -> ExtensionCapabilitySnapshot:
    """Return the host-owned, side-effect-free extension contract snapshot."""
    return {
        "schema_version": EXTENSION_CAPABILITIES_SCHEMA,
        "host_api_version": HOST_EXTENSION_API_VERSION,
        "protocols": _PROTOCOLS,
    }


__all__ = [
    "EXTENSION_CAPABILITIES_SCHEMA",
    "HOST_EXTENSION_API_VERSION",
    "ExtensionCapabilitySnapshot",
    "get_extension_capabilities",
]
