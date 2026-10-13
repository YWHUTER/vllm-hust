# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Default-off typed scheduler and worker lifecycle events.

This module provides a process-local typed event outlet for the vLLM
scheduler. It is intentionally generic: it exposes request-level lifecycle
events (a request finished, a request was preempted, its KV state was
actually reclaimed) with no experiment- or product-specific semantics.
Out-of-tree consumers (e.g. context-lifecycle management plugins) register a
sink via :meth:`EventBus.register_sink` and receive typed events; with no
sink registered the bus is disabled and every :meth:`EventBus.emit` is a
single no-op guard, so serving overhead is zero.

This is the request/session-level counterpart of the worker-level
``AsyncOutput*`` lifecycle events. The two layers are orthogonal: worker
events describe output-token movement on the GPU path; these events describe
request admission/finish and KV reclamation at the scheduler.

Event timing contract (enforced by ``tests/v1/test_request_lifecycle_events.py``):

* :class:`RequestFinished` - emitted by ``Scheduler._free_request`` *after*
  the encoder cache has been released and, unless an async connector asked
  for a delayed release, after the KV blocks have been returned. The
  ``kv_reclaim_deferred`` flag is ``True`` in the delayed case: the request
  is finished but its blocks are still owned by the connector. The matching
  :class:`RequestKvReclaimed` arrives later, when the fence step completes.
* :class:`RequestPreempted` - emitted by ``Scheduler._preempt_request`` after
  the preemption has released (or scheduled the deferred release of) the
  request's KV blocks and the request has been put back on the waiting
  queue. This event describes the *scheduling decision*; it is not a claim
  that the KV was returned to the pool.
* :class:`RequestKvReclaimed` - emitted at the exact point the blocks are
  handed back to the block pool, with ``path="immediate"`` for the
  synchronous path and ``path="deferred"`` when a deferred free is drained.
  This is the only event whose occurrence means "the KV for this request is
  really gone".

Token accounting is deliberately split into three non-overlapping quantities
so that consumers never have to guess what a single ``tokens`` field
measured: ``prompt_tokens`` (input length), ``output_tokens`` (generated
tokens), and ``sequence_tokens`` (their sum). ``Request.num_computed_tokens``
is intentionally not mixed in: it tracks prefill progress and re-counts
prompt tokens.
"""

from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Protocol

REQUEST_LIFECYCLE_EVENTS_API_VERSION = "1.0"
ASYNC_OUTPUT_EVENTS_API_VERSION = "1.0"


@dataclass(frozen=True)
class SchedulerEvent:
    """Base class for process-local scheduler events."""

    ts_monotonic_ns: int = field(init=False, default_factory=time.monotonic_ns)


@dataclass(frozen=True)
class RequestFinished(SchedulerEvent):
    """A request finished decoding (stopped or aborted).

    payload: request id, optional session id (vllm-hust session-scoped
    requests), the three non-overlapping token counts, the number of KV
    blocks the request held when its blocks were released, whether that
    release was deferred to a connector, and the terminal finish reason.
    Consumers use this to learn that one turn of a long conversation
    completed and how large its KV footprint was.
    """

    request_id: str
    session_id: str | None
    prompt_tokens: int
    output_tokens: int
    sequence_tokens: int
    kv_blocks: int
    kv_reclaim_deferred: bool
    finished_reason: str


@dataclass(frozen=True)
class RequestPreempted(SchedulerEvent):
    """A running request was preempted and re-queued.

    payload: request id, optional session id, the request's cumulative
    preemption count, the KV blocks the preemption released, and whether the
    release was deferred. Consumers use this to observe KV pressure and react
    (e.g. trigger context compaction) before latency degrades.

    The event reports the scheduling decision. When ``kv_reclaim_deferred``
    is ``True`` the blocks had not yet been returned to the pool at this
    point, and a later :class:`RequestKvReclaimed` will carry the reclaim.
    """

    request_id: str
    session_id: str | None
    num_preemptions: int
    kv_blocks: int
    kv_reclaim_deferred: bool


@dataclass(frozen=True)
class RequestKvReclaimed(SchedulerEvent):
    """KV blocks were returned to the block pool.

    payload: request id, optional session id, the number of blocks returned,
    and which release path performed it (``immediate`` for the synchronous
    return inside ``Scheduler._free_request_blocks``, ``deferred`` for a
    fence-satisfied entry drained by ``Scheduler._drain_deferred_frees``).
    """

    request_id: str
    session_id: str | None
    kv_blocks: int
    path: Literal["immediate", "deferred"]


@dataclass(frozen=True)
class WorkerEvent:
    """Base class for process-local asynchronous output events."""

    ts_monotonic_ns: int = field(init=False, default_factory=time.monotonic_ns)


@dataclass(frozen=True)
class AsyncOutputCreated(WorkerEvent):
    lifecycle_id: int
    request_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    runner_kind: Literal["mrv1", "mrv2"] = "mrv1"


@dataclass(frozen=True)
class AsyncOutputCopyIssued(WorkerEvent):
    lifecycle_id: int
    storage_id: int
    event_id: int
    nbytes: int
    dispatch_ns: int


@dataclass(frozen=True)
class AsyncOutputCopyFailed(WorkerEvent):
    lifecycle_id: int
    phase: str


@dataclass(frozen=True)
class AsyncOutputWaitComplete(WorkerEvent):
    lifecycle_id: int
    wait_ns: int


@dataclass(frozen=True)
class AsyncOutputMaterialized(WorkerEvent):
    lifecycle_id: int
    materialization_ns: int


@dataclass(frozen=True)
class AsyncOutputRetained(WorkerEvent):
    storage_id: int


@dataclass(frozen=True)
class AsyncOutputConsumed(WorkerEvent):
    storage_id: int
    wait_ns: int
    materialization_ns: int


class EventSink(Protocol):
    """Consumer interface for scheduler events."""

    def emit(self, event: SchedulerEvent) -> None: ...


class WorkerEventSink(Protocol):
    """Consumer interface for asynchronous output events."""

    def emit(self, event: WorkerEvent) -> None: ...


class EventBus:
    """Process-local scheduler and worker outlets with independent activation."""

    _sinks: list[EventSink] = []
    _worker_sinks: list[WorkerEventSink] = []
    _lock = threading.Lock()
    enabled = False
    worker_enabled = False

    @classmethod
    def register_sink(cls, sink: EventSink) -> None:
        """Register one sink instance."""
        with cls._lock:
            if sink not in cls._sinks:
                cls._sinks.append(sink)
            cls.enabled = bool(cls._sinks)

    @classmethod
    def unregister_sink(cls, sink: EventSink) -> None:
        """Remove one sink instance."""
        with cls._lock:
            if sink in cls._sinks:
                cls._sinks.remove(sink)
            cls.enabled = bool(cls._sinks)

    @classmethod
    def emit(cls, event: SchedulerEvent) -> None:
        """Dispatch an event, disabling sinks that fail."""
        if not cls.enabled:
            return
        with cls._lock:
            sinks = tuple(cls._sinks)
        for sink in sinks:
            try:
                sink.emit(event)
            except Exception:
                with contextlib.suppress(Exception):
                    cls.unregister_sink(sink)

    @classmethod
    def register_worker_sink(cls, sink: WorkerEventSink) -> None:
        """Enable worker events for one sink without enabling scheduler events."""
        with cls._lock:
            if not any(existing is sink for existing in cls._worker_sinks):
                cls._worker_sinks.append(sink)
            cls.worker_enabled = bool(cls._worker_sinks)

    @classmethod
    def unregister_worker_sink(cls, sink: WorkerEventSink) -> None:
        with cls._lock:
            cls._worker_sinks = [
                existing for existing in cls._worker_sinks if existing is not sink
            ]
            cls.worker_enabled = bool(cls._worker_sinks)

    @classmethod
    def emit_worker(cls, event: WorkerEvent) -> None:
        """Deliver only to worker sinks; isolate failing consumers."""
        if not cls.worker_enabled:
            return
        with cls._lock:
            sinks = tuple(cls._worker_sinks)
        for sink in sinks:
            try:
                sink.emit(event)
            except Exception:
                with contextlib.suppress(Exception):
                    cls.unregister_worker_sink(sink)


__all__ = [
    "ASYNC_OUTPUT_EVENTS_API_VERSION",
    "AsyncOutputConsumed",
    "AsyncOutputCopyFailed",
    "AsyncOutputCopyIssued",
    "AsyncOutputCreated",
    "AsyncOutputMaterialized",
    "AsyncOutputRetained",
    "AsyncOutputWaitComplete",
    "REQUEST_LIFECYCLE_EVENTS_API_VERSION",
    "EventBus",
    "EventSink",
    "RequestFinished",
    "RequestKvReclaimed",
    "RequestPreempted",
    "SchedulerEvent",
    "WorkerEvent",
    "WorkerEventSink",
]
