# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real async-output methods with host-only tensor/stream stubs."""

from __future__ import annotations

import ast
import importlib.util
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def events() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_async_output_test_events", ROOT / "vllm/v1/events.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def method_namespace(path: str, class_name: str, namespace: dict[str, Any]) -> None:
    """Execute unchanged methods from the checked-out runner, skipping imports."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    klass.bases = []
    klass.decorator_list = []
    tree.body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        klass,
    ]
    ast.fix_missing_locations(tree)
    exec(compile(tree, path, "exec"), namespace)


class FakeEvent:
    def record(self, *args: object) -> None:
        pass

    def synchronize(self) -> None:
        pass


class FakeStream:
    def wait_stream(self, other: object) -> None:
        pass


class FakeCpuTensor:
    shape = (1, 1)
    dtype = "int64"
    nbytes = 8

    def numel(self) -> int:
        return 1

    def element_size(self) -> int:
        return 8

    def tolist(self) -> list[list[int]]:
        return [[7]]


class FakeTensor:
    shape = (1, 1)
    dtype = "int64"

    def __init__(self, *, fail_copy: bool = False):
        self.fail_copy = fail_copy

    def to(self, device: str, *, non_blocking: bool) -> FakeCpuTensor:
        assert device == "cpu" and non_blocking
        if self.fail_copy:
            raise RuntimeError("copy failed")
        return FakeCpuTensor()


class FakeCountTensor(FakeTensor):
    def to(self, device: str, *, non_blocking: bool) -> SimpleNamespace:
        assert device == "cpu" and non_blocking
        return SimpleNamespace(tolist=lambda: [1])


def runner_namespace(events: ModuleType, path: str, class_name: str) -> dict[str, Any]:
    namespace = {name: getattr(events, name) for name in events.__all__}
    namespace.update(
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                Event=lambda **kwargs: FakeEvent(),
                current_stream=FakeStream,
                stream=lambda stream: nullcontext(),
            )
        ),
        time=time,
        get_ep_all2all_manager=lambda: None,
        envs=SimpleNamespace(VLLM_RAISE_ON_LOGIT_NANS=False),
        async_copy_to_np=lambda tensor: tensor.to("cpu", non_blocking=True),
        stream=lambda *args: nullcontext(),
    )
    method_namespace(path, class_name, namespace)
    return namespace


def model_output() -> SimpleNamespace:
    return SimpleNamespace(
        req_ids=["r"],
        req_id_to_index={"r": 0},
        prompt_logprobs_dict={},
        prompt_token_id_logprobs_dict={},
    )


def run_mrv1(events: ModuleType, *, fail_copy: bool = False) -> SimpleNamespace:
    namespace = runner_namespace(
        events, "vllm/v1/worker/gpu_model_runner.py", "AsyncGPUModelRunnerOutput"
    )
    wrapped = namespace["AsyncGPUModelRunnerOutput"](
        model_output(), FakeTensor(fail_copy=fail_copy), None, [], FakeStream(), 32
    )
    return wrapped.get_output()


def run_mrv2(events: ModuleType) -> SimpleNamespace:
    namespace = runner_namespace(
        events, "vllm/v1/worker/gpu/async_utils.py", "AsyncOutput"
    )
    sampler = SimpleNamespace(
        sampled_token_ids=FakeTensor(),
        logprobs_tensors=None,
        num_nans=None,
        sampling_mask_tensors=None,
    )
    wrapped = namespace["AsyncOutput"](
        model_output(),
        sampler,
        FakeCountTensor(),
        FakeStream(),
        FakeStream(),
        False,
        None,
    )
    return wrapped.get_output()


def test_scheduler_subscription_never_activates_worker_hot_paths(
    events: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[object] = []
    sink = SimpleNamespace(emit=received.append)
    events.EventBus.register_sink(sink)

    def unexpected_clock_read() -> int:
        raise AssertionError("worker timing ran with only a scheduler subscriber")

    monkeypatch.setattr(time, "perf_counter_ns", unexpected_clock_read)

    assert events.EventBus.enabled is True
    assert events.EventBus.worker_enabled is False
    assert run_mrv1(events).sampled_token_ids == [[7]]
    assert run_mrv2(events).sampled_token_ids == [[7]]
    assert received == []
    assert events.EventBus._worker_sinks == []


@pytest.mark.parametrize("runner", [run_mrv1, run_mrv2])
def test_late_subscription_does_not_emit_unmeasured_wait(
    events: ModuleType, monkeypatch: pytest.MonkeyPatch, runner: Any
) -> None:
    received: list[object] = []

    def subscribe_on_wait(self: FakeEvent) -> None:
        events.EventBus.register_worker_sink(SimpleNamespace(emit=received.append))

    monkeypatch.setattr(FakeEvent, "synchronize", subscribe_on_wait)
    assert runner(events).sampled_token_ids == [[7]]
    assert received == []


@pytest.mark.parametrize("runner", [run_mrv1, run_mrv2])
def test_worker_subscription_receives_only_its_runner_lifecycle(
    events: ModuleType, runner: Any
) -> None:
    worker_events: list[object] = []
    scheduler_events: list[object] = []
    events.EventBus.register_sink(SimpleNamespace(emit=scheduler_events.append))
    events.EventBus.register_worker_sink(SimpleNamespace(emit=worker_events.append))

    output = runner(events)

    assert output.sampled_token_ids == [[7]]
    assert [type(event) for event in worker_events] == [
        events.AsyncOutputCreated,
        events.AsyncOutputCopyIssued,
        events.AsyncOutputWaitComplete,
        events.AsyncOutputMaterialized,
    ]
    assert worker_events[0].runner_kind == ("mrv1" if runner is run_mrv1 else "mrv2")
    assert worker_events[1].nbytes == 8
    assert scheduler_events == []
    events.EventBus.emit(events.RequestFinished("r", None, 1, 1, 2, 0, False, "stop"))
    assert len(scheduler_events) == 1
    assert isinstance(scheduler_events[0], events.RequestFinished)
    assert len(worker_events) == 4


def test_failed_worker_sink_is_removed_without_disabling_scheduler(
    events: ModuleType,
) -> None:
    scheduler_events: list[object] = []
    worker_events: list[object] = []

    def fail(event: object) -> None:
        raise RuntimeError("sink failed")

    failing_sink = SimpleNamespace(emit=fail)
    healthy_sink = SimpleNamespace(emit=worker_events.append)
    events.EventBus.register_sink(SimpleNamespace(emit=scheduler_events.append))
    events.EventBus.register_worker_sink(failing_sink)
    events.EventBus.register_worker_sink(healthy_sink)

    assert run_mrv1(events).sampled_token_ids == [[7]]
    assert events.EventBus._worker_sinks == [healthy_sink]
    assert len(worker_events) == 4
    assert events.EventBus.enabled is True
    assert scheduler_events == []


def test_copy_failure_preserves_runtime_error_and_worker_receipt(
    events: ModuleType,
) -> None:
    received: list[object] = []
    events.EventBus.register_worker_sink(SimpleNamespace(emit=received.append))
    with pytest.raises(RuntimeError, match="copy failed"):
        run_mrv1(events, fail_copy=True)
    assert [type(item) for item in received] == [
        events.AsyncOutputCreated,
        events.AsyncOutputCopyFailed,
    ]


def test_second_consumer_receipt_follows_actual_input_batch_path(
    events: ModuleType,
) -> None:
    namespace = {name: getattr(events, name) for name in events.__all__}
    namespace.update(time=time, cast=cast)
    method_namespace("vllm/v1/worker/gpu_input_batch.py", "InputBatch", namespace)
    batch = object.__new__(namespace["InputBatch"])
    batch.sampling_metadata = SimpleNamespace(output_token_ids=[[-1]])
    batch.prev_req_id_to_index = {"r": 0}
    batch._req_ids = ["r"]
    received: list[object] = []
    events.EventBus.register_worker_sink(SimpleNamespace(emit=received.append))

    batch.set_async_sampled_token_ids(FakeCpuTensor(), FakeEvent())
    batch.update_async_output_token_ids()

    assert batch.sampling_metadata.output_token_ids == [[7]]
    assert [type(item) for item in received] == [
        events.AsyncOutputRetained,
        events.AsyncOutputConsumed,
    ]


def test_second_consumer_does_not_emit_without_retained_receipt(
    events: ModuleType,
) -> None:
    namespace = {name: getattr(events, name) for name in events.__all__}
    namespace.update(time=time, cast=cast)
    method_namespace("vllm/v1/worker/gpu_input_batch.py", "InputBatch", namespace)
    batch = object.__new__(namespace["InputBatch"])
    batch.sampling_metadata = SimpleNamespace(output_token_ids=[[-1]])
    batch.prev_req_id_to_index = {"r": 0}
    batch._req_ids = ["r"]
    batch.set_async_sampled_token_ids(FakeCpuTensor(), FakeEvent())

    received: list[object] = []
    events.EventBus.register_worker_sink(SimpleNamespace(emit=received.append))
    batch.update_async_output_token_ids()

    assert batch.sampling_metadata.output_token_ids == [[7]]
    assert received == []


def test_host_capability_names_the_worker_contract(events: ModuleType) -> None:
    path = ROOT / "vllm/plugins/extension_capabilities.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    protocols = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_PROTOCOLS"
    )
    snapshot = eval(
        compile(ast.Expression(protocols), str(path), "eval"),
        {
            "MappingProxyType": MappingProxyType,
            "ASYNC_OUTPUT_EVENTS_API_VERSION": events.ASYNC_OUTPUT_EVENTS_API_VERSION,
            "REQUEST_LIFECYCLE_EVENTS_API_VERSION": "1.0",
            "KV_TRANSFER_OBSERVABILITY_API_VERSION": "1.0",
        },
    )
    assert snapshot["vllm.async-output-events"] == "1.0"
