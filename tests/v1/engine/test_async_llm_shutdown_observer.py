# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU regressions for the API output task during intentional shutdown."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

import vllm.v1.engine.async_llm as async_llm_module
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError


@pytest.mark.parametrize(
    ("error", "shutdown_requested", "unfinished", "expected_shutdown"),
    [
        (EngineDeadError(), True, False, True),
        (EngineDeadError(), False, False, False),
        (EngineDeadError(), True, True, False),
        (RuntimeError("unexpected"), True, False, False),
    ],
)
def test_output_handler_only_expects_idle_engine_exit_during_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    shutdown_requested: bool,
    unfinished: bool,
    expected_shutdown: bool,
) -> None:
    engine = object.__new__(AsyncLLM)
    engine.engine_core = MagicMock()
    engine.engine_core.get_output_async = AsyncMock(side_effect=error)
    engine.output_processor = MagicMock()
    engine.output_processor.has_unfinished_requests.return_value = unfinished
    engine.log_stats = False
    engine.logger_manager = None
    engine.renderer = MagicMock()
    engine.output_handler = None
    engine._shutdown_requested = threading.Event()
    if shutdown_requested:
        engine._shutdown_requested.set()
    logger = MagicMock()
    monkeypatch.setattr(async_llm_module, "logger", logger)
    monkeypatch.setattr(async_llm_module, "shutdown_prometheus", lambda: None)

    async def run_handler() -> None:
        engine._run_output_handler()
        assert engine.output_handler is not None
        await engine.output_handler

    asyncio.run(run_handler())

    if expected_shutdown:
        logger.info.assert_called_once()
        logger.exception.assert_not_called()
        engine.output_processor.propagate_error.assert_not_called()
    else:
        logger.info.assert_not_called()
        logger.exception.assert_called_once()
        engine.output_processor.propagate_error.assert_called_once_with(error)


def test_shutdown_marks_intent_before_stopping_engine_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = object.__new__(AsyncLLM)
    engine._shutdown_requested = threading.Event()
    engine.renderer = MagicMock()
    engine.engine_core = MagicMock()
    engine.output_handler = None
    monkeypatch.setattr(async_llm_module, "shutdown_prometheus", lambda: None)

    def stop_core(timeout: float) -> None:
        assert engine._shutdown_requested.is_set()
        assert timeout == 20

    engine.engine_core.shutdown.side_effect = stop_core
    engine.shutdown(timeout=20)
    engine.renderer.shutdown.assert_called_once()
    engine.engine_core.shutdown.assert_called_once_with(timeout=20)
    engine.engine_core.shutdown.side_effect = None
