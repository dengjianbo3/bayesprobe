from __future__ import annotations

import math
import time
from collections.abc import Callable
from threading import Lock
from typing import Any

from bayesprobe_terminal_bench.actions import ShellAction
from bayesprobe_terminal_bench.config import BudgetExhausted


class TrialDeadline:
    _COMPLETION_MARGIN_SECONDS = 5

    def __init__(
        self,
        *,
        timeout_seconds: int,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(timeout_seconds) is not int or timeout_seconds < 1:
            raise ValueError("trial timeout must be a positive integer")
        self._monotonic = monotonic
        self._deadline = monotonic() + timeout_seconds
        self._lock = Lock()

    def remaining_seconds(self) -> float:
        with self._lock:
            return self._deadline - self._monotonic()

    def timeout_for(self, *, configured_timeout_seconds: int) -> int:
        if (
            type(configured_timeout_seconds) is not int
            or configured_timeout_seconds < 1
        ):
            raise ValueError("configured timeout must be a positive integer")
        with self._lock:
            remaining = (
                math.floor(self._deadline - self._monotonic())
                - self._COMPLETION_MARGIN_SECONDS
            )
            timeout = min(configured_timeout_seconds, remaining)
            if timeout <= 0:
                raise BudgetExhausted("trial deadline exhausted")
            return timeout


class _DeadlineCompletions:
    def __init__(self, owner: DeadlineOpenAIClient) -> None:
        self._owner = owner

    def create(self, **kwargs: object) -> Any:
        try:
            timeout = self._owner._deadline.timeout_for(
                configured_timeout_seconds=self._owner._configured_timeout_seconds
            )
        except BudgetExhausted as error:
            if self._owner._error_observer is not None:
                self._owner._error_observer(error)
            raise
        client = self._owner._base_client.with_options(
            timeout=timeout,
            max_retries=0,
        )
        response = client.chat.completions.create(**kwargs)
        if self._owner._response_observer is not None:
            self._owner._response_observer(response)
        return response


class _DeadlineChat:
    def __init__(self, owner: DeadlineOpenAIClient) -> None:
        self.completions = _DeadlineCompletions(owner)


class DeadlineOpenAIClient:
    def __init__(
        self,
        *,
        base_client: Any,
        deadline: TrialDeadline,
        configured_timeout_seconds: int,
        response_observer: Callable[[Any], None] | None = None,
        error_observer: Callable[[Exception], None] | None = None,
    ) -> None:
        self._base_client = base_client
        self._deadline = deadline
        self._configured_timeout_seconds = configured_timeout_seconds
        self._response_observer = response_observer
        self._error_observer = error_observer
        self.chat = _DeadlineChat(self)

    def with_options(self, **kwargs: object) -> DeadlineOpenAIClient:
        max_retries = kwargs.get("max_retries", 0)
        if max_retries != 0:
            raise ValueError("deadline client requires max_retries=0")
        return self


class DeadlineEnvironmentBridge:
    def __init__(
        self,
        *,
        delegate: Any,
        deadline: TrialDeadline,
        configured_timeout_seconds: int,
    ) -> None:
        self._delegate = delegate
        self._deadline = deadline
        self._configured_timeout_seconds = configured_timeout_seconds
        self._delegate_timeout_lock = Lock()

    def execute(self, action: Any, action_index: int) -> Any:
        timeout = self._deadline.timeout_for(
            configured_timeout_seconds=self._configured_timeout_seconds
        )
        if isinstance(action, ShellAction):
            action = action.model_copy(
                update={"timeout_seconds": min(action.timeout_seconds, timeout)}
            )
            return self._delegate.execute(action, action_index)
        if not hasattr(self._delegate, "_NON_SHELL_TIMEOUT_SECONDS"):
            return self._delegate.execute(action, action_index)
        with self._delegate_timeout_lock:
            instance_values = getattr(self._delegate, "__dict__", {})
            had_instance_value = "_NON_SHELL_TIMEOUT_SECONDS" in instance_values
            previous = getattr(self._delegate, "_NON_SHELL_TIMEOUT_SECONDS")
            setattr(self._delegate, "_NON_SHELL_TIMEOUT_SECONDS", timeout)
            try:
                return self._delegate.execute(action, action_index)
            finally:
                if had_instance_value:
                    setattr(self._delegate, "_NON_SHELL_TIMEOUT_SECONDS", previous)
                else:
                    delattr(self._delegate, "_NON_SHELL_TIMEOUT_SECONDS")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class DeadlineTerminalPlanner:
    def __init__(
        self,
        *,
        delegate: Any,
        deadline: TrialDeadline,
        accounting_observer: Any,
    ) -> None:
        self._delegate = delegate
        self._deadline = deadline
        self._accounting_observer = accounting_observer

    def plan(self, **kwargs: Any) -> Any:
        try:
            result = self._delegate.plan(**kwargs)
        except Exception:
            self._accounting_observer.raise_pending()
            raise
        self._accounting_observer.raise_pending()
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)
