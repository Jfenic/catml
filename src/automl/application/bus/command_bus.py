from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

TCommand = TypeVar("TCommand")
TResult = TypeVar("TResult")


@dataclass
class CommandBus:
    _handlers: dict[type, Callable[[Any], Any]]

    def __init__(self) -> None:
        self._handlers = {}

    def register(self, command_type: type[TCommand], handler: Callable[[TCommand], TResult]) -> None:
        self._handlers[command_type] = handler

    def dispatch(self, command: Any) -> Any:
        handler = self._handlers.get(type(command))
        if handler is None:
            raise KeyError(f"No handler registered for {type(command).__name__}")
        return handler(command)
