from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

TQuery = TypeVar("TQuery")
TResult = TypeVar("TResult")


@dataclass
class QueryBus:
    _handlers: dict[type, Callable[[Any], Any]]

    def __init__(self) -> None:
        self._handlers = {}

    def register(self, query_type: type[TQuery], handler: Callable[[TQuery], TResult]) -> None:
        self._handlers[query_type] = handler

    def dispatch(self, query: Any) -> Any:
        handler = self._handlers.get(type(query))
        if handler is None:
            raise KeyError(f"No handler registered for {type(query).__name__}")
        return handler(query)
