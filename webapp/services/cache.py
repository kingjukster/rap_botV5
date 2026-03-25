"""Lightweight in-memory TTL cache for expensive read-only service calls."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_lock = threading.Lock()
_store: dict[str, tuple[float, Any]] = {}


def ttl_cached(key: str, ttl_seconds: float, fn: Callable[[], T]) -> T:
    """Return cached result for *key* if still fresh, otherwise call *fn* and cache."""
    now = time.monotonic()
    with _lock:
        entry = _store.get(key)
        if entry and (now - entry[0]) < ttl_seconds:
            return entry[1]
    result = fn()
    with _lock:
        _store[key] = (time.monotonic(), result)
    return result


def invalidate(key: str) -> None:
    """Drop a single cache entry."""
    with _lock:
        _store.pop(key, None)


def clear() -> None:
    """Drop all cached entries."""
    with _lock:
        _store.clear()
