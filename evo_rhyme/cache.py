"""
Small in-process caches (no external dependencies).

Used for verse score caching to avoid expensive recomputation while avoiding
pathological \"clear-all\" behavior.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Optional, TypeVar

import time

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired: int = 0

    def as_dict(self) -> dict:
        return {
            "hits": int(self.hits),
            "misses": int(self.misses),
            "evictions": int(self.evictions),
            "expired": int(self.expired),
        }


class LRUTTLCache(Generic[K, V]):
    """
    LRU cache with per-entry TTL.

    - Access updates recency.
    - Expired entries are treated as misses and removed.
    """

    def __init__(
        self,
        *,
        max_size: int,
        ttl_seconds: float,
        time_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.max_size = int(max_size)
        self.ttl_seconds = float(ttl_seconds)
        self._time = time_fn or time.monotonic
        self._data: "OrderedDict[K, tuple[float, V]]" = OrderedDict()
        self.stats = CacheStats()

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()

    def get(self, key: K) -> Optional[V]:
        item = self._data.get(key)
        if item is None:
            self.stats.misses += 1
            return None

        ts, value = item
        now = self._time()
        if (now - ts) > self.ttl_seconds:
            # Expired
            try:
                del self._data[key]
            except Exception:
                pass
            self.stats.expired += 1
            self.stats.misses += 1
            return None

        # Fresh: mark as most-recent
        self._data.move_to_end(key, last=True)
        self.stats.hits += 1
        return value

    def set(self, key: K, value: V) -> None:
        now = self._time()
        if key in self._data:
            self._data[key] = (now, value)
            self._data.move_to_end(key, last=True)
            return

        self._data[key] = (now, value)
        self._data.move_to_end(key, last=True)

        if self.max_size > 0 and len(self._data) > self.max_size:
            # Evict least-recent
            try:
                self._data.popitem(last=False)
                self.stats.evictions += 1
            except Exception:
                pass

    def snapshot(self) -> dict:
        hits = self.stats.hits
        misses = self.stats.misses
        total = hits + misses
        hit_rate = (hits / total) if total else 0.0
        return {
            "size": len(self._data),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hit_rate": hit_rate,
            **self.stats.as_dict(),
        }

