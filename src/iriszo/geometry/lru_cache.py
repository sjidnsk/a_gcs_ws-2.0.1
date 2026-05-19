"""
Local LRU cache used by IrisZo collision checks.

Keeping this implementation inside ``iriszo`` avoids importing ``iris_pkg``
when only the custom IrisZo package is needed.
"""

from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class LRUCache:
    """O(1) least-recently-used cache for grid collision queries."""

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, key: Tuple[int, int]) -> Optional[bool]:
        if key not in self.cache:
            self.misses += 1
            return None

        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]

    def put(self, key: Tuple[int, int], value: bool) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
            return

        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0.0,
            'size': len(self.cache),
            'capacity': self.capacity,
        }

    def __repr__(self) -> str:
        return (
            f"LRUCache(capacity={self.capacity}, size={len(self.cache)}, "
            f"hits={self.hits}, misses={self.misses})"
        )


__all__ = ['LRUCache']
