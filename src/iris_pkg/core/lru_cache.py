"""
LRU缓存共享实现

提供O(1)时间复杂度的LRU(Least Recently Used)缓存。

作者: Path Planning Team
"""

from collections import OrderedDict
from typing import Tuple, Optional, Dict, Any


class LRUCache:
    """
    LRU(Least Recently Used)缓存实现

    使用OrderedDict实现O(1)时间复杂度的get和put操作。

    Attributes:
        cache: OrderedDict存储缓存数据
        capacity: 缓存容量
        hits: 缓存命中次数
        misses: 缓存未命中次数

    Example:
        >>> cache = LRUCache(capacity=1000)
        >>> cache.put((10, 20), True)
        >>> result = cache.get((10, 20))  # 返回True
        >>> stats = cache.get_stats()
    """

    def __init__(self, capacity: int):
        """
        初始化LRU缓存

        Args:
            capacity: 缓存容量
        """
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0

    def get(self, key: Tuple[int, int]) -> Optional[bool]:
        """
        从缓存中获取值

        Args:
            key: 缓存键,通常为栅格坐标(gx, gy)

        Returns:
            缓存值(True/False),如果不存在则返回None
        """
        if key not in self.cache:
            self.misses += 1
            return None

        # 将访问的键移到末尾(最近使用)
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]

    def put(self, key: Tuple[int, int], value: bool) -> None:
        """
        向缓存中添加值

        Args:
            key: 缓存键
            value: 缓存值
        """
        if key in self.cache:
            # 如果键已存在,移到末尾
            self.cache.move_to_end(key)
        else:
            # 添加新键值对
            self.cache[key] = value

            # 如果超过容量,移除最久未使用的项
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            包含命中率、命中次数、未命中次数等统计信息的字典
        """
        total = self.hits + self.misses
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total if total > 0 else 0.0,
            'size': len(self.cache),
            'capacity': self.capacity
        }

    def __repr__(self) -> str:
        return (f"LRUCache(capacity={self.capacity}, size={len(self.cache)}, "
                f"hits={self.hits}, misses={self.misses})")


__all__ = ['LRUCache']
