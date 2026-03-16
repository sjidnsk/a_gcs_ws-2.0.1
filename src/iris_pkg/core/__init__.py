"""
IrisNp 核心模块

包含区域生成器的核心功能。

作者: Path Planning Team
"""

from .iris_np_region_data import (
    IrisNpRegion,
    IrisNpResult,
    RegionIndex
)

from .iris_np_collision import (
    SimpleCollisionCheckerForIrisNp,
    LRUCache
)

from .iris_np_region import (
    IrisNpRegionGenerator,
    visualize_iris_np_result,
    check_drake_availability,
    DRAKE_AVAILABLE
)

__all__ = [
    # 数据结构
    'IrisNpRegion',
    'IrisNpResult',
    'RegionIndex',

    # 碰撞检测
    'SimpleCollisionCheckerForIrisNp',
    'LRUCache',

    # 主要功能
    'IrisNpRegionGenerator',
    'visualize_iris_np_result',
    'check_drake_availability',
    'DRAKE_AVAILABLE'
]
