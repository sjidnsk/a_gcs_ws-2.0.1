"""
IrisNp 凸区域生成模块

基于 Drake IrisNp 的配置空间凸区域生成算法。

主要导出:
- IrisNpRegionGenerator: 凸区域生成器
- IrisNpConfig / IrisNpConfigOptimized: 配置参数
- IrisNpRegion / IrisNpResult: 数据结构
- SimpleCollisionCheckerForIrisNp: 碰撞检测器
- visualize_iris_np_result: 可视化函数
- check_drake_availability: Drake可用性检查

子模块:
- config: 配置模块
- core: 核心功能模块

作者: Path Planning Team
"""

import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if path not in sys.path:
    sys.path.insert(0, path)

from config.iris import (
    IrisNpConfig,
    IrisNpConfigOptimized,
    get_high_safety_config,
    get_fast_processing_config,
    get_balanced_config,
    DEFAULT_ITERATION_LIMIT,
    DEFAULT_TERMINATION_THRESHOLD,
    DEFAULT_REGION_SIZE,
    DEFAULT_MAX_REGION_SIZE,
    DEFAULT_NUM_DIRECTIONS,
    DEFAULT_CACHE_SIZE,
    DEFAULT_NUM_WORKERS
)

from .core import (
    IrisNpRegion,
    IrisNpResult,
    RegionIndex,
    SimpleCollisionCheckerForIrisNp,
    LRUCache,
    IrisNpRegionGenerator,
    visualize_iris_np_result,
    check_drake_availability,
    DRAKE_AVAILABLE
)

__all__ = [
    # 配置
    'IrisNpConfig',
    'IrisNpConfigOptimized',
    'get_high_safety_config',
    'get_fast_processing_config',
    'get_balanced_config',

    # 数据结构
    'IrisNpRegion',
    'IrisNpResult',

    # 碰撞检测
    'SimpleCollisionCheckerForIrisNp',

    # 主要功能
    'IrisNpRegionGenerator',
    'visualize_iris_np_result',
    'check_drake_availability',
    'DRAKE_AVAILABLE'
]

__version__ = '2.0.0'
