"""
IRIS配置模块

提供IRIS算法的所有配置类和工具函数。
"""

from .iris_np_config import (
    IrisNpConfig,
    get_high_safety_config,
    get_fast_processing_config,
    get_balanced_config,
    DEFAULT_ITERATION_LIMIT,
    DEFAULT_TERMINATION_THRESHOLD,
    DEFAULT_REGION_SIZE,
    DEFAULT_MAX_REGION_SIZE,
    DEFAULT_NUM_DIRECTIONS,
    DEFAULT_CACHE_SIZE,
    DEFAULT_NUM_WORKERS,
)

from .iris_np_config_optimized import IrisNpConfigOptimized


__all__ = [
    'IrisNpConfig',
    'IrisNpConfigOptimized',
    'get_high_safety_config',
    'get_fast_processing_config',
    'get_balanced_config',
    'DEFAULT_ITERATION_LIMIT',
    'DEFAULT_TERMINATION_THRESHOLD',
    'DEFAULT_REGION_SIZE',
    'DEFAULT_MAX_REGION_SIZE',
    'DEFAULT_NUM_DIRECTIONS',
    'DEFAULT_CACHE_SIZE',
    'DEFAULT_NUM_WORKERS',
]
