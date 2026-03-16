"""
IrisNp 配置模块

包含配置参数类和预定义配置模板。

作者: Path Planning Team
"""

from .iris_np_config import (
    IrisNpConfig,
    DEFAULT_ITERATION_LIMIT,
    DEFAULT_TERMINATION_THRESHOLD,
    DEFAULT_REGION_SIZE,
    DEFAULT_MAX_REGION_SIZE,
    DEFAULT_NUM_DIRECTIONS,
    DEFAULT_CACHE_SIZE,
    DEFAULT_NUM_WORKERS
)

from .iris_np_config_optimized import (
    IrisNpConfigOptimized,
    get_high_safety_config,
    get_fast_processing_config,
    get_balanced_config
)

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
    'DEFAULT_NUM_WORKERS'
]
