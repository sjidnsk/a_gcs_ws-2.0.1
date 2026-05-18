"""
IRIS配置模块（向后兼容接口）

注意：配置文件已移动到 config/iris/ 目录
此文件仅提供向后兼容的导入接口
"""

from config.iris import (
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

__all__ = [
    'IrisNpConfig',
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
