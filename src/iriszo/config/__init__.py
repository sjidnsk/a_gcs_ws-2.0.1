"""
配置模块

导出配置类和预定义配置模板。
"""

from .iriszo_config import (
    IrisZoConfig,
    get_high_safety_config,
    get_fast_processing_config,
    get_balanced_config
)

__all__ = [
    'IrisZoConfig',
    'get_high_safety_config',
    'get_fast_processing_config',
    'get_balanced_config'
]
