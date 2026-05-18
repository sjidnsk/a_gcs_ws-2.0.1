"""
GCS配置模块

提供轻量级GCS成本配置。月面车策略模板请显式从
``config.gcs.lunar_rover_config`` 导入。
"""

from .cost_configurator import (
    CostConfigurator,
    CostWeights,
    OptimizationPriority,
    get_lunar_standard_config,
    get_lunar_high_risk_config,
    get_lunar_emergency_config,
    get_lunar_complex_config,
)

__all__ = [
    'CostConfigurator',
    'CostWeights',
    'OptimizationPriority',
    'get_lunar_standard_config',
    'get_lunar_high_risk_config',
    'get_lunar_emergency_config',
    'get_lunar_complex_config',
]
