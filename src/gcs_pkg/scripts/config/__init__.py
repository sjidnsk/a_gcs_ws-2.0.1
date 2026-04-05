"""
应用配置模块

提供成本配置和特定应用场景（如月面月球车）的配置模板。
"""

from .cost_configurator import (
    OptimizationPriority,
    CostWeights,
    CostConfigurator,
    CostOptimizer,
    get_lunar_standard_config,
    get_lunar_high_risk_config,
    get_lunar_emergency_config,
    get_lunar_complex_config,
)

from .lunar_rover_config import (
    LunarRoverGCSConfig,
    get_standard_lunar_config,
    get_high_risk_lunar_config,
    get_emergency_lunar_config,
    get_complex_terrain_config,
    apply_lunar_config_to_gcs,
    get_gcs_solve_options,
)

__all__ = [
    # Cost configurator
    'OptimizationPriority',
    'CostWeights',
    'CostConfigurator',
    'CostOptimizer',
    'get_lunar_standard_config',
    'get_lunar_high_risk_config',
    'get_lunar_emergency_config',
    'get_lunar_complex_config',
    # Lunar rover config
    'LunarRoverGCSConfig',
    'get_standard_lunar_config',
    'get_high_risk_lunar_config',
    'get_emergency_lunar_config',
    'get_complex_terrain_config',
    'apply_lunar_config_to_gcs',
    'get_gcs_solve_options',
]
