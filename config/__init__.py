"""
统一配置模块入口

提供所有配置类的统一导入接口。

模块结构：
- config.iris: IRIS算法配置
- config.visualization: 可视化配置
- config.gcs: GCS路径规划配置
- config.solver: 求解器配置
- config.planner: 规划器配置

使用示例：
    >>> from config.iris import IrisNpConfig
    >>> from config.planner import PlannerConfig
    >>> iris_config = IrisNpConfig()
    >>> planner_config = PlannerConfig()

注意：由于模块间依赖关系，建议从子模块直接导入：
    >>> from config.iris import IrisNpConfig  # 推荐
    >>> from config import IrisNpConfig  # 不推荐（可能导致循环导入）
"""

# 为了向后兼容，提供延迟导入功能
__all__ = [
    # IRIS
    'IrisNpConfig',
    'IrisNpConfigOptimized',
    'get_high_safety_config',
    'get_fast_processing_config',
    'get_balanced_config',
    # 可视化
    'VisualizationConfig',
    'ControlPointConfig',
    'ControlPointData',
    'PlotConfig',
    # GCS
    'CostConfigurator',
    'CostWeights',
    'CostOptimizer',
    'OptimizationPriority',
    'LunarRoverGCSConfig',
    'get_lunar_standard_config',
    'get_lunar_high_risk_config',
    'get_lunar_emergency_config',
    'get_lunar_complex_config',
    'get_standard_lunar_config',
    'get_high_risk_lunar_config',
    'get_emergency_lunar_config',
    'get_complex_terrain_config',
    'apply_lunar_config_to_gcs',
    'get_gcs_solve_options',
    # 求解器
    'SolverPerformanceProfile',
    'AdaptiveSolverConfig',
    'ProblemSize',
    'SolverType',
    'create_optimized_gcs_options',
    'get_fast_solver_config',
    'get_accurate_solver_config',
    'get_balanced_solver_config',
    # 规划器
    'PlannerConfig',
    'PlannerResult',
]


def __getattr__(name):
    """延迟导入以避免循环依赖"""
    if name in __all__:
        # 根据名称确定从哪个子模块导入
        if name in ['IrisNpConfig', 'IrisNpConfigOptimized', 'get_high_safety_config',
                    'get_fast_processing_config', 'get_balanced_config']:
            from config.iris import (
                IrisNpConfig, IrisNpConfigOptimized,
                get_high_safety_config, get_fast_processing_config, get_balanced_config
            )
            return locals()[name]

        elif name in ['VisualizationConfig', 'ControlPointConfig', 'ControlPointData', 'PlotConfig']:
            from config.visualization import (
                VisualizationConfig, ControlPointConfig, ControlPointData, PlotConfig
            )
            return locals()[name]

        elif name in ['CostConfigurator', 'CostWeights', 'CostOptimizer', 'OptimizationPriority',
                      'LunarRoverGCSConfig', 'get_lunar_standard_config', 'get_lunar_high_risk_config',
                      'get_lunar_emergency_config', 'get_lunar_complex_config',
                      'get_standard_lunar_config', 'get_high_risk_lunar_config',
                      'get_emergency_lunar_config', 'get_complex_terrain_config',
                      'apply_lunar_config_to_gcs', 'get_gcs_solve_options']:
            from config.gcs import (
                CostConfigurator, CostWeights, CostOptimizer, OptimizationPriority,
                LunarRoverGCSConfig, get_lunar_standard_config, get_lunar_high_risk_config,
                get_lunar_emergency_config, get_lunar_complex_config,
                get_standard_lunar_config, get_high_risk_lunar_config,
                get_emergency_lunar_config, get_complex_terrain_config,
                apply_lunar_config_to_gcs, get_gcs_solve_options
            )
            return locals()[name]

        elif name in ['SolverPerformanceProfile', 'AdaptiveSolverConfig', 'ProblemSize', 'SolverType',
                      'create_optimized_gcs_options', 'get_fast_solver_config',
                      'get_accurate_solver_config', 'get_balanced_solver_config']:
            from config.solver import (
                SolverPerformanceProfile, AdaptiveSolverConfig, ProblemSize, SolverType,
                create_optimized_gcs_options, get_fast_solver_config,
                get_accurate_solver_config, get_balanced_solver_config
            )
            return locals()[name]

        elif name in ['PlannerConfig', 'PlannerResult']:
            from config.planner import PlannerConfig, PlannerResult
            return locals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
