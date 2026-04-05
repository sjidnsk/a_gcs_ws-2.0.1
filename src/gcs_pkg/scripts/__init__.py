"""
GCS (Graph of Convex Sets) 路径规划模块

该模块提供了基于凸集图的路径规划算法，包括：
- 核心GCS算法实现（线性路径和贝塞尔曲线）
- 路径舍入策略
- 求解器配置优化
- 应用配置模板（如月面月球车）
- 工具函数

主要类:
    BaseGCS: GCS基础类
    LinearGCS: 线性路径规划
    BezierGCS: 贝塞尔曲线轨迹规划
    CostConfigurator: 成本配置器
    AdaptiveSolverConfig: 自适应求解器配置

使用示例:
    >>> from gcs_pkg.scripts import BezierGCS, CostConfigurator
    >>> gcs = BezierGCS(regions, order=4, continuity=2)
    >>> configurator = CostConfigurator().set_preset('lunar_standard')
    >>> configurator.apply_to_gcs(gcs)
"""

# 导入核心模块
from .core import (
    BaseGCS,
    LinearGCS,
    BezierGCS,
    BezierTrajectory,
    AckermannGCS,
    AckermannTrajectory,
    polytopeDimension,
    convexSetDimension,
    intersectionDimension,
)

# 导入舍入策略
from .rounding import (
    greedyForwardPathSearch,
    greedyBackwardPathSearch,
    randomForwardPathSearch,
    randomBackwardPathSearch,
    MipPathExtraction,
    averageVertexPositionGcs,
)

# 导入求解器配置
from config.solver import (
    ProblemSize,
    SolverType,
    SolverPerformanceProfile,
    AdaptiveSolverConfig,
    create_optimized_gcs_options,
    get_fast_solver_config,
    get_accurate_solver_config,
    get_balanced_solver_config,
)

# 导入配置模块
from config.gcs import (
    OptimizationPriority,
    CostWeights,
    CostConfigurator,
    CostOptimizer,
    LunarRoverGCSConfig,
    get_lunar_standard_config,
    get_lunar_high_risk_config,
    get_lunar_emergency_config,
    get_lunar_complex_config,
    get_standard_lunar_config,
    get_high_risk_lunar_config,
    get_emergency_lunar_config,
    get_complex_terrain_config,
    apply_lunar_config_to_gcs,
    get_gcs_solve_options,
)

# 导入工具函数
from .utils import removeRedundancies

__all__ = [
    # 核心类
    'BaseGCS',
    'LinearGCS',
    'BezierGCS',
    'BezierTrajectory',
    'AckermannGCS',
    'AckermannTrajectory',
    # 维度计算函数
    'polytopeDimension',
    'convexSetDimension',
    'intersectionDimension',
    # 舍入策略
    'greedyForwardPathSearch',
    'greedyBackwardPathSearch',
    'randomForwardPathSearch',
    'randomBackwardPathSearch',
    'MipPathExtraction',
    'averageVertexPositionGcs',
    # 求解器配置
    'ProblemSize',
    'SolverType',
    'SolverPerformanceProfile',
    'AdaptiveSolverConfig',
    'create_optimized_gcs_options',
    'get_fast_solver_config',
    'get_accurate_solver_config',
    'get_balanced_solver_config',
    # 成本配置
    'OptimizationPriority',
    'CostWeights',
    'CostConfigurator',
    'CostOptimizer',
    'get_lunar_standard_config',
    'get_lunar_high_risk_config',
    'get_lunar_emergency_config',
    'get_lunar_complex_config',
    # 月面配置
    'LunarRoverGCSConfig',
    'get_standard_lunar_config',
    'get_high_risk_lunar_config',
    'get_emergency_lunar_config',
    'get_complex_terrain_config',
    'apply_lunar_config_to_gcs',
    'get_gcs_solve_options',
    # 工具函数
    'removeRedundancies',
]

__version__ = '2.0.1'
