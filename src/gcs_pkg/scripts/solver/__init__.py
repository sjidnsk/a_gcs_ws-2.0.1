"""
求解器配置模块

提供自适应求解器配置、预热和缓存功能。
"""

from .solver_config import (
    ProblemSize,
    SolverType,
    SolverPerformanceProfile,
    AdaptiveSolverConfig,
    create_optimized_gcs_options,
    get_fast_solver_config,
    get_accurate_solver_config,
    get_balanced_solver_config,
)

__all__ = [
    'ProblemSize',
    'SolverType',
    'SolverPerformanceProfile',
    'AdaptiveSolverConfig',
    'create_optimized_gcs_options',
    'get_fast_solver_config',
    'get_accurate_solver_config',
    'get_balanced_solver_config',
]
