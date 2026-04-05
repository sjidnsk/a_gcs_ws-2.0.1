"""
SCP优化模块

本模块提供SCP求解器的优化功能，包括：
- TrustRegionManager: 信任区域管理器
- EarlyTerminationChecker: 提前终止检查器
- ParallelCurvatureLinearizer: 并行曲率线性化器
- PerformanceStats: 性能统计收集器
- ConstraintViolationCalculator: 约束违反量计算器
"""

from .trust_region_manager import TrustRegionManager
from .early_termination_checker import EarlyTerminationChecker
from .parallel_curvature_linearizer import ParallelCurvatureLinearizer
from .performance_stats import PerformanceStats
from .constraint_violation_calculator import ConstraintViolationCalculator

__all__ = [
    'TrustRegionManager',
    'EarlyTerminationChecker',
    'ParallelCurvatureLinearizer',
    'PerformanceStats',
    'ConstraintViolationCalculator',
]
