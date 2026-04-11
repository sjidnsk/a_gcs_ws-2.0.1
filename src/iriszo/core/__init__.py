"""
IrisZo核心模块

导出所有核心功能。
"""

from .iriszo_algorithm import *
from .iriszo_bisection import *
from .iriszo_collision import *
from .iriszo_coverage import CoverageValidator, CoverageResult
from .iriszo_coverage_radius import RadiusCalculator
from .iriszo_distance_query import (
    DistanceQueryEngine,
    DistanceTransformEngine,
    KDTreeEngine
)
from .iriszo_obstacle_detector import ObstacleDetector
from .iriszo_coverage_checker import CoverageChecker
from .iriszo_coverage_validator_enhanced import (
    EnhancedCoverageValidator,
    EnhancedCoverageResult
)
from .iriszo_hyperplane import *
from .iriszo_performance import *
from .iriszo_pruning import *
from .iriszo_region import *
from .iriszo_region_data import *
from .iriszo_sampler import *
from .iriszo_seed_extractor import *

__all__ = [
    # 覆盖验证
    'CoverageValidator',
    'CoverageResult',
    'EnhancedCoverageValidator',
    'EnhancedCoverageResult',
    'RadiusCalculator',
    'ObstacleDetector',
    'CoverageChecker',
    'DistanceQueryEngine',
    'DistanceTransformEngine',
    'KDTreeEngine',
]
