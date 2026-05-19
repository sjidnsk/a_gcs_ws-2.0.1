"""
Coverage validation and distance-query utilities for IrisZo.
"""

from .coverage import CoverageResult, CoverageValidator
from .coverage_checker import CoverageChecker
from .coverage_radius import RadiusCalculator
from .coverage_validator_enhanced import (
    EnhancedCoverageResult,
    EnhancedCoverageValidator,
)
from .distance_query import DistanceQueryEngine, DistanceTransformEngine, KDTreeEngine
from .obstacle_detector import ObstacleDetector

__all__ = [
    'CoverageChecker',
    'CoverageResult',
    'CoverageValidator',
    'DistanceQueryEngine',
    'DistanceTransformEngine',
    'EnhancedCoverageResult',
    'EnhancedCoverageValidator',
    'KDTreeEngine',
    'ObstacleDetector',
    'RadiusCalculator',
]
