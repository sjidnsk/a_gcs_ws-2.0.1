"""
核心GCS算法实现模块

包含BaseGCS基础类、LinearGCS和BezierGCS两种路径规划实现。
"""

from .base import BaseGCS, polytopeDimension, convexSetDimension, intersectionDimension
from .linear import LinearGCS
from .bezier import BezierGCS, BezierTrajectory

__all__ = [
    'BaseGCS',
    'LinearGCS',
    'BezierGCS',
    'BezierTrajectory',
    'polytopeDimension',
    'convexSetDimension',
    'intersectionDimension',
]
