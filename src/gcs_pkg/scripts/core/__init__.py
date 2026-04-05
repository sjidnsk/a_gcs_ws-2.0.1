"""
核心GCS算法实现模块

包含BaseGCS基础类、LinearGCS、BezierGCS和AckermannGCS三种路径规划实现。
重构后，AckermannGCS相关代码已拆分为以下模块：
- utils.py: 工具函数
- steering_constraint.py: 转向角约束管理器
- ackermann_trajectory.py: 阿克曼轨迹封装类
- ackermann_gcs.py: 阿克曼GCS轨迹优化器
"""

from .base import BaseGCS, polytopeDimension, convexSetDimension, intersectionDimension
from .linear import LinearGCS
from .bezier import BezierGCS, BezierTrajectory

# 从重构后的模块导入
from .ackermann_gcs import AckermannGCS
from .ackermann_trajectory import AckermannTrajectory
from .steering_constraint import SteeringConstraintManager
from .utils import sample_unit_sphere


__all__ = [
    'BaseGCS',
    'LinearGCS',
    'BezierGCS',
    'BezierTrajectory',
    'AckermannGCS',
    'AckermannTrajectory',
    'SteeringConstraintManager',
    'sample_unit_sphere',
    'polytopeDimension',
    'convexSetDimension',
    'intersectionDimension',
]
