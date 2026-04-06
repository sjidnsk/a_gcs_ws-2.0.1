"""
核心GCS算法实现模块

包含BaseGCS基础类、LinearGCS、BezierGCS三种路径规划实现。

注意：AckermannGCS已迁移到独立的ackermann_gcs_pkg包
请使用: from ackermann_gcs_pkg.ackermann_gcs_planner import AckermannGCSPlanner

遗留模块已移除：
- utils.py: 工具函数（功能已在其他模块实现）
- steering_constraint.py: 转向角约束管理器（已被ackermann_gcs_pkg的SCP框架替代）
- ackermann_trajectory.py: 阿克曼轨迹封装类（功能已被FlatOutputMapper替代）
- quadratic_constraint_manager.py: 二次约束管理器（已被ackermann_gcs_pkg的SCP方法替代）
- ackermann_gcs.py: 阿克曼GCS轨迹优化器（已迁移到ackermann_gcs_pkg）
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
