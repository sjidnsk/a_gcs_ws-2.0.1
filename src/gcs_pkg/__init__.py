"""GCS (Graph of Convex Sets) 路径规划模块

该模块提供了基于凸集图的路径规划算法。

注意：实际实现位于 gcs_pkg.scripts 子模块，
此文件提供便捷的顶层导入接口。

使用示例:
    >>> from gcs_pkg import BezierGCS, LinearGCS
    >>> from gcs_pkg.scripts import CostConfigurator
"""

from .scripts import (
    BaseGCS,
    LinearGCS,
    BezierGCS,
    BezierTrajectory,
    CostConfigurator,
    AdaptiveSolverConfig,
)

__all__ = [
    'BaseGCS',
    'LinearGCS',
    'BezierGCS',
    'BezierTrajectory',
    'CostConfigurator',
    'AdaptiveSolverConfig',
]
