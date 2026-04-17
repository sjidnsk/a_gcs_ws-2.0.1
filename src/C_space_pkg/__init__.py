"""配置空间处理模块

提供 SE(2) 配置空间生成、障碍物处理、部分走廊生成等功能。
"""

from .se2 import SE2ConfigurationSpace, RobotShape
from .partial_corridor import CorridorGenerator, CorridorConfig

__all__ = [
    'SE2ConfigurationSpace',
    'RobotShape',
    'CorridorGenerator',
    'CorridorConfig',
]
