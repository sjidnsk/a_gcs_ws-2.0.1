"""
核心可视化模块

提供可视化基类和配置管理
"""

from config.visualization import PlotConfig, VisualizationConfig
from .base_visualizer import BaseVisualizer
from .output_manager import VisualizationOutputManager
from .models import OutputConfig, OutputFileInfo, RunInstanceInfo
from .path_builder import PathBuilder
from .exceptions import (
    OutputManagerError,
    InvalidDimensionError,
    InvalidRunIdError,
    DirectoryCreationError,
    PathValidationError,
    FilenameTooLongError
)

__all__ = [
    # 配置类
    'PlotConfig',
    'VisualizationConfig',
    'OutputConfig',
    
    # 基类
    'BaseVisualizer',
    
    # 输出管理器
    'VisualizationOutputManager',
    'PathBuilder',
    
    # 数据模型
    'OutputFileInfo',
    'RunInstanceInfo',
    
    # 异常类
    'OutputManagerError',
    'InvalidDimensionError',
    'InvalidRunIdError',
    'DirectoryCreationError',
    'PathValidationError',
    'FilenameTooLongError',
]
