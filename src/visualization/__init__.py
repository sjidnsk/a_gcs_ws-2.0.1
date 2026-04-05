"""
可视化模块

统一的可视化接口，提供：
- 核心基类和配置
- Ackermann轨迹可视化
- 路径规划轨迹可视化
- 环境可视化
- 输出路径管理
"""

# 核心模块
from .core import (
    PlotConfig,
    VisualizationConfig,
    BaseVisualizer,
    VisualizationOutputManager,
    OutputConfig,
    PathBuilder,
    OutputFileInfo,
    RunInstanceInfo,
    OutputManagerError,
    InvalidDimensionError,
    InvalidRunIdError,
    DirectoryCreationError,
    PathValidationError,
    FilenameTooLongError,
)

# Ackermann可视化
from .ackermann import (
    visualize_trajectory,
    visualize_ackermann_gcs_enhanced,
    visualize_3d_trajectory,
    VisualizationConfig as AckermannVisualizationConfig,
    AckermannGCSVisualizer,
)

# 轨迹可视化
from .trajectory import TrajectoryVisualizer

# 环境可视化
from .environment import visualize_environment_with_bezier

__all__ = [
    # 核心配置类
    'PlotConfig',
    'VisualizationConfig',
    'OutputConfig',
    
    # 核心基类
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
    
    # Ackermann可视化
    'visualize_trajectory',
    'visualize_ackermann_gcs_enhanced',
    'visualize_3d_trajectory',
    'AckermannVisualizationConfig',
    'AckermannGCSVisualizer',
    
    # 轨迹可视化
    'TrajectoryVisualizer',
    
    # 环境可视化
    'visualize_environment_with_bezier',
]
