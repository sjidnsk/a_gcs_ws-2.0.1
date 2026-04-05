"""
AckermannGCS可视化模块

提供增强的轨迹可视化功能，包括：
- 2D轨迹视图（带IRIS区域、走廊、障碍物）
- 3D配置空间轨迹视图
- 速度、航向角、曲率等曲线图
- 综合可视化布局
"""

from .config import VisualizationConfig
from .trajectory_sampler import TrajectorySampler, TrajectoryData
from .region_renderer import RegionRenderer
from .path_comparator import PathComparator
from .plot_2d_trajectory import Plot2DTrajectory
from .plot_3d_trajectory import Plot3DTrajectory
from .plot_profiles import PlotProfiles
from .ackermann_visualizer_enhanced import AckermannGCSVisualizer

# 基础可视化器
from .ackermann_visualizer import visualize_trajectory

# 便捷接口函数
from .ackermann_visualizer_enhanced import visualize_ackermann_gcs_enhanced
from .plot_3d_trajectory import visualize_3d_trajectory

__all__ = [
    # 核心类
    'VisualizationConfig',
    'TrajectorySampler',
    'TrajectoryData',
    'RegionRenderer',
    'PathComparator',
    'Plot2DTrajectory',
    'Plot3DTrajectory',
    'PlotProfiles',
    'AckermannGCSVisualizer',
    # 便捷接口
    'visualize_trajectory',
    'visualize_ackermann_gcs_enhanced',
    'visualize_3d_trajectory',
]
