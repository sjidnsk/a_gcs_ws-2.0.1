"""
可视化配置模块
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class PlotConfig:
    """通用绘图配置类
    
    管理所有可视化相关的配置参数
    """
    
    # 采样参数
    num_samples: int = 200  # 轨迹采样点数
    
    # 2D视图参数
    show_iris_regions: bool = True  # 显示IRIS区域
    show_obstacles: bool = True  # 显示障碍物
    show_corridor: bool = True  # 显示走廊约束
    show_astar_path: bool = True  # 显示A*路径
    iris_alpha: float = 0.2  # IRIS区域透明度
    obstacle_alpha: float = 0.5  # 障碍物透明度
    
    # 3D视图参数
    show_3d_trajectory: bool = True  # 显示3D轨迹
    elev: float = 25.0  # 3D视角仰角
    azim: float = 45.0  # 3D视角方位角
    
    # 曲线图参数
    show_velocity: bool = True  # 显示速度曲线
    show_heading: bool = True  # 显示航向角曲线
    show_curvature: bool = True  # 显示曲率曲线
    show_steering: bool = True  # 显示转向角曲线
    show_acceleration: bool = True  # 显示加速度曲线
    show_theta_profile: bool = True  # 显示θ随路径变化曲线
    
    # 样式参数
    trajectory_color: str = 'red'  # 轨迹颜色
    trajectory_linewidth: float = 2.0  # 轨迹线宽
    astar_color: str = 'green'  # A*路径颜色
    astar_linestyle: str = '--'  # A*路径线型
    source_color: str = 'green'  # 起点颜色
    target_color: str = 'red'  # 终点颜色
    source_marker: str = 's'  # 起点标记（方块）
    target_marker: str = '*'  # 终点标记（星形）
    
    # 字体参数
    font_family: str = 'DejaVu Sans'  # 使用系统默认字体,避免字体警告
    font_size: int = 12
    title_font_size: int = 14
    label_font_size: int = 12
    
    # 图表参数
    figsize: Tuple[float, float] = (20, 14)  # 图表大小
    dpi: int = 150  # 分辨率
    subplot_hspace: float = 0.3  # 子图垂直间距
    subplot_wspace: float = 0.3  # 子图水平间距
    
    # 保存参数
    save_format: str = 'png'  # 保存格式
    bbox_inches: str = 'tight'  # 边界处理
    
    # 颜色映射
    iris_colors: List[str] = field(default_factory=lambda: [
        'lightblue', 'lightgreen', 'lightyellow', 'lightpink', 
        'lightcoral', 'lightsalmon', 'lightseagreen', 'lightskyblue'
    ])
    
    def validate(self) -> bool:
        """验证配置参数
        
        Returns:
            bool: 配置是否有效
        """
        if self.num_samples <= 0:
            raise ValueError("num_samples必须大于0")
        
        if not 0 <= self.iris_alpha <= 1:
            raise ValueError("iris_alpha必须在[0, 1]范围内")
        
        if not 0 <= self.obstacle_alpha <= 1:
            raise ValueError("obstacle_alpha必须在[0, 1]范围内")
        
        if self.figsize[0] <= 0 or self.figsize[1] <= 0:
            raise ValueError("figsize必须为正数")
        
        if self.dpi <= 0:
            raise ValueError("dpi必须大于0")
        
        return True
    
    def update(self, **kwargs) -> 'PlotConfig':
        """更新配置参数
        
        Args:
            **kwargs: 要更新的参数
            
        Returns:
            PlotConfig: 更新后的配置对象
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"未知参数: {key}")
        
        self.validate()
        return self


# 为了向后兼容，保留VisualizationConfig别名
VisualizationConfig = PlotConfig
