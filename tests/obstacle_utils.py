"""
障碍物设置工具模块

本模块提供测试中障碍物设置的安全工具，包括：
- 数组边界检查
- 安全障碍物设置
- 障碍物配置管理
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

# 导入数值安全工具
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ackermann_gcs_pkg.numerical_safety_utils import (
    check_array_bounds,
    safe_array_slice,
    NumericalSafetyError
)


@dataclass
class ObstacleConfig:
    """障碍物配置"""
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    value: float = 1.0
    name: str = "obstacle"
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'row_start': self.row_start,
            'row_end': self.row_end,
            'col_start': self.col_start,
            'col_end': self.col_end,
            'value': self.value,
            'name': self.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ObstacleConfig':
        """从字典创建"""
        return cls(**data)


class ObstacleMapBuilder:
    """
    障碍物地图构建器
    
    提供安全的障碍物设置功能，自动处理边界检查
    """
    
    def __init__(self, height: int, width: int, default_value: float = 0.0):
        """
        初始化障碍物地图构建器
        
        Args:
            height: 地图高度
            width: 地图宽度
            default_value: 默认值
        """
        self.height = height
        self.width = width
        self.map = np.full((height, width), default_value, dtype=np.float64)
        self.obstacles: List[ObstacleConfig] = []
    
    def add_obstacle(
        self,
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int,
        value: float = 1.0,
        name: str = "obstacle",
        check_bounds: bool = True,
        auto_clip: bool = True
    ) -> bool:
        """
        添加障碍物
        
        Args:
            row_start: 起始行
            row_end: 结束行
            col_start: 起始列
            col_end: 结束列
            value: 障碍物值
            name: 障碍物名称
            check_bounds: 是否检查边界
            auto_clip: 是否自动裁剪到边界内
        
        Returns:
            是否成功添加
        
        Examples:
            >>> builder = ObstacleMapBuilder(100, 100)
            >>> builder.add_obstacle(40, 80, 60, 100)  # OK
            >>> builder.add_obstacle(90, 110, 60, 100)  # 会自动裁剪或报错
        """
        # 边界检查
        if check_bounds:
            is_valid = check_array_bounds(
                self.map,
                (row_start, row_end, col_start, col_end)
            )
            
            if not is_valid:
                if auto_clip:
                    # 自动裁剪到边界内
                    row_start = max(0, row_start)
                    row_end = min(self.height, row_end)
                    col_start = max(0, col_start)
                    col_end = min(self.width, col_end)
                    
                    # 检查裁剪后是否有效
                    if row_start >= row_end or col_start >= col_end:
                        print(f"[Warning] Obstacle '{name}' is completely outside the map, skipped.")
                        return False
                else:
                    raise NumericalSafetyError(
                        f"Obstacle '{name}' is out of bounds: "
                        f"rows=[{row_start}, {row_end}), cols=[{col_start}, {col_end}), "
                        f"map size=({self.height}, {self.width})"
                    )
        
        # 设置障碍物
        self.map[row_start:row_end, col_start:col_end] = value
        
        # 记录障碍物配置
        config = ObstacleConfig(
            row_start=row_start,
            row_end=row_end,
            col_start=col_start,
            col_end=col_end,
            value=value,
            name=name
        )
        self.obstacles.append(config)
        
        return True
    
    def add_rectangular_obstacle(
        self,
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        resolution: float = 1.0,
        value: float = 1.0,
        name: str = "rect_obstacle"
    ) -> bool:
        """
        添加矩形障碍物（物理坐标）
        
        Args:
            center_x: 中心x坐标
            center_y: 中心y坐标
            width: 宽度
            height: 高度
            resolution: 分辨率（米/像素）
            value: 障碍物值
            name: 障碍物名称
        
        Returns:
            是否成功添加
        """
        # 转换为像素坐标
        half_width_px = int(width / resolution / 2)
        half_height_px = int(height / resolution / 2)
        center_x_px = int(center_x / resolution)
        center_y_px = int(center_y / resolution)
        
        # 计算行列范围
        row_start = center_y_px - half_height_px
        row_end = center_y_px + half_height_px
        col_start = center_x_px - half_width_px
        col_end = center_x_px + half_width_px
        
        return self.add_obstacle(
            row_start, row_end, col_start, col_end,
            value=value, name=name
        )
    
    def add_circular_obstacle(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        resolution: float = 1.0,
        value: float = 1.0,
        name: str = "circle_obstacle"
    ) -> bool:
        """
        添加圆形障碍物（物理坐标）
        
        Args:
            center_x: 中心x坐标
            center_y: 中心y坐标
            radius: 半径
            resolution: 分辨率（米/像素）
            value: 障碍物值
            name: 障碍物名称
        
        Returns:
            是否成功添加
        """
        # 转换为像素坐标
        center_x_px = int(center_x / resolution)
        center_y_px = int(center_y / resolution)
        radius_px = int(radius / resolution)
        
        # 创建圆形掩码
        y, x = np.ogrid[:self.height, :self.width]
        dist_sq = (x - center_x_px)**2 + (y - center_y_px)**2
        mask = dist_sq <= radius_px**2
        
        # 裁剪到边界内
        mask = mask & (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        
        # 设置障碍物
        self.map[mask] = value
        
        # 记录（简化记录，只记录边界框）
        row_indices, col_indices = np.where(mask)
        if len(row_indices) > 0:
            config = ObstacleConfig(
                row_start=int(row_indices.min()),
                row_end=int(row_indices.max()) + 1,
                col_start=int(col_indices.min()),
                col_end=int(col_indices.max()) + 1,
                value=value,
                name=name
            )
            self.obstacles.append(config)
            return True
        
        return False
    
    def get_map(self) -> np.ndarray:
        """获取障碍物地图"""
        return self.map.copy()
    
    def clear(self) -> None:
        """清除所有障碍物"""
        self.map.fill(0.0)
        self.obstacles.clear()
    
    def get_obstacle_configs(self) -> List[ObstacleConfig]:
        """获取所有障碍物配置"""
        return self.obstacles.copy()
    
    def visualize(self, show: bool = True) -> Optional[np.ndarray]:
        """
        可视化障碍物地图
        
        Args:
            show: 是否显示
        
        Returns:
            图像数组（如果show=False）
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            im = ax.imshow(self.map, cmap='binary', origin='lower')
            ax.set_title('Obstacle Map')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            plt.colorbar(im, ax=ax, label='Obstacle Value')
            
            # 标注障碍物
            for obs in self.obstacles:
                rect = plt.Rectangle(
                    (obs.col_start, obs.row_start),
                    obs.col_end - obs.col_start,
                    obs.row_end - obs.row_start,
                    fill=False,
                    edgecolor='red',
                    linewidth=2
                )
                ax.add_patch(rect)
                ax.text(
                    (obs.col_start + obs.col_end) / 2,
                    (obs.row_start + obs.row_end) / 2,
                    obs.name,
                    ha='center',
                    va='center',
                    color='red'
                )
            
            if show:
                plt.show()
                return None
            else:
                return fig
        except ImportError:
            print("[Warning] matplotlib not available, cannot visualize")
            return None


def create_standard_test_map(
    width: int = 150,
    height: int = 150,
    obstacle_configs: Optional[List[Dict]] = None
) -> np.ndarray:
    """
    创建标准测试障碍物地图
    
    Args:
        width: 地图宽度
        height: 地图高度
        obstacle_configs: 障碍物配置列表，每个配置为字典格式
    
    Returns:
        障碍物地图
    
    Examples:
        >>> # 使用默认配置
        >>> obstacle_map = create_standard_test_map()
        
        >>> # 自定义配置
        >>> configs = [
        ...     {'row_start': 40, 'row_end': 80, 'col_start': 60, 'col_end': 100, 'name': 'obs1'},
        ...     {'row_start': 20, 'row_end': 40, 'col_start': 20, 'col_end': 60, 'name': 'obs2'}
        ... ]
        >>> obstacle_map = create_standard_test_map(150, 150, configs)
    """
    builder = ObstacleMapBuilder(height, width)
    
    if obstacle_configs is None:
        # 默认测试配置
        obstacle_configs = [
            {'row_start': 40, 'row_end': 80, 'col_start': 60, 'col_end': 100, 'name': 'obstacle1'},
            {'row_start': 20, 'row_end': 40, 'col_start': 20, 'col_end': 60, 'name': 'obstacle2'},
        ]
    
    for config in obstacle_configs:
        builder.add_obstacle(**config)
    
    return builder.get_map()


def safe_obstacle_set(
    obstacle_map: np.ndarray,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    value: float = 1.0
) -> bool:
    """
    安全设置障碍物（单次操作）
    
    Args:
        obstacle_map: 障碍物地图
        row_start: 起始行
        row_end: 结束行
        col_start: 起始列
        col_end: 结束列
        value: 障碍物值
    
    Returns:
        是否成功设置
    
    Examples:
        >>> obstacle_map = np.zeros((100, 100))
        >>> safe_obstacle_set(obstacle_map, 40, 80, 60, 100)  # OK
        >>> safe_obstacle_set(obstacle_map, 90, 110, 60, 100)  # 自动裁剪
    """
    height, width = obstacle_map.shape
    
    # 边界检查和自动裁剪
    row_start = max(0, row_start)
    row_end = min(height, row_end)
    col_start = max(0, col_start)
    col_end = min(width, col_end)
    
    # 检查是否有效
    if row_start >= row_end or col_start >= col_end:
        return False
    
    # 设置障碍物
    obstacle_map[row_start:row_end, col_start:col_end] = value
    
    return True
