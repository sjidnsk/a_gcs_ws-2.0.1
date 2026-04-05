"""
2D轨迹绘制模块
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from pydrake.geometry.optimization import HPolyhedron

from .config import VisualizationConfig, ControlPointConfig, ControlPointData
from .trajectory_sampler import TrajectoryData
from .region_renderer import RegionRenderer
from .path_comparator import PathComparator
from ackermann_gcs_pkg.ackermann_data_structures import EndpointState


class Plot2DTrajectory:
    """2D轨迹绘制器
    
    绘制增强的2D轨迹视图，包含IRIS区域、障碍物、走廊等
    """
    
    def __init__(self, config: VisualizationConfig):
        """初始化2D轨迹绘制器
        
        Args:
            config: 可视化配置
        """
        self.config = config
        self.region_renderer = RegionRenderer(config.iris_colors)
        self.path_comparator = PathComparator(
            trajectory_color=config.trajectory_color,
            trajectory_linewidth=config.trajectory_linewidth,
            astar_color=config.astar_color,
            astar_linestyle=config.astar_linestyle
        )
    
    def plot(
        self,
        ax: plt.Axes,
        trajectory_data: TrajectoryData,
        workspace_regions: Optional[List[HPolyhedron]] = None,
        obstacle_map: Optional[np.ndarray] = None,
        astar_path: Optional[np.ndarray] = None,
        source: Optional[EndpointState] = None,
        target: Optional[EndpointState] = None,
        corridor_width: Optional[float] = None,
        resolution: float = 0.1,
        control_point_data: Optional[ControlPointData] = None,
        control_point_config: Optional[ControlPointConfig] = None
    ) -> None:
        """绘制2D轨迹视图
        
        Args:
            ax: matplotlib坐标轴
            trajectory_data: 轨迹数据
            workspace_regions: IRIS区域列表
            obstacle_map: 障碍物地图
            astar_path: A*路径
            source: 起点状态
            target: 终点状态
            corridor_width: 走廊宽度
            resolution: 地图分辨率（米/像素）
            control_point_data: 控制点数据（可选）
            control_point_config: 控制点配置（可选）
        """
        # 1. 绘制障碍物地图
        if self.config.show_obstacles and obstacle_map is not None:
            self._plot_obstacles(ax, obstacle_map, resolution)
        
        # 2. 绘制IRIS区域
        if self.config.show_iris_regions and workspace_regions is not None:
            self.region_renderer.plot_regions(
                ax, workspace_regions,
                alpha=self.config.iris_alpha,
                show_labels=True
            )
        
        # 3. 绘制走廊约束
        if self.config.show_corridor and corridor_width is not None:
            self._plot_corridor(ax, trajectory_data.position, corridor_width)
        
        # 4. 绘制路径对比
        self.path_comparator.plot_paths(
            ax,
            trajectory_data.position,
            astar_path=astar_path if self.config.show_astar_path else None,
            source=source,
            target=target,
            show_endpoints=True,
            show_heading_arrows=True
        )
        
        # 5. 绘制控制点
        if control_point_data is not None:
            config = control_point_config or ControlPointConfig()
            if config.show_control_points:
                self.plot_control_points(ax, control_point_data, config)
        
        # 6. 设置坐标轴
        self._setup_axes(ax, len(workspace_regions) if workspace_regions else 0)
    
    def _plot_obstacles(
        self,
        ax: plt.Axes,
        obstacle_map: np.ndarray,
        resolution: float = 0.1
    ) -> None:
        """绘制障碍物地图
        
        Args:
            ax: matplotlib坐标轴
            obstacle_map: 障碍物地图
            resolution: 地图分辨率（米/像素）
        """
        try:
            # 计算物理坐标范围
            # obstacle_map的像素坐标范围: [0, shape[0]-1] x [0, shape[1]-1]
            # 物理坐标范围: [0, shape*resolution]
            map_height, map_width = obstacle_map.shape
            x_max = map_width * resolution
            y_max = map_height * resolution
            
            # 设置extent: [x_min, x_max, y_min, y_max]
            extent = [0, x_max, 0, y_max]
            
            # 使用RdYlGn_r颜色映射
            ax.imshow(
                obstacle_map,
                origin='lower',
                cmap='RdYlGn_r',
                alpha=self.config.obstacle_alpha,
                aspect='auto',
                extent=extent,
                zorder=1
            )
        except Exception as e:
            print(f"警告: 障碍物地图绘制失败: {e}")
    
    def _plot_corridor(
        self,
        ax: plt.Axes,
        position: np.ndarray,
        width: float
    ) -> None:
        """绘制走廊约束
        
        Args:
            ax: matplotlib坐标轴
            position: 轨迹位置 (2, N)
            width: 走廊宽度
        """
        # 计算轨迹的法向量
        tangent = np.diff(position, axis=1)
        tangent_norm = np.sqrt(np.sum(tangent**2, axis=0))
        
        # 避免除零
        tangent_norm = np.maximum(tangent_norm, 1e-6)
        
        # 法向量（垂直于切向量）
        normal = np.array([-tangent[1, :], tangent[0, :]]) / tangent_norm
        
        # 计算走廊边界
        half_width = width / 2.0
        upper_boundary = position[:, :-1] + half_width * normal
        lower_boundary = position[:, :-1] - half_width * normal
        
        # 绘制走廊边界
        ax.plot(
            upper_boundary[0, :],
            upper_boundary[1, :],
            'k--',
            linewidth=0.5,
            alpha=0.3,
            label='Corridor Boundary'
        )
        ax.plot(
            lower_boundary[0, :],
            lower_boundary[1, :],
            'k--',
            linewidth=0.5,
            alpha=0.3
        )
    
    def _setup_axes(
        self,
        ax: plt.Axes,
        num_regions: int
    ) -> None:
        """设置坐标轴
        
        Args:
            ax: matplotlib坐标轴
            num_regions: IRIS区域数量
        """
        # 设置标签
        ax.set_xlabel("x (m)", fontsize=self.config.label_font_size)
        ax.set_ylabel("y (m)", fontsize=self.config.label_font_size)
        
        # 设置标题
        title = "2D Trajectory"
        if num_regions > 0:
            title += f" with {num_regions} IRIS Regions"
        ax.set_title(title, fontsize=self.config.title_font_size)
        
        # 设置图例
        ax.legend(loc='best', fontsize=self.config.font_size)
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        
        # 设置等比例
        ax.axis('equal')
    
    def plot_control_points(
        self,
        ax: plt.Axes,
        control_point_data: ControlPointData,
        config: ControlPointConfig
    ) -> None:
        """绘制2D控制点（xy平面投影）
        
        Args:
            ax: matplotlib坐标轴
            control_point_data: 控制点数据
            config: 控制点配置
        """
        points = control_point_data.points
        
        ax.scatter(
            points[:, 0],  # x
            points[:, 1],  # y
            c=config.control_point_color,
            marker=config.control_point_marker,
            s=config.control_point_size,
            alpha=config.control_point_alpha,
            label=config.control_point_label,
            zorder=config.control_point_zorder,
            edgecolors='black',
            linewidths=0.5
        )
    
    def plot_with_keypoints(
        self,
        ax: plt.Axes,
        trajectory_data: TrajectoryData,
        keypoints: Optional[List[Tuple[float, float, str]]] = None,
        **kwargs
    ) -> None:
        """绘制带关键点的2D轨迹视图
        
        Args:
            ax: matplotlib坐标轴
            trajectory_data: 轨迹数据
            keypoints: 关键点列表 [(x, y, label), ...]
            **kwargs: 其他参数
        """
        # 绘制基本2D视图
        self.plot(ax, trajectory_data, **kwargs)
        
        # 绘制关键点
        if keypoints:
            for x, y, label in keypoints:
                ax.plot(x, y, 'ko', markersize=8, zorder=5)
                ax.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=self.config.font_size - 2,
                    zorder=5
                )
    
    def plot_velocity_heatmap(
        self,
        ax: plt.Axes,
        trajectory_data: TrajectoryData,
        **kwargs
    ) -> None:
        """绘制速度热力图
        
        Args:
            ax: matplotlib坐标轴
            trajectory_data: 轨迹数据
            **kwargs: 其他参数
        """
        # 绘制轨迹，颜色表示速度
        points = trajectory_data.position.T  # (N, 2)
        velocities = trajectory_data.velocity
        
        # 创建颜色映射
        from matplotlib.collections import LineCollection
        
        # 创建线段
        segments = np.stack([points[:-1], points[1:]], axis=1)
        
        # 创建LineCollection
        lc = LineCollection(
            segments,
            cmap='viridis',
            norm=plt.Normalize(velocities.min(), velocities.max())
        )
        
        # 设置颜色
        lc.set_array(velocities[:-1])
        lc.set_linewidth(self.config.trajectory_linewidth)
        
        # 添加到坐标轴
        ax.add_collection(lc)
        
        # 添加颜色条
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Velocity (m/s)', fontsize=self.config.font_size)
        
        # 设置坐标轴
        ax.set_xlim(points[:, 0].min(), points[:, 0].max())
        ax.set_ylim(points[:, 1].min(), points[:, 1].max())
        ax.set_xlabel("x (m)", fontsize=self.config.label_font_size)
        ax.set_ylabel("y (m)", fontsize=self.config.label_font_size)
        ax.set_title("Trajectory Velocity Profile", fontsize=self.config.title_font_size)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
