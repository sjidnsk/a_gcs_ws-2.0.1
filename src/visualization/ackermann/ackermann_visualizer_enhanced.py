"""
增强版AckermannGCS可视化器
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from pydrake.geometry.optimization import HPolyhedron
from pydrake.trajectories import BsplineTrajectory

from config.visualization import VisualizationConfig
from .trajectory_sampler import TrajectorySampler, TrajectoryData
from .plot_2d_trajectory import Plot2DTrajectory
from .plot_3d_trajectory import Plot3DTrajectory
from .plot_profiles import PlotProfiles
from ackermann_gcs_pkg.ackermann_data_structures import VehicleParams, EndpointState
from ..core.output_manager import VisualizationOutputManager


class AckermannGCSVisualizer:
    """增强版AckermannGCS可视化器
    
    提供综合可视化功能，包括：
    - 2D轨迹视图（带IRIS区域、障碍物、走廊）
    - 3D配置空间轨迹视图
    - 速度、航向角、曲率等曲线图
    """
    
    def __init__(
        self,
        config: Optional[VisualizationConfig] = None,
        output_manager: Optional[VisualizationOutputManager] = None
    ):
        """初始化可视化器
        
        Args:
            config: 可视化配置，如果为None则使用默认配置
            output_manager: 输出管理器，如果为None则使用单例实例
        """
        self.config = config or VisualizationConfig()
        self.config.validate()
        
        # 初始化输出管理器
        self.output_manager = (
            output_manager or VisualizationOutputManager.get_instance()
        )
        
        # 初始化子模块
        self.sampler = TrajectorySampler(num_samples=self.config.num_samples)
        self.plot_2d = Plot2DTrajectory(self.config)
        self.plot_3d = Plot3DTrajectory(self.config)
        self.plot_profiles = PlotProfiles(self.config)
    
    def visualize(
        self,
        trajectory: BsplineTrajectory,
        vehicle_params: VehicleParams,
        workspace_regions: Optional[List[HPolyhedron]] = None,
        source: Optional[EndpointState] = None,
        target: Optional[EndpointState] = None,
        obstacle_map: Optional[np.ndarray] = None,
        astar_path: Optional[np.ndarray] = None,
        corridor_width: Optional[float] = None,
        resolution: float = 0.1,
        save_path: Optional[str] = None,
        run_id: Optional[str] = None,
        show: bool = False
    ) -> None:
        """执行综合可视化
        
        Args:
            trajectory: B样条轨迹
            vehicle_params: 车辆参数
            workspace_regions: IRIS区域列表
            source: 起点状态
            target: 终点状态
            obstacle_map: 障碍物地图
            astar_path: A*路径
            corridor_width: 走廊宽度
            resolution: 地图分辨率（米/像素）
            save_path: 保存路径（向后兼容，推荐使用run_id）
            run_id: 运行实例标识，用于自动生成规范化路径
            show: 是否显示图表
        """
        # 采样轨迹
        trajectory_data = self.sampler.sample(trajectory, vehicle_params)
        
        # 创建图表布局
        fig, axes = plt.subplots(2, 3, figsize=self.config.figsize)
        
        # 设置字体
        plt.rcParams['font.family'] = self.config.font_family
        
        # 1. 绘制2D轨迹视图
        ax_2d = axes[0, 0]
        self.plot_2d.plot(
            ax_2d,
            trajectory_data,
            workspace_regions=workspace_regions,
            obstacle_map=obstacle_map,
            astar_path=astar_path,
            source=source,
            target=target,
            corridor_width=corridor_width,
            resolution=resolution
        )
        
        # 2. 绘制3D轨迹视图
        if self.config.show_3d_trajectory:
            ax_3d = fig.add_subplot(2, 3, 4, projection='3d')
            self.plot_3d.plot(
                ax_3d,
                trajectory_data,
                astar_path=astar_path,
                source=source,
                target=target
            )
            # 隐藏原来的axes[1, 0]
            axes[1, 0].set_visible(False)
        
        # 3. 绘制曲线图
        self.plot_profiles.plot_all_profiles(
            axes,
            trajectory_data,
            vehicle_params,
            astar_path=astar_path
        )
        
        # 4. 绘制θ随路径长度变化曲线（如果启用）
        if self.config.show_theta_profile:
            # 使用axes[1, 0]（如果3D轨迹未启用）
            if not self.config.show_3d_trajectory:
                ax_theta = axes[1, 0]
                self.plot_profiles.plot_theta_vs_path_length(
                    ax_theta,
                    trajectory_data,
                    astar_path=astar_path
                )
        
        # 设置子图间距
        plt.subplots_adjust(
            hspace=self.config.subplot_hspace,
            wspace=self.config.subplot_wspace
        )
        
        # 保存图表
        if save_path or run_id:
            # 如果提供了run_id，使用输出管理器生成路径
            if run_id and not save_path:
                save_path = self.output_manager.generate_output_path(
                    filename="ackermann_gcs_enhanced.png",
                    dimension="2d",  # 主视图是2D
                    run_id=run_id
                )
            self._save_figure(fig, save_path)
        
        # 显示图表
        if show:
            plt.show()
        else:
            plt.close(fig)
    
    def _save_figure(self, fig: plt.Figure, save_path: str) -> None:
        """保存图表
        
        Args:
            fig: matplotlib图表对象
            save_path: 保存路径
        """
        # 创建目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存图表
        fig.savefig(
            save_path,
            dpi=self.config.dpi,
            format=self.config.save_format,
            bbox_inches=self.config.bbox_inches
        )
        
        print(f"[Visualizer] Visualization saved to {save_path}")
    
    def visualize_2d_only(
        self,
        trajectory: BsplineTrajectory,
        vehicle_params: VehicleParams,
        workspace_regions: Optional[List[HPolyhedron]] = None,
        source: Optional[EndpointState] = None,
        target: Optional[EndpointState] = None,
        obstacle_map: Optional[np.ndarray] = None,
        astar_path: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> None:
        """仅绘制2D轨迹视图
        
        Args:
            trajectory: B样条轨迹
            vehicle_params: 车辆参数
            workspace_regions: IRIS区域列表
            source: 起点状态
            target: 终点状态
            obstacle_map: 障碍物地图
            astar_path: A*路径
            save_path: 保存路径（向后兼容）
            run_id: 运行实例标识
        """
        # 采样轨迹
        trajectory_data = self.sampler.sample(trajectory, vehicle_params)
        
        # 创建图表
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 绘制2D轨迹
        self.plot_2d.plot(
            ax,
            trajectory_data,
            workspace_regions=workspace_regions,
            obstacle_map=obstacle_map,
            astar_path=astar_path,
            source=source,
            target=target
        )
        
        # 保存图表
        if save_path or run_id:
            if run_id and not save_path:
                save_path = self.output_manager.generate_output_path(
                    filename="trajectory_2d.png",
                    dimension="2d",
                    run_id=run_id
                )
            self._save_figure(fig, save_path)
        
        plt.close(fig)
    
    def visualize_3d_only(
        self,
        trajectory: BsplineTrajectory,
        vehicle_params: VehicleParams,
        source: Optional[EndpointState] = None,
        target: Optional[EndpointState] = None,
        astar_path: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        run_id: Optional[str] = None
    ) -> None:
        """仅绘制3D轨迹视图
        
        Args:
            trajectory: B样条轨迹
            vehicle_params: 车辆参数
            source: 起点状态
            target: 终点状态
            astar_path: A*路径
            save_path: 保存路径（向后兼容）
            run_id: 运行实例标识
        """
        # 采样轨迹
        trajectory_data = self.sampler.sample(trajectory, vehicle_params)
        
        # 创建图表
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制3D轨迹
        self.plot_3d.plot(
            ax,
            trajectory_data,
            astar_path=astar_path,
            source=source,
            target=target
        )
        
        # 保存图表
        if save_path or run_id:
            if run_id and not save_path:
                save_path = self.output_manager.generate_output_path(
                    filename="trajectory_3d.png",
                    dimension="3d",
                    run_id=run_id
                )
            self._save_figure(fig, save_path)
        
        plt.close(fig)


def visualize_ackermann_gcs_enhanced(
    trajectory: BsplineTrajectory,
    vehicle_params: VehicleParams,
    workspace_regions: Optional[List[HPolyhedron]] = None,
    source: Optional[EndpointState] = None,
    target: Optional[EndpointState] = None,
    obstacle_map: Optional[np.ndarray] = None,
    astar_path: Optional[np.ndarray] = None,
    corridor_width: Optional[float] = None,
    resolution: float = 0.1,
    save_path: str = "./ackermann_gcs_enhanced.png",
    config: Optional[VisualizationConfig] = None
) -> None:
    """便捷接口：增强版AckermannGCS可视化
    
    Args:
        trajectory: B样条轨迹
        vehicle_params: 车辆参数
        workspace_regions: IRIS区域列表
        source: 起点状态
        target: 终点状态
        obstacle_map: 障碍物地图
        astar_path: A*路径
        corridor_width: 走廊宽度
        resolution: 地图分辨率（米/像素）
        save_path: 保存路径
        config: 可视化配置
    """
    # 创建可视化器
    visualizer = AckermannGCSVisualizer(config)
    
    # 执行可视化
    visualizer.visualize(
        trajectory=trajectory,
        vehicle_params=vehicle_params,
        workspace_regions=workspace_regions,
        source=source,
        target=target,
        obstacle_map=obstacle_map,
        astar_path=astar_path,
        corridor_width=corridor_width,
        resolution=resolution,
        save_path=save_path,
        show=False
    )
