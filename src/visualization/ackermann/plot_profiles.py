"""
曲线图绘制模块
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .config import VisualizationConfig
from .trajectory_sampler import TrajectoryData
from ackermann_gcs_pkg.ackermann_data_structures import VehicleParams


class PlotProfiles:
    """曲线图绘制器
    
    绘制速度、航向角、曲率、转向角等曲线图
    """
    
    def __init__(self, config: VisualizationConfig):
        """初始化曲线图绘制器
        
        Args:
            config: 可视化配置
        """
        self.config = config
    
    def plot_velocity(
        self,
        ax: plt.Axes,
        trajectory_data: TrajectoryData,
        vehicle_params: VehicleParams
    ) -> None:
        """绘制速度曲线
        
        Args:
            ax: matplotlib坐标轴
            trajectory_data: 轨迹数据
            vehicle_params: 车辆参数
        """
        t = trajectory_data.t_samples
        v = trajectory_data.velocity
        
        # 绘制速度曲线
        ax.plot(t, v, 'b-', linewidth=2, label='Velocity')
        
        # 绘制最大速度限制
        ax.axhline(
            y=vehicle_params.max_velocity,
            color='r',
            linestyle='--',
            label=f'Max Velocity ({vehicle_params.max_velocity} m/s)'
        )
        
        # 设置坐标轴
        ax.set_xlabel('Time (s)', fontsize=self.config.label_font_size)
        ax.set_ylabel('Velocity (m/s)', fontsize=self.config.label_font_size)
        ax.set_title('Velocity Profile', fontsize=self.config.title_font_size)
        ax.legend(loc='best', fontsize=self.config.font_size)
        ax.grid(True, alpha=0.3)
    
    def plot_heading(
        self,
        ax: plt.Axes,
        trajectory_data: TrajectoryData
    ) -> None:
        """绘制航向角曲线
        
        Args:
            ax: matplotlib坐标轴
            trajectory_data: 轨迹数据
        """
        t = trajectory_data.t_samples
        theta = trajectory_data.heading
        
        # 绘制航向角曲线
        ax.plot(t, np.degrees(theta), 'b-', linewidth=2, label='Heading')
        
        # 设置坐标轴
        ax.set_xlabel('Time (s)', fontsize=self.config.label_font_size)
        ax.set_ylabel('Heading (deg)', fontsize=self.config.label_font_size)
        ax.set_title('Heading Angle Profile', fontsize=self.config.title_font_size)
        ax.legend(loc='best', fontsize=self.config.font_size)
        ax.grid(True, alpha=0.3)
        
        # 设置y轴范围（-180到180度）
        ax.set_ylim(-180, 180)
    
    def plot_curvature(
        self,
        ax: plt.Axes,
        trajectory_data: TrajectoryData,
        vehicle_params: VehicleParams
    ) -> None:
        """绘制曲率曲线
        
        Args:
            ax: matplotlib坐标轴
            trajectory_data: 轨迹数据
            vehicle_params: 车辆参数
        """
        t = trajectory_data.t_samples
        kappa = trajectory_data.curvature
        
        # 绘制曲率曲线
        ax.plot(t, kappa, 'b-', linewidth=2, label='Curvature')
        
        # 绘制最大曲率限制
        ax.axhline(
            y=vehicle_params.max_curvature,
            color='r',
            linestyle='--',
            label=f'Max Curvature ({vehicle_params.max_curvature:.3f})'
        )
        ax.axhline(
            y=-vehicle_params.max_curvature,
            color='r',
            linestyle='--'
        )
        
        # 设置坐标轴
        ax.set_xlabel('Time (s)', fontsize=self.config.label_font_size)
        ax.set_ylabel('Curvature (1/m)', fontsize=self.config.label_font_size)
        ax.set_title('Curvature Profile', fontsize=self.config.title_font_size)
        ax.legend(loc='best', fontsize=self.config.font_size)
        ax.grid(True, alpha=0.3)
    
    def plot_steering(
        self,
        ax: plt.Axes,
        trajectory_data: TrajectoryData,
        vehicle_params: VehicleParams
    ) -> None:
        """绘制转向角曲线
        
        Args:
            ax: matplotlib坐标轴
            trajectory_data: 轨迹数据
            vehicle_params: 车辆参数
        """
        t = trajectory_data.t_samples
        delta = trajectory_data.steering_angle
        
        # 绘制转向角曲线（转换为度）
        ax.plot(t, np.degrees(delta), 'b-', linewidth=2, label='Steering Angle')
        
        # 绘制最大转向角限制
        max_delta_deg = np.degrees(vehicle_params.max_steering_angle)
        ax.axhline(
            y=max_delta_deg,
            color='r',
            linestyle='--',
            label=f'Max Steering ({max_delta_deg:.1f}°)'
        )
        ax.axhline(
            y=-max_delta_deg,
            color='r',
            linestyle='--'
        )
        
        # 设置坐标轴
        ax.set_xlabel('Time (s)', fontsize=self.config.label_font_size)
        ax.set_ylabel('Steering Angle (deg)', fontsize=self.config.label_font_size)
        ax.set_title('Steering Angle Profile', fontsize=self.config.title_font_size)
        ax.legend(loc='best', fontsize=self.config.font_size)
        ax.grid(True, alpha=0.3)
    
    def plot_acceleration(
        self,
        ax: plt.Axes,
        trajectory_data: TrajectoryData,
        vehicle_params: VehicleParams
    ) -> None:
        """绘制加速度曲线
        
        Args:
            ax: matplotlib坐标轴
            trajectory_data: 轨迹数据
            vehicle_params: 车辆参数
        """
        t = trajectory_data.t_samples
        a = trajectory_data.acceleration
        
        # 绘制加速度曲线
        ax.plot(t, a, 'b-', linewidth=2, label='Acceleration')
        
        # 绘制最大加速度限制
        ax.axhline(
            y=vehicle_params.max_acceleration,
            color='r',
            linestyle='--',
            label=f'Max Acceleration ({vehicle_params.max_acceleration} m/s²)'
        )
        ax.axhline(
            y=-vehicle_params.max_acceleration,
            color='r',
            linestyle='--'
        )
        
        # 设置坐标轴
        ax.set_xlabel('Time (s)', fontsize=self.config.label_font_size)
        ax.set_ylabel('Acceleration (m/s²)', fontsize=self.config.label_font_size)
        ax.set_title('Acceleration Profile', fontsize=self.config.title_font_size)
        ax.legend(loc='best', fontsize=self.config.font_size)
        ax.grid(True, alpha=0.3)
    
    def plot_theta_vs_path_length(
        self,
        ax: plt.Axes,
        trajectory_data: TrajectoryData,
        astar_path: Optional[np.ndarray] = None
    ) -> None:
        """绘制θ随路径长度变化曲线
        
        Args:
            ax: matplotlib坐标轴
            trajectory_data: 轨迹数据
            astar_path: A*路径
        """
        # 绘制GCS轨迹的θ曲线
        if trajectory_data.path_length is not None:
            s = trajectory_data.path_length
            theta = trajectory_data.heading
            
            ax.plot(s, np.degrees(theta), 'r-', linewidth=2, label='GCS Trajectory')
        
        # 绘制A*路径的θ曲线（如果有）
        if astar_path is not None and astar_path.shape[1] >= 3:
            # 计算A*路径长度
            diff = np.diff(astar_path[:, :2], axis=0)
            distances = np.sqrt(np.sum(diff**2, axis=1))
            s_astar = np.zeros(len(astar_path))
            s_astar[1:] = np.cumsum(distances)
            
            theta_astar = astar_path[:, 2]
            
            ax.plot(s_astar, np.degrees(theta_astar), 'g--', linewidth=1.5, label='A* Path')
        
        # 设置坐标轴
        ax.set_xlabel('Path Length (m)', fontsize=self.config.label_font_size)
        ax.set_ylabel('Heading θ (deg)', fontsize=self.config.label_font_size)
        ax.set_title('Heading vs Path Length', fontsize=self.config.title_font_size)
        ax.legend(loc='best', fontsize=self.config.font_size)
        ax.grid(True, alpha=0.3)
    
    def plot_all_profiles(
        self,
        axes: np.ndarray,
        trajectory_data: TrajectoryData,
        vehicle_params: VehicleParams,
        astar_path: Optional[np.ndarray] = None
    ) -> None:
        """绘制所有曲线图
        
        Args:
            axes: matplotlib坐标轴数组 (2, 3)
            trajectory_data: 轨迹数据
            vehicle_params: 车辆参数
            astar_path: A*路径
        """
        # 速度曲线
        if self.config.show_velocity:
            self.plot_velocity(axes[0, 1], trajectory_data, vehicle_params)
        
        # 航向角曲线
        if self.config.show_heading:
            self.plot_heading(axes[0, 2], trajectory_data)
        
        # 曲率曲线
        if self.config.show_curvature:
            self.plot_curvature(axes[1, 0], trajectory_data, vehicle_params)
        
        # 转向角曲线
        if self.config.show_steering:
            self.plot_steering(axes[1, 1], trajectory_data, vehicle_params)
        
        # 加速度曲线
        if self.config.show_acceleration:
            self.plot_acceleration(axes[1, 2], trajectory_data, vehicle_params)
