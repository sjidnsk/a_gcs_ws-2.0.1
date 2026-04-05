"""
3D轨迹绘制模块
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional

from .config import VisualizationConfig, ControlPointConfig, ControlPointData
from .trajectory_sampler import TrajectoryData
from ackermann_gcs_pkg.ackermann_data_structures import EndpointState


class Plot3DTrajectory:
    """3D轨迹绘制器
    
    绘制(x, y, θ)配置空间中的3D轨迹
    """
    
    def __init__(self, config: VisualizationConfig):
        """初始化3D轨迹绘制器
        
        Args:
            config: 可视化配置
        """
        self.config = config
    
    def plot(
        self,
        ax: Axes3D,
        trajectory_data: TrajectoryData,
        astar_path: Optional[np.ndarray] = None,
        source: Optional[EndpointState] = None,
        target: Optional[EndpointState] = None,
        control_point_data: Optional[ControlPointData] = None,
        control_point_config: Optional[ControlPointConfig] = None
    ) -> None:
        """绘制3D轨迹视图
        
        Args:
            ax: matplotlib 3D坐标轴
            trajectory_data: 轨迹数据
            astar_path: A*路径 (M, 2) 或 (M, 3)
            source: 起点状态
            target: 终点状态
            control_point_data: 控制点数据（可选）
            control_point_config: 控制点配置（可选）
        """
        # 提取数据
        x = trajectory_data.position[0, :]
        y = trajectory_data.position[1, :]
        theta = trajectory_data.heading
        
        # 1. 绘制GCS轨迹
        ax.plot(
            x, y, theta,
            color=self.config.trajectory_color,
            linewidth=self.config.trajectory_linewidth,
            label='GCS Trajectory',
            zorder=3
        )
        
        # 2. 绘制A*路径
        if astar_path is not None and len(astar_path) > 0:
            self._plot_astar_path_3d(ax, astar_path)
        
        # 3. 绘制起终点
        if source is not None:
            self._plot_endpoint_3d(ax, source, is_source=True)
        if target is not None:
            self._plot_endpoint_3d(ax, target, is_source=False)
        
        # 4. 绘制控制点
        if control_point_data is not None:
            config = control_point_config or ControlPointConfig()
            if config.show_control_points:
                self.plot_control_points(ax, control_point_data, config)
        
        # 5. 设置坐标轴
        self._setup_axes(ax)
    
    def _plot_astar_path_3d(
        self,
        ax: Axes3D,
        path: np.ndarray
    ) -> None:
        """绘制A*路径的3D投影
        
        Args:
            ax: matplotlib 3D坐标轴
            path: 路径点 (M, 2) 或 (M, 3)
        """
        if path.shape[1] >= 3:
            # 路径包含θ信息
            x = path[:, 0]
            y = path[:, 1]
            theta = path[:, 2]
        elif path.shape[1] == 2:
            # 路径只有x, y，θ设为0
            x = path[:, 0]
            y = path[:, 1]
            theta = np.zeros(len(path))
        else:
            return
        
        ax.plot(
            x, y, theta,
            color=self.config.astar_color,
            linestyle=self.config.astar_linestyle,
            linewidth=1.5,
            label='A* Path',
            zorder=2
        )
    
    def _plot_endpoint_3d(
        self,
        ax: Axes3D,
        endpoint: EndpointState,
        is_source: bool = True
    ) -> None:
        """绘制3D端点
        
        Args:
            ax: matplotlib 3D坐标轴
            endpoint: 端点状态
            is_source: 是否为起点
        """
        # 设置颜色和标记
        if is_source:
            color = 'green'
            marker = 's'  # 方块
            label = 'Source'
            markersize = 10
        else:
            color = 'red'
            marker = '*'  # 星形
            label = 'Target'
            markersize = 15
        
        # 绘制端点
        ax.scatter(
            endpoint.position[0],
            endpoint.position[1],
            endpoint.heading,
            c=color,
            marker=marker,
            s=markersize**2,
            label=label,
            zorder=4
        )
    
    def _setup_axes(self, ax: Axes3D) -> None:
        """设置3D坐标轴
        
        Args:
            ax: matplotlib 3D坐标轴
        """
        # 设置标签
        ax.set_xlabel('x (m)', fontsize=self.config.label_font_size, fontweight='bold')
        ax.set_ylabel('y (m)', fontsize=self.config.label_font_size, fontweight='bold')
        ax.set_zlabel('θ (rad)', fontsize=self.config.label_font_size, fontweight='bold')
        
        # 设置标题
        ax.set_title('3D Configuration Space Trajectory', fontsize=self.config.title_font_size)
        
        # 设置视角
        ax.view_init(elev=self.config.elev, azim=self.config.azim)
        
        # 设置图例
        ax.legend(loc='best', fontsize=self.config.font_size)
        
        # 设置网格
        ax.grid(True, alpha=0.3)
    
    def plot_control_points(
        self,
        ax: Axes3D,
        control_point_data: ControlPointData,
        config: ControlPointConfig
    ) -> None:
        """绘制3D控制点
        
        Args:
            ax: matplotlib 3D坐标轴
            control_point_data: 控制点数据
            config: 控制点配置
        """
        points = control_point_data.points
        
        ax.scatter(
            points[:, 0],  # x
            points[:, 1],  # y
            points[:, 2],  # theta
            c=config.control_point_color,
            marker=config.control_point_marker,
            s=config.control_point_size,
            alpha=config.control_point_alpha,
            label=config.control_point_label,
            zorder=config.control_point_zorder,
            edgecolors='black',
            linewidths=0.5
        )
    
    def plot_with_projection(
        self,
        ax: Axes3D,
        trajectory_data: TrajectoryData,
        show_xy_projection: bool = True,
        show_xtheta_projection: bool = True,
        show_ytheta_projection: bool = True,
        **kwargs
    ) -> None:
        """绘制带投影的3D轨迹
        
        Args:
            ax: matplotlib 3D坐标轴
            trajectory_data: 轨迹数据
            show_xy_projection: 是否显示xy投影
            show_xtheta_projection: 是否显示xθ投影
            show_ytheta_projection: 是否显示yθ投影
            **kwargs: 其他参数
        """
        # 绘制基本3D轨迹
        self.plot(ax, trajectory_data, **kwargs)
        
        # 提取数据
        x = trajectory_data.position[0, :]
        y = trajectory_data.position[1, :]
        theta = trajectory_data.heading
        
        # 绘制xy投影（在θ=0平面）
        if show_xy_projection:
            ax.plot(
                x, y, np.zeros_like(theta),
                color='gray',
                linewidth=1,
                alpha=0.3,
                linestyle=':',
                label='xy Projection'
            )
        
        # 绘制xθ投影（在y=0平面）
        if show_xtheta_projection:
            ax.plot(
                x, np.zeros_like(y), theta,
                color='gray',
                linewidth=1,
                alpha=0.3,
                linestyle='--',
                label='xθ Projection'
            )
        
        # 绘制yθ投影（在x=0平面）
        if show_ytheta_projection:
            ax.plot(
                np.zeros_like(x), y, theta,
                color='gray',
                linewidth=1,
                alpha=0.3,
                linestyle='-.',
                label='yθ Projection'
            )


def visualize_3d_trajectory(
    trajectory_data: TrajectoryData,
    source: Optional[EndpointState] = None,
    target: Optional[EndpointState] = None,
    astar_path: Optional[np.ndarray] = None,
    elev: float = 25.0,
    azim: float = 45.0,
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
    dpi: int = 150
) -> None:
    """独立的3D轨迹可视化接口
    
    Args:
        trajectory_data: 轨迹数据
        source: 起点状态
        target: 终点状态
        astar_path: A*路径
        elev: 视角仰角
        azim: 视角方位角
        figsize: 图表大小
        save_path: 保存路径
        dpi: 分辨率
    """
    # 创建配置
    config = VisualizationConfig(elev=elev, azim=azim)
    
    # 创建图表
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建绘制器
    plotter = Plot3DTrajectory(config)
    
    # 绘制
    plotter.plot(ax, trajectory_data, astar_path, source, target)
    
    # 保存或显示
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"[Visualizer] 3D trajectory saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
