"""
路径对比模块
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

from ackermann_gcs_pkg.ackermann_data_structures import EndpointState


class PathComparator:
    """路径对比器
    
    负责绘制和对比A*路径与GCS轨迹
    """
    
    def __init__(
        self,
        trajectory_color: str = 'red',
        trajectory_linewidth: float = 2.0,
        astar_color: str = 'green',
        astar_linestyle: str = '--',
        astar_linewidth: float = 1.5
    ):
        """初始化路径对比器
        
        Args:
            trajectory_color: 轨迹颜色
            trajectory_linewidth: 轨迹线宽
            astar_color: A*路径颜色
            astar_linestyle: A*路径线型
            astar_linewidth: A*路径线宽
        """
        self.trajectory_color = trajectory_color
        self.trajectory_linewidth = trajectory_linewidth
        self.astar_color = astar_color
        self.astar_linestyle = astar_linestyle
        self.astar_linewidth = astar_linewidth
    
    def plot_paths(
        self,
        ax: plt.Axes,
        trajectory_position: np.ndarray,
        astar_path: Optional[np.ndarray] = None,
        source: Optional[EndpointState] = None,
        target: Optional[EndpointState] = None,
        show_endpoints: bool = True,
        show_heading_arrows: bool = True
    ) -> None:
        """绘制路径对比
        
        Args:
            ax: matplotlib坐标轴
            trajectory_position: GCS轨迹位置 (2, N)
            astar_path: A*路径点 (M, 2) 或 (M, 3)
            source: 起点状态
            target: 终点状态
            show_endpoints: 是否显示起终点
            show_heading_arrows: 是否显示航向角箭头
        """
        # 绘制A*路径
        if astar_path is not None and len(astar_path) > 0:
            self._plot_astar_path(ax, astar_path)
        
        # 绘制GCS轨迹
        self._plot_trajectory(ax, trajectory_position)
        
        # 绘制起终点
        if show_endpoints:
            if source is not None:
                self._plot_endpoint(
                    ax, source, is_source=True,
                    show_arrow=show_heading_arrows
                )
            if target is not None:
                self._plot_endpoint(
                    ax, target, is_source=False,
                    show_arrow=show_heading_arrows
                )
    
    def _plot_trajectory(
        self,
        ax: plt.Axes,
        position: np.ndarray
    ) -> None:
        """绘制GCS轨迹
        
        Args:
            ax: matplotlib坐标轴
            position: 位置数组 (2, N)
        """
        ax.plot(
            position[0, :],
            position[1, :],
            color=self.trajectory_color,
            linewidth=self.trajectory_linewidth,
            label='GCS Trajectory',
            zorder=3
        )
    
    def _plot_astar_path(
        self,
        ax: plt.Axes,
        path: np.ndarray
    ) -> None:
        """绘制A*路径
        
        Args:
            ax: matplotlib坐标轴
            path: 路径点 (M, 2) 或 (M, 3)
        """
        # 提取x, y坐标
        if path.shape[1] >= 2:
            x = path[:, 0]
            y = path[:, 1]
        else:
            return
        
        ax.plot(
            x, y,
            color=self.astar_color,
            linestyle=self.astar_linestyle,
            linewidth=self.astar_linewidth,
            label='A* Path',
            zorder=2
        )
    
    def _plot_endpoint(
        self,
        ax: plt.Axes,
        endpoint: EndpointState,
        is_source: bool = True,
        show_arrow: bool = True,
        arrow_length: float = 2.0
    ) -> None:
        """绘制端点
        
        Args:
            ax: matplotlib坐标轴
            endpoint: 端点状态
            is_source: 是否为起点
            show_arrow: 是否显示航向角箭头
            arrow_length: 箭头长度
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
        
        # 绘制端点标记
        ax.plot(
            endpoint.position[0],
            endpoint.position[1],
            color=color,
            marker=marker,
            markersize=markersize,
            label=label,
            zorder=4
        )
        
        # 绘制航向角箭头
        if show_arrow:
            ax.arrow(
                endpoint.position[0],
                endpoint.position[1],
                arrow_length * np.cos(endpoint.heading),
                arrow_length * np.sin(endpoint.heading),
                head_width=0.5,
                head_length=0.5,
                fc=color,
                ec=color,
                zorder=4
            )
    
    def compute_path_length(
        self,
        path: np.ndarray
    ) -> np.ndarray:
        """计算累积路径长度
        
        Args:
            path: 路径点 (N, 2) 或 (N, 3)
            
        Returns:
            np.ndarray: 累积路径长度 (N,)
        """
        if len(path) == 0:
            return np.array([])
        
        # 提取x, y坐标
        if path.ndim == 1:
            # 单个点
            return np.array([0.0])
        
        if path.shape[1] >= 2:
            xy = path[:, :2]
        else:
            return np.array([0.0])
        
        # 计算相邻点之间的距离
        diff = np.diff(xy, axis=0)
        distances = np.sqrt(np.sum(diff**2, axis=1))
        
        # 累积求和
        path_length = np.zeros(len(path))
        path_length[1:] = np.cumsum(distances)
        
        return path_length
    
    def compute_path_similarity(
        self,
        trajectory_position: np.ndarray,
        astar_path: np.ndarray
    ) -> dict:
        """计算路径相似度
        
        Args:
            trajectory_position: GCS轨迹位置 (2, N)
            astar_path: A*路径点 (M, 2) 或 (M, 3)
            
        Returns:
            dict: 相似度指标
        """
        # 计算路径长度
        traj_length = self._compute_total_length(trajectory_position.T)
        astar_length = self.compute_path_length(astar_path)[-1]
        
        # 计算长度差异
        length_diff = abs(traj_length - astar_length)
        length_ratio = traj_length / astar_length if astar_length > 0 else 0
        
        return {
            'trajectory_length': traj_length,
            'astar_length': astar_length,
            'length_difference': length_diff,
            'length_ratio': length_ratio
        }
    
    def _compute_total_length(
        self,
        path: np.ndarray
    ) -> float:
        """计算路径总长度
        
        Args:
            path: 路径点 (N, 2)
            
        Returns:
            float: 总长度
        """
        if len(path) == 0:
            return 0.0
        
        diff = np.diff(path, axis=0)
        distances = np.sqrt(np.sum(diff**2, axis=1))
        
        return np.sum(distances)
