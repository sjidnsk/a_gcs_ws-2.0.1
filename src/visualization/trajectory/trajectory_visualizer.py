"""
A*与GCS分层轨迹规划可视化模块

提供2D和阿克曼模式下的可视化功能，包括：
- 配置空间可视化
- 走廊可视化
- 障碍物凸分解可视化
- IRIS区域可视化
- GCS轨迹可视化
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from typing import List, Tuple, Optional, Any
from ..core.output_manager import VisualizationOutputManager


# === 数值容差常量 ===
# 用于数值计算中的容差判断

NUMERICAL_TOLERANCE: float = 1e-10  # 通用数值计算容差，用于避免除零
POLYTOPE_TOLERANCE: float = 1e-9  # 多胞体转换容差

# 设置全局字体为新罗马字体并放大
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 14  # 基础字体大小
plt.rcParams['axes.titlesize'] = 16  # 标题字体大小
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 12  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 12  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12  # 图例字体大小


class TrajectoryVisualizer:
    """A*与GCS分层轨迹规划可视化器"""
    
    def __init__(
        self,
        c_space,
        config,
        output_manager: Optional[VisualizationOutputManager] = None
    ):
        """
        初始化可视化器
        
        Args:
            c_space: 配置空间对象
            config: CorridorDecompositionConfig配置对象
            output_manager: 输出管理器，如果为None则使用单例实例
        """
        self.c_space = c_space
        self.config = config
        self.output_manager = (
            output_manager or VisualizationOutputManager.get_instance()
        )
    
    def visualize(self, result, original_path: List[Tuple[float, float, float]]):
        """生成可视化 - 支持2D和阿克曼模式"""
        # 优先检查阿克曼模式
        if result.used_gcs and hasattr(result, 'gcs_mode') and result.gcs_mode == 'ackermann':
            self._visualize_ackermann_gcs(result, original_path)
            return

        # 2D可视化（原有逻辑）
        self._visualize_2d(result, original_path)
    
    def _visualize_2d(self, result, original_path: List[Tuple[float, float, float]]):
        """2D可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        extent = [
            self.c_space.origin[0],
            self.c_space.origin[0] + self.c_space.width * self.c_space.resolution,
            self.c_space.origin[1],
            self.c_space.origin[1] + self.c_space.height * self.c_space.resolution
        ]
        
        # 子图1: 原始配置空间
        self._plot_cspace(axes[0, 0], extent, original_path, 'Original C-Space with A* Path')
        
        # 子图2: 走廊
        self._plot_corridor(axes[0, 1], result, extent, f'Corridor (Area: {result.corridor_area:.1f}m²)')
        
        # 子图3: 障碍物凸分解结果
        self._plot_obstacles(axes[1, 0], result, extent)
        
        # 子图4: IRIS凸区域结果
        self._plot_regions(axes[1, 1], result, extent)
        
        plt.tight_layout()
        
        if self.config.save_visualization:
            iris_mode = result.iris_mode_used if result.used_iris else 'traditional'
            # 添加GCS模式标识
            gcs_mode = '_gcs' if result.used_gcs else ''
            output_file = self.output_manager.generate_output_path(
                filename=f'corridor_decomposition_{iris_mode}_2d{gcs_mode}.png',
                dimension='2d'
            )
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"2D可视化保存至: {output_file}")
        
        plt.show()
        plt.close()
    
    def _visualize_3d_trajectory_standalone(self, result, original_path: List[Tuple[float, float, float]]):
        """独立的3D轨迹可视化（美化版）"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 设置背景色
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # 绘制原始路径（A*路径）
        if original_path:
            path_array = np.array(original_path)
            ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                   color='#2ecc71', linewidth=2.5, label='A* Path',
                   linestyle='--', alpha=0.8, zorder=10)
        
        # 绘制GCS轨迹（纯色）
        if result.gcs_waypoints is not None:
            waypoints = result.gcs_waypoints
            ax.plot(waypoints[0, :], waypoints[1, :], waypoints[2, :],
                   color='#e74c3c', linewidth=3, label='GCS Trajectory',
                   alpha=0.9, zorder=11)
        
        # 绘制起点和终点（更醒目）
        if original_path:
            # 起点
            ax.scatter([original_path[0][0]], [original_path[0][1]], [original_path[0][2]],
                      c='#27ae60', s=300, marker='s', label='Start',
                      edgecolors='black', linewidths=2, zorder=15)
            # 终点
            ax.scatter([original_path[-1][0]], [original_path[-1][1]], [original_path[-1][2]],
                      c='#e74c3c', s=400, marker='*', label='Goal',
                      edgecolors='black', linewidths=2, zorder=15)
        
        # 设置标签和标题
        ax.set_xlabel('X (m)', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_ylabel('Y (m)', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_zlabel('θ (rad)', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_title('3D Trajectory Visualization (x, y, θ)',
                    fontsize=20, fontweight='bold', pad=20)
        
        # 设置图例
        ax.legend(loc='upper left', fontsize=14, framealpha=0.95,
                 edgecolor='black', fancybox=True, shadow=True)
        
        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 设置视角
        ax.view_init(elev=25, azim=45)
        
        # 调整坐标轴刻度
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        
        if self.config.save_visualization:
            iris_mode = result.iris_mode_used if result.used_iris else 'traditional'
            # 添加GCS模式标识（独立的3D轨迹可视化）
            gcs_mode = '_gcs_3d' if result.used_gcs else ''
            output_file = self.output_manager.generate_output_path(
                filename=f'corridor_decomposition_{iris_mode}_3d_trajectory{gcs_mode}.png',
                dimension='3d'
            )
            plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
            print(f"3D轨迹可视化保存至: {output_file}")
        
        plt.show()
        plt.close()
    
    # ==================== 绘图辅助方法 ====================
    
    def _plot_cspace(self, ax, extent, path, title):
        """绘制配置空间"""
        ax.imshow(self.c_space.obstacle_map, cmap='RdYlGn_r',
                origin='lower', extent=extent, alpha=0.7)
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='A* Path')
            ax.scatter([path_x[0]], [path_y[0]], c='#27ae60', s=100, marker='o', zorder=5, label='Start')
            ax.scatter([path_x[-1]], [path_y[-1]], c='#e74c3c', s=100, marker='*', zorder=5, label='Goal')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_corridor(self, ax, result, extent, title):
        """绘制走廊"""
        ax.imshow(result.corridor_result.adjusted_c_space_2d,
                cmap='RdYlGn_r', origin='lower', extent=extent, alpha=0.7)
        if result.corridor_result.smoothed_path:
            path_x = [p[0] for p in result.corridor_result.smoothed_path]
            path_y = [p[1] for p in result.corridor_result.smoothed_path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Smoothed Path')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_obstacles(self, ax, result, extent):
        """绘制障碍物凸分解"""
        ax.imshow(result.corridor_result.adjusted_c_space_2d,
                cmap='gray', origin='lower', extent=extent, alpha=0.3)
        
        title = 'Obstacle Convex Decomposition'
        
        if result.corridor_result.smoothed_path:
            path_x = [p[0] for p in result.corridor_result.smoothed_path]
            path_y = [p[1] for p in result.corridor_result.smoothed_path]
            ax.plot(path_x, path_y, 'g-', linewidth=2, label='Path')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_regions(self, ax, result, extent):
        """绘制凸区域"""
        ax.imshow(result.corridor_result.adjusted_c_space_2d,
                cmap='gray', origin='lower', extent=extent, alpha=0.3)
        
        # 确定要绘制的区域
        regions_to_draw = []
        seed_points = []
        
        if result.used_iris:
            iris_result = result.iris_np_result or result.iris_np2_zo_result
            if iris_result:
                for r in iris_result.regions:
                    vertices = r.get_vertices_ordered() if hasattr(r, 'get_vertices_ordered') else r.vertices
                    regions_to_draw.append((vertices, 'blue'))
                    if hasattr(r, 'seed_point'):
                        seed_points.append(r.seed_point)
            title = f'IRIS Regions ({result.num_obstacles}) - {result.iris_mode_used}'
        else:
            regions_to_draw = [(obs, 'darkred') for obs in result.convex_obstacles]
            title = f'Traditional Decomposition ({result.num_obstacles})'
        
        # 绘制凸区域
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(regions_to_draw), 1)))
        for i, (vertices, edge_color) in enumerate(regions_to_draw):
            if len(vertices) >= 3:
                polygon = MplPolygon(vertices, closed=True,
                                    facecolor=colors[i % len(colors)], edgecolor=edge_color,
                                    alpha=0.5, linewidth=1.5)
                ax.add_patch(polygon)
                
                if hasattr(vertices, 'shape') and len(vertices) > 0:
                    centroid = np.mean(vertices, axis=0) if isinstance(vertices, np.ndarray) else (
                        sum(v[0] for v in vertices) / len(vertices),
                        sum(v[1] for v in vertices) / len(vertices)
                    )
                    ax.text(centroid[0], centroid[1], f'R{i+1}', 
                            fontsize=12, ha='center', va='center',
                            color='darkblue', fontweight='bold')
        
        # 绘制种子点
        if seed_points:
            seed_points = np.array(seed_points)
            ax.scatter(seed_points[:, 0], seed_points[:, 1], 
                      c='red', s=50, marker='*', zorder=10, label='Seed Points')
        
        # 绘制路径
        if result.corridor_result.smoothed_path:
            path_x = [p[0] for p in result.corridor_result.smoothed_path]
            path_y = [p[1] for p in result.corridor_result.smoothed_path]
            ax.plot(path_x, path_y, 'g-', linewidth=2, label='Path')
        
        # 绘制GCS轨迹
        if result.used_gcs and result.gcs_waypoints is not None:
            ax.plot(result.gcs_waypoints[0], result.gcs_waypoints[1], 
                   'm-', linewidth=2, label='GCS Trajectory')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_2d_topview(self, ax, result, original_path: List[Tuple[float, float, float]], extent):
        """绘制2D俯视图（带theta范围标注）"""
        # 绘制障碍物地图
        ax.imshow(result.corridor_result.adjusted_c_space_2d,
                 cmap='gray', origin='lower', extent=extent, alpha=0.3)

        # 绘制原始路径
        if original_path:
            path_x = [p[0] for p in original_path]
            path_y = [p[1] for p in original_path]
            ax.plot(path_x, path_y, 'g-', linewidth=2, label='A* Path')
        
        # 绘制GCS轨迹
        if result.gcs_waypoints is not None:
            ax.plot(result.gcs_waypoints[0], result.gcs_waypoints[1],
                   'r-', linewidth=2, label='GCS Trajectory')
        
        # 绘制起点和终点
        if original_path:
            ax.scatter([original_path[0][0]], [original_path[0][1]],
                      c='#27ae60', s=100, marker='s', label='Start', zorder=10)
            ax.scatter([original_path[-1][0]], [original_path[-1][1]],
                      c='#e74c3c', s=100, marker='*', label='Goal', zorder=10)
        
        ax.set_title('2D Top View (3D Mode)', fontsize=16, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_theta_profile(self, ax, result, original_path: List[Tuple[float, float, float]]):
        """绘制theta随路径变化图"""
        # 计算路径长度
        def compute_path_length(path):
            lengths = [0.0]
            for i in range(1, len(path)):
                dx = path[i][0] - path[i-1][0]
                dy = path[i][1] - path[i-1][1]
                lengths.append(lengths[-1] + np.sqrt(dx**2 + dy**2))
            return np.array(lengths)
        
        # 绘制原始路径的theta
        if original_path:
            path_lengths = compute_path_length(original_path)
            path_thetas = [p[2] for p in original_path]
            ax.plot(path_lengths, path_thetas, 'g-',
                   linewidth=2, label='A* Path (θ profile)', marker='o', markersize=4)

        # 绘制GCS轨迹的theta
        if result.gcs_waypoints is not None:
            waypoints = result.gcs_waypoints
            # 检查是否有theta维度（3D或4D模式）
            if waypoints.shape[0] >= 3:
                # 计算GCS轨迹的路径长度
                gcs_lengths = [0.0]
                for i in range(1, waypoints.shape[1]):
                    dx = waypoints[0, i] - waypoints[0, i-1]
                    dy = waypoints[1, i] - waypoints[1, i-1]
                    gcs_lengths.append(gcs_lengths[-1] + np.sqrt(dx**2 + dy**2))

                ax.plot(gcs_lengths, waypoints[2, :], 'r-',
                       linewidth=2, label='GCS Trajectory (θ profile)')

        ax.set_title('θ Profile Along Path', fontsize=16, fontweight='bold')
        ax.set_xlabel('Path Length (m)')
        ax.set_ylabel('θ (rad)')
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-np.pi, np.pi])
    
    # ==================== 4D可视化方法 ====================
    