"""
A*与GCS分层轨迹规划可视化模块

提供2D、3D和4D模式下的可视化功能，包括：
- 配置空间可视化
- 走廊可视化
- 障碍物凸分解可视化
- IRIS区域可视化
- GCS轨迹可视化
- 单位向量轨迹可视化
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
        """生成可视化 - 支持2D、3D和4D模式"""
        # 检查是否使用4D GCS（单位向量）
        use_4d = result.used_gcs and hasattr(result, 'gcs_regions_4d') and result.gcs_regions_4d is not None

        # 检查是否使用3D GCS（需要同时检查regions和waypoints的维度）
        use_3d = False
        if result.used_gcs and hasattr(result, 'gcs_regions_3d') and result.gcs_regions_3d is not None:
            # 检查waypoints的维度是否为3D（x, y, theta）
            if result.gcs_waypoints is not None and result.gcs_waypoints.shape[0] == 3:
                use_3d = True

        if use_4d:
            # 4D可视化（单位向量模式）
            self._visualize_4d_gcs(result, original_path)
        elif use_3d:
            # 3D可视化
            self._visualize_3d_gcs(result, original_path)
        else:
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
    
    def _visualize_3d_gcs(self, result, original_path: List[Tuple[float, float, float]]):
        """3D GCS可视化"""
        fig = plt.figure(figsize=(18, 12))
        
        # 创建子图布局
        # 左侧：3D视图（大图）
        ax_3d = fig.add_subplot(121, projection='3d')
        # 右侧上：2D俯视图
        ax_2d = fig.add_subplot(222)
        # 右侧下：theta随路径变化图
        ax_theta = fig.add_subplot(224)
        
        # 绘制3D视图
        self._plot_3d_regions(ax_3d, result, original_path)
        
        # 绘制2D俯视图
        extent = [
            self.c_space.origin[0],
            self.c_space.origin[0] + self.c_space.width * self.c_space.resolution,
            self.c_space.origin[1],
            self.c_space.origin[1] + self.c_space.height * self.c_space.resolution
        ]
        self._plot_2d_topview(ax_2d, result, original_path, extent)
        
        # 绘制theta变化图
        self._plot_theta_profile(ax_theta, result, original_path)
        
        plt.tight_layout()
        
        if self.config.save_visualization:
            iris_mode = result.iris_mode_used if result.used_iris else 'traditional'
            # 添加GCS模式标识（3D模式）
            gcs_mode = '_gcs_3d' if result.used_gcs else ''
            output_file = self.output_manager.generate_output_path(
                filename=f'corridor_decomposition_{iris_mode}_3d{gcs_mode}.png',
                dimension='3d'
            )
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"3D可视化保存至: {output_file}")
        
        plt.show()
        plt.close()
    
    def _visualize_4d_gcs(self, result, original_path: List[Tuple[float, float, float]]):
        """4D GCS可视化（单位向量模式）"""
        # 第一张图：独立的3D (x, y, theta)可视化
        self._visualize_3d_trajectory_standalone(result, original_path)
        
        # 第二张图：其他子图
        fig = plt.figure(figsize=(20, 16))
        
        # 创建子图布局
        # 第一行：2D俯视图、单位向量轨迹图、theta随路径变化图
        ax_2d = fig.add_subplot(331)
        ax_uv = fig.add_subplot(332)
        ax_theta = fig.add_subplot(333)
        # 第二行：u随时间变化、w随时间变化、单位圆约束验证
        ax_u = fig.add_subplot(334)
        ax_w = fig.add_subplot(335)
        ax_circle = fig.add_subplot(336)
        # 第三行：速度可视化（线速度、角速度、速度大小）
        ax_v_linear = fig.add_subplot(337)
        ax_v_angular = fig.add_subplot(338)
        ax_v_magnitude = fig.add_subplot(339)
        
        # 绘制2D俯视图
        extent = [
            self.c_space.origin[0],
            self.c_space.origin[0] + self.c_space.width * self.c_space.resolution,
            self.c_space.origin[1],
            self.c_space.origin[1] + self.c_space.height * self.c_space.resolution
        ]
        self._plot_2d_topview_4d(ax_2d, result, original_path, extent)
        
        # 绘制单位向量轨迹
        self._plot_unit_vector_trajectory(ax_uv, result)
        
        # 绘制theta变化图
        self._plot_theta_profile_4d(ax_theta, result, original_path)
        
        # 绘制u和w随时间变化
        self._plot_uw_time_profile(ax_u, ax_w, result)
        
        # 绘制单位圆约束验证
        self._plot_unit_circle_validation(ax_circle, result)
        
        # 绘制速度可视化
        self._plot_velocity_profile(ax_v_linear, ax_v_angular, ax_v_magnitude, result)
        
        plt.tight_layout()
        
        if self.config.save_visualization:
            iris_mode = result.iris_mode_used if result.used_iris else 'traditional'
            # 添加GCS模式标识（4D单位向量模式）
            gcs_mode = '_gcs_4d_unit_vector' if result.used_gcs else ''
            output_file = self.output_manager.generate_output_path(
                filename=f'corridor_decomposition_{iris_mode}_4d{gcs_mode}.png',
                dimension='4d'
            )
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"4D可视化保存至: {output_file}")
        
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
    
    def _plot_3d_regions(self, ax, result, original_path: List[Tuple[float, float, float]]):
        """绘制3D凸区域和轨迹"""
        try:
            from pydrake.geometry.optimization import HPolyhedron, VPolytope
            
            # 获取3D区域
            regions_3d = result.gcs_regions_3d
            if not regions_3d:
                return
            
            # 颜色映射
            colors = plt.cm.Set3(np.linspace(0, 1, len(regions_3d)))
            
            # 绘制每个3D区域
            for i, region_3d in enumerate(regions_3d):
                try:
                    # 转换为HPolyhedron
                    hpoly = region_3d.to_hpolyhedron()
                    
                    # 转换为VPolytope获取顶点
                    vpoly = VPolytope(hpoly, tol=1e-9)
                    vertices = vpoly.vertices()  # shape: (3, N)
                    
                    if vertices.shape[1] < 4:
                        continue
                    
                    # 计算凸包
                    hull = ConvexHull(vertices.T)
                    
                    # 绘制面片
                    faces = []
                    for simplex in hull.simplices:
                        face = vertices[:, simplex].T
                        faces.append(face)
                    
                    poly3d = Poly3DCollection(faces, alpha=0.3, 
                                             facecolor=colors[i], 
                                             edgecolor='darkblue',
                                             linewidth=0.5)
                    ax.add_collection3d(poly3d)
                    
                    # 标注区域编号
                    centroid = np.mean(vertices, axis=1)
                    ax.text(centroid[0], centroid[1], centroid[2], f'R{i+1}',
                           fontsize=12, ha='center', va='center', color='darkblue')
                    
                except Exception as e:
                    warnings.warn(f"绘制区域{i}失败: {e}")
                    continue
            
            # 绘制原始路径
            if original_path:
                path_array = np.array(original_path)
                ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2],
                       'g-', linewidth=2, label='A* Path', zorder=10)
            
            # 绘制GCS轨迹
            if result.gcs_waypoints is not None:
                waypoints = result.gcs_waypoints
                ax.plot(waypoints[0, :], waypoints[1, :], waypoints[2, :],
                       'r-', linewidth=3, label='GCS Trajectory', zorder=11)
            
            # 绘制起点和终点
            if original_path:
                ax.scatter([original_path[0][0]], [original_path[0][1]], [original_path[0][2]],
                          c='#27ae60', s=150, marker='s', label='Start', zorder=12)
                ax.scatter([original_path[-1][0]], [original_path[-1][1]], [original_path[-1][2]],
                          c='#e74c3c', s=150, marker='*', label='Goal', zorder=12)
            
            # 设置标签和标题
            ax.set_xlabel('X (m)', fontsize=14)
            ax.set_ylabel('Y (m)', fontsize=14)
            ax.set_zlabel('θ (rad)', fontsize=14)
            ax.set_title(f'3D GCS Regions ({len(regions_3d)} regions)', 
                        fontsize=16, fontweight='bold')
            ax.legend(loc='upper left')
            
            # 设置视角
            ax.view_init(elev=20, azim=45)
            
        except Exception as e:
            warnings.warn(f"3D区域绘制失败: {e}")
    
    def _plot_2d_topview(self, ax, result, original_path: List[Tuple[float, float, float]], extent):
        """绘制2D俯视图（带theta范围标注）"""
        # 绘制障碍物地图
        ax.imshow(result.corridor_result.adjusted_c_space_2d,
                 cmap='gray', origin='lower', extent=extent, alpha=0.3)
        
        # 获取3D区域
        regions_3d = result.gcs_regions_3d if hasattr(result, 'gcs_regions_3d') else None
        
        if regions_3d:
            colors = plt.cm.Set3(np.linspace(0, 1, len(regions_3d)))
            
            for i, region_3d in enumerate(regions_3d):
                # 绘制2D投影
                vertices_2d = region_3d.vertices_2d
                if len(vertices_2d) >= 3:
                    polygon = MplPolygon(vertices_2d, closed=True,
                                        facecolor=colors[i], edgecolor='darkblue',
                                        alpha=0.4, linewidth=1.5)
                    ax.add_patch(polygon)
                    
                    # 标注区域编号
                    centroid = region_3d.centroid_2d
                    ax.text(centroid[0], centroid[1], f'R{i+1}',
                           fontsize=12, ha='center', va='center', color='darkblue', fontweight='bold')
        
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
    
    def _plot_2d_topview_4d(self, ax, result, original_path: List[Tuple[float, float, float]], extent):
        """绘制2D俯视图（4D模式）"""
        # 绘制障碍物地图
        ax.imshow(result.corridor_result.adjusted_c_space_2d,
                 cmap='gray', origin='lower', extent=extent, alpha=0.3)
        
        # 获取4D区域
        regions_4d = result.gcs_regions_4d if hasattr(result, 'gcs_regions_4d') else None
        
        if regions_4d:
            colors = plt.cm.Set3(np.linspace(0, 1, len(regions_4d)))
            
            for i, region_4d in enumerate(regions_4d):
                # 绘制2D投影
                vertices_2d = region_4d.vertices_2d
                if len(vertices_2d) >= 3:
                    polygon = MplPolygon(vertices_2d, closed=True,
                                        facecolor=colors[i], edgecolor='darkblue',
                                        alpha=0.4, linewidth=1.5)
                    ax.add_patch(polygon)
        
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
        
        ax.set_title('2D Top View (4D Mode)', fontsize=16, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    def _plot_unit_vector_trajectory(self, ax, result):
        """绘制单位向量轨迹（简化版）"""
        if result.gcs_waypoints_4d is not None:
            waypoints_4d = result.gcs_waypoints_4d
            u = waypoints_4d[2, :]
            w = waypoints_4d[3, :]

            # 绘制单位圆
            theta_circle = np.linspace(0, 2*np.pi, 200)
            ax.plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', linewidth=1.5, label='Unit Circle', alpha=0.4)

            # 绘制GCS优化轨迹（实际轨迹）- 单色蓝色
            ax.plot(u, w, 'b-', linewidth=2.5, label='GCS Trajectory', zorder=3)

            # 绘制起点和终点
            ax.scatter([u[0]], [w[0]], c='#27ae60', s=200, marker='o', label='Start', zorder=4,
                      edgecolors='black', linewidths=2)
            ax.scatter([u[-1]], [w[-1]], c='#e74c3c', s=250, marker='*', label='Goal', zorder=4,
                      edgecolors='black', linewidths=2)

            ax.set_title('Unit Vector Trajectory (u, w)', fontsize=16, fontweight='bold')
            ax.set_xlabel('u = cos(θ)', fontsize=12, fontweight='bold')
            ax.set_ylabel('w = sin(θ)', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_xlim([-1.5, 1.5])
            ax.set_ylim([-1.5, 1.5])

            # 添加十字参考线
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    
    def _plot_theta_profile_4d(self, ax, result, original_path: List[Tuple[float, float, float]]):
        """绘制theta随路径变化图（4D模式）"""
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
            # 检查是否有theta维度（4D模式）
            if waypoints.shape[0] >= 3:
                # 计算GCS轨迹的路径长度
                gcs_lengths = [0.0]
                for i in range(1, waypoints.shape[1]):
                    dx = waypoints[0, i] - waypoints[0, i-1]
                    dy = waypoints[1, i] - waypoints[1, i-1]
                    gcs_lengths.append(gcs_lengths[-1] + np.sqrt(dx**2 + dy**2))

                ax.plot(gcs_lengths, waypoints[2, :], 'r-',
                       linewidth=2, label='GCS Trajectory (θ profile)')

        ax.set_title('θ Profile Along Path (4D Mode)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Path Length (m)')
        ax.set_ylabel('θ (rad)')
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-np.pi, np.pi])
    
    def _plot_uw_time_profile(self, ax_u, ax_w, result):
        """绘制u和w随时间变化"""
        if result.gcs_waypoints_4d is not None:
            waypoints_4d = result.gcs_waypoints_4d
            num_points = waypoints_4d.shape[1]
            time_steps = np.arange(num_points)
            
            # 绘制u随时间变化
            u = waypoints_4d[2, :]
            ax_u.plot(time_steps, u, 'b-', linewidth=2, label='u = cos(θ)')
            ax_u.axhline(y=1, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax_u.axhline(y=-1, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax_u.set_title('u = cos(θ) vs Time', fontsize=16, fontweight='bold')
            ax_u.set_xlabel('Time Step')
            ax_u.set_ylabel('u')
            ax_u.legend(loc='upper right', fontsize=12, framealpha=0.9)
            ax_u.grid(True, alpha=0.3)
            ax_u.set_ylim([-1.5, 1.5])
            
            # 绘制w随时间变化
            w = waypoints_4d[3, :]
            ax_w.plot(time_steps, w, 'r-', linewidth=2, label='w = sin(θ)')
            ax_w.axhline(y=1, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax_w.axhline(y=-1, color='k', linestyle='--', linewidth=1, alpha=0.5)
            ax_w.set_title('w = sin(θ) vs Time', fontsize=16, fontweight='bold')
            ax_w.set_xlabel('Time Step')
            ax_w.set_ylabel('w')
            ax_w.legend(loc='upper right', fontsize=12, framealpha=0.9)
            ax_w.grid(True, alpha=0.3)
            ax_w.set_ylim([-1.5, 1.5])
    
    def _plot_unit_circle_validation(self, ax, result):
        """绘制单位圆约束验证"""
        if result.gcs_waypoints_4d is not None:
            waypoints_4d = result.gcs_waypoints_4d
            u = waypoints_4d[2, :]
            w = waypoints_4d[3, :]
            
            # 计算u² + w²
            norm_squared = u**2 + w**2
            
            # 绘制
            time_steps = np.arange(len(norm_squared))
            ax.plot(time_steps, norm_squared, 'g-', linewidth=2, label='u² + w²')
            ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Constraint: u² + w² ≤ 1')
            ax.axhline(y=0.99, color='orange', linestyle=':', linewidth=2, alpha=1.0, label='Relaxed: 0.99')
            
            # 计算统计信息
            max_violation = np.max(norm_squared) - 1.0
            mean_norm = np.mean(norm_squared)
            min_norm = np.min(norm_squared)
            max_norm = np.max(norm_squared)
            
            # 动态调整y轴范围，确保所有数据可见
            y_min = max(0.0, min_norm - 0.05)
            y_max = max_norm + 0.05
            
            # 确保关键参考线在范围内
            y_min = min(y_min, 0.95)
            y_max = max(y_max, 1.05)
            
            ax.set_title(f'Unit Circle Constraint Validation\nMax Violation: {max_violation:.6f}, Mean: {mean_norm:.6f}',
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('u² + w²')
            ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim([y_min, y_max])
            
            # 添加文本标注，显示范围信息
            ax.text(0.02, 0.98, f'Range: [{min_norm:.4f}, {max_norm:.4f}]',
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_velocity_profile(self, ax_v_linear, ax_v_angular, ax_v_magnitude, result):
        """绘制速度可视化（使用EvalDerivative计算理论精确速度）"""
        # 优先使用trajectory对象计算速度（理论精确值）
        if result.gcs_trajectory is not None:
            trajectory = result.gcs_trajectory
            
            # 获取实际时间戳
            if hasattr(result, 'gcs_sample_times') and result.gcs_sample_times is not None:
                sample_times = result.gcs_sample_times
            else:
                sample_times = np.linspace(trajectory.start_time(), trajectory.end_time(), 1000)
            
            # 使用EvalDerivative计算速度（理论精确值）
            print("使用EvalDerivative计算速度...")
            velocities = np.array([trajectory.EvalDerivative(t, 1).flatten() for t in sample_times]).T
            
            vx = velocities[0, :]
            vy = velocities[1, :]
            v_speed = np.sqrt(vx**2 + vy**2)
            
            # 计算角速度（使用正确的曲率公式）
            # ω = (vx·ay - vy·ax) / (vx² + vy²)
            # 这是基于轨迹几何曲率的正确计算方法
            accelerations = np.array([trajectory.EvalDerivative(t, 2).flatten() for t in sample_times]).T
            ax_accel = accelerations[0, :]
            ay_accel = accelerations[1, :]

            omega = (vx * ay_accel - vy * ax_accel) / (vx**2 + vy**2 + 1e-10)
            
            # 检查边界速度（验证zero_deriv_boundary效果）
            start_velocity = trajectory.EvalDerivative(trajectory.start_time(), 1)
            end_velocity = trajectory.EvalDerivative(trajectory.end_time(), 1)
            start_speed = np.linalg.norm(start_velocity)
            end_speed = np.linalg.norm(end_velocity)
            
            # 检查边界角速度
            start_omega = omega[0]
            end_omega = omega[-1]
            
            print(f"\n=== 边界速度检查 ===")
            print(f"起点速度: [{float(start_velocity[0]):.6f}, {float(start_velocity[1]):.6f}] m/s")
            print(f"起点速度范数: {start_speed:.8f} m/s")
            print(f"起点角速度: {start_omega:.8f} rad/s")
            print(f"终点速度: [{float(end_velocity[0]):.6f}, {float(end_velocity[1]):.6f}] m/s")
            print(f"终点速度范数: {end_speed:.8f} m/s")
            print(f"终点角速度: {end_omega:.8f} rad/s")
            print(f"===================\n")
            
            # 使用实际时间作为x轴
            time_axis = sample_times
            
            # 绘制线速度分量
            ax_v_linear.plot(time_axis, vx, 'b-', linewidth=2, label='vx (X velocity)')
            ax_v_linear.plot(time_axis, vy, 'r-', linewidth=2, label='vy (Y velocity)')
            ax_v_linear.set_title('Linear Velocity Components (EvalDerivative)', fontsize=16, fontweight='bold')
            ax_v_linear.set_xlabel('Time (s)')
            ax_v_linear.set_ylabel('Velocity (m/s)')
            ax_v_linear.legend(loc='upper right', fontsize=12, framealpha=0.9)
            ax_v_linear.grid(True, alpha=0.3)
            
            # 绘制角速度
            ax_v_angular.plot(time_axis, omega, 'g-', linewidth=2, label='ω (Angular velocity)')
            ax_v_angular.set_title('Angular Velocity (EvalDerivative)', fontsize=16, fontweight='bold')
            ax_v_angular.set_xlabel('Time (s)')
            ax_v_angular.set_ylabel('ω (rad/s)')
            ax_v_angular.legend(loc='upper right', fontsize=12, framealpha=0.9)
            ax_v_angular.grid(True, alpha=0.3)
            
            # 绘制速度（标量，沿运动方向）
            ax_v_magnitude.plot(time_axis, v_speed, 'm-', linewidth=2, label='v (Speed)')
            ax_v_magnitude.set_title('Speed (EvalDerivative)', fontsize=16, fontweight='bold')
            ax_v_magnitude.set_xlabel('Time (s)')
            ax_v_magnitude.set_ylabel('Speed (m/s)')
            ax_v_magnitude.legend(loc='upper right', fontsize=12, framealpha=0.9)
            ax_v_magnitude.grid(True, alpha=0.3)
            
            # 添加统计信息
            max_speed = np.max(v_speed)
            min_speed = np.min(v_speed)
            mean_speed = np.mean(v_speed)
            max_omega = np.max(np.abs(omega))
            
            ax_v_magnitude.text(0.02, 0.98,
                               f'Max Speed: {max_speed:.3f} m/s\nMin Speed: {min_speed:.3f} m/s\nMean Speed: {mean_speed:.3f} m/s\nMax |ω|: {max_omega:.3f} rad/s\n\nStart Speed: {start_speed:.6f} m/s\nEnd Speed: {end_speed:.6f} m/s\nStart ω: {start_omega:.6f} rad/s\nEnd ω: {end_omega:.6f} rad/s',
                               transform=ax_v_magnitude.transAxes, fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 回退方案：使用数值差分（如果trajectory不可用）
        elif result.gcs_waypoints_4d is not None:
            print("警告：trajectory对象不可用，使用数值差分计算速度（精度较低）")
            waypoints_4d = result.gcs_waypoints_4d
            num_points = waypoints_4d.shape[1]
            
            # 获取实际时间戳
            if hasattr(result, 'gcs_sample_times') and result.gcs_sample_times is not None:
                sample_times = result.gcs_sample_times
            else:
                sample_times = np.arange(num_points)
            
            # 计算线速度（x, y方向的导数）
            vx = np.zeros(num_points)
            vy = np.zeros(num_points)
            for i in range(1, num_points):
                dt = sample_times[i] - sample_times[i-1] if len(sample_times) > i else 1.0
                vx[i] = (waypoints_4d[0, i] - waypoints_4d[0, i-1]) / dt
                vy[i] = (waypoints_4d[1, i] - waypoints_4d[1, i-1]) / dt
            
            # 计算角速度（使用正确的曲率公式）
            # ω = (vx·ay - vy·ax) / (vx² + vy²)
            omega = np.zeros(num_points)

            # 计算加速度（数值差分）
            ax = np.zeros(num_points)
            ay = np.zeros(num_points)
            for i in range(1, num_points-1):
                dt = sample_times[i] - sample_times[i-1] if len(sample_times) > i else 1.0
                dt_next = sample_times[i+1] - sample_times[i] if len(sample_times) > i+1 else 1.0
                # 中心差分计算加速度
                ax[i] = (waypoints_4d[0, i+1] - 2*waypoints_4d[0, i] + waypoints_4d[0, i-1]) / (dt * dt_next)
                ay[i] = (waypoints_4d[1, i+1] - 2*waypoints_4d[1, i] + waypoints_4d[1, i-1]) / (dt * dt_next)

            # 计算角速度
            v_squared = vx**2 + vy**2
            omega = (vx * ay - vy * ax) / (v_squared + 1e-10)

            # 计算速度大小
            v_speed = np.sqrt(vx**2 + vy**2)

            # 使用实际时间作为x轴
            time_axis = sample_times[:num_points] if len(sample_times) >= num_points else np.arange(num_points)

            # 绘制线速度分量
            ax_v_linear.plot(time_axis, vx, 'b-', linewidth=2, label='vx (X velocity)')
            ax_v_linear.plot(time_axis, vy, 'r-', linewidth=2, label='vy (Y velocity)')
            ax_v_linear.set_title('Linear Velocity Components (Numerical Difference)', fontsize=16, fontweight='bold')
            ax_v_linear.set_xlabel('Time (s)')
            ax_v_linear.set_ylabel('Velocity (m/s)')
            ax_v_linear.legend(loc='upper right', fontsize=12, framealpha=0.9)
            ax_v_linear.grid(True, alpha=0.3)

            # 绘制角速度
            ax_v_angular.plot(time_axis, omega, 'g-', linewidth=2, label='ω (Angular velocity)')
            ax_v_angular.set_title('Angular Velocity', fontsize=16, fontweight='bold')
            ax_v_angular.set_xlabel('Time (s)')
            ax_v_angular.set_ylabel('ω (rad/s)')
            ax_v_angular.legend(loc='upper right', fontsize=12, framealpha=0.9)
            ax_v_angular.grid(True, alpha=0.3)

            # 绘制速度（标量，沿运动方向）
            ax_v_magnitude.plot(time_axis, v_speed, 'm-', linewidth=2, label='v (Speed)')
            ax_v_magnitude.set_title('Speed (Numerical Difference)', fontsize=16, fontweight='bold')
            ax_v_magnitude.set_xlabel('Time (s)')
            ax_v_magnitude.set_ylabel('Speed (m/s)')
            ax_v_magnitude.legend(loc='upper right', fontsize=12, framealpha=0.9)
            ax_v_magnitude.grid(True, alpha=0.3)

            # 添加统计信息
            max_speed = np.max(v_speed)
            min_speed = np.min(v_speed)
            mean_speed = np.mean(v_speed)
            max_omega = np.max(np.abs(omega))

            ax_v_magnitude.text(0.02, 0.98,
                               f'Max Speed: {max_speed:.3f} m/s\nMin Speed: {min_speed:.3f} m/s\nMean Speed: {mean_speed:.3f} m/s\nMax |ω|: {max_omega:.3f} rad/s',
                               transform=ax_v_magnitude.transAxes, fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
