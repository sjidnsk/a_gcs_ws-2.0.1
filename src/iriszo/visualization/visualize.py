"""
自定义IrisZo算法可视化模块

提供2D和3D可视化功能,绘制障碍物、凸区域、种子点和路径。

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    MplPolygon = None

from ..core.iriszo_region_data import IrisZoResult


def visualize_iriszo_result(
    result: IrisZoResult,
    obstacle_map: np.ndarray,
    resolution: float,
    origin: Tuple[float, float] = (0.0, 0.0),
    path: Optional[List[Tuple[float, float, float]]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    figsize: Tuple[float, float] = (12, 10)
):
    """
    可视化IrisZo结果

    Args:
        result: IrisZo结果对象
        obstacle_map: 障碍物地图
        resolution: 地图分辨率
        origin: 地图原点
        path: 路径点列表(可选)
        save_path: 保存路径(可选)
        show_plot: 是否显示图形
        figsize: 图形尺寸

    Example:
        >>> visualize_iriszo_result(
        ...     result, obstacle_map, 0.05,
        ...     path=path, save_path="result.png"
        ... )
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("Matplotlib不可用,无法可视化")
        return

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 计算地图范围
    height, width = obstacle_map.shape
    extent = [
        origin[0],
        origin[0] + width * resolution,
        origin[1],
        origin[1] + height * resolution
    ]

    # 1. 绘制障碍物地图
    ax.imshow(
        obstacle_map,
        cmap='gray',
        origin='lower',
        extent=extent,
        alpha=0.5,
        aspect='auto'
    )

    # 2. 绘制凸区域
    if len(result.regions) > 0:
        colors = plt.cm.Set3(np.linspace(0, 1, len(result.regions)))

        for i, region in enumerate(result.regions):
            # Drake的HPolyhedron没有GetVertices()方法
            # 我们需要通过采样来绘制多面体
            try:
                # 方法1: 尝试使用MaybeGetFeasiblePoint获取点
                # 方法2: 使用ChebyshevCenter和采样
                
                # 获取多面体的约束
                A = region.polyhedron.A()
                b = region.polyhedron.b()
                
                # 使用线性规划找到顶点(简化方法)
                # 对于2D,我们可以通过求解边界点来绘制
                
                # 获取中心
                center = region.centroid
                
                # 在多个方向上采样边界点
                num_directions = 72  # 每5度一个方向(增加密度)
                boundary_points = []
                
                for angle in np.linspace(0, 2*np.pi, num_directions, endpoint=False):
                    direction = np.array([np.cos(angle), np.sin(angle)])
                    
                    # 沿方向搜索边界
                    # 使用二分搜索
                    t_min = 0.0
                    t_max = 5.0  # 最大搜索距离
                    
                    for _ in range(20):  # 二分搜索步数
                        t_mid = (t_min + t_max) / 2
                        point = center + t_mid * direction
                        
                        # 检查点是否满足所有约束
                        if np.all(A @ point <= b):
                            t_min = t_mid
                        else:
                            t_max = t_mid
                    
                    boundary_point = center + t_min * direction
                    boundary_points.append(boundary_point)
                
                boundary_points = np.array(boundary_points)
                
                # 按角度排序
                angles = np.arctan2(
                    boundary_points[:, 1] - center[1],
                    boundary_points[:, 0] - center[0]
                )
                order = np.argsort(angles)
                boundary_points = boundary_points[order]
                
                if len(boundary_points) >= 3:
                    # 创建多边形
                    polygon = MplPolygon(
                        boundary_points,
                        closed=True,
                        facecolor=colors[i],
                        edgecolor='blue',
                        alpha=0.4,
                        linewidth=1  # 减小线条宽度,避免视觉误差
                    )
                    ax.add_patch(polygon)
                    
                    # 标注区域编号
                    ax.text(
                        center[0],
                        center[1],
                        f'{i+1}',
                        fontsize=10,
                        ha='center',
                        va='center',
                        fontweight='bold',
                        color='darkblue'
                    )
                    
                    # 绘制种子点
                    ax.plot(
                        region.seed_point[0],
                        region.seed_point[1],
                        'r*',
                        markersize=10,
                        label='Seed' if i == 0 else ''
                    )
                    
            except Exception as e:
                warnings.warn(f"绘制区域{i+1}失败: {e}")
                # 如果失败,至少绘制种子点
                ax.plot(
                    region.seed_point[0],
                    region.seed_point[1],
                    'r*',
                    markersize=10,
                    label='Seed' if i == 0 else ''
                )

    # 3. 绘制路径
    if path and len(path) > 0:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]

        ax.plot(
            path_x,
            path_y,
            'g-',
            linewidth=2,
            label='Path',
            zorder=3
        )

        # 标记起点和终点
        ax.scatter(
            path_x[0],
            path_y[0],
            c='green',
            s=100,
            marker='o',
            label='Start',
            zorder=5,
            edgecolors='black',
            linewidths=2
        )
        ax.scatter(
            path_x[-1],
            path_y[-1],
            c='red',
            s=100,
            marker='*',
            label='Goal',
            zorder=5,
            edgecolors='black',
            linewidths=2
        )

    # 4. 设置图形属性
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)

    # 设置标题
    title = f'IrisZo Convex Regions\n'
    title += f'Regions: {result.num_regions}, '
    title += f'Area: {result.total_area:.2f} m², '
    title += f'Coverage: {result.coverage_ratio:.1%}'
    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)

    plt.tight_layout()

    # 5. 保存图形
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")

    # 6. 显示图形
    if show_plot:
        plt.show()

    plt.close(fig)


def visualize_iriszo_result_detailed(
    result: IrisZoResult,
    obstacle_map: np.ndarray,
    resolution: float,
    origin: Tuple[float, float] = (0.0, 0.0),
    path: Optional[List[Tuple[float, float, float]]] = None,
    save_path: Optional[str] = None
):
    """
    详细可视化IrisZo结果(包含统计信息)

    Args:
        result: IrisZo结果对象
        obstacle_map: 障碍物地图
        resolution: 地图分辨率
        origin: 地图原点
        path: 路径点列表(可选)
        save_path: 保存路径(可选)
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("Matplotlib不可用,无法可视化")
        return

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # 左图: 区域可视化
    ax1 = axes[0]

    # 绘制障碍物
    height, width = obstacle_map.shape
    extent = [
        origin[0],
        origin[0] + width * resolution,
        origin[1],
        origin[1] + height * resolution
    ]

    ax1.imshow(
        obstacle_map,
        cmap='gray',
        origin='lower',
        extent=extent,
        alpha=0.5
    )

    # 绘制区域
    if len(result.regions) > 0:
        colors = plt.cm.Set3(np.linspace(0, 1, len(result.regions)))

        for i, region in enumerate(result.regions):
            vertices = region.get_vertices_ordered()
            if len(vertices) >= 3:
                polygon = MplPolygon(
                    vertices,
                    closed=True,
                    facecolor=colors[i],
                    edgecolor='blue',
                    alpha=0.4,
                    linewidth=2
                )
                ax1.add_patch(polygon)

                ax1.plot(
                    region.seed_point[0],
                    region.seed_point[1],
                    'r*',
                    markersize=8
                )

    # 绘制路径
    if path and len(path) > 0:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax1.plot(path_x, path_y, 'g-', linewidth=2)

    ax1.set_xlim(extent[0], extent[1])
    ax1.set_ylim(extent[2], extent[3])
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Convex Regions', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')

    # 右图: 统计信息
    ax2 = axes[1]
    ax2.axis('off')

    # 显示统计信息
    stats_text = (
        f"IrisZo Generation Results\n"
        f"{'='*40}\n\n"
        f"Regions:\n"
        f"  - Number of regions: {result.num_regions}\n"
        f"  - Total area: {result.total_area:.6f} m²\n"
        f"  - Coverage ratio: {result.coverage_ratio:.2%}\n\n"
        f"Performance:\n"
        f"  - Algorithm time: {result.iris_time:.3f} s\n"
        f"  - Postprocess time: {result.postprocess_time:.3f} s\n"
        f"  - Total time: {result.total_time:.3f} s\n\n"
        f"Cache:\n"
        f"  - Hit rate: {result.cache_hit_rate:.2%}\n\n"
    )

    if result.config:
        stats_text += (
            f"Configuration:\n"
            f"  - Epsilon: {result.config.epsilon}\n"
            f"  - Delta: {result.config.delta}\n"
            f"  - Iterations: {result.config.iteration_limit}\n"
            f"  - Bisection steps: {result.config.bisection_steps}\n"
        )

    ax2.text(
        0.1, 0.9,
        stats_text,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"详细可视化结果已保存至: {save_path}")

    plt.show()
    plt.close(fig)


def visualize_region_only(
    region,
    obstacle_map: np.ndarray,
    resolution: float,
    origin: Tuple[float, float] = (0.0, 0.0),
    save_path: Optional[str] = None
):
    """
    可视化单个区域

    Args:
        region: IrisZoRegion对象
        obstacle_map: 障碍物地图
        resolution: 地图分辨率
        origin: 地图原点
        save_path: 保存路径(可选)
    """
    if not MATPLOTLIB_AVAILABLE:
        warnings.warn("Matplotlib不可用,无法可视化")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制障碍物
    height, width = obstacle_map.shape
    extent = [
        origin[0],
        origin[0] + width * resolution,
        origin[1],
        origin[1] + height * resolution
    ]

    ax.imshow(
        obstacle_map,
        cmap='gray',
        origin='lower',
        extent=extent,
        alpha=0.5
    )

    # 绘制区域
    vertices = region.get_vertices_ordered()
    if len(vertices) >= 3:
        polygon = MplPolygon(
            vertices,
            closed=True,
            facecolor='lightblue',
            edgecolor='blue',
            alpha=0.5,
            linewidth=2
        )
        ax.add_patch(polygon)

        # 绘制种子点
        ax.plot(
            region.seed_point[0],
            region.seed_point[1],
            'r*',
            markersize=15,
            label='Seed Point'
        )

        # 绘制中心
        ax.plot(
            region.centroid[0],
            region.centroid[1],
            'go',
            markersize=10,
            label='Centroid'
        )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()

    title = f'Single Region\nArea: {region.area:.6f} m²'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"区域可视化已保存至: {save_path}")

    plt.show()
    plt.close(fig)
