import matplotlib.pyplot as plt
import numpy as np
from pydrake.geometry.optimization import (Iris, IrisOptions, HPolyhedron, VPolytope)
from typing import Any, Dict, List, Optional
from gcs_pkg.scripts import BezierGCS, BezierTrajectory


def visualize_environment_with_bezier(
    result: Dict[str, Any],
    regions: List[HPolyhedron],
    trajectory: Optional[Any] = None,
    bezier_trajectory_obj: Optional[BezierTrajectory] = None, # 新增参数
    num_waypoints: int = 100,
    trajectory_color: str = 'blue',
    trajectory_linewidth: float = 2.0,
    region_colors: Optional[List] = None,
    show_region_labels: bool = True,
    show_bezier_controls: bool = True, # 新增参数
    bezier_control_color: str = 'orange', # 新增参数
    bezier_curve_color: str = 'purple', # 新增参数
    bezier_curve_linewidth: float = 1.5 # 新增参数
):
    """
    可视化生成的环境和多个IRIS区域，以及可选的轨迹和贝塞尔曲线及其控制点。
    """
    fig, ax = plt.subplots(figsize=(12, 12))

    # 绘制边界
    domain = result['domain']
    boundary = [(0, 0), (domain['width'], 0), (domain['width'], domain['height']), (0, domain['height'])]
    boundary.append(boundary[0])
    bx, by = zip(*boundary)
    ax.plot(bx, by, 'k-', linewidth=2)

    # 修复1: 正确绘制凸多边形障碍物
    for polygon in result['obstacles']:
        # 将多边形顶点转换为适合matplotlib的格式
        poly_x = [v[0] for v in polygon]
        poly_y = [v[1] for v in polygon]
        # 闭合多边形
        poly_x.append(poly_x[0])
        poly_y.append(poly_y[0])
        # 绘制多边形
        ax.plot(poly_x, poly_y, 'r-', linewidth=1.5)
        # 填充区域（半透明）
        ax.fill(poly_x, poly_y, 'red', alpha=0.4)

    # 修复2: 正确引用采样点
    # if 'samples' in result and result['samples']:
        # sx, sy = zip(*result['samples'])
        # ax.scatter(sx, sy, color='blue', s=30, alpha=0.8, zorder=5, label='Sample Points')

    # 绘制所有IRIS区域
    if regions:
        # 为每个区域生成颜色 (使用viridis色谱)
        if region_colors is None:
            region_colors = plt.cm.viridis(np.linspace(0, 1, len(regions)))

        for i, region in enumerate(regions):
            try:
                vrep = VPolytope(region)
                vertices = vrep.vertices()

                # 确保顶点按几何顺序排列
                from scipy.spatial import ConvexHull
                points = vertices.T
                if points.shape[0] >= 3:
                    try:
                        hull = ConvexHull(points)
                        ordered_vertices = points[hull.vertices].T
                    except:
                        center = np.mean(points, axis=0)
                        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
                        sorted_idx = np.argsort(angles)
                        ordered_vertices = points[sorted_idx].T
                else:
                    ordered_vertices = vertices

                # 绘制多边形
                poly_x = np.append(ordered_vertices[0, :], ordered_vertices[0, 0])
                poly_y = np.append(ordered_vertices[1, :], ordered_vertices[1, 0])

                # 使用不同的颜色
                color = region_colors[3]
                # color = region_colors[1]
                label = f'IRIS Region {i+1}' if show_region_labels and i < 10 else None

                # 绘制边界线
                ax.plot(poly_x, poly_y, '-', linewidth=1.5, color=color, label=label)

                # 填充区域（半透明）
                ax.fill(poly_x, poly_y, color=color, alpha=0.2)

                # # 添加区域中心标记
                # center_x = np.mean(poly_x[:-1])
                # center_y = np.mean(poly_y[:-1])
                # ax.plot(center_x, center_y, 'o', markersize=5, color=color, alpha=0.8)
            except Exception as e:
                print(f"可视化区域 {i+1} 失败: {str(e)}")

    # # 绘制优化轨迹
    if trajectory is not None:
        # 生成时间点并计算轨迹点
        times = np.linspace(trajectory.start_time(), trajectory.end_time(), num_waypoints)
        waypoints = trajectory.vector_values(times)

        # 提取 x 和 y 坐标
        wx = waypoints[0, :]
        wy = waypoints[1, :]

        # # 绘制轨迹
        # ax.plot(wx, wy, color=trajectory_color, linewidth=trajectory_linewidth, linestyle='-', zorder=10, label='Optimal Path')

        # 在轨迹起点和终点添加标记
        ax.plot(wx[0], wy[0], 'go', markersize=8, markeredgewidth=1.5, markeredgecolor='blue', label='Start',zorder=20)
        ax.plot(wx[-1], wy[-1], 'ro', markersize=8, markeredgewidth=1.5, markeredgecolor='red', label='Goal',zorder=20)



    # --- 新增：绘制贝塞尔曲线和控制点 ---
    if bezier_trajectory_obj is not None:
        print("--- 开始绘制贝塞尔曲线和控制点 ---")
        # 1. 提取控制点
        # BezierTrajectory 类封装了 BsplineTrajectory 对象
        path_traj = bezier_trajectory_obj.path_traj
        time_traj = bezier_trajectory_obj.time_traj

        # 从 BsplineTrajectory 对象中获取控制点
        path_control_points = np.array(path_traj.control_points()) # 确保这里转换为numpy数组
        time_control_points = np.array(time_traj.control_points()) # 同样确保时间控制点也转换为numpy数组
        knots = path_traj.basis().knots() # 获取节点向量
        order = path_traj.basis().order()
        num_control_points = path_control_points.shape[0] # 现在可以安全地访问.shape属性

        print(f"  - 控制点数量: {num_control_points}")
        print(f"  - 空间控制点形状: {path_control_points.shape}")
        print(f"  - 时间控制点形状: {time_control_points.shape}")
        print(f"  - 节点向量: {knots}")
        print(f"  - 阶数: {order}")

        # 2. 可视化控制点
        if show_bezier_controls:
            ax.scatter(path_control_points[:, 0], path_control_points[:, 1], color=bezier_control_color, s=50, marker='s', zorder=15, label='Bezier Control Points', edgecolors='black', linewidth=0.5)

        # 3. 可视化贝塞尔曲线 (通过密集采样Bspline近似)
        if show_bezier_controls: # 为了清晰，只在显示控制点时绘制曲线
            # 使用与轨迹相同的采样方式，但可以更密集
            s_values_for_curve = np.linspace(path_traj.start_time(), path_traj.end_time(), num=num_waypoints*2) # 更密集的采样
            path_values_for_curve = path_traj.vector_values(s_values_for_curve)

            ax.plot(path_values_for_curve[0, :], path_values_for_curve[1, :], color=bezier_curve_color, linewidth=bezier_curve_linewidth, linestyle='--', zorder=12, label='Bezier Curve')

        print("--- 贝塞尔曲线和控制点绘制完成 ---")


    # 设置图形属性
    ax.set_xlim(0, domain['width'])
    ax.set_ylim(0, domain['height'])
    ax.set_aspect('equal')
    ax.set_title('Generated Environment with Multiple IRIS Regions, Optimal Path, and Bezier Controls')

    # 添加图例（仅当有标签时）
    # 替换原有的图例处理代码
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # 创建颜色条作为图例的一部分
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch



        # 组合所有图例元素
        all_handles = handles 
        ax.legend(handles=all_handles, loc='best', fontsize=9)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()