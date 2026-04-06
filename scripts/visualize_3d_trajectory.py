#!/usr/bin/env python3
"""
独立的3D配置空间轨迹可视化工具

支持交互式视角调整，可以滚动、旋转、缩放查看3D轨迹
"""

import os
import sys

# 路径设置 - 必须在所有导入之前
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 从scripts回到项目根目录
src_dir = os.path.join(project_root, 'src')

# 移除空字符串(如果存在)
if '' in sys.path:
    sys.path.remove('')

# 添加项目根目录和src目录到Python路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple

# from scripts.test_hybrid_astar_gcs_planner import 
from scripts.hybrid_astar_gcs_planner import (
    SCENARIO_CONFIGS, 
    convert_iris_to_hpolyhedron, create_endpoint_state,
    DEFAULT_VEHICLE_PARAMS,
    plan_path,
    create_test_map
)
from C_space_pkg.se2 import SE2ConfigurationSpace
from path_planner.scripts.hybrid_astar_gcs_planner import HybridAStarGCSPlanner
from path_planner.scripts.planner_support import PlannerConfig
# from tests.unit.test_hybrid_astar_gcs_planner import plan_path
from ackermann_gcs_pkg.ackermann_gcs_planner import AckermannGCSPlanner
from ackermann_gcs_pkg.ackermann_data_structures import BezierConfig, SCPConfig
from ackermann_gcs_pkg.flat_output_mapper import compute_flat_output_mapping

# 导入输出管理器
from visualization import VisualizationOutputManager

# 导入控制点相关模块
from visualization.ackermann.control_point_extractor import extract_control_points
from config.visualization import ControlPointConfig


def visualize_3d_trajectory_interactive(
    scenario: str = 'narrow',
    elev: float = 25.0,
    azim: float = 45.0,
    save_path: Optional[str] = None,
    auto_save: bool = True,
    show_2d: bool = True,
    show_control_points: bool = True,
    control_point_size: Optional[int] = None,
    control_point_color: Optional[str] = None,
    control_point_marker: Optional[str] = None
):
    """
    交互式3D轨迹可视化
    
    Args:
        scenario: 测试场景名称
        elev: 初始仰角（度）
        azim: 初始方位角（度）
        save_path: 保存路径（可选）
        auto_save: 是否自动保存（默认True）
        show_2d: 是否同时显示2D视图（默认True）
        show_control_points: 是否显示控制点（默认True）
        control_point_size: 控制点标记大小（默认60）
        control_point_color: 控制点颜色（默认'orange'）
        control_point_marker: 控制点标记形状（默认'D'菱形）
    """
    print("=" * 60)
    print(f"3D配置空间轨迹可视化 - 场景: {scenario}")
    print("=" * 60)
    
    # 1. 获取场景配置
    config = SCENARIO_CONFIGS[scenario]
    
    # 2. 创建障碍物地图和配置空间
    print("\n步骤1: 创建配置空间...")
    obstacle_map = create_test_map(config['map_size'], scenario)
    c_space = SE2ConfigurationSpace(obstacle_map, resolution=0.1)
    
    # 3. A*路径规划
    print("步骤2: A*路径规划...")
    start = config['start']
    goal = config['goal']
    path = plan_path(c_space, start, goal)
    
    if not path:
        print("✗ A*路径规划失败")
        return
    
    print(f"✓ A*路径规划成功，路径点数: {len(path)}")
    
    # 4. IRIS分解
    print("步骤3: IRIS分解...")
    planner_config = PlannerConfig(
        use_iris=True,
        corridor_width=config['corridor_width'],
        enable_visualization=False,
        save_visualization=False,
        enable_gcs_optimization=False
    )
    
    planner = HybridAStarGCSPlanner(c_space, planner_config)
    result = planner.process(path)
    
    workspace_regions = convert_iris_to_hpolyhedron(result.iris_np_result.regions)
    print(f"✓ IRIS分解完成，区域数: {len(workspace_regions)}")
    
    # 5. AckermannGCS规划
    print("步骤4: AckermannGCS轨迹规划...")
    vehicle_params = DEFAULT_VEHICLE_PARAMS
    source = create_endpoint_state(start[:2], start[2])
    target = create_endpoint_state(goal[:2], goal[2])
    
    ackermann_planner = AckermannGCSPlanner(
        vehicle_params=vehicle_params,
        bezier_config=BezierConfig(order=5, continuity=1),
        scp_config=SCPConfig(max_iterations=10, convergence_tolerance=1e-3)
    )
    
    planning_result = ackermann_planner.plan_trajectory(
        source=source,
        target=target,
        workspace_regions=workspace_regions,
        constraints=None,
        cost_weights={"time": 1.0, "path_length": 0.1, "energy": 100.0 ,    # 添加曲率惩罚成本（新增）
    "curvature_squared": 0.5,      # 曲率平方积分权重
    "curvature_derivative": 0.05,  # 曲率导数平方积分权重
    "curvature_peak": 0.1         # 曲率峰值惩罚权重
    },
        verbose=True  # 启用详细输出，显示约束信息
    )
    
    if not planning_result.success:
        print("✗ 轨迹规划失败")
        return
    
    print(f"✓ 轨迹规划成功，求解时间: {planning_result.solve_time:.2f}s")
    
    # 6. 提取控制点（如果启用）
    control_point_data = None
    control_point_config = None
    
    if show_control_points:
        print("步骤5b: 提取控制点...")
        try:
            control_point_data = extract_control_points(planning_result.trajectory)
            print(f"✓ 提取到 {control_point_data.num_points} 个控制点")
            
            # 创建控制点配置
            control_point_config = ControlPointConfig(
                show_control_points=True,
                control_point_size=control_point_size or 60,
                control_point_color=control_point_color or 'orange',
                control_point_marker=control_point_marker or 'D'
            )
        except Exception as e:
            print(f"⚠ 控制点提取失败: {str(e)}")
    
    # 7. 采样轨迹
    print("步骤5: 采样轨迹数据...")
    mapping = compute_flat_output_mapping(
        planning_result.trajectory, vehicle_params, num_samples=200
    )
    
    position = mapping["position"]
    heading = mapping["heading"].flatten()
    
    # 8. 创建2D可视化（如果启用）
    if show_2d:
        print("步骤7a: 创建2D可视化...")
        fig_2d, ax_2d = plt.subplots(1, 1, figsize=(10, 8))
        
        # 绘制障碍物
        resolution = 0.1
        map_height, map_width = obstacle_map.shape
        x_max = map_width * resolution
        y_max = map_height * resolution
        extent = [0, x_max, 0, y_max]
        
        ax_2d.imshow(obstacle_map, origin='lower', cmap='RdYlGn_r', alpha=0.3, 
                     aspect='auto', extent=extent)
        
        # 绘制IRIS区域
        from pydrake.geometry.optimization import VPolytope
        from matplotlib.patches import Polygon as MplPolygon
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']
        
        for i, region in enumerate(workspace_regions):
            try:
                vpoly = VPolytope(region)
                vertices = vpoly.vertices().T
                
                # 按角度排序顶点
                center = np.mean(vertices, axis=0)
                angles = np.arctan2(vertices[:, 1] - center[1], 
                                   vertices[:, 0] - center[0])
                sorted_indices = np.argsort(angles)
                vertices_ordered = vertices[sorted_indices]
                
                color = colors[i % len(colors)]
                polygon = MplPolygon(vertices_ordered, closed=True, 
                                    facecolor=color, edgecolor='blue',
                                    alpha=0.3, linewidth=1)
                ax_2d.add_patch(polygon)
            except:
                pass
        
        # 绘制轨迹
        ax_2d.plot(position[0, :], position[1, :], 'r-', linewidth=2, label='Trajectory')
        
        # 绘制A*路径
        path_array_2d = np.array([(p[0], p[1]) for p in path])
        ax_2d.plot(path_array_2d[:, 0], path_array_2d[:, 1], 'g--', 
                   linewidth=1.5, alpha=0.5, label='A* Path')
        
        # 绘制起终点
        ax_2d.plot(source.position[0], source.position[1], 'go', 
                   markersize=10, label='Source')
        ax_2d.plot(target.position[0], target.position[1], 'r*', 
                   markersize=15, label='Target')
        
        # 绘制控制点（如果启用）
        if control_point_data is not None and control_point_config is not None:
            points = control_point_data.points
            ax_2d.scatter(
                points[:, 0],  # x
                points[:, 1],  # y
                c=control_point_config.control_point_color,
                marker=control_point_config.control_point_marker,
                s=control_point_config.control_point_size,
                alpha=control_point_config.control_point_alpha,
                label=control_point_config.control_point_label,
                zorder=control_point_config.control_point_zorder,
                edgecolors='black',
                linewidths=0.5
            )
        
        ax_2d.set_xlabel('x (m)', fontsize=12)
        ax_2d.set_ylabel('y (m)', fontsize=12)
        ax_2d.set_title(f'2D Trajectory View - {scenario}', fontsize=14, fontweight='bold')
        ax_2d.legend(loc='best', fontsize=10)
        ax_2d.grid(True, alpha=0.3)
        ax_2d.axis('equal')
        
        # 自动保存2D图
        if auto_save:
            # 使用输出管理器生成路径（自动生成时间戳run_id）
            output_manager = VisualizationOutputManager.get_instance()
            save_path_2d = output_manager.generate_output_path(
                filename=f'2d_trajectory_{scenario}.png',
                dimension='2d'
                # 不指定run_id，自动生成时间戳
            )
            fig_2d.tight_layout()
            fig_2d.savefig(save_path_2d, dpi=150, bbox_inches='tight')
            print(f"✓ 2D可视化已保存到: {save_path_2d}")
            plt.close(fig_2d)
    
    # 9. 创建3D可视化
    print("步骤7b: 创建3D可视化...")
    
    # 创建图表
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取x, y, theta
    x = position[0, :]
    y = position[1, :]
    theta = heading
    
    # 绘制3D轨迹
    ax.plot(x, y, theta, 'r-', linewidth=2.5, label='GCS Trajectory')
    
    # 绘制起点
    ax.scatter(
        source.position[0], source.position[1], source.heading,
        c='green', marker='s', s=100, label='Source', zorder=5
    )
    
    # 绘制终点
    ax.scatter(
        target.position[0], target.position[1], target.heading,
        c='red', marker='*', s=150, label='Target', zorder=5
    )
    
    # 绘制A*路径（如果有theta信息）
    path_array = np.array([(p[0], p[1], p[2]) for p in path])
    ax.plot(
        path_array[:, 0], path_array[:, 1], path_array[:, 2],
        'g--', linewidth=1.5, alpha=0.5, label='A* Path'
    )
    
    # 绘制xy平面投影
    ax.plot(x, y, np.full_like(theta, theta.min()), 
            'b:', linewidth=1, alpha=0.3, label='xy Projection')
    
    # 绘制控制点（如果启用）
    if control_point_data is not None and control_point_config is not None:
        points = control_point_data.points
        ax.scatter(
            points[:, 0],  # x
            points[:, 1],  # y
            points[:, 2],  # theta
            c=control_point_config.control_point_color,
            marker=control_point_config.control_point_marker,
            s=control_point_config.control_point_size,
            alpha=control_point_config.control_point_alpha,
            label=control_point_config.control_point_label,
            zorder=control_point_config.control_point_zorder,
            edgecolors='black',
            linewidths=0.5
        )
    
    # 设置坐标轴
    ax.set_xlabel('x (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('y (m)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_zlabel('θ (rad)', fontsize=12, fontweight='bold', labelpad=10)
    
    # 设置标题
    ax.set_title(
        f'3D Configuration Space Trajectory\n'
        f'Scenario: {scenario} | Regions: {len(workspace_regions)}',
        fontsize=14, fontweight='bold', pad=20
    )
    
    # 设置初始视角
    ax.view_init(elev=elev, azim=azim)
    
    # 设置图例
    ax.legend(loc='upper left', fontsize=10)
    
    # 设置网格
    ax.grid(True, alpha=0.3)
    
    # 添加说明文字
    info_text = (
        f"Interactive Controls:\n"
        f"• Drag to rotate view\n"
        f"• Scroll to zoom\n"
        f"• Right-drag to pan\n"
        f"\n"
        f"Trajectory Info:\n"
        f"• Points: {len(x)}\n"
        f"• x: [{x.min():.2f}, {x.max():.2f}]\n"
        f"• y: [{y.min():.2f}, {y.max():.2f}]\n"
        f"• θ: [{np.degrees(theta.min()):.1f}°, {np.degrees(theta.max()):.1f}°]"
    )
    
    fig.text(0.02, 0.98, info_text, fontsize=9, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 调整布局
    plt.tight_layout()
    
    # 自动保存逻辑
    if auto_save and save_path is None:
        # 使用输出管理器生成路径（自动生成时间戳run_id）
        output_manager = VisualizationOutputManager.get_instance()
        save_path = output_manager.generate_output_path(
            filename=f'3d_trajectory_{scenario}.png',
            dimension='3d'
            # 不指定run_id，自动生成时间戳
        )
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ 3D可视化已保存到: {save_path}")
    
    # 显示交互式窗口
    print("\n" + "=" * 60)
    print("交互式控制说明:")
    print("=" * 60)
    print("• 鼠标左键拖动：旋转视角")
    print("• 鼠标滚轮：缩放")
    print("• 鼠标右键拖动：平移")
    print("• 关闭窗口退出")
    print("=" * 60)
    
    plt.show()
    
    return fig, ax


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='3D配置空间轨迹可视化工具'
    )
    parser.add_argument(
        'scenario', 
        type=str, 
        nargs='?', 
        default='narrow',
        help='测试场景名称 (default: narrow)'
    )
    parser.add_argument(
        '--elev', 
        type=float, 
        default=25.0,
        help='初始仰角（度）(default: 25.0)'
    )
    parser.add_argument(
        '--azim', 
        type=float, 
        default=45.0,
        help='初始方位角（度）(default: 45.0)'
    )
    parser.add_argument(
        '--save', 
        type=str, 
        default=None,
        help='保存路径（可选）'
    )
    parser.add_argument(
        '--no-auto-save', 
        action='store_true',
        help='禁用自动保存'
    )
    parser.add_argument(
        '--no-2d', 
        action='store_true',
        help='不显示2D视图'
    )
    parser.add_argument(
        '--no-control-points',
        action='store_true',
        help='不显示控制点'
    )
    parser.add_argument(
        '--control-point-size',
        type=int,
        default=None,
        help='控制点标记大小（默认60）'
    )
    parser.add_argument(
        '--control-point-color',
        type=str,
        default=None,
        help='控制点颜色（默认orange）'
    )
    parser.add_argument(
        '--control-point-marker',
        type=str,
        default=None,
        help='控制点标记形状（默认D菱形）'
    )
    
    args = parser.parse_args()
    
    # 执行可视化
    visualize_3d_trajectory_interactive(
        scenario=args.scenario,
        elev=args.elev,
        azim=args.azim,
        save_path=args.save,
        auto_save=not args.no_auto_save,
        show_2d=not args.no_2d,
        show_control_points=not args.no_control_points,
        control_point_size=args.control_point_size,
        control_point_color=args.control_point_color,
        control_point_marker=args.control_point_marker
    )


if __name__ == '__main__':
    main()
