"""
HybridAStarGCSPlanner 测试脚本

测试 A*与GCS分层轨迹规划器的各种场景和配置。
"""

import os
import sys
import numpy as np
from typing import Optional, Tuple
import argparse

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')

# 添加路径
# 移除空字符串(如果存在)
if '' in sys.path:
    sys.path.remove('')

# 确保当前仓库的src_dir在最前面,这样不会误导入其他工作区的同名包
if src_dir in sys.path:
    sys.path.remove(src_dir)
sys.path.insert(0, src_dir)

if project_root in sys.path:
    sys.path.remove(project_root)
sys.path.insert(1, project_root)

from C_space_pkg.se2 import SE2ConfigurationSpace
from ackermann_gcs_pkg.ackermann_data_structures import PlanningResult
from config.project import ProjectConfig, VALID_PLANNER_MODES, load_project_config
from path_planner.scenario_utils import (
    convert_iris_to_hpolyhedron,
    create_endpoint_state,
    create_test_map,
    default_project_config,
    plan_path,
)


def print_ackermann_result(result: PlanningResult):
    """
    打印阿克曼GCS规划结果

    Args:
        result: 规划结果
    """
    print(f"\n{'='*60}")
    print("阿克曼GCS规划结果")
    print(f"{'='*60}")

    if result.success:
        print("✓ 规划成功")
        print(f"  求解时间: {result.solve_time:.2f}s")
        print(f"  SCP迭代次数: {result.num_iterations}")
        print(f"  收敛原因: {result.convergence_reason}")

        if result.trajectory_report:
            report = result.trajectory_report
            print(f"\n轨迹可行性: {'✓ 可行' if report.is_feasible else '✗ 不可行'}")

            if report.velocity_violation:
                v = report.velocity_violation
                print(f"  速度约束: {'✓ 满足' if not v.is_violated else f'✗ 违反 (max: {v.max_violation:.6f})'}")

            if report.acceleration_violation:
                a = report.acceleration_violation
                print(f"  加速度约束: {'✓ 满足' if not a.is_violated else f'✗ 违反 (max: {a.max_violation:.6f})'}")

            if report.curvature_violation:
                c = report.curvature_violation
                print(f"  曲率约束: {'✓ 满足' if not c.is_violated else f'✗ 违反 (max: {c.max_violation:.6f})'}")

            if report.workspace_violation:
                w = report.workspace_violation
                print(f"  工作空间约束: {'✓ 满足' if not w.is_violated else f'✗ 违反 (max: {w.max_violation:.6f})'}")
    else:
        print("✗ 规划失败")
        print(f"  求解时间: {result.solve_time:.2f}s")
        print(f"  错误信息: {result.error_message}")

    print(f"{'='*60}\n")


def run_ackermann_gcs_test(scenario: str,
                          obstacle_map: np.ndarray,
                          start: Tuple,
                          goal: Tuple,
                          corridor_width: float,
                          project_config: Optional[ProjectConfig] = None):
    """
    运行阿克曼GCS测试

    Args:
        scenario: 测试场景名称
        obstacle_map: 障碍物地图
        start: 起点 (x, y, theta)
        goal: 终点 (x, y, theta)
        corridor_width: 通道宽度
    """
    print(f"\n{'='*60}")
    print(f"测试场景: {scenario}, 模式: ackermann_gcs")
    print(f"{'='*60}")
    project_config = project_config or default_project_config()

    try:
        from path_planner import HybridAStarGCSPlanner
        from ackermann_gcs_pkg.ackermann_gcs_planner import AckermannGCSPlanner

        # 步骤1：创建配置空间
        c_space = SE2ConfigurationSpace(obstacle_map, resolution=0.1)

        # 步骤2：A*路径规划（用于生成IRIS区域）
        include_astar_gear = (
            project_config.ackermann.constraints.reference_gear_source == "astar"
        )
        path_result = plan_path(
            c_space, start, goal, project_config, include_gear=include_astar_gear
        )
        if isinstance(path_result, dict):
            path = path_result["path"]
            reference_path = path_result.get("gear_path") or path
        else:
            path = path_result
            reference_path = path
        if not path:
            raise ValueError("A*路径规划失败")

        # 步骤3：执行IRIS分解
        config = project_config.planner_config(
            scenario,
            enable_visualization=False,
            save_visualization=False,
            enable_gcs_optimization=False,
        )
        config.corridor_width = corridor_width
        planner = HybridAStarGCSPlanner(c_space, config)
        result = planner.process(path)

        if result.num_obstacles == 0:
            raise ValueError("IRIS分解失败，未生成可行区域")

        # 步骤4：转换IRIS区域为HPolyhedron（与 visualize_3d_trajectory.py 保持一致）
        iris_result = result.iris_zo_result or result.iris_np_result
        if not iris_result:
            raise ValueError("IRIS分解失败，未生成可行区域")
        workspace_regions = convert_iris_to_hpolyhedron(iris_result.regions)

        if not workspace_regions:
            raise ValueError("IRIS区域转换失败")

        # 步骤5：配置车辆参数
        vehicle_params = project_config.vehicle_params()

        # 步骤6：创建起终点状态
        source = create_endpoint_state(start[:2], start[2])
        target = create_endpoint_state(goal[:2], goal[2])
        direction_cone_overrides = (
            project_config.direction_cone_overrides()
            if project_config.ackermann.constraints.curvature_constraint_mode == "direction_cone"
            else {}
        )
        if direction_cone_overrides:
            print(f"使用 direction_cone 参数预设: {project_config.ackermann.direction_cone_profile}")
            for key, value in direction_cone_overrides.items():
                print(f"  {key}: {value}")

        # 步骤7：初始化AckermannGCSPlanner
        ackermann_planner = AckermannGCSPlanner(
            vehicle_params=vehicle_params,
            bezier_config=project_config.bezier_config()
        )

        # 步骤8：执行轨迹规划
        constraints = project_config.trajectory_constraints(workspace_regions)

        planning_result = ackermann_planner.plan_trajectory(
            source=source,
            target=target,
            workspace_regions=workspace_regions,
            constraints=constraints,
            cost_weights=project_config.cost_weights(),
            reference_path=reference_path,
            verbose=project_config.ackermann.verbose
        )

        # 步骤9：打印结果
        print_ackermann_result(planning_result)

        # 步骤10：可视化
        if planning_result.success:
            try:
                # 使用新的模块化可视化系统
                from visualization.ackermann.visualizer import visualize_ackermann_gcs_enhanced
                
                # 创建配置
                viz_config = project_config.visualization_config()
                
                # 执行可视化
                visualize_ackermann_gcs_enhanced(
                    trajectory=planning_result.trajectory,
                    vehicle_params=vehicle_params,
                    workspace_regions=workspace_regions,
                    source=source,
                    target=target,
                    obstacle_map=obstacle_map,
                    astar_path=None,  # TODO: 从planner获取A*路径
                    corridor_width=corridor_width,
                    resolution=0.1,  # C_space的分辨率
                    save_path=os.path.join(
                        project_config.visualization.output_dir,
                        f"ackermann_gcs_{scenario}_enhanced.png",
                    ),
                    config=viz_config
                )
                print(
                    "增强版可视化已保存到: "
                    f"{project_config.visualization.output_dir}/ackermann_gcs_{scenario}_enhanced.png"
                )
                
            except Exception as e:
                print(f"增强版可视化失败: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # 回退到简化可视化
                try:
                    visualize_ackermann_gcs_enhanced(
                        trajectory=planning_result.trajectory,
                        vehicle_params=vehicle_params,
                        source=source,
                        target=target,
                        save_path=os.path.join(
                            project_config.visualization.output_dir,
                            f"ackermann_gcs_{scenario}.png",
                        ),
                    )
                    print(
                        "简化可视化已保存到: "
                        f"{project_config.visualization.output_dir}/ackermann_gcs_{scenario}.png"
                    )
                except Exception as e2:
                    print(f"简化可视化也失败: {str(e2)}")

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ 测试失败: {scenario}")
        print(f"  错误: {str(e)}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()


def run_test(scenario: str = 'basic',
             mode: str = 'hybrid_astar_gcs',
             gcs_strategy: str = 'standard',
             gcs_cost: str = 'lunar_standard',
             project_config: Optional[ProjectConfig] = None):
    """
    运行测试

    Args:
        scenario: 测试场景
        mode: 规划模式
            - 'hybrid_astar_gcs': HybridAStar + GCS（默认）
            - 'ackermann_gcs': 阿克曼转向GCS
        gcs_strategy: GCS策略预设（仅hybrid_astar_gcs模式有效）
        gcs_cost: GCS成本预设（仅hybrid_astar_gcs模式有效）
    """
    project_config = project_config or default_project_config()
    if mode is None:
        mode = project_config.planner_mode
    if gcs_strategy is None:
        gcs_strategy = project_config.gcs.strategy_preset
    if gcs_cost is None:
        gcs_cost = project_config.gcs.cost_preset

    # 获取场景配置
    config = project_config.scenario_dict(scenario)

    # 创建地图
    obstacle_map = create_test_map(config['map_size'], scenario)

    # 根据模式选择测试函数
    if mode == 'ackermann_gcs':
        run_ackermann_gcs_test(
            scenario=scenario,
            obstacle_map=obstacle_map,
            start=config['start'],
            goal=config['goal'],
            corridor_width=config['corridor_width'],
            project_config=project_config
        )
    else:
        from path_planner import HybridAStarGCSPlanner

        # 原有HybridAStarGCS测试流程
        print(f"\n{'='*60}")
        print(f"测试场景: {scenario}, IRIS模式: np")
        print(f"GCS策略: {gcs_strategy}, GCS成本: {gcs_cost}")
        print(f"{'='*60}")

        c_space = SE2ConfigurationSpace(obstacle_map, resolution=0.1)

        path = plan_path(c_space, config['start'], config['goal'], project_config)
        if not path:
            print("路径规划失败")
            return

        # 执行分解（使用预设配置）
        planner_config = project_config.planner_config(
            scenario,
            enable_visualization=True,
            save_visualization=True,
        )
        planner_config.gcs_strategy_preset = gcs_strategy
        planner_config.gcs_cost_preset = gcs_cost
        planner_config._apply_gcs_strategy_preset()
        planner_config._apply_gcs_cost_preset()

        planner = HybridAStarGCSPlanner(c_space, planner_config)
        result = planner.process(path)

        # 打印结果
        print(f"\n结果: 区域数={result.num_obstacles}, 面积={result.obstacle_area:.2f}m², 时间={result.total_time:.2f}s")
        print(f"IRIS模式: {result.iris_mode_used}")
        print(f"GCS策略: {gcs_strategy}, GCS成本: {gcs_cost}")


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid A* + IRIS + GCS测试入口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scenario", nargs="?", default=None, help="测试场景名称")
    parser.add_argument("mode", nargs="?", default=None, choices=VALID_PLANNER_MODES, help="规划模式")
    parser.add_argument("gcs_strategy", nargs="?", default=None, help="GCS策略预设")
    parser.add_argument("gcs_cost", nargs="?", default=None, help="GCS成本预设")
    parser.add_argument("--config", default=None, help="YAML配置文件路径")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="覆盖配置项，例如 --set ackermann.vehicle.max_velocity=8.0",
    )
    parser.add_argument("--dump-config", default=None, help="导出合并后的配置到指定YAML文件")
    args = parser.parse_args()

    project_config = load_project_config(
        args.config,
        args.overrides,
        export_resolved_path=args.dump_config,
    )
    scenario = args.scenario or project_config.scenario.name
    run_test(
        scenario=scenario,
        mode=args.mode,
        gcs_strategy=args.gcs_strategy,
        gcs_cost=args.gcs_cost,
        project_config=project_config,
    )


if __name__ == "__main__":
    main()
