"""
批量测试曲率约束规划成功率
运行10次，统计成功率，不输出可视化
"""

import os
import sys
import time
import io
import numpy as np
import argparse

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')

if '' in sys.path:
    sys.path.remove('')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，不显示窗口

from path_planner.scenario_utils import (
    convert_iris_to_hpolyhedron,
    create_endpoint_state,
    create_test_map,
    plan_path,
)
from C_space_pkg.se2 import SE2ConfigurationSpace
from config.project import ProjectConfig, load_project_config


def run_single_test(run_id: int, scenario: str, project_config: ProjectConfig):
    """运行单次测试，返回结果字典"""
    result = {
        'run_id': run_id,
        'success': False,
        'feasible': False,
        'solve_time': 0.0,
        'max_curvature': 0.0,
        'curvature_violation': 0.0,
        'velocity_violation': 0.0,
        'acceleration_violation': 0.0,
        'workspace_violation': 0.0,
        'error': None,
    }

    try:
        config = project_config.scenario_dict(scenario)
        start = config['start']
        goal = config['goal']

        # 1. 配置空间
        c_space = SE2ConfigurationSpace(
            create_test_map(config['map_size'], scenario), resolution=0.1
        )

        # 2. A*路径规划
        astar_path = plan_path(c_space, start, goal, project_config)
        if not astar_path:
            result['error'] = 'A* path planning failed'
            return result

        # 3. IRIS分解
        from path_planner import HybridAStarGCSPlanner

        planner_config = project_config.planner_config(
            scenario,
            enable_visualization=False,
            save_visualization=False,
            enable_gcs_optimization=False,
        )
        planner = HybridAStarGCSPlanner(c_space, planner_config)
        iris_planner_result = planner.process(astar_path)

        # 获取IRIS结果（优先使用zo模式）
        iris_result = iris_planner_result.iris_zo_result or iris_planner_result.iris_np_result
        if not iris_result:
            result['error'] = 'IRIS decomposition failed'
            return result

        workspace_regions = convert_iris_to_hpolyhedron(iris_result.regions)

        # 4. AckermannGCS规划（启用曲率硬约束）
        vehicle_params = project_config.vehicle_params()
        source = create_endpoint_state(start[:2], start[2])
        target = create_endpoint_state(goal[:2], goal[2])
        constraints = project_config.trajectory_constraints(workspace_regions)

        from ackermann_gcs_pkg.ackermann_gcs_planner import AckermannGCSPlanner

        ackermann_planner = AckermannGCSPlanner(
            vehicle_params=vehicle_params,
            bezier_config=project_config.bezier_config()
        )

        t_start = time.time()
        planning_result = ackermann_planner.plan_trajectory(
            source=source,
            target=target,
            workspace_regions=workspace_regions,
            constraints=constraints,
            cost_weights=project_config.cost_weights(),
            reference_path=astar_path,
            verbose=False
        )
        result['solve_time'] = time.time() - t_start

        if planning_result.trajectory is None:
            result['error'] = 'No trajectory returned'
            return result

        result['success'] = bool(planning_result.success)
        if not planning_result.success:
            result['error'] = planning_result.error_message or 'Trajectory infeasible'

        # 提取可行性信息
        report = planning_result.trajectory_report
        if report is not None:
            result['feasible'] = report.is_feasible
            if report.curvature_violation:
                result['curvature_violation'] = report.curvature_violation.max_violation
            if report.velocity_violation:
                result['velocity_violation'] = report.velocity_violation.max_violation
            if report.acceleration_violation:
                result['acceleration_violation'] = report.acceleration_violation.max_violation
            if report.workspace_violation:
                result['workspace_violation'] = report.workspace_violation.max_violation
            if report.curvature_stats:
                result['max_curvature'] = report.curvature_stats.max_curvature

    except Exception as e:
        result['error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="批量测试曲率约束规划成功率",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scenario", nargs="?", default=None, help="测试场景名称")
    parser.add_argument("--num-runs", type=int, default=None, help="运行次数")
    parser.add_argument("--config", default="config/experiments/curvature_direction_cone.yaml", help="YAML配置文件路径")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="覆盖配置项，例如 --set batch.num_runs=3",
    )
    parser.add_argument("--dump-config", default=None, help="导出合并后的配置到指定YAML文件")
    args = parser.parse_args()

    project_config = load_project_config(
        args.config,
        args.overrides,
        export_resolved_path=args.dump_config,
    )
    num_runs = args.num_runs if args.num_runs is not None else project_config.batch.num_runs
    scenario = args.scenario or project_config.batch.scenario
    curvature_mode = project_config.ackermann.constraints.curvature_constraint_mode

    print("=" * 70)
    print(
        f"批量测试曲率约束规划 - 场景: {scenario}, "
        f"模式: {curvature_mode}, 次数: {num_runs}"
    )
    if curvature_mode == "direction_cone":
        print(f"方向锥参数预设: {project_config.ackermann.direction_cone_profile}")
    print("=" * 70)

    results = []
    for i in range(num_runs):
        # 进度条
        pct = (i + 1) / num_runs * 100
        bar_len = 30
        filled = int(bar_len * (i + 1) / num_runs)
        bar = '█' * filled + '░' * (bar_len - filled)
        sys.stdout.write(
            f"\r  [{bar}] {i+1}/{num_runs} ({pct:.0f}%)")
        sys.stdout.flush()

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if project_config.batch.quiet_each_run:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
        try:
            t0 = time.time()
            result = run_single_test(i + 1, scenario, project_config)
            elapsed = time.time() - t0
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        result['elapsed'] = elapsed
        results.append(result)

    # 进度条换行
    print()

    # 统计分析
    print("\n" + "=" * 70)
    print("测试结果统计分析")
    print("=" * 70)

    successes = [r for r in results if r['success']]
    feasibles = [r for r in results if r['feasible']]

    print(f"\n总运行次数: {num_runs}")
    print(f"规划成功次数: {len(successes)} ({len(successes)/num_runs*100:.1f}%)")
    print(f"轨迹可行次数: {len(feasibles)} ({len(feasibles)/num_runs*100:.1f}%)")

    if successes:
        solve_times = [r['solve_time'] for r in successes]
        max_curvatures = [r['max_curvature'] for r in successes]
        curv_violations = [r['curvature_violation'] for r in successes]
        vel_violations = [r['velocity_violation'] for r in successes]
        acc_violations = [r['acceleration_violation'] for r in successes]
        workspace_violations = [r['workspace_violation'] for r in successes]

        print(f"\n--- 求解时间 (s) ---")
        print(f"  均值: {np.mean(solve_times):.3f}")
        print(f"  标准差: {np.std(solve_times):.3f}")
        print(f"  最小: {np.min(solve_times):.3f}")
        print(f"  最大: {np.max(solve_times):.3f}")

        print(f"\n--- 最大曲率 (1/m) ---")
        print(f"  均值: {np.mean(max_curvatures):.6f}")
        print(f"  标准差: {np.std(max_curvatures):.6f}")
        print(f"  最小: {np.min(max_curvatures):.6f}")
        print(f"  最大: {np.max(max_curvatures):.6f}")
        print(f"  κ_max限制: {project_config.vehicle_params().max_curvature:.6f}")

        print(f"\n--- 曲率违反量 ---")
        print(f"  均值: {np.mean(curv_violations):.6f}")
        print(f"  最大: {np.max(curv_violations):.6f}")
        print(f"  零违反次数: {sum(1 for v in curv_violations if v < 1e-4)}/{len(successes)}")

        print(f"\n--- 速度违反量 ---")
        print(f"  均值: {np.mean(vel_violations):.6f}")
        print(f"  最大: {np.max(vel_violations):.6f}")
        print(f"  零违反次数: {sum(1 for v in vel_violations if v < 1e-4)}/{len(successes)}")

        print(f"\n--- 加速度违反量 ---")
        print(f"  均值: {np.mean(acc_violations):.6f}")
        print(f"  最大: {np.max(acc_violations):.6f}")
        print(f"  零违反次数: {sum(1 for v in acc_violations if v < 1e-4)}/{len(successes)}")

        print(f"\n--- 工作空间违反量 ---")
        print(f"  均值: {np.mean(workspace_violations):.6f}")
        print(f"  最大: {np.max(workspace_violations):.6f}")
        print(f"  零违反次数: {sum(1 for v in workspace_violations if v < 1e-4)}/{len(successes)}")

        # 逐次详细结果
        print(f"\n--- 逐次详细结果 ---")
        for r in successes:
            feas_str = "✓" if r['feasible'] else "✗"
            print(f"  运行{r['run_id']:2d}: {feas_str} | "
                  f"κ_max={r['max_curvature']:.4f} | "
                  f"κ_viol={r['curvature_violation']:.4f} | "
                  f"v_viol={r['velocity_violation']:.4f} | "
                  f"a_viol={r['acceleration_violation']:.4f} | "
                  f"ws_viol={r['workspace_violation']:.4f}")

    errors = [r for r in results if r['error']]
    if errors:
        print(f"\n--- 错误汇总 ---")
        for r in errors:
            print(f"  运行{r['run_id']}: {r['error']}")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
