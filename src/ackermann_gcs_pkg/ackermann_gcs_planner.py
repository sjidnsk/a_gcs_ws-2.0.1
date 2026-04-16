"""
阿克曼转向车辆GCS规划器

本模块实现了阿克曼转向车辆的GCS轨迹规划器，整合所有子模块。
"""

import time
import numpy as np
from typing import Optional, List

from pydrake.geometry.optimization import HPolyhedron
from pydrake.trajectories import BsplineTrajectory

from .ackermann_bezier_gcs import AckermannBezierGCS
from .ackermann_scp_solver import AckermannSCPSolver
from .ackermann_data_structures import (
    VehicleParams,
    EndpointState,
    TrajectoryConstraints,
    SCPConfig,
    BezierConfig,
    PlanningResult,
    # CurvatureCostConfig,  # [2025-04-06] 曲率成本功能暂时禁用
    # CurvatureCostWeights,  # [2025-04-06] 曲率成本功能暂时禁用
)
from .trajectory_evaluator import TrajectoryEvaluator
# from .curvature_cost_module import CurvatureCostModule  # [2025-04-06] 曲率成本功能暂时禁用
from .curvature_statistics import CurvatureStatistics


class AckermannGCSPlanner:
    """
    阿克曼转向车辆GCS规划器

    整合平坦输出映射器、AckermannBezierGCS、SCP求解器、轨迹评估器，提供完整的轨迹规划功能。
    """

    def __init__(
        self,
        vehicle_params: VehicleParams,
        bezier_config: Optional[BezierConfig] = None,
        scp_config: Optional[SCPConfig] = None,
    ):
        """
        初始化规划器

        Args:
            vehicle_params: 车辆参数
            bezier_config: 贝塞尔曲线配置，如果为None则使用默认配置
            scp_config: SCP配置，如果为None则使用默认配置
        """
        # 参数验证
        if not isinstance(vehicle_params, VehicleParams):
            raise TypeError(f"vehicle_params must be VehicleParams instance, got {type(vehicle_params)}")

        self.vehicle_params = vehicle_params
        self.bezier_config = bezier_config if bezier_config is not None else BezierConfig()
        self.scp_config = scp_config if scp_config is not None else SCPConfig()

        # 初始化轨迹评估器
        self.evaluator = TrajectoryEvaluator(vehicle_params)

    def plan_trajectory(
        self,
        source: EndpointState,
        target: EndpointState,
        workspace_regions: List[HPolyhedron],
        constraints: Optional[TrajectoryConstraints] = None,
        cost_weights: Optional[dict] = None,
        verbose: bool = True,
    ) -> PlanningResult:
        """
        规划轨迹

        执行以下步骤：
        1. 构建轨迹约束（如果未提供）
        2. 初始化AckermannBezierGCS
        3. 添加起终点约束
        4. 添加凸约束（速度、加速度）
        5. 添加成本函数（时间、路径长度、能量）
        6. 初始化AckermannSCPSolver
        7. SCP迭代求解
        8. 评估轨迹
        9. 返回规划结果

        Args:
            source: 起点状态
            target: 终点状态
            workspace_regions: 工作空间区域列表
            constraints: 轨迹约束，如果为None则从vehicle_params推导
            cost_weights: 成本权重字典，包含：
                - time: 时间成本权重
                - path_length: 路径长度成本权重
                - energy: 能量成本权重
            verbose: 是否打印详细日志

        Returns:
            规划结果
        """
        start_time = time.time()

        if verbose:
            print("[Planner] Starting trajectory planning...")
            print(f"[Planner] Source: position={source.position}, heading={source.heading:.3f}")
            print(f"[Planner] Target: position={target.position}, heading={target.heading:.3f}")
            print(f"[Planner] Workspace regions: {len(workspace_regions)}")

        # 步骤1：构建轨迹约束（如果未提供）
        if constraints is None:
            # 从成本权重推导 min_velocity（如果提供了成本权重）
            min_vel = 1.58  # 默认值，对应 w_time=1.0, w_energy=0.1
            if cost_weights is not None and "time" in cost_weights and "energy" in cost_weights:
                w_time = cost_weights["time"]
                w_energy = cost_weights["energy"]
                if w_energy > 0:
                    min_vel = TrajectoryConstraints.compute_min_velocity_from_weights(
                        w_time, w_energy
                    )
            constraints = TrajectoryConstraints(
                max_velocity=self.vehicle_params.max_velocity,
                max_acceleration=self.vehicle_params.max_acceleration,
                max_curvature=self.vehicle_params.max_curvature,
                workspace_regions=workspace_regions,
                min_velocity=min_vel,
            )

        # 步骤2：初始化AckermannBezierGCS
        if verbose:
            print("[Planner] Initializing AckermannBezierGCS...")

        bezier_gcs = AckermannBezierGCS(
            regions=workspace_regions,
            vehicle_params=self.vehicle_params,
            bezier_config=self.bezier_config,
        )

        # 步骤3：添加起终点约束
        if verbose:
            print("[Planner] Adding source and target constraints...")

        bezier_gcs.addSourceTargetWithHeading(
            source_position=source.position,
            source_heading=source.heading,
            target_position=target.position,
            target_heading=target.heading,
            source_velocity=(
                source.velocity * np.array([np.cos(source.heading), np.sin(source.heading)])
                if source.velocity is not None
                else None
            ),
            target_velocity=(
                target.velocity * np.array([np.cos(target.heading), np.sin(target.heading)])
                if target.velocity is not None
                else None
            ),
        )

        # 步骤4：添加凸约束（速度）
        if verbose:
            print("\n" + "=" * 70)
            print("速度约束添加")
            print("=" * 70)
            print(f"约束类型: 标量速度约束（SOCP）")
            print(f"约束形式: ||v||_2 <= {constraints.max_velocity} m/s")
            print(f"物理模型: 阿克曼车辆速度模长限制")
            print(f"凸性保证: 二阶锥约束（Lorentz锥）")
            print(f"优势: 可行域比矢量约束大27%，更符合物理实际")
            print("=" * 70)

        # 使用标量速度约束（SOCP）
        # 约束形式：||v||_2 <= max_velocity
        # 更符合阿克曼车辆的物理模型
        bezier_gcs.addScalarVelocityLimit(constraints.max_velocity)
        
        # 原有矢量约束（已弃用，保留作为参考）
        # bezier_gcs.addVelocityLimits(np.zeros(2), np.full(2, constraints.max_velocity))

        # 步骤4.5：添加曲率硬约束（如果启用）
        if constraints.enable_curvature_hard_constraint or \
           constraints.curvature_constraint_mode == "hard":
            if verbose:
                print("\n" + "=" * 70)
                print("曲率硬约束添加")
                print("=" * 70)
                print(f"约束类型: 凸硬约束（Lorentz锥）")
                print(f"约束形式: ||Q_j||_2 <= C = kappa_max * rho_min^2")
                print(f"最大曲率: {constraints.max_curvature:.4f} 1/m")
                print(f"最小速度: {constraints.min_velocity:.4f} m/s")
                rho_min = constraints.min_velocity * 1.0  # h_bar_prime默认1.0
                C = constraints.max_curvature * rho_min ** 2
                print(f"rho_min: {rho_min:.4f}")
                print(f"约束阈值 C: {C:.6f}")
                print(f"凸性保证: 二阶锥约束（Lorentz锥）")
                print(f"保守性: Cauchy-Schwarz + 凸包 + 速度下界")
                if constraints.min_velocity > 0:
                    alpha = (constraints.max_velocity / constraints.min_velocity) ** 2
                    print(f"保守因子 alpha: {alpha:.1f}")
                print("=" * 70)

            try:
                bezier_gcs.addCurvatureHardConstraint(
                    max_curvature=constraints.max_curvature,
                    min_velocity=constraints.min_velocity,
                )
                if verbose:
                    print("✓ 曲率硬约束添加完成")
            except ValueError as e:
                if verbose:
                    print(f"⚠️  曲率硬约束添加失败: {e}")
                    print("   将跳过曲率硬约束，仅使用成本项软引导")

        # 步骤5：添加成本函数
        if verbose:
            print("[Planner] Adding cost functions...")

        if cost_weights is None:
            cost_weights = {"time": 1.0, "path_length": 0.1, "energy": 0.01}

        if "time" in cost_weights and cost_weights["time"] > 0:
            bezier_gcs.addTimeCost(cost_weights["time"])
            if verbose:
                print(f"[Planner]   Time cost weight: {cost_weights['time']}")

        if "path_length" in cost_weights and cost_weights["path_length"] > 0:
            bezier_gcs.addPathLengthCost(cost_weights["path_length"])
            if verbose:
                print(f"[Planner]   Path length cost weight: {cost_weights['path_length']}")

        if "energy" in cost_weights and cost_weights["energy"] > 0:
            bezier_gcs.addPathEnergyCost(cost_weights["energy"])
            if verbose:
                print(f"[Planner]   Energy cost weight: {cost_weights['energy']}")

        # [2025-04-06] 曲率成本功能暂时禁用 - 开始
        # # 新增：添加曲率惩罚成本
        # curvature_weights = CurvatureCostWeights(
        #     curvature_squared=cost_weights.get("curvature_squared", 0.0),
        #     curvature_derivative=cost_weights.get("curvature_derivative", 0.0),
        #     curvature_peak=cost_weights.get("curvature_peak", 0.0)
        # )
        #
        # if curvature_weights.is_enabled():
        #     if verbose:
        #         print("[Planner] Adding curvature penalty cost...")
        #
        #     curvature_cost_config = CurvatureCostConfig()
        #     curvature_cost_module = CurvatureCostModule(curvature_cost_config)
        #     curvature_cost_module.add_curvature_cost_to_gcs(
        #         bezier_gcs, curvature_weights, verbose
        #     )
        # [2025-04-06] 曲率成本功能暂时禁用 - 结束

        # 步骤6/7：求解轨迹
        # 当曲率硬约束启用时，曲率已由Lorentz锥凸约束保证，
        # 无需SCP迭代线性化曲率约束，直接用GCS凸松弛求解
        use_curvature_hard_constraint = (
            constraints.enable_curvature_hard_constraint or
            constraints.curvature_constraint_mode == "hard"
        )

        if use_curvature_hard_constraint:
            # 曲率硬约束路径：多次GCS求解，选择曲率违反最小的轨迹
            # GCS rounding具有随机性，多次求解可提高可行率
            max_solve_attempts = 3
            if verbose:
                print(f"[Planner] 曲率硬约束已启用，跳过SCP迭代，"
                      f"GCS求解（最多{max_solve_attempts}次尝试）...")

            best_trajectory = None
            best_curvature_violation = np.inf
            best_attempt = 0

            for attempt in range(max_solve_attempts):
                result = bezier_gcs.SolvePathWithConstraints(
                    rounding=True, preprocessing=True, verbose=(verbose and attempt == 0)
                )
                trajectory_candidate = result[0] if result is not None else None

                if trajectory_candidate is None:
                    continue

                # 评估曲率违反量
                try:
                    report = self.evaluator.evaluate_trajectory(
                        trajectory_candidate, constraints
                    )
                    curv_viol = 0.0
                    if report.curvature_violation:
                        curv_viol = report.curvature_violation.max_violation
                except Exception:
                    curv_viol = np.inf

                if verbose and attempt > 0:
                    print(f"  尝试 {attempt+1}: κ_viol={curv_viol:.6f}")

                # 选择曲率违反最小的轨迹
                if curv_viol < best_curvature_violation:
                    best_curvature_violation = curv_viol
                    best_trajectory = trajectory_candidate
                    best_attempt = attempt + 1

                # 如果已找到可行解，提前退出
                if best_curvature_violation < 1e-4:
                    break

            trajectory = best_trajectory
            converged = trajectory is not None

            if verbose:
                if trajectory is not None:
                    print(f"[Planner] ✓ GCS求解成功（尝试{best_attempt}/{max_solve_attempts}，"
                          f"κ_viol={best_curvature_violation:.6f}）")
                else:
                    print("[Planner] ✗ GCS求解失败")
        else:
            # 标准路径：SCP迭代求解
            if verbose:
                print("[Planner] Initializing AckermannSCPSolver...")

            scp_solver = AckermannSCPSolver(
                bezier_gcs=bezier_gcs,
                vehicle_params=self.vehicle_params,
                scp_config=self.scp_config,
                constraints=constraints,
            )

            if verbose:
                print("[Planner] Solving with SCP...")

            trajectory, converged = scp_solver.solve(verbose=verbose)

        if trajectory is None:
            solve_time = time.time() - start_time
            if verbose:
                print(f"[Planner] Failed to solve trajectory!")
                print(f"[Planner] Total solve time: {solve_time:.2f}s")

            return PlanningResult(
                success=False,
                trajectory=None,
                trajectory_report=None,
                solve_time=solve_time,
                num_iterations=0 if use_curvature_hard_constraint else scp_solver.iteration,
                convergence_reason="solve_failed",
                error_message="Failed to solve trajectory",
            )

        # 步骤8：评估轨迹
        if verbose:
            print("[Planner] Evaluating trajectory...")

        trajectory_report = self.evaluator.evaluate_trajectory(trajectory, constraints)

        # 步骤7.5：后验验证边界段曲率（如果启用了曲率硬约束）
        if constraints.enable_curvature_hard_constraint or \
           constraints.curvature_constraint_mode == "hard":
            if trajectory_report.curvature_violation and \
               trajectory_report.curvature_violation.is_violated:
                if verbose:
                    print("\n" + "=" * 70)
                    print("曲率后验验证")
                    print("=" * 70)
                    print(f"⚠️  曲率约束违反！最大违反量: "
                          f"{trajectory_report.curvature_violation.max_violation:.6f}")
                    print("可能原因：")
                    print("  1. 边界段（v=0附近）曲率超限")
                    print("  2. 保守性不足：min_velocity设置过低")
                    print("  3. 成本权重不当：w_time/w_energy比值过小导致速度过低")
                    print("调参建议：")
                    print("  - 增大 min_velocity（当前: {:.4f}）".format(
                        constraints.min_velocity))
                    print("  - 增大 w_time/w_energy 比值（推荐 0.5~2.5）")
                    print("  - 增大二阶正则化权重 w_reg2（推荐 5.0~10.0）")
                    print("  - 使用曲率约束预设: curvature_constrained")
                    print("=" * 70)

        # 新增：计算曲率统计
        if trajectory is not None:
            try:
                curvature_stats_module = CurvatureStatistics(num_samples=100)
                trajectory_report.curvature_stats = curvature_stats_module.compute_curvature_stats(trajectory)

                if verbose:
                    stats = trajectory_report.curvature_stats
                    print(f"[Planner] Curvature statistics:")
                    print(f"  Max curvature: {stats.max_curvature:.6f}")
                    print(f"  Mean curvature: {stats.mean_curvature:.6f}")
                    print(f"  Std curvature: {stats.std_curvature:.6f}")
            except Exception as e:
                if verbose:
                    print(f"[Planner] Warning: Failed to compute curvature stats: {e}")

        if verbose:
            print(f"[Planner] Trajectory feasible: {trajectory_report.is_feasible}")
            if trajectory_report.velocity_violation:
                print(
                    f"[Planner]   Velocity violation: {trajectory_report.velocity_violation.max_violation:.6f}"
                )
            if trajectory_report.acceleration_violation:
                print(
                    f"[Planner]   Acceleration violation: {trajectory_report.acceleration_violation.max_violation:.6f}"
                )
            if trajectory_report.curvature_violation:
                print(
                    f"[Planner]   Curvature violation: {trajectory_report.curvature_violation.max_violation:.6f}"
                )
            if trajectory_report.workspace_violation:
                print(
                    f"[Planner]   Workspace violation: {trajectory_report.workspace_violation.max_violation:.6f}"
                )

        # 步骤9：返回规划结果
        solve_time = time.time() - start_time
        num_iterations = 0 if use_curvature_hard_constraint else scp_solver.iteration

        if verbose:
            print(f"[Planner] Total solve time: {solve_time:.2f}s")
            if use_curvature_hard_constraint:
                print(f"[Planner] Solver: GCS convex relaxation (curvature hard constraint)")
            else:
                print(f"[Planner] SCP iterations: {num_iterations}")
            print(f"[Planner] Converged: {converged}")

        convergence_reason = "converged" if converged else "max_iterations_reached"

        return PlanningResult(
            success=trajectory_report.is_feasible,
            trajectory=trajectory,
            trajectory_report=trajectory_report,
            solve_time=solve_time,
            num_iterations=num_iterations,
            convergence_reason=convergence_reason,
            error_message="",
        )
