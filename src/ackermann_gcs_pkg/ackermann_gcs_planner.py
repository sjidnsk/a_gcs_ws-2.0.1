"""
阿克曼转向车辆GCS规划器

本模块实现了阿克曼转向车辆的GCS轨迹规划器，整合所有子模块。
使用曲率硬约束（Lorentz锥凸约束）作为唯一的曲率约束实现方式。
"""

import time
import numpy as np
from typing import Optional, List

from pydrake.geometry.optimization import HPolyhedron
from pydrake.trajectories import BsplineTrajectory

from .ackermann_bezier_gcs import AckermannBezierGCS
from .ackermann_data_structures import (
    VehicleParams,
    EndpointState,
    TrajectoryConstraints,
    BezierConfig,
    PlanningResult,
)
from .trajectory_evaluator import TrajectoryEvaluator
from .curvature_statistics import CurvatureStatistics
from .h_bar_prime_iteration import iterate_h_bar_prime, HBarPrimeIterationResult


class AckermannGCSPlanner:
    """
    阿克曼转向车辆GCS规划器

    整合平坦输出映射器、AckermannBezierGCS、轨迹评估器，提供完整的轨迹规划功能。
    使用曲率硬约束（Lorentz锥凸约束）保证曲率约束满足。
    """

    def __init__(
        self,
        vehicle_params: VehicleParams,
        bezier_config: Optional[BezierConfig] = None,
    ):
        """
        初始化规划器

        Args:
            vehicle_params: 车辆参数
            bezier_config: 贝塞尔曲线配置，如果为None则使用默认配置
        """
        # 参数验证
        if not isinstance(vehicle_params, VehicleParams):
            raise TypeError(f"vehicle_params must be VehicleParams instance, got {type(vehicle_params)}")

        self.vehicle_params = vehicle_params
        self.bezier_config = bezier_config if bezier_config is not None else BezierConfig()

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
        4. 添加凸约束（速度、曲率硬约束）
        5. 添加成本函数（时间、路径长度、能量）
        6. GCS求解（曲率硬约束，多次舍入尝试）
        7. 评估轨迹
        8. 返回规划结果

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
            curvature_constraint_version=constraints.curvature_constraint_version,
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

        # 步骤4.5：添加曲率硬约束
        h_bar_prime_iteration_result = None
        if constraints.enable_curvature_hard_constraint or \
           constraints.curvature_constraint_mode == "hard":

            # v2模式：旋转二阶锥速度耦合方案
            if getattr(constraints, 'curvature_constraint_version', 'v1') == 'v2':
                if verbose:
                    print("\n" + "=" * 70)
                    print("曲率硬约束v2添加（旋转二阶锥速度耦合）")
                    print("=" * 70)
                    print(f"约束类型: 旋转二阶锥 + 线性速度下界")
                    print(f"约束形式: A1:q_i·d_θ≥σ_e, A2:τ_e≥σ_e², B:κ_max·τ_e≥‖Q_j‖, C:σ_e≥σ_min")
                    print(f"最大曲率: {constraints.max_curvature:.4f} 1/m")
                    print(f"sigma_min: {constraints.sigma_min}")
                    print(f"凸性保证: SOCP（旋转锥+Lorentz锥+线性）")
                    print(f"保守性: 消除(v_max/v_min)²因子")
                    print("=" * 70)

                try:
                    result_v2 = bezier_gcs.addCurvatureHardConstraintV2(
                        max_curvature=constraints.max_curvature,
                        sigma_min=constraints.sigma_min,
                    )
                    if verbose:
                        if result_v2 is not None:
                            print(f"✓ v2曲率约束添加完成: {result_v2.summary()}")
                        else:
                            print("⚠️  v2前提不满足，已回退到v1")
                except Exception as e:
                    if verbose:
                        print(f"⚠️  v2曲率约束添加失败: {e}")
                        print("   将跳过曲率硬约束")

            else:
                # v1模式：原有Lorentz锥方案
                if verbose:
                    print("\n" + "=" * 70)
                    print("曲率硬约束添加")
                    print("=" * 70)
                    print(f"约束类型: 凸硬约束（Lorentz锥）")
                    print(f"约束形式: ||Q_j||_2 <= C = kappa_max * rho_min^2")
                    print(f"最大曲率: {constraints.max_curvature:.4f} 1/m")
                    print(f"最小速度: {constraints.min_velocity:.4f} m/s")
                    print(f"h_bar_prime: {constraints.h_bar_prime}")
                    print(f"safety_factor: {constraints.h_bar_prime_safety_factor}")
                    print(f"max_iterations: {constraints.max_h_bar_prime_iterations}")
                    print(f"凸性保证: 二阶锥约束（Lorentz锥）")
                    print(f"保守性: Cauchy-Schwarz + 凸包 + 速度下界")
                    if constraints.min_velocity > 0:
                        alpha = (constraints.max_velocity / constraints.min_velocity) ** 2
                        print(f"保守因子 alpha: {alpha:.1f}")
                    print("=" * 70)

                # 判断是否使用迭代修正模式
                use_iteration = (
                    constraints.h_bar_prime is None
                    and constraints.max_h_bar_prime_iterations > 1
                )

                if use_iteration:
                    # 迭代修正模式：先求解无曲率约束，再迭代修正 h̄'
                    if verbose:
                        print("[Planner] Using iterative h̄' refinement mode")

                    # 先添加成本函数（迭代修正需要）
                    if cost_weights is None:
                        cost_weights = {"time": 1.0, "path_length": 0.1, "energy": 0.01}

                    if "time" in cost_weights and cost_weights["time"] > 0:
                        bezier_gcs.addTimeCost(cost_weights["time"])
                    if "path_length" in cost_weights and cost_weights["path_length"] > 0:
                        bezier_gcs.addPathLengthCost(cost_weights["path_length"])
                    if "energy" in cost_weights and cost_weights["energy"] > 0:
                        bezier_gcs.addPathEnergyCost(cost_weights["energy"])
                    if "time_derivative_reg" in cost_weights and cost_weights["time_derivative_reg"] > 0:
                        h_ref = cost_weights.get("h_ref", None)
                        bezier_gcs.addTimeDerivativeRegularization(
                            cost_weights["time_derivative_reg"], h_ref=h_ref
                        )
                    if "regularization_r" in cost_weights and cost_weights["regularization_r"] > 0:
                        weight_r = cost_weights["regularization_r"]
                        weight_h = cost_weights.get("regularization_h", 0.0)
                        reg_order = cost_weights.get("regularization_order", 2)
                        bezier_gcs.addDerivativeRegularization(weight_r, weight_h, reg_order)

                    # 执行迭代修正
                    iter_traj, h_bar_prime_iteration_result = iterate_h_bar_prime(
                        bezier_gcs=bezier_gcs,
                        constraints=constraints,
                        cost_weights=cost_weights,
                        source=source,
                        target=target,
                        workspace_regions=workspace_regions,
                        verbose=verbose,
                    )

                    if iter_traj is not None:
                        if verbose:
                            hbp_res = h_bar_prime_iteration_result
                            rho_min = constraints.min_velocity * hbp_res.effective_h_bar_prime
                            C = constraints.max_curvature * rho_min ** 2
                            print(f"✓ 迭代修正完成: h̄'={hbp_res.h_bar_prime:.6f}, "
                                  f"effective={hbp_res.effective_h_bar_prime:.6f}, "
                                  f"C={C:.6f}")
                    else:
                        if verbose:
                            print("⚠️  迭代修正求解失败，将跳过曲率硬约束")

                else:
                    # 直接模式：使用指定 h̄' 或默认值 1.0
                    try:
                        bezier_gcs.addCurvatureHardConstraint(
                            max_curvature=constraints.max_curvature,
                            min_velocity=constraints.min_velocity,
                            h_bar_prime=constraints.h_bar_prime,
                            h_bar_prime_safety_factor=constraints.h_bar_prime_safety_factor,
                        )
                        if verbose:
                            hbp = constraints.h_bar_prime if constraints.h_bar_prime is not None else 1.0
                            effective = hbp * constraints.h_bar_prime_safety_factor
                            rho_min = constraints.min_velocity * effective
                            C = constraints.max_curvature * rho_min ** 2
                            print(f"rho_min: {rho_min:.4f}")
                            print(f"约束阈值 C: {C:.6f}")
                            print(f"effective h̄': {effective:.6f}")
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

        if "time_derivative_reg" in cost_weights and cost_weights["time_derivative_reg"] > 0:
            h_ref = cost_weights.get("h_ref", None)
            bezier_gcs.addTimeDerivativeRegularization(
                cost_weights["time_derivative_reg"], h_ref=h_ref
            )
            if verbose:
                h_ref_str = f", h_ref={h_ref}" if h_ref is not None else ""
                print(
                    f"[Planner]   Time derivative regularization weight: "
                    f"{cost_weights['time_derivative_reg']}{h_ref_str}"
                )

        # 空间/时间二阶导数正则化：直接惩罚||r''(s)||²和||h''(s)||²
        if "regularization_r" in cost_weights and cost_weights["regularization_r"] > 0:
            weight_r = cost_weights["regularization_r"]
            weight_h = cost_weights.get("regularization_h", 0.0)
            reg_order = cost_weights.get("regularization_order", 2)
            bezier_gcs.addDerivativeRegularization(weight_r, weight_h, reg_order)
            if verbose:
                print(
                    f"[Planner]   Derivative regularization: "
                    f"r={weight_r}, h={weight_h}, order={reg_order}"
                )

        # 步骤6：GCS求解（曲率硬约束，多次舍入尝试）
        # GCS rounding具有随机性，多次求解可提高可行率

        # 设置自定义舍入策略：随机前向+随机后向（探索更多候选路径）
        from gcs_pkg.scripts.rounding import (
            randomForwardPathSearch,
            randomBackwardPathSearch,
        )
        from config.solver.mosek_opt_config import MosekOptimizationConfig
        _mosek_opt_config = MosekOptimizationConfig()
        bezier_gcs.setRoundingStrategy(
            [randomForwardPathSearch, randomBackwardPathSearch],
            flow_tol=1e-5,
            max_paths=_mosek_opt_config.effective_max_paths(),
            max_trials=100,
        )

        max_solve_attempts = self.bezier_config.max_rounding_attempts
        max_rounded_paths = self.bezier_config.max_rounded_paths
        if verbose:
            print(f"[Planner] GCS求解（最多{max_solve_attempts}次尝试，"
                  f"每次{max_rounded_paths}条舍入路径）...")

        best_trajectory = None
        best_velocity_violation = np.inf
        best_curvature_violation = np.inf
        best_combined_violation = np.inf
        best_attempt = 0

        def _evaluate_trajectory_violations(traj):
            """评估轨迹的约束违反量，返回 (vel, curv, acc, combined)。"""
            try:
                report = self.evaluator.evaluate_trajectory(
                    traj, constraints
                )
                vel_viol = 0.0
                curv_viol = 0.0
                acc_viol = 0.0
                if report.velocity_violation:
                    vel_viol = report.velocity_violation.max_violation
                if report.curvature_violation:
                    curv_viol = report.curvature_violation.max_violation
                if report.acceleration_violation:
                    acc_viol = report.acceleration_violation.max_violation
                combined_viol = vel_viol * 10.0 + curv_viol + acc_viol * 5.0
                return vel_viol, curv_viol, acc_viol, combined_viol
            except Exception:
                return np.inf, np.inf, np.inf, np.inf

        for attempt in range(max_solve_attempts):
            # 配置舍入路径数量
            bezier_gcs.options.max_rounded_paths = max_rounded_paths

            result = bezier_gcs.SolvePathWithConstraints(
                rounding=True, preprocessing=True, verbose=(verbose and attempt == 0)
            )
            trajectory_candidate = result[0] if result is not None else None

            if trajectory_candidate is None:
                continue

            # 评估最优轨迹（成本最低的舍入路径）
            vel_viol, curv_viol, acc_viol, combined_viol = \
                _evaluate_trajectory_violations(trajectory_candidate)

            if verbose and attempt > 0:
                print(f"  尝试 {attempt+1}: "
                      f"v_viol={vel_viol:.6f}, "
                      f"κ_viol={curv_viol:.6f}, "
                      f"a_viol={acc_viol:.6f}")

            # 选择综合违反量最小的轨迹
            if combined_viol < best_combined_violation:
                best_velocity_violation = vel_viol
                best_curvature_violation = curv_viol
                best_combined_violation = combined_viol
                best_trajectory = trajectory_candidate
                best_attempt = attempt + 1

            # 如果速度、曲率和加速度均可行，提前退出
            if (best_velocity_violation < 1e-4
                    and best_curvature_violation < 1e-4
                    and acc_viol < 1e-4):
                break

            # 在所有候选舍入轨迹中筛选违反量更小的
            results_dict = result[1] if len(result) > 1 else {}
            all_candidates = results_dict.get(
                "all_candidate_trajectories", [])
            if len(all_candidates) > 1 and best_combined_violation >= 1e-4:
                if verbose:
                    print(f"  筛选 {len(all_candidates)} 条候选轨迹...")
                for idx, cand_traj in enumerate(all_candidates):
                    if cand_traj is trajectory_candidate:
                        continue  # 已评估过
                    cv, ccv, cav, ccv_comb = \
                        _evaluate_trajectory_violations(cand_traj)
                    if ccv_comb < best_combined_violation:
                        best_velocity_violation = cv
                        best_curvature_violation = ccv
                        best_combined_violation = ccv_comb
                        best_trajectory = cand_traj
                        best_attempt = attempt + 1
                        if verbose:
                            print(f"    候选{idx}: "
                                  f"v={cv:.6f}, κ={ccv:.6f}, "
                                  f"a={cav:.6f} → 更优")
                    if (best_velocity_violation < 1e-4
                            and best_curvature_violation < 1e-4
                            and best_combined_violation < 1e-4):
                        break
                if (best_velocity_violation < 1e-4
                        and best_curvature_violation < 1e-4):
                    break

        trajectory = best_trajectory
        converged = trajectory is not None

        if verbose:
            if trajectory is not None:
                print(f"[Planner] ✓ GCS求解成功"
                      f"（尝试{best_attempt}/{max_solve_attempts}，"
                      f"v_viol={best_velocity_violation:.6f}，"
                      f"κ_viol={best_curvature_violation:.6f}）")
                if best_velocity_violation >= 1e-4:
                    print(f"[Planner] ⚠ 速度约束仍有违反: "
                          f"{best_velocity_violation:.6f} m/s")
            else:
                print("[Planner] ✗ GCS求解失败")

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
                num_iterations=0,
                convergence_reason="solve_failed",
                error_message="Failed to solve trajectory",
            )

        # 步骤7：评估轨迹
        if verbose:
            print("[Planner] Evaluating trajectory...")

        trajectory_report = self.evaluator.evaluate_trajectory(trajectory, constraints)

        # 后验验证边界段曲率（如果启用了曲率硬约束）
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

        # 计算曲率统计
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

        # 步骤8：返回规划结果
        solve_time = time.time() - start_time

        if verbose:
            print(f"[Planner] Total solve time: {solve_time:.2f}s")
            print(f"[Planner] Solver: GCS convex relaxation (curvature hard constraint)")
            print(f"[Planner] Converged: {converged}")

        convergence_reason = "converged" if converged else "max_iterations_reached"

        return PlanningResult(
            success=trajectory_report.is_feasible,
            trajectory=trajectory,
            trajectory_report=trajectory_report,
            solve_time=solve_time,
            num_iterations=0,
            convergence_reason=convergence_reason,
            error_message="",
        )
