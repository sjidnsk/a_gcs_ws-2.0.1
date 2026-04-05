"""
阿克曼转向车辆SCP求解器

本模块实现了序列凸规划（SCP）求解器，用于处理非凸的曲率约束。
优化版本包含：动态信任区域调整、多层次提前终止、并行化计算。
"""

import numpy as np
import time
from typing import List, Tuple, Optional

from pydrake.solvers import (
    Binding,
    Constraint,
    LinearConstraint,
    LorentzConeConstraint,
)
from pydrake.symbolic import DecomposeLinearExpressions, Expression
from pydrake.trajectories import BsplineTrajectory

from .ackermann_bezier_gcs import AckermannBezierGCS
from .ackermann_data_structures import (
    VehicleParams,
    SCPConfig,
    TrustRegionConfig,
    TerminationConfig,
    ParallelConfig,
    TerminationReason,
    TrajectoryConstraints,
)

# 导入优化模块
from .scp_optimization import (
    TrustRegionManager,
    EarlyTerminationChecker,
    ParallelCurvatureLinearizer,
    PerformanceStats,
    ConstraintViolationCalculator,
)


class AckermannSCPSolver:
    """
    序列凸规划（SCP）求解器

    通过迭代线性化曲率约束，将非凸约束转化为凸约束，逐步逼近最优解。
    """

    def __init__(
        self,
        bezier_gcs: AckermannBezierGCS,
        vehicle_params: VehicleParams,
        scp_config: Optional[SCPConfig] = None,
        constraints: Optional[TrajectoryConstraints] = None,  # 新增参数
    ):
        """
        初始化SCP求解器

        Args:
            bezier_gcs: AckermannBezierGCS实例
            vehicle_params: 车辆参数
            scp_config: SCP配置，如果为None则使用默认配置
            constraints: 轨迹约束，如果为None则从vehicle_params推导（新增）
        """
        # 参数验证
        if not isinstance(bezier_gcs, AckermannBezierGCS):
            raise TypeError(f"bezier_gcs must be AckermannBezierGCS instance, got {type(bezier_gcs)}")

        # 使用默认配置
        if scp_config is None:
            scp_config = SCPConfig()

        self.bezier_gcs = bezier_gcs
        self.vehicle_params = vehicle_params
        self.scp_config = scp_config

        # 初始化当前轨迹和迭代计数
        self.current_trajectory: Optional[BsplineTrajectory] = None
        self.iteration = 0

        # 初始化优化模块
        self.trust_manager: Optional[TrustRegionManager] = None
        self.termination_checker: Optional[EarlyTerminationChecker] = None
        self.parallel_linearizer: Optional[ParallelCurvatureLinearizer] = None
        self.performance_stats: Optional[PerformanceStats] = None
        self.violation_calculator: Optional[ConstraintViolationCalculator] = None  # 新增

        # 根据配置初始化优化模块
        if scp_config.enable_optimizations:
            # 信任区域管理器
            if scp_config.enable_dynamic_trust_region:
                trust_config = scp_config.trust_region_config or TrustRegionConfig()
                self.trust_manager = TrustRegionManager(trust_config)

            # 提前终止检查器
            if scp_config.enable_early_termination:
                term_config = scp_config.termination_config or TerminationConfig()
                self.termination_checker = EarlyTerminationChecker(term_config)

            # 并行线性化器
            if scp_config.enable_parallel_linearization:
                parallel_config = scp_config.parallel_config or ParallelConfig()
                self.parallel_linearizer = ParallelCurvatureLinearizer(
                    parallel_config, vehicle_params, bezier_gcs
                )

            # 性能统计
            if scp_config.enable_performance_stats:
                num_processes = (
                    parallel_config.num_processes if scp_config.parallel_config
                    else None
                )
                self.performance_stats = PerformanceStats(num_processes)

            # 约束违反量计算器（新增）
            if constraints is None:
                constraints = TrajectoryConstraints(
                    max_velocity=vehicle_params.max_velocity,
                    max_acceleration=vehicle_params.max_acceleration,
                    max_curvature=vehicle_params.max_curvature,
                    workspace_regions=None,
                )

            # 创建轨迹评估器和约束违反量计算器
            from .trajectory_evaluator import TrajectoryEvaluator
            evaluator = TrajectoryEvaluator(vehicle_params, constraints)
            self.violation_calculator = ConstraintViolationCalculator(
                evaluator=evaluator,
                constraints=constraints,
                num_samples=100
            )

    def _linearize_curvature_constraint(
        self,
        trajectory: BsplineTrajectory,
        num_samples: int = 50,
    ) -> List[LinearConstraint]:
        """
        线性化曲率约束

        对曲率约束进行泰勒展开线性化：
        κ(s) ≈ κ₀(s) + ∇κ(s) · Δz(s)

        构建线性约束：
        κ₀ + ∇κ · Δz ≤ κ_max
        κ₀ + ∇κ · Δz ≥ -κ_max

        Args:
            trajectory: 当前轨迹
            num_samples: 采样点数

        Returns:
            线性约束列表
        """
        # 采样时间点
        t_samples = np.linspace(trajectory.start_time(), trajectory.end_time(), num_samples)

        # 计算当前曲率κ₀和梯度∇κ
        curvature_list = []
        grad_x_dot_list = []
        grad_y_dot_list = []
        grad_x_ddot_list = []
        grad_y_ddot_list = []

        for t in t_samples:
            # 计算一阶和二阶导数
            first_deriv = trajectory.EvalDerivative(t, 1)  # (2,)
            second_deriv = trajectory.EvalDerivative(t, 2)  # (2,)

            x_dot = first_deriv[0]
            y_dot = first_deriv[1]
            x_ddot = second_deriv[0]
            y_ddot = second_deriv[1]

            # 计算当前曲率κ₀
            denominator = (x_dot**2 + y_dot**2) ** 1.5
            epsilon = 1e-10
            if denominator < epsilon:
                denominator = 1.0
            numerator = x_dot * y_ddot - y_dot * x_ddot
            curvature = numerator / denominator
            curvature_list.append(curvature)

            # 计算曲率梯度∇κ = [∂κ/∂x', ∂κ/∂y', ∂κ/∂x'', ∂κ/∂y'']
            # ∂κ/∂x' = (3 * x' * y' * (y' * x'' - x' * y'')) / (x'^2 + y'^2)^(5/2)
            # ∂κ/∂y' = (3 * x' * y' * (x' * y'' - y' * x'')) / (x'^2 + y'^2)^(5/2)
            # ∂κ/∂x'' = -y' / (x'^2 + y'^2)^(3/2)
            # ∂κ/∂y'' = x' / (x'^2 + y'^2)^(3/2)

            denom_1_5 = (x_dot**2 + y_dot**2) ** 2.5
            if denom_1_5 < epsilon:
                denom_1_5 = 1.0
            denom_0_5 = (x_dot**2 + y_dot**2) ** 0.5
            if denom_0_5 < epsilon:
                denom_0_5 = 1.0

            dk_dx_dot = (3 * x_dot * y_dot * (y_dot * x_ddot - x_dot * y_ddot)) / denom_1_5
            dk_dy_dot = (3 * x_dot * y_dot * (x_dot * y_ddot - y_dot * x_ddot)) / denom_1_5
            dk_dx_ddot = -y_dot / denom_0_5
            dk_dy_ddot = x_dot / denom_0_5

            grad_x_dot_list.append(dk_dx_dot)
            grad_y_dot_list.append(dk_dy_dot)
            grad_x_ddot_list.append(dk_dx_ddot)
            grad_y_ddot_list.append(dk_dy_ddot)

        # 构建线性约束
        constraints = []
        u_vars = self.bezier_gcs.u_vars
        u_r_trajectory = self.bezier_gcs.u_r_trajectory

        for i in range(num_samples):
            t = t_samples[i]
            kappa_0 = curvature_list[i]
            grad_x_dot = grad_x_dot_list[i]
            grad_y_dot = grad_y_dot_list[i]
            grad_x_ddot = grad_x_ddot_list[i]
            grad_y_ddot = grad_y_ddot_list[i]

            # 计算控制点对导数的贡献
            # dq/ds = sum(B_i'(s) * P_i)
            first_deriv_at_s = u_r_trajectory.MakeDerivative(1).value(t)
            second_deriv_at_s = u_r_trajectory.MakeDerivative(2).value(t)

            # 构建线性表达式：κ₀ + ∇κ · Δz
            # Δz = [Δx', Δy', Δx'', Δy'']
            delta_x_dot = first_deriv_at_s[0]
            delta_y_dot = first_deriv_at_s[1]
            delta_x_ddot = second_deriv_at_s[0]
            delta_y_ddot = second_deriv_at_s[1]

            # 只构建线性部分（不包含常数项kappa_0）
            linear_expr = (
                grad_x_dot * delta_x_dot
                + grad_y_dot * delta_y_dot
                + grad_x_ddot * delta_x_ddot
                + grad_y_ddot * delta_y_ddot
            )

            # 分解线性表达式
            A = DecomposeLinearExpressions(linear_expr, u_vars)
            
            # 常数项kappa_0需要在约束边界中考虑
            # 上界约束：κ₀ + ∇κ · Δz ≤ κ_max  =>  ∇κ · Δz ≤ κ_max - κ₀
            constraints.append(
                LinearConstraint(
                    A,
                    np.array([-np.inf]),
                    np.array([self.vehicle_params.max_curvature - kappa_0]),
                )
            )
            # 下界约束：κ₀ + ∇κ · Δz ≥ -κ_max  =>  ∇κ · Δz ≥ -κ_max - κ₀
            constraints.append(
                LinearConstraint(
                    A,
                    np.array([-self.vehicle_params.max_curvature - kappa_0]),
                    np.array([np.inf]),
                )
            )

        return constraints

    def _build_trust_region_constraint(
        self,
        delta: float,
    ) -> LorentzConeConstraint:
        """
        构建信任区域约束

        约束形式：||[Δx', Δy', Δx'', Δy'']||₂ ≤ delta

        使用Drake的LorentzConeConstraint实现。
        LorentzConeConstraint定义：||Ax + b||₂ ≤ x[0]
        我们需要构造：||Δx||₂ ≤ delta
        通过添加一个虚拟变量来实现。

        Args:
            delta: 信任区域半径

        Returns:
            LorentzConeConstraint
        """
        # 获取控制点变量
        u_vars = self.bezier_gcs.u_vars
        num_control_points = self.bezier_gcs.order + 1
        
        # LorentzConeConstraint: ||Ax + b||₂ ≤ x[0]
        # 我们需要：||Δx||₂ ≤ delta
        # 构造：||[Δx₁, Δx₂, ..., Δxₙ]||₂ ≤ delta
        # 这可以通过设置 A = [I], b = [0], 然后约束 x[0] = delta 来实现
        # 但更简单的方法是直接约束控制点的范数
        
        # 构建A矩阵：选择所有控制点
        # A的形状应该是 (m, n)，其中m是约束维度，n是变量数量
        # 我们约束所有控制点的变化
        A = np.eye(len(u_vars))
        
        # b向量：偏移量
        b = np.zeros(len(u_vars))
        
        # 构建LorentzConeConstraint
        # ||Ax + b||_2 <= x[0]
        # 这意味着我们需要第一个变量x[0] >= ||x||_2
        # 但我们想要的是 ||x||_2 <= delta
        # 所以我们需要一个不同的方法
        
        # 实际上，LorentzConeConstraint的定义是：
        # x ∈ LorentzCone <=> x[0] >= sqrt(x[1]^2 + ... + x[n]^2)
        # 即 x[0] >= ||x[1:]||_2
        
        # 为了实现 ||Δx||_2 <= delta，我们需要：
        # delta >= ||Δx||_2
        # 这等价于 [delta, Δx₁, Δx₂, ..., Δxₙ] ∈ LorentzCone
        
        # 构建A矩阵：[0, I]，使得 Ax + b = [0, Δx₁, Δx₂, ..., Δxₙ]
        A = np.zeros((len(u_vars) + 1, len(u_vars)))
        A[1:, :] = np.eye(len(u_vars))
        
        # b向量：[delta, 0, 0, ..., 0]
        b = np.zeros(len(u_vars) + 1)
        b[0] = delta
        
        # 构建LorentzConeConstraint
        constraint = LorentzConeConstraint(A, b)

        return constraint

    def _compute_curvature_violation(
        self,
        trajectory: BsplineTrajectory,
        num_samples: int = 100,
    ) -> float:
        """
        计算曲率约束的最大违反量

        Args:
            trajectory: 轨迹
            num_samples: 采样点数

        Returns:
            最大违反量
        """
        # 采样时间点
        t_samples = np.linspace(trajectory.start_time(), trajectory.end_time(), num_samples)

        # 计算曲率
        curvature_list = []
        for t in t_samples:
            # 计算一阶和二阶导数
            first_deriv = trajectory.EvalDerivative(t, 1)  # (2,)
            second_deriv = trajectory.EvalDerivative(t, 2)  # (2,)

            x_dot = first_deriv[0]
            y_dot = first_deriv[1]
            x_ddot = second_deriv[0]
            y_ddot = second_deriv[1]

            # 计算曲率
            denominator = (x_dot**2 + y_dot**2) ** 1.5
            epsilon = 1e-10
            if denominator < epsilon:
                denominator = 1.0
            numerator = x_dot * y_ddot - y_dot * x_ddot
            curvature = numerator / denominator
            curvature_list.append(curvature)

        curvature = np.array(curvature_list)

        # 计算违反量：max(0, |κ| - κ_max)
        violation = np.maximum(0, np.abs(curvature) - self.vehicle_params.max_curvature)

        return np.max(violation)

    def solve(
        self,
        verbose: bool = True,
    ) -> Tuple[Optional[BsplineTrajectory], bool]:
        """
        SCP迭代求解（优化版本）

        执行以下步骤：
        1. 求解初始轨迹（忽略曲率约束）
        2. 初始化信任区域半径和优化模块
        3. SCP迭代（最多max_iterations次）：
           - 计算曲率违反量
           - 提前终止检查（如果启用）
           - 线性化曲率约束（并行或串行）
           - 构建信任区域约束
           - 求解GCS优化问题
           - 计算改进率和改进量
           - 动态调整信任区域（如果启用）
           - 记录性能统计
        4. 生成性能报告
        5. 返回最终轨迹和收敛标志

        Args:
            verbose: 是否打印迭代日志

        Returns:
            (trajectory, converged): 轨迹和收敛标志
        """
        start_time = time.time()
        
        # 启动性能统计
        if self.performance_stats:
            self.performance_stats.start()

        # 步骤1：求解初始轨迹（忽略曲率约束）
        if verbose:
            print("[SCP] Solving initial trajectory (ignoring curvature constraint)...")

        # 使用SolvePathWithConstraints以应用航向角约束
        initial_result = self.bezier_gcs.SolvePathWithConstraints(rounding=True, preprocessing=True, verbose=verbose)

        if initial_result is None or initial_result[0] is None:
            if verbose:
                print("[SCP] Failed to solve initial trajectory!")
            return None, False

        # SolvePath返回(path_traj, time_traj)元组
        self.current_trajectory = initial_result[0]  # path_traj
        self.iteration = 0

        # 步骤2：初始化信任区域半径
        delta = self.scp_config.initial_trust_region_radius
        
        # 初始化改进率
        improvement_ratio = 0.0
        previous_violation = 0.0
        termination_reason = TerminationReason.CONTINUE

        # 步骤3：SCP迭代
        while self.iteration < self.scp_config.max_iterations:
            iter_start_time = time.time()
            self.iteration += 1

            if verbose:
                print(f"[SCP] Iteration {self.iteration}/{self.scp_config.max_iterations}")

            # 计算所有约束违反量（改进）
            if self.violation_calculator:
                violation_report = self.violation_calculator.compute_all_violations(
                    self.current_trajectory
                )
                curvature_violation = violation_report.max_violation  # 使用最大违反量

                if verbose:
                    # 输出所有约束违反量（改进）
                    print(f"[SCP]   Velocity violation: {violation_report.velocity_violation:.6f}")
                    print(f"[SCP]   Acceleration violation: {violation_report.acceleration_violation:.6f}")
                    print(f"[SCP]   Curvature violation: {violation_report.curvature_violation:.6f}")
                    if violation_report.workspace_violation > 0:
                        print(f"[SCP]   Workspace violation: {violation_report.workspace_violation:.6f}")

                    # 严重违反提示（新增）
                    if violation_report.severe_violation_type:
                        print(f"[SCP]   WARNING: Severe {violation_report.severe_violation_type} violation detected!")
            else:
                # 向后兼容：仅计算曲率违反量
                curvature_violation = self._compute_curvature_violation(self.current_trajectory)
                violation_report = None

                if verbose:
                    print(f"[SCP]   Curvature violation: {curvature_violation:.6f}")

            # 提前终止检查（改进）
            if self.termination_checker and violation_report:
                should_terminate, termination_reason = self.termination_checker.check_all_with_report(
                    violation_report,
                    improvement_ratio,
                    delta,
                    self.iteration,
                    self.scp_config.max_iterations
                )

                if should_terminate:
                    if verbose:
                        print(f"[SCP] Early termination: {termination_reason.value}")
                    break

            # 原始收敛检查（向后兼容）
            elif violation_report is None:
                if curvature_violation < self.scp_config.convergence_tolerance:
                    if verbose:
                        print(f"[SCP] Converged! Curvature violation < {self.scp_config.convergence_tolerance}")
                    termination_reason = TerminationReason.CONVERGED
                    break

            if delta < self.scp_config.min_trust_region_radius:
                if verbose:
                    print(f"[SCP] Trust region too small ({delta:.6f} < {self.scp_config.min_trust_region_radius})")
                termination_reason = TerminationReason.TRUST_REGION_EXHAUSTED
                break

            # 线性化曲率约束（并行或串行）
            if self.parallel_linearizer:
                # 使用并行线性化器
                linearized_constraints = self.parallel_linearizer.linearize_curvature_constraints(
                    self.current_trajectory
                )
            else:
                # 使用原始串行方法
                linearized_constraints = self._linearize_curvature_constraint(self.current_trajectory)

            # 构建信任区域约束
            trust_region_constraint = self._build_trust_region_constraint(delta)

            # 添加约束到GCS
            for constraint in linearized_constraints:
                self.bezier_gcs.addCustomConstraint(constraint)
            self.bezier_gcs.addCustomConstraint(trust_region_constraint)

            # 求解GCS优化问题
            if verbose:
                print(f"[SCP]   Solving GCS with trust region delta = {delta:.6f}...")

            # 第一次迭代时输出约束应用信息
            show_constraint_info = (self.iteration == 1) and verbose
            new_result = self.bezier_gcs.SolvePathWithConstraints(rounding=True, preprocessing=True, verbose=show_constraint_info)

            iter_solve_time = time.time() - iter_start_time
            
            if new_result is not None and new_result[0] is not None:
                # 计算改进量
                if self.violation_calculator:
                    new_violation_report = self.violation_calculator.compute_all_violations(new_result[0])
                    new_violation = new_violation_report.max_violation
                else:
                    new_violation = self._compute_curvature_violation(new_result[0])
                    new_violation_report = None

                improvement = curvature_violation - new_violation

                # 计算改进率
                if self.trust_manager:
                    improvement_ratio = self.trust_manager.compute_improvement_ratio(
                        improvement, curvature_violation
                    )
                else:
                    # 简单改进率计算
                    improvement_ratio = improvement / curvature_violation if curvature_violation > 0 else 0.0

                if verbose:
                    print(f"[SCP]   New curvature violation: {new_violation:.6f}")
                    print(f"[SCP]   Improvement: {improvement:.6f}, Ratio: {improvement_ratio:.3f}")

                # 动态信任区域调整（优化功能）
                if self.trust_manager and improvement > 0:
                    # 使用动态调整策略
                    adjustment_factor = self.trust_manager.adjust_trust_region(improvement_ratio)
                    delta *= adjustment_factor

                    # 记录改进
                    self.trust_manager.record_improvement(improvement, new_violation, delta)

                    if verbose:
                        print(f"[SCP]   Accepted! Adjusted trust region to {delta:.6f} (factor: {adjustment_factor:.2f})")

                    # 接受新解
                    self.current_trajectory = new_result[0]
                    previous_violation = curvature_violation

                elif improvement > 0:
                    # 原始策略：改进有效，接受新解并扩大信任区域
                    self.current_trajectory = new_result[0]
                    delta *= self.scp_config.trust_region_expand_factor
                    previous_violation = curvature_violation
                    if verbose:
                        print(f"[SCP]   Accepted! Expanding trust region to {delta:.6f}")
                else:
                    # 改进无效，缩小信任区域
                    delta *= self.scp_config.trust_region_shrink_factor
                    if verbose:
                        print(f"[SCP]   Rejected! Shrinking trust region to {delta:.6f}")

                # 记录性能统计（改进）
                if self.performance_stats:
                    # 提取约束违反量
                    vel_viol = violation_report.velocity_violation if violation_report else 0.0
                    accel_viol = violation_report.acceleration_violation if violation_report else 0.0
                    curv_viol = violation_report.curvature_violation if violation_report else curvature_violation
                    work_viol = violation_report.workspace_violation if violation_report else 0.0

                    self.performance_stats.record_iteration(
                        self.iteration,
                        curvature_violation,
                        improvement,
                        improvement_ratio,
                        delta,
                        iter_solve_time,
                        velocity_violation=vel_viol,
                        acceleration_violation=accel_viol,
                        curvature_violation=curv_viol,
                        workspace_violation=work_viol
                    )
            else:
                # 求解失败，缩小信任区域
                if verbose:
                    print(f"[SCP]   Solve failed! Shrinking trust region to {delta * self.scp_config.trust_region_shrink_factor:.6f}")
                delta *= self.scp_config.trust_region_shrink_factor

        # 达到最大迭代次数
        if self.iteration >= self.scp_config.max_iterations:
            termination_reason = TerminationReason.MAX_ITERATIONS
            if verbose:
                print(f"[SCP] Reached max iterations ({self.scp_config.max_iterations})")

        solve_time = time.time() - start_time
        
        # 生成性能报告
        if verbose and self.performance_stats:
            print("\n" + self.performance_stats.generate_report())
        elif verbose:
            print(f"[SCP] Total solve time: {solve_time:.2f}s")

        # 判断是否收敛
        converged = termination_reason in [
            TerminationReason.CONVERGED,
            TerminationReason.CONSTRAINT_SATISFIED
        ]
        
        return self.current_trajectory, converged
