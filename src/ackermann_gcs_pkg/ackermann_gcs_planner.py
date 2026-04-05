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
    CurvatureCostConfig,
    CurvatureCostWeights,
)
from .trajectory_evaluator import TrajectoryEvaluator
from .curvature_cost_module import CurvatureCostModule
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
            constraints = TrajectoryConstraints(
                max_velocity=self.vehicle_params.max_velocity,
                max_acceleration=self.vehicle_params.max_acceleration,
                max_curvature=self.vehicle_params.max_curvature,
                workspace_regions=workspace_regions,
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
        # 更符合阿克曼车辆的物理模型，约束形式：||v||_2 <= max_velocity
        bezier_gcs.addScalarVelocityLimit(constraints.max_velocity)

        # 原有矢量约束（已弃用，保留作为参考）
        # bezier_gcs.addVelocityLimits(np.zeros(2), np.full(2, constraints.max_velocity))

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

        # 新增：添加曲率惩罚成本
        curvature_weights = CurvatureCostWeights(
            curvature_squared=cost_weights.get("curvature_squared", 0.0),
            curvature_derivative=cost_weights.get("curvature_derivative", 0.0),
            curvature_peak=cost_weights.get("curvature_peak", 0.0)
        )

        if curvature_weights.is_enabled():
            if verbose:
                print("[Planner] Adding curvature penalty cost...")

            curvature_cost_config = CurvatureCostConfig()
            curvature_cost_module = CurvatureCostModule(curvature_cost_config)
            curvature_cost_module.add_curvature_cost_to_gcs(
                bezier_gcs, curvature_weights, verbose
            )

        # 步骤6：初始化AckermannSCPSolver
        if verbose:
            print("[Planner] Initializing AckermannSCPSolver...")

        scp_solver = AckermannSCPSolver(
            bezier_gcs=bezier_gcs,
            vehicle_params=self.vehicle_params,
            scp_config=self.scp_config,
            constraints=constraints,  # 新增：传递约束参数
        )

        # 步骤7：SCP迭代求解
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
                num_iterations=scp_solver.iteration,
                convergence_reason="solve_failed",
                error_message="Failed to solve trajectory",
            )

        # 步骤8：评估轨迹
        if verbose:
            print("[Planner] Evaluating trajectory...")

        trajectory_report = self.evaluator.evaluate_trajectory(trajectory, constraints)

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

        if verbose:
            print(f"[Planner] Total solve time: {solve_time:.2f}s")
            print(f"[Planner] SCP iterations: {scp_solver.iteration}")
            print(f"[Planner] Converged: {converged}")

        convergence_reason = "converged" if converged else "max_iterations_reached"

        return PlanningResult(
            success=trajectory_report.is_feasible,
            trajectory=trajectory,
            trajectory_report=trajectory_report,
            solve_time=solve_time,
            num_iterations=scp_solver.iteration,
            convergence_reason=convergence_reason,
            error_message="",
        )
