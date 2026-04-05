"""
约束违反量计算器

本模块实现统一的约束违反量计算接口，用于SCP求解器的收敛判断。
"""

from typing import Optional
from pydrake.trajectories import BsplineTrajectory

from ..trajectory_evaluator import TrajectoryEvaluator
from ..ackermann_data_structures import (
    TrajectoryConstraints,
    ViolationReport,
    ConstraintThresholds,
)


class ConstraintViolationCalculator:
    """
    约束违反量计算器

    统一计算所有约束类型的违反量，提供综合评估接口。
    复用TrajectoryEvaluator的约束检查方法。
    """

    def __init__(
        self,
        evaluator: TrajectoryEvaluator,
        constraints: TrajectoryConstraints,
        thresholds: Optional[ConstraintThresholds] = None,
        num_samples: int = 100
    ):
        """
        初始化计算器

        Args:
            evaluator: 轨迹评估器
            constraints: 轨迹约束
            thresholds: 约束阈值配置，None则使用默认值
            num_samples: 采样点数

        Raises:
            ValueError: 如果evaluator为None
        """
        if evaluator is None:
            raise ValueError("evaluator cannot be None")
        if constraints is None:
            raise ValueError("constraints cannot be None")

        self.evaluator = evaluator
        self.constraints = constraints
        self.thresholds = thresholds or ConstraintThresholds()
        self.num_samples = num_samples

    def compute_all_violations(
        self,
        trajectory: BsplineTrajectory
    ) -> ViolationReport:
        """
        计算所有约束违反量

        Args:
            trajectory: 轨迹对象

        Returns:
            ViolationReport: 约束违反量报告

        Raises:
            ValueError: 如果trajectory为None
        """
        if trajectory is None:
            raise ValueError("trajectory cannot be None")

        # 计算各约束违反量
        velocity_violation = self.compute_velocity_violation(trajectory)
        acceleration_violation = self.compute_acceleration_violation(trajectory)
        curvature_violation = self.compute_curvature_violation(trajectory)
        workspace_violation = self.compute_workspace_violation(trajectory)

        # 计算最大违反量
        max_violation = max(
            velocity_violation,
            acceleration_violation,
            curvature_violation,
            workspace_violation
        )

        # 判断可行性
        is_feasible = (
            velocity_violation < self.thresholds.velocity_threshold and
            acceleration_violation < self.thresholds.acceleration_threshold and
            curvature_violation < self.thresholds.curvature_threshold
        )

        # 检测严重违反
        severe_violation_type = self._detect_severe_violation(
            velocity_violation,
            acceleration_violation,
            curvature_violation
        )

        return ViolationReport(
            velocity_violation=velocity_violation,
            acceleration_violation=acceleration_violation,
            curvature_violation=curvature_violation,
            workspace_violation=workspace_violation,
            max_violation=max_violation,
            is_feasible=is_feasible,
            severe_violation_type=severe_violation_type
        )

    def compute_velocity_violation(
        self,
        trajectory: BsplineTrajectory
    ) -> float:
        """
        计算速度约束违反量

        Args:
            trajectory: 轨迹对象

        Returns:
            速度约束最大违反量
        """
        violation = self.evaluator.check_velocity_constraint(
            trajectory, self.num_samples
        )
        return violation.max_violation

    def compute_acceleration_violation(
        self,
        trajectory: BsplineTrajectory
    ) -> float:
        """
        计算加速度约束违反量

        Args:
            trajectory: 轨迹对象

        Returns:
            加速度约束最大违反量
        """
        violation = self.evaluator.check_acceleration_constraint(
            trajectory, self.num_samples
        )
        return violation.max_violation

    def compute_curvature_violation(
        self,
        trajectory: BsplineTrajectory
    ) -> float:
        """
        计算曲率约束违反量

        Args:
            trajectory: 轨迹对象

        Returns:
            曲率约束最大违反量
        """
        violation = self.evaluator.check_curvature_constraint(
            trajectory, self.num_samples
        )
        return violation.max_violation

    def compute_workspace_violation(
        self,
        trajectory: BsplineTrajectory
    ) -> float:
        """
        计算工作空间约束违反量

        Args:
            trajectory: 轨迹对象

        Returns:
            工作空间约束最大违反量
        """
        violation = self.evaluator.check_workspace_constraint(
            trajectory, self.num_samples
        )
        return violation.max_violation

    def _detect_severe_violation(
        self,
        velocity_violation: float,
        acceleration_violation: float,
        curvature_violation: float
    ) -> Optional[str]:
        """
        检测严重违反的约束类型

        优先级：加速度 > 速度 > 曲率

        Args:
            velocity_violation: 速度违反量
            acceleration_violation: 加速度违反量
            curvature_violation: 曲率违反量

        Returns:
            严重违反的约束类型，None表示无严重违反
        """
        # 检查加速度严重违反（最高优先级）
        if acceleration_violation > self.thresholds.acceleration_severe_threshold:
            return "acceleration"

        # 检查速度严重违反
        if velocity_violation > self.thresholds.velocity_severe_threshold:
            return "velocity"

        # 检查曲率严重违反
        if curvature_violation > self.thresholds.curvature_severe_threshold:
            return "curvature"

        return None
