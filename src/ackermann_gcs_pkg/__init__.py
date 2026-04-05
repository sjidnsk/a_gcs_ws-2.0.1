"""
阿克曼转向车辆轨迹规划系统 - 核心模块

本包提供阿克曼转向车辆的轨迹规划功能，基于微分平坦性理论和GCS（图凸集）算法。
"""

from .ackermann_data_structures import (
    VehicleParams,
    EndpointState,
    TrajectoryConstraints,
    SCPConfig,
    BezierConfig,
    ConstraintViolation,
    ContinuityReport,
    TrajectoryReport,
    PlanningResult,
    # 新增数据结构
    LinearizedCostCoefficients,
    CurvatureDerivatives,
    CurvatureCostConfig,
    CurvatureCostWeights,
)

from .ackermann_bezier_gcs import AckermannBezierGCS
from .ackermann_scp_solver import AckermannSCPSolver
from .ackermann_gcs_planner import AckermannGCSPlanner
from .trajectory_evaluator import TrajectoryEvaluator
from .flat_output_mapper import FlatOutputMapper, compute_flat_output_mapping

# 新增模块
from .curvature_cost_module import CurvatureCostModule
from .curvature_cost_linearizer import CurvatureCostLinearizer
from .curvature_derivative_cost import CurvatureDerivativeCost
from .curvature_peak_cost import CurvaturePeakCost
from .analytic_gradient_calculator import AnalyticGradientCalculator

__all__ = [
    # 数据结构
    "VehicleParams",
    "EndpointState",
    "TrajectoryConstraints",
    "SCPConfig",
    "BezierConfig",
    "ConstraintViolation",
    "ContinuityReport",
    "TrajectoryReport",
    "PlanningResult",
    # 新增数据结构
    "LinearizedCostCoefficients",
    "CurvatureDerivatives",
    "CurvatureCostConfig",
    "CurvatureCostWeights",
    # 核心类
    "AckermannBezierGCS",
    "AckermannSCPSolver",
    "AckermannGCSPlanner",
    "TrajectoryEvaluator",
    "FlatOutputMapper",
    # 曲率成本模块
    "CurvatureCostModule",
    "CurvatureCostLinearizer",
    "CurvatureDerivativeCost",
    "CurvaturePeakCost",
    "AnalyticGradientCalculator",
    # 函数
    "compute_flat_output_mapping",
]

__version__ = "0.2.0"
