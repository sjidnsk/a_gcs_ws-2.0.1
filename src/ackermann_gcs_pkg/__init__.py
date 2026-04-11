"""
阿克曼转向车辆轨迹规划系统 - 核心模块

本包提供阿克曼转向车辆的轨迹规划功能，基于微分平坦性理论和GCS（图凸集）算法。
"""

from typing import TYPE_CHECKING

# 类型检查时导入（运行时不执行）
if TYPE_CHECKING:
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
        LinearizedCostCoefficients,
        CurvatureDerivatives,
        # CurvatureCostConfig,  # [2025-04-06] 曲率成本功能暂时禁用
        # CurvatureCostWeights,  # [2025-04-06] 曲率成本功能暂时禁用
    )
    from .ackermann_bezier_gcs import AckermannBezierGCS
    from .ackermann_scp_solver import AckermannSCPSolver
    from .ackermann_gcs_planner import AckermannGCSPlanner
    from .trajectory_evaluator import TrajectoryEvaluator
    from .flat_output_mapper import FlatOutputMapper, compute_flat_output_mapping
    # from .curvature_cost_module import CurvatureCostModule  # [2025-04-06] 曲率成本功能暂时禁用
    # from .curvature_cost_linearizer import CurvatureCostLinearizer  # [2025-04-06] 曲率成本功能暂时禁用
    # from .curvature_derivative_cost import CurvatureDerivativeCost  # [2025-04-06] 曲率成本功能暂时禁用
    # from .curvature_peak_cost import CurvaturePeakCost  # [2025-04-06] 曲率成本功能暂时禁用
    from .analytic_gradient_calculator import AnalyticGradientCalculator
    from .cost_calculator_interface import CostCalculatorInterface
    from .curvature_squared_cost_calculator import CurvatureSquaredCostCalculator
    # 新增工具模块
    from .constants import *
    from .trajectory_utils import *
    from .curvature_utils import *
    from .constraint_utils import *
    from .formatting_utils import *

# 延迟导入缓存
_import_cache = {}

def __getattr__(name: str):
    """
    延迟导入机制
    
    只在真正需要时才导入模块，避免初始化时的循环依赖。
    """
    if name in _import_cache:
        return _import_cache[name]
    
    # 数据结构
    if name in ("VehicleParams", "EndpointState", "TrajectoryConstraints",
                "SCPConfig", "BezierConfig", "ConstraintViolation",
                "ContinuityReport", "TrajectoryReport", "PlanningResult",
                "LinearizedCostCoefficients", "CurvatureDerivatives"):
        # [2025-04-06] 曲率成本功能暂时禁用: 移除 CurvatureCostConfig, CurvatureCostWeights
        from . import ackermann_data_structures
        obj = getattr(ackermann_data_structures, name)
        _import_cache[name] = obj
        return obj
    
    # 核心类
    if name == "AckermannBezierGCS":
        from .ackermann_bezier_gcs import AckermannBezierGCS
        _import_cache[name] = AckermannBezierGCS
        return AckermannBezierGCS
    
    if name == "AckermannSCPSolver":
        from .ackermann_scp_solver import AckermannSCPSolver
        _import_cache[name] = AckermannSCPSolver
        return AckermannSCPSolver
    
    if name == "AckermannGCSPlanner":
        from .ackermann_gcs_planner import AckermannGCSPlanner
        _import_cache[name] = AckermannGCSPlanner
        return AckermannGCSPlanner
    
    if name == "TrajectoryEvaluator":
        from .trajectory_evaluator import TrajectoryEvaluator
        _import_cache[name] = TrajectoryEvaluator
        return TrajectoryEvaluator
    
    if name in ("FlatOutputMapper", "compute_flat_output_mapping"):
        from . import flat_output_mapper
        obj = getattr(flat_output_mapper, name)
        _import_cache[name] = obj
        return obj
    
    # [2025-04-06] 曲率成本功能暂时禁用 - 开始
    # # 曲率成本模块
    # if name == "CurvatureCostModule":
    #     from .curvature_cost_module import CurvatureCostModule
    #     _import_cache[name] = CurvatureCostModule
    #     return CurvatureCostModule
    #
    # if name == "CurvatureCostLinearizer":
    #     from .curvature_cost_linearizer import CurvatureCostLinearizer
    #     _import_cache[name] = CurvatureCostLinearizer
    #     return CurvatureCostLinearizer
    #
    # if name == "CurvatureDerivativeCost":
    #     from .curvature_derivative_cost import CurvatureDerivativeCost
    #     _import_cache[name] = CurvatureDerivativeCost
    #     return CurvatureDerivativeCost
    #
    # if name == "CurvaturePeakCost":
    #     from .curvature_peak_cost import CurvaturePeakCost
    #     _import_cache[name] = CurvaturePeakCost
    #     return CurvaturePeakCost
    #
    # if name == "CurvatureSquaredCostCalculator":
    #     from .curvature_squared_cost_calculator import CurvatureSquaredCostCalculator
    #     _import_cache[name] = CurvatureSquaredCostCalculator
    #     return CurvatureSquaredCostCalculator
    #
    # if name == "AnalyticGradientCalculator":
    #     from .analytic_gradient_calculator import AnalyticGradientCalculator
    #     _import_cache[name] = AnalyticGradientCalculator
    #     return AnalyticGradientCalculator
    # [2025-04-06] 曲率成本功能暂时禁用 - 结束

    if name == "CostCalculatorInterface":
        from .cost_calculator_interface import CostCalculatorInterface
        _import_cache[name] = CostCalculatorInterface
        return CostCalculatorInterface
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# 公共 API 导出列表
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
    "LinearizedCostCoefficients",
    "CurvatureDerivatives",
    # "CurvatureCostConfig",  # [2025-04-06] 曲率成本功能暂时禁用
    # "CurvatureCostWeights",  # [2025-04-06] 曲率成本功能暂时禁用
    # 核心类
    "AckermannBezierGCS",
    "AckermannSCPSolver",
    "AckermannGCSPlanner",
    "TrajectoryEvaluator",
    "FlatOutputMapper",
    # [2025-04-06] 曲率成本功能暂时禁用 - 开始
    # # 曲率成本模块
    # "CurvatureCostModule",
    # "CurvatureCostLinearizer",
    # "CurvatureDerivativeCost",
    # "CurvaturePeakCost",
    # "CurvatureSquaredCostCalculator",
    # "AnalyticGradientCalculator",
    # [2025-04-06] 曲率成本功能暂时禁用 - 结束
    # 新增接口
    "CostCalculatorInterface",
    # 函数
    "compute_flat_output_mapping",
]

__version__ = "0.2.0"
