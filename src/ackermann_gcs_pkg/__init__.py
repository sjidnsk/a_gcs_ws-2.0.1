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
        BezierConfig,
        ConstraintViolation,
        ContinuityReport,
        TrajectoryReport,
        PlanningResult,
        CurvatureDerivatives,
    )
    from .ackermann_bezier_gcs import AckermannBezierGCS
    from .ackermann_gcs_planner import AckermannGCSPlanner
    from .trajectory_evaluator import TrajectoryEvaluator
    from .h_bar_prime_iteration import (
        HBarPrimeIterationResult,
        iterate_h_bar_prime,
    )
    from .flat_output_mapper import FlatOutputMapper, compute_flat_output_mapping
    # 工具模块
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
                "BezierConfig", "ConstraintViolation",
                "ContinuityReport", "TrajectoryReport", "PlanningResult",
                "CurvatureDerivatives"):
        from . import ackermann_data_structures
        obj = getattr(ackermann_data_structures, name)
        _import_cache[name] = obj
        return obj
    
    # 核心类
    if name == "AckermannBezierGCS":
        from .ackermann_bezier_gcs import AckermannBezierGCS
        _import_cache[name] = AckermannBezierGCS
        return AckermannBezierGCS
    
    if name == "AckermannGCSPlanner":
        from .ackermann_gcs_planner import AckermannGCSPlanner
        _import_cache[name] = AckermannGCSPlanner
        return AckermannGCSPlanner
    
    if name == "TrajectoryEvaluator":
        from .trajectory_evaluator import TrajectoryEvaluator
        _import_cache[name] = TrajectoryEvaluator
        return TrajectoryEvaluator
    
    if name in ("HBarPrimeIterationResult", "iterate_h_bar_prime"):
        from . import h_bar_prime_iteration
        obj = getattr(h_bar_prime_iteration, name)
        _import_cache[name] = obj
        return obj
    
    if name in ("FlatOutputMapper", "compute_flat_output_mapping"):
        from . import flat_output_mapper
        obj = getattr(flat_output_mapper, name)
        _import_cache[name] = obj
        return obj
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# 公共 API 导出列表
__all__ = [
    # 数据结构
    "VehicleParams",
    "EndpointState",
    "TrajectoryConstraints",
    "BezierConfig",
    "ConstraintViolation",
    "ContinuityReport",
    "TrajectoryReport",
    "PlanningResult",
    "CurvatureDerivatives",
    # 核心类
    "AckermannBezierGCS",
    "AckermannGCSPlanner",
    "TrajectoryEvaluator",
    "HBarPrimeIterationResult",
    "iterate_h_bar_prime",
    "FlatOutputMapper",
    # 函数
    "compute_flat_output_mapping",
]

__version__ = "0.2.0"
