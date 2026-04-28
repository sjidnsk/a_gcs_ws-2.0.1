"""
曲率约束v2：旋转二阶锥速度耦合方案

本包实现了基于旋转二阶锥(RotatedLorentzCone)的曲率约束体系，
消除v1方案中(v_max/v_min)^2保守因子，实现逐边自适应速度-曲率耦合约束。

约束体系：
- A1: qᵢ·d_θ ≥ σ_e          (线性速度下界, n+1个/边)
- A2: τ_e·1 ≥ σ_e²           (旋转二阶锥, 1个/边)
- B:  κ_max·τ_e ≥ ‖Qⱼ‖₂     (Lorentz锥, n-1个/边)
- C:  σ_e ≥ σ_min            (下界保证, 1个/边)

模块结构：
- M1: builder.py              - CurvatureConstraintV2Builder
- M2: auxiliary_var_manager.py - AuxiliaryVarManager
- M3: heading_dir_extractor.py - HeadingDirExtractor
- M4: rotated_cone_factory.py  - RotatedConeFactory
- M5: solver_adapter.py        - SolverAdapter
- M6: constraint_validator.py  - ConstraintValidator
- M7: coordinator.py           - CurvatureConstraintCoordinator
"""

# 不依赖pydrake的模块：始终可导入
from .exceptions import (
    CurvatureV2Error,
    InvalidParameterError,
    SolverNotSupportedError,
    PrerequisiteViolationError,
    ConstraintConstructionError,
    VertexExtensionError,
    ConstraintValidationError,
)
from .data_structures import CurvatureV2Result, ValidationReport
from .constants import (
    CURVATURE_V2_FLOAT_TOL,
    CURVATURE_V2_SINGULAR_TOL,
    CURVATURE_V2_COND_WARN,
    CURVATURE_V2_COND_MAX,
    SIGMA_MIN_LOWER_BOUND,
    SIGMA_MIN_UPPER_BOUND,
    KAPPA_MAX_LOWER_BOUND,
    KAPPA_MAX_UPPER_BOUND,
)

# 依赖pydrake的模块：延迟导入，pydrake不可用时跳过
try:
    from .builder import CurvatureConstraintV2Builder
    from .auxiliary_var_manager import AuxiliaryVarManager
    from .heading_dir_extractor import HeadingDirExtractor
    from .rotated_cone_factory import RotatedConeFactory
    from .solver_adapter import SolverAdapter
    from .constraint_validator import ConstraintValidator
    from .coordinator import CurvatureConstraintCoordinator
    _DRAKE_AVAILABLE = True
except ImportError:
    _DRAKE_AVAILABLE = False

__all__ = [
    # 异常
    "CurvatureV2Error",
    "InvalidParameterError",
    "SolverNotSupportedError",
    "PrerequisiteViolationError",
    "ConstraintConstructionError",
    "VertexExtensionError",
    "ConstraintValidationError",
    # 数据结构
    "CurvatureV2Result",
    "ValidationReport",
    # 常量
    "CURVATURE_V2_FLOAT_TOL",
    "CURVATURE_V2_SINGULAR_TOL",
    "CURVATURE_V2_COND_WARN",
    "CURVATURE_V2_COND_MAX",
    "SIGMA_MIN_LOWER_BOUND",
    "SIGMA_MIN_UPPER_BOUND",
    "KAPPA_MAX_LOWER_BOUND",
    "KAPPA_MAX_UPPER_BOUND",
    # 模块 (需要pydrake)
    "CurvatureConstraintV2Builder",
    "AuxiliaryVarManager",
    "HeadingDirExtractor",
    "RotatedConeFactory",
    "SolverAdapter",
    "ConstraintValidator",
    "CurvatureConstraintCoordinator",
]
