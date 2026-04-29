"""
曲率约束v2：旋转二阶锥速度耦合方案

本模块实现基于旋转二阶锥(RotatedLorentzCone)的曲率约束体系，
消除v1方案中(v_max/v_min)^2保守因子，实现逐边自适应速度-曲率耦合约束。

约束体系：
- A1: q_i · d_θ ≥ σ_e          (线性速度下界, n+1个/边)
- A2: τ_e · 1 ≥ σ_e²           (旋转二阶锥, 1个/边)
- B:  κ_max · τ_e ≥ ‖Q_j‖_2   (Lorentz锥, n-1个/边)
- C:  σ_e ≥ σ_min              (下界保证, 1个/边)

模块结构：
- M1: CurvatureConstraintV2Builder (builder.py)
- M2: AuxiliaryVarManager (auxiliary_var_manager.py)
- M3: HeadingDirExtractor (heading_dir_extractor.py)
- M4: RotatedConeFactory (rotated_cone_factory.py)
- M5: SolverAdapter (solver_adapter.py)
- M6: ConstraintValidator (constraint_validator.py)
- M7: CurvatureConstraintCoordinator (coordinator.py)
"""

from .exceptions import (
    CurvatureV2Error,
    InvalidParameterError,
    SolverNotSupportedError,
    PrerequisiteViolationError,
    ConstraintConstructionError,
    VertexExtensionError,
    ConstraintValidationError,
)

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

from .data_structures import CurvatureV2Result, ValidationReport

from .coordinator import CurvatureConstraintCoordinator

__all__ = [
    # 异常
    'CurvatureV2Error',
    'InvalidParameterError',
    'SolverNotSupportedError',
    'PrerequisiteViolationError',
    'ConstraintConstructionError',
    'VertexExtensionError',
    'ConstraintValidationError',
    # 常量
    'CURVATURE_V2_FLOAT_TOL',
    'CURVATURE_V2_SINGULAR_TOL',
    'CURVATURE_V2_COND_WARN',
    'CURVATURE_V2_COND_MAX',
    'SIGMA_MIN_LOWER_BOUND',
    'SIGMA_MIN_UPPER_BOUND',
    'KAPPA_MAX_LOWER_BOUND',
    'KAPPA_MAX_UPPER_BOUND',
    # 数据结构
    'CurvatureV2Result',
    'ValidationReport',
    # 协调器
    'CurvatureConstraintCoordinator',
]
