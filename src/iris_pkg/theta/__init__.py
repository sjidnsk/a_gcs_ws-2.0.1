"""
Theta 处理模块

包含 Theta 单位向量处理和混合约束策略。

作者: Path Planning Team
"""

from .theta_unit_vector_handler import (
    ThetaUnitVectorHandler,
    UnitVectorConfig,
    theta_to_unit_vector,
    unit_vector_to_theta
)

from .hybrid_theta_constraint import (
    HybridThetaConstraintStrategy,
    HybridConstraintConfig
)

from .bezier_reparameterization import (
    BezierReparameterizer,
    ReparameterizationConfig,
    reparameterize_bezier_trajectory
)

__all__ = [
    'ThetaUnitVectorHandler',
    'UnitVectorConfig',
    'theta_to_unit_vector',
    'unit_vector_to_theta',
    'HybridThetaConstraintStrategy',
    'HybridConstraintConfig',
    'BezierReparameterizer',
    'ReparameterizationConfig',
    'reparameterize_bezier_trajectory'
]
