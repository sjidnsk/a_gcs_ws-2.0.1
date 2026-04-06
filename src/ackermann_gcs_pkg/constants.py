"""数值常量定义模块

本模块统一管理所有数值容差常量和默认参数，避免分散在多个模块中的重复定义。
这些常量用于数值计算、轨迹采样、约束验证等场景。

创建日期: 2025-01-06
作者: 代码重构任务
"""

# ============================================================================
# 数值容差常量
# ============================================================================

# 通用数值计算容差，用于避免除零和浮点数精度问题
NUMERICAL_TOLERANCE: float = 1e-10

# 小量阈值，用于判断速度、距离等是否为零
SMALL_VALUE_THRESHOLD: float = 1e-6

# 零速度判断阈值，用于判断车辆是否静止
ZERO_SPEED_THRESHOLD: float = 1e-6

# 小量容差，用于判断向量是否为零
SMALL_EPSILON: float = 1e-6

# 角度容差，用于判断特殊角度（如0°、90°等）
ANGLE_TOLERANCE: float = 1e-6

# ============================================================================
# 采样参数常量
# ============================================================================

# 轨迹评估默认采样点数
DEFAULT_NUM_SAMPLES: int = 100

# 曲率线性化默认采样点数
DEFAULT_LINEARIZATION_SAMPLES: int = 50

# 连续性检查默认采样点数
DEFAULT_CONTINUITY_SAMPLES: int = 11

# 曲率统计默认采样点数
DEFAULT_CURVATURE_STATISTICS_SAMPLES: int = 100

# ============================================================================
# 约束违反阈值常量
# ============================================================================

# 速度约束违反阈值 (m/s)
VELOCITY_VIOLATION_THRESHOLD: float = 1e-2

# 加速度约束违反阈值 (m/s²)
ACCELERATION_VIOLATION_THRESHOLD: float = 1e-4

# 曲率约束违反阈值 (1/m)
CURVATURE_VIOLATION_THRESHOLD: float = 1e-4

# 连续性检查默认容差
DEFAULT_CONTINUITY_TOLERANCE: float = 1e-3

# ============================================================================
# 优化参数常量
# ============================================================================

# 最小信任区域半径
DEFAULT_MIN_TRUST_REGION: float = 1e-6

# 正则化参数
DEFAULT_REGULARIZATION: float = 1e-6

# Hessian对角线最小值
DEFAULT_HESSIAN_DIAGONAL: float = 1e-6

# ============================================================================
# 曲率计算相关常量
# ============================================================================

# 曲率计算默认容差
CURVATURE_EPSILON: float = 1e-7

# 曲率梯度计算默认容差
CURVATURE_GRADIENT_EPSILON: float = 1e-10

# ============================================================================
# 数值安全相关常量
# ============================================================================

# 默认除零保护值
DEFAULT_EPSILON: float = 1e-10

# 默认小量保护值
DEFAULT_SMALL_VALUE: float = 1e-6

# ============================================================================
# 模块导出
# ============================================================================

__all__ = [
    # 数值容差常量
    'NUMERICAL_TOLERANCE',
    'SMALL_VALUE_THRESHOLD',
    'ZERO_SPEED_THRESHOLD',
    'SMALL_EPSILON',
    'ANGLE_TOLERANCE',
    
    # 采样参数常量
    'DEFAULT_NUM_SAMPLES',
    'DEFAULT_LINEARIZATION_SAMPLES',
    'DEFAULT_CONTINUITY_SAMPLES',
    'DEFAULT_CURVATURE_STATISTICS_SAMPLES',
    
    # 约束违反阈值常量
    'VELOCITY_VIOLATION_THRESHOLD',
    'ACCELERATION_VIOLATION_THRESHOLD',
    'CURVATURE_VIOLATION_THRESHOLD',
    'DEFAULT_CONTINUITY_TOLERANCE',
    
    # 优化参数常量
    'DEFAULT_MIN_TRUST_REGION',
    'DEFAULT_REGULARIZATION',
    'DEFAULT_HESSIAN_DIAGONAL',
    
    # 曲率计算相关常量
    'CURVATURE_EPSILON',
    'CURVATURE_GRADIENT_EPSILON',
    
    # 数值安全相关常量
    'DEFAULT_EPSILON',
    'DEFAULT_SMALL_VALUE',
]
