"""
曲率约束v2数值安全常量

遵循IEEE 754双精度规范和MOSEK Modeling Cookbook §6.2数值稳定性建议。
"""

# === 数值容差常量 ===

# 零值判定容差：用于浮点数与0的比较
CURVATURE_V2_FLOAT_TOL: float = 1e-12

# 奇异值判定容差：用于检测向量退化（如航向角方向向量范数过小）
CURVATURE_V2_SINGULAR_TOL: float = 1e-8

# 条件数警告阈值：cond(A)超过此值时发出警告
CURVATURE_V2_COND_WARN: float = 1e6

# 条件数最大允许值：cond(A)超过此值时视为数值不可靠
CURVATURE_V2_COND_MAX: float = 1e8

# === 参数边界 ===

# σ_min绝对下界：低于此值数值不可靠
SIGMA_MIN_LOWER_BOUND: float = 1e-6

# σ_min上界：超过此值约束过强，可能无可行解
SIGMA_MIN_UPPER_BOUND: float = 1.0

# κ_max绝对下界
KAPPA_MAX_LOWER_BOUND: float = 1e-3

# κ_max上界：超过此值物理不合理（R_min < 0.01m）
KAPPA_MAX_UPPER_BOUND: float = 100.0

# === 默认参数 ===

# σ_min默认值（当sigma_min="auto"时的推导结果典型值）
SIGMA_MIN_DEFAULT: float = 0.01

# 默认航向角方向（当无法提取时的回退值：x轴正方向）
DEFAULT_HEADING_DIRECTION = None  # 延迟初始化为np.array([1.0, 0.0])

# === 日志前缀 ===
LOG_PREFIX = "CURVATURE_V2"
