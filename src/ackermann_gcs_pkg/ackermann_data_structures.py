"""
阿克曼转向车辆轨迹规划系统 - 数据结构定义

本模块定义了阿克曼转向车辆轨迹规划所需的所有数据结构，包括：
- 车辆参数
- 起终点状态
- 轨迹约束
- 贝塞尔曲线配置
- 约束违反报告
- 连续性报告
- 轨迹评估报告
- 规划结果
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import numpy as np

# 导入数值安全工具
from .numerical_safety_utils import check_tan_safety, NumericalSafetyError


# === 时间缩放导数下界常量 ===
# 用于确保时间缩放导数h'(s)不会过小，避免速度/加速度数值不稳定

HDOT_MIN_DEFAULT: float = 0.01  # hdot_min物理合理默认值
HDOT_MIN_WARNING_THRESHOLD: float = 0.001  # hdot_min过小警告阈值


@dataclass
class VehicleParams:
    """
    车辆参数

    Attributes:
        wheelbase: 车辆轴距（米）
        max_steering_angle: 最大转向角（弧度）
        max_velocity: 最大速度（米/秒）
        max_acceleration: 最大加速度（米/秒²）
        max_curvature: 最大曲率（1/米），由 max_steering_angle 和 wheelbase 计算
        min_turning_radius: 最小转弯半径（米），由 wheelbase / tan(δ_max) 计算
    """
    wheelbase: float
    max_steering_angle: float
    max_velocity: float
    max_acceleration: float
    max_curvature: float = field(init=False)
    min_turning_radius: float = field(init=False)

    def __post_init__(self):
        """参数验证和计算"""
        if self.wheelbase <= 0:
            raise ValueError(f"wheelbase must be positive, got {self.wheelbase}")
        if self.max_steering_angle <= 0:
            raise ValueError(f"max_steering_angle must be positive, got {self.max_steering_angle}")
        if self.max_velocity <= 0:
            raise ValueError(f"max_velocity must be positive, got {self.max_velocity}")
        if self.max_acceleration <= 0:
            raise ValueError(f"max_acceleration must be positive, got {self.max_acceleration}")
        
        # 检查max_steering_angle是否接近π/2，避免tan函数发散
        try:
            check_tan_safety(self.max_steering_angle, angle_name="max_steering_angle")
        except NumericalSafetyError as e:
            raise ValueError(str(e))

        # 计算最大曲率：κ_max = tan(δ_max) / L
        self.max_curvature = np.tan(self.max_steering_angle) / self.wheelbase
        # 计算最小转弯半径：R_min = L / tan(δ_max) = 1 / κ_max
        self.min_turning_radius = self.wheelbase / np.tan(self.max_steering_angle)

    @classmethod
    def fromdict(cls, data: Dict) -> 'VehicleParams':
        """从字典创建实例"""
        return cls(**data)

    def todict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


@dataclass
class EndpointState:
    """
    起终点状态

    Attributes:
        position: 位置坐标（米），形状为(2,)
        heading: 航向角（弧度），范围[-π, π]
        velocity: 速度（米/秒），可选，默认为None
    """
    position: np.ndarray
    heading: float
    velocity: Optional[float] = None

    def __post_init__(self):
        """参数验证"""
        if not isinstance(self.position, np.ndarray) or self.position.shape != (2,):
            raise ValueError(f"position must be a 2D numpy array, got shape {self.position.shape}")
        if not (-np.pi <= self.heading <= np.pi):
            raise ValueError(f"heading must be in [-π, π], got {self.heading}")
        if self.velocity is not None and self.velocity < 0:
            raise ValueError(f"velocity must be non-negative, got {self.velocity}")

    @classmethod
    def fromdict(cls, data: Dict) -> 'EndpointState':
        """从字典创建实例"""
        data['position'] = np.array(data['position'])
        return cls(**data)

    def todict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['position'] = data['position'].tolist()
        return data


class CurvatureConstraintMode(Enum):
    """曲率约束模式枚举"""
    NONE = "none"            # 无曲率硬约束（仅评估检查）
    HARD = "hard"            # 凸硬约束（Lorentz锥，保守但可靠）
    TURNING_RADIUS = "turning_radius"  # 旧转弯半径约束（盒约束，已弃用）


@dataclass
class TrajectoryConstraints:
    """
    轨迹约束

    Attributes:
        max_velocity: 最大速度（米/秒）
        max_acceleration: 最大加速度（米/秒²）
        max_curvature: 最大曲率（1/米）
        workspace_regions: 工作空间区域（HPolyhedron列表）
        enable_curvature_hard_constraint: 是否启用曲率硬约束
        min_velocity: 最小速度（米/秒），用于曲率硬约束计算rho_min
            推导方式: v_optimal = sqrt(w_time / w_energy), min_velocity = v_optimal * 0.5
            默认1.58 m/s 对应 w_time=1.0, w_energy=0.1
        curvature_constraint_mode: 曲率约束模式
        h_bar_prime: h'(s)均值估计。None表示未估计（将使用默认值1.0
            或迭代修正）。指定具体值时跳过迭代修正。
        h_bar_prime_safety_factor: 保守修正因子，范围(0, 1.0]。
            用于防止h̄'过估导致约束过松。默认0.7。
        max_h_bar_prime_iterations: h̄'迭代修正最大次数。
            1表示禁用迭代修正，>1启用。默认3。
        h_bar_prime_convergence_threshold: 迭代收敛判定阈值。
            相对变化小于此值判定收敛。默认0.15（15%）。
        h_bar_prime_relax_factor: 求解失败时h̄'放宽因子。默认1.3。
        max_h_bar_prime_relax_attempts: 求解失败放宽重试最大次数。默认3。
        h_bar_prime_safety_factor_decay: 当h̄'在迭代中显著下降时，
            safety_factor自动乘以此衰减因子以收紧约束。范围(0, 1.0)。
            默认0.8（即safety_factor从0.7降至0.56）。
            设为1.0禁用动态收紧。
    """
    max_velocity: float
    max_acceleration: float
    max_curvature: float
    workspace_regions: Optional[List] = None
    enable_curvature_hard_constraint: bool = False
    min_velocity: float = 1.58
    curvature_constraint_mode: str = "none"
    h_bar_prime: Optional[float] = None
    h_bar_prime_safety_factor: float = 0.7
    max_h_bar_prime_iterations: int = 3
    h_bar_prime_convergence_threshold: float = 0.15
    h_bar_prime_relax_factor: float = 1.3
    max_h_bar_prime_relax_attempts: int = 3
    h_bar_prime_safety_factor_decay: float = 0.8
    # --- v2曲率约束字段 ---
    curvature_constraint_version: str = "v1"
    """曲率约束版本: 'v1'(Lorentz锥) 或 'v2'(旋转二阶锥)"""
    sigma_min: Union[float, str] = "auto"
    """σ_e的最小下界，防止σ_e→0导致约束退化。仅v2模式使用。
    "auto"自动推导，或用户显式指定正数。"""

    def __post_init__(self):
        """参数验证"""
        if self.max_velocity <= 0:
            raise ValueError(f"max_velocity must be positive, got {self.max_velocity}")
        if self.max_acceleration <= 0:
            raise ValueError(f"max_acceleration must be positive, got {self.max_acceleration}")
        if self.max_curvature <= 0:
            raise ValueError(f"max_curvature must be positive, got {self.max_curvature}")
        if self.min_velocity < 0:
            raise ValueError(f"min_velocity must be non-negative, got {self.min_velocity}")
        valid_modes = [m.value for m in CurvatureConstraintMode]
        if self.curvature_constraint_mode not in valid_modes:
            raise ValueError(
                f"curvature_constraint_mode must be one of {valid_modes}, "
                f"got {self.curvature_constraint_mode}"
            )
        # 当启用硬约束时，自动设置模式
        if self.enable_curvature_hard_constraint and self.curvature_constraint_mode == "none":
            self.curvature_constraint_mode = "hard"
        # h_bar_prime 相关参数验证
        if self.h_bar_prime is not None and self.h_bar_prime <= 0:
            raise ValueError(
                f"h_bar_prime must be positive when specified, "
                f"got {self.h_bar_prime}"
            )
        if not (0 < self.h_bar_prime_safety_factor <= 1.0):
            raise ValueError(
                f"h_bar_prime_safety_factor must be in (0, 1.0], "
                f"got {self.h_bar_prime_safety_factor}"
            )
        if self.max_h_bar_prime_iterations < 1:
            raise ValueError(
                f"max_h_bar_prime_iterations must be >= 1, "
                f"got {self.max_h_bar_prime_iterations}"
            )
        if not (0 < self.h_bar_prime_convergence_threshold < 1.0):
            raise ValueError(
                f"h_bar_prime_convergence_threshold must be in (0, 1.0), "
                f"got {self.h_bar_prime_convergence_threshold}"
            )
        if not (0 < self.h_bar_prime_safety_factor_decay <= 1.0):
            raise ValueError(
                f"h_bar_prime_safety_factor_decay must be in (0, 1.0], "
                f"got {self.h_bar_prime_safety_factor_decay}"
            )
        # v2曲率约束字段验证
        if self.curvature_constraint_version not in ("v1", "v2"):
            raise ValueError(
                f"curvature_constraint_version must be 'v1' or 'v2', "
                f"got {self.curvature_constraint_version}"
            )
        if isinstance(self.sigma_min, (int, float)) and self.sigma_min <= 0:
            raise ValueError(
                f"sigma_min must be positive when specified as a number, "
                f"got {self.sigma_min}"
            )

    @classmethod
    def fromdict(cls, data: Dict) -> 'TrajectoryConstraints':
        """从字典创建实例"""
        return cls(**data)

    @staticmethod
    def compute_min_velocity_from_weights(
        w_time: float, w_energy: float, safety_factor: float = 0.5
    ) -> float:
        """从成本权重推导最小速度。

        最优速度 v_optimal = sqrt(w_time / w_energy)，
        min_velocity = v_optimal * safety_factor。

        Args:
            w_time: 时间成本权重
            w_energy: 能量成本权重
            safety_factor: 保守系数，默认0.5

        Returns:
            推导的最小速度（米/秒）
        """
        if w_energy <= 0:
            raise ValueError(f"w_energy must be positive, got {w_energy}")
        v_optimal = np.sqrt(w_time / w_energy)
        return v_optimal * safety_factor

    def todict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


@dataclass
class BezierConfig:
    """
    贝塞尔曲线配置

    Attributes:
        order: 贝塞尔曲线阶数（控制点数量 = order + 1）
        continuity: 连续性阶数（0=C0，1=C1，2=C2）
        hdot_min: 时间导数最小值（避免时间倒流和数值不稳定）
        full_dim_overlap: 是否使用全维重叠
        hyperellipsoid_num_samples_per_dim_factor: 超椭球采样因子
        max_rounding_attempts: 舍入验证重试次数
        max_rounded_paths: 每次舍入尝试的路径数
    """
    order: int = 5
    continuity: int = 1
    hdot_min: float = HDOT_MIN_DEFAULT
    full_dim_overlap: bool = False
    hyperellipsoid_num_samples_per_dim_factor: int = 32
    max_rounding_attempts: int = 3
    max_rounded_paths: int = 5

    def __post_init__(self):
        """参数验证"""
        if self.order < 1:
            raise ValueError(f"order must be >= 1, got {self.order}")
        if self.continuity < 0 or self.continuity >= self.order:
            raise ValueError(f"continuity must be in [0, {self.order-1}], got {self.continuity}")
        if self.hdot_min <= 0:
            raise ValueError(f"hdot_min must be positive, got {self.hdot_min}")
        if self.hdot_min < HDOT_MIN_WARNING_THRESHOLD:
            import warnings
            warnings.warn(
                f"hdot_min={self.hdot_min} is below recommended threshold "
                f"{HDOT_MIN_WARNING_THRESHOLD}. This may cause numerical "
                f"instability in velocity/acceleration constraints.",
                UserWarning,
                stacklevel=2,
            )

    @classmethod
    def fromdict(cls, data: Dict) -> 'BezierConfig':
        """从字典创建实例"""
        return cls(**data)

    def todict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


@dataclass
class ConstraintViolation:
    """
    约束违反报告

    Attributes:
        constraint_name: 约束名称
        is_violated: 是否违反约束
        max_violation: 最大违反量
        violation_points: 违反点列表（参数值s）
    """
    constraint_name: str
    is_violated: bool
    max_violation: float
    violation_points: List[float] = field(default_factory=list)

    def todict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


@dataclass
class ContinuityReport:
    """
    连续性报告

    Attributes:
        order: 连续性阶数（0=C0，1=C1，2=C2）
        is_continuous: 是否连续
        max_discontinuity: 最大不连续量
        discontinuity_points: 不连续点列表（参数值s）
    """
    order: int
    is_continuous: bool
    max_discontinuity: float
    discontinuity_points: List[float] = field(default_factory=list)

    def todict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


@dataclass
class CurvatureDerivatives:
    """
    曲率及各阶导数

    存储曲率计算过程中的中间结果，用于解析梯度计算和缓存复用

    Attributes:
        curvature: 曲率 κ = (ẋÿ - ẏẍ) / (ẋ² + ẏ²)^(3/2)
        curvature_derivative: 曲率导数 dκ/ds = (dκ/dt) / ||r'(s)||
        first_deriv: 一阶导数 r'(s) = [ẋ, ẏ]，形状为(2,)
        second_deriv: 二阶导数 r''(s) = [ẍ, ÿ]，形状为(2,)
        third_deriv: 三阶导数 r'''(s) = [x''', y''']，形状为(2,)
        speed: 速度 ||r'(s)|| = sqrt(ẋ² + ẏ²)
    """
    curvature: float
    curvature_derivative: float
    first_deriv: np.ndarray
    second_deriv: np.ndarray
    third_deriv: np.ndarray
    speed: float = 0.0

    def __post_init__(self):
        """参数验证"""
        if not isinstance(self.first_deriv, np.ndarray) or self.first_deriv.shape != (2,):
            raise ValueError(f"first_deriv must be a (2,) numpy array, got shape {self.first_deriv.shape}")
        if not isinstance(self.second_deriv, np.ndarray) or self.second_deriv.shape != (2,):
            raise ValueError(f"second_deriv must be a (2,) numpy array, got shape {self.second_deriv.shape}")
        if not isinstance(self.third_deriv, np.ndarray) or self.third_deriv.shape != (2,):
            raise ValueError(f"third_deriv must be a (2,) numpy array, got shape {self.third_deriv.shape}")
        # 如果speed未提供，自动计算
        if self.speed == 0.0:
            self.speed = np.linalg.norm(self.first_deriv)

    def todict(self) -> Dict:
        """转换为字典"""
        return {
            'curvature': self.curvature,
            'curvature_derivative': self.curvature_derivative,
            'first_deriv': self.first_deriv.tolist(),
            'second_deriv': self.second_deriv.tolist(),
            'third_deriv': self.third_deriv.tolist(),
            'speed': self.speed,
        }

    @classmethod
    def fromdict(cls, data: Dict) -> 'CurvatureDerivatives':
        """从字典创建实例"""
        return cls(
            curvature=data['curvature'],
            curvature_derivative=data['curvature_derivative'],
            first_deriv=np.array(data['first_deriv']),
            second_deriv=np.array(data['second_deriv']),
            third_deriv=np.array(data['third_deriv']),
            speed=data.get('speed', 0.0),
        )


@dataclass
class CurvatureStats:
    """
    曲率统计信息

    所有统计量（max、min、mean、std）均基于 |κ|（曲率绝对值）计算，
    描述轨迹弯曲程度的分布特征，不含方向信息。

    Attributes:
        max_curvature: 最大曲率绝对值，max(|κ|)
        min_curvature: 最小曲率绝对值，min(|κ|)
        mean_curvature: 平均曲率绝对值，mean(|κ|)
        std_curvature: 曲率绝对值的标准差，std(|κ|)
        max_curvature_location: 最大曲率位置（参数s）
        curvature_samples: 曲率采样点（带符号，可选）
    """
    max_curvature: float
    min_curvature: float
    mean_curvature: float
    std_curvature: float
    max_curvature_location: float
    curvature_samples: Optional[np.ndarray] = None

    def todict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        if self.curvature_samples is not None:
            data['curvature_samples'] = self.curvature_samples.tolist()
        return data


@dataclass
class ImprovementMetrics:
    """
    平滑度改善指标

    Attributes:
        peak_reduction_percent: 峰值降低百分比
        std_reduction_percent: 标准差降低百分比
        mean_reduction_percent: 平均值降低百分比
        baseline_stats: 基准统计（可选）
        optimized_stats: 优化后统计（可选）
    """
    peak_reduction_percent: float
    std_reduction_percent: float
    mean_reduction_percent: float
    baseline_stats: Optional[CurvatureStats] = None
    optimized_stats: Optional[CurvatureStats] = None

    def todict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        if self.baseline_stats:
            data['baseline_stats'] = self.baseline_stats.todict()
        if self.optimized_stats:
            data['optimized_stats'] = self.optimized_stats.todict()
        return data


@dataclass
class TrajectoryReport:
    """
    轨迹评估报告

    Attributes:
        is_feasible: 轨迹是否可行
        velocity_violation: 速度约束违反报告
        acceleration_violation: 加速度约束违反报告
        curvature_violation: 曲率约束违反报告
        workspace_violation: 工作空间约束违反报告
        c0_continuity: C0连续性报告
        c1_continuity: C1连续性报告
        c2_continuity: C2连续性报告
        curvature_stats: 曲率统计信息（新增）
        improvement_metrics: 平滑度改善指标（新增）
    """
    is_feasible: bool
    velocity_violation: Optional[ConstraintViolation] = None
    acceleration_violation: Optional[ConstraintViolation] = None
    curvature_violation: Optional[ConstraintViolation] = None
    workspace_violation: Optional[ConstraintViolation] = None
    c0_continuity: Optional[ContinuityReport] = None
    c1_continuity: Optional[ContinuityReport] = None
    c2_continuity: Optional[ContinuityReport] = None
    # 新增字段
    curvature_stats: Optional[CurvatureStats] = None
    improvement_metrics: Optional[ImprovementMetrics] = None

    def todict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        # 转换嵌套的dataclass
        if self.velocity_violation:
            data['velocity_violation'] = self.velocity_violation.todict()
        if self.acceleration_violation:
            data['acceleration_violation'] = self.acceleration_violation.todict()
        if self.curvature_violation:
            data['curvature_violation'] = self.curvature_violation.todict()
        if self.workspace_violation:
            data['workspace_violation'] = self.workspace_violation.todict()
        if self.c0_continuity:
            data['c0_continuity'] = self.c0_continuity.todict()
        if self.c1_continuity:
            data['c1_continuity'] = self.c1_continuity.todict()
        if self.c2_continuity:
            data['c2_continuity'] = self.c2_continuity.todict()
        # 新增字段转换
        if self.curvature_stats:
            data['curvature_stats'] = self.curvature_stats.todict()
        if self.improvement_metrics:
            data['improvement_metrics'] = self.improvement_metrics.todict()
        return data


@dataclass
class PlanningResult:
    """
    规划结果

    Attributes:
        success: 规划是否成功
        trajectory: 轨迹对象（BezierTrajectory）
        trajectory_report: 轨迹评估报告
        solve_time: 求解时间（秒）
        num_iterations: 求解尝试次数（GCS多次舍入尝试）
        convergence_reason: 收敛原因
        error_message: 错误消息（如果失败）
    """
    success: bool
    trajectory: Optional[object] = None
    trajectory_report: Optional[TrajectoryReport] = None
    solve_time: float = 0.0
    num_iterations: int = 0
    convergence_reason: str = ""
    error_message: str = ""

    def todict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        if self.trajectory_report:
            data['trajectory_report'] = self.trajectory_report.todict()
        return data
