"""
阿克曼转向车辆轨迹规划系统 - 数据结构定义

本模块定义了阿克曼转向车辆轨迹规划所需的所有数据结构，包括：
- 车辆参数
- 起终点状态
- 轨迹约束
- SCP配置
- 贝塞尔曲线配置
- 约束违反报告
- 连续性报告
- 轨迹评估报告
- 规划结果
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from enum import Enum
import numpy as np
import time

# 导入数值安全工具
from .numerical_safety_utils import check_tan_safety, NumericalSafetyError


# === 约束违反阈值常量 ===
# 用于判断约束是否违反的阈值

VELOCITY_VIOLATION_THRESHOLD: float = 1e-2  # 速度约束违反阈值 (m/s)
ACCELERATION_VIOLATION_THRESHOLD: float = 1e-4  # 加速度约束违反阈值 (m/s²)
CURVATURE_VIOLATION_THRESHOLD: float = 1e-4  # 曲率约束违反阈值 (1/m)

# === 收敛与终止阈值常量 ===
# 用于SCP迭代收敛判断

DEFAULT_CONVERGENCE_TOLERANCE: float = 1e-3  # 默认收敛容差
DEFAULT_STAGNATION_THRESHOLD: float = 0.01  # 改进停滞阈值
DEFAULT_MIN_TRUST_REGION: float = 1e-6  # 最小信任区域半径

# === 迭代参数常量 ===
# 用于SCP迭代、优化算法等

DEFAULT_MAX_ITERATIONS: int = 10  # SCP默认最大迭代次数
DEFAULT_STAGNATION_WINDOW: int = 5  # 停滞检测窗口大小
DEFAULT_MAX_SHRINK_COUNT: int = 3  # 最大连续缩小次数

# === 信任区域参数常量 ===
# 用于信任区域调整策略

DEFAULT_TRUST_REGION_RADIUS: float = 1.0  # 默认初始信任区域半径
TRUST_REGION_EXPAND_FACTOR: float = 2.0  # 信任区域扩大因子
TRUST_REGION_SHRINK_FACTOR: float = 0.5  # 信任区域缩小因子
AGGRESSIVE_SHRINK_FACTOR: float = 0.25  # 激进缩小因子

# === 采样参数常量 ===
# 用于轨迹评估、曲率计算等过程中的采样

DEFAULT_BATCH_SIZE: int = 20  # 并行处理默认批次大小

# === 数值容差常量 ===
# 用于数值计算中的容差判断

SMALL_EPSILON: float = 1e-6  # 小量容差，用于判断向量是否为零


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
    """
    wheelbase: float
    max_steering_angle: float
    max_velocity: float
    max_acceleration: float
    max_curvature: float = field(init=False)

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


@dataclass
class TrajectoryConstraints:
    """
    轨迹约束

    Attributes:
        max_velocity: 最大速度（米/秒）
        max_acceleration: 最大加速度（米/秒²）
        max_curvature: 最大曲率（1/米）
        workspace_regions: 工作空间区域（HPolyhedron列表）
    """
    max_velocity: float
    max_acceleration: float
    max_curvature: float
    workspace_regions: Optional[List] = None

    def __post_init__(self):
        """参数验证"""
        if self.max_velocity <= 0:
            raise ValueError(f"max_velocity must be positive, got {self.max_velocity}")
        if self.max_acceleration <= 0:
            raise ValueError(f"max_acceleration must be positive, got {self.max_acceleration}")
        if self.max_curvature <= 0:
            raise ValueError(f"max_curvature must be positive, got {self.max_curvature}")

    @classmethod
    def fromdict(cls, data: Dict) -> 'TrajectoryConstraints':
        """从字典创建实例"""
        return cls(**data)

    def todict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


# ==================== SCP优化相关数据结构 ====================

class TerminationReason(Enum):
    """
    终止原因枚举
    
    定义SCP迭代可能的所有终止原因
    """
    CONTINUE = "continue"                       # 继续迭代
    CONVERGED = "converged"                     # 收敛成功
    IMPROVEMENT_STAGNATION = "improvement_stagnation"  # 改进停滞
    CONSTRAINT_SATISFIED = "constraint_satisfied"      # 约束满足
    TRUST_REGION_EXHAUSTED = "trust_region_exhausted"  # 信任区域耗尽
    LOCAL_OPTIMUM = "local_optimum"            # 局部最优
    MAX_ITERATIONS = "max_iterations"          # 达到最大迭代


@dataclass
class TrustRegionConfig:
    """
    信任区域配置
    
    用于动态信任区域调整策略的配置参数
    
    Attributes:
        high_improvement_threshold: 高改进阈值，超过此值扩大信任区域
        medium_improvement_threshold: 中等改进阈值，在此范围内保持信任区域
        low_improvement_threshold: 低改进阈值，低于此值缩小信任区域
        expand_factor: 信任区域扩大因子
        maintain_factor: 信任区域保持因子（通常为1.0）
        shrink_factor: 信任区域缩小因子
        aggressive_shrink_factor: 激进缩小因子（用于负改进情况）
        max_history_length: 改进历史记录最大长度
        enable_dynamic_adjustment: 是否启用动态调整功能
    """
    high_improvement_threshold: float = 0.3
    medium_improvement_threshold: float = 0.1
    low_improvement_threshold: float = 0.01
    expand_factor: float = 1.5
    maintain_factor: float = 1.0
    shrink_factor: float = 0.5
    aggressive_shrink_factor: float = 0.25
    max_history_length: int = 20
    enable_dynamic_adjustment: bool = True
    
    def __post_init__(self):
        """参数验证"""
        # 验证阈值范围和顺序
        if not (0 < self.low_improvement_threshold < self.medium_improvement_threshold < self.high_improvement_threshold < 1):
            raise ValueError(
                f"Thresholds must satisfy: 0 < low < medium < high < 1, "
                f"got low={self.low_improvement_threshold}, medium={self.medium_improvement_threshold}, "
                f"high={self.high_improvement_threshold}"
            )
        
        # 验证调整因子
        if self.expand_factor <= 1.0:
            raise ValueError(f"expand_factor must be > 1.0, got {self.expand_factor}")
        if self.maintain_factor != 1.0:
            raise ValueError(f"maintain_factor should be 1.0, got {self.maintain_factor}")
        if not (0 < self.shrink_factor < 1.0):
            raise ValueError(f"shrink_factor must be in (0, 1), got {self.shrink_factor}")
        if not (0 < self.aggressive_shrink_factor < self.shrink_factor):
            raise ValueError(
                f"aggressive_shrink_factor must be in (0, shrink_factor), "
                f"got {self.aggressive_shrink_factor}"
            )
        
        # 验证历史长度
        if self.max_history_length <= 0:
            raise ValueError(f"max_history_length must be positive, got {self.max_history_length}")


@dataclass
class TerminationConfig:
    """
    终止条件配置

    用于多层次提前终止机制的配置参数

    Attributes:
        convergence_tolerance: 收敛阈值，曲率违反量小于此值认为收敛
        stagnation_threshold: 改进停滞阈值，改进率低于此值认为停滞
        stagnation_window: 停滞检测窗口大小，连续N次低改进率则终止
        oscillation_threshold: 震荡检测阈值
        engineering_tolerance_factor: 工程阈值因子，工程阈值 = convergence_tolerance * 此因子
        min_iteration_ratio: 最小迭代比例，至少迭代此比例的最大迭代次数
        min_delta: 最小信任区域半径，小于此值认为信任区域耗尽
        max_shrink_count: 最大连续缩小次数，超过此值且无改进则终止
        enable_early_termination: 是否启用提前终止功能
        velocity_threshold: 速度收敛阈值（新增）
        acceleration_threshold: 加速度收敛阈值（新增）
        velocity_severe_threshold: 速度严重违反阈值（新增）
        acceleration_severe_threshold: 加速度严重违反阈值（新增）
        curvature_severe_threshold: 曲率严重违反阈值（新增）
    """
    convergence_tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE
    stagnation_threshold: float = DEFAULT_STAGNATION_THRESHOLD
    stagnation_window: int = DEFAULT_STAGNATION_WINDOW
    oscillation_threshold: float = 0.1
    engineering_tolerance_factor: float = 10.0
    min_iteration_ratio: float = 0.3
    min_delta: float = DEFAULT_MIN_TRUST_REGION
    max_shrink_count: int = DEFAULT_MAX_SHRINK_COUNT
    enable_early_termination: bool = True
    # 新增字段（带默认值，向后兼容）
    velocity_threshold: float = VELOCITY_VIOLATION_THRESHOLD
    acceleration_threshold: float = 1.0
    velocity_severe_threshold: float = 1.0
    acceleration_severe_threshold: float = 10.0
    curvature_severe_threshold: float = 0.1

    def __post_init__(self):
        """参数验证"""
        if self.convergence_tolerance <= 0:
            raise ValueError(f"convergence_tolerance must be positive, got {self.convergence_tolerance}")
        if self.stagnation_threshold <= 0:
            raise ValueError(f"stagnation_threshold must be positive, got {self.stagnation_threshold}")
        if self.stagnation_window <= 0:
            raise ValueError(f"stagnation_window must be positive, got {self.stagnation_window}")
        if self.engineering_tolerance_factor <= 1.0:
            raise ValueError(f"engineering_tolerance_factor must be > 1.0, got {self.engineering_tolerance_factor}")
        if not (0 < self.min_iteration_ratio < 1.0):
            raise ValueError(f"min_iteration_ratio must be in (0, 1), got {self.min_iteration_ratio}")
        if self.min_delta <= 0:
            raise ValueError(f"min_delta must be positive, got {self.min_delta}")
        if self.max_shrink_count <= 0:
            raise ValueError(f"max_shrink_count must be positive, got {self.max_shrink_count}")
        # 新增字段验证
        if self.velocity_threshold <= 0:
            raise ValueError(f"velocity_threshold must be positive, got {self.velocity_threshold}")
        if self.acceleration_threshold <= 0:
            raise ValueError(f"acceleration_threshold must be positive, got {self.acceleration_threshold}")
        if self.velocity_severe_threshold <= self.velocity_threshold:
            raise ValueError(
                f"velocity_severe_threshold must be greater than velocity_threshold, "
                f"got {self.velocity_severe_threshold} <= {self.velocity_threshold}"
            )
        if self.acceleration_severe_threshold <= self.acceleration_threshold:
            raise ValueError(
                f"acceleration_severe_threshold must be greater than acceleration_threshold, "
                f"got {self.acceleration_severe_threshold} <= {self.acceleration_threshold}"
            )
        if self.curvature_severe_threshold <= self.convergence_tolerance:
            raise ValueError(
                f"curvature_severe_threshold must be greater than convergence_tolerance, "
                f"got {self.curvature_severe_threshold} <= {self.convergence_tolerance}"
            )


@dataclass
class ParallelConfig:
    """
    并行计算配置
    
    用于曲率约束并行线性化的配置参数
    
    Attributes:
        num_processes: 并行进程数，None表示使用CPU核心数
        batch_size: 每批次采样点数，用于批次处理避免内存溢出
        enable_batching: 是否启用批次处理
        enable_parallel: 是否启用并行计算
    """
    num_processes: Optional[int] = None
    batch_size: int = DEFAULT_BATCH_SIZE
    enable_batching: bool = True
    enable_parallel: bool = True
    
    def __post_init__(self):
        """参数验证"""
        if self.num_processes is not None and self.num_processes <= 0:
            raise ValueError(f"num_processes must be positive or None, got {self.num_processes}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


@dataclass
class ImprovementRecord:
    """
    改进记录
    
    记录单次迭代的改进信息
    
    Attributes:
        iteration: 迭代次数
        improvement: 改进量（previous_violation - current_violation）
        improvement_ratio: 改进率（improvement / previous_violation）
        violation: 当前曲率违反量
        delta: 当前信任区域半径
        timestamp: 时间戳
    """
    iteration: int
    improvement: float
    improvement_ratio: float
    violation: float
    delta: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class IterationStats:
    """
    迭代统计信息

    记录单次迭代的完整统计信息

    Attributes:
        iteration: 迭代次数
        violation: 曲率违反量
        improvement: 改进量
        improvement_ratio: 改进率
        delta: 信任区域半径
        solve_time: 求解时间（秒）
        timestamp: 时间戳
        velocity_violation: 速度违反量（新增）
        acceleration_violation: 加速度违反量（新增）
        curvature_violation: 曲率违反量（新增）
        workspace_violation: 工作空间违反量（新增）
    """
    iteration: int
    violation: float
    improvement: float
    improvement_ratio: float
    delta: float
    solve_time: float
    timestamp: float = field(default_factory=time.time)
    # 新增字段（带默认值，向后兼容）
    velocity_violation: float = 0.0
    acceleration_violation: float = 0.0
    curvature_violation: float = 0.0
    workspace_violation: float = 0.0


@dataclass
class ViolationReport:
    """
    约束违反量报告（新增）

    统一存储所有约束类型的违反量，用于收敛判断和性能统计

    Attributes:
        velocity_violation: 速度约束违反量
        acceleration_violation: 加速度约束违反量
        curvature_violation: 曲率约束违反量
        workspace_violation: 工作空间约束违反量
        max_violation: 最大违反量
        is_feasible: 是否可行（所有约束都满足）
        severe_violation_type: 严重违反的约束类型（None表示无严重违反）
    """
    velocity_violation: float
    acceleration_violation: float
    curvature_violation: float
    workspace_violation: float
    max_violation: float
    is_feasible: bool
    severe_violation_type: Optional[str] = None

    def __post_init__(self):
        """验证违反量非负"""
        if self.velocity_violation < 0:
            raise ValueError(f"velocity_violation must be non-negative, got {self.velocity_violation}")
        if self.acceleration_violation < 0:
            raise ValueError(f"acceleration_violation must be non-negative, got {self.acceleration_violation}")
        if self.curvature_violation < 0:
            raise ValueError(f"curvature_violation must be non-negative, got {self.curvature_violation}")
        if self.workspace_violation < 0:
            raise ValueError(f"workspace_violation must be non-negative, got {self.workspace_violation}")
        if self.max_violation < 0:
            raise ValueError(f"max_violation must be non-negative, got {self.max_violation}")

    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'velocity': self.velocity_violation,
            'acceleration': self.acceleration_violation,
            'curvature': self.curvature_violation,
            'workspace': self.workspace_violation,
            'max': self.max_violation,
        }


@dataclass
class ConstraintThresholds:
    """
    约束阈值配置（新增）

    定义收敛阈值和严重违反阈值，用于收敛判断和严重违反检测

    Attributes:
        velocity_threshold: 速度收敛阈值
        acceleration_threshold: 加速度收敛阈值
        curvature_threshold: 曲率收敛阈值
        velocity_severe_threshold: 速度严重违反阈值
        acceleration_severe_threshold: 加速度严重违反阈值
        curvature_severe_threshold: 曲率严重违反阈值
    """
    velocity_threshold: float = VELOCITY_VIOLATION_THRESHOLD
    acceleration_threshold: float = 1.0
    curvature_threshold: float = DEFAULT_CONVERGENCE_TOLERANCE
    velocity_severe_threshold: float = 1.0
    acceleration_severe_threshold: float = 10.0
    curvature_severe_threshold: float = 0.1

    def __post_init__(self):
        """验证阈值合理性"""
        if self.velocity_threshold <= 0:
            raise ValueError(f"velocity_threshold must be positive, got {self.velocity_threshold}")
        if self.acceleration_threshold <= 0:
            raise ValueError(f"acceleration_threshold must be positive, got {self.acceleration_threshold}")
        if self.curvature_threshold <= 0:
            raise ValueError(f"curvature_threshold must be positive, got {self.curvature_threshold}")
        if self.velocity_severe_threshold <= self.velocity_threshold:
            raise ValueError(
                f"velocity_severe_threshold must be greater than velocity_threshold, "
                f"got {self.velocity_severe_threshold} <= {self.velocity_threshold}"
            )
        if self.acceleration_severe_threshold <= self.acceleration_threshold:
            raise ValueError(
                f"acceleration_severe_threshold must be greater than acceleration_threshold, "
                f"got {self.acceleration_severe_threshold} <= {self.acceleration_threshold}"
            )
        if self.curvature_severe_threshold <= self.curvature_threshold:
            raise ValueError(
                f"curvature_severe_threshold must be greater than curvature_threshold, "
                f"got {self.curvature_severe_threshold} <= {self.curvature_threshold}"
            )


@dataclass
class PerformanceMetrics:
    """
    性能指标
    
    记录整体性能指标
    
    Attributes:
        total_solve_time: 总求解时间（秒）
        average_iteration_time: 平均迭代时间（秒）
        num_iterations: 迭代次数
        final_violation: 最终曲率违反量
        convergence_rate: 收敛率（final_violation / initial_violation）
        parallel_speedup: 并行加速比（可选）
        parallel_efficiency: 并行效率（可选）
    """
    total_solve_time: float
    average_iteration_time: float
    num_iterations: int
    final_violation: float
    convergence_rate: float
    parallel_speedup: Optional[float] = None
    parallel_efficiency: Optional[float] = None


@dataclass
class SCPConfig:
    """
    SCP（序列凸规划）配置

    Attributes:
        max_iterations: 最大迭代次数
        convergence_tolerance: 收敛阈值（曲率违反量）
        initial_trust_region_radius: 初始信任区域半径
        trust_region_shrink_factor: 信任区域缩小因子
        trust_region_expand_factor: 信任区域扩大因子
        min_trust_region_radius: 最小信任区域半径
        trust_region_config: 信任区域优化配置（可选）
        termination_config: 终止条件优化配置（可选）
        parallel_config: 并行计算配置（可选）
        enable_optimizations: 是否启用所有优化功能
        enable_dynamic_trust_region: 是否启用动态信任区域调整
        enable_early_termination: 是否启用提前终止机制
        enable_parallel_linearization: 是否启用并行线性化
        enable_performance_stats: 是否启用性能统计
    """
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    convergence_tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE
    initial_trust_region_radius: float = DEFAULT_TRUST_REGION_RADIUS
    trust_region_shrink_factor: float = TRUST_REGION_SHRINK_FACTOR
    trust_region_expand_factor: float = TRUST_REGION_EXPAND_FACTOR
    min_trust_region_radius: float = DEFAULT_MIN_TRUST_REGION
    
    # 优化配置
    trust_region_config: Optional[TrustRegionConfig] = None
    termination_config: Optional[TerminationConfig] = None
    parallel_config: Optional[ParallelConfig] = None
    
    # 功能开关
    enable_optimizations: bool = True
    enable_dynamic_trust_region: bool = True
    enable_early_termination: bool = True
    enable_parallel_linearization: bool = True
    enable_performance_stats: bool = True

    def __post_init__(self):
        """参数验证"""
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}")
        if self.convergence_tolerance <= 0:
            raise ValueError(f"convergence_tolerance must be positive, got {self.convergence_tolerance}")
        if self.initial_trust_region_radius <= 0:
            raise ValueError(f"initial_trust_region_radius must be positive, got {self.initial_trust_region_radius}")
        if self.trust_region_shrink_factor <= 0 or self.trust_region_shrink_factor >= 1:
            raise ValueError(f"trust_region_shrink_factor must be in (0, 1), got {self.trust_region_shrink_factor}")
        if self.trust_region_expand_factor <= 1:
            raise ValueError(f"trust_region_expand_factor must be > 1, got {self.trust_region_expand_factor}")
        if self.min_trust_region_radius <= 0:
            raise ValueError(f"min_trust_region_radius must be positive, got {self.min_trust_region_radius}")

    @classmethod
    def fromdict(cls, data: Dict) -> 'SCPConfig':
        """从字典创建实例"""
        return cls(**data)

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
        hdot_min: 时间导数最小值（避免时间倒流）
        full_dim_overlap: 是否使用全维重叠
        hyperellipsoid_num_samples_per_dim_factor: 超椭球采样因子
    """
    order: int = 5
    continuity: int = 1
    hdot_min: float = SMALL_EPSILON
    full_dim_overlap: bool = False
    hyperellipsoid_num_samples_per_dim_factor: int = 32

    def __post_init__(self):
        """参数验证"""
        if self.order < 1:
            raise ValueError(f"order must be >= 1, got {self.order}")
        if self.continuity < 0 or self.continuity >= self.order:
            raise ValueError(f"continuity must be in [0, {self.order-1}], got {self.continuity}")
        if self.hdot_min <= 0:
            raise ValueError(f"hdot_min must be positive, got {self.hdot_min}")

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


# ==================== 曲率惩罚成本相关数据结构 ====================
# [2025-04-06] 曲率成本功能暂时禁用 - 开始
'''
@dataclass
class CurvatureCostConfig:
    """
    曲率成本配置

    Attributes:
        integration_method: 数值积分方法（gauss_legendre/simpson/trapezoid）
        num_integration_points: 积分采样点数
        enable_convex_relaxation: 是否启用凸松弛
        relaxation_method: 松弛方法（quadratic/linear）
        numerical_tolerance: 数值计算容差
    """
    integration_method: str = "gauss_legendre"
    num_integration_points: int = 20
    enable_convex_relaxation: bool = True
    relaxation_method: str = "quadratic"
    numerical_tolerance: float = 1e-6

    def __post_init__(self):
        """参数验证"""
        valid_methods = ["gauss_legendre", "simpson", "trapezoid"]
        if self.integration_method not in valid_methods:
            raise ValueError(
                f"integration_method must be one of {valid_methods}, "
                f"got {self.integration_method}"
            )
        if self.num_integration_points <= 0:
            raise ValueError(
                f"num_integration_points must be positive, "
                f"got {self.num_integration_points}"
            )
        if self.numerical_tolerance <= 0:
            raise ValueError(
                f"numerical_tolerance must be positive, "
                f"got {self.numerical_tolerance}"
            )

    @classmethod
    def fromdict(cls, data: Dict) -> 'CurvatureCostConfig':
        """从字典创建实例"""
        return cls(**data)

    def todict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


@dataclass
class CurvatureCostWeights:
    """
    曲率成本权重

    Attributes:
        curvature_squared: 曲率平方积分权重
        curvature_derivative: 曲率导数平方积分权重
        curvature_peak: 曲率峰值惩罚权重
    """
    curvature_squared: float = 0.0
    curvature_derivative: float = 0.0
    curvature_peak: float = 0.0

    def __post_init__(self):
        """参数验证"""
        if self.curvature_squared < 0:
            raise ValueError(
                f"curvature_squared must be non-negative, "
                f"got {self.curvature_squared}"
            )
        if self.curvature_derivative < 0:
            raise ValueError(
                f"curvature_derivative must be non-negative, "
                f"got {self.curvature_derivative}"
            )
        if self.curvature_peak < 0:
            raise ValueError(
                f"curvature_peak must be non-negative, "
                f"got {self.curvature_peak}"
            )

    def is_enabled(self) -> bool:
        """检查是否启用曲率成本"""
        return (
            self.curvature_squared > 0 or
            self.curvature_derivative > 0 or
            self.curvature_peak > 0
        )

    @classmethod
    def fromdict(cls, data: Dict) -> 'CurvatureCostWeights':
        """从字典创建实例"""
        return cls(**data)

    def todict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
'''
# [2025-04-06] 曲率成本功能暂时禁用 - 结束


@dataclass
class LinearizedCostCoefficients:
    """
    线性化成本系数

    用于SCP迭代中曲率成本的线性化近似，包含泰勒展开的各阶系数

    泰勒展开形式: J ≈ J₀ + gᵀΔP + 0.5 * ΔPᵀHΔP

    Attributes:
        gradient: 梯度向量 g，形状为(n*2,)，n为控制点数量
        hessian_diag: Hessian对角近似，形状为(n*2,)，用于保证正定性
        constant: 常数项 J₀，当前轨迹的成本值
    """
    gradient: np.ndarray
    hessian_diag: np.ndarray
    constant: float

    def __post_init__(self):
        """参数验证"""
        if not isinstance(self.gradient, np.ndarray):
            raise ValueError(f"gradient must be a numpy array, got {type(self.gradient)}")
        if not isinstance(self.hessian_diag, np.ndarray):
            raise ValueError(f"hessian_diag must be a numpy array, got {type(self.hessian_diag)}")
        if self.gradient.shape != self.hessian_diag.shape:
            raise ValueError(
                f"gradient and hessian_diag must have the same shape, "
                f"got gradient.shape={self.gradient.shape}, hessian_diag.shape={self.hessian_diag.shape}"
            )
        if len(self.gradient.shape) != 1:
            raise ValueError(f"gradient must be 1D array, got shape {self.gradient.shape}")
        # 验证Hessian对角正定性（所有元素应为正）
        if np.any(self.hessian_diag <= 0):
            raise ValueError("hessian_diag must be positive definite (all elements > 0)")

    def todict(self) -> Dict:
        """转换为字典"""
        return {
            'gradient': self.gradient.tolist(),
            'hessian_diag': self.hessian_diag.tolist(),
            'constant': self.constant,
        }

    @classmethod
    def fromdict(cls, data: Dict) -> 'LinearizedCostCoefficients':
        """从字典创建实例"""
        return cls(
            gradient=np.array(data['gradient']),
            hessian_diag=np.array(data['hessian_diag']),
            constant=data['constant'],
        )


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

    Attributes:
        max_curvature: 最大曲率
        min_curvature: 最小曲率
        mean_curvature: 平均曲率
        std_curvature: 曲率标准差
        max_curvature_location: 最大曲率位置（参数s）
        curvature_samples: 曲率采样点（可选）
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
        num_iterations: SCP迭代次数
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
