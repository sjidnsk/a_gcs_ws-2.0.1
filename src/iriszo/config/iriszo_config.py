"""
自定义IrisZo算法配置模块

定义IrisZo算法的所有配置参数。

作者: Path Planning Team
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IrisZoConfig:
    """
    自定义IrisZo算法配置参数

    该类定义了自定义IrisZo算法的所有可配置参数,包括概率保证参数、
    迭代控制参数、并行化参数等。

    Attributes:
        epsilon: 碰撞体积占比上限ε,控制区域中允许的碰撞比例。
                 取值范围(0, 1),默认1e-3。
        delta: 置信水平δ,控制概率保证的严格程度。
               取值范围(0, 1),默认1e-3。
        iteration_limit: 最大外迭代次数,默认20（实测平均2.5次迭代）。
        bisection_steps: 二分搜索步数,控制零阶方向搜索的精度,默认7（精度约0.8%）。
        num_workers: 并行工作进程数,控制种子点并行处理的进程数,默认8。
        verbose: 详细输出,默认False。
        starting_ellipsoid_radius: 初始椭球体半径,默认0.01。
        num_samples_per_iteration: 每次迭代采样点数,默认100。
        termination_threshold: 终止阈值,当区域增长低于此值时停止,默认1e-3。
        relative_termination_threshold: 相对终止阈值,默认0.05（碰撞比例<5%时终止）。
        configuration_space_margin: 配置空间边界冗余量,默认1e-3。
        enable_two_batch_expansion: 启用两批扩张策略,默认True。
        strict_coverage_check: 严格覆盖检查,默认True。
        enable_cache: 启用碰撞检测缓存,默认True。
        cache_size: 缓存大小,默认10000。

    Example:
        >>> # 使用默认配置
        >>> config = IrisZoConfig()
        >>>
        >>> # 自定义配置
        >>> config = IrisZoConfig(
        ...     epsilon=1e-4,
        ...     delta=1e-6,
        ...     iteration_limit=200,
        ...     verbose=True
        ... )
        >>>
        >>> # 验证配置
        >>> config.validate()
    """

    # 概率保证参数
    epsilon: float = 1e-3
    delta: float = 1e-3

    # 迭代控制参数
    iteration_limit: int = 20  # 实测平均2.5次迭代，最大5次，20提供4倍余量
    bisection_steps: int = 7  # 精度2^-7≈0.8%，对走廊区域足够
    termination_threshold: float = 1e-3
    relative_termination_threshold: float = 0.05  # 碰撞比例<5%时迭代收益极低

    # 采样参数
    num_samples_per_iteration: int = 100

    # 并行化参数
    num_workers: int = 8  # 并行工作进程数

    # 初始椭球体参数
    starting_ellipsoid_radius: float = 0.5  # 从0.01增大到0.5，提供更大初始膨胀空间

    # 边界参数
    configuration_space_margin: float = 1e-3 # 配置空间边界冗余量，防止数值问题导致的边界碰撞误判

    # 终止条件参数（新增）
    min_iterations: int = 2  # 最小迭代次数，防止过早终止
    zero_collision_threshold: int = 2  # 连续多少次碰撞比例为0才终止

    # 策略参数
    enable_two_batch_expansion: bool = True
    strict_coverage_check: bool = True

    # 缓存参数
    enable_cache: bool = True
    cache_size: int = 10000

    # 输出参数
    verbose: bool = False

    # 覆盖半径参数（新增）
    coverage_radius_grids: int = 5  # 覆盖半径栅格数
    coverage_radius_meters: float = 1.0  # 覆盖半径实际距离（米）
    safety_factor: float = 0.8  # 安全余量因子
    min_effective_radius: float = 0.1  # 最小有效半径（米）
    enable_distance_transform: bool = True  # 启用距离变换优化
    enable_batch_processing: bool = True  # 启用批量处理

    # 自适应采样参数
    enable_adaptive_sampling: bool = True  # 启用自适应采样
    adaptive_high_threshold: float = 0.3  # 高碰撞比例阈值（>此值使用完整参数）
    adaptive_low_threshold: float = 0.1  # 低碰撞比例阈值（<此值使用精简参数）
    adaptive_mid_samples: int = 50  # 中期采样点数
    adaptive_low_samples: int = 20  # 后期采样点数
    adaptive_mid_mix_steps: int = 15  # 中期混合步数
    adaptive_low_mix_steps: int = 10  # 后期混合步数
    adaptive_min_samples: int = 10  # 最小采样点数
    adaptive_first_iter_mix_steps: int = 10  # 第1次迭代混合步数（高碰撞时降低，加速采样）

    # 自适应二分搜索参数
    enable_adaptive_bisection: bool = True  # 启用二分搜索精度自适应
    min_bisection_steps: int = 5  # 最小二分搜索步数（精度2^-5≈3.1%，确保不低于安全边距）
    max_bisection_steps: int = 8  # 最大二分搜索步数（精度2^-8≈0.4%）
    adaptive_bisection_distance_scale: float = 2.0  # 距离指标缩放因子

    # 并行二分搜索参数
    enable_parallel_bisection: bool = True  # 启用二分搜索并行化
    parallel_bisection_workers: int = 4  # 并行线程数
    parallel_bisection_threshold: int = 4  # 启用并行的最小碰撞点数

    # Drake调用批量化参数
    enable_drake_batch: bool = True  # 启用Drake调用批量化（A/b缓存持久化+延迟构造）
    point_in_set_tolerance: float = 1e-8  # numpy PointInSet容差

    # Numba JIT加速参数
    enable_numba_jit: bool = True  # 启用Numba JIT加速（Numba不可用时自动禁用）

    # RTree空间索引参数
    enable_rtree_pruning: bool = True  # 启用RTree加速修剪（rtree不可用时自动禁用）
    rtree_leaf_capacity: int = 100  # RTree叶节点容量

    def validate(self) -> bool:
        """
        验证配置参数有效性

        检查所有配置参数是否在有效范围内。

        Returns:
            True如果所有参数有效

        Raises:
            ValueError: 如果任何参数无效

        Example:
            >>> config = IrisZoConfig(epsilon=0.5, delta=0.5)
            >>> config.validate()
            True
            >>>
            >>> config = IrisZoConfig(epsilon=1.5)  # 无效值
            >>> config.validate()  # 抛出ValueError
        """
        # 验证概率保证参数
        if not (0 < self.epsilon < 1):
            raise ValueError(
                f"epsilon必须在(0, 1)范围内,当前值: {self.epsilon}"
            )

        if not (0 < self.delta < 1):
            raise ValueError(
                f"delta必须在(0, 1)范围内,当前值: {self.delta}"
            )

        # 验证迭代控制参数
        if self.iteration_limit <= 0:
            raise ValueError(
                f"iteration_limit必须>0,当前值: {self.iteration_limit}"
            )

        if self.bisection_steps <= 0:
            raise ValueError(
                f"bisection_steps必须>0,当前值: {self.bisection_steps}"
            )

        if self.termination_threshold <= 0:
            raise ValueError(
                f"termination_threshold必须>0,当前值: {self.termination_threshold}"
            )

        if self.relative_termination_threshold <= 0:
            raise ValueError(
                f"relative_termination_threshold必须>0, "
                f"当前值: {self.relative_termination_threshold}"
            )

        # 验证采样参数
        if self.num_samples_per_iteration <= 0:
            raise ValueError(
                f"num_samples_per_iteration必须>0, "
                f"当前值: {self.num_samples_per_iteration}"
            )

        # 验证初始椭球体参数
        if self.starting_ellipsoid_radius <= 0:
            raise ValueError(
                f"starting_ellipsoid_radius必须>0, "
                f"当前值: {self.starting_ellipsoid_radius}"
            )

        # 验证边界参数
        if self.configuration_space_margin < 0:
            raise ValueError(
                f"configuration_space_margin必须>=0, "
                f"当前值: {self.configuration_space_margin}"
            )

        # 验证终止条件参数
        if self.min_iterations < 1:
            raise ValueError(
                f"min_iterations必须>=1,当前值: {self.min_iterations}"
            )

        if self.zero_collision_threshold < 1:
            raise ValueError(
                f"zero_collision_threshold必须>=1,当前值: {self.zero_collision_threshold}"
            )

        # 验证缓存参数
        if self.cache_size <= 0:
            raise ValueError(
                f"cache_size必须>0,当前值: {self.cache_size}"
            )

        # 验证自适应二分搜索参数
        if self.min_bisection_steps < 1:
            raise ValueError(
                f"min_bisection_steps必须>=1,当前值: {self.min_bisection_steps}"
            )

        if self.max_bisection_steps < self.min_bisection_steps:
            raise ValueError(
                f"max_bisection_steps必须>=min_bisection_steps, "
                f"当前值: max={self.max_bisection_steps}, min={self.min_bisection_steps}"
            )

        if self.adaptive_bisection_distance_scale <= 0:
            raise ValueError(
                f"adaptive_bisection_distance_scale必须>0, "
                f"当前值: {self.adaptive_bisection_distance_scale}"
            )

        # 验证并行二分搜索参数
        if self.parallel_bisection_workers < 1:
            raise ValueError(
                f"parallel_bisection_workers必须>=1, "
                f"当前值: {self.parallel_bisection_workers}"
            )

        if self.parallel_bisection_threshold < 1:
            raise ValueError(
                f"parallel_bisection_threshold必须>=1, "
                f"当前值: {self.parallel_bisection_threshold}"
            )

        # 验证Drake调用批量化参数
        if self.point_in_set_tolerance <= 0:
            raise ValueError(
                f"point_in_set_tolerance必须>0, "
                f"当前值: {self.point_in_set_tolerance}"
            )

        # 验证RTree参数
        if self.rtree_leaf_capacity < 1:
            raise ValueError(
                f"rtree_leaf_capacity必须>=1, 当前值: {self.rtree_leaf_capacity}"
            )

        # 验证覆盖半径参数
        self.validate_coverage_config()

        return True

    def validate_coverage_config(self) -> bool:
        """
        验证覆盖判定配置参数

        检查覆盖半径相关参数是否在有效范围内。

        Returns:
            True如果所有参数有效

        Raises:
            ValueError: 如果任何参数无效
        """
        # 验证safety_factor
        if not (0 < self.safety_factor <= 1):
            raise ValueError(
                f"safety_factor必须在(0, 1]范围内，当前值: {self.safety_factor}"
            )

        # 验证min_effective_radius
        if self.min_effective_radius <= 0:
            raise ValueError(
                f"min_effective_radius必须>0，当前值: {self.min_effective_radius}"
            )

        # 验证coverage_radius_grids
        if self.coverage_radius_grids < 1:
            raise ValueError(
                f"coverage_radius_grids必须≥1，当前值: {self.coverage_radius_grids}"
            )

        # 验证coverage_radius_meters
        if self.coverage_radius_meters <= 0:
            raise ValueError(
                f"coverage_radius_meters必须>0，当前值: {self.coverage_radius_meters}"
            )

        # 验证num_workers
        if self.num_workers < 1:
            raise ValueError(
                f"num_workers必须≥1，当前值: {self.num_workers}"
            )

        return True

    def __str__(self) -> str:
        """
        返回配置参数的字符串表示

        Returns:
            配置参数的格式化字符串
        """
        return (
            f"IrisZoConfig(\n"
            f"  概率保证: epsilon={self.epsilon}, delta={self.delta}\n"
            f"  迭代控制: iteration_limit={self.iteration_limit}, "
            f"bisection_steps={self.bisection_steps}\n"
            f"  采样: num_samples_per_iteration={self.num_samples_per_iteration}\n"
            f"  并行: num_workers={self.num_workers}\n"
            f"  初始椭球体: radius={self.starting_ellipsoid_radius}\n"
            f"  终止条件: min_iterations={self.min_iterations}, "
            f"zero_collision_threshold={self.zero_collision_threshold}\n"
            f"  策略: two_batch={self.enable_two_batch_expansion}, "
            f"strict_coverage={self.strict_coverage_check}\n"
            f"  自适应采样: enable={self.enable_adaptive_sampling}, "
            f"high={self.adaptive_high_threshold}, low={self.adaptive_low_threshold}\n"
            f"  自适应二分: enable={self.enable_adaptive_bisection}, "
            f"min_steps={self.min_bisection_steps}, max_steps={self.max_bisection_steps}\n"
            f"  并行二分: enable={self.enable_parallel_bisection}, "
            f"workers={self.parallel_bisection_workers}, threshold={self.parallel_bisection_threshold}\n"
            f"  Drake批量化: enable={self.enable_drake_batch}, "
            f"point_in_set_tol={self.point_in_set_tolerance}\n"
            f"  Numba JIT: enable={self.enable_numba_jit}\n"
            f"  RTree修剪: enable={self.enable_rtree_pruning}, "
            f"leaf_capacity={self.rtree_leaf_capacity}\n"
            f")"
        )


# 预定义配置模板

def get_high_safety_config() -> IrisZoConfig:
    """
    获取高安全配置

    适用于对安全性要求极高的场景。

    Returns:
        高安全配置对象
    """
    return IrisZoConfig(
        epsilon=1e-4,
        delta=1e-6,
        iteration_limit=50,
        bisection_steps=10,
        num_samples_per_iteration=150,
        starting_ellipsoid_radius=0.005,
        strict_coverage_check=True,
        min_bisection_steps=5,
        max_bisection_steps=15,
        parallel_bisection_workers=2,
        point_in_set_tolerance=1e-10
    )


def get_fast_processing_config() -> IrisZoConfig:
    """
    获取快速处理配置

    适用于对速度要求高,可以接受较低安全性的场景。

    Returns:
        快速处理配置对象
    """
    return IrisZoConfig(
        epsilon=1e-2,
        delta=1e-2,
        iteration_limit=20,
        bisection_steps=5,
        num_samples_per_iteration=50,
        starting_ellipsoid_radius=0.02,
        min_iterations=1,
        zero_collision_threshold=1,
        strict_coverage_check=False,
        min_bisection_steps=4,
        max_bisection_steps=7,
        parallel_bisection_workers=4,
        point_in_set_tolerance=1e-7
    )


def get_balanced_config() -> IrisZoConfig:
    """
    获取平衡配置

    在安全性和速度之间取得平衡。

    Returns:
        平衡配置对象
    """
    return IrisZoConfig(
        epsilon=1e-3,
        delta=1e-3,
        iteration_limit=30,
        bisection_steps=7,
        num_samples_per_iteration=100,
        starting_ellipsoid_radius=0.01,
        strict_coverage_check=True,
        min_bisection_steps=5,
        max_bisection_steps=8,
        parallel_bisection_workers=4,
        point_in_set_tolerance=1e-8
    )
