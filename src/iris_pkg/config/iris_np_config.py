"""
IrisNp 配置模块

包含 IrisNp 算法的配置参数和常量定义。

作者: Path Planning Team
"""

from dataclasses import dataclass
from typing import Optional

# ============================================================================
# 常量定义
# ============================================================================

# 默认配置
DEFAULT_ITERATION_LIMIT = 100  # 迭代次数上限
DEFAULT_TERMINATION_THRESHOLD = 0.01  # 终止阈值
DEFAULT_REGION_SIZE = 0.5  # 区域大小
DEFAULT_MAX_REGION_SIZE = 10.0  # 最大区域大小
DEFAULT_NUM_DIRECTIONS = 8  # 方向数量

# 性能相关
DEFAULT_CACHE_SIZE = 10000  # 缓存大小
DEFAULT_NUM_WORKERS = 8  # 工作线程数量


@dataclass
class IrisNpConfig:
    """IrisNp 配置参数"""
    # 迭代参数
    iteration_limit: int = 1000  # 迭代次数上限
    termination_threshold: float = 0.01  # 终止阈值
    relative_termination_threshold: float = 0.01  # 相对终止阈值

    # 概率保证参数（向后兼容）
    num_collision_infeasible_samples: int = 50  # 碰撞不可行样本数量
    num_additional_constraints_infeasible_samples: int = 50  # 额外的约束不可行样本数量
    configuration_space_margin: float = 0.2  # 配置空间边界的冗余

    # 种子点选择
    min_seed_distance: float = 1.0  # 种子点之间的最小距离
    max_seed_points: int = 50  # 最大种子点数量

    # 区域生成参数
    initial_region_size: float = 0.1  # 初始区域大小
    max_region_size: float = 100.0  # 最大区域大小
    size_increment: float = 0.1  # 区域膨胀步长（向后兼容）
    use_ellipse_expansion: bool = True  # 向后兼容

    # 多方向独立膨胀
    # use_adaptive_expansion: bool = True  # 使用自适应膨胀（向后兼容）
    num_expansion_directions: int = 32  # 膨胀方向数量
    direction_tolerance: float = 0.01  # 方向膨胀容差（向后兼容）

    use_adaptive_expansion = True
    use_ellipse_expansion = False
    # enable_two_batch_expansion = True

    # 区域合并（向后兼容）
    merge_overlapping_regions: bool = True
    overlap_threshold: float = 0.3

    # 碰撞检测优化
    enable_collision_cache: bool = True  # 启用碰撞检测缓存
    collision_cache_size: int = 1000000  # 向后兼容 # 碰撞检测缓存大小
    use_batch_collision_check: bool = True  # 向后兼容 # 启用批量碰撞检测

    # 可视化（向后兼容）
    enable_visualization: bool = True
    region_alpha: float = 0.3

    # 性能监控
    enable_profiling: bool = True  # 向后兼容
    verbose: bool = True  # 详细输出

    # 性能优化
    enable_parallel_processing: bool = True  # 启用并行处理
    num_parallel_workers: int = 8  # 并行工作进程数

    # 自适应步长参数
    adaptive_initial_step: float = 1.0  # 初始步长（米）
    adaptive_min_step: float = 0.001  # 最小步长（米）
    adaptive_step_reduction: float = 0.2  # 步长缩减因子

    # 两批种子点扩张参数
    enable_two_batch_expansion: bool = True  # 启用两批种子点扩张
    first_batch_seed_interval: int = 1  # 第一批种子点采样间隔
    tangent_normal_ratio: float = 2.0  # 切线/法向膨胀比例（第二批）
    strict_coverage_check: bool = True  # 严格检查路径覆盖

    # 椭圆膨胀参数
    ellipse_aspect_ratio: float = 1.5  # 椭圆长宽比

    # Voronoi优化参数
    enable_voronoi_optimization: bool = False  # 启用Voronoi优化（在两批扩张中）
    enable_voronoi_only_mode: bool = True  # 启用Voronoi优化模式（实验性，仅使用Voronoi）
    voronoi_max_iterations: int = 10  # Voronoi优化最大迭代次数
    voronoi_max_new_seeds: int = 20  # Voronoi优化最多新增种子点数量
