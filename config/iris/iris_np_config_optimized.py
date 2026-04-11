"""
优化后的 IrisNp 配置类

将参数按功能模块分组，添加详细注释，提高可读性和可维护性。

作者: Path Planning Team
"""

from dataclasses import dataclass


@dataclass
class IrisNpConfigOptimized:
    """
    IrisNp 配置参数（优化版）
    
    参数按功能模块分组：
    1. 碰撞检测参数 - 最重要，影响安全性
    2. 区域生成参数 - 控制区域大小和形状
    3. 种子点参数 - 控制种子点选择和分布
    4. 膨胀参数 - 控制膨胀策略和步长
    5. 性能优化参数 - 控制并行处理和缓存
    6. 区域合并参数 - 控制合并策略
    7. 可视化参数 - 控制输出和调试
    
    使用建议：
    - 首先调整碰撞检测参数，确保安全性
    - 然后调整区域生成参数，优化区域质量
    - 最后调整性能参数，提升处理速度
    """
    
    # ========================================================================
    # 1. 碰撞检测参数 - 最重要，影响安全性
    # ========================================================================
    
    # 概率保证参数
    num_collision_infeasible_samples: int = 50
    """碰撞约束采样数量
    
    说明：在区域内随机采样点进行碰撞检测的次数
    影响：采样次数越多，检测到碰撞的概率越高，安全性越好
    性能：采样次数增加会线性增加计算时间
    推荐值：
        - 一般应用：10-50
        - 高安全要求：50-100
        - 关键应用：100-200
    """
    
    num_additional_constraints_infeasible_samples: int = 50
    """其他约束采样数量
    
    说明：用于验证其他约束（如边界约束）的采样次数
    影响：确保区域满足所有约束条件
    推荐值：与 num_collision_infeasible_samples 相同
    """
    
    configuration_space_margin: float = 0.1
    """配置空间安全裕度（米）
    
    说明：区域与障碍物之间的最小距离
    影响：裕度越大，安全性越高，但可用空间越小
    推荐值：
        - 精确应用：0.05-0.1
        - 一般应用：0.1-0.3
        - 保守应用：0.3-0.5
    """
    
    # 碰撞检测优化参数
    enable_collision_cache: bool = True
    """启用碰撞检测缓存
    
    说明：缓存已检测点的结果，避免重复计算
    影响：大幅提升性能，特别是多次检测相同点时
    推荐值：True（除非内存受限）
    """
    
    collision_cache_size: int = 10000
    """碰撞检测缓存大小
    
    说明：缓存最多存储的检测结果数量
    影响：缓存越大，命中率越高，但内存占用越多
    推荐值：5000-20000
    """
    
    use_batch_collision_check: bool = True
    """使用批量碰撞检测
    
    说明：批量检测多个点，利用向量化优化
    影响：显著提升检测速度
    推荐值：True
    """
    
    # ========================================================================
    # 2. 区域生成参数 - 控制区域大小和形状
    # ========================================================================
    
    initial_region_size: float = 0.1
    """初始区域大小（米）
    
    说明：每个种子点生成的初始区域大小
    影响：初始大小应足够小，确保不与障碍物碰撞
    推荐值：0.05-0.2
    """
    
    max_region_size: float = 100.0
    """最大区域大小（米）
    
    说明：区域膨胀的最大尺寸限制
    影响：防止区域过大，影响后续处理
    推荐值：根据应用场景调整，一般50-200
    """
    
    size_increment: float = 0.1
    """区域膨胀步长（米）- 向后兼容
    
    说明：每次膨胀增加的尺寸
    影响：步长越小，区域边界越精确，但计算时间越长
    推荐值：0.05-0.2
    注意：自适应膨胀模式下此参数被忽略
    """
    
    use_ellipse_expansion: bool = True
    """使用椭圆膨胀 - 向后兼容
    
    说明：使用椭圆形状膨胀区域
    影响：椭圆更适应复杂环境
    推荐值：True
    注意：自适应膨胀模式下此参数被忽略
    """
    
    # ========================================================================
    # 3. 种子点参数 - 控制种子点选择和分布
    # ========================================================================
    
    min_seed_distance: float = 1.0
    """种子点之间的最小距离（米）
    
    说明：相邻种子点之间的最小距离
    影响：距离越大，种子点越稀疏，区域数量越少
    推荐值：0.5-2.0
    """
    
    max_seed_points: int = 50
    """最大种子点数量
    
    说明：从路径中提取的最大种子点数
    影响：种子点越多，区域越多，覆盖越完整
    推荐值：20-100
    """
    
    enable_two_batch_expansion: bool = True
    """启用两批种子点扩张
    
    说明：
        - 第一批：正常扩张，覆盖主要路径
        - 第二批：检查未覆盖路径点，优先沿切线方向膨胀
    影响：提高路径覆盖率，减少未覆盖区域
    推荐值：True
    """
    
    first_batch_seed_interval: int = 5
    """第一批种子点采样间隔
    
    说明：从路径中每隔N个点提取一个种子点
    影响：间隔越小，种子点越密集
    推荐值：3-10
    """
    
    tangent_normal_ratio: float = 2.0
    """切线/法向膨胀比例（第二批种子点）
    
    说明：第二批种子点沿切线方向的膨胀速度是法向的多少倍
    影响：比例越大，区域沿路径方向延伸越长
    推荐值：1.5-3.0
    """
    
    strict_coverage_check: bool = True
    """严格检查路径覆盖
    
    说明：验证所有路径点是否被凸区域覆盖
    影响：确保路径完全在凸区域内
    推荐值：True
    """
    
    # ========================================================================
    # 4. 膨胀参数 - 控制膨胀策略和步长
    # ========================================================================
    
    # 迭代参数
    iteration_limit: int = 100
    """最大迭代次数
    
    说明：区域膨胀的最大迭代次数
    影响：迭代次数越多，区域可能越大，但计算时间越长
    推荐值：50-200
    """
    
    termination_threshold: float = 0.01
    """终止阈值
    
    说明：当区域增长小于此阈值时停止膨胀
    影响：阈值越小，区域越精确
    推荐值：0.001-0.05
    """
    
    relative_termination_threshold: float = 0.01
    """相对终止阈值
    
    说明：当区域增长比例小于此阈值时停止膨胀
    影响：阈值越小，区域越精确
    推荐值：0.001-0.05
    """
    
    # 多方向独立膨胀参数
    use_adaptive_expansion: bool = True
    """使用自适应膨胀
    
    说明：使用真正的IRIS算法，多方向独立膨胀
    影响：生成更自然、更适应环境的凸区域
    推荐值：True
    """
    
    num_expansion_directions: int = 8
    """膨胀方向数量
    
    说明：区域膨胀的方向数（均匀分布在360度）
    影响：方向越多，区域形状越精确，但计算量越大
    推荐值：6-16
    """
    
    direction_tolerance: float = 0.01
    """方向膨胀容差 - 向后兼容
    
    说明：方向膨胀的精度容差
    影响：容差越小，边界越精确
    推荐值：0.001-0.05
    """
    
    # 自适应步长参数
    adaptive_initial_step: float = 1.0
    """自适应膨胀初始步长（米）
    
    说明：自适应膨胀的初始步长
    影响：步长越大，膨胀越快，但可能错过障碍物
    推荐值：0.5-2.0
    """
    
    adaptive_min_step: float = 0.001
    """自适应膨胀最小步长（米）
    
    说明：步长的最小值，防止无限缩小
    影响：最小步长越小，边界越精确
    推荐值：0.0001-0.01
    """
    
    adaptive_step_reduction: float = 0.5
    """步长缩减因子
    
    说明：遇到障碍物时步长缩减的比例
    影响：缩减因子越小，收敛越快，但可能错过精确边界
    推荐值：0.3-0.7
    """
    
    # ========================================================================
    # 5. 性能优化参数 - 控制并行处理和性能
    # ========================================================================
    
    enable_parallel_processing: bool = True
    """启用并行处理
    
    说明：使用多进程并行处理多个种子点
    影响：大幅提升处理速度
    推荐值：True（种子点数>1时）
    """
    
    num_parallel_workers: int = 8
    """并行工作进程数
    
    说明：并行处理的进程数量
    影响：进程数越多，速度越快，但内存占用越多
    推荐值：CPU核心数的1-2倍
    """
    
    # ========================================================================
    # 6. 区域合并参数 - 向后兼容
    # ========================================================================
    
    merge_overlapping_regions: bool = True
    """合并重叠区域 - 向后兼容
    
    说明：合并有重叠的凸区域
    影响：减少区域数量，简化后续处理
    推荐值：根据应用需求
    """
    
    overlap_threshold: float = 0.3
    """重叠阈值 - 向后兼容
    
    说明：判断区域是否重叠的阈值
    影响：阈值越大，合并越激进
    推荐值：0.2-0.5
    """
    
    # ========================================================================
    # 7. 可视化和调试参数
    # ========================================================================
    
    enable_visualization: bool = True
    """启用可视化
    
    说明：生成可视化结果
    影响：便于调试和结果展示
    推荐值：True（调试时），False（生产环境）
    """
    
    region_alpha: float = 0.3
    """区域透明度
    
    说明：可视化时区域的透明度
    影响：透明度越小，越容易看到底层地图
    推荐值：0.2-0.5
    """
    
    enable_profiling: bool = True
    """启用性能分析 - 向后兼容
    
    说明：输出详细的性能统计信息
    影响：便于性能优化和调试
    推荐值：True（调试时），False（生产环境）
    """
    
    verbose: bool = True
    """详细输出
    
    说明：输出详细的处理过程信息
    影响：便于调试和监控
    推荐值：True（调试时），False（生产环境）
    """


# ============================================================================
# 预定义配置模板
# ============================================================================

def get_high_safety_config() -> IrisNpConfigOptimized:
    """获取高安全要求配置
    
    特点：
    - 高采样密度，确保检测到所有障碍物
    - 大安全裕度，增加安全性
    - 精确边界，减少误差
    
    适用场景：
    - 关键路径规划
    - 高精度要求
    - 安全优先
    """
    return IrisNpConfigOptimized(
        # 碰撞检测
        num_collision_infeasible_samples=100,
        num_additional_constraints_infeasible_samples=100,
        configuration_space_margin=0.25,
        
        # 膨胀参数
        adaptive_initial_step=1.0,
        adaptive_min_step=0.0001,
        adaptive_step_reduction=0.3,
        num_expansion_directions=16,
        termination_threshold=0.001,
        
        # 性能
        enable_parallel_processing=True,
        num_parallel_workers=8
    )


def get_fast_processing_config() -> IrisNpConfigOptimized:
    """获取快速处理配置
    
    特点：
    - 低采样密度，快速检测
    - 小安全裕度，最大化空间
    - 大步长，快速膨胀
    
    适用场景：
    - 实时应用
    - 非关键路径
    - 性能优先
    """
    return IrisNpConfigOptimized(
        # 碰撞检测
        num_collision_infeasible_samples=20,
        num_additional_constraints_infeasible_samples=20,
        configuration_space_margin=0.1,
        
        # 膨胀参数
        adaptive_initial_step=2.0,
        adaptive_min_step=0.01,
        adaptive_step_reduction=0.6,
        num_expansion_directions=6,
        iteration_limit=50,
        
        # 性能
        enable_parallel_processing=True,
        num_parallel_workers=16,
        enable_collision_cache=True,
        collision_cache_size=20000
    )


def get_balanced_config() -> IrisNpConfigOptimized:
    """获取平衡配置
    
    特点：
    - 中等采样密度，平衡安全和性能
    - 中等安全裕度
    - 中等步长
    
    适用场景：
    - 一般应用
    - 平衡安全和性能
    """
    return IrisNpConfigOptimized(
        # 碰撞检测
        num_collision_infeasible_samples=50,
        num_additional_constraints_infeasible_samples=50,
        configuration_space_margin=0.2,
        
        # 膨胀参数
        adaptive_initial_step=1.0,
        adaptive_min_step=0.001,
        adaptive_step_reduction=0.5,
        num_expansion_directions=8,
        
        # 性能
        enable_parallel_processing=True,
        num_parallel_workers=8
    )


if __name__ == "__main__":
    print("IrisNp 优化配置类")
    print("=" * 70)
    print("\n预定义配置模板：")
    print("1. get_high_safety_config() - 高安全要求")
    print("2. get_fast_processing_config() - 快速处理")
    print("3. get_balanced_config() - 平衡配置")
    print("\n使用示例：")
    print("  config = get_high_safety_config()")
    print("  generator = IrisNpRegionGenerator(config)")
