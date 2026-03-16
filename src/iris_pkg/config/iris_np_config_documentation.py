"""
IrisNp 配置参数详细说明文档

本文档详细说明 IrisNpConfig 类中所有参数的含义、影响和推荐值。
参数按功能模块分组，便于理解和调整。

作者: Path Planning Team
日期: 2024
"""

# ============================================================================
# 配置参数分组说明
# ============================================================================

"""
IrisNpConfig 参数分为以下7个主要模块：

1. 碰撞检测参数 - 最重要，直接影响安全性
2. 区域生成参数 - 控制区域的大小和形状
3. 种子点参数 - 控制种子点的选择和分布
4. 膨胀参数 - 控制区域膨胀的策略和步长
5. 性能优化参数 - 控制并行处理和缓存
6. 区域合并参数 - 控制区域合并策略
7. 可视化和调试参数 - 控制输出和调试

使用建议：
- 首先调整碰撞检测参数，确保安全性
- 然后调整区域生成参数，优化区域质量
- 最后调整性能参数，提升处理速度
"""

# ============================================================================
# 1. 碰撞检测参数 - 最重要，影响安全性
# ============================================================================

# 概率保证参数
NUM_COLLISION_INFEASIBLE_SAMPLES = {
    'default': 50,
    'description': '碰撞约束采样数量',
    'detail': '在区域内随机采样点进行碰撞检测的次数',
    'impact': '采样次数越多，检测到碰撞的概率越高，安全性越好',
    'performance': '采样次数增加会线性增加计算时间',
    'recommendations': {
        '一般应用': '10-50',
        '高安全要求': '50-100',
        '关键应用': '100-200'
    }
}

NUM_ADDITIONAL_CONSTRAINTS_INFEASIBLE_SAMPLES = {
    'default': 50,
    'description': '其他约束采样数量',
    'detail': '用于验证其他约束（如边界约束）的采样次数',
    'impact': '确保区域满足所有约束条件',
    'recommendations': '与 num_collision_infeasible_samples 相同'
}

CONFIGURATION_SPACE_MARGIN = {
    'default': 0.2,
    'unit': '米',
    'description': '配置空间安全裕度',
    'detail': '区域与障碍物之间的最小距离',
    'impact': '裕度越大，安全性越高，但可用空间越小',
    'recommendations': {
        '精确应用': '0.05-0.1',
        '一般应用': '0.1-0.3',
        '保守应用': '0.3-0.5'
    }
}

# 碰撞检测优化参数
ENABLE_COLLISION_CACHE = {
    'default': True,
    'description': '启用碰撞检测缓存',
    'detail': '缓存已检测点的结果，避免重复计算',
    'impact': '大幅提升性能，特别是多次检测相同点时',
    'recommendations': 'True（除非内存受限）'
}

COLLISION_CACHE_SIZE = {
    'default': 10000,
    'description': '碰撞检测缓存大小',
    'detail': '缓存最多存储的检测结果数量',
    'impact': '缓存越大，命中率越高，但内存占用越多',
    'recommendations': '5000-20000'
}

USE_BATCH_COLLISION_CHECK = {
    'default': True,
    'description': '使用批量碰撞检测',
    'detail': '批量检测多个点，利用向量化优化',
    'impact': '显著提升检测速度',
    'recommendations': 'True'
}

# ============================================================================
# 2. 区域生成参数 - 控制区域大小和形状
# ============================================================================

INITIAL_REGION_SIZE = {
    'default': 0.1,
    'unit': '米',
    'description': '初始区域大小',
    'detail': '每个种子点生成的初始区域大小',
    'impact': '初始大小应足够小，确保不与障碍物碰撞',
    'recommendations': '0.05-0.2'
}

MAX_REGION_SIZE = {
    'default': 100.0,
    'unit': '米',
    'description': '最大区域大小',
    'detail': '区域膨胀的最大尺寸限制',
    'impact': '防止区域过大，影响后续处理',
    'recommendations': '根据应用场景调整，一般50-200'
}

SIZE_INCREMENT = {
    'default': 0.1,
    'unit': '米',
    'description': '区域膨胀步长（向后兼容）',
    'detail': '每次膨胀增加的尺寸',
    'impact': '步长越小，区域边界越精确，但计算时间越长',
    'recommendations': '0.05-0.2',
    'note': '自适应膨胀模式下此参数被忽略'
}

USE_ELLIPSE_EXPANSION = {
    'default': True,
    'description': '使用椭圆膨胀（向后兼容）',
    'detail': '使用椭圆形状膨胀区域',
    'impact': '椭圆更适应复杂环境',
    'recommendations': 'True',
    'note': '自适应膨胀模式下此参数被忽略'
}

# ============================================================================
# 3. 种子点参数 - 控制种子点的选择和分布
# ============================================================================

MIN_SEED_DISTANCE = {
    'default': 1.0,
    'unit': '米',
    'description': '种子点之间的最小距离',
    'detail': '相邻种子点之间的最小距离',
    'impact': '距离越大，种子点越稀疏，区域数量越少',
    'recommendations': '0.5-2.0'
}

MAX_SEED_POINTS = {
    'default': 50,
    'description': '最大种子点数量',
    'detail': '从路径中提取的最大种子点数',
    'impact': '种子点越多，区域越多，覆盖越完整',
    'recommendations': '20-100'
}

ENABLE_TWO_BATCH_EXPANSION = {
    'default': True,
    'description': '启用两批种子点扩张',
    'detail': '第一批：正常扩张，覆盖主要路径；第二批：检查未覆盖路径点，优先沿切线方向膨胀',
    'impact': '提高路径覆盖率，减少未覆盖区域',
    'recommendations': 'True'
}

FIRST_BATCH_SEED_INTERVAL = {
    'default': 5,
    'description': '第一批种子点采样间隔',
    'detail': '从路径中每隔N个点提取一个种子点',
    'impact': '间隔越小，种子点越密集',
    'recommendations': '3-10'
}

TANGENT_NORMAL_RATIO = {
    'default': 2.0,
    'description': '切线/法向膨胀比例（第二批种子点）',
    'detail': '第二批种子点沿切线方向的膨胀速度是法向的多少倍',
    'impact': '比例越大，区域沿路径方向延伸越长',
    'recommendations': '1.5-3.0'
}

STRICT_COVERAGE_CHECK = {
    'default': True,
    'description': '严格检查路径覆盖',
    'detail': '验证所有路径点是否被凸区域覆盖',
    'impact': '确保路径完全在凸区域内',
    'recommendations': 'True'
}

# ============================================================================
# 4. 膨胀参数 - 控制区域膨胀的策略和步长
# ============================================================================

# 迭代参数
ITERATION_LIMIT = {
    'default': 100,
    'description': '最大迭代次数',
    'detail': '区域膨胀的最大迭代次数',
    'impact': '迭代次数越多，区域可能越大，但计算时间越长',
    'recommendations': '50-200'
}

TERMINATION_THRESHOLD = {
    'default': 0.01,
    'description': '终止阈值',
    'detail': '当区域增长小于此阈值时停止膨胀',
    'impact': '阈值越小，区域越精确',
    'recommendations': '0.001-0.05'
}

RELATIVE_TERMINATION_THRESHOLD = {
    'default': 0.01,
    'description': '相对终止阈值',
    'detail': '当区域增长比例小于此阈值时停止膨胀',
    'impact': '阈值越小，区域越精确',
    'recommendations': '0.001-0.05'
}

# 多方向独立膨胀参数
USE_ADAPTIVE_EXPANSION = {
    'default': True,
    'description': '使用自适应膨胀',
    'detail': '使用真正的IRIS算法，多方向独立膨胀',
    'impact': '生成更自然、更适应环境的凸区域',
    'recommendations': 'True'
}

NUM_EXPANSION_DIRECTIONS = {
    'default': 8,
    'description': '膨胀方向数量',
    'detail': '区域膨胀的方向数（均匀分布在360度）',
    'impact': '方向越多，区域形状越精确，但计算量越大',
    'recommendations': '6-16'
}

DIRECTION_TOLERANCE = {
    'default': 0.01,
    'description': '方向膨胀容差（向后兼容）',
    'detail': '方向膨胀的精度容差',
    'impact': '容差越小，边界越精确',
    'recommendations': '0.001-0.05'
}

# 自适应步长参数
ADAPTIVE_INITIAL_STEP = {
    'default': 1.0,
    'unit': '米',
    'description': '自适应膨胀初始步长',
    'detail': '自适应膨胀的初始步长',
    'impact': '步长越大，膨胀越快，但可能错过障碍物',
    'recommendations': '0.5-2.0'
}

ADAPTIVE_MIN_STEP = {
    'default': 0.001,
    'unit': '米',
    'description': '自适应膨胀最小步长',
    'detail': '步长的最小值，防止无限缩小',
    'impact': '最小步长越小，边界越精确',
    'recommendations': '0.0001-0.01'
}

ADAPTIVE_STEP_REDUCTION = {
    'default': 0.5,
    'description': '步长缩减因子',
    'detail': '遇到障碍物时步长缩减的比例',
    'impact': '缩减因子越小，收敛越快，但可能错过精确边界',
    'recommendations': '0.3-0.7'
}

# ============================================================================
# 5. 性能优化参数 - 控制并行处理和性能
# ============================================================================

ENABLE_PARALLEL_PROCESSING = {
    'default': True,
    'description': '启用并行处理',
    'detail': '使用多进程并行处理多个种子点',
    'impact': '大幅提升处理速度',
    'recommendations': 'True（种子点数>1时）'
}

NUM_PARALLEL_WORKERS = {
    'default': 8,
    'description': '并行工作进程数',
    'detail': '并行处理的进程数量',
    'impact': '进程数越多，速度越快，但内存占用越多',
    'recommendations': 'CPU核心数的1-2倍'
}

# ============================================================================
# 6. 区域合并参数 - 向后兼容
# ============================================================================

MERGE_OVERLAPPING_REGIONS = {
    'default': True,
    'description': '合并重叠区域（向后兼容）',
    'detail': '合并有重叠的凸区域',
    'impact': '减少区域数量，简化后续处理',
    'recommendations': '根据应用需求'
}

OVERLAP_THRESHOLD = {
    'default': 0.3,
    'description': '重叠阈值（向后兼容）',
    'detail': '判断区域是否重叠的阈值',
    'impact': '阈值越大，合并越激进',
    'recommendations': '0.2-0.5'
}

# ============================================================================
# 7. 可视化和调试参数
# ============================================================================

ENABLE_VISUALIZATION = {
    'default': True,
    'description': '启用可视化',
    'detail': '生成可视化结果',
    'impact': '便于调试和结果展示',
    'recommendations': {
        '调试时': 'True',
        '生产环境': 'False'
    }
}

REGION_ALPHA = {
    'default': 0.3,
    'description': '区域透明度',
    'detail': '可视化时区域的透明度',
    'impact': '透明度越小，越容易看到底层地图',
    'recommendations': '0.2-0.5'
}

ENABLE_PROFILING = {
    'default': True,
    'description': '启用性能分析（向后兼容）',
    'detail': '输出详细的性能统计信息',
    'impact': '便于性能优化和调试',
    'recommendations': {
        '调试时': 'True',
        '生产环境': 'False'
    }
}

VERBOSE = {
    'default': True,
    'description': '详细输出',
    'detail': '输出详细的处理过程信息',
    'impact': '便于调试和监控',
    'recommendations': {
        '调试时': 'True',
        '生产环境': 'False'
    }
}

# ============================================================================
# 使用示例
# ============================================================================

"""
# 示例1：高安全要求配置
config = IrisNpConfig(
    num_collision_infeasible_samples=100,
    configuration_space_margin=0.3,
    adaptive_min_step=0.0001,
    num_expansion_directions=12
)

# 示例2：快速处理配置
config = IrisNpConfig(
    num_collision_infeasible_samples=20,
    configuration_space_margin=0.1,
    enable_parallel_processing=True,
    num_parallel_workers=16,
    iteration_limit=50
)

# 示例3：精确边界配置
config = IrisNpConfig(
    adaptive_initial_step=0.5,
    adaptive_min_step=0.0001,
    adaptive_step_reduction=0.3,
    num_expansion_directions=16,
    termination_threshold=0.001
)
"""

if __name__ == "__main__":
    print("IrisNp 配置参数文档")
    print("=" * 70)
    print("\n本模块提供了所有配置参数的详细说明。")
    print("请参考上述文档了解每个参数的含义和推荐值。")
