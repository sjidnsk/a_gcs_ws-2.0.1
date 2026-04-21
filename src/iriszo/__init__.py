"""
自定义IrisZo算法模块公共接口

导出所有公共类和函数。

作者: Path Planning Team
"""

# 检查Drake可用性
try:
    from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False

# 导入配置
from .config import (
    IrisZoConfig,
    get_high_safety_config,
    get_fast_processing_config,
    get_balanced_config
)

# 导入数据结构
from .core.iriszo_region_data import IrisZoRegion, IrisZoResult

# 导入碰撞检测
from .core.iriszo_collision import CollisionCheckerAdapter, LRUCache

# 导入核心算法
from .core.iriszo_algorithm import CustomIrisZoAlgorithm

# 导入采样器
from .core.iriszo_sampler import HitAndRunSampler

# 导入二分搜索
from .core.iriszo_bisection import BisectionSearcher

# 导入超平面生成
from .core.iriszo_hyperplane import SeparatingHyperplaneGenerator

# 导入种子点提取
from .core.iriszo_seed_extractor import IrisZoSeedExtractor

# 导入主区域生成器
from .core.iriszo_region import IrisZoRegionGenerator

# 导入覆盖验证
from .core.iriszo_coverage import CoverageValidator, CoverageResult
from .core.iriszo_coverage_validator_enhanced import (
    EnhancedCoverageValidator,
    EnhancedCoverageResult
)
from .core.iriszo_coverage_radius import RadiusCalculator
from .core.iriszo_obstacle_detector import ObstacleDetector
from .core.iriszo_coverage_checker import CoverageChecker
from .core.iriszo_distance_query import (
    DistanceQueryEngine,
    DistanceTransformEngine,
    KDTreeEngine
)

# 导入区域修剪
from .core.iriszo_pruning import RegionPruner, PruningResult, RTreeIndex, RTREE_AVAILABLE

# 导入性能报告
from .core.iriszo_performance import (
    PerformanceReporter,
    PerformanceMetrics,
    PerformanceDataCollector,
    TimeMetrics,
    MemoryMetrics,
    AlgorithmMetrics,
    CacheMetrics
)

# 导入可视化
from .visualization import (
    visualize_iriszo_result,
    visualize_iriszo_result_detailed,
    visualize_region_only
)

__all__ = [
    # 配置
    'IrisZoConfig',
    'get_high_safety_config',
    'get_fast_processing_config',
    'get_balanced_config',

    # 数据结构
    'IrisZoRegion',
    'IrisZoResult',

    # 碰撞检测
    'CollisionCheckerAdapter',

    # 核心算法
    'CustomIrisZoAlgorithm',

    # 采样器
    'HitAndRunSampler',

    # 二分搜索

    # 超平面生成

    # 种子点提取
    'IrisZoSeedExtractor',

    # 主区域生成器
    'IrisZoRegionGenerator',

    # 覆盖验证
    'CoverageValidator',
    'CoverageResult',
    'EnhancedCoverageValidator',
    'EnhancedCoverageResult',
    'RadiusCalculator',
    'CoverageChecker',
    'DistanceQueryEngine',

    # 区域修剪
    'RegionPruner',
    'PruningResult',
    'RTreeIndex',
    'RTREE_AVAILABLE',

    # 性能报告
    'PerformanceReporter',
    'PerformanceMetrics',
    'PerformanceDataCollector',
    'TimeMetrics',
    'MemoryMetrics',
    'AlgorithmMetrics',
    'CacheMetrics',

    # 可视化
    'visualize_iriszo_result',
    'visualize_iriszo_result_detailed',
    'visualize_region_only',

    # Drake可用性
    'DRAKE_AVAILABLE'
]

__version__ = '2.0.0'
