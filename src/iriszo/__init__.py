"""
Public API for the custom IrisZo implementation.

Exports are loaded lazily so importing ``iriszo`` or ``IrisZoConfig`` does not
pull in Drake, Matplotlib, diagnostics, or the full generation stack.
"""

from importlib import import_module
from typing import Any, Dict

from .availability import check_drake_availability


_EXPORTS: Dict[str, str] = {
    # 配置
    'IrisZoConfig': 'iriszo.config',
    'get_high_safety_config': 'iriszo.config',
    'get_fast_processing_config': 'iriszo.config',
    'get_balanced_config': 'iriszo.config',

    # 数据结构与几何工具
    'IrisZoRegion': 'iriszo.geometry',
    'IrisZoResult': 'iriszo.geometry',
    'CollisionCheckerAdapter': 'iriszo.geometry',
    'LRUCache': 'iriszo.geometry',
    'HitAndRunSampler': 'iriszo.geometry',
    'BisectionSearcher': 'iriszo.geometry',
    'SeparatingHyperplaneGenerator': 'iriszo.geometry',

    # 区域生成
    'CustomIrisZoAlgorithm': 'iriszo.generation',
    'IrisZoSeedExtractor': 'iriszo.generation',
    'IrisZoRegionGenerator': 'iriszo.generation',

    # 覆盖验证
    'CoverageValidator': 'iriszo.validation',
    'CoverageResult': 'iriszo.validation',
    'EnhancedCoverageValidator': 'iriszo.validation',
    'EnhancedCoverageResult': 'iriszo.validation',
    'RadiusCalculator': 'iriszo.validation',
    'ObstacleDetector': 'iriszo.validation',
    'CoverageChecker': 'iriszo.validation',
    'DistanceQueryEngine': 'iriszo.validation',
    'DistanceTransformEngine': 'iriszo.validation',
    'KDTreeEngine': 'iriszo.validation',

    # 诊断
    'RegionPruner': 'iriszo.diagnostics',
    'PruningResult': 'iriszo.diagnostics',
    'RTreeIndex': 'iriszo.diagnostics',
    'RTREE_AVAILABLE': 'iriszo.diagnostics',
    'PerformanceReporter': 'iriszo.diagnostics',
    'PerformanceMetrics': 'iriszo.diagnostics',
    'PerformanceDataCollector': 'iriszo.diagnostics',
    'TimeMetrics': 'iriszo.diagnostics',
    'MemoryMetrics': 'iriszo.diagnostics',
    'AlgorithmMetrics': 'iriszo.diagnostics',
    'CacheMetrics': 'iriszo.diagnostics',

    # 可视化
    'visualize_iriszo_result': 'iriszo.visualization',
    'visualize_iriszo_result_detailed': 'iriszo.visualization',
    'visualize_region_only': 'iriszo.visualization',
}

__all__ = [*_EXPORTS, 'DRAKE_AVAILABLE', 'check_drake_availability']

__version__ = '2.0.0'


def __getattr__(name: str) -> Any:
    if name == 'DRAKE_AVAILABLE':
        return check_drake_availability()

    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'iriszo' has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))
