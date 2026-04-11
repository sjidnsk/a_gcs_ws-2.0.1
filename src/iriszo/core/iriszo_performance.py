"""
性能报告模块

实现性能报告功能，包括数据收集、指标分析、报告生成和导出。

作者: Path Planning Team
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Union
import numpy as np
import time
import datetime
import json
import os

from ..config.iriszo_config import IrisZoConfig


@dataclass
class TimeMetrics:
    """
    时间性能指标

    Attributes:
        total_time: 总耗时（秒）
        sampling_time: 采样耗时（秒）
        collision_check_time: 碰撞检测耗时（秒）
        bisection_time: 二分搜索耗时（秒）
        hyperplane_generation_time: 超平面生成耗时（秒）
        polyhedron_update_time: 多面体更新耗时（秒）
        ellipsoid_computation_time: 椭球计算耗时（秒）
    """

    total_time: float = 0.0
    sampling_time: float = 0.0
    collision_check_time: float = 0.0
    bisection_time: float = 0.0
    hyperplane_generation_time: float = 0.0
    polyhedron_update_time: float = 0.0
    ellipsoid_computation_time: float = 0.0

    def get_breakdown(self) -> dict:
        """
        获取时间分解

        Returns:
            各阶段耗时字典
        """
        return {
            'sampling': self.sampling_time,
            'collision_check': self.collision_check_time,
            'bisection': self.bisection_time,
            'hyperplane_generation': self.hyperplane_generation_time,
            'polyhedron_update': self.polyhedron_update_time,
            'ellipsoid_computation': self.ellipsoid_computation_time
        }


@dataclass
class MemoryMetrics:
    """
    内存使用指标

    Attributes:
        peak_memory_mb: 峰值内存使用（MB）
        cache_memory_mb: 缓存内存使用（MB）
        region_memory_mb: 区域内存使用（MB）
        total_memory_mb: 总内存使用（MB）
    """

    peak_memory_mb: float = 0.0
    cache_memory_mb: float = 0.0
    region_memory_mb: float = 0.0
    total_memory_mb: float = 0.0


@dataclass
class AlgorithmMetrics:
    """
    算法统计指标

    Attributes:
        num_iterations: 迭代次数
        num_samples_total: 总采样数
        num_collision_checks: 碰撞检测次数
        num_hyperplanes_generated: 生成的超平面数
        final_region_volume: 最终区域体积
        convergence_rate: 收敛率
        avg_iteration_time: 平均迭代时间（秒）
    """

    num_iterations: int = 0
    num_samples_total: int = 0
    num_collision_checks: int = 0
    num_hyperplanes_generated: int = 0
    final_region_volume: float = 0.0
    convergence_rate: float = 0.0
    avg_iteration_time: float = 0.0


@dataclass
class CacheMetrics:
    """
    缓存性能指标

    Attributes:
        enabled: 是否启用缓存
        cache_hits: 缓存命中次数
        cache_misses: 缓存未命中次数
        cache_hit_rate: 缓存命中率
        cache_size: 缓存大小
        cache_memory_mb: 缓存内存使用（MB）
    """

    enabled: bool = False
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    cache_size: int = 0
    cache_memory_mb: float = 0.0


@dataclass
class PerformanceMetrics:
    """
    综合性能指标

    Attributes:
        time_metrics: 时间性能指标
        memory_metrics: 内存使用指标
        algorithm_metrics: 算法统计指标
        cache_metrics: 缓存性能指标
        start_time: 开始时间戳
        end_time: 结束时间戳
    """

    time_metrics: TimeMetrics = field(default_factory=TimeMetrics)
    memory_metrics: MemoryMetrics = field(default_factory=MemoryMetrics)
    algorithm_metrics: AlgorithmMetrics = field(default_factory=AlgorithmMetrics)
    cache_metrics: CacheMetrics = field(default_factory=CacheMetrics)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def to_dict(self) -> dict:
        """
        转换为字典

        Returns:
            包含所有指标的字典

        Example:
            >>> metrics_dict = metrics.to_dict()
            >>> print(metrics_dict['time']['total_time'])
        """
        return {
            'time': asdict(self.time_metrics),
            'memory': asdict(self.memory_metrics),
            'algorithm': asdict(self.algorithm_metrics),
            'cache': asdict(self.cache_metrics),
            'start_time': self.start_time,
            'end_time': self.end_time
        }


class PerformanceDataCollector:
    """
    性能数据收集器

    该类用于在算法执行过程中收集性能数据。

    Example:
        >>> collector = PerformanceDataCollector()
        >>> collector.start()
        >>> # ... 执行算法 ...
        >>> collector.record_time('sampling', 0.1)
        >>> collector.stop()
        >>> metrics = collector.get_metrics()
    """

    def __init__(self):
        """
        初始化数据收集器
        """
        self.time_data = {}
        self.memory_data = {}
        self.algorithm_data = {}
        self.cache_data = {}
        self.start_time = None
        self.end_time = None
        self._collecting = False

    def start(self) -> None:
        """
        开始收集数据

        Example:
            >>> collector.start()
        """
        self.start_time = time.time()
        self._collecting = True

    def stop(self) -> None:
        """
        停止收集数据

        Example:
            >>> collector.stop()
        """
        self.end_time = time.time()
        self._collecting = False

    def record_time(self, phase: str, duration: float) -> None:
        """
        记录阶段耗时

        Args:
            phase: 阶段名称
            duration: 耗时（秒）

        Example:
            >>> collector.record_time('sampling', 0.1)
        """
        self.time_data[phase] = self.time_data.get(phase, 0.0) + duration

    def record_memory(self, metric: str, value: float) -> None:
        """
        记录内存使用

        Args:
            metric: 指标名称
            value: 内存使用值（MB）

        Example:
            >>> collector.record_memory('peak', 50.0)
        """
        self.memory_data[metric] = value

    def record_algorithm_stat(self, stat: str, value: Any) -> None:
        """
        记录算法统计

        Args:
            stat: 统计项名称
            value: 统计值

        Example:
            >>> collector.record_algorithm_stat('iterations', 10)
        """
        self.algorithm_data[stat] = value

    def record_cache_stat(self, stat: str, value: Any) -> None:
        """
        记录缓存统计

        Args:
            stat: 统计项名称
            value: 统计值

        Example:
            >>> collector.record_cache_stat('hits', 100)
        """
        self.cache_data[stat] = value

    def get_metrics(self) -> PerformanceMetrics:
        """
        获取收集的性能指标

        Returns:
            PerformanceMetrics对象

        Example:
            >>> metrics = collector.get_metrics()
            >>> print(f"总耗时: {metrics.time_metrics.total_time:.3f}s")
        """
        # 构建TimeMetrics
        time_metrics = TimeMetrics(
            total_time=(self.end_time - self.start_time) if self.start_time and self.end_time else 0.0,
            sampling_time=self.time_data.get('sampling', 0.0),
            collision_check_time=self.time_data.get('collision_check', 0.0),
            bisection_time=self.time_data.get('bisection', 0.0),
            hyperplane_generation_time=self.time_data.get('hyperplane_generation', 0.0),
            polyhedron_update_time=self.time_data.get('polyhedron_update', 0.0),
            ellipsoid_computation_time=self.time_data.get('ellipsoid_computation', 0.0)
        )

        # 构建MemoryMetrics
        memory_metrics = MemoryMetrics(
            peak_memory_mb=self.memory_data.get('peak', 0.0),
            cache_memory_mb=self.memory_data.get('cache', 0.0),
            region_memory_mb=self.memory_data.get('region', 0.0),
            total_memory_mb=self.memory_data.get('total', 0.0)
        )

        # 构建AlgorithmMetrics
        algorithm_metrics = AlgorithmMetrics(
            num_iterations=self.algorithm_data.get('iterations', 0),
            num_samples_total=self.algorithm_data.get('samples', 0),
            num_collision_checks=self.algorithm_data.get('collision_checks', 0),
            num_hyperplanes_generated=self.algorithm_data.get('hyperplanes', 0),
            final_region_volume=self.algorithm_data.get('volume', 0.0),
            convergence_rate=self.algorithm_data.get('convergence_rate', 0.0),
            avg_iteration_time=self.algorithm_data.get('avg_iteration_time', 0.0)
        )

        # 构建CacheMetrics
        cache_metrics = CacheMetrics(
            enabled=self.cache_data.get('enabled', False),
            cache_hits=self.cache_data.get('hits', 0),
            cache_misses=self.cache_data.get('misses', 0),
            cache_hit_rate=self.cache_data.get('hit_rate', 0.0),
            cache_size=self.cache_data.get('size', 0),
            cache_memory_mb=self.cache_data.get('memory', 0.0)
        )

        return PerformanceMetrics(
            time_metrics=time_metrics,
            memory_metrics=memory_metrics,
            algorithm_metrics=algorithm_metrics,
            cache_metrics=cache_metrics,
            start_time=self.start_time,
            end_time=self.end_time
        )


class PerformanceReporter:
    """
    性能报告生成器主类

    该类提供性能报告生成和导出功能。

    Attributes:
        config: IrisZo配置对象
        collector: 性能数据收集器

    Example:
        >>> reporter = PerformanceReporter()
        >>> report = reporter.generate_report(format='text')
        >>> print(report)
    """

    def __init__(
        self,
        config: Optional[IrisZoConfig] = None,
        collector: Optional[PerformanceDataCollector] = None
    ):
        """
        初始化性能报告生成器

        Args:
            config: IrisZo配置对象，可选
            collector: 性能数据收集器，可选
        """
        self.config = config or IrisZoConfig()
        self.collector = collector or PerformanceDataCollector()

    def generate_report(self, format: str = 'text') -> Union[str, dict]:
        """
        生成性能报告

        Args:
            format: 报告格式，'text'或'json'

        Returns:
            报告内容（字符串或字典）

        Raises:
            ValueError: 如果格式不支持

        Example:
            >>> report = reporter.generate_report('text')
            >>> print(report)
        """
        metrics = self.collector.get_metrics()

        if format == 'text':
            return self.generate_text_report(metrics)
        elif format == 'json':
            return self.generate_json_report(metrics)
        else:
            raise ValueError(f"不支持的报告格式: {format}")

    def generate_text_report(self, metrics: Optional[PerformanceMetrics] = None) -> str:
        """
        生成文本格式报告

        Args:
            metrics: 性能指标对象，可选

        Returns:
            文本格式报告

        Example:
            >>> report = reporter.generate_text_report()
            >>> print(report)
        """
        if metrics is None:
            metrics = self.collector.get_metrics()

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        lines = [
            "=" * 80,
            " " * 20 + "IrisZo Performance Report",
            "=" * 80,
            f"Generated at: {timestamp}",
            "",
            "-" * 80,
            " " * 30 + "Time Performance",
            "-" * 80,
            f"Total Time:                    {metrics.time_metrics.total_time:.3f} s",
            f"  Sampling:                    {metrics.time_metrics.sampling_time:.3f} s",
            f"  Collision Check:             {metrics.time_metrics.collision_check_time:.3f} s",
            f"  Bisection Search:            {metrics.time_metrics.bisection_time:.3f} s",
            f"  Hyperplane Generation:       {metrics.time_metrics.hyperplane_generation_time:.3f} s",
            f"  Polyhedron Update:           {metrics.time_metrics.polyhedron_update_time:.3f} s",
            f"  Ellipsoid Computation:       {metrics.time_metrics.ellipsoid_computation_time:.3f} s",
            "",
            "-" * 80,
            " " * 32 + "Memory Usage",
            "-" * 80,
            f"Peak Memory:                   {metrics.memory_metrics.peak_memory_mb:.2f} MB",
            f"  Cache Memory:                {metrics.memory_metrics.cache_memory_mb:.2f} MB",
            f"  Region Memory:               {metrics.memory_metrics.region_memory_mb:.2f} MB",
            "",
            "-" * 80,
            " " * 28 + "Algorithm Statistics",
            "-" * 80,
            f"Iterations:                    {metrics.algorithm_metrics.num_iterations}",
            f"Total Samples:                 {metrics.algorithm_metrics.num_samples_total}",
            f"Collision Checks:              {metrics.algorithm_metrics.num_collision_checks}",
            f"Hyperplanes Generated:         {metrics.algorithm_metrics.num_hyperplanes_generated}",
            f"Final Region Volume:           {metrics.algorithm_metrics.final_region_volume:.3f}",
            f"Convergence Rate:              {metrics.algorithm_metrics.convergence_rate:.4f}",
            f"Avg Iteration Time:            {metrics.algorithm_metrics.avg_iteration_time:.4f} s",
            "",
            "-" * 80,
            " " * 30 + "Cache Performance",
            "-" * 80,
            f"Cache Enabled:                 {'Yes' if metrics.cache_metrics.enabled else 'No'}",
            f"Cache Hits:                    {metrics.cache_metrics.cache_hits}",
            f"Cache Misses:                  {metrics.cache_metrics.cache_misses}",
            f"Cache Hit Rate:                {metrics.cache_metrics.cache_hit_rate:.2%}",
            f"Cache Size:                    {metrics.cache_metrics.cache_size}",
            "",
            "=" * 80
        ]

        return "\n".join(lines)

    def generate_json_report(self, metrics: Optional[PerformanceMetrics] = None) -> dict:
        """
        生成JSON格式报告

        Args:
            metrics: 性能指标对象，可选

        Returns:
            JSON格式报告（字典）

        Example:
            >>> report = reporter.generate_json_report()
            >>> print(report['time']['total_time'])
        """
        if metrics is None:
            metrics = self.collector.get_metrics()
        return metrics.to_dict()

    def export_report(self, filepath: str, format: str = 'text') -> None:
        """
        导出报告到文件

        Args:
            filepath: 文件路径
            format: 报告格式，'text'或'json'

        Example:
            >>> reporter.export_report('report.txt', format='text')
        """
        # 创建目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 生成报告
        report = self.generate_report(format)

        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            if format == 'json':
                json.dump(report, f, indent=2, ensure_ascii=False)
            else:
                f.write(report)
