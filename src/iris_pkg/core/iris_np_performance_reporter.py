"""
IrisNp 性能报告模块

负责生成和输出性能报告，包括区域统计、性能指标和配置信息。

作者: Path Planning Team
"""

from config.iris import IrisNpConfig
from .iris_np_region_data import IrisNpResult
from .iris_np_collision import SimpleCollisionCheckerForIrisNp


class IrisNpPerformanceReporter:
    """IrisNp 性能报告生成器"""

    def __init__(self, config: IrisNpConfig):
        """
        初始化性能报告生成器

        Args:
            config: IrisNp 配置参数
        """
        self.config = config

    def print_performance_report(
        self,
        result: IrisNpResult,
        checker: SimpleCollisionCheckerForIrisNp
    ):
        """打印性能报告"""
        print("\n" + "="*70)
        print("IrisNp 处理完成")
        print("="*70)
        print(f"总区域数: {result.num_regions}")
        print(f"总面积: {result.total_area:.2f} 平方米")
        print(f"覆盖率: {result.coverage_ratio*100:.1f}%")
        print(f"\n性能统计:")
        print(f"  - 总耗时: {result.total_time:.4f}秒")
        print(f"  - IrisNp 耗时: {result.iris_time:.4f}秒")
        print(f"  - 后处理耗时: {result.postprocess_time:.4f}秒")

        if self.config.enable_collision_cache:
            cache_stats = checker.get_cache_stats()
            print(f"  - 缓存命中率: {cache_stats['hit_rate']*100:.1f}%")
            print(f"  - 缓存大小: {cache_stats['cache_size']}")

        print(f"\n配置信息:")
        print(f"  - 膨胀模式: {'椭圆' if self.config.use_ellipse_expansion else '自适应膨胀'}")
        print(f"  - 初始区域大小: {self.config.initial_region_size}")
        print(f"  - 最大区域大小: {self.config.max_region_size}")
        print(f"  - 膨胀步长: {self.config.size_increment}")
