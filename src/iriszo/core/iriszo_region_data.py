"""
自定义IrisZo算法数据结构模块

定义IrisZo算法的核心数据结构,包括IrisZoRegion和IrisZoResult。

作者: Path Planning Team
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np

try:
    from pydrake.geometry.optimization import HPolyhedron
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    HPolyhedron = None


@dataclass
class IrisZoRegion:
    """
    IrisZo生成的单个凸区域

    该类表示一个由自定义IrisZo算法生成的凸多面体区域。
    使用Drake的HPolyhedron作为底层表示。

    Attributes:
        polyhedron: Drake HPolyhedron对象,表示凸多面体
        seed_point: 种子点坐标,shape=(dim,)
        area: 区域面积(2D)或体积(3D+),默认0.0
        centroid: 区域中心坐标,shape=(dim,),默认空数组
        vertices: 顶点数组,shape=(num_vertices, dim),可选
        iteration_count: 生成该区域所需的迭代次数,默认0

    Example:
        >>> import numpy as np
        >>> from pydrake.geometry.optimization import HPolyhedron
        >>>
        >>> # 创建一个简单的矩形区域
        >>> lb = np.array([0.0, 0.0])
        >>> ub = np.array([1.0, 1.0])
        >>> polyhedron = HPolyhedron.MakeBox(lb, ub)
        >>> seed_point = np.array([0.5, 0.5])
        >>>
        >>> region = IrisZoRegion(polyhedron=polyhedron, seed_point=seed_point)
        >>> print(f"区域面积: {region.area}")
        >>> print(f"区域中心: {region.centroid}")
    """

    polyhedron: HPolyhedron
    seed_point: np.ndarray
    area: float = 0.0
    centroid: np.ndarray = field(default_factory=lambda: np.array([]))
    vertices: Optional[np.ndarray] = None
    iteration_count: int = 0

    def __post_init__(self):
        """
        初始化后处理

        计算区域的面积、中心和顶点。
        """
        if not DRAKE_AVAILABLE:
            return

        try:
            # 计算中心(Chebyshev中心)
            self.centroid = self.polyhedron.ChebyshevCenter()
        except Exception:
            # 如果计算失败,使用种子点作为中心
            self.centroid = self.seed_point.copy()

        try:
            # 计算面积(2D)或体积(3D+)
            # Drake的HPolyhedron不能计算精确体积,需要使用采样方法
            # CalcVolumeViaSampling()需要RandomGenerator
            from pydrake.common import RandomGenerator
            generator = RandomGenerator()
            sampled_volume = self.polyhedron.CalcVolumeViaSampling(
                generator=generator,
                desired_rel_accuracy=0.01,  # 1%相对精度
                max_num_samples=10000  # 最大采样数
            )
            self.area = sampled_volume.volume
        except Exception as e:
            # 如果采样方法也失败,尝试使用CalcVolume()
            try:
                self.area = self.polyhedron.CalcVolume()
            except Exception:
                # 如果都失败,保持默认值0.0
                import warnings
                warnings.warn(f"计算区域面积失败: {e}")
                pass

        # Drake的HPolyhedron没有GetVertices()方法
        # 顶点需要通过其他方式获取,暂时保持None
        # 可视化时可以使用其他方法绘制
        self.vertices = None

    def contains(self, point: np.ndarray) -> bool:
        """
        检查点是否在区域内

        Args:
            point: 待检查的点,shape=(dim,)

        Returns:
            True如果点在区域内,False否则

        Example:
            >>> point = np.array([0.5, 0.5])
            >>> region.contains(point)
            True
        """
        if not DRAKE_AVAILABLE:
            return False

        try:
            return self.polyhedron.PointInSet(point)
        except Exception:
            return False

    def get_vertices_ordered(self) -> np.ndarray:
        """
        获取有序顶点(用于可视化)

        对于2D区域,按角度排序顶点。
        对于高维区域,返回原始顶点。

        Returns:
            有序顶点数组,shape=(num_vertices, dim)

        Example:
            >>> vertices = region.get_vertices_ordered()
            >>> # vertices按逆时针顺序排列(2D)
        """
        if self.vertices is None or len(self.vertices) == 0:
            return np.array([])

        # 2D情况:按角度排序
        if self.vertices.shape[1] == 2:
            # 计算每个顶点相对于中心的角度
            angles = np.arctan2(
                self.vertices[:, 1] - self.centroid[1],
                self.vertices[:, 0] - self.centroid[0]
            )
            # 按角度排序
            order = np.argsort(angles)
            return self.vertices[order]

        # 高维情况:返回原始顶点
        return self.vertices

    def get_bounds(self) -> tuple:
        """
        获取区域的边界框

        Returns:
            (lower_bound, upper_bound) 元组

        Example:
            >>> lb, ub = region.get_bounds()
            >>> print(f"下界: {lb}, 上界: {ub}")
        """
        if self.vertices is None or len(self.vertices) == 0:
            return (np.array([]), np.array([]))

        lower_bound = np.min(self.vertices, axis=0)
        upper_bound = np.max(self.vertices, axis=0)
        return (lower_bound, upper_bound)

    def __str__(self) -> str:
        """
        返回区域的字符串表示

        Returns:
            格式化的字符串
        """
        return (
            f"IrisZoRegion(\n"
            f"  seed_point={self.seed_point},\n"
            f"  area={self.area:.6f},\n"
            f"  centroid={self.centroid},\n"
            f"  num_vertices={len(self.vertices) if self.vertices is not None else 0},\n"
            f"  iterations={self.iteration_count}\n"
            f")"
        )


@dataclass
class IrisZoResult:
    """
    IrisZo生成结果

    该类表示自定义IrisZo算法的完整生成结果,包含所有生成的凸区域、
    统计信息和性能数据。

    与iris_pkg.IrisNpResult接口保持一致。

    Attributes:
        regions: 凸区域列表
        num_regions: 区域数量
        total_area: 总面积/体积
        coverage_ratio: 路径覆盖率
        iris_time: IrisZo算法耗时(秒)
        postprocess_time: 后处理耗时(秒)
        total_time: 总耗时(秒)
        config: 配置参数
        cache_hit_rate: 缓存命中率
        pruning_time: 区域修剪耗时(秒)
        pruned_count: 被修剪的区域数量
        iteration_stats: 迭代统计信息

    Example:
        >>> result = generator.generate_from_path(path, obstacle_map, 0.05)
        >>> print(f"生成了 {result.num_regions} 个凸区域")
        >>> print(f"覆盖率: {result.coverage_ratio:.2%}")
        >>> print(f"总耗时: {result.total_time:.2f}秒")
    """

    # 区域数据
    regions: List[IrisZoRegion] = field(default_factory=list)
    num_regions: int = 0

    # 统计数据
    total_area: float = 0.0
    coverage_ratio: float = 0.0

    # 性能数据
    iris_time: float = 0.0
    postprocess_time: float = 0.0
    total_time: float = 0.0

    # 配置
    config: Optional['IrisZoConfig'] = None

    # 缓存统计
    cache_hit_rate: float = 0.0

    # 修剪统计
    pruning_time: float = 0.0
    pruned_count: int = 0

    # 迭代统计
    iteration_stats: dict = field(default_factory=dict)

    # 新增功能结果
    coverage_result: Optional['CoverageResult'] = None
    pruning_result: Optional['PruningResult'] = None
    performance_metrics: Optional['PerformanceMetrics'] = None

    def add_region(self, region: IrisZoRegion):
        """
        添加一个区域

        Args:
            region: 待添加的区域
        """
        self.regions.append(region)
        self.num_regions = len(self.regions)
        self.total_area += region.area

    def calculate_coverage_ratio(self, path_length: int) -> float:
        """
        计算覆盖率

        Args:
            path_length: 路径点总数

        Returns:
            覆盖率
        """
        if path_length == 0:
            return 0.0

        # 统计被覆盖的路径点数量
        # 注意:这里需要外部提供路径点覆盖信息
        # 暂时返回已存储的coverage_ratio
        return self.coverage_ratio

    def get_summary(self) -> dict:
        """
        获取结果摘要

        Returns:
            包含关键统计信息的字典
        """
        return {
            'num_regions': self.num_regions,
            'total_area': self.total_area,
            'coverage_ratio': self.coverage_ratio,
            'iris_time': self.iris_time,
            'postprocess_time': self.postprocess_time,
            'total_time': self.total_time,
            'cache_hit_rate': self.cache_hit_rate,
            'pruned_count': self.pruned_count
        }

    def __str__(self) -> str:
        """
        返回结果的字符串表示

        Returns:
            格式化的字符串
        """
        return (
            f"IrisZoResult(\n"
            f"  区域数量: {self.num_regions}\n"
            f"  总面积: {self.total_area:.6f}\n"
            f"  覆盖率: {self.coverage_ratio:.2%}\n"
            f"  算法耗时: {self.iris_time:.3f}秒\n"
            f"  后处理耗时: {self.postprocess_time:.3f}秒\n"
            f"  总耗时: {self.total_time:.3f}秒\n"
            f"  缓存命中率: {self.cache_hit_rate:.2%}\n"
            f")"
        )

    def __repr__(self) -> str:
        """
        返回结果的详细表示

        Returns:
            详细字符串
        """
        summary = self.get_summary()
        return (
            f"IrisZoResult(\n"
            f"  regions=[{self.num_regions} regions],\n"
            f"  total_area={summary['total_area']:.6f},\n"
            f"  coverage_ratio={summary['coverage_ratio']:.4f},\n"
            f"  total_time={summary['total_time']:.3f}s\n"
            f")"
        )

    def get_coverage_report(self) -> str:
        """
        获取覆盖验证报告

        Returns:
            覆盖验证报告字符串

        Example:
            >>> print(result.get_coverage_report())
        """
        if self.coverage_result is None:
            return "未执行覆盖验证"
        return (
            f"覆盖率: {self.coverage_result.coverage_ratio:.2%}\n"
            f"覆盖点数: {self.coverage_result.covered_points}/{self.coverage_result.total_points}\n"
            f"未覆盖段数: {len(self.coverage_result.uncovered_segments)}"
        )

    def get_pruning_report(self) -> str:
        """
        获取修剪报告

        Returns:
            修剪报告字符串

        Example:
            >>> print(result.get_pruning_report())
        """
        if self.pruning_result is None:
            return "未执行区域修剪"
        return (
            f"原始区域数: {self.pruning_result.original_count}\n"
            f"修剪区域数: {self.pruning_result.pruned_count}\n"
            f"剩余区域数: {self.pruning_result.remaining_count}\n"
            f"修剪比例: {self.pruning_result.pruning_ratio:.2%}"
        )

    def get_performance_report(self, format: str = 'text') -> Union[str, dict]:
        """
        获取性能报告

        Args:
            format: 报告格式，'text'或'json'

        Returns:
            性能报告（字符串或字典）

        Example:
            >>> print(result.get_performance_report())
        """
        if self.performance_metrics is None:
            return "未收集性能数据"

        # 导入PerformanceReporter
        from .iriszo_performance import PerformanceReporter
        reporter = PerformanceReporter()

        if format == 'text':
            return reporter.generate_text_report(self.performance_metrics)
        else:
            return reporter.generate_json_report(self.performance_metrics)
