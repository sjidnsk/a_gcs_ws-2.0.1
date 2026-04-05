"""
基于 Drake IrisNp 的凸区域生成模块

使用 IrisNp (IRIS for Configuration Space with Nonlinear Programming) 算法
在配置空间中生成无碰撞凸区域。

IrisNp 是 Drake 中实际存在的算法，专门用于机器人配置空间。

核心优势：
1. 直接在配置空间工作
2. 使用非线性优化处理碰撞
3. 自动处理非凸障碍物
4. 提供概率保证的无碰撞区域

配置说明：
- 使用 IrisNpConfigOptimized 进行配置（推荐）
- 提供预定义配置模板：高安全、快速处理、平衡配置
- 详细参数说明见 iris_np_config_documentation.py

重构说明：
- 已将功能模块提取到独立文件中：
  - iris_np_seed_extractor.py: 种子点提取
  - iris_np_processor.py: 种子点处理（串行/并行）
  - iris_np_coverage_checker.py: 路径覆盖验证
  - iris_np_performance_reporter.py: 性能报告生成
  - iris_np_region_pruner.py: 区域修剪（移除被完全覆盖的冗余区域）

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import warnings

from pydrake.geometry.optimization import HPolyhedron

# 导入优化配置
try:
    from ..config.iris_np_config_optimized import (
        IrisNpConfigOptimized,
        get_high_safety_config,
        get_fast_processing_config,
        get_balanced_config
    )
    OPTIMIZED_CONFIG_AVAILABLE = True
except ImportError:
    OPTIMIZED_CONFIG_AVAILABLE = False
    warnings.warn("优化配置模块未找到，使用默认配置")

# 导入内部模块
from ..config.iris_np_config import IrisNpConfig
from .iris_np_region_data import IrisNpRegion, IrisNpResult
from .iris_np_collision import SimpleCollisionCheckerForIrisNp
from .iris_np_expansion import IrisNpExpansion
from .iris_np_seed_extractor import IrisNpSeedExtractor
from .iris_np_processor import IrisNpProcessor
from .iris_np_coverage_checker import IrisNpCoverageChecker
from .iris_np_performance_reporter import IrisNpPerformanceReporter
from .iris_np_voronoi_optimizer import VoronoiSeedOptimizer
from .iris_np_region_pruner import RegionPruner, PruningResult


# 尝试导入 Drake
try:
    from pydrake.geometry.optimization import IrisNp, IrisOptions

    # MultibodyPlant 和 Context 的正确导入路径
    try:
        from pydrake.multibody.plant import MultibodyPlant
    except ImportError:
        from pydrake.multibody import MultibodyPlant

    try:
        from pydrake.systems.framework import Context
    except ImportError:
        from pydrake.systems import Context

    DRAKE_AVAILABLE = True
except ImportError as e:
    DRAKE_AVAILABLE = False
    warnings.warn(
        f"Drake (pydrake) 导入失败: {e}\n"
        "IrisNp 功能将不可用。\n"
        "安装方法: pip install drake"
    )


class IrisNpRegionGenerator:
    """
    基于 IrisNp 的凸区域生成器

    这是模块的主类，负责协调各个子模块完成凸区域生成任务。

    工作流程：
    1. 初始化：创建碰撞检测器、种子点提取器、处理器、覆盖验证器等子模块
    2. 生成区域：从路径中提取种子点，使用 IrisNp 算法生成凸区域
    3. 验证覆盖：检查路径是否被凸区域完全覆盖
    4. 输出结果：返回包含所有凸区域和性能统计的结果对象

    支持两种模式：
    - 单批扩张模式：一次性提取所有种子点并生成区域（向后兼容）
    - 两批扩张模式：分两批提取种子点，第二批专注于未覆盖的路径点（推荐）

    Attributes:
        config: IrisNp 配置参数
        expansion: IrisNp 膨胀器，负责执行实际的区域膨胀算法
        seed_extractor: 种子点提取器，负责从路径中提取种子点
        processor: 种子点处理器，负责处理种子点并生成凸区域
        coverage_checker: 覆盖验证器，负责验证路径是否被完全覆盖
        reporter: 性能报告生成器，负责生成性能统计报告
    """

    def __init__(self, config: Optional[IrisNpConfig] = None):
        """
        初始化 IrisNp 凸区域生成器

        Args:
            config: IrisNp 配置参数，如果为 None 则使用默认配置

        Raises:
            RuntimeError: 如果 Drak e 未安装

        Note:
            初始化时会创建以下子模块：
            - IrisNpExpansion: 执行区域膨胀的核心算法
            - IrisNpSeedExtractor: 从路径中提取种子点
            - IrisNpProcessor: 处理种子点并生成凸区域
            - IrisNpCoverageChecker: 验证路径覆盖情况
            - IrisNpPerformanceReporter: 生成性能报告
        """
        # 检查 Drake 是否可用
        if not DRAKE_AVAILABLE:
            raise RuntimeError(
                "Drake 未安装。请使用以下命令安装：\n"
                "pip install drake"
            )

        # 初始化配置
        self.config = config or IrisNpConfig()

        # 初始化核心膨胀器（负责执行 IrisNp 算法）
        self.expansion = IrisNpExpansion(self.config)

        # 初始化子模块（各司其职，提高代码可维护性）
        self.seed_extractor = IrisNpSeedExtractor(self.config)
        self.processor = IrisNpProcessor(self.config, self.expansion)
        self.coverage_checker = IrisNpCoverageChecker(self.config, self.expansion)
        self.reporter = IrisNpPerformanceReporter(self.config)
        self.pruner = RegionPruner(
            verbose=self.config.verbose,
            use_rtree=self.config.enable_region_pruning,
            sample_resolution=self.config.pruning_sample_resolution
        )

        # 初始化Voronoi优化器（可选）
        self.voronoi_optimizer = None

    def generate_from_path(
        self,
        path: List[Tuple[float, float, float]],
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float] = (0.0, 0.0)
    ) -> IrisNpResult:
        """
        从路径生成凸区域（使用 IrisNp）

        这是模块的主入口方法，协调整个凸区域生成流程。

        工作流程：
        1. 创建碰撞检测器（支持缓存优化）
        2. 定义搜索域（边界框）
        3. 根据配置选择单批或两批扩张模式
        4. 提取种子点并生成凸区域
        5. 验证路径覆盖情况
        6. 收集性能统计并生成报告

        支持两批种子点扩张策略（推荐）：
        - 第一批：正常扩张，均匀采样路径点
        - 第二批：检查未覆盖路径点，优先沿路径切线方向膨胀
        - 第三批（可选）：针对仍未覆盖的点生成小区域

        Args:
            path: 路径点列表，每个元素为 (x, y, theta)
                  x, y: 位置坐标（米）
                  theta: 朝向角度（弧度）
            obstacle_map: 障碍物地图，2D numpy 数组
                          0 表示自由空间，1 表示障碍物
            resolution: 地图分辨率（米/像素）
            origin: 地图原点坐标 (x, y)，默认为 (0.0, 0.0)

        Returns:
            IrisNpResult: 生成的凸区域结果，包含：
                - regions: 生成的凸区域列表
                - num_regions: 区域数量
                - total_area: 总面积
                - coverage_ratio: 覆盖率
                - iris_time: IrisNp 算法耗时
                - postprocess_time: 后处理耗时
                - total_time: 总耗时
                - cache_hit_rate: 缓存命中率（如果启用）

        Example:
            >>> from src.iris_pkg.core.iris_np_region import IrisNpRegionGenerator
            >>> generator = IrisNpRegionGenerator()
            >>> result = generator.generate_from_path(path, obstacle_map, 0.05, (0.0, 0.0))
            >>> print(f"生成了 {result.num_regions} 个凸区域")
        """
        import time
        total_start = time.time()

        # 初始化结果对象
        result = IrisNpResult(config=self.config)

        # ========================================================================
        # Step 1: 创建碰撞检测器
        # ========================================================================
        # 碰撞检测器负责检查点是否在障碍物内
        # 支持缓存优化，大幅提升重复查询的性能
        if self.config.verbose:
            print("\n" + "="*70)
            print("Step 1: 创建碰撞检测器")
            print("="*70)

        checker = SimpleCollisionCheckerForIrisNp(
            obstacle_map, resolution, origin,
            enable_cache=self.config.enable_collision_cache,
            cache_size=self.config.collision_cache_size
        )

        if self.config.verbose:
            print("碰撞检测器创建完成")
            print(f"  - 缓存启用: {self.config.enable_collision_cache}")
            print(f"  - 批量检测: {self.config.use_batch_collision_check}")

        # ========================================================================
        # Step 2: 定义搜索域（边界框）
        # ========================================================================
        # 搜索域限制了凸区域的最大范围，避免无限膨胀
        # 使用 HPolyhedron.MakeBox 创建矩形边界框
        domain = self._create_domain(obstacle_map, resolution, origin)

        # ========================================================================
        # Step 3-5: 根据配置选择扩张模式
        # ========================================================================
        # 三种模式：
        # 1. 两批扩张模式（推荐）：分两批提取种子点，提高覆盖质量
        # 2. 单批扩张模式（向后兼容）：一次性提取所有种子点
        # 3. Voronoi优化模式（实验性）：仅使用Voronoi优化，数学最优

        if self.config.enable_voronoi_only_mode:
            # ========== Voronoi优化模式（实验性） ==========
            # 优势：
            # - 数学最优性：基于最大空圆定理
            # - 几何自适应性：自动适应障碍物和路径几何
            # - 种子点最少：避免冗余采样
            # 缺点：
            # - 计算复杂度高：需要迭代生成Voronoi图
            # - 依赖初始种子点：需要至少3个种子点
            regions = self._process_voronoi_only_mode(
                path, obstacle_map, resolution, origin, checker, domain, result
            )
        elif self.config.enable_two_batch_expansion:
            # ========== 两批扩张模式（推荐） ==========
            # 优势：
            # - 第一批快速覆盖主要路径
            # - 第二批针对性处理未覆盖区域
            # - 第三轮（可选）确保完全覆盖
            regions = self._process_two_batch_expansion(
                path, obstacle_map, resolution, origin, checker, domain, result
            )
        else:
            # ========== 单批扩张模式（向后兼容） ==========
            # 优势：
            # - 逻辑简单，易于理解
            # - 适用于简单场景
            regions = self._process_single_batch_expansion(
                path, obstacle_map, resolution, origin, checker, domain, result
            )

        # ========================================================================
        # Step 6: 后处理
        # ========================================================================
        postprocess_start = time.time()

        # 验证路径覆盖
        # 检查路径上的每个点是否至少在一个凸区域内
        if self.config.strict_coverage_check:
            coverage_result = self.coverage_checker.verify_path_coverage(path, regions)
            if self.config.verbose:
                print(f"\n路径覆盖验证: {'通过' if coverage_result else '未通过'}")

        # 区域修剪：移除被完全覆盖的冗余区域
        # 在路径覆盖验证之后进行，确保修剪不会影响路径覆盖
        if self.config.enable_region_pruning:
            pruning_result = self.pruner.prune(regions)
            regions = pruning_result.pruned_regions
            result.pruning_time = pruning_result.pruning_time
            result.pruned_count = pruning_result.removed_count
        else:
            result.pruning_time = 0.0
            result.pruned_count = 0

        result.postprocess_time = time.time() - postprocess_start

        # ========================================================================
        # Step 7: 收集性能统计
        # ========================================================================
        # 碰撞检测缓存统计
        if self.config.enable_collision_cache:
            cache_stats = checker.get_cache_stats()
            result.cache_hit_rate = cache_stats['hit_rate']

        # 区域统计
        result.regions = regions
        result.num_regions = len(regions)
        result.total_area = sum(r.area for r in regions)

        # 覆盖率计算
        # 覆盖率 = 凸区域总面积 / 自由空间总面积
        total_free_space = np.sum(obstacle_map == 0) * resolution * resolution
        result.coverage_ratio = result.total_area / total_free_space if total_free_space > 0 else 0.0

        # 总耗时
        result.total_time = time.time() - total_start

        # ========================================================================
        # Step 8: 输出性能报告
        # ========================================================================
        if self.config.enable_profiling:
            self.reporter.print_performance_report(result, checker)

        return result

    def _process_two_batch_expansion(
        self,
        path: List[Tuple[float, float, float]],
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        checker: SimpleCollisionCheckerForIrisNp,
        domain: HPolyhedron,
        result: IrisNpResult
    ) -> List[IrisNpRegion]:
        """
        处理两批扩张模式

        这是最推荐的扩张策略，分三个阶段完成路径覆盖：

        第一阶段：正常扩张
        - 均匀采样路径点作为种子点
        - 使用标准膨胀参数生成凸区域
        - 快速覆盖路径的主要部分

        第二阶段：针对性扩张
        - 检查路径上未被覆盖的点
        - 在未覆盖点附近提取新的种子点
        - 使用各向异性膨胀（沿切线方向优先）

        第三阶段：补充扩张（可选）
        - 如果仍有未覆盖点，进行聚类
        - 为每个簇生成小区域
        - 确保路径完全被覆盖

        Args:
            path: 路径点列表
            obstacle_map: 障碍物地图
            resolution: 地图分辨率
            origin: 地图原点
            checker: 碰撞检测器
            domain: 搜索域
            result: 结果对象（用于记录耗时）

        Returns:
            生成的凸区域列表

        Note:
            两批扩张的优势：
            1. 提高覆盖质量：第二批针对性处理遗漏区域
            2. 减少区域数量：避免过度重叠
            3. 优化计算效率：第一批快速，第二批精准
        """
        import time

        # ========================================================================
        # 第一批：正常扩张
        # ========================================================================
        # 目标：快速覆盖路径的主要部分
        # 策略：均匀采样路径点，使用标准膨胀参数

        if self.config.verbose:
            print("\n" + "="*70)
            print("Step 2: 第一批种子点提取（正常扩张）")
            print("="*70)

        # 提取第一批种子点
        # batch=1 表示第一批，使用均匀采样策略
        first_batch_seeds = self.seed_extractor.extract_seed_points(
            path, obstacle_map, resolution, origin, batch=1
        )

        if self.config.verbose:
            print(f"第一批提取了 {len(first_batch_seeds)} 个种子点")

        # 执行第一批 IrisNp 算法
        if self.config.verbose:
            print("\n" + "="*70)
            print("Step 3: 第一批执行 IrisNp 算法（正常扩张）")
            print("="*70)
            print(f"膨胀模式: {'椭圆' if self.config.use_ellipse_expansion else '方形'}")
            print(f"并行处理: {'启用' if self.config.enable_parallel_processing else '禁用'}")
            if self.config.enable_parallel_processing:
                print(f"工作进程数: {self.config.num_parallel_workers}")

        iris_start = time.time()

        # 处理种子点，生成凸区域
        # processor 会自动选择串行或并行处理模式
        first_batch_regions = self.processor.process_seeds(
            first_batch_seeds, checker, domain, obstacle_map, resolution, origin
        )

        result.iris_time = time.time() - iris_start

        if self.config.verbose:
            print(f"\n第一批 IrisNp 完成，耗时: {result.iris_time:.4f}秒")
            print(f"成功生成 {len(first_batch_regions)} 个凸区域")

        # ========================================================================
        # Voronoi优化（可选）
        # ========================================================================
        # 目标：基于Voronoi图优化种子点位置，提高覆盖效率
        # 策略：
        # 1. 提取当前种子点
        # 2. 生成Voronoi图
        # 3. 评估Voronoi顶点
        # 4. 添加最优顶点作为新种子点

        if self.config.enable_voronoi_optimization and len(first_batch_regions) >= 3:
            if self.config.verbose:
                print("\n" + "="*70)
                print("Voronoi优化：基于最大空圆定理优化种子点")
                print("="*70)

            # 初始化Voronoi优化器
            if self.voronoi_optimizer is None:
                self.voronoi_optimizer = VoronoiSeedOptimizer(
                    obstacle_map, resolution, origin, self.coverage_checker
                )

            # 提取当前种子点
            current_seeds = [region.seed_point for region in first_batch_regions]

            # Voronoi优化
            voronoi_start = time.time()
            optimized_seeds = self.voronoi_optimizer.optimize(
                current_seeds, path,
                max_iterations=self.config.voronoi_max_iterations,
                max_new_seeds=self.config.voronoi_max_new_seeds
            )
            voronoi_time = time.time() - voronoi_start

            if self.config.verbose:
                print(f"Voronoi优化完成，耗时: {voronoi_time:.4f}秒")
                print(f"种子点数量: {len(current_seeds)} → {len(optimized_seeds)}")

            # 如果有新增种子点，生成新区域
            if len(optimized_seeds) > len(current_seeds):
                new_seeds = optimized_seeds[len(current_seeds):]

                if self.config.verbose:
                    print(f"新增 {len(new_seeds)} 个种子点，生成新区域")

                # 处理新增种子点
                new_seeds_formatted = [(seed, None) for seed in new_seeds]
                new_regions = self.processor.process_seeds(
                    new_seeds_formatted, checker, domain, obstacle_map, resolution, origin
                )

                # 合并区域
                first_batch_regions = first_batch_regions + new_regions

                if self.config.verbose:
                    print(f"Voronoi优化后总区域数: {len(first_batch_regions)}")

                # 更新耗时
                result.iris_time += voronoi_time

        # ========================================================================
        # 第二批：针对性扩张
        # ========================================================================
        # 目标：覆盖第一批遗漏的路径点
        # 策略：
        # 1. 检查路径上未被覆盖的点
        # 2. 在未覆盖点附近提取种子点
        # 3. 使用各向异性膨胀（沿切线方向优先）

        if self.config.verbose:
            print("\n" + "="*70)
            print("Step 4: 第二批种子点提取（检查未覆盖路径点）")
            print("="*70)

        # 提取第二批种子点
        # batch=2 表示第二批，会检查已生成的区域
        # existing_regions 参数用于识别未覆盖的路径点
        second_batch_seeds = self.seed_extractor.extract_seed_points(
            path, obstacle_map, resolution, origin, batch=2, existing_regions=first_batch_regions
        )

        if self.config.verbose:
            print(f"第二批提取了 {len(second_batch_seeds)} 个种子点")

        # 执行第二批 IrisNp 算法
        if len(second_batch_seeds) > 0:
            if self.config.verbose:
                print("\n" + "="*70)
                print("Step 5: 第二批执行 IrisNp 算法（各向异性膨胀）")
                print("="*70)
                print(f"切线/法向膨胀比例: {self.config.tangent_normal_ratio}:1")
                print(f"并行处理: {'启用' if self.config.enable_parallel_processing else '禁用'}")
                if self.config.enable_parallel_processing:
                    print(f"工作进程数: {self.config.num_parallel_workers}")

            iris_start_2 = time.time()

            # 处理第二批种子点
            # 种子点包含切线方向信息，用于各向异性膨胀
            second_batch_regions = self.processor.process_seeds(
                second_batch_seeds, checker, domain, obstacle_map, resolution, origin
            )

            result.iris_time += time.time() - iris_start_2

            if self.config.verbose:
                print(f"\n第二批 IrisNp 完成，耗时: {time.time() - iris_start_2:.4f}秒")
                print(f"成功生成 {len(second_batch_regions)} 个凸区域")

            # 合并两批区域
            regions = first_batch_regions + second_batch_regions

            # ========================================================================
            # 第三轮：补充覆盖（可选）
            # ========================================================================
            # 目标：确保路径完全被覆盖
            # 策略：
            # 1. 检查是否仍有未覆盖的路径点
            # 2. 对未覆盖点进行聚类
            # 3. 为每个簇生成小区域

            if self.config.strict_coverage_check:
                # 查找未覆盖的路径点索引
                uncovered_indices = self.coverage_checker.find_uncovered_points(path, regions)

                if len(uncovered_indices) > 0:
                    if self.config.verbose:
                        print(f"\n" + "="*70)
                        print("Step 6: 第三轮针对性覆盖处理")
                        print("="*70)
                        print(f"发现 {len(uncovered_indices)} 个未覆盖点，进行第三轮处理")

                    # 为未覆盖点生成小区域
                    # 使用更小的初始区域和更密集的采样
                    third_batch_regions = self.coverage_checker.generate_regions_for_uncovered_points(
                        path, uncovered_indices, checker, domain, obstacle_map, resolution, origin
                    )

                    if len(third_batch_regions) > 0:
                        regions = regions + third_batch_regions
                        if self.config.verbose:
                            print(f"第三轮成功生成 {len(third_batch_regions)} 个凸区域")
                            print(f"总区域数: {len(regions)}")
        else:
            # 第二批没有种子点，说明第一批已经完全覆盖
            regions = first_batch_regions

        return regions

    def _process_single_batch_expansion(
        self,
        path: List[Tuple[float, float, float]],
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        checker: SimpleCollisionCheckerForIrisNp,
        domain: HPolyhedron,
        result: IrisNpResult
    ) -> List[IrisNpRegion]:
        """
        处理单批扩张模式（向后兼容）

        这是原始的扩张策略，一次性提取所有种子点并生成凸区域。

        优点：
        - 逻辑简单，易于理解
        - 适用于简单场景
        - 向后兼容旧代码

        缺点：
        - 可能遗漏某些路径段
        - 区域数量可能过多
        - 覆盖质量不如两批扩张

        Args:
            path: 路径点列表
            obstacle_map: 障碍物地图
            resolution: 地图分辨率
            origin: 地图原点
            checker: 碰撞检测器
            domain: 搜索域
            result: 结果对象（用于记录耗时）

        Returns:
            生成的凸区域列表

        Note:
            推荐使用两批扩张模式（enable_two_batch_expansion=True）
            以获得更好的覆盖质量和计算效率
        """
        import time

        # ========================================================================
        # 提取种子点
        # ========================================================================
        # 使用均匀采样策略
        # batch=1 表示单批模式

        if self.config.verbose:
            print("\n" + "="*70)
            print("Step 2: 提取种子点")
            print("="*70)

        seed_points = self.seed_extractor.extract_seed_points(
            path, obstacle_map, resolution, origin, batch=1
        )

        if self.config.verbose:
            print(f"提取了 {len(seed_points)} 个种子点")

        # ========================================================================
        # 执行 IrisNp 算法
        # ========================================================================
        # 处理所有种子点，生成凸区域
        # processor 会自动选择串行或并行处理模式

        if self.config.verbose:
            print("\n" + "="*70)
            print("Step 3: 执行 IrisNp 算法")
            print("="*70)
            print(f"膨胀模式: {'椭圆' if self.config.use_ellipse_expansion else '方形'}")
            print(f"并行处理: {'启用' if self.config.enable_parallel_processing else '禁用'}")
            if self.config.enable_parallel_processing:
                print(f"工作进程数: {self.config.num_parallel_workers}")

        iris_start = time.time()

        regions = self.processor.process_seeds(
            seed_points, checker, domain, obstacle_map, resolution, origin
        )

        result.iris_time = time.time() - iris_start

        if self.config.verbose:
            print(f"\nIrisNp 完成，耗时: {result.iris_time:.4f}秒")
            print(f"成功生成 {len(regions)} 个凸区域")

        return regions

    def _process_voronoi_only_mode(
        self,
        path: List[Tuple[float, float, float]],
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        checker: SimpleCollisionCheckerForIrisNp,
        domain: HPolyhedron,
        result: IrisNpResult
    ) -> List[IrisNpRegion]:
        """
        处理Voronoi优化模式（实验性）

        这是基于Voronoi图的纯优化策略，不使用均匀采样。

        核心思想：
        1. 粗采样：稀疏采样路径点作为初始种子点
        2. Voronoi优化：迭代添加最优种子点
        3. 覆盖检查：验证路径是否完全覆盖
        4. 补充采样：针对未覆盖点进行聚类和补充

        优点：
        - 数学最优性：基于最大空圆定理，保证种子点位置最优
        - 几何自适应性：自动适应障碍物形状和路径几何
        - 种子点最少：避免冗余采样，提高覆盖效率
        - 全局优化：考虑所有种子点的相互作用

        缺点：
        - 计算复杂度高：需要迭代生成Voronoi图，O(n log n)
        - 依赖初始种子点：需要至少3个种子点才能生成Voronoi图
        - 可能收敛慢：在复杂场景中需要多次迭代

        Args:
            path: 路径点列表
            obstacle_map: 障碍物地图
            resolution: 地图分辨率
            origin: 地图原点
            checker: 碰撞检测器
            domain: 搜索域
            result: 结果对象（用于记录耗时）

        Returns:
            生成的凸区域列表

        Note:
            这是实验性功能，适合复杂场景（不规则障碍物、多分支路径等）。
            对于简单场景，推荐使用两批扩张模式。
        """
        import time

        # ========================================================================
        # Step 1: 粗采样（初始种子点）
        # ========================================================================
        # 目标：提供初始种子点，用于生成Voronoi图
        # 策略：稀疏采样路径点，避免过度密集
        # 采样间隔：每隔20个点采样一次

        if self.config.verbose:
            print("\n" + "="*70)
            print("Step 2: 粗采样（初始种子点）")
            print("="*70)

        initial_seeds = []
        for i in range(0, len(path), 20):
            x, y, _ = path[i]
            seed_point = np.array([x, y])

            # 检查是否在自由空间
            gx = int((x - origin[0]) / resolution)
            gy = int((y - origin[1]) / resolution)

            if 0 <= gx < obstacle_map.shape[1] and 0 <= gy < obstacle_map.shape[0]:
                if obstacle_map[gy, gx] == 0:
                    # 检查与已有种子点的距离
                    if len(initial_seeds) == 0 or all(
                        np.linalg.norm(seed_point - sp) >= self.config.min_seed_distance
                        for sp in initial_seeds
                    ):
                        initial_seeds.append(seed_point)

        # 强制包含起点和终点
        x_start, y_start, _ = path[0]
        start_point = np.array([x_start, y_start])
        gx_start = int((x_start - origin[0]) / resolution)
        gy_start = int((y_start - origin[1]) / resolution)

        if 0 <= gx_start < obstacle_map.shape[1] and 0 <= gy_start < obstacle_map.shape[0]:
            if obstacle_map[gy_start, gx_start] == 0:
                if len(initial_seeds) == 0 or all(
                    np.linalg.norm(start_point - sp) >= self.config.min_seed_distance
                    for sp in initial_seeds
                ):
                    initial_seeds.insert(0, start_point)

        x_end, y_end, _ = path[-1]
        end_point = np.array([x_end, y_end])
        gx_end = int((x_end - origin[0]) / resolution)
        gy_end = int((y_end - origin[1]) / resolution)

        if 0 <= gx_end < obstacle_map.shape[1] and 0 <= gy_end < obstacle_map.shape[0]:
            if obstacle_map[gy_end, gx_end] == 0:
                if len(initial_seeds) == 0 or all(
                    np.linalg.norm(end_point - sp) >= self.config.min_seed_distance
                    for sp in initial_seeds
                ):
                    initial_seeds.append(end_point)

        if self.config.verbose:
            print(f"粗采样完成，提取了 {len(initial_seeds)} 个初始种子点")

        # ========================================================================
        # Step 2: Voronoi优化（核心）
        # ========================================================================
        # 目标：基于Voronoi图迭代优化种子点
        # 策略：
        # 1. 生成初始区域
        # 2. 提取种子点，生成Voronoi图
        # 3. 评估Voronoi顶点，添加最优顶点
        # 4. 重复直到覆盖完整或达到迭代上限

        if self.config.verbose:
            print("\n" + "="*70)
            print("Step 3: Voronoi优化（核心）")
            print("="*70)

        # 初始化Voronoi优化器
        if self.voronoi_optimizer is None:
            self.voronoi_optimizer = VoronoiSeedOptimizer(
                obstacle_map, resolution, origin, self.coverage_checker
            )

        # 迭代优化
        current_seeds = initial_seeds.copy()
        max_iterations = self.config.voronoi_max_iterations
        iteration = 0

        voronoi_start = time.time()

        while iteration < max_iterations:
            # 生成当前区域
            seeds_formatted = [(seed, None) for seed in current_seeds]
            regions = self.processor.process_seeds(
                seeds_formatted, checker, domain, obstacle_map, resolution, origin
            )

            # 检查覆盖
            if self.voronoi_optimizer._check_coverage(path, regions):
                if self.config.verbose:
                    print(f"迭代 {iteration + 1}: 路径已完全覆盖，停止优化")
                break

            # Voronoi优化（每次迭代只优化一次，内部可能添加多个种子点）
            optimized_seeds = self.voronoi_optimizer.optimize(
                current_seeds, path,
                max_iterations=1,  # 外层循环控制迭代次数
                max_new_seeds=self.config.voronoi_max_new_seeds
            )

            # 检查是否有新增种子点
            if len(optimized_seeds) <= len(current_seeds):
                if self.config.verbose:
                    print(f"迭代 {iteration + 1}: 无新增种子点，停止优化")
                break

            # 添加所有新增种子点
            new_seeds = optimized_seeds[len(current_seeds):]
            current_seeds.extend(new_seeds)

            if self.config.verbose:
                print(f"迭代 {iteration + 1}: 添加 {len(new_seeds)} 个种子点")
                print(f"  当前种子点数: {len(current_seeds)}")

            iteration += 1

        # 最终生成区域
        seeds_formatted = [(seed, None) for seed in current_seeds]
        regions = self.processor.process_seeds(
            seeds_formatted, checker, domain, obstacle_map, resolution, origin
        )

        result.iris_time = time.time() - voronoi_start

        if self.config.verbose:
            print(f"\nVoronoi优化完成")
            print(f"  总迭代次数: {iteration}")
            print(f"  初始种子点: {len(initial_seeds)}")
            print(f"  最终种子点: {len(current_seeds)}")
            print(f"  新增种子点: {len(current_seeds) - len(initial_seeds)}")
            print(f"  生成区域数: {len(regions)}")
            print(f"  耗时: {result.iris_time:.4f}秒")

        # ========================================================================
        # Step 3: 补充覆盖（可选）
        # ========================================================================
        # 目标：针对仍未覆盖的点进行聚类和补充

        if self.config.strict_coverage_check:
            uncovered_indices = self.coverage_checker.find_uncovered_points(path, regions)

            if len(uncovered_indices) > 0:
                if self.config.verbose:
                    print(f"\n" + "="*70)
                    print("Step 4: 补充覆盖")
                    print("="*70)
                    print(f"发现 {len(uncovered_indices)} 个未覆盖点")

                # 迭代补充覆盖，直到所有点都被覆盖或达到最大迭代次数
                max_supplemental_iterations = 5
                supplemental_iteration = 0
                total_supplemental_regions = 0

                while supplemental_iteration < max_supplemental_iterations:
                    # 检查当前覆盖情况
                    uncovered_indices = self.coverage_checker.find_uncovered_points(path, regions)
                    
                    if len(uncovered_indices) == 0:
                        if self.config.verbose:
                            print(f"\n补充覆盖完成：所有路径点已被覆盖")
                        break
                    
                    if self.config.verbose:
                        if supplemental_iteration == 0:
                            print(f"  未覆盖点聚类为 {len(uncovered_indices)} 个簇")
                        else:
                            print(f"\n  迭代 {supplemental_iteration + 1}: 发现 {len(uncovered_indices)} 个未覆盖点")

                    # 为未覆盖点生成小区域
                    third_batch_regions = self.coverage_checker.generate_regions_for_uncovered_points(
                        path, uncovered_indices, checker, domain, obstacle_map, resolution, origin
                    )

                    if len(third_batch_regions) == 0:
                        if self.config.verbose:
                            print(f"  迭代 {supplemental_iteration + 1}: 无法生成新区域，停止补充覆盖")
                        break
                    
                    regions = regions + third_batch_regions
                    total_supplemental_regions += len(third_batch_regions)
                    
                    if self.config.verbose:
                        print(f"  迭代 {supplemental_iteration + 1}: 生成 {len(third_batch_regions)} 个凸区域")
                        print(f"  总区域数: {len(regions)}")
                    
                    supplemental_iteration += 1
                
                if self.config.verbose and supplemental_iteration == max_supplemental_iterations:
                    print(f"\n警告: 达到最大补充覆盖迭代次数 ({max_supplemental_iterations})")
                    print(f"  总共生成 {total_supplemental_regions} 个补充区域")

        return regions

    def _create_domain(
        self,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ) -> HPolyhedron:
        """
        创建搜索域（边界框）

        搜索域限制了凸区域的最大范围，避免无限膨胀。
        使用 HPolyhedron.MakeBox 创建矩形边界框。

        Args:
            obstacle_map: 障碍物地图，2D numpy 数组
            resolution: 地图分辨率（米/像素）
            origin: 地图原点坐标 (x, y)

        Returns:
            HPolyhedron: 表示边界框的多面体

        Note:
            边界框的计算：
            - x_min = origin[0]
            - x_max = origin[0] + width * resolution
            - y_min = origin[1]
            - y_max = origin[1] + height * resolution

            其中 width 和 height 是 obstacle_map 的列数和行数
        """
        # 获取地图尺寸
        height, width = obstacle_map.shape

        # 计算边界框坐标
        x_min = origin[0]
        x_max = origin[0] + width * resolution
        y_min = origin[1]
        y_max = origin[1] + height * resolution

        # 创建边界框
        # lb: lower bound (下界)
        # ub: upper bound (上界)
        lb = np.array([x_min, y_min])
        ub = np.array([x_max, y_max])

        domain = HPolyhedron.MakeBox(lb, ub)

        return domain


def visualize_iris_np_result(
    result: IrisNpResult,
    obstacle_map: np.ndarray,
    resolution: float,
    origin: Tuple[float, float] = (0.0, 0.0),
    path: Optional[List[Tuple[float, float, float]]] = None,
    save_path: Optional[str] = None
):
    """
    可视化 IrisNp 结果

    在地图上绘制：
    1. 障碍物（灰色）
    2. 凸区域（彩色半透明）
    3. 种子点（红色星号）
    4. 路径（绿色线条）
    5. 起点和终点（绿色圆圈和红色星号）

    Args:
        result: IrisNp 结果对象
        obstacle_map: 障碍物地图
        resolution: 地图分辨率
        origin: 地图原点
        path: 路径点列表（可选）
        save_path: 保存路径（可选），如果提供则保存图片

    Example:
        >>> visualize_iris_np_result(result, obstacle_map, 0.05, save_path="result.png")
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))

    # ========================================================================
    # 绘制障碍物地图
    # ========================================================================
    # 使用 imshow 绘制 2D 地图
    # extent 参数指定坐标范围
    extent = [
        origin[0],
        origin[0] + obstacle_map.shape[1] * resolution,
        origin[1],
        origin[1] + obstacle_map.shape[0] * resolution
    ]
    ax.imshow(obstacle_map, cmap='gray', origin='lower', extent=extent, alpha=0.5)

    # ========================================================================
    # 绘制凸区域
    # ========================================================================
    # 使用不同的颜色区分不同的区域
    # 半透明显示，可以看到重叠部分
    colors = plt.cm.Set3(np.linspace(0, 1, len(result.regions)))

    for i, region in enumerate(result.regions):
        # 获取有序顶点
        vertices = region.get_vertices_ordered()

        if len(vertices) >= 3:
            # 创建多边形
            polygon = MplPolygon(
                vertices,
                closed=True,
                facecolor=colors[i],
                edgecolor='blue',
                alpha=0.4,
                linewidth=2,
                label=f'Region {i+1}'
            )
            ax.add_patch(polygon)

            # 标注区域编号
            ax.text(
                region.centroid[0], region.centroid[1],
                f'{i+1}',
                fontsize=10, ha='center', va='center',
                fontweight='bold', color='darkblue'
            )

            # 绘制种子点
            ax.plot(
                region.seed_point[0], region.seed_point[1],
                'r*', markersize=10
            )

    # ========================================================================
    # 绘制路径
    # ========================================================================
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]

        # 绘制路径线
        ax.plot(path_x, path_y, 'g-', linewidth=2, label='Path')

        # 绘制起点（绿色圆圈）
        ax.scatter(path_x[0], path_y[0], c='green', s=100, marker='o', label='Start', zorder=5)

        # 绘制终点（红色星号）
        ax.scatter(path_x[-1], path_y[-1], c='red', s=100, marker='*', label='Goal', zorder=5)

    # ========================================================================
    # 设置图形属性
    # ========================================================================
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # 设置标题
    ax.set_title(
        f'IrisNp Convex Regions ({result.num_regions} regions, '
        f'Area: {result.total_area:.2f} m²)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")

    plt.show()
    plt.close()


# ============================================================================
# 兼容性检查函数
# ============================================================================

def check_drake_availability() -> bool:
    """
    检查 Drake 是否可用

    Returns:
        True 如果 Drake 可用，False 否则

    Example:
        >>> if check_drake_availability():
        ...     print("Drake 可用")
        ... else:
        ...     print("请安装 Drake: pip install drake")
    """
    return DRAKE_AVAILABLE


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("IrisNp 凸区域生成模块")
    print("="*70)

    if not DRAKE_AVAILABLE:
        print("\n错误: Drake 未安装")
        print("请使用以下命令安装 Drake:")
        print("  pip install drake")
    else:
        print("\n✓ Drake 已安装，IrisNp 功能可用")
