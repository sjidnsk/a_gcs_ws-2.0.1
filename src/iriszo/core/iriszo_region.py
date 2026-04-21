"""
自定义IrisZo算法区域生成主模块

提供主要的区域生成接口,协调所有子模块完成端到端流程。

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional
import time
import warnings
import multiprocessing

try:
    from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    HPolyhedron = None
    Hyperellipsoid = None

from ..config.iriszo_config import IrisZoConfig
from .iriszo_algorithm import CustomIrisZoAlgorithm
from .iriszo_collision import CollisionCheckerAdapter
from .iriszo_seed_extractor import IrisZoSeedExtractor
from .iriszo_region_data import IrisZoRegion, IrisZoResult
from .iriszo_coverage_validator_enhanced import EnhancedCoverageValidator, EnhancedCoverageResult


def _process_single_seed_worker(args):
    """
    模块级工作函数，被multiprocessing调用。
    在工作进程内独立创建所有Drake对象和算法实例。

    Args:
        args: 元组 (seed_index, seed_point, tangent, config, obstacle_map, resolution, domain_lb, domain_ub, origin)

    Returns:
        (seed_index, region_or_none) 元组
    """
    (seed_index, seed_point, tangent, config,
     obstacle_map, resolution, domain_lb, domain_ub, origin) = args

    try:
        # 在工作进程内独立创建碰撞检测器
        checker = CollisionCheckerAdapter(
            obstacle_map=obstacle_map,
            resolution=resolution,
            origin=origin,
            enable_cache=config.enable_cache,
            cache_size=config.cache_size
        )

        # 在工作进程内独立创建搜索域
        domain = HPolyhedron.MakeBox(domain_lb, domain_ub)

        # 在工作进程内独立创建算法实例
        algorithm = CustomIrisZoAlgorithm(config)

        # 碰撞检查
        if not checker.check_config_collision_free(seed_point):
            return (seed_index, None)

        # 创建初始椭球体
        starting_ellipsoid = Hyperellipsoid.MakeHypersphere(
            config.starting_ellipsoid_radius, seed_point
        )

        # 执行算法
        polyhedron = algorithm.run(checker, starting_ellipsoid, domain)

        if polyhedron is not None:
            region = IrisZoRegion(
                polyhedron=polyhedron,
                seed_point=seed_point,
                iteration_count=algorithm.get_iteration_count()
            )
            return (seed_index, region)

        return (seed_index, None)

    except Exception:
        return (seed_index, None)


class IrisZoRegionGenerator:
    """
    基于自定义IrisZo的凸区域生成器

    这是模块的主类,负责协调各个子模块完成凸区域生成任务。
    与iris_pkg.IrisNpRegionGenerator接口保持一致。

    工作流程:
        1. 创建碰撞检测器
        2. 创建搜索域
        3. 提取种子点
        4. 执行自定义IrisZo算法生成凸区域
        5. 验证路径覆盖
        6. 返回结果

    Attributes:
        config: 配置参数
        algorithm: 自定义IrisZo算法实例
        seed_extractor: 种子点提取器

    Example:
        >>> from src.iriszo import IrisZoRegionGenerator, IrisZoConfig
        >>>
        >>> config = IrisZoConfig(verbose=True)
        >>> generator = IrisZoRegionGenerator(config)
        >>>
        >>> result = generator.generate_from_path(
        ...     path, obstacle_map, resolution=0.05
        ... )
        >>>
        >>> print(f"生成了 {result.num_regions} 个凸区域")
    """

    def __init__(self, config: Optional[IrisZoConfig] = None):
        """
        初始化IrisZo区域生成器

        Args:
            config: 配置参数,如果为None则使用默认配置

        Raises:
            RuntimeError: 如果Drake未安装
        """
        if not DRAKE_AVAILABLE:
            raise RuntimeError(
                "Drake未安装。请使用以下命令安装:\n"
                "pip install drake"
            )

        self.config = config or IrisZoConfig()
        self.config.validate()

        # 初始化核心算法
        self.algorithm = CustomIrisZoAlgorithm(self.config)

        # 初始化种子点提取器
        self.seed_extractor = IrisZoSeedExtractor(self.config)

        if self.config.verbose:
            print(f"IrisZoRegionGenerator初始化完成")
            print(f"配置: {self.config}")

    def generate_from_path(
        self,
        path: List[Tuple[float, float, float]],
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float] = (0.0, 0.0)
    ) -> IrisZoResult:
        """
        从路径生成凸区域(主入口方法)

        Args:
            path: 路径点列表,每个元素为(x, y, theta)
            obstacle_map: 障碍物地图,0=自由空间,1=障碍物
            resolution: 地图分辨率(米/像素)
            origin: 地图原点坐标(x, y)

        Returns:
            IrisZoResult: 生成结果,包含区域列表、统计信息等

        Example:
            >>> result = generator.generate_from_path(
            ...     path, obstacle_map, 0.05, (0.0, 0.0)
            ... )
        """
        total_start = time.time()
        result = IrisZoResult(config=self.config)

        if self.config.verbose:
            print("\n" + "=" * 60)
            print("开始生成凸区域")
            print("=" * 60)
            print(f"路径点数: {len(path)}")
            print(f"地图尺寸: {obstacle_map.shape}")
            print(f"分辨率: {resolution}")

        # Step 1: 创建碰撞检测器
        if self.config.verbose:
            print("\nStep 1: 创建碰撞检测器")

        checker = CollisionCheckerAdapter(
            obstacle_map=obstacle_map,
            resolution=resolution,
            origin=origin,
            enable_cache=self.config.enable_cache,
            cache_size=self.config.cache_size
        )

        # Step 1.5: 创建增强版覆盖验证器（如果启用）
        if self.config.strict_coverage_check:
            self._coverage_validator = EnhancedCoverageValidator(
                obstacle_map=obstacle_map,
                resolution=resolution,
                origin=origin,
                config=self.config
            )
            self._coverage_validator.initialize()
            if self.config.verbose:
                print("✓ 增强版覆盖验证器已启用")
        else:
            self._coverage_validator = None
            if self.config.verbose:
                print("⚠ 使用简单点包含判定")

        # Step 2: 创建搜索域
        if self.config.verbose:
            print("Step 2: 创建搜索域")

        domain = self._create_domain(obstacle_map, resolution, origin)

        # Step 3: 提取种子点
        if self.config.verbose:
            print("Step 3: 提取种子点")

        seeds = self.seed_extractor.extract_seed_points(
            path, obstacle_map, resolution, origin, batch=1
        )

        if self.config.verbose:
            print(f"提取到 {len(seeds)} 个种子点")

        if len(seeds) == 0:
            warnings.warn("未提取到有效的种子点")
            result.total_time = time.time() - total_start
            return result

        # Step 4: 处理种子点生成区域（第一批）
        if self.config.verbose:
            print("\nStep 4: 处理种子点生成区域（第一批）")

        iris_start = time.time()
        regions_batch1 = self._process_seeds(seeds, checker, domain)

        if self.config.verbose:
            print(f"第一批成功生成 {len(regions_batch1)} 个区域")

        # 计算第一批覆盖率（保存完整验证结果以复用）
        coverage_result_batch1 = self._calculate_coverage_ratio(path, regions_batch1)
        if isinstance(coverage_result_batch1, EnhancedCoverageResult):
            coverage_batch1 = coverage_result_batch1.coverage_ratio
        else:
            coverage_batch1 = coverage_result_batch1

        # Step 5: 第二批扩张（如果启用且覆盖率未达100%）
        regions = regions_batch1
        if coverage_batch1 < 1.0 and self.config.enable_two_batch_expansion:
            if self.config.verbose:
                print(f"\n第一批覆盖率: {coverage_batch1:.2%}，启动第二批扩张")

            # 识别未覆盖段（复用已有的覆盖验证结果）
            uncovered_segments = self._identify_uncovered_segments(
                path, regions_batch1, coverage_result=coverage_result_batch1
            )

            if len(uncovered_segments) > 0:
                if self.config.verbose:
                    print(f"识别到 {len(uncovered_segments)} 个未覆盖段")

                # 从未覆盖段提取种子点
                seeds_batch2 = self._extract_seeds_from_segments(uncovered_segments, path, obstacle_map, resolution, origin)

                if len(seeds_batch2) > 0:
                    if self.config.verbose:
                        print(f"第二批提取到 {len(seeds_batch2)} 个种子点")

                    # 生成第二批区域
                    regions_batch2 = self._process_seeds(seeds_batch2, checker, domain)
                    regions = regions_batch1 + regions_batch2

                    if self.config.verbose:
                        print(f"第二批成功生成 {len(regions_batch2)} 个区域")

        result.iris_time = time.time() - iris_start

        if self.config.verbose:
            print(f"总共生成 {len(regions)} 个区域,耗时 {result.iris_time:.3f}秒")

        # Step 5: 后处理
        postprocess_start = time.time()

        # 计算统计信息
        result.regions = regions
        result.num_regions = len(regions)
        result.total_area = sum(r.area for r in regions)

        # 计算覆盖率（避免重复计算：无第二批扩张时复用已有结果）
        if regions is regions_batch1:
            # 无第二批扩张，直接复用第一批结果
            result.coverage_ratio = coverage_batch1
        else:
            # 有第二批扩张，需要重新计算全部区域的覆盖率
            result.coverage_ratio = self._calculate_coverage_ratio(
                path, regions
            )
            if isinstance(result.coverage_ratio, EnhancedCoverageResult):
                result.coverage_ratio = result.coverage_ratio.coverage_ratio

        result.postprocess_time = time.time() - postprocess_start
        result.total_time = time.time() - total_start

        # 获取缓存统计
        cache_stats = checker.get_cache_stats()
        result.cache_hit_rate = cache_stats['hit_rate']

        if self.config.verbose:
            print("\n" + "=" * 60)
            print("生成完成")
            print("=" * 60)
            print(f"区域数量: {result.num_regions}")
            print(f"总面积: {result.total_area:.6f}")
            print(f"覆盖率: {result.coverage_ratio:.2%}")
            print(f"算法耗时: {result.iris_time:.3f}秒")
            print(f"后处理耗时: {result.postprocess_time:.3f}秒")
            print(f"总耗时: {result.total_time:.3f}秒")
            print(f"缓存命中率: {result.cache_hit_rate:.2%}")

        return result

    def _create_domain(
        self,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ) -> HPolyhedron:
        """
        创建搜索域(边界框)

        Args:
            obstacle_map: 障碍物地图
            resolution: 地图分辨率
            origin: 地图原点

        Returns:
            搜索域多面体
        """
        height, width = obstacle_map.shape

        x_min = origin[0]
        x_max = origin[0] + width * resolution
        y_min = origin[1]
        y_max = origin[1] + height * resolution

        lb = np.array([x_min, y_min])
        ub = np.array([x_max, y_max])

        return HPolyhedron.MakeBox(lb, ub)

    def _process_seeds(
        self,
        seeds: List[Tuple[np.ndarray, Optional[np.ndarray]]],
        checker: CollisionCheckerAdapter,
        domain: HPolyhedron
    ) -> List[IrisZoRegion]:
        """
        处理种子点生成区域（路由方法）

        根据配置选择并行或串行处理。并行失败时自动降级为串行。

        Args:
            seeds: 种子点列表
            checker: 碰撞检测器
            domain: 搜索域

        Returns:
            生成的区域列表
        """
        use_parallel = (
            self.config.num_workers > 1
            and self.config.enable_batch_processing
            and len(seeds) > 1
        )

        if use_parallel:
            try:
                return self._process_seeds_parallel(seeds, checker, domain)
            except Exception as e:
                if self.config.verbose:
                    print(f"  ⚠ 并行处理失败，降级为串行: {e}")
                return self._process_seeds_serial(seeds, checker, domain)
        else:
            return self._process_seeds_serial(seeds, checker, domain)

    def _process_seeds_serial(
        self,
        seeds: List[Tuple[np.ndarray, Optional[np.ndarray]]],
        checker: CollisionCheckerAdapter,
        domain: HPolyhedron
    ) -> List[IrisZoRegion]:
        """
        串行处理种子点生成区域

        Args:
            seeds: 种子点列表
            checker: 碰撞检测器
            domain: 搜索域

        Returns:
            生成的区域列表
        """
        regions = []

        for i, (seed_point, tangent) in enumerate(seeds):
            if self.config.verbose:
                print(f"\n处理种子点 {i+1}/{len(seeds)}: {seed_point}")

            # 检查种子点是否无碰撞
            if not checker.check_config_collision_free(seed_point):
                if self.config.verbose:
                    print(f"  ✗ 种子点有碰撞,跳过")
                continue

            # 创建初始椭球体
            radius = self.config.starting_ellipsoid_radius
            starting_ellipsoid = Hyperellipsoid.MakeHypersphere(
                radius, seed_point
            )

            # 执行自定义IrisZo算法
            try:
                polyhedron = self.algorithm.run(
                    checker, starting_ellipsoid, domain
                )

                if polyhedron is not None:
                    # 创建IrisZoRegion
                    region = IrisZoRegion(
                        polyhedron=polyhedron,
                        seed_point=seed_point,
                        iteration_count=self.algorithm.get_iteration_count()
                    )
                    regions.append(region)

                    if self.config.verbose:
                        print(f"  ✓ 成功生成区域,面积: {region.area:.6f}")
                else:
                    if self.config.verbose:
                        print(f"  ✗ 算法返回None")

            except Exception as e:
                if self.config.verbose:
                    print(f"  ✗ 算法执行失败: {e}")
                continue

        return regions

    def _process_seeds_parallel(
        self,
        seeds: List[Tuple[np.ndarray, Optional[np.ndarray]]],
        checker: CollisionCheckerAdapter,
        domain: HPolyhedron
    ) -> List[IrisZoRegion]:
        """
        并行处理种子点生成区域

        Args:
            seeds: 种子点列表
            checker: 碰撞检测器
            domain: 搜索域

        Returns:
            生成的区域列表（按种子点原始顺序排列）
        """
        # 从碰撞检测器提取可序列化参数
        obstacle_map = checker.obstacle_map
        resolution = checker.resolution
        origin = checker.origin

        # 从obstacle_map计算搜索域边界
        height, width = obstacle_map.shape
        domain_lb = np.array([origin[0], origin[1]])
        domain_ub = np.array([origin[0] + width * resolution, origin[1] + height * resolution])

        # 构造worker参数列表
        task_args = []
        for i, (seed_point, tangent) in enumerate(seeds):
            task_args.append((
                i, seed_point, tangent, self.config,
                obstacle_map, resolution, domain_lb, domain_ub, origin
            ))

        num_workers = min(self.config.num_workers, len(seeds))

        if self.config.verbose:
            print(f"\n并行处理 {len(seeds)} 个种子点, 工作进程数: {num_workers}")

        # 使用进程池并行执行
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(_process_single_seed_worker, task_args)

        # 按seed_index排序，保持确定性顺序
        results.sort(key=lambda x: x[0])

        # 收集有效区域
        regions = []
        for seed_index, region in results:
            if region is not None:
                regions.append(region)
                if self.config.verbose:
                    print(f"  ✓ 种子点 {seed_index+1}: 成功生成区域,面积: {region.area:.6f}")
            else:
                if self.config.verbose:
                    print(f"  ✗ 种子点 {seed_index+1}: 未生成区域")

        return regions

    def _calculate_coverage_ratio(
        self,
        path: List[Tuple[float, float, float]],
        regions: List[IrisZoRegion]
    ):
        """
        计算路径覆盖率（使用增强版覆盖验证）

        Args:
            path: 路径点列表
            regions: 区域列表

        Returns:
            覆盖率(float)或EnhancedCoverageResult（增强版验证器启用时返回完整结果以供复用）
        """
        if len(path) == 0 or len(regions) == 0:
            return 0.0

        # 使用增强版覆盖验证器（如果配置启用）
        if hasattr(self, '_coverage_validator') and self._coverage_validator is not None:
            # 转换路径格式
            path_array = np.array([[x, y, theta] for x, y, theta in path])

            # 使用增强版验证器，返回完整结果以供复用
            result = self._coverage_validator.validate(path_array, regions)
            return result

        # 降级为简单点包含判定
        covered_count = 0

        for x, y, _ in path:
            point = np.array([x, y])

            # 检查是否被任一区域覆盖
            if any(region.contains(point) for region in regions):
                covered_count += 1

        return covered_count / len(path)

    def _identify_uncovered_segments(
        self,
        path: List[Tuple[float, float, float]],
        regions: List[IrisZoRegion],
        coverage_result=None
    ) -> List[Tuple[int, int]]:
        """
        识别未覆盖的连续段（使用增强版覆盖验证）

        Args:
            path: 路径点列表
            regions: 区域列表
            coverage_result: 已有的覆盖验证结果（EnhancedCoverageResult），传入时直接复用避免重复计算

        Returns:
            未覆盖连续段列表，每个元素为(start_idx, end_idx)
        """
        # 复用已有的覆盖验证结果
        if coverage_result is not None and isinstance(coverage_result, EnhancedCoverageResult):
            uncovered_indices = coverage_result.uncovered_indices
        else:
            uncovered_indices = []

            # 使用增强版覆盖验证器（如果启用）
            if hasattr(self, '_coverage_validator') and self._coverage_validator is not None:
                # 转换路径格式
                path_array = np.array([[x, y, theta] for x, y, theta in path])

                # 计算一次base_radius（结果恒定，无需每次重新计算）
                base_radius = self._coverage_validator.radius_calculator.calculate_base_radius()

                # 使用增强版验证器检查每个点
                for i, point in enumerate(path_array):
                    point_2d = point[:2]

                    # 检查是否在障碍物内
                    if self._coverage_validator.obstacle_detector.is_in_obstacle(point_2d):
                        uncovered_indices.append(i)
                        continue

                    # 查询障碍物距离
                    obs_distance = self._coverage_validator.obstacle_detector.query_distance(point_2d)

                    # 调整有效半径
                    effective_radius = self._coverage_validator.radius_calculator.adjust_effective_radius(
                        base_radius, obs_distance
                    )

                    # 判定覆盖
                    is_covered, _ = self._coverage_validator.coverage_checker.check_point_coverage(
                        point_2d, effective_radius, regions
                    )

                    if not is_covered:
                        uncovered_indices.append(i)
            else:
                # 降级为简单点包含判定
                for i, (x, y, _) in enumerate(path):
                    point = np.array([x, y])
                    if not any(region.contains(point) for region in regions):
                        uncovered_indices.append(i)

        # 合并为连续段
        if not uncovered_indices:
            return []

        segments = []
        start = uncovered_indices[0]
        prev = start

        for idx in uncovered_indices[1:]:
            if idx != prev + 1:
                segments.append((start, prev))
                start = idx
            prev = idx

        segments.append((start, prev))
        return segments

    def _extract_seeds_from_segments(
        self,
        segments: List[Tuple[int, int]],
        path: List[Tuple[float, float, float]],
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        从未覆盖段提取种子点

        Args:
            segments: 未覆盖段列表
            path: 路径点列表
            obstacle_map: 障碍物地图
            resolution: 地图分辨率
            origin: 地图原点

        Returns:
            种子点列表
        """
        seeds = []
        existing_points = []  # 仅存seed_point的numpy数组，用于向量化距离检查
        min_distance = 0.3  # 第二批种子点最小距离

        for start_idx, end_idx in segments:
            # 取段的中点作为种子点
            mid_idx = (start_idx + end_idx) // 2
            x, y, _ = path[mid_idx]
            seed_point = np.array([x, y])

            # 检查是否在自由空间
            gx = int((x - origin[0]) / resolution)
            gy = int((y - origin[1]) / resolution)

            if 0 <= gx < obstacle_map.shape[1] and 0 <= gy < obstacle_map.shape[0]:
                if obstacle_map[gy, gx] == 0:
                    # 向量化距离检查
                    if len(existing_points) == 0:
                        seeds.append((seed_point, None))
                        existing_points.append(seed_point)
                    else:
                        existing = np.array(existing_points)
                        distances = np.linalg.norm(existing - seed_point, axis=1)
                        if np.all(distances >= min_distance):
                            seeds.append((seed_point, None))
                            existing_points.append(seed_point)

        return seeds

    def __str__(self) -> str:
        """
        返回生成器的字符串表示

        Returns:
            格式化的字符串
        """
        return (
            f"IrisZoRegionGenerator(\n"
            f"  config={self.config}\n"
            f")"
        )
