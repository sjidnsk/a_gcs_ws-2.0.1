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

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial import KDTree

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
from .iris_np_parallel import init_worker, process_single_seed


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
    """基于 IrisNp 的凸区域生成器"""

    def __init__(self, config: Optional[IrisNpConfig] = None):
        """
        初始化 IrisNp 凸区域生成器

        Args:
            config: IrisNp 配置参数
        """
        if not DRAKE_AVAILABLE:
            raise RuntimeError(
                "Drake 未安装。请使用以下命令安装：\n"
                "pip install drake"
            )

        self.config = config or IrisNpConfig()
        self.expansion = IrisNpExpansion(self.config)

    def generate_from_path(
        self,
        path: List[Tuple[float, float, float]],
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float] = (0.0, 0.0)
    ) -> IrisNpResult:
        """
        从路径生成凸区域（使用IrisNp）

        支持两批种子点扩张：
        - 第一批：正常扩张
        - 第二批：检查路径点不在任何凸区域内，优先沿路径切线方向膨胀

        Args:
            path: 路径点列表 [(x, y, theta), ...]
            obstacle_map: 障碍物地图
            resolution: 地图分辨率
            origin: 地图原点

        Returns:
            IrisNpResult: 生成的凸区域结果
        """
        import time
        total_start = time.time()

        result = IrisNpResult(config=self.config)

        # Step 1: 创建碰撞检测器
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

        # Step 2: 定义域（边界框）
        domain = self._create_domain(obstacle_map, resolution, origin)

        # 判断是否启用两批扩张
        if self.config.enable_two_batch_expansion:
            # ========== 两批扩张模式 ==========
            if self.config.verbose:
                print("\n" + "="*70)
                print("Step 2: 第一批种子点提取（正常扩张）")
                print("="*70)

            # 第一批：提取种子点
            first_batch_seeds = self._extract_seed_points(
                path, obstacle_map, resolution, origin, batch=1
            )

            if self.config.verbose:
                print(f"第一批提取了 {len(first_batch_seeds)} 个种子点")

            # 第一批：执行 IrisNp 算法
            if self.config.verbose:
                print("\n" + "="*70)
                print("Step 3: 第一批执行 IrisNp 算法（正常扩张）")
                print("="*70)
                print(f"膨胀模式: {'椭圆' if self.config.use_ellipse_expansion else '方形'}")
                print(f"并行处理: {'启用' if self.config.enable_parallel_processing else '禁用'}")
                if self.config.enable_parallel_processing:
                    print(f"工作进程数: {self.config.num_parallel_workers}")

            iris_start = time.time()

            # 选择处理模式：并行或串行
            if self.config.enable_parallel_processing and len(first_batch_seeds) > 1:
                first_batch_regions = self._process_seeds_parallel(
                    first_batch_seeds, checker, domain, obstacle_map, resolution, origin
                )
            else:
                first_batch_regions = self._process_seeds_serial(
                    first_batch_seeds, checker, domain, obstacle_map, resolution, origin
                )

            result.iris_time = time.time() - iris_start

            if self.config.verbose:
                print(f"\n第一批 IrisNp 完成，耗时: {result.iris_time:.4f}秒")
                print(f"成功生成 {len(first_batch_regions)} 个凸区域")

            # 第二批：提取种子点（检查路径点不在任何凸区域内）
            if self.config.verbose:
                print("\n" + "="*70)
                print("Step 4: 第二批种子点提取（检查未覆盖路径点）")
                print("="*70)

            second_batch_seeds = self._extract_seed_points(
                path, obstacle_map, resolution, origin, batch=2, existing_regions=first_batch_regions
            )

            if self.config.verbose:
                print(f"第二批提取了 {len(second_batch_seeds)} 个种子点")

            # 第二批：执行 IrisNp 算法（各向异性膨胀）
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

                # 选择处理模式：并行或串行
                if self.config.enable_parallel_processing and len(second_batch_seeds) > 1:
                    second_batch_regions = self._process_seeds_parallel(
                        second_batch_seeds, checker, domain, obstacle_map, resolution, origin
                    )
                else:
                    second_batch_regions = self._process_seeds_serial(
                        second_batch_seeds, checker, domain, obstacle_map, resolution, origin
                    )

                result.iris_time += time.time() - iris_start_2

                if self.config.verbose:
                    print(f"\n第二批 IrisNp 完成，耗时: {time.time() - iris_start_2:.4f}秒")
                    print(f"成功生成 {len(second_batch_regions)} 个凸区域")

                # 合并两批区域
                regions = first_batch_regions + second_batch_regions

                # 方案2: 第三轮覆盖检查
                # 检查是否仍有未覆盖的路径点
                if self.config.strict_coverage_check:
                    uncovered_indices = self._find_uncovered_points(path, regions)
                    if len(uncovered_indices) > 0:
                        if self.config.verbose:
                            print(f"\n" + "="*70)
                            print("Step 6: 第三轮针对性覆盖处理")
                            print("="*70)
                            print(f"发现 {len(uncovered_indices)} 个未覆盖点，进行第三轮处理")

                        # 第三轮: 为未覆盖点生成小区域
                        third_batch_regions = self._generate_regions_for_uncovered_points(
                            path, uncovered_indices, checker, domain, obstacle_map, resolution, origin
                        )

                        if len(third_batch_regions) > 0:
                            regions = regions + third_batch_regions
                            if self.config.verbose:
                                print(f"第三轮成功生成 {len(third_batch_regions)} 个凸区域")
                                print(f"总区域数: {len(regions)}")
            else:
                regions = first_batch_regions

        else:
            # ========== 单批扩张模式（向后兼容） ==========
            if self.config.verbose:
                print("\n" + "="*70)
                print("Step 2: 提取种子点")
                print("="*70)

            seed_points = self._extract_seed_points(
                path, obstacle_map, resolution, origin, batch=1
            )

            if self.config.verbose:
                print(f"提取了 {len(seed_points)} 个种子点")

            # 执行 IrisNp 算法
            if self.config.verbose:
                print("\n" + "="*70)
                print("Step 3: 执行 IrisNp 算法")
                print("="*70)
                print(f"膨胀模式: {'椭圆' if self.config.use_ellipse_expansion else '方形'}")
                print(f"并行处理: {'启用' if self.config.enable_parallel_processing else '禁用'}")
                if self.config.enable_parallel_processing:
                    print(f"工作进程数: {self.config.num_parallel_workers}")

            iris_start = time.time()

            # 选择处理模式：并行或串行
            if self.config.enable_parallel_processing and len(seed_points) > 1:
                regions = self._process_seeds_parallel(
                    seed_points, checker, domain, obstacle_map, resolution, origin
                )
            else:
                regions = self._process_seeds_serial(
                    seed_points, checker, domain, obstacle_map, resolution, origin
                )

            result.iris_time = time.time() - iris_start

            if self.config.verbose:
                print(f"\nIrisNp 完成，耗时: {result.iris_time:.4f}秒")
                print(f"成功生成 {len(regions)} 个凸区域")

        # Step 6: 后处理
        postprocess_start = time.time()

        # 验证路径覆盖
        if self.config.strict_coverage_check:
            coverage_result = self._verify_path_coverage(path, regions)
            if self.config.verbose:
                print(f"\n路径覆盖验证: {'通过' if coverage_result else '未通过'}")

        result.postprocess_time = time.time() - postprocess_start

        # 收集性能统计
        if self.config.enable_collision_cache:
            cache_stats = checker.get_cache_stats()
            result.cache_hit_rate = cache_stats['hit_rate']

        # 统计信息
        result.regions = regions
        result.num_regions = len(regions)
        result.total_area = sum(r.area for r in regions)

        # 计算覆盖率
        total_free_space = np.sum(obstacle_map == 0) * resolution * resolution
        result.coverage_ratio = result.total_area / total_free_space if total_free_space > 0 else 0.0

        result.total_time = time.time() - total_start

        # 输出性能报告
        if self.config.enable_profiling:
            self._print_performance_report(result, checker)

        return result

    def _print_performance_report(
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
        print(f"  - 膨胀模式: {'椭圆' if self.config.use_ellipse_expansion else '方形'}")
        print(f"  - 初始区域大小: {self.config.initial_region_size}")
        print(f"  - 最大区域大小: {self.config.max_region_size}")
        print(f"  - 膨胀步长: {self.config.size_increment}")

    def _process_seeds_serial(
        self,
        seed_points: List[Tuple[np.ndarray, Optional[np.ndarray]]],
        checker: SimpleCollisionCheckerForIrisNp,
        domain: HPolyhedron,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ) -> List[IrisNpRegion]:
        """串行处理种子点（支持切线方向）"""
        regions = []

        for i, (seed_point, tangent_direction) in enumerate(seed_points):
            if self.config.verbose:
                print(f"处理种子点 {i+1}/{len(seed_points)}: ({seed_point[0]:.2f}, {seed_point[1]:.2f})")
                if tangent_direction is not None:
                    print(f"  切线方向: ({tangent_direction[0]:.2f}, {tangent_direction[1]:.2f})")

            try:
                region = self.expansion.simplified_iris_with_sampling(
                    checker, seed_point, domain, obstacle_map, resolution, origin, tangent_direction
                )

                if region is not None:
                    regions.append(region)
                    if self.config.verbose:
                        print(f"  ✓ 生成区域，面积: {region.area:.2f} 平方米")
                else:
                    if self.config.verbose:
                        print(f"  ✗ 未能生成有效区域")
            except Exception as e:
                if self.config.verbose:
                    print(f"  ✗ IrisNp 失败: {e}")

        return regions

    def _process_seeds_parallel(
        self,
        seed_points: List[Tuple[np.ndarray, Optional[np.ndarray]]],
        checker: SimpleCollisionCheckerForIrisNp,
        domain: HPolyhedron,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ) -> List[IrisNpRegion]:
        """并行处理种子点（支持切线方向，使用initializer共享资源）"""
        # 准备参数
        num_workers = min(self.config.num_parallel_workers, len(seed_points))

        if self.config.verbose:
            print(f"使用 {num_workers} 个工作进程并行处理 {len(seed_points)} 个种子点")

        # 创建任务列表（简化任务，减少数据传递）
        tasks = []
        for i, (seed_point, tangent_direction) in enumerate(seed_points):
            tasks.append({
                'seed_id': i,
                'seed_point': seed_point,
                'tangent_direction': tangent_direction,
                'obstacle_map': obstacle_map,
                'resolution': resolution,
                'origin': origin,
                'config': self.config,
                'domain': domain
            })

        regions = []

        # 使用进程池并行处理（带initializer）
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=init_worker,
            initargs=(obstacle_map, resolution, origin, self.config, domain)
        ) as executor:
            # 提交所有任务
            future_to_task = {
                executor.submit(process_single_seed, task): task for task in tasks
            }

            # 收集结果
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                seed_id = task['seed_id']

                try:
                    region = future.result()
                    if region is not None:
                        regions.append(region)
                        if self.config.verbose:
                            print(f"  ✓ 种子点 {seed_id+1} 完成，面积: {region.area:.2f} 平方米")
                    else:
                        if self.config.verbose:
                            print(f"  ✗ 种子点 {seed_id+1} 未能生成有效区域")
                except Exception as e:
                    if self.config.verbose:
                        print(f"  ✗ 种子点 {seed_id+1} 失败: {e}")

        return regions

    def _create_domain(
        self,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ) -> HPolyhedron:
        """创建域（边界框）"""
        height, width = obstacle_map.shape

        x_min = origin[0]
        x_max = origin[0] + width * resolution
        y_min = origin[1]
        y_max = origin[1] + height * resolution

        # 创建边界框
        lb = np.array([x_min, y_min])
        ub = np.array([x_max, y_max])

        domain = HPolyhedron.MakeBox(lb, ub)

        return domain

    def _extract_seed_points(
        self,
        path: List[Tuple[float, float, float]],
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        batch: int = 1,
        existing_regions: Optional[List[IrisNpRegion]] = None
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """
        从路径中提取种子点

        Args:
            path: 路径点列表
            obstacle_map: 障碍物地图
            resolution: 地图分辨率
            origin: 地图原点
            batch: 批次 (1=第一批, 2=第二批)
            existing_regions: 已存在的凸区域列表（用于第二批）

        Returns:
            种子点列表，每个元素为 (seed_point, tangent_direction)
            tangent_direction 为路径切线方向，用于第二批各向异性膨胀
        """
        seed_points = []

        path_length = len(path)
        if path_length == 0:
            return seed_points

        if batch == 1:
            # 第一批：根据路径长度自适应采样
            # 计算路径总长度
            path_total_length = 0.0
            for i in range(1, path_length):
                x0, y0, _ = path[i-1]
                x1, y1, _ = path[i]
                path_total_length += np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

            # 改进3: 减小密度系数，提高区域重叠度
            # 从0.3改为0.2，增加种子点密度，提高区域重叠
            density_factor = 0.2  # 提高种子点密度

            # 根据路径长度和最大种子点数计算采样间隔
            # 目标：均匀分布种子点，确保路径覆盖
            if path_total_length > 0:
                # 计算理想的种子点间距
                ideal_spacing = path_total_length / (self.config.max_seed_points * density_factor)
                # 根据分辨率计算采样间隔（点数）
                sample_interval = max(1, int(ideal_spacing / resolution))
            else:
                # 路径长度为0，使用默认间隔
                sample_interval = max(1, self.config.first_batch_seed_interval)

            # 改进1: 强制包含起点
            x_start, y_start, _ = path[0]
            start_point = np.array([x_start, y_start])
            gx_start = int((x_start - origin[0]) / resolution)
            gy_start = int((y_start - origin[1]) / resolution)

            if 0 <= gx_start < obstacle_map.shape[1] and 0 <= gy_start < obstacle_map.shape[0]:
                if obstacle_map[gy_start, gx_start] == 0:  # 自由空间
                    seed_points.append((start_point, None))
                    if self.config.verbose:
                        print(f"  强制添加起点: ({x_start:.2f}, {y_start:.2f})")

            # 均匀采样路径点
            for i in range(sample_interval, path_length - 1, sample_interval):
                x, y, _ = path[i]
                seed_point = np.array([x, y])

                # 检查种子点是否有效（不在障碍物内）
                gx = int((x - origin[0]) / resolution)
                gy = int((y - origin[1]) / resolution)

                if 0 <= gx < obstacle_map.shape[1] and 0 <= gy < obstacle_map.shape[0]:
                    if obstacle_map[gy, gx] == 0:  # 自由空间
                        # 改进2: 自适应最小距离（根据局部路径密度）
                        adaptive_min_distance = self._compute_adaptive_min_distance(
                            path, i, path_total_length
                        )
                        # 检查与已有种子点的距离
                        if self._is_valid_seed_adaptive(seed_point, [sp[0] for sp in seed_points], adaptive_min_distance):
                            # 第一批不需要切线方向
                            seed_points.append((seed_point, None))

            # 改进1: 强制包含终点
            x_end, y_end, _ = path[-1]
            end_point = np.array([x_end, y_end])
            gx_end = int((x_end - origin[0]) / resolution)
            gy_end = int((y_end - origin[1]) / resolution)

            if 0 <= gx_end < obstacle_map.shape[1] and 0 <= gy_end < obstacle_map.shape[0]:
                if obstacle_map[gy_end, gx_end] == 0:  # 自由空间
                    # 检查终点是否与已有种子点距离过近
                    if self._is_valid_seed(end_point, [sp[0] for sp in seed_points]):
                        seed_points.append((end_point, None))
                        if self.config.verbose:
                            print(f"  强制添加终点: ({x_end:.2f}, {y_end:.2f})")

        elif batch == 2:
            # 第二批：优化策略 - 确保凸区域之间有重叠，避免仅点接触
            # 选取：完全未被覆盖的路径点 + 周围邻域内未被完全覆盖的路径点
            if existing_regions is None or len(existing_regions) == 0:
                return seed_points

            # 构建区域中心的KDTree用于快速查找
            region_centers = np.array([r.centroid for r in existing_regions])
            kdtree = KDTree(region_centers)

            # 预计算最大搜索半径（基于最大区域大小）
            max_region_radius = max(
                np.max(np.linalg.norm(r.vertices - r.centroid, axis=1))
                for r in existing_regions
            )
            search_radius = max_region_radius * 1.5  # 增加50%的安全裕度

            # 遍历所有路径点
            for i in range(path_length):
                x, y, _ = path[i]
                point = np.array([x, y])

                # 使用KDTree快速查找附近的区域
                nearby_indices = kdtree.query_ball_point(point, search_radius)

                # 检查点本身是否被覆盖
                is_covered = False
                for idx in nearby_indices:
                    region = existing_regions[idx]
                    if region.contains(point, tol=1e-6):
                        is_covered = True
                        break

                # 判断是否应该作为种子点
                should_add_seed = False

                # 情况1：点完全未被覆盖
                if not is_covered:
                    should_add_seed = True
                else:
                    # 情况2：点被覆盖，但检查周围邻域的覆盖情况
                    # 目的：识别区域边界附近的点，确保区域重叠
                    coverage_info = self._check_neighborhood_coverage(
                        point, existing_regions, kdtree, search_radius, resolution
                    )

                    # 如果邻域内有未覆盖的点，说明该点在区域边界附近
                    # 添加为种子点可以确保新生成的区域与已有区域重叠
                    if coverage_info['uncovered_count'] >= 3:
                        should_add_seed = True
                        if self.config.verbose:
                            print(f"  边界点候选: ({x:.2f}, {y:.2f}), "
                                  f"未覆盖邻居数: {coverage_info['uncovered_count']}/9")

                # 添加种子点
                if should_add_seed:
                    # 检查是否在障碍物内
                    gx = int((x - origin[0]) / resolution)
                    gy = int((y - origin[1]) / resolution)

                    if 0 <= gx < obstacle_map.shape[1] and 0 <= gy < obstacle_map.shape[0]:
                        if obstacle_map[gy, gx] == 0:  # 自由空间
                            # 计算路径切线方向
                            tangent = self._compute_path_tangent(path, i)

                            # 使用放宽的距离限制
                            if self._is_valid_seed_relaxed(point, [sp[0] for sp in seed_points], min_distance=0.8):
                                seed_points.append((point, tangent))

        return seed_points

    def _is_valid_seed(
        self,
        candidate: np.ndarray,
        existing: List[np.ndarray]
    ) -> bool:
        """
        检查种子点是否有效（向量化优化版本）

        Args:
            candidate: 候选种子点
            existing: 已存在的种子点列表

        Returns:
            True 如果候选点有效，False 否则
        """
        if len(existing) == 0:
            return True

        # 向量化计算：将列表转换为数组
        existing_array = np.array(existing)

        # 计算所有距离（向量化操作）
        distances = np.linalg.norm(existing_array - candidate, axis=1)

        # 检查是否所有距离都大于最小距离
        return np.all(distances >= self.config.min_seed_distance)

    def _is_valid_seed_relaxed(
        self,
        candidate: np.ndarray,
        existing: List[np.ndarray],
        min_distance: float = 0.3
    ) -> bool:
        """
        方案1: 放宽距离限制的种子点验证

        对于未覆盖的路径点,使用更宽松的距离限制

        Args:
            candidate: 候选种子点
            existing: 已存在的种子点列表
            min_distance: 最小距离（默认0.3米，比标准的1.0米更宽松）

        Returns:
            True 如果候选点有效，False 否则
        """
        if len(existing) == 0:
            return True

        # 向量化计算：将列表转换为数组
        existing_array = np.array(existing)

        # 计算所有距离（向量化操作）
        distances = np.linalg.norm(existing_array - candidate, axis=1)

        # 检查是否所有距离都大于最小距离
        return np.all(distances >= min_distance)

    def _check_neighborhood_coverage(
        self,
        point: np.ndarray,
        existing_regions: List[IrisNpRegion],
        kdtree: KDTree,
        search_radius: float,
        resolution: float
    ) -> Dict[str, Any]:
        """
        检查点周围邻域的覆盖情况

        目的：识别区域边界附近的点，确保新生成的区域与已有区域有重叠

        Args:
            point: 查询点
            existing_regions: 已存在的凸区域列表
            kdtree: 区域中心的KDTree
            search_radius: 搜索半径
            resolution: 地图分辨率

        Returns:
            coverage_info: 包含覆盖信息的字典
                - covered_count: 被覆盖的邻居数量
                - uncovered_count: 未被覆盖的邻居数量
                - coverage_ratio: 覆盖比例
        """
        # 定义邻域的9个方向（包括中心点）
        # 使用网格距离，确保覆盖相邻区域
        directions = [
            (0, 0),    # 中心
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 4邻域
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 对角
        ]

        covered_count = 0
        uncovered_count = 0

        for dx, dy in directions:
            # 计算邻居点坐标
            neighbor = point + np.array([dx, dy]) * resolution

            # 使用KDTree查找附近的区域
            nearby_indices = kdtree.query_ball_point(neighbor, search_radius)

            # 检查邻居点是否被任何区域覆盖
            is_covered = False
            for idx in nearby_indices:
                region = existing_regions[idx]
                if region.contains(neighbor, tol=1e-6):
                    is_covered = True
                    break

            if is_covered:
                covered_count += 1
            else:
                uncovered_count += 1

        total_count = len(directions)
        coverage_ratio = covered_count / total_count

        return {
            'covered_count': covered_count,
            'uncovered_count': uncovered_count,
            'coverage_ratio': coverage_ratio
        }

    def _compute_adaptive_min_distance(
        self,
        path: List[Tuple[float, float, float]],
        index: int,
        path_total_length: float
    ) -> float:
        """
        改进2: 计算自适应最小距离

        根据局部路径密度动态调整最小距离：
        - 在狭窄通道或密集区域：减小距离，增加种子点密度
        - 在开阔区域：保持正常距离

        Args:
            path: 路径点列表
            index: 当前点索引
            path_total_length: 路径总长度

        Returns:
            自适应的最小距离
        """
        # 基础最小距离
        base_distance = self.config.min_seed_distance

        # 计算局部路径密度（使用前后窗口）
        window_size = min(10, len(path) // 4)  # 窗口大小

        start_idx = max(0, index - window_size)
        end_idx = min(len(path) - 1, index + window_size)

        # 计算局部路径长度
        local_length = 0.0
        for i in range(start_idx + 1, end_idx + 1):
            x0, y0, _ = path[i-1]
            x1, y1, _ = path[i]
            local_length += np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

        # 计算局部密度（点数/长度）
        local_density = (end_idx - start_idx + 1) / max(local_length, 0.1)

        # 计算全局平均密度
        global_density = len(path) / max(path_total_length, 0.1)

        # 根据密度比调整最小距离
        density_ratio = local_density / global_density

        if density_ratio > 1.5:
            # 密集区域：减小距离到70%
            adaptive_distance = base_distance * 0.7
        elif density_ratio < 0.7:
            # 稀疏区域：保持正常距离
            adaptive_distance = base_distance
        else:
            # 正常区域：略微减小距离到85%
            adaptive_distance = base_distance * 0.85

        return adaptive_distance

    def _is_valid_seed_adaptive(
        self,
        candidate: np.ndarray,
        existing: List[np.ndarray],
        adaptive_min_distance: float
    ) -> bool:
        """
        改进2: 使用自适应最小距离验证种子点

        Args:
            candidate: 候选种子点
            existing: 已存在的种子点列表
            adaptive_min_distance: 自适应的最小距离

        Returns:
            True 如果候选点有效，False 否则
        """
        if len(existing) == 0:
            return True

        # 向量化计算：将列表转换为数组
        existing_array = np.array(existing)

        # 计算所有距离（向量化操作）
        distances = np.linalg.norm(existing_array - candidate, axis=1)

        # 检查是否所有距离都大于自适应最小距离
        return np.all(distances >= adaptive_min_distance)

    def _compute_path_tangent(
        self,
        path: List[Tuple[float, float, float]],
        index: int
    ) -> np.ndarray:
        """
        计算路径在指定点的切线方向

        Args:
            path: 路径点列表
            index: 当前点索引

        Returns:
            单位切线向量 [tx, ty]
        """
        path_length = len(path)

        # 使用前后点计算切线
        if path_length == 1:
            # 只有一个点，返回默认方向
            return np.array([1.0, 0.0])

        if index == 0:
            # 第一个点：使用前向差分
            p0 = np.array([path[0][0], path[0][1]])
            p1 = np.array([path[1][0], path[1][1]])
            tangent = p1 - p0
        elif index == path_length - 1:
            # 最后一个点：使用后向差分
            p_prev = np.array([path[index-1][0], path[index-1][1]])
            p_curr = np.array([path[index][0], path[index][1]])
            tangent = p_curr - p_prev
        else:
            # 中间点：使用中心差分
            p_prev = np.array([path[index-1][0], path[index-1][1]])
            p_next = np.array([path[index+1][0], path[index+1][1]])
            tangent = p_next - p_prev

        # 归一化
        norm = np.linalg.norm(tangent)
        if norm > 1e-6:
            tangent = tangent / norm
        else:
            tangent = np.array([1.0, 0.0])

        return tangent

    def _verify_path_coverage(
        self,
        path: List[Tuple[float, float, float]],
        regions: List[IrisNpRegion]
    ) -> bool:
        """
        验证路径是否完全被凸区域覆盖

        Args:
            path: 路径点列表
            regions: 凸区域列表

        Returns:
            True 如果路径完全被覆盖，False 否则
        """
        if len(path) == 0:
            return True

        if len(regions) == 0:
            return False

        # 检查每个路径点
        uncovered_points = []
        for i, (x, y, _) in enumerate(path):
            point = np.array([x, y])
            is_covered = False

            # 检查点是否在任何区域内
            for region in regions:
                if region.contains(point, tol=1e-6):
                    is_covered = True
                    break

            if not is_covered:
                uncovered_points.append(i)

        if len(uncovered_points) > 0:
            if self.config.verbose:
                print(f"  警告: 发现 {len(uncovered_points)} 个路径点未被覆盖")
                print(f"  未覆盖点索引: {uncovered_points[:10]}{'...' if len(uncovered_points) > 10 else ''}")
            return False

        return True

    def _find_uncovered_points(
        self,
        path: List[Tuple[float, float, float]],
        regions: List[IrisNpRegion]
    ) -> List[int]:
        """
        方案2: 查找未覆盖的路径点索引

        Args:
            path: 路径点列表
            regions: 凸区域列表

        Returns:
            未覆盖点的索引列表
        """
        if len(path) == 0 or len(regions) == 0:
            return list(range(len(path)))

        uncovered_indices = []
        for i, (x, y, _) in enumerate(path):
            point = np.array([x, y])
            is_covered = False

            # 检查点是否在任何区域内
            for region in regions:
                if region.contains(point, tol=1e-6):
                    is_covered = True
                    break

            if not is_covered:
                uncovered_indices.append(i)

        return uncovered_indices

    def _generate_regions_for_uncovered_points(
        self,
        path: List[Tuple[float, float, float]],
        uncovered_indices: List[int],
        checker: SimpleCollisionCheckerForIrisNp,
        domain: HPolyhedron,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ) -> List[IrisNpRegion]:
        """
        方案2: 为未覆盖点生成小区域

        使用更小的初始区域和更密集的采样,确保覆盖

        Args:
            path: 路径点列表
            uncovered_indices: 未覆盖点索引列表
            checker: 碰撞检测器
            domain: 定义域
            obstacle_map: 障碍物地图
            resolution: 分辨率
            origin: 原点

        Returns:
            生成的凸区域列表
        """
        regions = []

        # 对未覆盖点进行聚类,避免生成过多重叠区域
        # 使用简单的距离聚类
        clusters = self._cluster_uncovered_points(path, uncovered_indices, cluster_distance=2.0)

        if self.config.verbose:
            print(f"  未覆盖点聚类为 {len(clusters)} 个簇")

        # 为每个簇生成一个区域
        for cluster_idx, cluster_indices in enumerate(clusters):
            # 使用簇的中心点作为种子点
            cluster_points = [path[i] for i in cluster_indices]
            center_x = sum(p[0] for p in cluster_points) / len(cluster_points)
            center_y = sum(p[1] for p in cluster_points) / len(cluster_points)
            seed_point = np.array([center_x, center_y])

            # 检查种子点是否在障碍物内
            gx = int((center_x - origin[0]) / resolution)
            gy = int((center_y - origin[1]) / resolution)

            if not (0 <= gx < obstacle_map.shape[1] and 0 <= gy < obstacle_map.shape[0]):
                continue

            if obstacle_map[gy, gx] > 0:
                # 如果中心点在障碍物内,尝试使用簇中的其他点
                found_valid_seed = False
                for idx in cluster_indices:
                    x, y, _ = path[idx]
                    gx = int((x - origin[0]) / resolution)
                    gy = int((y - origin[1]) / resolution)
                    if 0 <= gx < obstacle_map.shape[1] and 0 <= gy < obstacle_map.shape[0]:
                        if obstacle_map[gy, gx] == 0:
                            seed_point = np.array([x, y])
                            found_valid_seed = True
                            break
                if not found_valid_seed:
                    continue

            # 使用更小的初始区域生成凸区域
            try:
                # 临时修改配置,使用更小的初始区域
                original_initial_size = self.config.initial_region_size
                original_max_size = self.config.max_region_size

                # 使用更小的初始区域(0.05米)和更小的最大区域(20米)
                self.config.initial_region_size = 0.05
                self.config.max_region_size = 20.0

                region = self.expansion.simplified_iris_with_sampling(
                    checker, seed_point, domain, obstacle_map, resolution, origin
                )

                # 恢复原始配置
                self.config.initial_region_size = original_initial_size
                self.config.max_region_size = original_max_size

                if region is not None and region.area > 0:
                    regions.append(region)
                    if self.config.verbose:
                        print(f"    ✓ 簇 {cluster_idx + 1} 生成区域成功，面积: {region.area:.2f} 平方米")

            except Exception as e:
                if self.config.verbose:
                    print(f"    ✗ 簇 {cluster_idx + 1} 生成区域失败: {e}")
                continue

        return regions

    def _cluster_uncovered_points(
        self,
        path: List[Tuple[float, float, float]],
        uncovered_indices: List[int],
        cluster_distance: float = 2.0
    ) -> List[List[int]]:
        """
        对未覆盖点进行聚类

        Args:
            path: 路径点列表
            uncovered_indices: 未覆盖点索引列表
            cluster_distance: 聚类距离阈值

        Returns:
            聚类结果,每个元素是一个簇的索引列表
        """
        if len(uncovered_indices) == 0:
            return []

        # 简单的贪心聚类算法
        clusters = []
        remaining = set(uncovered_indices)

        while remaining:
            # 选择一个种子点
            seed_idx = remaining.pop()
            cluster = [seed_idx]

            # 扩展簇
            changed = True
            while changed:
                changed = False
                for idx in list(remaining):
                    # 检查是否与簇中任何点距离小于阈值
                    for cluster_idx in cluster:
                        x1, y1, _ = path[idx]
                        x2, y2, _ = path[cluster_idx]
                        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        if dist <= cluster_distance:
                            cluster.append(idx)
                            remaining.remove(idx)
                            changed = True
                            break

            clusters.append(cluster)

        return clusters


def visualize_iris_np_result(
    result: IrisNpResult,
    obstacle_map: np.ndarray,
    resolution: float,
    origin: Tuple[float, float] = (0.0, 0.0),
    path: Optional[List[Tuple[float, float, float]]] = None,
    save_path: Optional[str] = None
):
    """可视化 IrisNp 结果"""
    fig, ax = plt.subplots(figsize=(12, 10))

    # 绘制障碍物地图
    extent = [
        origin[0],
        origin[0] + obstacle_map.shape[1] * resolution,
        origin[1],
        origin[1] + obstacle_map.shape[0] * resolution
    ]
    ax.imshow(obstacle_map, cmap='gray', origin='lower', extent=extent, alpha=0.5)

    # 绘制凸区域
    colors = plt.cm.Set3(np.linspace(0, 1, len(result.regions)))

    for i, region in enumerate(result.regions):
        vertices = region.get_vertices_ordered()
        if len(vertices) >= 3:
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

    # 绘制路径
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'g-', linewidth=2, label='Path')
        ax.scatter(path_x[0], path_y[0], c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(path_x[-1], path_y[-1], c='red', s=100, marker='*', label='Goal', zorder=5)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title(
        f'IrisNp Convex Regions ({result.num_regions} regions, '
        f'Area: {result.total_area:.2f} m²)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化结果已保存至: {save_path}")

    plt.show()
    plt.close()


# 兼容性检查函数
def check_drake_availability() -> bool:
    """检查 Drake 是否可用"""
    return DRAKE_AVAILABLE


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
