"""
IrisNp 种子点处理模块

负责处理种子点并生成凸区域，支持串行和并行两种处理模式。

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from pydrake.geometry.optimization import HPolyhedron

from config.iris import IrisNpConfig
from .iris_np_region_data import IrisNpRegion
from .iris_np_collision import SimpleCollisionCheckerForIrisNp
from .iris_np_expansion import IrisNpExpansion
from .iris_np_parallel import init_worker, process_single_seed


class IrisNpProcessor:
    """IrisNp 种子点处理器"""

    def __init__(
        self,
        config: IrisNpConfig,
        expansion: IrisNpExpansion
    ):
        """
        初始化种子点处理器

        Args:
            config: IrisNp 配置参数
            expansion: IrisNp 膨胀器
        """
        self.config = config
        self.expansion = expansion

    def process_seeds(
        self,
        seed_points: List[Tuple[np.ndarray, Optional[np.ndarray]]],
        checker: SimpleCollisionCheckerForIrisNp,
        domain: HPolyhedron,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ) -> List[IrisNpRegion]:
        """
        处理种子点，生成凸区域

        Args:
            seed_points: 种子点列表，每个元素为 (seed_point, tangent_direction)
            checker: 碰撞检测器
            domain: 定义域
            obstacle_map: 障碍物地图
            resolution: 地图分辨率
            origin: 地图原点

        Returns:
            生成的凸区域列表
        """
        # 选择处理模式：并行或串行
        if self.config.enable_parallel_processing and len(seed_points) > 1:
            return self._process_seeds_parallel(
                seed_points, checker, domain, obstacle_map, resolution, origin
            )
        else:
            return self._process_seeds_serial(
                seed_points, checker, domain, obstacle_map, resolution, origin
            )

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
