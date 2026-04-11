"""
自定义IrisZo算法核心模块

实现自定义IrisZo算法的核心逻辑,基于技术文档中的Algorithm 2从零实现。

作者: Path Planning Team
"""

import numpy as np
from typing import Optional, Tuple
import time
import warnings

try:
    from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    HPolyhedron = None
    Hyperellipsoid = None

from ..config.iriszo_config import IrisZoConfig
from .iriszo_collision import CollisionCheckerAdapter
from .iriszo_sampler import HitAndRunSampler
from .iriszo_bisection import BisectionSearcher
from .iriszo_hyperplane import SeparatingHyperplaneGenerator


class CustomIrisZoAlgorithm:
    """
    自定义IrisZo算法核心实现

    基于技术文档中的Algorithm 2 (ZeroOrderSeparatingPlanes)实现零阶优化凸区域生成。
    不依赖Drake的IrisZo API,从零实现算法逻辑。

    算法工作流程:
        1. 初始化多面体P为搜索域
        2. 外迭代循环:
           a. 在P内均匀采样
           b. 碰撞检测找出碰撞点
           c. 检查终止条件
           d. 二分搜索优化碰撞点位置
           e. 生成并添加分离超平面
           f. 更新多面体P
           g. 计算新的内接椭球体
        3. 返回最终多面体

    Attributes:
        config: 配置参数
        sampler: Hit-and-Run采样器
        bisectioner: 二分搜索器
        hyperplane_gen: 分离超平面生成器
        iteration_count: 外迭代计数
        verbose: 是否详细输出

    Example:
        >>> algorithm = CustomIrisZoAlgorithm(config)
        >>> region = algorithm.run(checker, starting_ellipsoid, domain)
    """

    def __init__(self, config: IrisZoConfig):
        """
        初始化自定义IrisZo算法

        Args:
            config: 配置参数
        """
        self.config = config
        self.sampler = HitAndRunSampler(config)
        self.bisectioner = BisectionSearcher(config)
        self.hyperplane_gen = SeparatingHyperplaneGenerator(config)
        self.iteration_count = 0
        self.verbose = config.verbose

        # 终止条件相关计数器
        self.zero_collision_count = 0
        self.prev_collision_ratio = 1.0

    def run(
        self,
        checker: CollisionCheckerAdapter,
        starting_ellipsoid: 'Hyperellipsoid',
        domain: HPolyhedron
    ) -> Optional[HPolyhedron]:
        """
        执行自定义IrisZo算法

        Args:
            checker: 碰撞检测器
            starting_ellipsoid: 初始椭球体
            domain: 搜索域

        Returns:
            生成的HPolyhedron区域,如果失败则返回None

        Raises:
            RuntimeError: 如果Drake不可用
            ValueError: 如果初始椭球体中心有碰撞

        Example:
            >>> region = algorithm.run(checker, ellipsoid, domain)
        """
        if not DRAKE_AVAILABLE:
            raise RuntimeError("Drake不可用,无法执行IrisZo算法")

        start_time = time.time()

        # 检查初始椭球体中心是否无碰撞
        try:
            center = starting_ellipsoid.center()
        except Exception:
            raise ValueError("无法获取初始椭球体中心")

        if not checker.check_config_collision_free(center):
            raise ValueError(f"初始椭球体中心{center}有碰撞,无法生成区域")

        if self.verbose:
            print(f"开始自定义IrisZo算法,初始中心: {center}")

        # 初始化多面体P为搜索域
        P = domain
        E = starting_ellipsoid

        # 外迭代循环
        self.iteration_count = 0
        prev_volume = 0.0

        for iteration in range(self.config.iteration_limit):
            self.iteration_count = iteration + 1

            if self.verbose:
                print(f"\n外迭代 {iteration + 1}/{self.config.iteration_limit}")

            # 执行一次外迭代
            P_new, collision_ratio, num_collision = self._outer_iteration(
                P, E, checker
            )

            if P_new is None:
                if self.verbose:
                    print("外迭代失败,保持当前多面体")
                break

            P = P_new

            # 计算新的内接椭球体
            try:
                E = P.MaximumVolumeInscribedEllipsoid()
            except Exception as e:
                warnings.warn(f"计算内接椭球体失败: {e}")
                break

            # 检查终止条件
            if self._check_termination(collision_ratio, iteration):
                if self.verbose:
                    print(f"满足终止条件,碰撞比例: {collision_ratio:.4f}")
                break

            # 检查体积增长
            try:
                current_volume = P.Volume()
                if iteration > 0:
                    volume_growth = (current_volume - prev_volume) / prev_volume
                    if self.verbose:
                        print(f"体积增长: {volume_growth:.4f}")

                    if volume_growth < self.config.termination_threshold:
                        if self.verbose:
                            print("体积增长不足,终止迭代")
                        break

                prev_volume = current_volume
            except Exception:
                pass

        elapsed_time = time.time() - start_time

        if self.verbose:
            print(f"\n算法完成,迭代次数: {self.iteration_count}, 耗时: {elapsed_time:.3f}秒")

        return P

    def _outer_iteration(
        self,
        P: HPolyhedron,
        E: 'Hyperellipsoid',
        checker: CollisionCheckerAdapter
    ) -> Tuple[Optional[HPolyhedron], float, int]:
        """
        执行一次外迭代

        Args:
            P: 当前多面体
            E: 当前内接椭球体
            checker: 碰撞检测器

        Returns:
            (更新后的多面体, 碰撞点比例, 碰撞点数量)
        """
        # Step 1: 采样并执行碰撞检测
        samples, collision_samples = self._sample_and_check(P, checker)

        num_samples = len(samples)
        num_collision = len(collision_samples)

        if num_samples == 0:
            return None, 0.0, 0

        collision_ratio = num_collision / num_samples

        if self.verbose:
            print(f"采样: {num_samples}个点, 碰撞: {num_collision}个, 比例: {collision_ratio:.4f}")

        # 如果没有碰撞点,返回当前多面体
        if num_collision == 0:
            return P, 0.0, 0

        # Step 2: 二分搜索优化碰撞点位置
        try:
            ellipsoid_center = E.center()
        except Exception:
            ellipsoid_center = np.mean(collision_samples, axis=0)

        boundary_points = self.bisectioner.search_boundary(
            collision_samples, ellipsoid_center, checker
        )

        if self.verbose:
            print(f"二分搜索完成,边界点: {len(boundary_points)}个")

        # Step 3: 生成并添加分离超平面
        hyperplanes = self.hyperplane_gen.generate(boundary_points, E, P)

        if len(hyperplanes) == 0:
            return P, collision_ratio, num_collision

        # Step 4: 更新多面体
        P_new = self.hyperplane_gen.update_polyhedron(P, hyperplanes)

        if P_new is None:
            return P, collision_ratio, num_collision

        return P_new, collision_ratio, num_collision

    def _sample_and_check(
        self,
        P: HPolyhedron,
        checker: CollisionCheckerAdapter
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样并执行碰撞检测

        Args:
            P: 当前多面体
            checker: 碰撞检测器

        Returns:
            (所有采样点, 碰撞采样点)
        """
        # 在多面体内采样
        try:
            samples = self.sampler.sample(
                P, self.config.num_samples_per_iteration
            )
        except Exception as e:
            warnings.warn(f"采样失败: {e}")
            return np.array([]), np.array([])

        # 批量碰撞检测
        collision_results = checker.check_configs_collision_free(samples)

        # 分离碰撞点和无碰撞点
        collision_mask = np.array([not result for result in collision_results])
        collision_samples = samples[collision_mask]

        return samples, collision_samples

    def _check_termination(self, collision_ratio: float, iteration: int) -> bool:
        """
        检查终止条件（改进版）

        改进点：
        1. 强制最小迭代次数，防止过早终止
        2. 要求连续多次碰撞比例为0才终止
        3. 保留原有的相对终止阈值检查

        Args:
            collision_ratio: 碰撞点比例
            iteration: 当前迭代次数

        Returns:
            True如果应终止
        """
        # 条件1: 强制最小迭代次数
        if iteration < self.config.min_iterations - 1:
            if self.verbose and iteration == 0:
                print(f"强制最小迭代次数: {self.config.min_iterations}")
            return False

        # 条件2: 连续多次碰撞比例为0才终止
        if collision_ratio == 0 and self.prev_collision_ratio == 0:
            self.zero_collision_count += 1
            if self.zero_collision_count >= self.config.zero_collision_threshold:
                if self.verbose:
                    print(f"连续{self.zero_collision_count}次碰撞比例为0,终止迭代")
                return True
        else:
            self.zero_collision_count = 0

        # 条件3: 碰撞比例低于相对终止阈值
        if collision_ratio < self.config.relative_termination_threshold:
            return True

        # 条件4: 达到最大迭代次数
        if iteration >= self.config.iteration_limit - 1:
            return True

        # 更新前一次碰撞比例
        self.prev_collision_ratio = collision_ratio

        return False

    def get_iteration_count(self) -> int:
        """
        获取迭代次数

        Returns:
            外迭代次数
        """
        return self.iteration_count

    def __str__(self) -> str:
        """
        返回算法的字符串表示

        Returns:
            格式化的字符串
        """
        return (
            f"CustomIrisZoAlgorithm(\n"
            f"  iteration_limit={self.config.iteration_limit}\n"
            f"  bisection_steps={self.config.bisection_steps}\n"
            f"  num_samples={self.config.num_samples_per_iteration}\n"
            f")"
        )
