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

        # Drake调用批量化：A/b缓存
        self._A_cached: Optional[np.ndarray] = None
        self._b_cached: Optional[np.ndarray] = None

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

        # Drake调用批量化：初始化A/b缓存
        use_drake_batch = self.config.enable_drake_batch
        if use_drake_batch:
            self._A_cached = domain.A()
            self._b_cached = domain.b()

        # 外迭代循环
        self.iteration_count = 0
        self.zero_collision_count = 0
        prev_volume = 0.0
        prev_collision_ratio = 1.0  # 初始假设高碰撞比例

        for iteration in range(self.config.iteration_limit):
            self.iteration_count = iteration + 1

            if self.verbose:
                print(f"\n外迭代 {iteration + 1}/{self.config.iteration_limit}")

            # 计算自适应采样参数
            adaptive_num, adaptive_mix = self._compute_adaptive_params(
                prev_collision_ratio, iteration
            )

            # 执行一次外迭代
            if use_drake_batch:
                P_new, collision_ratio, num_collision = self._outer_iteration_batch(
                    P, E, checker, adaptive_num, adaptive_mix
                )
            else:
                P_new, collision_ratio, num_collision = self._outer_iteration(
                    P, E, checker, adaptive_num, adaptive_mix
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
                # 更新碰撞比例（无论是否终止都要更新，供下次自适应计算使用）
                prev_collision_ratio = collision_ratio
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
                        # 更新碰撞比例（无论是否终止都要更新，供下次自适应计算使用）
                        prev_collision_ratio = collision_ratio
                        break

                prev_volume = current_volume
            except Exception:
                pass

            # 更新碰撞比例供下次自适应计算使用
            prev_collision_ratio = collision_ratio

        elapsed_time = time.time() - start_time

        if self.verbose:
            print(f"\n算法完成,迭代次数: {self.iteration_count}, 耗时: {elapsed_time:.3f}秒")

        return P

    def _outer_iteration(
        self,
        P: HPolyhedron,
        E: 'Hyperellipsoid',
        checker: CollisionCheckerAdapter,
        adaptive_num_samples: int,
        adaptive_mix_steps: int
    ) -> Tuple[Optional[HPolyhedron], float, int]:
        """
        执行一次外迭代

        Args:
            P: 当前多面体
            E: 当前内接椭球体
            checker: 碰撞检测器
            adaptive_num_samples: 自适应采样点数
            adaptive_mix_steps: 自适应混合步数

        Returns:
            (更新后的多面体, 碰撞点比例, 碰撞点数量)
        """
        # Step 1: 采样并执行碰撞检测
        samples, collision_samples = self._sample_and_check(
            P, checker, adaptive_num_samples, adaptive_mix_steps
        )

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

    def _outer_iteration_batch(
        self,
        P: HPolyhedron,
        E: 'Hyperellipsoid',
        checker: CollisionCheckerAdapter,
        adaptive_num_samples: int,
        adaptive_mix_steps: int
    ) -> Tuple[Optional[HPolyhedron], float, int]:
        """
        执行一次外迭代（Drake调用批量化路径）

        使用A/b缓存避免重复Drake跨语言调用：
        - sample()直接使用缓存的A/b，不调用polyhedron.A()/b()
        - update_polyhedron_cache()纯numpy增量更新，不调用Drake
        - 仅在需要MVIE/Volume时构造HPolyhedron

        Args:
            P: 当前多面体（仅用于generate()中的_remove_redundant）
            E: 当前内接椭球体
            checker: 碰撞检测器
            adaptive_num_samples: 自适应采样点数
            adaptive_mix_steps: 自适应混合步数

        Returns:
            (更新后的多面体, 碰撞点比例, 碰撞点数量)
        """
        # 获取椭球中心（用于采样起始点和二分搜索）
        try:
            ellipsoid_center = E.center()
        except Exception:
            ellipsoid_center = None

        # Step 1: 采样并执行碰撞检测（使用A/b缓存）
        samples, collision_samples = self._sample_and_check_batch(
            P, checker, adaptive_num_samples, adaptive_mix_steps,
            ellipsoid_center=ellipsoid_center
        )

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
        if ellipsoid_center is None:
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

        # Step 4: 增量更新A/b缓存（纯numpy，无Drake调用）
        self._A_cached, self._b_cached = self.hyperplane_gen.update_polyhedron_cache(
            self._A_cached, self._b_cached, hyperplanes
        )

        # 延迟构造HPolyhedron（仅在MVIE/Volume需要时）
        try:
            P_new = HPolyhedron(self._A_cached, self._b_cached)
        except Exception as e:
            warnings.warn(f"从A/b缓存构造HPolyhedron失败: {e}，回退到原始路径")
            P_new = self.hyperplane_gen.update_polyhedron(P, hyperplanes)

        if P_new is None:
            return P, collision_ratio, num_collision

        return P_new, collision_ratio, num_collision

    def _sample_and_check(
        self,
        P: HPolyhedron,
        checker: CollisionCheckerAdapter,
        num_samples: int,
        mix_steps: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样并执行碰撞检测

        Args:
            P: 当前多面体
            checker: 碰撞检测器
            num_samples: 采样点数
            mix_steps: 混合步数

        Returns:
            (所有采样点, 碰撞采样点)
        """
        # 在多面体内采样
        try:
            samples = self.sampler.sample(
                P, num_samples, mix_steps=mix_steps
            )
        except Exception as e:
            warnings.warn(f"采样失败: {e}")
            return np.array([]), np.array([])

        # 批量碰撞检测
        collision_results = checker.check_configs_collision_free(samples)

        # 分离碰撞点和无碰撞点
        collision_mask = ~np.array(collision_results, dtype=np.bool_)
        collision_samples = samples[collision_mask]

        return samples, collision_samples

    def _sample_and_check_batch(
        self,
        P: HPolyhedron,
        checker: CollisionCheckerAdapter,
        num_samples: int,
        mix_steps: int,
        ellipsoid_center: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样并执行碰撞检测（Drake调用批量化路径）

        使用A/b缓存避免从Drake对象提取约束矩阵。

        Args:
            P: 当前多面体（仅作为fallback使用）
            checker: 碰撞检测器
            num_samples: 采样点数
            mix_steps: 混合步数
            ellipsoid_center: 椭球中心（用作采样起始点）

        Returns:
            (所有采样点, 碰撞采样点)
        """
        # 在多面体内采样（使用A/b缓存）
        try:
            samples = self.sampler.sample(
                P, num_samples, mix_steps=mix_steps,
                A=self._A_cached, b=self._b_cached,
                ellipsoid_center=ellipsoid_center
            )
        except Exception as e:
            warnings.warn(f"采样失败: {e}")
            return np.array([]), np.array([])

        # 批量碰撞检测
        collision_results = checker.check_configs_collision_free(samples)

        # 分离碰撞点和无碰撞点
        collision_mask = ~np.array(collision_results, dtype=np.bool_)
        collision_samples = samples[collision_mask]

        return samples, collision_samples

    def _compute_adaptive_params(
        self,
        prev_collision_ratio: float,
        iteration: int
    ) -> Tuple[int, int]:
        """
        根据碰撞比例计算自适应采样参数

        Args:
            prev_collision_ratio: 上一次迭代的碰撞比例
            iteration: 当前迭代序号

        Returns:
            (采样点数, 混合步数)
        """
        # 首次迭代：使用混合精度采样（降低mix_steps加速）
        if iteration == 0:
            if self.config.enable_adaptive_sampling:
                return self.config.num_samples_per_iteration, self.config.adaptive_first_iter_mix_steps
            else:
                return self.config.num_samples_per_iteration, 20

        # 未启用自适应采样：使用默认参数
        if not self.config.enable_adaptive_sampling:
            return self.config.num_samples_per_iteration, 20

        ratio = prev_collision_ratio

        # 异常值保护
        if not np.isfinite(ratio) or ratio < 0:
            return self.config.num_samples_per_iteration, 20

        if ratio > self.config.adaptive_high_threshold:
            # 早期：碰撞比例高，使用完整参数
            return self.config.num_samples_per_iteration, 20
        elif ratio > self.config.adaptive_low_threshold:
            # 中期：适度减少
            return self.config.adaptive_mid_samples, self.config.adaptive_mid_mix_steps
        else:
            # 后期：大幅减少
            return max(self.config.adaptive_low_samples, self.config.adaptive_min_samples), self.config.adaptive_low_mix_steps

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
