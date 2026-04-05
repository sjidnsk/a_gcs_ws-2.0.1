"""
IrisNp 区域膨胀算法模块

包含各种区域膨胀算法的实现。

作者: Path Planning Team
"""

import numpy as np
import warnings
from typing import List, Tuple, Optional
from pydrake.geometry.optimization import HPolyhedron

from ..config.iris_np_config import IrisNpConfig
from .iris_np_region_data import IrisNpRegion
from .iris_np_collision import SimpleCollisionCheckerForIrisNp
from .iris_np_utils import (
    compute_polyhedron_vertices_optimized,
    compute_polygon_area,
    quick_boundary_collision_check,
    estimate_area_fast
)


class IrisNpExpansion:
    """区域膨胀算法基类"""

    def __init__(self, config: IrisNpConfig):
        self.config = config

    def simplified_iris_with_sampling(
        self,
        checker: SimpleCollisionCheckerForIrisNp,
        seed_point: np.ndarray,
        domain: HPolyhedron,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        tangent_direction: Optional[np.ndarray] = None
    ) -> Optional[IrisNpRegion]:
        """
        改进的IRIS方法（支持椭圆和方形膨胀）

        根据配置选择膨胀策略，生成更自然的凸区域
        支持各向异性膨胀（切线优先，法向限制）
        """
        # 选择膨胀策略
        if self.config.use_adaptive_expansion:
            # 使用自适应膨胀（真正的IRIS）
            return self.adaptive_iris_expansion(
                checker, seed_point, domain, obstacle_map, resolution, origin, tangent_direction
            )
        elif self.config.use_ellipse_expansion:
            # 使用椭圆膨胀（统一膨胀）
            return self.ellipse_expansion(
                checker, seed_point, domain, obstacle_map, resolution, origin
            )
        else:
            # 使用方形膨胀（统一膨胀）
            return self.box_expansion(
                checker, seed_point, domain, obstacle_map, resolution, origin
            )

    def adaptive_iris_expansion(
        self,
        checker: SimpleCollisionCheckerForIrisNp,
        seed_point: np.ndarray,
        domain: HPolyhedron,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        tangent_direction: Optional[np.ndarray] = None
    ) -> Optional[IrisNpRegion]:
        """
        自适应 IRIS 膨胀算法（优化版）

        使用二分搜索加速膨胀过程
        支持各向异性膨胀（切线优先，法向限制）
        """
        # 初始化
        initial_size = self.config.initial_region_size
        num_directions = self.config.num_expansion_directions

        # 定义膨胀方向
        angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
        directions = np.array([[np.cos(a), np.sin(a)] for a in angles])

        # 每个方向的膨胀距离
        expansion_distances = np.ones(num_directions) * initial_size

        # 检查初始区域
        initial_region = self._create_region_from_directions(
            seed_point, directions, expansion_distances, domain, seed_point
        )

        if initial_region is None:
            return None

        from .iris_np_utils import check_region_collision_optimized
        if check_region_collision_optimized(self.config, checker, initial_region, resolution):
            # 尝试缩小初始区域
            for scale in [0.3, 0.2, 0.1, 0.05]:
                test_distances = np.ones(num_directions) * initial_size * scale
                test_region = self._create_region_from_directions(
                    seed_point, directions, test_distances, domain, seed_point
                )

                if test_region is not None:
                    if not check_region_collision_optimized(self.config, checker, test_region, resolution):
                        expansion_distances = test_distances
                        break
            else:
                return None

        # 使用自适应步长膨胀（传递切线方向）
        return self._adaptive_expansion_adaptive_step(
            checker, seed_point, directions, expansion_distances, domain, resolution, tangent_direction
        )

    def _adaptive_expansion_adaptive_step(
        self,
        checker: SimpleCollisionCheckerForIrisNp,
        seed_point: np.ndarray,
        directions: np.ndarray,
        expansion_distances: np.ndarray,
        domain: HPolyhedron,
        resolution: float,
        tangent_direction: Optional[np.ndarray] = None
    ) -> Optional[IrisNpRegion]:
        """
        优化的自适应步长膨胀（改进版）

        改进点：
        1. 方向级别的独立收敛机制
        2. 每个方向维护独立的碰撞计数器
        3. 碰撞后更激进的步长衰减
        4. 基于收敛方向比例的全局退出策略

        特点：
        1. 遇到障碍物的方向快速收敛，不再浪费计算
        2. 未遇到障碍物的方向继续膨胀到最大尺寸
        3. 所有方向均衡膨胀
        4. 支持各向异性膨胀（切线优先，法向限制）
        """
        num_directions = len(directions)
        max_size = self.config.max_region_size

        # 每个方向维护独立的步长
        step_sizes = np.ones(num_directions) * self.config.adaptive_initial_step
        min_step = self.config.adaptive_min_step
        step_reduction = self.config.adaptive_step_reduction

        # 改进1: 添加方向级别的收敛状态跟踪
        collision_counts = np.zeros(num_directions, dtype=int)  # 每个方向的连续碰撞次数
        max_collision_count = 3  # 连续碰撞3次后认为该方向收敛

        # 如果提供了切线方向，应用各向异性膨胀（向量化）
        if tangent_direction is not None:
            cos_angles = np.abs(np.dot(directions, tangent_direction))
            step_sizes *= (1.0 + (self.config.tangent_normal_ratio - 1.0) * cos_angles)

        # 动态调整最大迭代次数
        max_iterations = min(50, int(max_size / min_step) + 10)

        # 改进2: 基于收敛方向比例的退出策略
        converged_stable_count = 0
        max_converged_stable = 5  # 收敛方向数量稳定5次后退出

        for iteration in range(max_iterations):
            improved_count = 0
            converged_count = 0

            for i in range(num_directions):
                # 改进3: 如果该方向已收敛，跳过
                if collision_counts[i] >= max_collision_count:
                    converged_count += 1
                    continue

                # 如果步长已经很小，也认为收敛
                if step_sizes[i] < min_step:
                    collision_counts[i] = max_collision_count
                    converged_count += 1
                    continue

                # 尝试膨胀
                new_distance = expansion_distances[i] + step_sizes[i]

                # 限制最大尺寸
                if new_distance > max_size:
                    new_distance = max_size
                    # 达到最大尺寸也算收敛
                    if expansion_distances[i] >= max_size * 0.95:
                        collision_counts[i] = max_collision_count
                        converged_count += 1
                        continue

                # 使用快速碰撞检查
                if quick_boundary_collision_check(
                    self.config, checker, seed_point, directions[i], new_distance, resolution
                ):
                    # 改进4: 碰撞后更激进的步长衰减
                    step_sizes[i] *= step_reduction
                    collision_counts[i] += 1

                    # 改进5: 连续碰撞多次后，大幅减小步长以快速收敛
                    if collision_counts[i] >= 2:
                        step_sizes[i] *= step_reduction  # 再次衰减
                else:
                    # 无碰撞，接受膨胀
                    expansion_distances[i] = new_distance
                    improved_count += 1
                    collision_counts[i] = 0  # 重置碰撞计数

                    # 如果接近最大尺寸，减小步长
                    if expansion_distances[i] > max_size * 0.8:
                        step_sizes[i] *= step_reduction

            # 改进6: 基于收敛方向比例的退出策略
            if converged_count >= num_directions * 1.0:  # 100%的方向收敛
                converged_stable_count += 1
                if converged_stable_count >= max_converged_stable:
                    if self.config.verbose:
                        print(f"    退出: {converged_count}/{num_directions} 方向已收敛")
                    break
            else:
                converged_stable_count = 0

            # 改进7: 如果所有方向都无改进且大部分已收敛
            if improved_count == 0 and converged_count >= num_directions * 0.6:
                if self.config.verbose:
                    print(f"    退出: 无改进且 {converged_count}/{num_directions} 方向已收敛")
                break

        # # 应用方向协同处理
        # expansion_distances = self._apply_direction_coordination(
        #     directions, expansion_distances, seed_point, checker,
        #     enable_smoothing=False,
        #     enable_corner_detection=False,
        #     enable_local_verification=False
        # )

        # 创建最终区域
        final_region = self._create_region_from_directions(
            seed_point, directions, expansion_distances, domain, seed_point
        )

        return final_region

    def ellipse_expansion(
        self,
        checker: SimpleCollisionCheckerForIrisNp,
        seed_point: np.ndarray,
        domain: HPolyhedron,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ) -> Optional[IrisNpRegion]:
        """
        椭圆膨胀方法

        生成椭圆形凸区域，更适应复杂环境
        """
        # 使用配置参数
        initial_size = self.config.initial_region_size
        max_size = self.config.max_region_size
        size_increment = self.config.size_increment
        aspect_ratio = self.config.ellipse_aspect_ratio

        current_size = initial_size
        best_region = None

        # 椭圆膨胀：在不同方向上使用不同的膨胀速率
        for iteration in range(self.config.iteration_limit):
            # 创建椭圆区域（传递原始种子点）
            region = self._create_ellipse_region(
                seed_point, current_size, aspect_ratio, domain, seed_point
            )

            if region is None:
                break

            # 检查区域是否碰撞
            from .iris_np_utils import check_region_collision_optimized
            if check_region_collision_optimized(self.config, checker, region, resolution):
                break

            best_region = region
            current_size += size_increment

            if current_size > max_size:
                break

        if best_region is None:
            best_region = self._create_ellipse_region(
                seed_point, initial_size, aspect_ratio, domain, seed_point
            )

        return best_region

    def box_expansion(
        self,
        checker: SimpleCollisionCheckerForIrisNp,
        seed_point: np.ndarray,
        domain: HPolyhedron,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ) -> Optional[IrisNpRegion]:
        """
        方形膨胀方法（优化版）

        生成方形凸区域，使用配置参数
        """
        current_size = self.config.initial_region_size
        best_region = None

        for iteration in range(self.config.iteration_limit):
            region = self._create_box_region(
                seed_point, current_size, domain, seed_point
            )

            if region is None:
                break

            from .iris_np_utils import check_region_collision_optimized
            if check_region_collision_optimized(self.config, checker, region, resolution):
                break

            best_region = region
            current_size += self.config.size_increment

            if current_size > self.config.max_region_size:
                break

        if best_region is None:
            best_region = self._create_box_region(
                seed_point, self.config.initial_region_size, domain, seed_point
            )

        return best_region

    def _create_ellipse_region(
        self,
        center: np.ndarray,
        size: float,
        aspect_ratio: float,
        domain: HPolyhedron,
        seed_point: np.ndarray
    ) -> Optional[IrisNpRegion]:
        """
        创建椭圆区域（近似为多边形）

        Args:
            center: 膨胀中心点
            size: 基础尺寸
            aspect_ratio: 长宽比
            domain: 域约束
            seed_point: 原始种子点
        """
        try:
            # 椭圆参数
            a = size  # 长轴
            b = size / aspect_ratio  # 短轴

            # 使用多边形近似椭圆（16边形）
            num_vertices = 16
            angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)

            # 椭圆顶点
            vertices_ellipse = np.array([
                [center[0] + a * np.cos(angle), center[1] + b * np.sin(angle)]
                for angle in angles
            ])

            # 转换为半空间表示
            A_region = []
            b_region = []

            for i in range(num_vertices):
                v1 = vertices_ellipse[i]
                v2 = vertices_ellipse[(i + 1) % num_vertices]

                # 计算边的法向量（指向内部）
                edge = v2 - v1
                normal = np.array([edge[1], -edge[0]])  # 法向量
                normal = normal / np.linalg.norm(normal)

                # 半空间约束：normal · x <= normal · v1
                A_region.append(normal)
                b_region.append(np.dot(normal, v1))

            A_region = np.array(A_region)
            b_region = np.array(b_region)

            # 与domain求交
            A_domain = domain.A()
            b_domain = domain.b()

            A_combined = np.vstack([A_region, A_domain])
            b_combined = np.concatenate([b_region, b_domain])

            # 计算顶点
            vertices = compute_polyhedron_vertices_optimized(A_combined, b_combined)

            if len(vertices) < 3:
                return None

            # 计算实际几何中心（可能与膨胀中心不同）
            centroid = np.mean(vertices, axis=0)
            area = compute_polygon_area(vertices)

            return IrisNpRegion(
                A=A_combined,
                b=b_combined,
                vertices=vertices,
                centroid=centroid,
                area=area,
                seed_point=seed_point,  # 原始种子点
                expansion_center=center  # 膨胀中心
            )

        except Exception as e:
            warnings.warn(f"创建椭圆区域失败: {e}")
            return None

    def _create_box_region(
        self,
        center: np.ndarray,
        size: float,
        domain: HPolyhedron,
        seed_point: np.ndarray
    ) -> Optional[IrisNpRegion]:
        """
        创建方形区域

        Args:
            center: 膨胀中心点
            size: 区域大小
            domain: 域约束
            seed_point: 原始种子点
        """
        try:
            # 创建方形约束
            A_region = np.array([
                [1, 0],    # x <= x_max
                [-1, 0],   # -x <= -x_min
                [0, 1],    # y <= y_max
                [0, -1]    # -y <= -y_min
            ])

            b_region = np.array([
                center[0] + size,
                -(center[0] - size),
                center[1] + size,
                -(center[1] - size)
            ])

            # 与domain求交
            A_domain = domain.A()
            b_domain = domain.b()

            A_combined = np.vstack([A_region, A_domain])
            b_combined = np.concatenate([b_region, b_domain])

            # 计算顶点
            vertices = compute_polyhedron_vertices_optimized(A_combined, b_combined)

            if len(vertices) < 3:
                return None

            # 计算几何属性（实际中心可能与膨胀中心不同）
            centroid = np.mean(vertices, axis=0)
            area = compute_polygon_area(vertices)

            return IrisNpRegion(
                A=A_combined,
                b=b_combined,
                vertices=vertices,
                centroid=centroid,
                area=area,
                seed_point=seed_point,  # 原始种子点
                expansion_center=center  # 膨胀中心
            )

        except Exception as e:
            warnings.warn(f"创建区域失败: {e}")
            return None

    def _create_region_from_directions(
        self,
        center: np.ndarray,
        directions: np.ndarray,
        distances: np.ndarray,
        domain: HPolyhedron,
        seed_point: np.ndarray
    ) -> Optional[IrisNpRegion]:
        """
        根据方向和距离创建凸区域

        Args:
            center: 中心点
            directions: 方向向量数组 (N x 2)
            distances: 每个方向的距离 (N,)
            domain: 域约束
            seed_point: 原始种子点
        """
        try:
            num_directions = len(directions)

            # 创建半空间约束
            A_region = []
            b_region = []

            for i in range(num_directions):
                # 每个方向定义一个半空间
                # 法向量指向外部（即方向向量）
                normal = directions[i]

                # 半空间约束：normal · x <= normal · (center + direction * distance)
                # 这表示：在 direction 方向上，点不能超过 boundary_point
                boundary_point = center + directions[i] * distances[i]
                b_val = np.dot(normal, boundary_point)

                A_region.append(normal)
                b_region.append(b_val)

            A_region = np.array(A_region)
            b_region = np.array(b_region)

            # 与 domain 求交
            A_domain = domain.A()
            b_domain = domain.b()

            A_combined = np.vstack([A_region, A_domain])
            b_combined = np.concatenate([b_region, b_domain])

            # 计算顶点
            vertices = compute_polyhedron_vertices_optimized(A_combined, b_combined)

            if len(vertices) < 3:
                return None

            # 计算几何属性
            centroid = np.mean(vertices, axis=0)
            area = compute_polygon_area(vertices)

            return IrisNpRegion(
                A=A_combined,
                b=b_combined,
                vertices=vertices,
                centroid=centroid,
                area=area,
                seed_point=seed_point,
                expansion_center=center
            )

        except Exception as e:
            warnings.warn(f"创建方向区域失败: {e}")
            return None

    def _smooth_direction_distances(
        self,
        directions: np.ndarray,
        distances: np.ndarray,
        max_ratio: float = 1.5,
        iterations: int = 5
    ) -> np.ndarray:
        """
        平滑方向距离，确保连续性

        约束: distances[i] / distances[i-1] ≤ max_ratio
              distances[i] / distances[i+1] ≤ max_ratio

        Args:
            directions: 方向向量数组 (N x 2)
            distances: 距离数组 (N,)
            max_ratio: 最大平滑比率
            iterations: 平滑迭代次数

        Returns:
            平滑后的距离数组
        """
        num_directions = len(directions)
        smoothed = distances.copy()

        for _ in range(iterations):
            for i in range(num_directions):
                prev_idx = (i - 1) % num_directions
                next_idx = (i + 1) % num_directions

                # 计算相邻方向的平均距离
                avg_neighbor = (smoothed[prev_idx] + smoothed[next_idx]) / 2.0

                # 限制最大比率
                if smoothed[i] > avg_neighbor * max_ratio:
                    smoothed[i] = avg_neighbor * max_ratio
                elif smoothed[i] < avg_neighbor / max_ratio:
                    smoothed[i] = avg_neighbor / max_ratio

        return smoothed

    def _detect_and_handle_corners(
        self,
        directions: np.ndarray,
        distances: np.ndarray,
        seed_point: np.ndarray,
        checker: SimpleCollisionCheckerForIrisNp,
        corner_threshold: float = 0.5
    ) -> np.ndarray:
        """
        检测夹角情况并处理

        当某个方向距离远小于相邻方向时，可能是夹角限制
        尝试拉长该方向到相邻方向的平均值

        Args:
            directions: 方向向量数组 (N x 2)
            distances: 距离数组 (N,)
            seed_point: 种子点
            checker: 碰撞检测器
            corner_threshold: 夹角阈值（距离比）

        Returns:
            调整后的距离数组
        """
        num_directions = len(directions)
        adjusted = distances.copy()

        for i in range(num_directions):
            prev_idx = (i - 1) % num_directions
            next_idx = (i + 1) % num_directions

            # 检查是否为夹角：当前方向远小于相邻方向
            if (adjusted[i] < adjusted[prev_idx] * corner_threshold and
                adjusted[i] < adjusted[next_idx] * corner_threshold):

                # 可能是夹角，尝试拉长
                target_distance = (adjusted[prev_idx] + adjusted[next_idx]) / 2.0
                test_point = seed_point + directions[i] * target_distance

                if not checker.check_collision(test_point):
                    # 可以拉长
                    adjusted[i] = target_distance

        return adjusted

    def _check_convex_hull_collision(
        self,
        vertices: np.ndarray,
        checker: SimpleCollisionCheckerForIrisNp
    ) -> bool:
        """
        检查凸包是否碰撞

        通过采样凸包内部点进行碰撞检测

        Args:
            vertices: 凸包顶点 (N x 2)
            checker: 碰撞检测器

        Returns:
            True如果碰撞，False否则
        """
        if len(vertices) < 3:
            return True

        # 计算凸包中心
        centroid = np.mean(vertices, axis=0)

        # 检查中心点
        if checker.check_collision(centroid):
            return True

        # 检查边的中点
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % len(vertices)]
            midpoint = (v1 + v2) / 2.0

            if checker.check_collision(midpoint):
                return True

        # 检查顶点到中心的中间点
        for vertex in vertices:
            midpoint = (centroid + vertex) / 2.0

            if checker.check_collision(midpoint):
                return True

        return False

    def _verify_local_convexity(
        self,
        directions: np.ndarray,
        distances: np.ndarray,
        seed_point: np.ndarray,
        checker: SimpleCollisionCheckerForIrisNp,
        window_size: int = 3,
        shrink_factor: float = 0.9
    ) -> np.ndarray:
        """
        验证局部凸包的有效性

        检查由相邻方向确定的局部凸包是否碰撞
        如果碰撞，缩小该窗口内所有方向的距离

        Args:
            directions: 方向向量数组 (N x 2)
            distances: 距离数组 (N,)
            seed_point: 种子点
            checker: 碰撞检测器
            window_size: 局部窗口大小
            shrink_factor: 缩小因子

        Returns:
            验证后的距离数组
        """
        num_directions = len(directions)
        verified = distances.copy()

        for i in range(num_directions):
            # 获取局部窗口
            indices = [(i + j) % num_directions for j in range(-window_size, window_size + 1)]

            # 创建局部凸包顶点
            local_vertices = []
            for idx in indices:
                vertex = seed_point + directions[idx] * verified[idx]
                local_vertices.append(vertex)

            local_vertices = np.array(local_vertices)

            # 检查局部凸包是否碰撞
            if self._check_convex_hull_collision(local_vertices, checker):
                # 碰撞，需要调整
                # 缩小所有方向的距离
                for idx in indices:
                    verified[idx] *= shrink_factor

        return verified

    # def _apply_direction_coordination(
    #     self,
    #     directions: np.ndarray,
    #     distances: np.ndarray,
    #     seed_point: np.ndarray,
    #     checker: SimpleCollisionCheckerForIrisNp,
    #     enable_smoothing: bool = True,
    #     enable_corner_detection: bool = True,
    #     enable_local_verification: bool = True
    # ) -> np.ndarray:
    #     """
    #     应用方向协同处理

    #     组合平滑、夹角检测和局部验证

    #     Args:
    #         directions: 方向向量数组 (N x 2)
    #         distances: 距离数组 (N,)
    #         seed_point: 种子点
    #         checker: 碰撞检测器
    #         enable_smoothing: 是否启用方向平滑
    #         enable_corner_detection: 是否启用夹角检测
    #         enable_local_verification: 是否启用局部验证

    #     Returns:
    #         协同处理后的距离数组
    #     """
    #     coordinated = distances.copy()

    #     # 1. 方向平滑
    #     if enable_smoothing:
    #         coordinated = self._smooth_direction_distances(
    #             directions, coordinated, max_ratio=1.5, iterations=5
    #         )

    #     # 2. 夹角检测与处理
    #     if enable_corner_detection:
    #         coordinated = self._detect_and_handle_corners(
    #             directions, coordinated, seed_point, checker, corner_threshold=0.5
    #         )

    #     # 3. 局部凸包验证
    #     if enable_local_verification:
    #         coordinated = self._verify_local_convexity(
    #             directions, coordinated, seed_point, checker,
    #             window_size=3, shrink_factor=0.9
    #         )

    #     return coordinated
