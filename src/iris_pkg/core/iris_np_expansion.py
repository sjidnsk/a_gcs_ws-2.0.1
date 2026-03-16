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
        优化的自适应步长膨胀

        优化点：
        1. 动态调整迭代次数（基于最大尺寸和最小步长）
        2. 避免重复创建区域对象（使用快速碰撞检查）
        3. 多层次早期退出机制（收敛判断、步长检查、面积增长检查）
        4. 向量化操作减少循环开销

        特点：
        1. 大步长快速膨胀
        2. 遇到障碍物时自动减小步长
        3. 精确定位边界
        4. 所有方向均衡膨胀
        5. 支持各向异性膨胀（切线优先，法向限制）
        """
        num_directions = len(directions)
        max_size = self.config.max_region_size

        # 每个方向维护独立的步长
        step_sizes = np.ones(num_directions) * self.config.adaptive_initial_step
        min_step = self.config.adaptive_min_step
        step_reduction = self.config.adaptive_step_reduction

        # 如果提供了切线方向，应用各向异性膨胀（向量化）
        if tangent_direction is not None:
            # 向量化计算所有方向与切线的夹角余弦值
            cos_angles = np.abs(np.dot(directions, tangent_direction))
            # 根据夹角调整步长：切线方向步长更大
            step_sizes *= (1.0 + (self.config.tangent_normal_ratio - 1.0) * cos_angles)

        # 优化1: 动态调整最大迭代次数
        # 根据最大尺寸和最小步长计算合理的迭代次数
        max_iterations = min(50, int(max_size / min_step) + 10)

        # 优化2: 多层次早期退出机制
        no_improvement_count = 0
        max_no_improvement = 15  # 连续15次没有改进则退出
        prev_area = 0.0
        convergence_threshold = 0.001  # 面积增长小于1%时认为收敛
        area_check_interval = 5  # 每5次迭代检查一次面积

        for iteration in range(max_iterations):
            improved = False
            improved_count = 0

            for i in range(num_directions):
                # 如果步长已经很小，跳过
                if step_sizes[i] < min_step:
                    continue

                # 尝试膨胀
                new_distance = expansion_distances[i] + step_sizes[i]

                # 限制最大尺寸
                if new_distance > max_size:
                    new_distance = max_size

                # 优化3: 使用快速碰撞检查（避免创建完整区域对象）
                # 只检查边界点是否碰撞，而不是整个区域
                if quick_boundary_collision_check(
                    self.config, checker, seed_point, directions[i], new_distance, resolution
                ):
                    # 碰撞，减小步长
                    step_sizes[i] *= step_reduction
                else:
                    # 无碰撞，接受膨胀
                    expansion_distances[i] = new_distance
                    improved = True
                    improved_count += 1

                    # 如果接近最大尺寸，减小步长
                    if expansion_distances[i] > max_size * 0.8:
                        step_sizes[i] *= step_reduction

            # 优化4: 多层次早期退出
            # 层次1: 连续无改进检查
            if improved:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= max_no_improvement:
                    if self.config.verbose:
                        print(f"    早期退出: 连续{max_no_improvement}次无改进")
                    break

            # 层次2: 所有步长都很小
            if np.all(step_sizes < min_step):
                if self.config.verbose:
                    print(f"    早期退出: 所有步长已达到最小值")
                break

            # 层次3: 面积收敛检查（每N次迭代检查一次）
            if iteration > 0 and iteration % area_check_interval == 0:
                current_area = estimate_area_fast(expansion_distances, directions)
                if prev_area > 0:
                    area_growth = abs(current_area - prev_area) / prev_area
                    if area_growth < convergence_threshold:
                        if self.config.verbose:
                            print(f"    早期退出: 面积收敛 (增长: {area_growth*100:.2f}%)")
                        break
                prev_area = current_area

            # 层次4: 如果大部分方向已经达到最大尺寸
            if np.sum(expansion_distances >= max_size * 0.95) >= num_directions * 0.7:
                if self.config.verbose:
                    print(f"    早期退出: 大部分方向已达到最大尺寸")
                break

        # 创建最终区域（只创建一次）
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
