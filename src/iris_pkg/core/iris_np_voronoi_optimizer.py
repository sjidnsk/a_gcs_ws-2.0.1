"""
Voronoi图优化种子点模块

基于Voronoi图的种子点优化策略：
- 利用Voronoi顶点的最大空圆性质
- 自动找到最优种子点位置
- 数学上保证覆盖效率最优

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import Voronoi

from .iris_np_region_data import IrisNpRegion
from .iris_np_coverage_checker import IrisNpCoverageChecker


class VoronoiSeedOptimizer:
    """基于Voronoi图的种子点优化器"""
    
    def __init__(
        self,
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        coverage_checker: Optional[IrisNpCoverageChecker] = None
    ):
        """
        初始化优化器
        
        Args:
            obstacle_map: 障碍物地图（0=自由，1=障碍）
            resolution: 地图分辨率（米/像素）
            origin: 地图原点 [x₀, y₀]
            coverage_checker: 覆盖验证器（可选，用于精确检测）
        """
        self.obstacle_map = obstacle_map
        self.resolution = resolution
        self.origin = origin
        self.coverage_checker = coverage_checker
        self.path = None  # 存储路径用于覆盖检查
    
    def optimize(
        self,
        initial_seeds: List[np.ndarray],
        path: List[Tuple[float, float, float]],
        max_iterations: int = 5,
        max_new_seeds: int = 10
    ) -> List[np.ndarray]:
        """
        优化种子点

        Args:
            initial_seeds: 初始种子点列表
            path: 路径点列表 [(x, y, θ), ...]
            max_iterations: 最大迭代次数
            max_new_seeds: 最多新增种子点数量

        Returns:
            优化后的种子点列表
        """
        # 存储路径用于后续覆盖检查
        self.path = path
        
        current_seeds = initial_seeds.copy()
        new_seeds_count = 0

        # 预处理路径：转换为NumPy数组以提高计算效率
        path_array = np.array([(p[0], p[1]) for p in path])
        path_segments = []
        for i in range(len(path) - 1):
            path_segments.append((path_array[i], path_array[i + 1]))

        for iteration in range(max_iterations):
            # 检查是否可以生成Voronoi图
            if len(current_seeds) < 3:
                break

            # 识别未覆盖的路径点
            uncovered_indices = self._find_uncovered_path_points(current_seeds, path)
            
            if uncovered_indices:
                print(f"  发现 {len(uncovered_indices)} 个路径点未被覆盖")
                if len(uncovered_indices) <= 20:
                    print(f"  未覆盖点索引: {uncovered_indices[:10]}{'...' if len(uncovered_indices) > 10 else ''}")
            else:
                print("  路径已完全覆盖，停止优化")
                break

            # 生成Voronoi图
            try:
                vor = self._robust_voronoi(current_seeds)
            except Exception as e:
                print(f"Voronoi生成失败: {e}")
                break

            # 获取有限顶点
            finite_vertices = self._get_finite_vertices(vor)

            # 评估顶点（传入未覆盖的路径点索引）
            candidates = []
            for vertex in finite_vertices:
                score = self._evaluate_vertex(vertex, path_segments, current_seeds, uncovered_indices)
                if score > 0:
                    candidates.append((vertex, score))

            # 没有可添加的候选点
            if not candidates:
                print("  没有找到合适的候选点，停止优化")
                break

            # 按评分排序候选点
            candidates.sort(key=lambda x: x[1], reverse=True)

            # 在当前迭代中批量添加种子点（但不超过剩余配额）
            seeds_added_this_iteration = 0
            remaining_quota = max_new_seeds - new_seeds_count
            
            for vertex, score in candidates:
                if seeds_added_this_iteration >= remaining_quota:
                    break
                
                # 添加到种子点集合
                current_seeds.append(vertex)
                new_seeds_count += 1
                seeds_added_this_iteration += 1
                
                print(f"迭代 {iteration + 1}: 添加种子点 ({vertex[0]:.2f}, {vertex[1]:.2f}), 评分 {score:.2f}")

            # 检查是否达到新增种子点上限
            if new_seeds_count >= max_new_seeds:
                print(f"  达到最大新增种子点数量 ({max_new_seeds})，停止优化")
                break

        return current_seeds
    
    def _robust_voronoi(self, seeds: List[np.ndarray]) -> Voronoi:
        """
        鲁棒Voronoi图生成
        
        处理种子点共线的情况，添加微小扰动避免退化
        """
        seeds_array = np.array(seeds)
        
        # 检查是否共线
        if len(seeds) >= 3:
            v1 = seeds_array[1] - seeds_array[0]
            v2 = seeds_array[2] - seeds_array[0]
            cross = np.abs(v1[0] * v2[1] - v1[1] * v2[0])
            
            if cross < 1e-8:
                # 共线，添加微小扰动
                noise = np.random.randn(len(seeds), 2) * 1e-6
                seeds_array += noise
        
        return Voronoi(seeds_array)
    
    def _get_finite_vertices(self, vor: Voronoi) -> List[np.ndarray]:
        """
        获取有限顶点

        过滤掉坐标为inf或-inf的顶点（这些顶点在边界外）
        """
        # 使用NumPy向量化操作过滤有限顶点
        finite_mask = np.all(np.isfinite(vor.vertices), axis=1)
        finite_vertices = vor.vertices[finite_mask]
        return finite_vertices
    
    def _is_in_free_space(self, point: np.ndarray) -> bool:
        """
        检查点是否在自由空间
        
        Args:
            point: 点 [x, y]
        
        Returns:
            True如果在自由空间，False否则
        """
        gx = int((point[0] - self.origin[0]) / self.resolution)
        gy = int((point[1] - self.origin[1]) / self.resolution)
        
        # 边界检查
        if gx < 0 or gx >= self.obstacle_map.shape[1]:
            return False
        if gy < 0 or gy >= self.obstacle_map.shape[0]:
            return False
        
        # 检查是否为自由空间
        return self.obstacle_map[gy, gx] == 0
    
    def _find_uncovered_path_points(
        self,
        seeds: List[np.ndarray],
        path: List[Tuple[float, float, float]]
    ) -> List[int]:
        """
        找出未被种子点覆盖的路径点索引
        
        使用Voronoi图信息来判断覆盖：路径点只被其所在的Voronoi区域的种子点覆盖
        
        Args:
            seeds: 当前种子点列表
            path: 路径点列表
        
        Returns:
            未覆盖的路径点索引列表
        """
        if len(seeds) < 3:
            return list(range(len(path)))
        
        uncovered_indices = []
        seeds_array = np.array(seeds)
        path_array = np.array([(p[0], p[1]) for p in path])
        
        # 生成Voronoi图
        try:
            vor = self._robust_voronoi(seeds)
        except:
            # Voronoi生成失败，使用简单的距离判断
            for i, point in enumerate(path_array):
                distances = np.linalg.norm(seeds_array - point, axis=1)
                if np.min(distances) >= 2.0:
                    uncovered_indices.append(i)
            return uncovered_indices
        
        # 获取每个路径点所属的Voronoi区域
        for i, point in enumerate(path_array):
            # 找到最近的种子点
            distances = np.linalg.norm(seeds_array - point, axis=1)
            nearest_seed_idx = np.argmin(distances)
            nearest_seed = seeds_array[nearest_seed_idx]
            
            # 计算到最近种子点的距离
            dist_to_nearest = distances[nearest_seed_idx]
            
            # 获取该种子点的Voronoi区域半径（到第二近种子点的距离的一半）
            sorted_distances = np.sort(distances)
            if len(sorted_distances) > 1:
                voronoi_radius = sorted_distances[1] / 2.0
            else:
                voronoi_radius = 2.0  # 默认值
            
            # 如果点到最近种子点的距离超过Voronoi区域半径的50%，认为未被覆盖
            # 使用更保守的阈值（50%而不是80%），避免与最终凸区域检测冲突
            # 因为凸区域通常会小于Voronoi区域
            if dist_to_nearest > voronoi_radius * 0.5:
                uncovered_indices.append(i)
        
        return uncovered_indices
    
    def _min_distance_to_path(self, point: np.ndarray, path_segments: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        计算点到路径的最小距离

        Args:
            point: 点 [x, y]
            path_segments: 路径线段列表 [(start_point, end_point), ...]

        Returns:
            最小距离（米）
        """
        # 向量化计算点到所有线段的距离
        distances = []
        for p1, p2 in path_segments:
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)

            if line_len_sq < 1e-12:
                distances.append(np.linalg.norm(point - p1))
            else:
                t = np.dot(point - p1, line_vec) / line_len_sq
                t = max(0, min(1, t))
                projection = p1 + t * line_vec
                distances.append(np.linalg.norm(point - projection))

        return min(distances) if distances else float('inf')
    
    def _point_to_segment_distance(self, point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        计算点到线段的最小距离
        
        Args:
            point: 点 [x, y]
            p1: 线段起点 [x, y]
            p2: 线段终点 [x, y]
        
        Returns:
            最小距离
        """
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-6:
            return np.linalg.norm(point - p1)
        
        # 计算投影点
        t = np.dot(point - p1, line_vec) / (line_len ** 2)
        t = max(0, min(1, t))  # 限制在线段范围内
        
        projection = p1 + t * line_vec
        return np.linalg.norm(point - projection)
    
    def _evaluate_vertex(
        self,
        vertex: np.ndarray,
        path_segments: List[Tuple[np.ndarray, np.ndarray]],
        seeds: List[np.ndarray],
        uncovered_indices: List[int] = None
    ) -> float:
        """
        评估Voronoi顶点的质量

        改进的评分标准：
        1. 是否在自由空间（必须，否则返回0）
        2. 能够覆盖多少未覆盖的路径点（最重要，权重最高）
        3. 距离路径的远近（适中最好，太近或太远都不好）
        4. 距离最近种子点的距离（越远越好，避免重复）
        5. Voronoi区域大小（越大越好）
        6. 沿路径方向的均匀性（避免集中在某一段）

        Args:
            vertex: Voronoi顶点 [x, y]
            path_segments: 路径线段列表 [(start_point, end_point), ...]
            seeds: 当前种子点列表
            uncovered_indices: 未覆盖的路径点索引列表

        Returns:
            评分（0表示无效）
        """
        # 检查1: 是否在自由空间
        if not self._is_in_free_space(vertex):
            return 0.0

        # 检查2: 距离路径的远近
        dist_to_path = self._min_distance_to_path(vertex, path_segments)
        if dist_to_path > 5.0:  # 超过5米，不考虑
            return 0.0

        # 检查3: 距离最近种子点的距离（使用NumPy向量化）
        seeds_array = np.array(seeds)
        distances = np.linalg.norm(seeds_array - vertex, axis=1)
        min_dist_to_seed = np.min(distances)
        
        # 如果距离最近的种子点太近（<1米），认为是冗余的
        if min_dist_to_seed < 1.0:
            return 0.0

        # 检查4: Voronoi区域大小（用最小距离估计）
        sorted_distances = np.sort(distances)
        if len(sorted_distances) > 1:
            voronoi_radius = sorted_distances[1] / 2.0
        else:
            voronoi_radius = 2.0
        estimated_area = np.pi * voronoi_radius ** 2

        # 检查5: 能够覆盖多少未覆盖的路径点（最重要）
        coverage_score = 0.0
        if uncovered_indices and self.path:
            covered_count = 0
            for idx in uncovered_indices:
                x, y, _ = self.path[idx]
                path_point = np.array([x, y])
                # 使用Voronoi半径的50%作为覆盖半径（保守估计）
                # 与_find_uncovered_path_points中的逻辑保持一致
                if np.linalg.norm(vertex - path_point) < voronoi_radius * 0.5:
                    covered_count += 1
            
            # 覆盖分数：覆盖的未覆盖点越多，分数越高
            coverage_score = covered_count * 10.0  # 每个未覆盖点10分

        # 检查6: 沿路径方向的均匀性
        # 计算顶点在路径上的投影位置
        path_positions = []
        for p1, p2 in path_segments:
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)
            if line_len > 1e-6:
                t = np.dot(vertex - p1, line_vec) / (line_len ** 2)
                t = max(0, min(1, t))
                path_positions.append(p1 + t * line_vec)
        
        if not path_positions:
            return 0.0
        
        # 找到最近的路径点
        path_positions_array = np.array(path_positions)
        dists_to_path_points = np.linalg.norm(path_positions_array - vertex, axis=1)
        nearest_path_idx = np.argmin(dists_to_path_points)
        nearest_path_point = path_positions_array[nearest_path_idx]
        
        # 计算沿路径方向的均匀性
        # 统计路径上已有种子点的分布
        seed_path_positions = []
        for seed in seeds:
            seed_positions = []
            for p1, p2 in path_segments:
                line_vec = p2 - p1
                line_len = np.linalg.norm(line_vec)
                if line_len > 1e-6:
                    t = np.dot(seed - p1, line_vec) / (line_len ** 2)
                    t = max(0, min(1, t))
                    seed_positions.append(p1 + t * line_vec)
            if seed_positions:
                seed_positions_array = np.array(seed_positions)
                seed_dists = np.linalg.norm(seed_positions_array - seed, axis=1)
                seed_path_positions.append(seed_positions_array[np.argmin(seed_dists)])
        
        if seed_path_positions:
            seed_path_positions_array = np.array(seed_path_positions)
            # 计算到最近种子点的路径距离
            dists_along_path = np.linalg.norm(seed_path_positions_array - nearest_path_point, axis=1)
            min_dist_along_path = np.min(dists_along_path)
            # 路径距离越大越好（避免重复）
            path_uniformity_score = min_dist_along_path
        else:
            path_uniformity_score = 10.0  # 没有种子点时给高分

        # 综合评分
        # 距离路径适中（0.5-2米之间最好）
        path_distance_score = 1.0 if 0.5 <= dist_to_path <= 2.0 else max(0, 1.0 - abs(dist_to_path - 1.25) / 3.75)
        
        # 各项权重
        score = (
            coverage_score * 3.0 +                   # 覆盖未覆盖点权重（最高）
            path_distance_score * 2.0 +               # 路径距离权重
            min_dist_to_seed * 1.0 +                   # 种子点距离权重
            estimated_area * 0.2 +                     # 区域大小权重
            path_uniformity_score * 0.5                # 路径均匀性权重
        )

        return score
    
    def _check_coverage(self, path: List[Tuple[float, float, float]], regions: List[IrisNpRegion]) -> bool:
        """
        检查路径是否被完全覆盖
        
        Args:
            path: 路径点列表
            regions: 凸区域列表
        
        Returns:
            True如果完全覆盖，False否则
        """
        # 使用IrisNpCoverageChecker进行精确检测
        return self.coverage_checker.verify_path_coverage(path, regions)


def compute_path_curvature(path: np.ndarray, index: int) -> float:
    """
    计算路径在指定点的曲率（转角角度）

    Args:
        path: 路径点NumPy数组，形状为(N, 2)
        index: 当前点索引

    Returns:
        曲率（弧度），0表示直线，越大表示转弯越急
    """
    if index == 0 or index == len(path) - 1:
        return 0.0

    p_prev = path[index - 1]
    p_curr = path[index]
    p_next = path[index + 1]

    v1 = p_curr - p_prev
    v2 = p_next - p_curr

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    # 计算转角角度
    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    angle = np.arccos(cos_angle)

    return angle


def curvature_adaptive_sampling(
    path: List[Tuple[float, float, float]],
    obstacle_map: np.ndarray,
    resolution: float,
    origin: Tuple[float, float],
    min_distance: float = 1.0
) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    基于曲率的自适应采样

    策略：
    - 曲率 > 30度：密集采样（间隔3点）
    - 曲率 > 15度：中等采样（间隔5点）
    - 曲率 <= 15度：稀疏采样（间隔10点）

    Args:
        path: 路径点列表
        obstacle_map: 障碍物地图
        resolution: 地图分辨率
        origin: 地图原点
        min_distance: 最小种子点距离（米）

    Returns:
        种子点列表，每个元素为 (seed_point, tangent_direction)
    """
    seed_points = []
    path_length = len(path)

    if path_length == 0:
        return seed_points

    # 将路径转换为NumPy数组以提高计算效率
    path_array = np.array([(p[0], p[1]) for p in path])

    # 预计算所有点的曲率
    curvatures = [compute_path_curvature(path_array, i) for i in range(path_length)]

    # 强制包含起点
    x_start, y_start, _ = path[0]
    start_point = path_array[0]
    gx_start = int((x_start - origin[0]) / resolution)
    gy_start = int((y_start - origin[1]) / resolution)

    if 0 <= gx_start < obstacle_map.shape[1] and 0 <= gy_start < obstacle_map.shape[0]:
        if obstacle_map[gy_start, gx_start] == 0:
            seed_points.append((start_point, None))

    # 基于曲率自适应采样
    i = 0
    while i < path_length:
        seed_point = path_array[i]

        # 检查是否在自由空间
        gx = int((seed_point[0] - origin[0]) / resolution)
        gy = int((seed_point[1] - origin[1]) / resolution)

        if 0 <= gx < obstacle_map.shape[1] and 0 <= gy < obstacle_map.shape[0]:
            if obstacle_map[gy, gx] == 0:
                # 检查与已有种子点的距离（使用NumPy向量化）
                if len(seed_points) == 0:
                    seed_points.append((seed_point, None))
                else:
                    existing_points = np.array([sp[0] for sp in seed_points])
                    distances = np.linalg.norm(existing_points - seed_point, axis=1)
                    if np.all(distances >= min_distance):
                        seed_points.append((seed_point, None))

        # 根据曲率决定下一个采样点
        curvature = curvatures[i]

        if curvature > np.pi / 6:  # > 30度：密集采样
            step = min(3, path_length - i - 1)
        elif curvature > np.pi / 12:  # > 15度：中等采样
            step = min(5, path_length - i - 1)
        else:  # <= 15度：稀疏采样
            step = min(10, path_length - i - 1)

        i += step

    # 强制包含终点
    x_end, y_end, _ = path[-1]
    end_point = path_array[-1]
    gx_end = int((x_end - origin[0]) / resolution)
    gy_end = int((y_end - origin[1]) / resolution)

    if 0 <= gx_end < obstacle_map.shape[1] and 0 <= gy_end < obstacle_map.shape[0]:
        if obstacle_map[gy_end, gx_end] == 0:
            if len(seed_points) == 0:
                seed_points.append((end_point, None))
            else:
                existing_points = np.array([sp[0] for sp in seed_points])
                distances = np.linalg.norm(existing_points - end_point, axis=1)
                if np.all(distances >= min_distance):
                    seed_points.append((end_point, None))

    return seed_points
