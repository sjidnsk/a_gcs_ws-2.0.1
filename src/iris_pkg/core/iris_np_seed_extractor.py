"""
IrisNp 种子点提取模块

负责从路径中提取种子点，支持两批扩张策略：
- 第一批：正常扩张，均匀采样路径点
- 第二批：检查未覆盖路径点，优先沿切线方向膨胀

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from scipy.spatial import KDTree

from config.iris import IrisNpConfig
from .iris_np_region_data import IrisNpRegion


class IrisNpSeedExtractor:
    """IrisNp 种子点提取器"""

    def __init__(self, config: IrisNpConfig):
        """
        初始化种子点提取器

        Args:
            config: IrisNp 配置参数
        """
        self.config = config

    def extract_seed_points(
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
            seed_points = self._extract_first_batch_seeds(
                path, obstacle_map, resolution, origin
            )
        elif batch == 2:
            seed_points = self._extract_second_batch_seeds(
                path, obstacle_map, resolution, origin, existing_regions
            )

        return seed_points

    def _extract_first_batch_seeds(
        self,
        path: List[Tuple[float, float, float]],
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """提取第一批种子点（正常扩张）"""
        seed_points = []
        path_length = len(path)

        # 计算路径总长度
        path_total_length = 0.0
        for i in range(1, path_length):
            x0, y0, _ = path[i-1]
            x1, y1, _ = path[i]
            path_total_length += np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

        # 优化1: 调整密度系数，平衡种子点密度和覆盖完整性
        # 从0.2改为0.45，减少约55%的种子点，提高效率
        density_factor = 0.45  # 平衡密度和覆盖

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

        # 优化1: 强制包含起点
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
                    # 优化2: 自适应最小距离（根据局部路径密度）
                    adaptive_min_distance = self._compute_adaptive_min_distance(
                        path, i, path_total_length
                    )
                    # 检查与已有种子点的距离
                    if self._is_valid_seed_adaptive(seed_point, [sp[0] for sp in seed_points], adaptive_min_distance):
                        # 第一批不需要切线方向
                        seed_points.append((seed_point, None))

        # 优化1: 强制包含终点
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

        return seed_points

    def _extract_second_batch_seeds(
        self,
        path: List[Tuple[float, float, float]],
        obstacle_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        existing_regions: Optional[List[IrisNpRegion]]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """提取第二批种子点（检查未覆盖路径点）"""
        seed_points = []
        path_length = len(path)

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

                # 优化3: 提高边界点检测阈值，减少冗余种子点
                # 如果邻域内有未覆盖的点，说明该点在区域边界附近
                # 添加为种子点可以确保新生成的区域与已有区域重叠
                # 从3改为4，更严格的边界点条件
                if coverage_info['uncovered_count'] >= 4:
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

                        # 优化4: 使用放宽的距离限制（0.8米）
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
        优化4: 放宽距离限制的种子点验证

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
        优化2: 计算自适应最小距离

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
            # 密集区域：减小距离到85%（从70%改为85%，减少过度密集）
            adaptive_distance = base_distance * 0.85
        elif density_ratio < 0.7:
            # 稀疏区域：保持正常距离
            adaptive_distance = base_distance
        else:
            # 正常区域：略微减小距离到90%（从85%改为90%）
            adaptive_distance = base_distance * 0.9

        return adaptive_distance

    def _is_valid_seed_adaptive(
        self,
        candidate: np.ndarray,
        existing: List[np.ndarray],
        adaptive_min_distance: float
    ) -> bool:
        """
        优化2: 使用自适应最小距离验证种子点

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
