"""
局部走廊生成模块

对A*算法规划出的路径结果进行处理,在路径一定范围形成走廊,
走廊外的空间设置为障碍,输出调整过的配置空间。

特性:
1. 带状缓冲区走廊生成
2. 可配置走廊宽度
3. 支持SE(2)配置空间
4. 完整的可视化功能
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, PathPatch
from matplotlib.path import Path
from scipy.ndimage import binary_dilation
import warnings
import sys
import os

# 导入SE2配置空间模块
try:
    from .se2 import SE2ConfigurationSpace, RobotShape
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from se2 import SE2ConfigurationSpace, RobotShape


# === 数值容差常量 ===
# 用于数值计算中的容差判断

NUMERICAL_TOLERANCE: float = 1e-6  # 通用数值计算容差

# 尝试导入A*规划器
try:
    # 添加A*模块路径
    _astar_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "A_pkg", "scripts")
    if _astar_dir not in sys.path:
        sys.path.insert(0, _astar_dir)
    from A_star_fast_optimized import FastSE2AStarPlanner, PlannerConfig
    ASTAR_AVAILABLE = True
except ImportError:
    ASTAR_AVAILABLE = False


@dataclass
class CorridorConfig:
    """走廊配置参数"""
    # 走廊宽度(米)
    corridor_width: float = 5.0

    # 路径平滑参数
    smooth_path: bool = True
    smooth_window: int = 3

    # 走廊边界处理
    boundary_margin: float = 0.5  # 边界额外裕度(米)

    # 碰撞检测
    check_corridor_collision: bool = True
    min_corridor_clearance: float = 0.3  # 最小间隙(米)

    # 可视化参数
    visualization_resolution: int = 100


@dataclass
class CorridorResult:
    """走廊生成结果"""
    # 调整后的配置空间
    adjusted_c_space_2d: np.ndarray
    adjusted_c_space_3d: Optional[np.ndarray] = None

    # 走廊边界信息
    corridor_boundary: List[Tuple[float, float]] = field(default_factory=list)
    corridor_mask: np.ndarray = None  # 走廊区域的布尔掩码

    # 路径信息
    original_path: List[Tuple[float, float, float]] = field(default_factory=list)
    smoothed_path: List[Tuple[float, float, float]] = field(default_factory=list)

    # 统计信息
    corridor_area: float = 0.0  # 走廊面积(平方米)
    original_free_space: float = 0.0  # 原始自由空间
    adjusted_free_space: float = 0.0  # 调整后自由空间
    reduction_ratio: float = 0.0  # 空间缩减比例

    # 配置参数
    config: CorridorConfig = None

    def __post_init__(self):
        if self.config is None:
            self.config = CorridorConfig()


class PathSmoother:
    """路径平滑处理（优化版：向量化操作）"""

    @staticmethod
    def smooth_path(path: List[Tuple[float, float, float]],
                    window_size: int = 3) -> List[Tuple[float, float, float]]:
        """
        使用移动平均平滑路径（向量化优化版）

        Args:
            path: 原始路径点列表
            window_size: 平滑窗口大小

        Returns:
            平滑后的路径点列表
        """
        if len(path) < window_size:
            return path.copy()

        # 【优化】转换为numpy数组进行向量化操作
        path_array = np.array(path)  # shape: (N, 3)
        smoothed = np.zeros_like(path_array)

        # 使用卷积核进行平滑
        kernel = np.ones(window_size) / window_size

        # x, y坐标直接卷积（向量化操作）
        smoothed[:, 0] = np.convolve(path_array[:, 0], kernel, mode='same')
        smoothed[:, 1] = np.convolve(path_array[:, 1], kernel, mode='same')

        # 角度使用圆周均值（向量化）
        sin_vals = np.sin(path_array[:, 2])
        cos_vals = np.cos(path_array[:, 2])
        smoothed[:, 2] = np.arctan2(
            np.convolve(sin_vals, kernel, mode='same'),
            np.convolve(cos_vals, kernel, mode='same')
        )

        # 转换回列表格式
        return [tuple(p) for p in smoothed]

    @staticmethod
    def _circular_mean(angles: List[float]) -> float:
        """计算角度的圆周均值（保留用于兼容性）"""
        if not angles:
            return 0.0

        sin_sum = np.sum([np.sin(a) for a in angles])
        cos_sum = np.sum([np.cos(a) for a in angles])

        return np.arctan2(sin_sum, cos_sum)


class CorridorGenerator:
    """局部走廊生成器（优化版：内存缓存）"""

    def __init__(self, c_space: SE2ConfigurationSpace,
                 config: Optional[CorridorConfig] = None):
        """
        初始化走廊生成器

        Args:
            c_space: SE(2)配置空间对象
            config: 走廊配置参数
        """
        self.c_space = c_space
        self.config = config or CorridorConfig()
        self.path_smoother = PathSmoother()

        # 【优化】缓存坐标网格，避免重复创建
        self._cached_coords = None

    def generate_corridor(self,
                         path: List[Tuple[float, float, float]],
                         robot: Optional[RobotShape] = None) -> CorridorResult:
        """
        为路径生成局部走廊

        Args:
            path: A*算法规划出的路径点列表 [(x, y, theta), ...]
            robot: 机器人形状(可选,用于碰撞检测)

        Returns:
            CorridorResult: 走廊生成结果
        """
        if not path:
            raise ValueError("路径不能为空")

        # 1. 路径平滑处理
        smoothed_path = self._smooth_path(path)

        # 2. 生成走廊掩码
        corridor_mask = self._generate_corridor_mask(smoothed_path)

        # 3. 生成走廊边界
        corridor_boundary = self._extract_corridor_boundary(corridor_mask)

        # 4. 调整配置空间
        adjusted_c_space_2d = self._adjust_c_space_2d(corridor_mask)

        # 5. 如果需要,调整3D配置空间
        adjusted_c_space_3d = None
        if hasattr(self.c_space, '_cache') and self.c_space._cache:
            adjusted_c_space_3d = self._adjust_c_space_3d(corridor_mask)

        # 6. 计算统计信息
        stats = self._compute_statistics(corridor_mask, adjusted_c_space_2d)

        # 7. 构建结果
        result = CorridorResult(
            adjusted_c_space_2d=adjusted_c_space_2d,
            adjusted_c_space_3d=adjusted_c_space_3d,
            corridor_boundary=corridor_boundary,
            corridor_mask=corridor_mask,
            original_path=path,
            smoothed_path=smoothed_path,
            corridor_area=stats['corridor_area'],
            original_free_space=stats['original_free_space'],
            adjusted_free_space=stats['adjusted_free_space'],
            reduction_ratio=stats['reduction_ratio'],
            config=self.config
        )

        return result

    def _smooth_path(self, path: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """路径平滑处理"""
        if self.config.smooth_path:
            return self.path_smoother.smooth_path(path, self.config.smooth_window)
        return path.copy()

    def _generate_corridor_mask(self, path: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        生成走廊掩码(带状缓冲区)

        Args:
            path: 路径点列表

        Returns:
            走廊区域的布尔掩码(True表示走廊区域)
        """
        height, width = self.c_space.height, self.c_space.width
        corridor_mask = np.zeros((height, width), dtype=bool)

        # 转换为像素单位的走廊宽度
        corridor_width_pixels = self.config.corridor_width / self.c_space.resolution
        margin_pixels = self.config.boundary_margin / self.c_space.resolution
        total_width_pixels = corridor_width_pixels + 2 * margin_pixels

        # 为每个路径段生成带状区域
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]

            # 生成路径段的带状区域
            segment_mask = self._generate_segment_corridor(
                p1[:2], p2[:2], total_width_pixels
            )

            corridor_mask = corridor_mask | segment_mask

        # 为最后一个路径点生成圆形区域
        if path:
            last_point = path[-1]
            gx, gy = self.c_space.world_to_grid(last_point[0], last_point[1])
            if 0 <= gx < width and 0 <= gy < height:
                # 【优化】使用缓存的坐标网格
                y_coords, x_coords = self._get_coordinate_grids()
                distance = np.sqrt((x_coords - gx)**2 + (y_coords - gy)**2)
                circle_mask = distance <= total_width_pixels / 2
                corridor_mask = corridor_mask | circle_mask

        return corridor_mask

    def _get_coordinate_grids(self):
        """获取缓存的坐标网格，避免重复创建"""
        if self._cached_coords is None:
            h, w = self.c_space.height, self.c_space.width
            self._cached_coords = np.ogrid[:h, :w]
        return self._cached_coords

    def _generate_segment_corridor(self,
                                   p1: Tuple[float, float],
                                   p2: Tuple[float, float],
                                   width_pixels: float) -> np.ndarray:
        """
        为路径段生成带状走廊区域（优化版：局部区域计算）

        Args:
            p1: 起点坐标(x, y) - 世界坐标
            p2: 终点坐标(x, y) - 世界坐标
            width_pixels: 走廊宽度(像素)

        Returns:
            该段的走廊掩码
        """
        height, width = self.c_space.height, self.c_space.width

        # 转换为网格坐标
        gx1, gy1 = self.c_space.world_to_grid(p1[0], p1[1])
        gx2, gy2 = self.c_space.world_to_grid(p2[0], p2[1])

        # 计算路径段的方向向量
        dx = gx2 - gx1
        dy = gy2 - gy1
        length = np.sqrt(dx**2 + dy**2)

        half_width = width_pixels / 2

        if length < NUMERICAL_TOLERANCE:
            # 起点和终点重合,生成圆形区域（局部计算）
            margin = int(half_width) + 2
            x_min = max(0, int(gx1) - margin)
            x_max = min(width, int(gx1) + margin + 1)
            y_min = max(0, int(gy1) - margin)
            y_max = min(height, int(gy1) + margin + 1)

            if x_max <= x_min or y_max <= y_min:
                return np.zeros((height, width), dtype=bool)

            # 局部坐标网格
            y_local, x_local = np.ogrid[y_min:y_max, x_min:x_max]
            distance = np.sqrt((x_local - gx1)**2 + (y_local - gy1)**2)
            local_mask = distance <= half_width

            # 创建全尺寸mask并填充
            full_mask = np.zeros((height, width), dtype=bool)
            full_mask[y_min:y_max, x_min:x_max] = local_mask
            return full_mask

        # 【优化核心】计算局部边界框
        margin = int(half_width) + 2
        x_min = max(0, int(min(gx1, gx2)) - margin)
        x_max = min(width, int(max(gx1, gx2)) + margin + 1)
        y_min = max(0, int(min(gy1, gy2)) - margin)
        y_max = min(height, int(max(gy1, gy2)) + margin + 1)

        # 检查边界框有效性
        if x_max <= x_min or y_max <= y_min:
            return np.zeros((height, width), dtype=bool)

        # 归一化方向向量
        dx_norm = dx / length
        dy_norm = dy / length

        # 垂直方向向量
        perp_x = -dy_norm
        perp_y = dx_norm

        # 【优化】只在局部区域生成坐标网格
        y_local, x_local = np.ogrid[y_min:y_max, x_min:x_max]

        # 计算局部区域每个点到线段的距离
        # 向量: p1 -> point（使用全局坐标）
        px = x_local - gx1
        py = y_local - gy1

        # 投影到线段方向
        projection = px * dx_norm + py * dy_norm

        # 投影到垂直方向(距离)
        distance_perp = np.abs(px * perp_x + py * perp_y)

        # 判断是否在带状区域内
        extension = half_width  # 两端扩展

        in_corridor = (
            (projection >= -extension) &
            (projection <= length + extension) &
            (distance_perp <= half_width)
        )

        # 对于线段两端,使用圆形区域
        dist_to_start = np.sqrt((x_local - gx1)**2 + (y_local - gy1)**2)
        in_start_circle = dist_to_start <= half_width

        dist_to_end = np.sqrt((x_local - gx2)**2 + (y_local - gy2)**2)
        in_end_circle = dist_to_end <= half_width

        # 合并局部mask
        local_mask = in_corridor | in_start_circle | in_end_circle

        # 创建全尺寸mask并填充局部结果
        full_mask = np.zeros((height, width), dtype=bool)
        full_mask[y_min:y_max, x_min:x_max] = local_mask

        return full_mask

    def _extract_corridor_boundary(self, corridor_mask: np.ndarray) -> List[Tuple[float, float]]:
        """
        提取走廊边界坐标（优化版：使用OpenCV）

        Args:
            corridor_mask: 走廊掩码

        Returns:
            边界点列表[(x, y), ...]
        """
        # 【优化】优先使用OpenCV（性能更好）
        try:
            import cv2
            # 转换为uint8格式
            mask_uint8 = corridor_mask.astype(np.uint8) * 255

            # 提取轮廓（RETR_EXTERNAL只提取外轮廓）
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 转换为世界坐标
            boundary_world = []
            for contour in contours:
                for point in contour:
                    gx, gy = point[0]  # OpenCV格式
                    x, y = self.c_space.grid_to_world(gx, gy)
                    boundary_world.append((x, y))

            return boundary_world

        except ImportError:
            # 【回退】如果OpenCV不可用，使用scipy
            from scipy.ndimage import binary_erosion

            # 腐蚀操作获取边界
            eroded = binary_erosion(corridor_mask)
            boundary_mask = corridor_mask & ~eroded

            # 提取边界点坐标
            boundary_points = np.argwhere(boundary_mask)

            # 转换为世界坐标
            boundary_world = []
            for gy, gx in boundary_points:
                x, y = self.c_space.grid_to_world(gx, gy)
                boundary_world.append((x, y))

            return boundary_world

    def _adjust_c_space_2d(self, corridor_mask: np.ndarray) -> np.ndarray:
        """
        调整2D配置空间

        Args:
            corridor_mask: 走廊掩码

        Returns:
            调整后的配置空间
        """
        # 复制原始障碍物地图
        adjusted_c_space = self.c_space.obstacle_map.copy()

        # 走廊外的区域设置为障碍
        adjusted_c_space[~corridor_mask] = 1

        return adjusted_c_space

    def _adjust_c_space_3d(self, corridor_mask: np.ndarray) -> Optional[np.ndarray]:
        """
        调整3D配置空间

        Args:
            corridor_mask: 走廊掩码

        Returns:
            调整后的3D配置空间
        """
        # 检查是否有缓存的3D配置空间
        if not hasattr(self.c_space, '_cache') or not self.c_space._cache:
            return None

        # 获取第一个缓存的配置空间作为模板
        first_key = list(self.c_space._cache.keys())[0]
        template = self.c_space._cache[first_key]

        # 创建3D配置空间
        num_theta = len(self.c_space._cache)
        height, width = template.shape
        adjusted_3d = np.zeros((height, width, num_theta), dtype=np.uint8)

        # 对每个theta层应用相同的走廊掩码
        for i, (key, c_space_2d) in enumerate(self.c_space._cache.items()):
            adjusted_3d[:, :, i] = c_space_2d.copy()
            adjusted_3d[:, :, i][~corridor_mask] = 1

        return adjusted_3d

    def _compute_statistics(self, corridor_mask: np.ndarray,
                           adjusted_c_space: np.ndarray) -> Dict[str, float]:
        """计算统计信息"""
        resolution = self.c_space.resolution

        # 走廊面积
        corridor_area = np.sum(corridor_mask) * resolution**2

        # 原始自由空间
        original_free = np.sum(self.c_space.obstacle_map == 0) * resolution**2

        # 调整后自由空间
        adjusted_free = np.sum(adjusted_c_space == 0) * resolution**2

        # 缩减比例
        reduction_ratio = 1.0 - (adjusted_free / original_free) if original_free > 0 else 0.0

        return {
            'corridor_area': corridor_area,
            'original_free_space': original_free,
            'adjusted_free_space': adjusted_free,
            'reduction_ratio': reduction_ratio
        }


class CorridorVisualizer:
    """走廊可视化工具"""

    @staticmethod
    def visualize_2d_top_view(result: CorridorResult,
                              c_space: SE2ConfigurationSpace,
                              output_file: Optional[str] = None):
        """
        2D俯视图可视化

        Args:
            result: 走廊生成结果
            c_space: 配置空间对象
            output_file: 输出文件路径(可选)
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # 原始配置空间
        extent = [c_space.origin[0],
                 c_space.origin[0] + c_space.width * c_space.resolution,
                 c_space.origin[1],
                 c_space.origin[1] + c_space.height * c_space.resolution]

        axes[0].imshow(c_space.obstacle_map, cmap='RdYlGn_r',
                      origin='lower', extent=extent, alpha=0.7)
        axes[0].set_title('Original Configuration Space', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_aspect('equal')
        axes[0].grid(True, alpha=0.3)

        # 绘制原始路径
        if result.original_path:
            path_x = [p[0] for p in result.original_path]
            path_y = [p[1] for p in result.original_path]
            axes[0].plot(path_x, path_y, 'b-', linewidth=2, label='Original Path')
            axes[0].legend()

        # 调整后的配置空间
        axes[1].imshow(result.adjusted_c_space_2d, cmap='RdYlGn_r',
                      origin='lower', extent=extent, alpha=0.7)

        # 绘制走廊边界
        if result.corridor_boundary:
            boundary_x = [p[0] for p in result.corridor_boundary]
            boundary_y = [p[1] for p in result.corridor_boundary]
            axes[1].scatter(boundary_x, boundary_y, c='orange', s=1, alpha=0.5, label='Corridor Boundary')

        # 绘制平滑路径
        if result.smoothed_path:
            path_x = [p[0] for p in result.smoothed_path]
            path_y = [p[1] for p in result.smoothed_path]
            axes[1].plot(path_x, path_y, 'b-', linewidth=2, label='Smoothed Path')

        axes[1].set_title('Adjusted Configuration Space (Local Corridor)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X (m)')
        axes[1].set_ylabel('Y (m)')
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # 添加统计信息
        stats_text = (
            f"Corridor Width: {result.config.corridor_width:.1f}m\n"
            f"Corridor Area: {result.corridor_area:.1f}m²\n"
            f"Space Reduction: {result.reduction_ratio*100:.1f}%"
        )
        axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"2D top view saved: {output_file}")

        plt.show()
        plt.close()

    @staticmethod
    def visualize_comparison(result: CorridorResult,
                            c_space: SE2ConfigurationSpace,
                            output_file: Optional[str] = None):
        """
        对比图可视化

        Args:
            result: 走廊生成结果
            c_space: 配置空间对象
            output_file: 输出文件路径(可选)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))

        extent = [c_space.origin[0],
                 c_space.origin[0] + c_space.width * c_space.resolution,
                 c_space.origin[1],
                 c_space.origin[1] + c_space.height * c_space.resolution]

        # 1. 原始障碍物地图
        axes[0, 0].imshow(c_space.obstacle_map, cmap='gray',
                         origin='lower', extent=extent)
        axes[0, 0].set_title('Original Obstacle Map', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('X (m)')
        axes[0, 0].set_ylabel('Y (m)')
        axes[0, 0].set_aspect('equal')

        # 2. 走廊掩码
        axes[0, 1].imshow(result.corridor_mask.astype(float), cmap='Blues',
                         origin='lower', extent=extent)
        axes[0, 1].set_title('Corridor Mask', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('X (m)')
        axes[0, 1].set_ylabel('Y (m)')
        axes[0, 1].set_aspect('equal')

        # 绘制路径
        if result.smoothed_path:
            path_x = [p[0] for p in result.smoothed_path]
            path_y = [p[1] for p in result.smoothed_path]
            axes[0, 1].plot(path_x, path_y, 'r-', linewidth=2, label='Path')

        # 3. 调整后的配置空间
        axes[1, 0].imshow(result.adjusted_c_space_2d, cmap='RdYlGn_r',
                         origin='lower', extent=extent)
        axes[1, 0].set_title('Adjusted Configuration Space', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('X (m)')
        axes[1, 0].set_ylabel('Y (m)')
        axes[1, 0].set_aspect('equal')

        # 4. 差异图
        diff = result.adjusted_c_space_2d.astype(int) - c_space.obstacle_map.astype(int)
        axes[1, 1].imshow(diff, cmap='RdBu', origin='lower', extent=extent,
                         vmin=-1, vmax=1)
        axes[1, 1].set_title('Difference Map (New Obstacles)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('X (m)')
        axes[1, 1].set_ylabel('Y (m)')
        axes[1, 1].set_aspect('equal')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Comparison view saved: {output_file}")

        plt.show()
        plt.close()

    @staticmethod
    def visualize_3d_view(result: CorridorResult,
                         c_space: SE2ConfigurationSpace,
                         theta_index: int = 0,
                         output_file: Optional[str] = None):
        """
        3D视图可视化

        Args:
            result: 走廊生成结果
            c_space: 配置空间对象
            theta_index: theta层索引
            output_file: 输出文件路径(可选)
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(14, 10))

        # 如果有3D配置空间
        if result.adjusted_c_space_3d is not None:
            # 显示多个theta层
            num_theta = result.adjusted_c_space_3d.shape[2]
            num_slices = min(4, num_theta)
            indices = np.linspace(0, num_theta-1, num_slices, dtype=int)

            for i, idx in enumerate(indices):
                ax = fig.add_subplot(2, 2, i+1, projection='3d')

                # 创建网格
                x = np.arange(0, result.adjusted_c_space_3d.shape[1])
                y = np.arange(0, result.adjusted_c_space_3d.shape[0])
                X, Y = np.meshgrid(x, y)

                # 绘制表面
                Z = result.adjusted_c_space_3d[:, :, idx]
                ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

                theta_deg = 360 * idx / num_theta
                ax.set_title(f'theta = {theta_deg:.0f}°', fontsize=12)
                ax.set_xlabel('X (grid)')
                ax.set_ylabel('Y (grid)')
                ax.set_zlabel('Collision')

        else:
            # 仅显示2D配置空间的3D视图
            ax = fig.add_subplot(111, projection='3d')

            x = np.arange(0, result.adjusted_c_space_2d.shape[1])
            y = np.arange(0, result.adjusted_c_space_2d.shape[0])
            X, Y = np.meshgrid(x, y)

            Z = result.adjusted_c_space_2d
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

            ax.set_title('Adjusted Configuration Space (3D View)', fontsize=14, fontweight='bold')
            ax.set_xlabel('X (grid)')
            ax.set_ylabel('Y (grid)')
            ax.set_zlabel('Collision')

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"3D view saved: {output_file}")

        plt.show()
        plt.close()


# ============== 便捷函数 ==============

def create_local_corridor(c_space: SE2ConfigurationSpace,
                         path: List[Tuple[float, float, float]],
                         corridor_width: float = 5.0,
                         smooth_path: bool = True) -> CorridorResult:
    """
    创建局部走廊的便捷函数

    Args:
        c_space: SE(2)配置空间对象
        path: A*路径点列表
        corridor_width: 走廊宽度(米)
        smooth_path: 是否平滑路径

    Returns:
        CorridorResult: 走廊生成结果
    """
    config = CorridorConfig(
        corridor_width=corridor_width,
        smooth_path=smooth_path
    )

    generator = CorridorGenerator(c_space, config)
    return generator.generate_corridor(path)


# ============== A*集成类 ==============

@dataclass
class AStarCorridorConfig:
    """A*走廊集成配置"""
    # A*规划器参数
    min_radius: float = 1.5
    resolution: float = 0.5
    theta_resolution: int = 16
    max_iterations: int = 50000
    goal_tolerance: float = 1.5
    heuristic_weight: float = 1.2

    # 走廊参数
    corridor_width: float = 5.0
    smooth_path: bool = True
    smooth_window: int = 5
    boundary_margin: float = 0.5

    # 可视化
    visualize: bool = True
    save_visualization: bool = False
    output_dir: str = "."


@dataclass
class AStarCorridorResult:
    """A*走廊集成结果"""
    # A*规划结果
    path: Optional[List[Tuple[float, float, float]]] = None
    planning_success: bool = False
    planning_iterations: int = 0

    # 走廊结果
    corridor_result: Optional[CorridorResult] = None

    # 统计信息
    total_time: float = 0.0
    planning_time: float = 0.0
    corridor_time: float = 0.0


class AStarCorridorPlanner:
    """A*路径规划与走廊生成集成类"""

    def __init__(self,
                 c_space: SE2ConfigurationSpace,
                 robot: RobotShape,
                 config: Optional[AStarCorridorConfig] = None):
        """
        初始化A*走廊规划器

        Args:
            c_space: SE(2)配置空间对象
            robot: 机器人形状
            config: 配置参数
        """
        if not ASTAR_AVAILABLE:
            raise ImportError("A* planner not available. Please check A_star_fast_optimized.py")

        self.c_space = c_space
        self.robot = robot
        self.config = config or AStarCorridorConfig()

        # 创建A*规划器
        self._create_astar_planner()

    def _create_astar_planner(self):
        """创建A*规划器"""
        planner_config = PlannerConfig(
            max_iterations=self.config.max_iterations,
            goal_tolerance=self.config.goal_tolerance,
            heuristic_weight=self.config.heuristic_weight,
            verbose=True
        )

        self.astar_planner = FastSE2AStarPlanner(
            c_space=self.c_space,
            robot=self.robot,
            min_radius=self.config.min_radius,
            resolution=self.config.resolution,
            theta_resolution=self.config.theta_resolution,
            config=planner_config
        )

    def plan_with_corridor(self,
                          start: Tuple[float, float, float],
                          goal: Tuple[float, float, float]) -> AStarCorridorResult:
        """
        执行A*路径规划并生成走廊

        Args:
            start: 起点位姿 (x, y, theta)
            goal: 目标位姿 (x, y, theta)

        Returns:
            AStarCorridorResult: 集成结果
        """
        import time

        result = AStarCorridorResult()
        total_start = time.time()

        # 1. A*路径规划
        planning_start = time.time()
        path = self.astar_planner.plan(start, goal)
        result.planning_time = time.time() - planning_start

        if path is None:
            result.planning_success = False
            result.total_time = time.time() - total_start
            return result

        result.path = path
        result.planning_success = True
        result.planning_iterations = self.astar_planner._cache_hits + self.astar_planner._cache_misses

        # 2. 生成走廊
        corridor_start = time.time()
        corridor_config = CorridorConfig(
            corridor_width=self.config.corridor_width,
            smooth_path=self.config.smooth_path,
            smooth_window=self.config.smooth_window,
            boundary_margin=self.config.boundary_margin
        )

        generator = CorridorGenerator(self.c_space, corridor_config)
        result.corridor_result = generator.generate_corridor(path, self.robot)
        result.corridor_time = time.time() - corridor_start

        # 3. 可视化
        if self.config.visualize:
            self._visualize_result(result, start, goal)

        result.total_time = time.time() - total_start
        return result

    def _visualize_result(self, result: AStarCorridorResult,
                         start: Tuple[float, float, float],
                         goal: Tuple[float, float, float]):
        """可视化结果"""
        if result.corridor_result is None:
            return

        output_file = None
        if self.config.save_visualization:
            import os
            output_file = os.path.join(self.config.output_dir, "astar_corridor_result.png")

        CorridorVisualizer.visualize_2d_top_view(
            result.corridor_result, self.c_space,
            output_file=output_file
        )

    def get_adjusted_c_space(self, result: AStarCorridorResult) -> Optional[np.ndarray]:
        """
        获取调整后的配置空间

        Args:
            result: A*走廊集成结果

        Returns:
            调整后的2D配置空间,如果失败返回None
        """
        if result.corridor_result is None:
            return None
        return result.corridor_result.adjusted_c_space_2d


def plan_and_create_corridor(c_space: SE2ConfigurationSpace,
                             robot: RobotShape,
                             start: Tuple[float, float, float],
                             goal: Tuple[float, float, float],
                             corridor_width: float = 5.0,
                             min_radius: float = 1.5,
                             visualize: bool = True) -> Optional[AStarCorridorResult]:
    """
    便捷函数: 执行A*规划并创建走廊

    Args:
        c_space: SE(2)配置空间对象
        robot: 机器人形状
        start: 起点位姿
        goal: 目标位姿
        corridor_width: 走廊宽度(米)
        min_radius: 最小转弯半径(米)
        visualize: 是否可视化

    Returns:
        AStarCorridorResult: 集成结果,失败返回None
    """
    if not ASTAR_AVAILABLE:
        print("Error: A* planner not available")
        return None

    config = AStarCorridorConfig(
        min_radius=min_radius,
        corridor_width=corridor_width,
        visualize=visualize
    )

    planner = AStarCorridorPlanner(c_space, robot, config)
    return planner.plan_with_corridor(start, goal)


# ============== 测试代码 ==============

if __name__ == "__main__":
    print("=" * 60)
    print("局部走廊生成测试 (A*集成)")
    print("=" * 60)

    # 创建测试地图
    map_size = 200
    obstacle_map = np.zeros((map_size, map_size), dtype=np.uint8)

    # 添加一些障碍物
    obstacle_map[40:80, 60:100] = 1
    obstacle_map[120:160, 40:80] = 1
    obstacle_map[100:140, 120:160] = 1

    # 添加圆形障碍物
    for i in range(map_size):
        for j in range(map_size):
            if (i - 150)**2 + (j - 150)**2 < 25**2:
                obstacle_map[i, j] = 1

    # 添加边界墙
    obstacle_map[0:5, :] = 1
    obstacle_map[-5:, :] = 1
    obstacle_map[:, 0:5] = 1
    obstacle_map[:, -5:] = 1

    print(f"地图大小: {map_size}x{map_size}")

    # 创建配置空间
    c_space = SE2ConfigurationSpace(obstacle_map, resolution=0.1)

    # 创建机器人
    try:
        from se2 import create_rectangle_robot
        robot = create_rectangle_robot(length=1.0, width=0.6)
        print(f"机器人尺寸: {robot.length}m x {robot.width}m")
    except:
        robot = None
        print("警告: 无法创建机器人模型")

    # 检查A*是否可用
    if ASTAR_AVAILABLE and robot is not None:
        print("\n使用A*规划器进行路径规划...")

        # 使用A*集成功能
        start = (2.0, 2.0, 0.0)
        goal = (18.0, 18.0, 0.0)

        print(f"起点: {start}")
        print(f"终点: {goal}")

        # 使用便捷函数
        result = plan_and_create_corridor(
            c_space=c_space,
            robot=robot,
            start=start,
            goal=goal,
            corridor_width=5.0,
            min_radius=1.5,
            visualize=True
        )

        if result and result.planning_success:
            print("\n" + "=" * 60)
            print("A*走廊集成结果")
            print("=" * 60)
            print(f"路径规划: 成功")
            print(f"路径点数: {len(result.path)}")
            print(f"规划时间: {result.planning_time:.3f}s")
            print(f"走廊生成时间: {result.corridor_time:.3f}s")
            print(f"总时间: {result.total_time:.3f}s")

            if result.corridor_result:
                cr = result.corridor_result
                print(f"\n走廊统计:")
                print(f"  走廊宽度: {cr.config.corridor_width:.1f}m")
                print(f"  走廊面积: {cr.corridor_area:.1f}m²")
                print(f"  空间缩减: {cr.reduction_ratio*100:.1f}%")
        else:
            print("A*规划失败,使用模拟路径测试走廊生成...")

    else:
        print("\nA*不可用,使用模拟路径测试走廊生成...")

        # 创建测试路径(模拟A*规划结果)
        test_path = [
            (2.0, 2.0, 0.0),
            (4.0, 3.0, 0.2),
            (6.0, 5.0, 0.3),
            (8.0, 7.0, 0.1),
            (10.0, 10.0, 0.0),
            (12.0, 12.0, -0.2),
            (14.0, 14.0, 0.0),
            (16.0, 16.0, 0.1),
            (18.0, 18.0, 0.0)
        ]

        print(f"测试路径点数: {len(test_path)}")

        # 创建走廊生成器
        config = CorridorConfig(
            corridor_width=5.0,
            smooth_path=True,
            smooth_window=3
        )

        generator = CorridorGenerator(c_space, config)

        # 生成走廊
        print("\n生成局部走廊...")
        result = generator.generate_corridor(test_path)

        # 打印统计信息
        print("\n" + "=" * 60)
        print("走廊生成结果")
        print("=" * 60)
        print(f"走廊宽度: {result.config.corridor_width:.1f}m")
        print(f"走廊面积: {result.corridor_area:.1f}m²")
        print(f"原始自由空间: {result.original_free_space:.1f}m²")
        print(f"调整后自由空间: {result.adjusted_free_space:.1f}m²")
        print(f"空间缩减比例: {result.reduction_ratio*100:.1f}%")
        print(f"原始路径点数: {len(result.original_path)}")
        print(f"平滑路径点数: {len(result.smoothed_path)}")
        print(f"走廊边界点数: {len(result.corridor_boundary)}")

        # 可视化
        print("\n生成可视化结果...")

        # 2D俯视图
        CorridorVisualizer.visualize_2d_top_view(
            result, c_space,
            output_file="corridor_2d_top_view.png"
        )

        # 对比图
        CorridorVisualizer.visualize_comparison(
            result, c_space,
            output_file="corridor_comparison.png"
        )

        # 3D视图
        CorridorVisualizer.visualize_3d_view(
            result, c_space,
            output_file="corridor_3d_view.png"
        )

    print("\n测试完成!")
    print("=" * 60)
