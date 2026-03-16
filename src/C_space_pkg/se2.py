"""
SE2配置空间生成模块

将二维障碍物地图转化为SE(2)配置空间（x, y, theta），
考虑机器人的几何形状进行碰撞检测。

SE(2) = Special Euclidean Group in 2D
表示二维空间中的刚体变换：位置(x, y) + 朝向(theta)

优化策略:
1. Numba JIT编译加速核心循环（内部并行）
2. 距离场快速筛选（外接圆排除/内切圆确认）
3. 预计算距离场
"""

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from typing import Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Numba JIT编译加速（可选）
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    jit = lambda *args, **kwargs: lambda f: f
    prange = range


@dataclass
class RobotShape:
    """机器人形状定义"""
    shape_type: str  # 'circle', 'rectangle', 'polygon'
    radius: float = 0.0
    length: float = 0.0
    width: float = 0.0
    vertices: Optional[np.ndarray] = None
    
    def get_inscribed_radius(self) -> float:
        """获取内切圆半径"""
        if self.shape_type == 'circle':
            return self.radius
        elif self.shape_type == 'rectangle':
            return min(self.length, self.width) / 2
        elif self.shape_type == 'polygon' and self.vertices is not None:
            return np.min(np.linalg.norm(self.vertices, axis=1))
        return 0.0
    
    def get_circumscribed_radius(self) -> float:
        """获取外接圆半径"""
        if self.shape_type == 'circle':
            return self.radius
        elif self.shape_type == 'rectangle':
            return np.sqrt(self.length**2 + self.width**2) / 2
        elif self.shape_type == 'polygon' and self.vertices is not None:
            return np.max(np.linalg.norm(self.vertices, axis=1))
        return 0.0
    
    def get_vertices_at_pose(self, x: float, y: float, theta: float) -> np.ndarray:
        """获取机器人在指定位姿下的顶点坐标"""
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        if self.shape_type == 'circle':
            angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
            local = np.column_stack([self.radius * np.cos(angles), self.radius * np.sin(angles)])
        elif self.shape_type == 'rectangle':
            hl, hw = self.length / 2, self.width / 2
            local = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
        elif self.shape_type == 'polygon' and self.vertices is not None:
            local = self.vertices
        else:
            raise ValueError(f"未知形状: {self.shape_type}")
        
        return (R @ local.T).T + np.array([x, y])


# ============== Numba加速函数 ==============

@jit(nopython=True, cache=True)
def _check_collision_rectangle(
    obstacle_map: np.ndarray, distance_field: np.ndarray,
    height: int, width: int, resolution: float,
    origin_x: float, origin_y: float,
    x: float, y: float, length: float, width_robot: float, theta: float,
    inscribed_r: float, circumscribed_r: float
) -> bool:
    """Numba加速的矩形碰撞检测"""
    gx = int((x - origin_x) / resolution)
    gy = int((y - origin_y) / resolution)
    
    if gx < 0 or gx >= width or gy < 0 or gy >= height:
        return True
    
    dist = distance_field[gy, gx] * resolution
    if dist >= circumscribed_r:
        return False
    if dist < inscribed_r:
        return True
    
    half_l, half_w = length / 2, width_robot / 2
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    # 检查四个角点
    corners_l = np.array([half_l, half_l, -half_l, -half_l])
    corners_w = np.array([half_w, -half_w, -half_w, half_w])
    
    for i in range(4):
        wx = cos_t * corners_l[i] - sin_t * corners_w[i] + x
        wy = sin_t * corners_l[i] + cos_t * corners_w[i] + y
        cgx = int((wx - origin_x) / resolution)
        cgy = int((wy - origin_y) / resolution)
        if 0 <= cgx < width and 0 <= cgy < height:
            if obstacle_map[cgy, cgx] == 1:
                return True
    
    # 内部采样
    for i in range(5):
        for j in range(5):
            local_l = -half_l + length * i / 4
            local_w = -half_w + width_robot * j / 4
            wx = cos_t * local_l - sin_t * local_w + x
            wy = sin_t * local_l + cos_t * local_w + y
            cgx = int((wx - origin_x) / resolution)
            cgy = int((wy - origin_y) / resolution)
            if 0 <= cgx < width and 0 <= cgy < height:
                if obstacle_map[cgy, cgx] == 1:
                    return True
    return False


@jit(nopython=True, parallel=True, cache=True)
def _generate_c_space_2d_rectangle(
    obstacle_map: np.ndarray, distance_field: np.ndarray,
    height: int, width: int, resolution: float,
    origin_x: float, origin_y: float,
    length: float, width_robot: float, theta: float,
    inscribed_r: float, circumscribed_r: float
) -> np.ndarray:
    """Numba并行生成2D C-space（矩形机器人）"""
    c_space = np.zeros((height, width), dtype=np.uint8)
    
    for gy in prange(height):
        for gx in range(width):
            x = gx * resolution + origin_x
            y = gy * resolution + origin_y
            if _check_collision_rectangle(
                obstacle_map, distance_field, height, width, resolution,
                origin_x, origin_y, x, y, length, width_robot, theta,
                inscribed_r, circumscribed_r
            ):
                c_space[gy, gx] = 1
    return c_space


@jit(nopython=True, parallel=True, cache=True)
def _generate_c_space_2d_circle(
    distance_field: np.ndarray, height: int, width: int, radius_pixels: float
) -> np.ndarray:
    """Numba并行生成2D C-space（圆形机器人）"""
    c_space = np.zeros((height, width), dtype=np.uint8)
    for gy in prange(height):
        for gx in range(width):
            if distance_field[gy, gx] < radius_pixels:
                c_space[gy, gx] = 1
    return c_space


class SE2ConfigurationSpace:
    """SE(2)配置空间生成器"""
    
    def __init__(self, obstacle_map: np.ndarray, resolution: float = 0.1,
                 origin: Tuple[float, float] = (0.0, 0.0)):
        self.obstacle_map = (obstacle_map > 0).astype(np.uint8)
        self.resolution = resolution
        self.origin = np.array(origin, dtype=np.float64)
        self.height, self.width = obstacle_map.shape
        self._distance_field = distance_transform_edt(1 - self.obstacle_map)
        self._cache = {}
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        return int((x - self.origin[0]) / self.resolution), int((y - self.origin[1]) / self.resolution)
    
    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        return gx * self.resolution + self.origin[0], gy * self.resolution + self.origin[1]
    
    def get_distance_to_obstacle(self, x: float, y: float) -> float:
        gx, gy = self.world_to_grid(x, y)
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return self._distance_field[gy, gx] * self.resolution
        return 0.0
    
    def check_collision(self, robot: RobotShape, x: float, y: float, theta: float) -> bool:
        """检查机器人在指定位姿是否碰撞"""
        circ_r = robot.get_circumscribed_radius()
        ins_r = robot.get_inscribed_radius()
        
        dist = self.get_distance_to_obstacle(x, y)
        if dist >= circ_r:
            return False
        if dist < ins_r:
            return True
        
        if robot.shape_type == 'circle':
            return dist < robot.radius
        elif robot.shape_type == 'rectangle':
            return _check_collision_rectangle(
                self.obstacle_map, self._distance_field,
                self.height, self.width, self.resolution,
                self.origin[0], self.origin[1],
                x, y, robot.length, robot.width, theta, ins_r, circ_r
            )
        elif robot.shape_type == 'polygon':
            return self._check_polygon(robot, x, y, theta)
        raise ValueError(f"未知形状: {robot.shape_type}")
    
    def _check_polygon(self, robot: RobotShape, x: float, y: float, theta: float) -> bool:
        """多边形碰撞检测"""
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        world_v = (R @ robot.vertices.T).T + np.array([x, y])
        
        for vx, vy in world_v:
            gx, gy = self.world_to_grid(vx, vy)
            if 0 <= gx < self.width and 0 <= gy < self.height:
                if self.obstacle_map[gy, gx] == 1:
                    return True
        
        min_x, max_x = world_v[:, 0].min(), world_v[:, 0].max()
        min_y, max_y = world_v[:, 1].min(), world_v[:, 1].max()
        
        for sx in np.arange(min_x, max_x, self.resolution):
            for sy in np.arange(min_y, max_y, self.resolution):
                if self._point_in_polygon(sx, sy, world_v):
                    gx, gy = self.world_to_grid(sx, sy)
                    if 0 <= gx < self.width and 0 <= gy < self.height:
                        if self.obstacle_map[gy, gx] == 1:
                            return True
        return False
    
    def _point_in_polygon(self, x: float, y: float, polygon: np.ndarray) -> bool:
        """射线法判断点是否在多边形内"""
        inside = False
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
    
    def generate_c_space_2d(self, robot: RobotShape, theta: float = 0.0) -> np.ndarray:
        """生成固定朝向的2D C-space切片"""
        key = (robot.shape_type, robot.radius, robot.length, robot.width, round(theta, 4))
        if key in self._cache:
            return self._cache[key]
        
        if robot.shape_type == 'circle':
            result = _generate_c_space_2d_circle(
                self._distance_field, self.height, self.width,
                robot.radius / self.resolution
            )
        elif robot.shape_type == 'rectangle':
            result = _generate_c_space_2d_rectangle(
                self.obstacle_map, self._distance_field,
                self.height, self.width, self.resolution,
                self.origin[0], self.origin[1],
                robot.length, robot.width, theta,
                robot.get_inscribed_radius(), robot.get_circumscribed_radius()
            )
        else:
            result = self._generate_c_space_polygon(robot, theta)
        
        self._cache[key] = result
        return result
    
    def _generate_c_space_polygon(self, robot: RobotShape, theta: float) -> np.ndarray:
        """多边形机器人的C-space生成"""
        c_space = np.zeros((self.height, self.width), dtype=np.uint8)
        circ_r = robot.get_circumscribed_radius()
        ins_r = robot.get_inscribed_radius()
        
        for gy in range(self.height):
            for gx in range(self.width):
                x, y = self.grid_to_world(gx, gy)
                dist = self._distance_field[gy, gx] * self.resolution
                if dist >= circ_r:
                    continue
                if dist < ins_r or self._check_polygon(robot, x, y, theta):
                    c_space[gy, gx] = 1
        return c_space
    
    def generate_c_space_se2(self, robot: RobotShape, num_theta: int = 36,
                              theta_range: Tuple[float, float] = (0, 2*np.pi)) -> np.ndarray:
        """生成完整的SE(2)配置空间"""
        thetas = np.linspace(theta_range[0], theta_range[1], num_theta, endpoint=False)
        c_space_3d = np.zeros((self.height, self.width, num_theta), dtype=np.uint8)
        
        for i, theta in enumerate(thetas):
            c_space_3d[:, :, i] = self.generate_c_space_2d(robot, theta)
            print(f"生成C-space: theta层 {i+1}/{num_theta} 完成")
        
        return c_space_3d
    
    def generate_c_space_se2_fast(self, robot: RobotShape, num_theta: int = 36) -> np.ndarray:
        """
        快速生成SE(2)配置空间（使用外接圆近似）
        适用于快速原型开发，牺牲少量精度换取极大速度提升
        """
        radius_pixels = robot.get_circumscribed_radius() / self.resolution
        base_c_space = _generate_c_space_2d_circle(
            self._distance_field, self.height, self.width, radius_pixels
        )
        
        c_space_3d = np.zeros((self.height, self.width, num_theta), dtype=np.uint8)
        for i in range(num_theta):
            c_space_3d[:, :, i] = base_c_space
        
        print(f"快速模式: 使用外接圆近似 (r={robot.get_circumscribed_radius():.3f}m)")
        return c_space_3d
    
    def inflate_obstacles(self, radius: float) -> np.ndarray:
        """膨胀障碍物"""
        r_pix = int(np.ceil(radius / self.resolution))
        y, x = np.ogrid[-r_pix:r_pix+1, -r_pix:r_pix+1]
        struct = (x**2 + y**2 <= r_pix**2).astype(np.uint8)
        return binary_dilation(self.obstacle_map, structure=struct).astype(np.uint8)


class SE2Visualizer:
    """SE2 Configuration Space Visualization Tool"""
    
    @staticmethod
    def visualize_c_space_slice(c_space: np.ndarray, resolution: float = 0.1,
                                 origin: Tuple[float, float] = (0, 0), theta: float = 0.0,
                                 title: str = "C-Space Slice", output_file: Optional[str] = None):
        fig, ax = plt.subplots(figsize=(10, 10))
        extent = [origin[0], origin[0] + c_space.shape[1] * resolution,
                  origin[1], origin[1] + c_space.shape[0] * resolution]
        im = ax.imshow(c_space, cmap='RdYlGn_r', origin='lower', extent=extent)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{title}\ntheta = {np.degrees(theta):.1f} deg')
        plt.colorbar(im, ax=ax, label='Collision')
        ax.set_aspect('equal')
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()
    
    @staticmethod
    def visualize_c_space_3d_slices(c_space_3d: np.ndarray, num_slices: int = 6,
                                     resolution: float = 0.1, origin: Tuple[float, float] = (0, 0),
                                     output_file: Optional[str] = None):
        num_theta = c_space_3d.shape[2]
        indices = np.linspace(0, num_theta-1, num_slices, dtype=int)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        extent = [origin[0], origin[0] + c_space_3d.shape[1] * resolution,
                  origin[1], origin[1] + c_space_3d.shape[0] * resolution]
        
        for i, idx in enumerate(indices):
            theta = 2 * np.pi * idx / num_theta
            axes[i].imshow(c_space_3d[:, :, idx], cmap='RdYlGn_r', origin='lower', extent=extent)
            axes[i].set_title(f'theta = {np.degrees(theta):.1f} deg')
            axes[i].set_xlabel('X (m)')
            axes[i].set_ylabel('Y (m)')
            axes[i].set_aspect('equal')
        
        plt.suptitle('SE(2) Configuration Space Slices', fontsize=14)
        plt.tight_layout()
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()


# 便捷函数
def create_circle_robot(radius: float) -> RobotShape:
    return RobotShape(shape_type='circle', radius=radius)


def create_rectangle_robot(length: float, width: float) -> RobotShape:
    return RobotShape(shape_type='rectangle', length=length, width=width)


def create_polygon_robot(vertices: np.ndarray) -> RobotShape:
    return RobotShape(shape_type='polygon', vertices=vertices)


# 测试
if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("SE(2) 配置空间生成测试")
    print("=" * 60)
    
    # 生成测试地图
    map_size = 200
    obstacle_map = np.zeros((map_size, map_size), dtype=np.uint8)
    obstacle_map[40:80, 60:100] = 1
    obstacle_map[120:160, 40:80] = 1
    obstacle_map[100:140, 120:160] = 1
    
    for i in range(map_size):
        for j in range(map_size):
            if (i - 150)**2 + (j - 150)**2 < 25**2:
                obstacle_map[i, j] = 1
    
    print(f"地图大小: {map_size}x{map_size}")
    
    # 创建配置空间
    c_space = SE2ConfigurationSpace(obstacle_map, resolution=0.1)
    
    # 测试机器人
    rect_robot = create_rectangle_robot(length=1.0, width=0.6)
    circle_robot = create_circle_robot(radius=0.5)
    
    # 性能测试
    print("\n[测试1] 单层C-space生成")
    start = time.time()
    _ = c_space.generate_c_space_2d(rect_robot, theta=0.0)
    print(f"  耗时: {time.time() - start:.4f}秒")
    
    print("\n[测试2] SE(2)配置空间生成（精确模式）")
    start = time.time()
    c_space_3d = c_space.generate_c_space_se2(rect_robot, num_theta=36)
    elapsed_exact = time.time() - start
    print(f"  总耗时: {elapsed_exact:.4f}秒")
    
    print("\n[测试3] SE(2)配置空间生成（快速模式）")
    start = time.time()
    c_space_3d_fast = c_space.generate_c_space_se2_fast(rect_robot, num_theta=36)
    elapsed_fast = time.time() - start
    print(f"  总耗时: {elapsed_fast:.4f}秒")
    
    # ============== 精度对比分析 ==============
    print("\n" + "=" * 60)
    print("精度对比分析")
    print("=" * 60)
    
    total_pixels = c_space_3d.size
    exact_collision = np.sum(c_space_3d == 1)
    fast_collision = np.sum(c_space_3d_fast == 1)
    
    # 差异分析
    diff = c_space_3d_fast.astype(np.int8) - c_space_3d.astype(np.int8)
    false_positives = np.sum(diff == 1)   # 快速模式误判为碰撞
    false_negatives = np.sum(diff == -1)  # 快速模式误判为自由
    
    print(f"\n机器人尺寸: {rect_robot.length}m x {rect_robot.width}m")
    print(f"内切圆半径: {rect_robot.get_inscribed_radius():.3f}m")
    print(f"外接圆半径: {rect_robot.get_circumscribed_radius():.3f}m")
    print(f"外接圆/内切圆比值: {rect_robot.get_circumscribed_radius()/rect_robot.get_inscribed_radius():.3f}")
    
    print(f"\n总像素数: {total_pixels:,}")
    print(f"精确模式碰撞像素: {exact_collision:,} ({100*exact_collision/total_pixels:.2f}%)")
    print(f"快速模式碰撞像素: {fast_collision:,} ({100*fast_collision/total_pixels:.2f}%)")
    
    print(f"\n差异分析:")
    print(f"  快速模式多判为碰撞(假阳性): {false_positives:,} ({100*false_positives/total_pixels:.4f}%)")
    print(f"  快速模式少判为碰撞(假阴性): {false_negatives:,} ({100*false_negatives/total_pixels:.4f}%)")
    print(f"  总差异: {false_positives + false_negatives:,} ({100*(false_positives + false_negatives)/total_pixels:.4f}%)")
    
    # 计算IoU (Intersection over Union)
    intersection = np.sum((c_space_3d == 1) & (c_space_3d_fast == 1))
    union = np.sum((c_space_3d == 1) | (c_space_3d_fast == 1))
    iou = intersection / union if union > 0 else 1.0
    print(f"\nIoU (交并比): {iou:.4f} ({100*iou:.2f}%)")
    
    # 安全性分析
    print(f"\n安全性分析:")
    print(f"  快速模式是保守估计（外接圆），不会漏判碰撞")
    print(f"  假阳性率: {100*false_positives/(fast_collision if fast_collision > 0 else 1):.2f}%")
    print(f"  这意味着快速模式会排除一些实际可行的路径，但不会产生碰撞路径")
    
    # 速度对比
    print(f"\n速度对比:")
    print(f"  精确模式: {elapsed_exact:.4f}秒")
    print(f"  快速模式: {elapsed_fast:.4f}秒")
    print(f"  加速比: {elapsed_exact/elapsed_fast:.1f}x")
    
    # 碰撞检测测试
    print("\n[碰撞检测测试]")
    test_poses = [(5.0, 5.0, 0.0), (10.0, 5.0, np.pi/4)]
    for x, y, theta in test_poses:
        collision = c_space.check_collision(rect_robot, x, y, theta)
        dist = c_space.get_distance_to_obstacle(x, y)
        print(f"  位置({x:.1f}, {y:.1f}, θ={np.degrees(theta):.0f}°): 距离={dist:.2f}m, 碰撞={collision}")
    
    # 可视化对比
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    thetas_to_show = [0, np.pi/4, np.pi/2]
    theta_indices = [0, 9, 18]  # 对应36层中的索引
    
    for col, (theta, idx) in enumerate(zip(thetas_to_show, theta_indices)):
        # 精确模式
        axes[0, col].imshow(c_space_3d[:, :, idx], cmap='RdYlGn_r', origin='lower')
        axes[0, col].set_title(f'Exact Mode theta={np.degrees(theta):.0f} deg')
        axes[0, col].axis('off')
        
        # 快速模式
        axes[1, col].imshow(c_space_3d_fast[:, :, idx], cmap='RdYlGn_r', origin='lower')
        axes[1, col].set_title(f'Fast Mode theta={np.degrees(theta):.0f} deg')
        axes[1, col].axis('off')
    
    plt.suptitle(f'Exact Mode vs Fast Mode Comparison\nIoU={iou:.4f}, Speedup={elapsed_exact/elapsed_fast:.1f}x', fontsize=14)
    plt.tight_layout()
    plt.savefig('se2_precision_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: se2_precision_comparison.png")
    plt.show()
    plt.close()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)
