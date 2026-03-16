"""
IrisNp 2D区域到3D配置空间扩展模块

将IrisNp生成的2D凸区域 (x, y) 扩展到3D配置空间 (x, y, theta)，
用于GCS在SE(2)配置空间中进行路径规划。

核心功能：
1. 将2D凸区域扩展为3D凸区域
2. 处理theta维度的约束（周期性、连续性）
3. 支持GCS在(x, y, theta)空间进行规划

关键约束：
- theta ∈ [0, 2π)，具有周期性
- theta与(x, y)的约束关系需要根据路径信息确定
- 支持theta的连续性约束（避免角度跳变）

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import warnings

# Drake导入
try:
    from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    warnings.warn("Drake未安装，部分功能不可用")


@dataclass
class ThetaRangeConfig:
    """Theta范围配置"""
    # Theta范围
    theta_min: float = 0.0
    theta_max: float = 2 * np.pi
    
    # 是否允许theta跨越边界（0和2π）
    allow_wrap_around: bool = True
    
    # Theta变化率限制（rad/m）
    max_theta_change_rate: float = np.pi / 2  # 每米最大角度变化
    
    # Theta连续性约束
    enforce_continuity: bool = True
    max_theta_jump: float = np.pi / 4  # 相邻区域间最大角度跳变
    
    # 是否使用路径信息约束theta
    use_path_constraint: bool = True
    path_theta_margin: float = np.pi / 6  # 路径theta的扩展裕度


@dataclass
class IrisNpRegion3D:
    """
    3D凸区域 (x, y, theta)
    
    从2D区域扩展而来，添加theta维度约束
    """
    # 多面体表示 (Ax <= b)，维度为3
    A: np.ndarray  # 约束矩阵 (M x 3)
    b: np.ndarray  # 约束向量 (M,)
    
    # 几何信息
    vertices_2d: np.ndarray  # 2D顶点坐标 (N x 2)
    centroid_2d: np.ndarray  # 2D中心点 (2,)
    theta_range: Tuple[float, float]  # theta范围 [theta_min, theta_max]
    
    # 元数据
    seed_point: np.ndarray  # 原始种子点 (x, y, theta)
    region_id: int = 0
    
    # 原始2D区域信息
    original_A: np.ndarray = None  # 原始2D约束矩阵
    original_b: np.ndarray = None  # 原始2D约束向量
    
    def contains(self, point: np.ndarray, tol: float = 1e-6) -> bool:
        """
        检查点是否在区域内
        
        Args:
            point: 3D点 (x, y, theta)
            tol: 容差
        
        Returns:
            True如果点在区域内
        """
        if len(point) != 3:
            raise ValueError(f"点维度必须为3，当前为{len(point)}")
        
        # 检查theta是否在范围内（考虑周期性）
        theta = point[2]
        theta_min, theta_max = self.theta_range
        
        # 处理theta周期性
        if theta < theta_min - tol or theta > theta_max + tol:
            # 检查是否跨越边界
            if theta_min == 0.0 and theta_max == 2 * np.pi:
                # theta应该在[0, 2π)范围内
                theta_normalized = theta % (2 * np.pi)
                if theta_normalized < theta_min - tol or theta_normalized > theta_max + tol:
                    return False
            else:
                return False
        
        # 检查多面体约束
        return np.all(self.A @ point <= self.b + tol)
    
    def to_hpolyhedron(self):
        """
        转换为Drake HPolyhedron对象
        
        Returns:
            HPolyhedron对象
        """
        if not DRAKE_AVAILABLE:
            raise RuntimeError("Drake未安装，无法创建HPolyhedron")
        
        return HPolyhedron(self.A, self.b)
    
    def get_vertices_3d(self, num_theta_samples: int = 10) -> np.ndarray:
        """
        获取3D顶点（用于可视化）
        
        Args:
            num_theta_samples: theta方向的采样点数
        
        Returns:
            3D顶点数组 (K x 3)
        """
        vertices_3d = []
        theta_min, theta_max = self.theta_range
        thetas = np.linspace(theta_min, theta_max, num_theta_samples)
        
        for theta in thetas:
            for vertex_2d in self.vertices_2d:
                vertices_3d.append([vertex_2d[0], vertex_2d[1], theta])
        
        return np.array(vertices_3d)


class IrisRegion3DAdapter:
    """
    2D区域到3D区域的适配器
    
    将IrisNp生成的2D凸区域扩展到3D配置空间
    """
    
    def __init__(self, config: Optional[ThetaRangeConfig] = None):
        """
        初始化适配器
        
        Args:
            config: Theta范围配置
        """
        self.config = config or ThetaRangeConfig()
    
    def expand_region_to_3d(
        self,
        region_2d: Any,
        seed_point: np.ndarray,
        path_theta: Optional[float] = None,
        neighbor_thetas: Optional[List[float]] = None
    ) -> IrisNpRegion3D:
        """
        将2D区域扩展为3D区域
        
        Args:
            region_2d: 2D区域对象（IrisNpRegion或HPolyhedron）
            seed_point: 种子点 (x, y, theta)
            path_theta: 路径在该点的theta值
            neighbor_thetas: 相邻区域的theta值列表
        
        Returns:
            3D区域对象
        """
        # 提取2D区域信息
        if hasattr(region_2d, 'A') and hasattr(region_2d, 'b'):
            # IrisNpRegion对象
            A_2d = region_2d.A
            b_2d = region_2d.b
            vertices_2d = region_2d.vertices
            centroid_2d = region_2d.centroid
        elif hasattr(region_2d, 'A') and hasattr(region_2d, 'b'):
            # HPolyhedron对象
            A_2d = region_2d.A()
            b_2d = region_2d.b()
            # 需要计算顶点
            vertices_2d = self._compute_vertices_from_hpoly(region_2d)
            centroid_2d = vertices_2d.mean(axis=0)
        else:
            raise TypeError(f"不支持的区域类型: {type(region_2d)}")
        
        # 确定theta范围
        theta_range = self._determine_theta_range(
            seed_point, path_theta, neighbor_thetas
        )
        
        # 构建3D约束矩阵
        A_3d, b_3d = self._build_3d_constraints(
            A_2d, b_2d, theta_range
        )
        
        # 创建3D区域
        region_3d = IrisNpRegion3D(
            A=A_3d,
            b=b_3d,
            vertices_2d=vertices_2d,
            centroid_2d=centroid_2d,
            theta_range=theta_range,
            seed_point=seed_point,
            original_A=A_2d,
            original_b=b_2d
        )
        
        return region_3d
    
    def expand_regions_from_path(
        self,
        regions_2d: List[Any],
        path: List[Tuple[float, float, float]]
    ) -> List[IrisNpRegion3D]:
        """
        从路径批量扩展2D区域到3D
        
        Args:
            regions_2d: 2D区域列表
            path: 路径点列表 [(x, y, theta), ...]
        
        Returns:
            3D区域列表
        """
        regions_3d = []
        
        for i, region_2d in enumerate(regions_2d):
            # 获取路径theta
            path_theta = path[i][2] if i < len(path) else None
            
            # 获取相邻区域的theta
            neighbor_thetas = []
            if i > 0 and i - 1 < len(path):
                neighbor_thetas.append(path[i-1][2])
            if i < len(path) - 1:
                neighbor_thetas.append(path[i+1][2])
            
            # 扩展区域
            seed_point = np.array(path[i]) if i < len(path) else np.array([0, 0, 0])
            region_3d = self.expand_region_to_3d(
                region_2d, seed_point, path_theta, neighbor_thetas
            )
            region_3d.region_id = i
            regions_3d.append(region_3d)
        
        return regions_3d
    
    def _determine_theta_range(
        self,
        seed_point: np.ndarray,
        path_theta: Optional[float],
        neighbor_thetas: Optional[List[float]]
    ) -> Tuple[float, float]:
        """
        确定theta范围 - 取消限制，覆盖整个圆周

        Args:
            seed_point: 种子点
            path_theta: 路径theta值
            neighbor_thetas: 相邻区域theta值

        Returns:
            (theta_min, theta_max)
        """
        # 直接返回完整的theta范围（-π, π）
        # 这样所有IRIS区域在theta维度上都是完全连通的
        return (-np.pi, np.pi)
    
    def _build_3d_constraints(
        self,
        A_2d: np.ndarray,
        b_2d: np.ndarray,
        theta_range: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建3D约束矩阵
        
        将2D约束 (A_2d * [x, y]^T <= b_2d) 扩展为3D约束
        
        Args:
            A_2d: 2D约束矩阵 (M x 2)
            b_2d: 2D约束向量 (M,)
            theta_range: theta范围
        
        Returns:
            (A_3d, b_3d): 3D约束矩阵和向量
        """
        theta_min, theta_max = theta_range
        M = A_2d.shape[0]
        
        # 构建3D约束矩阵
        # 原始2D约束：A_2d * [x, y]^T <= b_2d
        # 扩展为：[A_2d, 0] * [x, y, theta]^T <= b_2d
        A_3d_xy = np.hstack([A_2d, np.zeros((M, 1))])
        b_3d_xy = b_2d
        
        # 添加theta约束
        # theta >= theta_min  =>  -theta <= -theta_min
        # theta <= theta_max
        A_theta = np.array([
            [0, 0, -1],  # -theta <= -theta_min
            [0, 0, 1]    # theta <= theta_max
        ])
        b_theta = np.array([-theta_min, theta_max])
        
        # 合并约束
        A_3d = np.vstack([A_3d_xy, A_theta])
        b_3d = np.concatenate([b_3d_xy, b_theta])
        
        return A_3d, b_3d
    
    def _compute_vertices_from_hpoly(
        self,
        hpoly: 'HPolyhedron',
        max_iterations: int = 1000
    ) -> np.ndarray:
        """
        从HPolyhedron计算顶点
        
        Args:
            hpoly: HPolyhedron对象
            max_iterations: 最大迭代次数
        
        Returns:
            顶点数组 (N x 2)
        """
        try:
            # 尝试转换为VPolytope
            vpoly = hpoly.ToVPolytope()
            vertices = np.array(vpoly.vertices())
            return vertices.T  # 转置为 (N x 2)
        except Exception as e:
            warnings.warn(f"无法从HPolyhedron提取顶点: {e}")
            # 返回空数组
            return np.array([])


def convert_iris_regions_to_3d(
    iris_result: Any,
    path: List[Tuple[float, float, float]],
    config: Optional[ThetaRangeConfig] = None
) -> List[IrisNpRegion3D]:
    """
    将IrisNp结果转换为3D区域列表
    
    便捷函数，用于将IrisNp生成的2D区域批量转换为3D区域
    
    Args:
        iris_result: IrisNp结果对象
        path: 路径点列表 [(x, y, theta), ...]
        config: Theta配置
    
    Returns:
        3D区域列表
    """
    adapter = IrisRegion3DAdapter(config)
    
    # 提取2D区域
    if hasattr(iris_result, 'regions'):
        regions_2d = iris_result.regions
    else:
        raise TypeError("iris_result必须包含regions属性")
    
    # 扩展为3D
    regions_3d = adapter.expand_regions_from_path(regions_2d, path)
    
    return regions_3d


def create_hpolyhedron_list_from_3d_regions(
    regions_3d: List[IrisNpRegion3D]
) -> List['HPolyhedron']:
    """
    从3D区域列表创建HPolyhedron列表
    
    用于GCS规划
    
    Args:
        regions_3d: 3D区域列表
    
    Returns:
        HPolyhedron列表
    """
    if not DRAKE_AVAILABLE:
        raise RuntimeError("Drake未安装，无法创建HPolyhedron")
    
    hpoly_list = []
    for region in regions_3d:
        try:
            hpoly = region.to_hpolyhedron()
            hpoly_list.append(hpoly)
        except Exception as e:
            warnings.warn(f"区域{region.region_id}转换失败: {e}")
    
    return hpoly_list


# ============================================================================
# 测试代码
# ============================================================================

def test_3d_region_adapter():
    """测试3D区域适配器"""
    print("=" * 60)
    print("测试3D区域适配器")
    print("=" * 60)
    
    # 创建模拟的2D区域
    A_2d = np.array([
        [1, 0],   # x <= 5
        [-1, 0],  # x >= 0
        [0, 1],   # y <= 5
        [0, -1]   # y >= 0
    ])
    b_2d = np.array([5, 0, 5, 0])
    
    # 创建模拟的IrisNpRegion
    from dataclasses import dataclass
    
    @dataclass
    class MockRegion2D:
        A: np.ndarray
        b: np.ndarray
        vertices: np.ndarray
        centroid: np.ndarray
    
    vertices_2d = np.array([
        [0, 0], [5, 0], [5, 5], [0, 5]
    ])
    centroid_2d = np.array([2.5, 2.5])
    
    region_2d = MockRegion2D(
        A=A_2d,
        b=b_2d,
        vertices=vertices_2d,
        centroid=centroid_2d
    )
    
    # 创建适配器
    config = ThetaRangeConfig(
        use_path_constraint=True,
        path_theta_margin=np.pi / 4
    )
    adapter = IrisRegion3DAdapter(config)
    
    # 测试单个区域扩展
    seed_point = np.array([2.5, 2.5, np.pi / 2])
    region_3d = adapter.expand_region_to_3d(
        region_2d, seed_point, path_theta=np.pi / 2
    )
    
    print(f"\n2D区域约束矩阵形状: {A_2d.shape}")
    print(f"3D区域约束矩阵形状: {region_3d.A.shape}")
    print(f"Theta范围: {region_3d.theta_range}")
    
    # 测试点包含检查
    test_point = np.array([2.5, 2.5, np.pi / 2])
    is_inside = region_3d.contains(test_point)
    print(f"\n测试点 {test_point} 在区域内: {is_inside}")
    
    test_point_outside = np.array([2.5, 2.5, 3 * np.pi / 2])
    is_inside_outside = region_3d.contains(test_point_outside)
    print(f"测试点 {test_point_outside} 在区域内: {is_inside_outside}")
    
    # 测试批量扩展
    path = [
        (2.5, 2.5, 0),
        (3.0, 3.0, np.pi / 4),
        (3.5, 3.5, np.pi / 2)
    ]
    
    regions_2d_list = [region_2d, region_2d, region_2d]
    regions_3d_list = adapter.expand_regions_from_path(regions_2d_list, path)
    
    print(f"\n批量扩展: {len(regions_2d_list)}个2D区域 -> {len(regions_3d_list)}个3D区域")
    for i, r in enumerate(regions_3d_list):
        print(f"  区域{i}: theta范围 = {r.theta_range}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_3d_region_adapter()
