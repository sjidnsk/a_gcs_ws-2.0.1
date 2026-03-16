"""
IrisNp 2D区域到4D配置空间扩展模块（增强版）

将IrisNp生成的2D凸区域 (x, y) 扩展到4D配置空间 (x, y, u, w)，
其中 (u, w) = (cos(θ), sin(θ)) 是theta的单位向量表示。

核心改进：
1. 使用单位向量替代theta，解决非凸性问题
2. 支持SOCP松弛约束（u² + w² ≤ 1）
3. 改进的theta连续性处理
4. 完善的周期性处理

关键优势：
- 将非凸的运动学约束转化为凸约束
- GCS优化问题保持凸性
- 更好的数值稳定性

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import warnings

# 导入单位向量处理器
from ..theta.theta_unit_vector_handler import (
        ThetaUnitVectorHandler,
        UnitVectorConfig,
        theta_to_unit_vector,
        unit_vector_to_theta
    )


# 导入混合约束策略
from ..theta.hybrid_theta_constraint import (
        HybridThetaConstraintStrategy,
        HybridConstraintConfig
    )


# Drake导入
try:
    from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
    from pydrake.solvers import LorentzConeConstraint
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    warnings.warn("Drake未安装，部分功能不可用")


@dataclass
class ThetaRangeConfigEnhanced:
    """
    增强的Theta范围配置
    
    支持两种表示方式：
    1. 传统theta表示：theta ∈ [theta_min, theta_max]
    2. 单位向量表示：(u, w) ∈ unit circle sector
    3. 混合约束策略：SOCP + 扇形约束
    """
    # Theta范围（传统表示）
    theta_min: float = 0.0
    theta_max: float = 2 * np.pi
    
    # 是否使用单位向量表示
    use_unit_vector: bool = True
    
    # 是否使用SOCP松弛
    use_socp_relaxation: bool = True
    
    # 是否使用混合约束策略（新增）
    use_hybrid_constraints: bool = True
    
    # 混合约束配置（新增）
    use_sector_constraints: bool = True       # 是否使用扇形约束
    sector_approximation_sides: int = 8       # 扇形近似边数
    auto_normalize_multi_period: bool = True  # 自动归一化多周期
    
    # Theta连续性约束
    enforce_continuity: bool = True
    max_theta_jump: float = np.pi / 4  # 相邻区域间最大角度跳变
    
    # 是否使用路径信息约束theta
    use_path_constraint: bool = True
    path_theta_margin: float = np.pi / 6  # 路径theta的扩展裕度
    
    # 周期性处理
    allow_wrap_around: bool = True
    normalize_theta: bool = True


@dataclass
class IrisNpRegion4D:
    """
    4D凸区域 (x, y, u, w)
    
    使用单位向量表示theta，解决非凸性问题
    """
    # 多面体表示 (Ax <= b)，维度为4
    A: np.ndarray  # 约束矩阵 (M x 4)
    b: np.ndarray  # 约束向量 (M,)
    
    # 几何信息
    vertices_2d: np.ndarray  # 2D顶点坐标 (N x 2)
    centroid_2d: np.ndarray  # 2D中心点 (2,)
    theta_range: Tuple[float, float]  # theta范围 [theta_min, theta_max]
    
    # 单位向量范围
    u_range: Tuple[float, float]  # u范围 [u_min, u_max]
    w_range: Tuple[float, float]  # w范围 [w_min, w_max]
    
    # 元数据
    seed_point: np.ndarray  # 原始种子点 (x, y, theta)
    seed_point_uv: np.ndarray  # 单位向量种子点 (x, y, u, w)
    
    # SOCP约束（可选）
    socp_constraint: Optional['LorentzConeConstraint'] = None
    
    region_id: int = 0
    
    # 原始2D区域信息
    original_A: Optional[np.ndarray] = None  # 原始2D约束矩阵
    original_b: Optional[np.ndarray] = None  # 原始2D约束向量
    
    # 配置
    use_unit_vector: bool = True
    use_socp: bool = True
    
    def contains(self, point: np.ndarray, tol: float = 1e-6) -> bool:
        """
        检查点是否在区域内
        
        Args:
            point: 可以是3D点 (x, y, theta) 或4D点 (x, y, u, w)
            tol: 容差
        
        Returns:
            True如果点在区域内
        """
        # 根据维度判断输入类型
        if len(point) == 3:
            # 3D输入 (x, y, theta)，转换为4D
            x, y, theta = point
            u, w = theta_to_unit_vector(theta)
            point_4d = np.array([x, y, u, w])
        elif len(point) == 4:
            # 4D输入 (x, y, u, w)
            point_4d = point
        else:
            raise ValueError(f"点维度必须为3或4，当前为{len(point)}")
        
        # 检查多面体约束
        if not np.all(self.A @ point_4d <= self.b + tol):
            return False
        
        # 检查SOCP约束（如果有）
        if self.use_socp and self.socp_constraint is not None:
            # SOCP约束：u² + w² ≤ 1
            u, w = point_4d[2], point_4d[3]
            if u**2 + w**2 > 1.0 + tol:
                return False
        
        return True
    
    def to_hpolyhedron(self):
        """
        转换为Drake HPolyhedron对象
        
        注意：SOCP约束无法直接用HPolyhedron表示，
        需要在GCS优化中单独处理
        
        Returns:
            HPolyhedron对象（仅包含线性约束）
        """
        if not DRAKE_AVAILABLE:
            raise RuntimeError("Drake未安装，无法创建HPolyhedron")
        
        return HPolyhedron(self.A, self.b)
    
    def get_vertices_4d(self, num_theta_samples: int = 10) -> np.ndarray:
        """
        获取4D顶点（用于可视化）
        
        Args:
            num_theta_samples: theta方向的采样点数
        
        Returns:
            4D顶点数组 (K x 4)
        """
        vertices_4d = []
        theta_min, theta_max = self.theta_range
        thetas = np.linspace(theta_min, theta_max, num_theta_samples)
        
        for theta in thetas:
            u, w = theta_to_unit_vector(theta)
            for vertex_2d in self.vertices_2d:
                vertices_4d.append([vertex_2d[0], vertex_2d[1], u, w])
        
        return np.array(vertices_4d)
    
    def get_3d_projection(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取3D投影 (x, y, theta)
        
        用于可视化和与传统方法兼容
        
        Returns:
            (A_3d, b_3d): 3D约束矩阵和向量
        """
        # 从4D约束中提取3D约束
        # 这是一个近似，因为单位向量约束是非线性的
        
        # 简化：使用theta范围约束
        theta_min, theta_max = self.theta_range
        
        # 构建3D约束
        M = self.original_A.shape[0] if self.original_A is not None else 0
        
        # 2D约束
        if self.original_A is not None:
            A_3d_xy = np.hstack([self.original_A, np.zeros((M, 1))])
            b_3d_xy = self.original_b
        else:
            A_3d_xy = np.zeros((0, 3))
            b_3d_xy = np.zeros(0)
        
        # theta约束
        A_theta = np.array([
            [0, 0, -1],  # -theta <= -theta_min
            [0, 0, 1]    # theta <= theta_max
        ])
        b_theta = np.array([-theta_min, theta_max])
        
        # 合并
        A_3d = np.vstack([A_3d_xy, A_theta])
        b_3d = np.concatenate([b_3d_xy, b_theta])
        
        return A_3d, b_3d


class IrisRegion4DAdapter:
    """
    2D区域到4D区域的适配器（增强版）
    
    使用单位向量表示theta，解决非凸性问题
    """
    
    def __init__(self, config: Optional[ThetaRangeConfigEnhanced] = None):
        """
        初始化适配器
        
        Args:
            config: Theta范围配置
        """
        self.config = config or ThetaRangeConfigEnhanced()
        self.theta_handler = ThetaUnitVectorHandler(
            UnitVectorConfig(
                use_socp_relaxation=self.config.use_socp_relaxation,
                max_theta_jump=self.config.max_theta_jump
            )
        )
        
        # 初始化混合约束策略（新增）
        if self.config.use_hybrid_constraints:
            self.hybrid_strategy = HybridThetaConstraintStrategy(
                HybridConstraintConfig(
                    use_socp=self.config.use_socp_relaxation,
                    use_sector_constraints=self.config.use_sector_constraints,
                    sector_approximation_sides=self.config.sector_approximation_sides,
                    auto_normalize_multi_period=self.config.auto_normalize_multi_period,
                    max_theta_jump=self.config.max_theta_jump,
                    allow_wrap_around=self.config.allow_wrap_around
                )
            )
        else:
            self.hybrid_strategy = None
    
    def expand_region_to_4d(
        self,
        region_2d: Any,
        seed_point: np.ndarray,
        path_theta: Optional[float] = None,
        neighbor_thetas: Optional[List[float]] = None
    ) -> IrisNpRegion4D:
        """
        将2D区域扩展为4D区域（使用单位向量）
        
        Args:
            region_2d: 2D区域对象（IrisNpRegion或HPolyhedron）
            seed_point: 种子点 (x, y, theta)
            path_theta: 路径在该点的theta值
            neighbor_thetas: 相邻区域的theta值列表
        
        Returns:
            4D区域对象
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
            vertices_2d = self._compute_vertices_from_hpoly(region_2d)
            centroid_2d = vertices_2d.mean(axis=0)
        else:
            raise TypeError(f"不支持的区域类型: {type(region_2d)}")
        
        # 确定theta范围
        theta_range = self._determine_theta_range_enhanced(
            seed_point, path_theta, neighbor_thetas
        )
        
        # 构建4D约束矩阵（使用单位向量）
        A_4d, b_4d, socp_constraint = self._build_4d_constraints_with_unit_vector(
            A_2d, b_2d, theta_range
        )
        
        # 计算单位向量范围
        theta_min, theta_max = theta_range
        u_min, w_min = theta_to_unit_vector(theta_min)
        u_max, w_max = theta_to_unit_vector(theta_max)
        
        # 扩展范围以包含整个扇形
        u_range = (min(u_min, u_max), max(u_min, u_max))
        w_range = (min(w_min, w_max), max(w_min, w_max))
        
        # 创建单位向量种子点
        x, y, theta = seed_point
        u, w = theta_to_unit_vector(theta)
        seed_point_uv = np.array([x, y, u, w])
        
        # 创建4D区域
        region_4d = IrisNpRegion4D(
            A=A_4d,
            b=b_4d,
            socp_constraint=socp_constraint,
            vertices_2d=vertices_2d,
            centroid_2d=centroid_2d,
            theta_range=theta_range,
            u_range=u_range,
            w_range=w_range,
            seed_point=seed_point,
            seed_point_uv=seed_point_uv,
            original_A=A_2d,
            original_b=b_2d,
            use_unit_vector=self.config.use_unit_vector,
            use_socp=self.config.use_socp_relaxation
        )
        
        return region_4d
    
    def expand_region_to_4d_with_hybrid(
        self,
        region_2d: Any,
        seed_point: np.ndarray,
        theta_min: float,
        theta_max: float
    ) -> IrisNpRegion4D:
        """
        使用混合约束策略将2D区域扩展为4D区域
        
        Args:
            region_2d: 2D区域对象
            seed_point: 种子点 (x, y, theta)
            theta_min: 最小theta值
            theta_max: 最大theta值
        
        Returns:
            4D区域对象
        """
        # 提取2D区域信息
        if hasattr(region_2d, 'A') and hasattr(region_2d, 'b'):
            A_2d = region_2d.A
            b_2d = region_2d.b
            vertices_2d = region_2d.vertices
            centroid_2d = region_2d.centroid
        elif hasattr(region_2d, 'A') and hasattr(region_2d, 'b'):
            A_2d = region_2d.A()
            b_2d = region_2d.b()
            vertices_2d = self._compute_vertices_from_hpoly(region_2d)
            centroid_2d = vertices_2d.mean(axis=0)
        else:
            raise TypeError(f"不支持的区域类型: {type(region_2d)}")
        
        # 使用混合约束策略
        if self.hybrid_strategy is not None:
            A_4d, b_4d, socp_constraint = self.hybrid_strategy.create_hybrid_constraints_for_region(
                A_2d, b_2d, theta_min, theta_max
            )
        else:
            # 回退到传统方法
            theta_range = (theta_min, theta_max)
            A_4d, b_4d, socp_constraint = self._build_4d_constraints_with_unit_vector(
                A_2d, b_2d, theta_range
            )
        
        # 计算单位向量范围
        u_min, w_min = theta_to_unit_vector(theta_min)
        u_max, w_max = theta_to_unit_vector(theta_max)
        u_range = (min(u_min, u_max), max(u_min, u_max))
        w_range = (min(w_min, w_max), max(w_min, w_max))
        
        # 创建单位向量种子点
        x, y, theta = seed_point
        u, w = theta_to_unit_vector(theta)
        seed_point_uv = np.array([x, y, u, w])
        
        # 创建4D区域
        region_4d = IrisNpRegion4D(
            A=A_4d,
            b=b_4d,
            socp_constraint=socp_constraint,
            vertices_2d=vertices_2d,
            centroid_2d=centroid_2d,
            theta_range=(theta_min, theta_max),
            u_range=u_range,
            w_range=w_range,
            seed_point=seed_point,
            seed_point_uv=seed_point_uv,
            original_A=A_2d,
            original_b=b_2d,
            use_unit_vector=self.config.use_unit_vector,
            use_socp=self.config.use_socp_relaxation
        )
        
        return region_4d
    
    def expand_regions_from_path(
        self,
        regions_2d: List[Any],
        path: List[Tuple[float, float, float]]
    ) -> List[IrisNpRegion4D]:
        """
        从路径批量扩展2D区域到4D
        
        Args:
            regions_2d: 2D区域列表
            path: 路径点列表 [(x, y, theta), ...]
        
        Returns:
            4D区域列表
        """
        regions_4d = []
        
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
            region_4d = self.expand_region_to_4d(
                region_2d, seed_point, path_theta, neighbor_thetas
            )
            region_4d.region_id = i
            regions_4d.append(region_4d)
        
        return regions_4d
    
    def _determine_theta_range_enhanced(
        self,
        seed_point: np.ndarray,
        path_theta: Optional[float],
        neighbor_thetas: Optional[List[float]]
    ) -> Tuple[float, float]:
        """
        确定theta范围（增强版）

        改进：
        1. 取消theta范围限制，直接覆盖整个圆周
        2. 确保所有区域在theta维度上完全连通
        3. 避免因theta范围限制导致的GCS不连通问题

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
    
    def _build_4d_constraints_with_unit_vector(
        self,
        A_2d: np.ndarray,
        b_2d: np.ndarray,
        theta_range: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray, Optional['LorentzConeConstraint']]:
        """
        构建4D约束矩阵（使用单位向量）

        将2D约束 (A_2d * [x, y]^T <= b_2d) 扩展为4D约束
        并添加单位向量约束

        改进：取消theta的线性约束，只保留SOCP约束（单位圆约束）
        这样可以避免theta范围覆盖整个圆周时约束过强的问题

        Args:
            A_2d: 2D约束矩阵 (M x 2)
            b_2d: 2D约束向量 (M,)
            theta_range: theta范围（已覆盖整个圆周，不再添加线性约束）

        Returns:
            (A_4d, b_4d, socp_constraint): 4D约束矩阵、向量和SOCP约束
        """
        M = A_2d.shape[0]

        # 扩展2D约束到4D
        # 原始：A_2d * [x, y]^T <= b_2d
        # 扩展：[A_2d, 0, 0] * [x, y, u, w]^T <= b_2d
        A_4d = np.hstack([A_2d, np.zeros((M, 2))])
        b_4d = b_2d

        # 不再添加theta的线性约束
        # 因为theta_range已经覆盖整个圆周（-π, π），
        # 添加线性约束会导致区域过窄，无法包含起点/终点
        # 只依赖SOCP约束（单位圆约束）即可

        # 创建SOCP约束
        socp_constraint = None
        if self.config.use_socp_relaxation and DRAKE_AVAILABLE:
            socp_constraint = self.theta_handler.create_socp_constraint_for_unit_vector()

        return A_4d, b_4d, socp_constraint
    
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
            return np.array([])


def convert_iris_regions_to_4d(
    iris_result: Any,
    path: List[Tuple[float, float, float]],
    config: Optional[ThetaRangeConfigEnhanced] = None
) -> List[IrisNpRegion4D]:
    """
    将IrisNp结果转换为4D区域列表（使用单位向量）
    
    便捷函数，用于将IrisNp生成的2D区域批量转换为4D区域
    
    Args:
        iris_result: IrisNp结果对象
        path: 路径点列表 [(x, y, theta), ...]
        config: Theta配置
    
    Returns:
        4D区域列表
    """
    adapter = IrisRegion4DAdapter(config)
    
    # 提取2D区域
    if hasattr(iris_result, 'regions'):
        regions_2d = iris_result.regions
    else:
        raise TypeError("iris_result必须包含regions属性")
    
    # 扩展为4D
    regions_4d = adapter.expand_regions_from_path(regions_2d, path)
    
    return regions_4d


def create_hpolyhedron_list_from_4d_regions(
    regions_4d: List[IrisNpRegion4D]
) -> List['HPolyhedron']:
    """
    从4D区域列表创建HPolyhedron列表
    
    注意：SOCP约束需要单独处理
    
    Args:
        regions_4d: 4D区域列表
    
    Returns:
        HPolyhedron列表（仅包含线性约束）
    """
    if not DRAKE_AVAILABLE:
        raise RuntimeError("Drake未安装，无法创建HPolyhedron")
    
    hpoly_list = []
    for region in regions_4d:
        try:
            hpoly = region.to_hpolyhedron()
            hpoly_list.append(hpoly)
        except Exception as e:
            warnings.warn(f"区域{region.region_id}转换失败: {e}")
    
    return hpoly_list


# ==================== 测试代码 ====================

def test_4d_region_adapter():
    """测试4D区域适配器"""
    print("=" * 60)
    print("测试4D区域适配器（单位向量表示）")
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
    config = ThetaRangeConfigEnhanced(
        use_unit_vector=True,
        use_socp_relaxation=True,
        use_path_constraint=True,
        path_theta_margin=np.pi / 4
    )
    adapter = IrisRegion4DAdapter(config)
    
    # 测试单个区域扩展
    seed_point = np.array([2.5, 2.5, np.pi / 2])
    region_4d = adapter.expand_region_to_4d(
        region_2d, seed_point, path_theta=np.pi / 2
    )
    
    print(f"\n2D区域约束矩阵形状: {A_2d.shape}")
    print(f"4D区域约束矩阵形状: {region_4d.A.shape}")
    print(f"Theta范围: {region_4d.theta_range}")
    print(f"单位向量范围: u={region_4d.u_range}, w={region_4d.w_range}")
    print(f"使用SOCP约束: {region_4d.socp_constraint is not None}")
    
    # 测试点包含检查
    test_point_3d = np.array([2.5, 2.5, np.pi / 2])
    is_inside_3d = region_4d.contains(test_point_3d)
    print(f"\n测试点(3D) {test_point_3d} 在区域内: {is_inside_3d}")
    
    u, w = theta_to_unit_vector(np.pi / 2)
    test_point_4d = np.array([2.5, 2.5, u, w])
    is_inside_4d = region_4d.contains(test_point_4d)
    print(f"测试点(4D) {test_point_4d} 在区域内: {is_inside_4d}")
    
    # 测试批量扩展
    path = [
        (2.5, 2.5, 0),
        (3.0, 3.0, np.pi / 4),
        (3.5, 3.5, np.pi / 2)
    ]
    
    regions_2d_list = [region_2d, region_2d, region_2d]
    regions_4d_list = adapter.expand_regions_from_path(regions_2d_list, path)
    
    print(f"\n批量扩展: {len(regions_2d_list)}个2D区域 -> {len(regions_4d_list)}个4D区域")
    for i, r in enumerate(regions_4d_list):
        print(f"  区域{i}: theta范围={r.theta_range}, SOCP={r.socp_constraint is not None}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_4d_region_adapter()
