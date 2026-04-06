"""
区域渲染模块
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from typing import List, Optional, Tuple
from pydrake.geometry.optimization import HPolyhedron


# === 数值容差常量 ===
# 用于数值计算中的容差判断

CONSTRAINT_CONTAINMENT_TOL: float = 1e-6  # 点在区域内判断的容差
NUMERICAL_TOLERANCE: float = 1e-10  # 通用数值计算容差，用于避免除零


class RegionRenderer:
    """IRIS区域渲染器
    
    负责绘制HPolyhedron区域
    """
    
    def __init__(self, colors: Optional[List[str]] = None):
        """初始化渲染器
        
        Args:
            colors: 颜色列表，用于区分不同区域
        """
        self.colors = colors or [
            'lightblue', 'lightgreen', 'lightyellow', 'lightpink',
            'lightcoral', 'lightsalmon', 'lightseagreen', 'lightskyblue',
            'lavender', 'thistle', 'paleturquoise', 'palegreen'
        ]
    
    def plot_regions(
        self,
        ax: plt.Axes,
        regions: List[HPolyhedron],
        alpha: float = 0.2,
        show_labels: bool = True,
        label_prefix: str = "R"
    ) -> None:
        """绘制多个IRIS区域
        
        Args:
            ax: matplotlib坐标轴
            regions: HPolyhedron区域列表
            alpha: 透明度
            show_labels: 是否显示区域标签
            label_prefix: 标签前缀
        """
        for i, region in enumerate(regions):
            color = self.colors[i % len(self.colors)]
            label = f"{label_prefix}{i}" if show_labels else None
            
            self.plot_single_region(
                ax, region, color=color, alpha=alpha,
                label=label if i == 0 else None
            )
    
    def plot_single_region(
        self,
        ax: plt.Axes,
        region: HPolyhedron,
        color: str = 'lightblue',
        alpha: float = 0.2,
        label: Optional[str] = None
    ) -> None:
        """绘制单个IRIS区域
        
        Args:
            ax: matplotlib坐标轴
            region: HPolyhedron区域
            color: 颜色
            alpha: 透明度
            label: 标签
        """
        # 尝试获取顶点
        vertices = self._get_vertices(region)
        
        if vertices is not None and len(vertices) > 0:
            # 使用顶点绘制多边形
            polygon = MplPolygon(
                vertices,
                closed=True,
                facecolor=color,
                edgecolor='blue',
                alpha=alpha,
                linewidth=1,
                label=label
            )
            ax.add_patch(polygon)
        else:
            # 回退到采样方法
            self._plot_region_by_sampling(ax, region, color, alpha, label)
    
    def _get_vertices(self, region: HPolyhedron) -> Optional[np.ndarray]:
        """获取区域顶点
        
        尝试多种方法获取顶点，并按正确顺序排列
        
        Args:
            region: HPolyhedron区域
            
        Returns:
            Optional[np.ndarray]: 顶点数组 (N, 2) 或 None
        """
        # 方法1：转换为VPolytope
        try:
            from pydrake.geometry.optimization import VPolytope
            vpoly = VPolytope(region)
            vertices = vpoly.vertices()
            if vertices.shape[1] > 0:
                # vertices是(2, N)格式，需要转置为(N, 2)
                vertices_T = vertices.T
                
                # 按角度排序顶点（相对于中心点）
                vertices_ordered = self._order_vertices(vertices_T)
                
                return vertices_ordered
        except Exception as e:
            pass
        
        # 方法2：边界采样
        return self._sample_boundary(region)
    
    def _order_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """按角度排序顶点
        
        Args:
            vertices: 顶点数组 (N, 2)
            
        Returns:
            np.ndarray: 排序后的顶点数组 (N, 2)
        """
        if len(vertices) < 3:
            return vertices
        
        # 计算中心点
        center = np.mean(vertices, axis=0)
        
        # 计算每个顶点相对于中心的角度
        angles = np.arctan2(vertices[:, 1] - center[1], 
                           vertices[:, 0] - center[0])
        
        # 按角度排序
        sorted_indices = np.argsort(angles)
        vertices_ordered = vertices[sorted_indices]
        
        return vertices_ordered
    
    def _sample_boundary(
        self,
        region: HPolyhedron,
        num_samples: int = 100
    ) -> Optional[np.ndarray]:
        """采样区域边界
        
        Args:
            region: HPolyhedron区域
            num_samples: 采样点数
            
        Returns:
            Optional[np.ndarray]: 边界点数组 (N, 2) 或 None
        """
        try:
            A = region.A()
            b = region.b()
            
            # 估计区域范围
            center = self._estimate_center(A, b)
            if center is None:
                return None
            
            # 在每个约束边界上采样
            boundary_points = []
            for i in range(A.shape[0]):
                # 沿着第i个约束的边界采样
                for t in np.linspace(0, 2*np.pi, num_samples // A.shape[0]):
                    # 生成边界点
                    direction = np.array([np.cos(t), np.sin(t)])
                    
                    # 求解与边界的交点
                    point = self._find_boundary_intersection(
                        center, direction, A, b, i
                    )
                    
                    if point is not None:
                        # 检查点是否在区域内
                        if np.all(A @ point <= b + CONSTRAINT_CONTAINMENT_TOL):
                            boundary_points.append(point)
            
            if len(boundary_points) > 0:
                return np.array(boundary_points)
            
        except:
            pass
        
        return None
    
    def _estimate_center(
        self,
        A: np.ndarray,
        b: np.ndarray
    ) -> Optional[np.ndarray]:
        """估计区域中心
        
        Args:
            A: 约束矩阵
            b: 约束向量
            
        Returns:
            Optional[np.ndarray]: 中心点或None
        """
        try:
            # 使用Chebyshev中心
            from scipy.optimize import linprog
            
            # 构造优化问题
            c = np.zeros(3)
            c[2] = -1  # 最大化r
            
            A_ub = np.zeros((A.shape[0], 3))
            A_ub[:, :2] = A
            A_ub[:, 2] = np.linalg.norm(A, axis=1)
            
            b_ub = b
            
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
            
            if result.success:
                return result.x[:2]
        except:
            pass
        
        # 回退到原点
        return np.zeros(2)
    
    def _find_boundary_intersection(
        self,
        center: np.ndarray,
        direction: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        constraint_idx: int
    ) -> Optional[np.ndarray]:
        """找到与指定约束边界的交点
        
        Args:
            center: 起点
            direction: 方向
            A: 约束矩阵
            b: 约束向量
            constraint_idx: 约束索引
            
        Returns:
            Optional[np.ndarray]: 交点或None
        """
        try:
            i = constraint_idx
            a_i = A[i, :]
            b_i = b[i]
            
            # 求解: a_i^T (center + t * direction) = b_i
            denom = np.dot(a_i, direction)
            
            if abs(denom) < NUMERICAL_TOLERANCE:
                return None
            
            t = (b_i - np.dot(a_i, center)) / denom
            
            if t < 0:
                return None
            
            return center + t * direction
            
        except:
            return None
    
    def _plot_region_by_sampling(
        self,
        ax: plt.Axes,
        region: HPolyhedron,
        color: str,
        alpha: float,
        label: Optional[str]
    ) -> None:
        """通过采样绘制区域
        
        Args:
            ax: matplotlib坐标轴
            region: HPolyhedron区域
            color: 颜色
            alpha: 透明度
            label: 标签
        """
        try:
            A = region.A()
            b = region.b()
            
            # 估计范围
            center = self._estimate_center(A, b)
            if center is None:
                center = np.zeros(2)
            
            # 创建网格
            range_val = 20.0
            x_range = np.linspace(center[0] - range_val, center[0] + range_val, 100)
            y_range = np.linspace(center[1] - range_val, center[1] + range_val, 100)
            X, Y = np.meshgrid(x_range, y_range)
            
            # 检查每个点是否在区域内
            inside = np.ones(X.shape, dtype=bool)
            for i in range(A.shape[0]):
                inside &= (A[i, 0] * X + A[i, 1] * Y <= b[i])
            
            # 绘制等高线
            ax.contourf(X, Y, inside, levels=[0.5, 1.5], colors=[color], alpha=alpha)
            
        except:
            pass
    
    def compute_region_center(
        self,
        region: HPolyhedron
    ) -> Optional[np.ndarray]:
        """计算区域中心
        
        Args:
            region: HPolyhedron区域
            
        Returns:
            Optional[np.ndarray]: 中心点或None
        """
        vertices = self._get_vertices(region)
        
        if vertices is not None and len(vertices) > 0:
            return np.mean(vertices, axis=0)
        
        return self._estimate_center(region.A(), region.b())
