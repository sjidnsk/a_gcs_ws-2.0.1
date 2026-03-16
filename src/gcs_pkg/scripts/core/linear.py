import numpy as np
import pydot
import time


from pydrake.geometry.optimization import (
    HPolyhedron,  # 表示多面体凸集
    Point,        # 表示单点凸集
    Hyperellipsoid,  # 表示椭球凸集
    VPolytope,  # 用于转换 Hyperellipsoid
)
from pydrake.solvers import (
    Binding,                   # 约束或成本与变量的绑定
    Constraint,                # 约束基类
    Cost,                      # 成本基类
    L2NormCost,                # L2范数成本
    LinearConstraint,          # 线性约束
)

from gcs_pkg.scripts.core import BaseGCS  # 导入GCS基础类

def sample_unit_sphere(dim, num_samples):
    """
    在 n 维单位超球面上生成均匀采样点。

    Args:
        dim (int): 空间维度 n。
        num_samples (int): 期望的采样点数量。

    Returns:
        np.ndarray: 形状为 (dim, num_samples) 的数组，每一列是一个单位超球面上的点。
    """
    # 在 n 维空间中，从标准正态分布 N(0, I) 采样
    # 然后将每个样本向量归一化到单位长度
    # 这样得到的点在单位超球面上是均匀分布的
    
    # 生成 (dim, num_samples) 的高斯随机矩阵
    gaussian_samples = np.random.normal(size=(dim, num_samples))
    
    # 计算每个样本向量的 L2 范数 (沿 dim 轴)
    norms = np.linalg.norm(gaussian_samples, axis=0, keepdims=True) # shape: (1, num_samples)
    
    # 避免除零错误（虽然概率极低）
    norms = np.where(norms == 0, 1.0, norms)
    
    # 将每个向量除以其范数，得到单位向量
    unit_vectors = gaussian_samples / norms # Broadcasting: (dim, num_samples) / (1, num_samples)
    
    return unit_vectors

class LinearGCS(BaseGCS):
    """
    线性图凸集(Graph of Convex Sets)实现，用于生成由直线段组成的路径规划。
    
    该类是最简单的GCS实现，每个边表示两个凸集区域之间的直线段，
    适用于不需要平滑轨迹的简单路径规划问题。
    """

    def __init__(self, regions, edges=None, path_weights=None, full_dim_overlap=False, hyperellipsoid_num_samples_per_dim_factor=32):
        """
        初始化线性GCS
        
        Args:
            regions (list or dict): 凸集列表或字典，每个凸集表示一个可行区域
            edges (list, optional): 指定的边连接关系，默认为None时自动计算
            path_weights (array or float, optional): 路径成本权重，可为标量或向量
            full_dim_overlap (bool): 是否要求区域交集维度为全维度
        """
        # 调用基类初始化，不自动添加顶点
        BaseGCS.__init__(self, regions, auto_add_vertices=False)
        # 存储采样点数量参数
        self.hyperellipsoid_num_samples_per_dim_factor = hyperellipsoid_num_samples_per_dim_factor

        # 处理路径权重参数
        if path_weights is None:
            # 默认权重：各维度权重均为1
            path_weights = np.ones(self.dimension)
        elif isinstance(path_weights, float) or isinstance(path_weights, int):
            # 如果提供标量权重，则扩展为各维度相同的向量
            path_weights = path_weights * np.ones(self.dimension)
        # 确保权重向量维度与问题维度一致
        assert len(path_weights) == self.dimension

        # 创建边的成本函数：L2范数成本
        # 成本函数形式：||W*(x_v - x_u)||，其中W是对角权重矩阵
        # np.hstack((np.diag(-path_weights), np.diag(path_weights))) 构建了系数矩阵
        # 使得成本 = ||W*x_v - W*x_u|| = ||W*(x_v - x_u)||
        self.edge_cost = L2NormCost(
            np.hstack((np.diag(-path_weights), np.diag(path_weights))),
            np.zeros(self.dimension))
        
        for i, r in enumerate(self.regions):
            # 检查区域类型
            if isinstance(r, Hyperellipsoid):
                 print(f"Warning: Converting Hyperellipsoid at index {i} to HPolyhedron approximation for LinearGCS vertex.")
                 
                 # --- 新增逻辑：n 维采样 ---  
                 dimension = r.A().shape[0] # 获取椭球所在的空间维度
                 num_samples = self.hyperellipsoid_num_samples_per_dim_factor * dimension # 计算总采样点数

                 # 使用新的采样函数生成单位超球面上的点
                 unit_sphere_points = sample_unit_sphere(dimension, num_samples) # shape: (dimension, num_samples)

                 # 获取椭球参数
                 A_inv_T = np.linalg.inv(r.A()).T  # 椭球形状矩阵的逆转置
                 center = r.center()               # 椭球中心

                 # 将单位超球面上的点转换到椭球表面上
                 # x_ellipse = A^{-T} * x_unit_sphere + center
                 ellipse_points = A_inv_T @ unit_sphere_points + center.reshape(-1, 1)
                 # --- 新增逻辑结束 ---

                 # 创建VPolytope
                 v_poly = VPolytope(ellipse_points)
                 # 转换为HPolyhedron
                 # 注意：VPolytope -> HPolyhedron 的转换在高维空间可能非常耗时且不稳定
                 try:
                     vertex_set = HPolyhedron(v_poly, tol=1e-8) # 调整容差
                     print(f"  Successfully converted Hyperellipsoid ({dimension}D) to HPolyhedron with {num_samples} vertices.")
                 except Exception as e:
                     print(f"  Error converting Hyperellipsoid ({dimension}D) to HPolyhedron: {e}")
                     # 根据需要决定如何处理失败的情况，例如重新采样或使用 Drake 的默认转换
                     # 这里我们暂时抛出异常，让调用者知道失败了
                     raise RuntimeError(f"Failed to convert Hyperellipsoid at index {i} to HPolyhedron after sampling.") from e

            elif isinstance(r, HPolyhedron):
                 # 如果已经是 HPolyhedron，直接使用
                 vertex_set = r
            # elif isinstance(r, Point):
            #      # 如果是 Point，也尝试转换为 HPolyhedron
            #      # Point 在 Drake 中可以转换为 VPolytope，然后 VPolytope 可以转换为 HPolyhedron
            #      # 虽然这会产生一个退化的多面体，但逻辑上是可行的。
            #      print(f"Warning: Converting Point at index {i} to HPolyhedron approximation for LinearGCS vertex.")
            #      vertex_set = r.ToVPolytope().MakeHPolyhedron()
            else:
                 # 如果是其他未知类型，抛出错误或警告
                 raise TypeError(f"Region type {type(r)} at index {i} is not supported by LinearGCS. "
                                 f"Supported types are HPolyhedron, Hyperellipsoid, and Point.")

            self.gcs.AddVertex(vertex_set, name = self.names[i] if not self.names is None else '')

        # # 为每个区域添加顶点到GCS图中
        # for i, r in enumerate(self.regions):
        #     self.gcs.AddVertex(r, name = self.names[i] if not self.names is None else '')

        # 确定边的连接关系
        if edges is None:
            if full_dim_overlap:
                # 使用全维度交集条件确定边
                edges = self.findEdgesViaFullDimensionOverlaps()
            else:
                # 使用简单交集条件确定边
                edges = self.findEdgesViaOverlaps()

        # 获取图中所有顶点
        vertices = self.gcs.Vertices()
        # 为每对连接的区域添加边
        for ii, jj in edges:
            u = vertices[ii]  # 起始顶点
            v = vertices[jj]  # 目标顶点
            # 添加边到图中，命名格式为"(u_name, v_name)"
            edge = self.gcs.AddEdge(u, v, f"({u.name()}, {v.name()})")

            # 为边添加成本：路径长度成本
            edge.AddCost(Binding[Cost](
                self.edge_cost, np.append(u.x(), v.x())))

            # 添加约束：v中的点必须在u的凸集中
            # 即：A_u * x_v <= b_u，确保从u到v的直线段完全在u区域内
            edge.AddConstraint(Binding[Constraint](
                LinearConstraint(u.set().A(),
                                 -np.inf*np.ones(len(u.set().b())),
                                 u.set().b()),
                v.x()))

    def addSourceTarget(self, source, target, edges=None):
        """
        添加源点和目标点，并设置边界条件
        
        Args:
            source (array): 源点位置，维度必须等于self.dimension
            target (array): 目标点位置，维度必须等于self.dimension
            edges (list, optional): 指定的边连接关系
            
        Returns:
            tuple: (源点边列表, 目标点边列表)
        """
        # 调用父类方法添加源点和目标点
        source_edges, target_edges = super().addSourceTarget(source, target, edges)

        # 为源点边添加约束：源点与连接的第一个区域顶点重合
        for edge in source_edges:
            for jj in range(self.dimension):
                # 约束：source[jj] == first_vertex[jj]
                edge.AddConstraint(edge.xu()[jj] == edge.xv()[jj])

        # 为目标点边添加成本：与普通边相同的路径长度成本
        for edge in target_edges:
            edge.AddCost(Binding[Cost](
                self.edge_cost, np.append(edge.xu(), edge.xv())))

        return source_edges, target_edges

    def SolvePath(self, rounding=False, verbose=False, preprocessing=False):
        """
        求解最优路径并返回航路点
        
        Args:
            rounding (bool): 是否使用舍入策略（先求解松弛问题）
            verbose (bool): 是否显示详细求解信息
            preprocessing (bool): 是否进行预处理
            
        Returns:
            tuple: (航路点数组, 结果字典)
                航路点数组 shape=(dimension, num_waypoints)
                结果字典包含求解过程的各种信息
        """
        # 调用基类方法求解GCS
        best_path, best_result, results_dict = self.solveGCS(
            rounding, preprocessing, verbose)

        # 如果没有找到可行路径，返回None
        if best_path is None:
            return None, results_dict

        # 提取轨迹的航路点
        # 初始化空的航路点数组，维度为(dimension, 0)
        waypoints = np.empty((self.dimension, 0))
        
        # 遍历最优路径中的每条边
        for edge in best_path:
            # 获取边的终点（v）的解（即航路点）
            new_waypoint = best_result.GetSolution(edge.xv())
            # 将新航路点添加到数组中
            waypoints = np.concatenate(
                [waypoints, np.expand_dims(new_waypoint, 1)], axis=1)

        # 返回航路点数组和结果字典
        return waypoints, results_dict