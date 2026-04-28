import pydot
import numpy as np
from typing import Optional

from pydrake.geometry.optimization import (
    GraphOfConvexSets,      # Drake 中的核心类，用于构建和求解凸集图上的最短路径问题
    GraphOfConvexSetsOptions, # Drake 中用于配置 GCS 求解选项的类
    Point,                  # Drake 中表示单个点的几何对象
    HPolyhedron,            # Drake 中表示多面体的几何对象
    Hyperellipsoid,              # Drake 中表示椭球的几何对象
)
from pydrake.solvers import (
    CommonSolverOption,     # 求解器通用选项
    MathematicalProgram,    # Drake 中的数学规划问题基类 (用于 polytopeDimension)
    MosekSolver,            # Drake 中 MOSEK 求解器的接口
    SolverOptions,          # Drake 中用于配置求解器参数的类
)
from pydrake.all import le # Drake 中用于添加线性不等式约束的辅助函数

from gcs_pkg.scripts.rounding import MipPathExtraction # 自定义的路径提取策略（例如，从 MIP 解中提取）
from gcs_pkg.scripts.solver import AdaptiveSolverConfig, SolverPerformanceProfile
from gcs_pkg.scripts.core.phi_updater import IncrementalPhiUpdater
from config.solver.mosek_opt_config import MosekOptimizationConfig

def polytopeDimension(A, b, tol=1e-4):
    """
    计算由 A*x <= b 定义的多面体 P 的维度。
    通过逐步识别构成 P 边界的约束（即在边界上的点使对应约束变为等式），
    并计算这些等式约束所形成的仿射子空间的维度来实现。
    该方法利用了优化理论中的对偶变量（拉格朗日乘数）信息。
    注意：此函数专门用于 HPolyhedron (A*x <= b 形式的多面体)，不适用于 Hyperellipsoid。

    Args:
        A (numpy.ndarray): 不等式约束的系数矩阵，形状 (m, n)。
        b (numpy.ndarray): 不等式约束的右端向量，形状 (m,)。
        tol (float): 判断对偶变量是否为零的容差值，也用于判断 r 是否大于 0。

    Returns:
        int: 多面体 P 的维度。如果计算失败或 P 为空，则返回 -1。
    """
    
    assert A.shape[0] == b.size # 确保 A 的行数与 b 的长度匹配
    
    m, n = A.shape # m: 约束数量, n: 变量维度
    eq = []        # 存储当前识别为等式约束的索引列表

    while True: # 循环直到找到所有边界约束
        
        # 获取当前仍视为不等式的约束索引
        ineq = [i for i in range(m) if i not in eq]
        A_ineq = A[ineq] # 当前不等式约束的系数矩阵
        b_ineq = b[ineq] # 当前不等式约束的右端向量

        # 创建一个数学规划问题，用于寻找多面体内部的一个点或确定边界
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(n) # 创建 n 维决策变量 x
        r = prog.NewContinuousVariables(1)[0] # 创建 1 维辅助变量 r (代表到边界的距离)

        # 添加等式约束 (之前识别出的边界约束)
        if len(eq) > 0:
            A_eq = A[eq] # 边界约束的系数矩阵
            b_eq = b[eq] # 边界约束的右端向量
            prog.AddLinearEqualityConstraint(A_eq.dot(x), b_eq) # 添加 A_eq * x = b_eq
        
        # 添加当前剩余的不等式约束 (A_ineq * x + r <= b_ineq)
        # 这个约束意味着 x 必须满足所有当前不等式约束，并且至少离每个不等式边界 r 距离远
        if len(ineq) > 0:
            # le(A_ineq.dot(x) + r * np.ones(len(ineq)), b_ineq) 代表 A_ineq * x + r * 1 <= b_ineq
            # 即 A_ineq * x <= b_ineq - r
            # 如果 r > 0，说明存在一个点 x 在所有 A_ineq * x <= b_ineq 约束内部 r 距离处
            c = prog.AddLinearConstraint(le(A_ineq.dot(x) + r * np.ones(len(ineq)), b_ineq))
        
        # 约束 r 的范围为 [0, 1]，防止 r 无界增大
        prog.AddBoundingBoxConstraint(0, 1, r)
        
        # 目标函数：最大化 r (即最小化 -r)
        # 如果原多面体有内部点，则最优 r > 0
        # 如果原多面体没有内部点（例如，维度低于 n），则最优 r = 0
        prog.AddLinearCost(-r)

        # 使用 MOSEK 求解器求解此问题
        solver = MosekSolver()
        result = solver.Solve(prog)

        # 如果求解失败，说明多面体可能是空的或存在其他数值问题
        if not result.is_success():
            return -1 # 返回 -1 表示计算失败
        
        # 获取最优解中的 r 值
        optimal_r = result.GetSolution(r)

        # 如果最优 r 大于容差 tol，说明多面体内部非空，其维度等于 n 减去已知等式约束的秩
        if optimal_r > tol:
            eq_rank = 0 if len(eq) == 0 else np.linalg.matrix_rank(A[eq]) # 计算已知等式约束矩阵的秩
            return n - eq_rank # 返回多面体维度
        
        # 如果最优 r <= tol，说明多面体没有严格内部点 (r* = 0)
        # 此时，对偶变量 (c_opt) 揭示了哪些不等式约束在最优解处是活跃的（即变成了等式）
        # 对偶变量绝对值大于 tol 的约束被认为是边界约束
        c_opt = np.abs(result.GetDualSolution(c)) # 获取与不等式约束 c 相关的对偶变量（拉格朗日乘数）
        # 将当前不等式中对偶变量值较大的约束索引添加到 eq 列表中
        eq += [ineq[i] for i, ci in enumerate(c_opt) if ci > tol]
        # A_eq = A[eq] # 更新等式约束矩阵 (循环开始时重新获取)
        # b_eq = b[eq] # 更新等式约束向量 (循环开始时重新获取)

def convexSetDimension(convex_set, tol=1e-4):
    """
    计算凸集的维度，支持 HPolyhedron 和 Hyperellipsoid。

    Args:
        convex_set: 凸集对象，可以是 HPolyhedron 或 Hyperellipsoid。
        tol (float): 判断对偶变量是否为零的容差值。

    Returns:
        int: 凸集的维度。如果计算失败或集为空，则返回 -1。
    """
    # 首先检查凸集的类型是否为 HPolyhedron
    if isinstance(convex_set, HPolyhedron):
        # 对于 HPolyhedron，使用 polytopeDimension 函数
        return polytopeDimension(convex_set.A(), convex_set.b(), tol)
    elif isinstance(convex_set, Hyperellipsoid):
        # 对于 Hyperellipsoid，其维度等于环境维度
        return convex_set.ambient_dimension()
    else:
        # 对于其他类型的凸集，返回环境维度
        return convex_set.ambient_dimension()

def _estimate_intersection_dimension_heuristic(set1, set2, ambient_dim, tol=1e-4):
    """
    使用启发式方法估计两个凸集交集的维度。
    此方法特别针对 Hyperellipsoid 或混合类型的交集，因为其精确维度计算较为复杂。
    该方法通过在交集内寻找多个线性无关的方向向量来估计维度。

    Args:
        set1: 第一个凸集。
        set2: 第二个凸集。
        ambient_dim (int): 环境空间的维度。
        tol (float): 数值容差。

    Returns:
        int: 估计的交集维度。如果计算失败或交集为空，返回 -1。
    """
    # 步骤 1: 尝试找到交集内的一个点 (起始点，可以是 Chebyshev 中心)
    # 这里选择 set1 的中心作为初始猜测，因为它已在 set1 内部
    initial_guess = set1.ChebyshevCenter()
    
    # 验证 initial_guess 是否也在 set2 内部（双重保险）
    if not set2.PointInSet(initial_guess):
        # 如果中心不在交集内，尝试用优化找到一个交集内的点
        prog_find_point = MathematicalProgram()
        x = prog_find_point.NewContinuousVariables(ambient_dim)
        set1.AddPointInSetConstraints(prog_find_point, x)
        set2.AddPointInSetConstraints(prog_find_point, x)
        # 任意一个成本函数，只需找到一个可行解即可
        prog_find_point.AddLinearCost(np.zeros(ambient_dim), 0, x[0])
        
        solver = MosekSolver()
        result_find_point = solver.Solve(prog_find_point)
        
        if not result_find_point.is_success():
             print(f"Warning: Could not find a point in the intersection during heuristic estimation.")
             return -1 # 无法找到交集内点，可能交集为空或求解失败

        initial_guess = result_find_point.GetSolution(x)
        # 验证找到的点确实位于交集中
        if not (set1.PointInSet(initial_guess) and set2.PointInSet(initial_guess)):
             print(f"Warning: Found point seems not to be in the intersection during heuristic estimation.")
             return -1 # 找到的点似乎不在交集内，数值问题？


    # 步骤 2: 尝试找到交集边界上的点，以估计其大小/维度
    # 方法：沿随机方向或标准基向量方向最大化距离
    # 初始化找到的独立方向数量
    num_independent_directions_found = 0
    # 存储找到的边界点相对于 initial_guess 的偏移向量
    offset_vectors = []

    # 为了更全面地探索，可以沿着标准基向量 e_i 和 -e_i 方向搜索
    # 也可以结合随机方向
    directions_to_check = []
    # 添加标准基向量及其反向
    for i in range(ambient_dim):
        e_i = np.zeros(ambient_dim)
        e_i[i] = 1.0
        directions_to_check.append(e_i)
        directions_to_check.append(-e_i)
    
    # 可选：添加一些随机方向
    # num_random_dirs = max(0, 2 * ambient_dim - 2 * ambient_dim) # 示例：添加一些随机方向
    # for _ in range(num_random_dirs):
    #     random_dir = np.random.randn(ambient_dim)
    #     random_dir /= np.linalg.norm(random_dir)
    #     directions_to_check.append(random_dir)

    for direction in directions_to_check:
        prog_boundary = MathematicalProgram()
        x = prog_boundary.NewContinuousVariables(ambient_dim)

        # 添加点在两个集合内的约束
        set1.AddPointInSetConstraints(prog_boundary, x)
        set2.AddPointInSetConstraints(prog_boundary, x)

        # 目标：最大化点 x 在给定方向上的投影
        # maximize (x - initial_guess)^T * direction
        # 等价于 minimize -(x - initial_guess)^T * direction
        # 等价于 minimize -x^T * direction + initial_guess^T * direction
        # 由于 constant 项不影响优化，成本为 -x.T @ direction
        cost_vec = -direction
        prog_boundary.AddLinearCost(cost_vec, 0, x) # AddLinearCost(c, d, vars) adds c^T vars + d

        solver = MosekSolver()
        result_boundary = solver.Solve(prog_boundary)

        if result_boundary.is_success():
            x_boundary = result_boundary.GetSolution(x)
            # 计算偏移向量
            offset = x_boundary - initial_guess
            
            # 检查这个偏移向量是否与之前找到的向量线性无关
            # 将新向量添加到列表中
            offset_vectors.append(offset)
            
            # 构建矩阵 V，其列为 offset_vectors
            V_matrix = np.column_stack(offset_vectors) if offset_vectors else np.empty((ambient_dim, 0))
            
            # 计算矩阵 V 的秩，即为找到的独立方向数
            current_rank = np.linalg.matrix_rank(V_matrix, tol=tol)
            
            # 更新找到的独立方向数
            num_independent_directions_found = max(num_independent_directions_found, current_rank)

            # 如果已经找到了 ambient_dim 个独立方向，维度就是 ambient_dim
            if num_independent_directions_found >= ambient_dim:
                 break # 已达到最大可能维度

    # 步骤 3: 基于找到的独立方向数估计维度
    estimated_dim = num_independent_directions_found

    # 可选：对于椭球交集，如果估计维度小于 ambient_dim，可以做一个保守估计
    # 但这个启发式规则本身就不够准确，所以这里直接返回估计值
    # if estimated_dim < ambient_dim:
    #     # For two ellipsoids intersecting, if they overlap significantly, it's often ambient_dim - 1
    #     # But this is highly specific and unreliable.
    #     # We stick with the calculated estimate from independent directions.
    #     pass 

    return estimated_dim


def intersectionDimension(set1, set2, tol=1e-4):
    """
    计算两个凸集交集的维度，支持 HPolyhedron 和 Hyperellipsoid。
    注意：当交集涉及 Hyperellipsoid 时，返回的维度是基于启发式方法的估计值，
         可能不精确，尤其是在交集形状复杂或维度较低的情况下。

    Args:
        set1: 第一个凸集，可以是 HPolyhedron 或 Hyperellipsoid。
        set2: 第二个凸集，可以是 HPolyhedron 或 Hyperellipsoid。
        tol (float): 数值容差。

    Returns:
        int: 交集的维度。如果计算失败或交集为空，则返回 -1。
    """
    # 检查两个集合是否有交集
    if not set1.IntersectsWith(set2):
        return -1  # 无交集

    # 如果两个集合都是 HPolyhedron，合并它们的约束并计算维度
    if isinstance(set1, HPolyhedron) and isinstance(set2, HPolyhedron):
        A = np.vstack((set1.A(), set2.A()))
        b = np.concatenate((set1.b(), set2.b()))
        return polytopeDimension(A, b, tol)

    # 至少有一个集合是 Hyperellipsoid，使用启发式方法
    ambient_dim = set1.ambient_dimension() # Assume consistency as per BaseGCS
    return _estimate_intersection_dimension_heuristic(set1, set2, ambient_dim, tol)

# BaseGCS 类是基于图搜索的凸优化方法 (GCS) 的基础实现
class BaseGCS:
    def __init__(self, regions, auto_add_vertices=True, 
                 solver_config: Optional[AdaptiveSolverConfig] = None,
                 mosek_opt_config: Optional[MosekOptimizationConfig] = None):
        """
        初始化 GCS 图结构。

        Args:
            regions (list or dict): 包含所有凸集（例如 HPolyhedron, Hyperellipsoid, Point）的列表，
                                    或者是以名称为键、凸集为值的字典。
            auto_add_vertices (bool): 是否自动添加顶点到图中。默认为True。
            solver_config (AdaptiveSolverConfig, optional): 自适应求解器配置。默认为None。
            mosek_opt_config (MosekOptimizationConfig, optional): MOSEK优化配置。默认为None(使用默认优化值)。
        """
        self.names = None # 存储顶点的名称
        # 检查输入是否为字典，如果是，则提取名称和区域
        if type(regions) is dict:
            self.names = list(regions.keys()) # 从字典键获取名称
            regions = list(regions.values())  # 从字典值获取区域对象
        else:
            # 如果输入是列表，则自动生成名称 v0, v1, ...
            self.names = ["v" + str(ii) for ii in range(len(regions))]
        
        # 所有区域必须具有相同的环境维度
        self.dimension = regions[0].ambient_dimension()
        self.regions = regions.copy() # 存储凸集区域列表
        self.rounding_fn = []         # 存储用于舍入（从松弛解中提取整数解）的策略函数列表
        self.rounding_kwargs = {}     # 存储传递给舍入函数的额外关键字参数

        # 验证所有区域的维度是否一致
        for r in self.regions:
            assert r.ambient_dimension() == self.dimension

        # 创建 Drake 的 GraphOfConvexSets 对象实例
        self.gcs = GraphOfConvexSets()
        # 创建 GCS 求解选项对象
        self.options = GraphOfConvexSetsOptions()
        # 初始化源点和目标点顶点引用
        self.source = None
        self.target = None
        
        # 初始化自适应求解器配置
        self.solver_config = solver_config if solver_config is not None else AdaptiveSolverConfig()
        self.solver_profile: Optional[SolverPerformanceProfile] = None

        # 初始化MOSEK优化配置
        self._mosek_opt_config = mosek_opt_config or MosekOptimizationConfig()
        self._phi_updater = IncrementalPhiUpdater() if self._mosek_opt_config.enable_incremental_phi else None

        # 自动添加顶点到图中
        if auto_add_vertices:
            for ii, region in enumerate(self.regions):
                self.gcs.AddVertex(region, self.names[ii])

    def addSourceTarget(self, source, target, edges=None):
        """
        在 GCS 图中添加源点和目标点顶点，并连接它们到图中的其他顶点。

        Args:
            source (array-like): 源点坐标，长度必须等于 self.dimension。
            target (array-like): 目标点坐标，长度必须等于 self.dimension。
            edges (list of lists, optional): 指定源点和目标点分别连接到哪些区域顶点的索引列表。
                                             格式为 [[src_connected_region_indices], [tgt_connected_region_indices]]
                                             如果为 None，则自动查找包含源/目标点的区域。

        Raises:
            ValueError: 如果源点或目标点没有连接到任何区域顶点。
        """
        # 如果之前已经添加过源点或目标点，则先移除它们
        if self.source is not None or self.target is not None:
            self.gcs.RemoveVertex(self.source)
            self.gcs.RemoveVertex(self.target)

        # 验证源点和目标点的维度
        assert len(source) == self.dimension
        assert len(target) == self.dimension

        # 获取图中现有的顶点列表（对应于 regions）
        vertices = self.gcs.Vertices()

        # 添加源点和目标点作为 Point 类型的顶点
        self.source = self.gcs.AddVertex(Point(source), "source") # 添加源点顶点
        self.target = self.gcs.AddVertex(Point(target), "target") # 添加目标点顶点

        # 添加连接源点/目标点到图中顶点的边
        if edges is None:
            # 如果没有提供边的连接信息，则自动查找
            edges = self.findStartGoalEdges(source, target)

        # 检查源点和目标点是否至少连接到一个区域顶点，否则无法形成有效路径
        if not (len(edges[0]) > 0): # edges[0] 包含源点连接的区域索引
            raise ValueError('Source vertex is not connected.')
        if not (len(edges[1]) > 0): # edges[1] 包含目标点连接的区域索引
            raise ValueError('Target vertex is not connected.')

        # 存储添加的边
        source_edges = []
        target_edges = []
        # 为源点连接的每个区域顶点添加一条边 (source -> region_vertex)
        for ii in edges[0]:
            u = vertices[ii] # 获取对应的区域顶点对象
            edge = self.gcs.AddEdge(self.source, u, f"(source, {u.name()})") # 添加边
            source_edges.append(edge) # 记录边

        # 为目标点连接的每个区域顶点添加一条边 (region_vertex -> target)
        for ii in edges[1]:
            u = vertices[ii] # 获取对应的区域顶点对象
            edge = self.gcs.AddEdge(u, self.target, f"({u.name()}, target)") # 添加边
            target_edges.append(edge) # 记录边

        return source_edges, target_edges # 返回添加的源点和目标点边

    def findEdgesViaOverlaps(self):
        """
        通过检查区域之间的交集来确定 GCS 图中的边。
        如果两个凸集有交集，则在它们对应的顶点之间添加双向边。

        Returns:
            list of tuples: 包含所有应添加的边的索引对列表 [(i,j), (j,i), ...]。
        """
        edges = [] # 初始化边列表
        # 遍历所有区域对 (ii, jj)，其中 jj > ii 避免重复计算
        for ii in range(len(self.regions)):
            for jj in range(ii + 1, len(self.regions)):
                # 使用 Drake 的 IntersectsWith 方法检查两个凸集是否有交集
                if self.regions[ii].IntersectsWith(self.regions[jj]):
                    # 如果有交集，则添加双向边 (ii -> jj) 和 (jj -> ii)
                    edges.append((ii, jj))
                    edges.append((jj, ii))
        return edges # 返回边列表

    def findEdgesViaFullDimensionOverlaps(self):
        """
        通过检查区域交集的维度是否足够大（>= ambient_dimension - 1）来确定 GCS 图中的边。
        该方法现在正确支持 HPolyhedron 和 Hyperellipsoid 类型的凸集。

        Returns:
            list of tuples: 包含所有应添加的边的索引对列表 [(i,j), (j,i), ...]。
        """
        edges = []  # 初始化边列表
        # 遍历所有区域对 (ii, jj)，其中 jj > ii 避免重复计算
        for ii in range(len(self.regions)):
            for jj in range(ii + 1, len(self.regions)):
                # 使用通用的 intersectionDimension 函数计算交集维度
                dim = intersectionDimension(self.regions[ii], self.regions[jj])
                
                # 检查维度是否有效且足够大
                # dim == -1 表示无交集或计算失败，应被忽略
                # 其他负值理论上不应出现，但也按失败处理
                if dim >= self.dimension - 1:
                    # 如果维度足够大，则添加双向边 (ii -> jj) 和 (jj -> ii)
                    # 注：只有当 dim != -1 且 dim >= self.dimension - 1 时才会执行到这里
                    edges.append((ii, jj))
                    edges.append((jj, ii))
        return edges  # 返回边列表

    def findStartGoalEdges(self, start, goal):
        """
        确定源点和目标点应该连接到哪些区域顶点。
        通过检查点是否在区域内来实现。

        Args:
            start (array-like): 源点坐标。
            goal (array-like): 目标点坐标。

        Returns:
            list of lists: 格式为 [[start_connected_region_indices], [goal_connected_region_indices]]。
        """
        edges = [[], []] # 初始化源点和目标点的连接列表
        # 遍历所有区域，检查源点和目标点是否在区域内
        for ii in range(len(self.regions)):
            # 使用 Drake 的 PointInSet 方法检查点是否在凸集中
            if self.regions[ii].PointInSet(start):
                edges[0].append(ii) # 如果源点在区域内，记录区域索引
            if self.regions[ii].PointInSet(goal):
                edges[1].append(ii) # 如果目标点在区域内，记录区域索引
        return edges # 返回连接列表

    def setSolver(self, solver):
        """
        设置 GCS 求解器。

        Args:
            solver: Drake 中的求解器实例 (e.g., MosekSolver()).
        """
        self.options.solver = solver # 将求解器实例赋值给选项对象
    
    def configureSolverAdaptive(self, 
                                problem_size: str = 'auto',
                                solver_type: str = 'mosek',
                                custom_profile: Optional[SolverPerformanceProfile] = None):
        """
        自适应配置求解器
        
        Args:
            problem_size: 问题规模 ('small', 'medium', 'large', 'auto')
            solver_type: 求解器类型 ('mosek', 'gurobi', 'clp', 'scs')
            custom_profile: 自定义性能配置
        """
        # 更新求解器配置
        if custom_profile is not None:
            self.solver_config.custom_profile = custom_profile
        
        if problem_size != 'auto':
            from gcs_pkg.scripts.solver import ProblemSize
            self.solver_config.problem_size = ProblemSize(problem_size)
        
        from gcs_pkg.scripts.solver import SolverType
        self.solver_config.solver_type = SolverType(solver_type)
        
        # 应用配置
        num_vertices = len(self.gcs.Vertices())
        num_edges = len(self.gcs.Edges())
        
        self.options, self.solver_profile = self.solver_config.configure(
            num_vertices, num_edges, self.dimension
        )
        
        print(f"求解器已配置: {solver_type.upper()}, 问题规模: {problem_size}")
        if self.solver_profile:
            print(f"  松弛容差: {self.solver_profile.relaxation_tol}")
            print(f"  MIP容差: {self.solver_profile.mip_tol}")
            print(f"  最大时间: {self.solver_profile.max_time}s")
    
    def enableWarmStart(self, enable: bool = True):
        """
        启用或禁用求解器预热
        
        Args:
            enable: 是否启用预热
        """
        if self.solver_profile:
            self.solver_profile.enable_warm_start = enable
            print(f"求解器预热: {'启用' if enable else '禁用'}")

    def setSolverOptions(self, options):
        """
        设置 GCS 求解器的具体参数。

        Args:
            options (SolverOptions): Drake 中的 SolverOptions 实例。
        """
        self.options.solver_options = options # 将参数对象赋值给选项对象

    def setPaperSolverOptions(self):
        """
        设置论文中常用的特定求解器选项，通常用于实验或复现结果。
        这里设置了 MOSEK 求解器的一些参数。
        使用MosekOptimizationConfig中的优化值替代硬编码。
        """
        solver_options = SolverOptions()
        # 在控制台打印求解过程信息
        solver_options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        # 设置内点法相对间隙容差
        solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-3)
        # 设置内点法求解形式 (通常用于原始或对偶问题)
        solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_INTPNT_SOLVE_FORM", 1)
        # 设置混合整数优化 (MIO) 相对间隙容差
        solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_TOL_REL_GAP", 1e-3)
        # 设置 MIO 最大求解时间 (秒) - 使用优化配置值替代硬编码3600.0
        solver_options.SetOption(MosekSolver.id(), "MSK_DPAR_MIO_MAX_TIME",
                                 self._mosek_opt_config.effective_mio_max_time())
        # 设置 MOSEK 线程数 - 使用优化配置值
        solver_options.SetOption(MosekSolver.id(), "MSK_IPAR_NUM_THREADS",
                                 self._mosek_opt_config.effective_num_threads())
        self.options.solver_options = solver_options # 应用设置的参数

    def setRoundingStrategy(self, rounding_fn, **kwargs):
        """
        设置用于从 GCS 松弛解中提取可行路径的舍入策略。

        Args:
            rounding_fn (function or list of functions): 用于舍入的函数或函数列表。
            **kwargs: 传递给舍入函数的额外关键字参数。
        """
        self.rounding_kwargs = kwargs # 存储额外参数
        if callable(rounding_fn): # 如果传入的是单个函数
            self.rounding_fn = [rounding_fn] # 将其放入列表
        elif isinstance(rounding_fn, list): # 如果传入的是函数列表
            assert len(rounding_fn) > 0 # 确保列表不为空
            for fn in rounding_fn: # 确保列表中的每个元素都是可调用的
                assert callable(fn)
            self.rounding_fn = rounding_fn # 存储列表
        else: # 如果都不是，则抛出错误
            raise ValueError("Rounding strategy must either be "
                             "a function or list of functions.")

    def ResetGraph(self, vertices=None):
        """
        重置 GCS 图，移除指定的顶点，并清除所有边上的 Phi 约束（用于舍入）。

        Args:
            vertices (list, optional): 要移除的顶点列表。如果为 None，则默认移除 source 和 target。
        """
        if vertices is None: # 如果未指定顶点
            vertices = [self.source, self.target] # 默认移除源点和目标点
            self.source = None # 清空源点引用
            self.target = None # 清空目标点引用
        for v in vertices: # 移除指定的顶点
            self.gcs.RemoveVertex(v)
        # 清除图中所有边的 Phi 约束 (这些约束用于固定边的启用/禁用状态，通常在舍入后设置)
        for edge in self.gcs.Edges():
            edge.ClearPhiConstraints()
        # 重置增量Phi更新器状态
        if self._phi_updater is not None:
            self._phi_updater.reset()


    def VisualizeGraph(self, file_type="svg"):
        """
        生成 GCS 图的可视化表示 (SVG 或 PNG)。

        Args:
            file_type (str): 输出文件类型，'svg' 或 'png'。

        Returns:
            bytes: 图像数据。
        """
        # 获取图的 Graphviz DOT 字符串表示
        graphviz = self.gcs.GetGraphvizString(None, False)
        # 使用 pydot 解析 DOT 字符串
        data = pydot.graph_from_dot_data(graphviz)[0]
        # 根据请求的类型生成图像数据
        if file_type == "svg":
            return data.create_svg()
        elif file_type == "png":
            return data.create_png()
        else:
            raise ValueError("Unrecognized file type:", file_type)


    def _extract_active_path(self, gcs, result, source, target):
        """从求解结果中提取活跃边组成的路径。

        通过检查每条边的流值 φ，将 φ > 0.5 的边视为活跃边，
        按拓扑顺序连接成 source→target 的完整路径。

        Args:
            gcs: GraphOfConvexSets 对象
            result: 求解结果
            source: 起始顶点
            target: 目标顶点

        Returns:
            list: 活跃边组成的路径，或 None（提取失败时）
        """
        active_edges = []
        for edge in gcs.Edges():
            phi_val = result.GetSolution(edge.phi())
            if phi_val > 0.5:
                active_edges.append(edge)

        # 按拓扑顺序排列活跃边
        path = []
        current = source
        visited = set()
        while current != target:
            found_next = False
            for edge in active_edges:
                if edge.u() == current and edge not in visited:
                    path.append(edge)
                    visited.add(edge)
                    current = edge.v()
                    found_next = True
                    break
            if not found_next:
                return None
        return path


    def solveGCS(self, rounding, preprocessing, verbose):
        """
        求解 GCS 最短路径问题。

        Args:
            rounding (bool): 是否使用舍入策略。True 求解松弛问题并舍入，False 求解混合整数规划 (MIP)。
            preprocessing (bool): 是否启用 GCS 内置的预处理。
            verbose (bool): 是否打印求解过程的详细信息。

        Returns:
            tuple: (best_path, best_result, results_dict)。
                   best_path: 最优路径上的边列表。
                   best_result: Drake 的 MathematicalProgramResult 对象。
                   results_dict: 包含各种求解统计信息的字典。
        """
        # 用于存储求解过程中的各种结果和统计信息
        results_dict = {}

        # 自适应配置求解器(如果尚未配置)
        if self.solver_profile is None:
            num_vertices = len(self.gcs.Vertices())
            num_edges = len(self.gcs.Edges())
            self.options, self.solver_profile = self.solver_config.configure(
                num_vertices, num_edges, self.dimension
            )
            if verbose:
                print(f"自动配置求解器: 顶点={num_vertices}, 边={num_edges}, 维度={self.dimension}")

        # 用MosekOptimizationConfig覆盖AdaptiveSolverConfig的MOSEK关键参数
        # 确保MosekOptimizationConfig作为统一配置入口的实际生效
        if self.options.solver_options is not None:
            solver_opts = self.options.solver_options
            solver_id = MosekSolver.id()
            # 覆盖MIO时间限制
            solver_opts.SetOption(solver_id, "MSK_DPAR_MIO_MAX_TIME",
                                  self._mosek_opt_config.effective_mio_max_time())
            # 覆盖线程数
            solver_opts.SetOption(solver_id, "MSK_IPAR_NUM_THREADS",
                                  self._mosek_opt_config.effective_num_threads())

        # 打印MOSEK优化配置摘要
        if verbose:
            print(self._mosek_opt_config.summary())

        # 配置 GCS 选项
        self.options.convex_relaxation = rounding # 是否求解松弛问题
        self.options.preprocessing = preprocessing # 是否启用预处理
        self.options.max_rounded_paths = 0 # 初始设置为 0
        
        # 检查是否有缓存的解(预热)
        problem_hash = None
        if self.solver_profile and self.solver_profile.enable_warm_start and self.source and self.target:
            # 计算问题哈希
            source_pos = self.source.set().x()
            target_pos = self.target.set().x()
            problem_hash = self.solver_config.compute_problem_hash(
                self.gcs, source_pos, target_pos
            )
            
            # 尝试获取缓存的解
            cached_solution = self.solver_config.get_cached_solution(problem_hash)
            if cached_solution is not None and verbose:
                print("找到缓存的解,尝试预热...")
                # 注意: Drake的GCS不直接支持warm start,这里只是记录
                # 实际预热需要在更底层实现

        # 第一次求解：求解松弛问题或 MIP
        result = self.gcs.SolveShortestPath(self.source, self.target, self.options)

        # 根据是否使用舍入来存储第一次求解的结果
        if rounding: # 如果使用舍入
            results_dict["relaxation_result"] = result
            results_dict["relaxation_solver_time"] = result.get_solver_details().optimizer_time
            results_dict["relaxation_cost"] = result.get_optimal_cost()
        else: # 如果直接求解 MIP
            results_dict["mip_result"] = result
            results_dict["mip_solver_time"] = result.get_solver_details().optimizer_time
            results_dict["mip_cost"] = result.get_optimal_cost()

        # 检查第一次求解是否成功
        if not result.is_success():
            print("First solve failed")
            print(f"  Solution result: {result.get_solution_result()}")
            try:
                details = result.get_solver_details()
                print(f"  Solver details: {details}")
                # 尝试获取MOSEK不可行证明
                if hasattr(details, 'solution_status'):
                    print(f"  Solution status: {details.solution_status}")
                if hasattr(details, 'code'):
                    print(f"  Code: {details.code}")
                if hasattr(details, 'msg'):
                    print(f"  Message: {details.msg}")
            except Exception as e:
                print(f"  Error getting details: {e}")
            return None, None, results_dict # 返回失败结果

        # 如果要求详细输出，则打印第一次求解的信息
        if verbose:
            print("Solution\t",
                  "Success:", result.get_solution_result(),
                  "Cost:", result.get_optimal_cost(),
                  "Solver time:", result.get_solver_details().optimizer_time)

        # --- 处理舍入步骤 (如果启用了舍入) ---
        if rounding and len(self.rounding_fn) > 0: # 如果使用舍入且提供了舍入策略
            # 每次进入Rounding循环前reset phi_updater，确保从干净状态开始
            # 避免跨solveGCS调用的_prev_path_edges残留导致增量更新错误
            if self._phi_updater is not None:
                self._phi_updater.reset()

            active_edges = [] # 存储所有舍入策略找到的路径
            found_path = False # 标记是否找到了至少一条路径

            # 尝试每一个指定的舍入策略
            for fn in self.rounding_fn:
                # 调用舍入函数，传入 GCS 对象、松弛解结果、源点、目标点和额外参数
                try:
                    rounded_edges = fn(self.gcs, result, self.source, self.target,
                                       **self.rounding_kwargs)
                except Exception as e:
                    print(f"{fn.__name__} raised exception: {e}")
                    rounded_edges = None

                if rounded_edges is None or len(rounded_edges) == 0:
                    print(fn.__name__, "could not find a path.")
                    continue
                else:
                    found_path = True
                    for path in rounded_edges:
                        active_edges.append(path)

            results_dict["rounded_paths"] = active_edges # 记录所有尝试的路径

            if not found_path: # 如果所有策略都失败了
                print("All rounding strategies failed to find a path.")
                return None, None, results_dict # 返回失败结果

            # 配置第二次求解的选项（关闭预处理，因为路径已固定）
            self.options.preprocessing = False
            rounded_results = [] # 存储每次固定路径后的求解结果
            best_cost = np.inf # 初始化最佳成本
            best_path = None # 初始化最佳路径
            best_result = None # 初始化最佳结果
            max_rounded_solver_time = 0.0 # 记录最长的舍入后求解时间
            total_rounded_solver_time = 0.0 # 记录总的舍入后求解时间

            # 对每个找到的路径进行第二次精确求解
            for path_edges in active_edges:
                # 使用增量或全量Phi约束更新
                if self._phi_updater is not None:
                    # 增量Phi约束更新：仅修改与上一条路径差异的边
                    self._phi_updater.update_phi_constraints(self.gcs, path_edges)
                else:
                    # 全量Phi约束更新（回退方式）：清除所有边再重建
                    for edge in self.gcs.Edges():
                        edge.ClearPhiConstraints()
                    for edge in self.gcs.Edges():
                        if edge in path_edges:
                            edge.AddPhiConstraint(True)
                        else:
                            edge.AddPhiConstraint(False)

                # 求解固定路径后的 GCS 问题（这是一个凸优化问题）
                rounded_results.append(self.gcs.SolveShortestPath(
                    self.source, self.target, self.options))
                
                # 统计求解时间和成本
                solve_time = rounded_results[-1].get_solver_details().optimizer_time
                max_rounded_solver_time = np.maximum(solve_time, max_rounded_solver_time)
                total_rounded_solver_time += solve_time
                
                # 检查当前路径的求解结果是否是目前最好的
                if (rounded_results[-1].is_success() # 结果成功
                    and rounded_results[-1].get_optimal_cost() < best_cost): # 成本更低
                    best_cost = rounded_results[-1].get_optimal_cost() # 更新最佳成本
                    best_path = path_edges # 更新最佳路径
                    best_result = rounded_results[-1] # 更新最佳结果

            # 将舍入相关的统计信息存入结果字典
            results_dict["best_path"] = best_path
            results_dict["best_result"] = best_result
            results_dict["rounded_results"] = rounded_results
            results_dict["max_rounded_solver_time"] =  max_rounded_solver_time
            results_dict["total_rounded_solver_time"] = total_rounded_solver_time
            results_dict["rounded_cost"] = best_result.get_optimal_cost()
            # 保存所有候选路径及其求解结果，供上层按约束违反量筛选
            results_dict["all_rounded_paths_results"] = list(
                zip(active_edges, rounded_results)
            )

            # 如果要求详细输出，打印所有舍入后求解的信息
            if verbose:
                print("Rounded Solutions:")
                for r in rounded_results:
                    if r is None:
                        print("\t\tNo path to solve")
                        continue
                    print("\t\t",
                        "Success:", r.get_solution_result(),
                        "Cost:", r.get_optimal_cost(),
                        "Solver time:", r.get_solver_details().optimizer_time)

            # 检查最终是否找到了有效的最佳路径
            if best_path is None:
                print("Second solve failed on all paths.") # 所有路径的第二次求解都失败
                return best_path, best_result, results_dict

        # --- 处理内置舍入或 MIP 求解 ---
        elif rounding: # 如果使用舍入但没有提供自定义策略 (使用 Drake 内置)
            # 配置选项，允许 Drake 尝试找到最多 10 条舍入路径
            self.options.max_rounded_paths = 10
            # 执行舍入求解
            rounded_result = self.gcs.SolveShortestPath(self.source, self.target, self.options)
            # 优先从精确解中提取活跃边路径
            best_path = self._extract_active_path(
                self.gcs, rounded_result, self.source, self.target)
            if best_path is None:
                # 回退到 MipPathExtraction
                best_path = MipPathExtraction(self.gcs, rounded_result, self.source, self.target)[0]
            best_result = rounded_result
            results_dict["rounded_result"] = rounded_result
            results_dict["rounded_solver_time"] = rounded_result.get_solver_details().optimizer_time
            results_dict["rounded_cost"] = rounded_result.get_optimal_cost()
            
            # 缓存解
            if problem_hash and self.solver_profile and self.solver_profile.cache_solutions:
                self.solver_config.cache_solution(problem_hash, best_result)
                self.solver_config.record_solve_stats(
                    problem_hash,
                    rounded_result.get_solver_details().optimizer_time,
                    rounded_result.get_optimal_cost(),
                    rounded_result.is_success()
                )
        else:
            # 直接求解 MIP
            best_path = MipPathExtraction(self.gcs, result, self.source, self.target)[0]
            best_result = result
            
            # 缓存解
            if problem_hash and self.solver_profile and self.solver_profile.cache_solutions:
                self.solver_config.cache_solution(problem_hash, best_result)
                self.solver_config.record_solve_stats(
                    problem_hash,
                    result.get_solver_details().optimizer_time,
                    result.get_optimal_cost(),
                    result.is_success()
                )

        # 如果要求详细输出，打印最终选择的最佳路径
        if verbose:
            for edge in best_path:
                print("Added", edge.name(), "to path.")

        # 返回最佳路径、最佳结果和包含统计信息的字典
        return best_path, best_result, results_dict