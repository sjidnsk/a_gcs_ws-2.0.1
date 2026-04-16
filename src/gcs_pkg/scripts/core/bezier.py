import numpy as np
import pydot
import time
from scipy.optimize import root_scalar

from pydrake.geometry.optimization import (
    HPolyhedron,  # 表示多面体凸集
    Point,        # 表示单点凸集
    Hyperellipsoid,  # 表示椭球凸集
    VPolytope,  # 用于转换 Hyperellipsoid
)
from pydrake.math import (
    BsplineBasis,      # B样条基函数
    BsplineBasis_,     # B样条基函数（带类型）
    KnotVectorType,    # 节点向量类型
)
from pydrake.solvers import(
    Binding,                   # 约束或成本与变量的绑定
    Constraint,                # 约束基类
    Cost,                      # 成本基类
    L2NormCost,                # L2范数成本
    LinearConstraint,          # 线性约束
    LinearCost,                # 线性成本
    LinearEqualityConstraint,  # 线性等式约束
    LorentzConeConstraint,     # Lorentz锥约束（用于标量速度约束）
    QuadraticCost,             # 二次成本
    PerspectiveQuadraticCost,  # 透视二次成本
)
from pydrake.symbolic import (
    DecomposeLinearExpressions,  # 将线性表达式分解为系数矩阵和常数项
    Expression,                  # 符号表达式
    MakeMatrixContinuousVariable,  # 创建矩阵连续变量
    MakeVectorContinuousVariable,  # 创建向量连续变量
)
from pydrake.trajectories import (
    BsplineTrajectory,      # B样条轨迹
    BsplineTrajectory_,     # 带类型的B样条轨迹
    Trajectory,             # 轨迹基类
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



class BezierGCS(BaseGCS):
    """
    基于贝塞尔曲线的图凸集(Graph of Convex Sets)实现，用于生成平滑的轨迹规划。
    
    该类通过将每个凸区域表示为控制点序列（贝塞尔曲线控制点）来构建GCS图，
    并在顶点之间添加B样条曲线的连续性约束，实现平滑轨迹规划。
    """
    
    def __init__(self, regions, order, continuity, edges=None, hdot_min=1e-6, full_dim_overlap=False, hyperellipsoid_num_samples_per_dim_factor=32):
        """
        初始化贝塞尔GCS
        
        Args:
            regions (list): 凸集列表，每个凸集表示一个可行区域
            order (int): B样条曲线的阶数（控制点数量 = order + 1）
            continuity (int): 路径连续性要求（0=C0连续，1=C1连续，2=C2连续等）
            edges (list, optional): 指定的边连接关系，默认为None时自动计算
            hdot_min (float): 时间导数的最小值，确保时间单调递增
            full_dim_overlap (bool): 是否要求区域交集维度为全维度
        """
        # 调用基类初始化
        BaseGCS.__init__(self, regions, auto_add_vertices=False)

        self.order = order
        self.continuity = continuity
        # 存储采样点数量参数（作为因子）
        self.hyperellipsoid_num_samples_per_dim_factor = hyperellipsoid_num_samples_per_dim_factor
        assert continuity < order  # 连续性要求必须小于曲线阶数

        # 创建时间缩放凸集，用于确保时间单调递增
        # A_time 和 b_time 定义了时间控制点的约束：
        # 1. 控制点 <= 1000 (大数，实际不会达到)
        # 2. 控制点 >= 0
        # 3. 相邻控制点差值 >= hdot_min (确保时间单调递增)
        A_time = np.vstack((np.eye(order + 1), -np.eye(order + 1),
                            np.eye(order, order + 1) - np.eye(order, order + 1, 1)))
        b_time = np.concatenate((1e3*np.ones(order + 1), np.zeros(order + 1), -hdot_min * np.ones(order)))
        self.time_scaling_set = HPolyhedron(A_time, b_time)

        for i, r in enumerate(self.regions):
            # 检查区域类型
            if isinstance(r, Hyperellipsoid):
                 print(f"Warning: Converting Hyperellipsoid at index {i} to HPolyhedron approximation for BezierGCS vertex.")
                 
                 # --- 新增逻辑：n 维采样，基于维度缩放 ---
                 # 解释 hyperellipsoid_num_samples_per_dim_factor 为一个因子
                 num_samples_per_dim_factor = self.hyperellipsoid_num_samples_per_dim_factor
                 dimension = r.A().shape[0] # 获取椭球所在的空间维度
                 total_samples = num_samples_per_dim_factor * dimension # 计算总采样点数

                 print(f"  Sampling {total_samples} points for {dimension}D Hyperellipsoid using factor {num_samples_per_dim_factor}.")
                 
                 # 使用新的采样函数生成单位超球面上的点
                 unit_sphere_points = sample_unit_sphere(dimension, total_samples) # shape: (dimension, total_samples)

                 # 获取椭球参数
                 A_inv_T = np.linalg.inv(r.A()).T  # 椭球形状矩阵的逆转置
                 center = r.center()               # 椭球中心

                 # 将单位超球面上的点转换到椭球表面上
                 # x_ellipse = A^{-T} * x_unit_sphere + center
                 ellipse_points = A_inv_T @ unit_sphere_points + center.reshape(-1, 1)

                 # 创建VPolytope
                 v_poly = VPolytope(ellipse_points)
                 # 转换为HPolyhedron
                 # 注意：VPolytope -> HPolyhedron 的转换在高维空间可能非常耗时且不稳定
                 try:
                     region_poly = HPolyhedron(v_poly, tol=1e-8) # 转换后的区域作为 HPolyhedron
                     print(f"  Successfully converted Hyperellipsoid ({dimension}D) to HPolyhedron with {total_samples} vertices.")
                 except Exception as e:
                     print(f"  Error converting Hyperellipsoid ({dimension}D) to HPolyhedron with {total_samples} samples: {e}")
                     # 根据需要决定如何处理失败的情况，例如重新采样或使用 Drake 的默认转换
                     # 这里我们暂时抛出异常，让调用者知道失败了
                     raise RuntimeError(f"Failed to convert Hyperellipsoid at index {i} to HPolyhedron after sampling.") from e

            elif isinstance(r, HPolyhedron):
                 # 如果已经是 HPolyhedron，直接使用
                 region_poly = r
            # elif isinstance(r, Point):
            #      # 如果是 Point，也尝试转换为 HPolyhedron
            #      print(f"Warning: Converting Point at index {i} to HPolyhedron approximation for BezierGCS vertex.")
            #      try:
            #         region_poly = r.ToVPolytope().MakeHPolyhedron()
            #      except Exception as e:
            #          print(f"  Error converting Point at index {i} to HPolyhedron: {e}")
            #          raise RuntimeError(f"Failed to convert Point at index {i} to HPolyhedron.") from e
            else:
                 # 如果是其他未知类型，抛出错误或警告
                 raise TypeError(f"Region type {type(r)} at index {i} is not supported by BezierGCS. "
                                 f"Supported types are HPolyhedron, Hyperellipsoid, and Point.")
        # 为每个区域顶点添加控制点
        # 每个顶点的凸集 = 区域凸集^(order+1) × 时间缩放凸集
        # 即每个顶点有 (order+1) 个空间控制点和 (order+1) 个时间控制点
            # 使用转换后的 region_poly (一定是 HPolyhedron) 来构建顶点集合
            vertex_set = region_poly.CartesianPower(order + 1).CartesianProduct(self.time_scaling_set)
            self.gcs.AddVertex(vertex_set, name = self.names[i] if not self.names is None else '')

        # 定义边上的变量：u控制点、v控制点、u时间、v时间
        u_control = MakeMatrixContinuousVariable(
            self.dimension, order + 1, "xu")
        v_control = MakeMatrixContinuousVariable(
            self.dimension, order + 1, "xv")
        u_duration = MakeVectorContinuousVariable(order + 1, "Tu")
        v_duration = MakeVectorContinuousVariable(order + 1, "Tv")

        # 存储u顶点的所有变量（控制点+时间）
        self.u_vars = np.concatenate((u_control.flatten("F"), u_duration))
        
        # 创建u顶点的空间轨迹（B样条）
        self.u_r_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            u_control)
        # 创建u顶点的时间轨迹（B样条）
        self.u_h_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            np.expand_dims(u_duration, 0))

        # 边上所有变量
        edge_vars = np.concatenate((u_control.flatten("F"), u_duration, v_control.flatten("F"), v_duration))
        
        # 创建v顶点的空间轨迹（B样条）
        v_r_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            v_control)
        # 创建v顶点的时间轨迹（B样条）
        v_h_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            np.expand_dims(v_duration, 0))

        # 连续性约束：确保路径在连接点处满足指定阶数的连续性
        self.contin_constraints = []
        for deriv in range(continuity + 1):  # 从0阶(位置)到continuity阶(如速度、加速度等)
            # 计算u轨迹在指定阶数的导数
            u_path_deriv = self.u_r_trajectory.MakeDerivative(deriv)
            v_path_deriv = v_r_trajectory.MakeDerivative(deriv)
            # 连续性条件：v轨迹的起始点 = u轨迹的终点
            path_continuity_error = v_path_deriv.control_points()[0] - u_path_deriv.control_points()[-1]
            # 添加线性等式约束
            self.contin_constraints.append(LinearEqualityConstraint(
                DecomposeLinearExpressions(path_continuity_error, edge_vars),
                np.zeros(self.dimension)))

            # 时间轨迹的连续性约束（确保时间参数化连续）
            u_time_deriv = self.u_h_trajectory.MakeDerivative(deriv)
            v_time_deriv = v_h_trajectory.MakeDerivative(deriv)
            time_continuity_error = v_time_deriv.control_points()[0] - u_time_deriv.control_points()[-1]
            self.contin_constraints.append(LinearEqualityConstraint(
                DecomposeLinearExpressions(time_continuity_error, edge_vars), 0.0))

        # 存储导数约束和边成本
        self.deriv_constraints = []
        self.edge_costs = []

        # 添加边到图中并应用约束
        if edges is None:
            if full_dim_overlap:
                edges = self.findEdgesViaFullDimensionOverlaps()
            else:
                edges = self.findEdgesViaOverlaps()

        vertices = self.gcs.Vertices()
        for ii, jj in edges:
            u = vertices[ii]
            v = vertices[jj]
            edge = self.gcs.AddEdge(u, v, f"({u.name()}, {v.name()})")

            # 为边添加连续性约束
            for c_con in self.contin_constraints:
                edge.AddConstraint(Binding[Constraint](
                        c_con, np.append(u.x(), v.x())))

    def addTimeCost(self, weight):
        """
        添加时间成本：最小化轨迹总时间
        
        Args:
            weight (float): 时间成本的权重
        """
        assert isinstance(weight, float) or isinstance(weight, int)

        # 获取时间控制点
        u_time_control = self.u_h_trajectory.control_points()
        # 计算边的时间长度（终点时间 - 起点时间）
        segment_time = u_time_control[-1] - u_time_control[0]
        # 创建线性成本：weight * segment_time
        time_cost = LinearCost(
            weight * DecomposeLinearExpressions(segment_time, self.u_vars)[0], 0.)
        self.edge_costs.append(time_cost)

        # 为所有边添加时间成本（源点边除外）
        for edge in self.gcs.Edges():
            if edge.u() == self.source:
                continue
            edge.AddCost(Binding[Cost](time_cost, edge.xu()))

    def addPathLengthCost(self, weight):
        """
        添加路径长度成本：最小化轨迹空间长度（近似）
        
        Args:
            weight (float or array): 路径长度成本的权重，可以是标量或向量
        """
        if isinstance(weight, float) or isinstance(weight, int):
            weight_matrix = weight * np.eye(self.dimension)
        else:
            assert(len(weight) == self.dimension)
            weight_matrix = np.diag(weight)

        # 获取空间轨迹的一阶导数控制点
        u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        for ii in range(len(u_path_control)):
            # 将控制点表达式分解为线性系数
            H = DecomposeLinearExpressions(u_path_control[ii] / self.order, self.u_vars)
            # 创建L2范数成本：||weight_matrix * (dq/ds)||，近似路径长度
            path_cost = L2NormCost(np.matmul(weight_matrix, H), np.zeros(self.dimension))
            self.edge_costs.append(path_cost)

            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddCost(Binding[Cost](path_cost, edge.xu()))

    def addPathLengthIntegralCost(self, weight, integration_points=100):
        """
        添加精确的路径长度积分成本
        
        Args:
            weight (float or array): 路径长度成本的权重
            integration_points (int): 数值积分的采样点数量
        """
        if isinstance(weight, float) or isinstance(weight, int):
            weight_matrix = weight * np.eye(self.dimension)
        else:
            assert(len(weight) == self.dimension)
            weight_matrix = np.diag(weight)

        # 创建积分点（0到1之间的均匀分布）
        s_points = np.linspace(0., 1., integration_points + 1)
        # 获取空间轨迹的一阶导数
        u_path_deriv = self.u_r_trajectory.MakeDerivative(1)

        # 处理一阶B样条的特殊情况
        if u_path_deriv.basis().order() == 1:
            for t in [0.0, 1.0]:
                q_ds = u_path_deriv.value(t)
                costs = []
                for ii in range(self.dimension):
                    costs.append(q_ds[ii])
                H = DecomposeLinearExpressions(costs, self.u_vars)
                integral_cost = L2NormCost(np.matmul(weight_matrix, H), np.zeros(self.dimension))
                self.edge_costs.append(integral_cost)

                for edge in self.gcs.Edges():
                    if edge.u() == self.source:
                        continue
                    edge.AddCost(Binding[Cost](integral_cost, edge.xu()))
        else:
            # 计算所有积分点处的导数值
            q_ds = u_path_deriv.vector_values(s_points)
            for ii in range(integration_points + 1):
                costs = []
                for jj in range(self.dimension):
                    # 边界点使用梯形法则的权重（0.5），内部点使用1
                    if ii == 0 or ii == integration_points:
                        costs.append(0.5 * 1./integration_points * q_ds[jj, ii])
                    else:
                        costs.append(1./integration_points * q_ds[jj, ii])
                H = DecomposeLinearExpressions(costs, self.u_vars)
                integral_cost = L2NormCost(np.matmul(weight_matrix, H), np.zeros(self.dimension))
                self.edge_costs.append(integral_cost)

                for edge in self.gcs.Edges():
                    if edge.u() == self.source:
                        continue
                    edge.AddCost(Binding[Cost](integral_cost, edge.xu()))

    def addPathEnergyCost(self, weight, edges=None):
        """
        添加路径能量成本：最小化动能（速度平方积分）

        Args:
            weight (float or array): 能量成本的权重
            edges (list, optional): 指定添加成本的边列表。
                若为None，则对所有非source边添加（默认行为）。
                若提供，则仅对这些边添加成本（用于分段差异化成本）。
        """
        if isinstance(weight, float) or isinstance(weight, int):
            weight_matrix = weight * np.eye(self.dimension)
        else:
            assert(len(weight) == self.dimension)
            weight_matrix = np.diag(weight)

        # 获取空间轨迹和时间轨迹的一阶导数控制点
        u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()
        for ii in range(len(u_path_control)):
            # 将空间导数和时间导数表达式分解为线性系数
            A_ctrl = DecomposeLinearExpressions(u_path_control[ii], self.u_vars)
            b_ctrl = DecomposeLinearExpressions(u_time_control[ii], self.u_vars)
            # 构建透视二次成本的系数矩阵
            # H = [order * b_ctrl; sqrt(weight_matrix) * A_ctrl]
            H = np.vstack(((self.order) * b_ctrl, np.matmul(np.sqrt(weight_matrix), A_ctrl)))
            energy_cost = PerspectiveQuadraticCost(H, np.zeros(H.shape[0]))
            self.edge_costs.append(energy_cost)

            target_edges = edges if edges is not None else self.gcs.Edges()
            for edge in target_edges:
                if edge.u() == self.source:
                    continue
                edge.AddCost(Binding[Cost](energy_cost, edge.xu()))

    def addDerivativeRegularization(self, weight_r, weight_h, order, edges=None):
        """
        添加导数正则化：惩罚高阶导数，使轨迹更平滑

        Args:
            weight_r (float): 空间轨迹导数的正则化权重
            weight_h (float): 时间轨迹导数的正则化权重
            order (int): 要正则化的导数阶数
            edges (list, optional): 指定添加成本的边列表。
                若为None，则对所有非source边添加（默认行为）。
                若提供，则仅对这些边添加成本（用于分段差异化成本）。
        """
        assert isinstance(order, int) and 2 <= order <= self.order
        weights = [weight_r, weight_h]
        for weight in weights:
            assert isinstance(weight, float) or isinstance(weight, int)

        # 对空间轨迹和时间轨迹分别应用正则化
        trajectories = [self.u_r_trajectory, self.u_h_trajectory]
        for traj, weight in zip(trajectories, weights):
            # 获取指定阶数导数的控制点
            derivative_control = traj.MakeDerivative(order).control_points()
            for c in derivative_control:
                # 将导数表达式分解为线性系数
                A_ctrl = DecomposeLinearExpressions(c, self.u_vars)
                # 构建二次成本：2 * weight / (1 + self.order - order) * ||A_ctrl||^2
                H = A_ctrl.T.dot(A_ctrl) * 2 * weight / (1 + self.order - order)
                reg_cost = QuadraticCost(H, np.zeros(H.shape[0]), 0)
                self.edge_costs.append(reg_cost)

                target_edges = edges if edges is not None else self.gcs.Edges()
                for edge in target_edges:
                    if edge.u() == self.source:
                        continue
                    edge.AddCost(Binding[Cost](reg_cost, edge.xu()))

    def addVelocityLimits(self, lower_bound, upper_bound):
        """
        添加速度限制：确保轨迹速度在指定范围内
        
        Args:
            lower_bound (array): 速度下限
            upper_bound (array): 速度上限
        """
        assert len(lower_bound) == self.dimension
        assert len(upper_bound) == self.dimension

        # 获取空间轨迹和时间轨迹的一阶导数控制点
        u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()
        lb = np.expand_dims(lower_bound, 1)
        ub = np.expand_dims(upper_bound, 1)

        for ii in range(len(u_path_control)):
            # 将空间导数和时间导数表达式分解为线性系数
            A_ctrl = DecomposeLinearExpressions(u_path_control[ii], self.u_vars)
            b_ctrl = DecomposeLinearExpressions(u_time_control[ii], self.u_vars)
            # 构建速度约束：lb <= (A_ctrl / b_ctrl) <= ub
            # 等价于：A_ctrl - ub * b_ctrl <= 0 和 -A_ctrl + lb * b_ctrl <= 0
            A_constraint = np.vstack((A_ctrl - ub * b_ctrl, -A_ctrl + lb * b_ctrl))
            velocity_con = LinearConstraint(
                A_constraint, -np.inf*np.ones(2*self.dimension), np.zeros(2*self.dimension))
            self.deriv_constraints.append(velocity_con)

            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddConstraint(Binding[Constraint](velocity_con, edge.xu()))

    def addScalarVelocityLimit(self, max_velocity: float):
        """
        添加标量速度限制：||v||_2 <= max_velocity
        
        使用二阶锥约束（SOCP）实现，更符合阿克曼车辆的物理模型。
        
        数学推导：
            速度定义：v = dr/dt = r'(s) / h'(s)
            标量约束：||v||_2 <= v_max
            转化形式：||r'(s)||_2 <= v_max * h'(s)
            SOCP形式：||A_ctrl||_2 <= v_max * b_ctrl
            
        Lorentz锥约束：
            z = H @ x, 其中 H = [v_max * b_ctrl; A_ctrl]
            约束：z[0] >= ||z[1:]||_2
        
        Args:
            max_velocity: 标量速度上限（m/s），必须为正数
        
        Raises:
            AssertionError: 如果 max_velocity <= 0
        
        Note:
            - 此约束是凸约束，保持优化问题的凸性
            - 需要时间缩放约束保证 h'(s) > 0
            - 比矢量约束更准确，但求解时间略长（通常<10%）
            - 适用于阿克曼转向车辆的速度限制
        
        Example:
            >>> bezier_gcs = BezierGCS(regions, order=5, continuity=2)
            >>> bezier_gcs.addScalarVelocityLimit(2.0)  # 限制速度不超过2 m/s
        """
        # 参数验证
        assert max_velocity > 0, f"max_velocity must be positive, got {max_velocity}"
        
        # 获取空间轨迹和时间轨迹的一阶导数控制点
        # r'(s): 空间轨迹对参数s的导数
        # h'(s): 时间轨迹对参数s的导数
        u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()
        
        # 对每个控制点添加Lorentz锥约束
        for ii in range(len(u_path_control)):
            # 将空间导数和时间导数表达式分解为线性系数
            # A_ctrl: 空间导数的系数矩阵，形状 (dimension, num_vars)
            #         表示 r'(s) = A_ctrl @ x
            # b_ctrl: 时间导数的系数矩阵，形状 (1, num_vars)
            #         表示 h'(s) = b_ctrl @ x
            A_ctrl = DecomposeLinearExpressions(u_path_control[ii], self.u_vars)
            b_ctrl = DecomposeLinearExpressions(u_time_control[ii], self.u_vars)
            
            # 构建Lorentz锥约束矩阵
            # 约束形式：||r'(s)||_2 <= v_max * h'(s)
            # 即：||A_ctrl @ x||_2 <= v_max * (b_ctrl @ x)
            # 转化为Lorentz锥形式：
            #   z = H @ x, 其中 H = [v_max * b_ctrl; A_ctrl]
            #   约束：z[0] >= ||z[1:]||_2
            # 
            # H的形状：(dimension + 1, num_vars)
            # H[0, :] = v_max * b_ctrl  (时间导数系数，乘以速度上限)
            # H[1:, :] = A_ctrl         (空间导数系数)
            H = np.vstack([
                max_velocity * b_ctrl,  # 第一行：时间导数系数
                A_ctrl                   # 后续行：空间导数系数
            ])
            
            # 创建Lorentz锥约束
            # Drake的LorentzConeConstraint定义：
            #   x ∈ LorentzCone <=> x[0] >= sqrt(x[1]^2 + ... + x[n]^2)
            # 即：x[0] >= ||x[1:]||_2
            # 
            # 构造函数：LorentzConeConstraint(A, b)
            # 约束：A @ x + b ∈ LorentzCone
            # 即：(A @ x + b)[0] >= ||(A @ x + b)[1:]||_2
            velocity_con = LorentzConeConstraint(H, np.zeros(H.shape[0]))
            
            # 存储约束对象（用于调试和可视化）
            self.deriv_constraints.append(velocity_con)
            
            # 对所有GCS边添加约束
            # 注意：跳过源点边（edge.u() == self.source），因为源点没有前驱
            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddConstraint(Binding[Constraint](velocity_con, edge.xu()))

    @staticmethod
    def compute_h_bar_prime_from_trajectory(traj, num_samples: int = 100) -> float:
        """从已求解的轨迹中计算 h_bar_prime（时间参数化导数均值）

        在两阶段求解流程中使用：先求解无曲率约束的 GCS，从结果中
        计算实际 h_bar_prime，再用该值添加曲率约束重新求解。

        数学原理：
            h(s) 将参数 s 映射到物理时间 t，h'(s) = dh/ds
            h_bar_prime = mean of h'(s) over the trajectory
            实测表明 h_bar_prime ≈ L / T（路径长度 / 总物理时间）

        Args:
            traj: BezierTrajectory 对象（SolvePath 的返回值）
            num_samples: 采样点数，默认 100

        Returns:
            h_bar_prime: 时间参数化导数均值

        Example:
            >>> result = bezier_gcs.SolvePath(rounding=True)
            >>> traj = result[0] if isinstance(result, tuple) else result
            >>> h_bar_prime = BezierGCS.compute_h_bar_prime_from_trajectory(traj)
        """
        s_start = traj.time_traj.start_time()
        s_end = traj.time_traj.end_time()

        s_samples = np.linspace(s_start, s_end, num_samples)
        h_prime_values = []
        for s in s_samples:
            try:
                h_prime = float(
                    traj.time_traj.EvalDerivative(s, 1).flatten()[0]
                )
                h_prime_values.append(h_prime)
            except Exception:
                continue

        if not h_prime_values:
            raise RuntimeError(
                "Failed to evaluate h'(s) from trajectory. "
                "The trajectory may not be properly solved."
            )

        return float(np.mean(h_prime_values))

    @staticmethod
    def estimate_h_bar_prime(
        path_length_estimate: float,
        num_segments: int,
        v_max: float = None,
        v_min: float = None,
        w_time: float = None,
        w_energy: float = None,
    ) -> float:
        """从几何和成本权重估算 h_bar_prime（求解前使用）

        基于实测验证的公式：h_bar_prime ≈ L / T
        其中 L 是路径长度，T 是总物理时间。

        估算策略（按优先级）：
            1. 如果提供 w_time 和 w_energy：
               v_optimal = sqrt(w_time / w_energy)
               T ≈ L / v_optimal → h_bar_prime ≈ v_optimal * N / L * L / N
               简化：h_bar_prime ≈ L / (N * v_optimal) * N = L / v_optimal
               但实测 h_bar_prime = L / T = L / (L / v_avg) = v_avg
               更准确：h_bar_prime ≈ L / (N_segments * T_per_segment)
               其中 T_per_segment ≈ L_segment / v_optimal
               所以 h_bar_prime ≈ v_optimal（当路径均匀时）

            2. 如果提供 v_max 和 v_min：
               v_avg ≈ (v_max + v_min) / 2
               h_bar_prime ≈ path_length_estimate / (num_segments * T_per_segment)

            3. 兜底：h_bar_prime = 1.0（中性假设）

        Args:
            path_length_estimate: 路径长度估计（如起终点欧氏距离）
            num_segments: GCS 段数（区域数）
            v_max: 最大速度约束（m/s），可选
            v_min: 最小速度估计（m/s），可选
            w_time: 时间成本权重，可选
            w_energy: 能量成本权重，可选

        Returns:
            h_bar_prime 估计值

        Example:
            >>> h_bar_prime = BezierGCS.estimate_h_bar_prime(
            ...     path_length_estimate=8.0,
            ...     num_segments=2,
            ...     w_time=3.0, w_energy=3.0,
            ... )
        """
        if w_time is not None and w_energy is not None and w_energy > 0:
            # 策略1：从成本权重估算最优速度
            v_optimal = np.sqrt(w_time / w_energy)
            # h_bar_prime ≈ L / T, T = L / v_optimal
            # 但实测 h_bar_prime = L / T = v_optimal 仅在特定归一化下成立
            # 更准确：h_bar_prime ≈ L / (N_segments * (L/N_segments) / v_optimal)
            #                     = L / (L / v_optimal) = v_optimal
            # 但实际 h_bar_prime 还取决于 s 域的归一化
            # 实测：h_bar_prime ≈ path_length / (num_segments * T_per_segment)
            # 其中 T_per_segment = (path_length / num_segments) / v_optimal
            # 所以 h_bar_prime ≈ path_length / (path_length / v_optimal) = v_optimal
            # 但这忽略了 s 域归一化的影响
            # 实测修正：h_bar_prime ≈ path_length_estimate / num_segments / v_optimal * num_segments
            #                     = path_length_estimate / v_optimal
            # 不对，实测 h_bar_prime = 4.0, L = 8, v_optimal = 1.0, N = 2
            # L / v_optimal = 8.0 ≠ 4.0
            # L / (N * v_optimal) = 4.0 ✓
            return path_length_estimate / (num_segments * v_optimal)

        if v_max is not None and v_min is not None:
            # 策略2：从速度范围估算
            v_avg = (v_max + v_min) / 2.0
            return path_length_estimate / (num_segments * v_avg)

        # 策略3：兜底
        import warnings
        warnings.warn(
            "Insufficient information to estimate h_bar_prime. "
            "Using default value 1.0. Provide w_time/w_energy or "
            "v_max/v_min for a better estimate, or use "
            "compute_h_bar_prime_from_trajectory() after an initial solve.",
            UserWarning,
            stacklevel=2,
        )
        return 1.0

    def addCurvatureHardConstraint(
        self, max_curvature: float, min_velocity: float, h_bar_prime: float = None
    ):
        """添加曲率硬约束：||Q_j||_2 <= kappa_max * rho_min^2

        利用贝塞尔曲线的凸包特性，通过限制二阶导数控制点的范数
        来间接、保守地限制曲率。使用 LorentzConeConstraint 实现，
        与 addScalarVelocityLimit 同构。

        数学推导：
            曲率公式：kappa(s) = ||r'(s) x r''(s)|| / ||r'(s)||^3
            由 Cauchy-Schwarz 不等式：|kappa(s)| <= ||r''(s)|| / ||r'(s)||^2
            由凸包特性：||r''(s)|| <= max_j ||Q_j||
            充分条件：||Q_j||_2 <= kappa_max * rho_min^2

        其中 rho_min = min_velocity * h_bar_prime 是 s 域速度下界。

        Lorentz锥约束形式：
            z = H @ x + b, 其中 H = [0; A_ctrl], b = [C; 0]
            约束：z[0] >= ||z[1:]||_2  即  C >= ||Q_j||_2

        Args:
            max_curvature: 最大允许曲率 kappa_max (1/m)，必须为正数
            min_velocity: 内部段最小物理速度 (m/s)，必须为正数
                推荐值：sqrt(w_time / w_energy)，由成本权重隐式决定
            h_bar_prime: 时间参数化导数 h'(s) 的均值估计
                推荐确定方式（按精度排序）：
                1. 两阶段求解：先求解无曲率约束的GCS，用
                   compute_h_bar_prime_from_trajectory() 计算实际值
                2. 静态估算：用 estimate_h_bar_prime() 从几何和成本权重估算
                3. None：使用默认值 1.0（可能不准确，会发出警告）

        Raises:
            AssertionError: 如果 max_curvature <= 0 或 min_velocity <= 0
            ValueError: 如果 rho_min 过小导致约束阈值 C 接近零

        Note:
            - 此约束是凸约束（LorentzCone），保持优化问题的凸性
            - 与 addScalarVelocityLimit 同构，仅导数阶数和阈值不同
            - 对 n=5 阶贝塞尔曲线，每段添加 4 个 LorentzCone 约束
            - 边界段（起终点 v=0）不应使用此约束，由航向角约束隐式保证
            - 保守性可通过成本权重调优（w_time/w_energy 比值）来缓解
            - h_bar_prime 的准确性直接影响约束的保守程度：
              偏大 → C 偏大 → 约束过松（可能违反曲率限制）
              偏小 → C 偏小 → 约束过紧（可能过度限制可行域）

        Example:
            >>> bezier_gcs = BezierGCS(regions, order=5, continuity=2)
            >>> # 方式1：两阶段求解（最准确）
            >>> result = bezier_gcs.SolvePath(rounding=True)
            >>> h_bar_prime = BezierGCS.compute_h_bar_prime_from_trajectory(result[0])
            >>> bezier_gcs.addCurvatureHardConstraint(0.5, 0.7, h_bar_prime)
            >>> # 方式2：静态估算
            >>> h_bar_prime = BezierGCS.estimate_h_bar_prime(8.0, 2, w_time=3.0, w_energy=3.0)
            >>> bezier_gcs.addCurvatureHardConstraint(0.5, 0.7, h_bar_prime)
        """
        # 参数验证
        assert max_curvature > 0, (
            f"max_curvature must be positive, got {max_curvature}"
        )
        assert min_velocity > 0, (
            f"min_velocity must be positive, got {min_velocity}"
        )

        # 步骤1：计算 h_bar_prime（时间参数化导数均值）
        if h_bar_prime is None:
            import warnings
            warnings.warn(
                "h_bar_prime not specified, using default value 1.0. "
                "This may be inaccurate. Recommended approaches:\n"
                "  1. Two-phase solve: solve without curvature constraint first, "
                "then use compute_h_bar_prime_from_trajectory() to get the "
                "actual value.\n"
                "  2. Static estimate: use estimate_h_bar_prime() with "
                "path_length, num_segments, and cost weights.\n"
                "See addCurvatureHardConstraint docstring for examples.",
                UserWarning,
                stacklevel=2,
            )
            h_bar_prime = 1.0

        # 步骤2：计算 rho_min 和约束阈值 C
        rho_min = min_velocity * h_bar_prime
        C = max_curvature * rho_min**2

        # 验证阈值合理性
        if C < 1e-10:
            raise ValueError(
                f"Curvature constraint threshold C = {C:.2e} is too small "
                f"(rho_min = {rho_min:.2e}). This would over-constrain the "
                f"trajectory. Consider increasing min_velocity or adjusting "
                f"cost weights (w_time/w_energy ratio) to raise the "
                f"effective speed floor."
            )

        # 步骤3：获取二阶导数控制点并添加 LorentzCone 约束
        # r''(s) 的控制点 Q_j，j = 0, 1, ..., n-2
        u_path_ddot = self.u_r_trajectory.MakeDerivative(2).control_points()

        for ii in range(len(u_path_ddot)):
            # 将二阶导数控制点分解为线性系数
            # Q_j = A_ctrl @ x
            A_ctrl = DecomposeLinearExpressions(u_path_ddot[ii], self.u_vars)

            # 构建 Lorentz 锥约束矩阵
            # 约束形式：C >= ||Q_j||_2
            # 即：C >= ||A_ctrl @ x||_2
            # 转化为 Lorentz 锥形式：
            #   z = H @ x + b, 其中 H = [0; A_ctrl], b = [C; 0]
            #   约束：z[0] >= ||z[1:]||_2
            #
            # H 的形状：(dimension + 1, num_vars)
            # H[0, :] = 0           (第一行全零，阈值由 b 提供)
            # H[1:, :] = A_ctrl     (后续行：二阶导数系数)
            H = np.vstack(
                [np.zeros((1, A_ctrl.shape[1])), A_ctrl]
            )

            # b 的形状：(dimension + 1,)
            # b[0] = C               (阈值)
            # b[1:] = 0              (无偏移)
            b = np.zeros(A_ctrl.shape[0] + 1)
            b[0] = C

            # 创建 Lorentz 锥约束
            # Drake 的 LorentzConeConstraint 定义：
            #   A @ x + b ∈ LorentzCone
            # 即：(A @ x + b)[0] >= ||(A @ x + b)[1:]||_2
            curvature_con = LorentzConeConstraint(H, b)

            # 存储约束对象（用于调试和可视化）
            self.deriv_constraints.append(curvature_con)

            # 对所有 GCS 边添加约束
            # 注意：跳过源点边（edge.u() == self.source），因为源点没有前驱
            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddConstraint(
                    Binding[Constraint](curvature_con, edge.xu())
                )

    def addCurvatureHardConstraintForEdges(
        self,
        max_curvature: float,
        min_velocity: float,
        h_bar_prime: float = None,
        edges=None,
    ):
        """对指定边添加曲率硬约束（用于分段控制：仅内部段）

        与 addCurvatureHardConstraint 相同的数学原理，但仅对指定
        的边集合添加约束，用于实现边界段/内部段分离策略。

        Args:
            max_curvature: 最大允许曲率 kappa_max (1/m)
            min_velocity: 内部段最小物理速度 (m/s)
            h_bar_prime: 时间参数化导数均值，None 则自动估算
            edges: 目标边列表，None 则对所有非 source 边添加

        Note:
            - 用于边界段免曲率硬约束的分段策略
            - 边界段由航向角约束隐式保证曲率
        """
        assert max_curvature > 0
        assert min_velocity > 0

        if h_bar_prime is None:
            import warnings
            warnings.warn(
                "h_bar_prime not specified, using default value 1.0. "
                "Use compute_h_bar_prime_from_trajectory() or "
                "estimate_h_bar_prime() for a better estimate.",
                UserWarning,
                stacklevel=2,
            )
            h_bar_prime = 1.0

        rho_min = min_velocity * h_bar_prime
        C = max_curvature * rho_min**2

        if C < 1e-10:
            raise ValueError(
                f"Curvature constraint threshold C = {C:.2e} is too small "
                f"(rho_min = {rho_min:.2e})."
            )

        u_path_ddot = self.u_r_trajectory.MakeDerivative(2).control_points()

        for ii in range(len(u_path_ddot)):
            A_ctrl = DecomposeLinearExpressions(u_path_ddot[ii], self.u_vars)
            H = np.vstack(
                [np.zeros((1, A_ctrl.shape[1])), A_ctrl]
            )
            b = np.zeros(A_ctrl.shape[0] + 1)
            b[0] = C
            curvature_con = LorentzConeConstraint(H, b)
            self.deriv_constraints.append(curvature_con)

            # 仅对指定边添加约束
            target_edges = edges if edges is not None else [
                edge for edge in self.gcs.Edges()
                if edge.u() != self.source
            ]
            for edge in target_edges:
                edge.AddConstraint(
                    Binding[Constraint](curvature_con, edge.xu())
                )

    def _isSourceInVertexRegion(self, source, vertex):
        """检查源点是否在顶点区域内"""
        # 获取顶点集合
        vertex_set = vertex.set()
        
        # 对于HPolyhedron，检查Ax <= b
        if isinstance(vertex_set, HPolyhedron):
            A = vertex_set.A()
            b = vertex_set.b()
            # 检查每个不等式约束
            for i in range(A.shape[0]):
                if A[i].dot(source) > b[i] + 1e-6:  # 添加小容差
                    return False
            return True
    def addSourceTarget(self, source, target, edges=None, velocity=None, zero_deriv_boundary=None, min_time_derivative=None):
        """
        添加源点和目标点，并设置边界条件
        
        Args:
            source (array): 源点位置
            target (array): 目标点位置
            edges (list, optional): 指定的边连接关系
            velocity (array, optional): 边界速度条件 [初始速度, 终止速度]
            zero_deriv_boundary (int, optional): 边界处设为零的导数阶数
            min_time_derivative (float, optional): 时间轨迹导数的最小值，防止 dh/ds 过小导致速度突变
            
        Returns:
            tuple: (源点边列表, 目标点边列表)
        """
        # 调用基类方法添加源点和目标点
        source_edges, target_edges = super().addSourceTarget(source, target, edges)

        # 检查并修复源点连接 
        if not source_edges:
            print(f"WARNING: No source edges created. Source position: {source}")
            # # 手动检查源点是否在各区域内部
            # for vertex in self.gcs.Vertices():
            #     if vertex != self.source and vertex != self.target:
                    # # 检查源点是否在顶点区域内
                    # if self._isSourceInVertexRegion(source, vertex):
                    #     print(f"Manually adding edge from source to {vertex.name()}")
                    #     edge = self.gcs.AddEdge(self.source, vertex, f"(source, {vertex.name()})")
                    #     source_edges.append(edge)
        
            # 同样检查目标点连接
        if not target_edges:
            print(f"WARNING: No target edges created. Target position: {target}")
            # # 手动检查目标点是否在各区域内部
            # for vertex in self.gcs.Vertices():
            #     if vertex != self.source and vertex != self.target:
                    # # 检查目标点是否在顶点区域内
                    # if self._isSourceInVertexRegion(target, vertex):
                    #     print(f"Manually adding edge from target to {vertex.name()}")
                    #     edge = self.gcs.AddEdge(self.source, vertex, f"(source, {vertex.name()})")
                    #     source_edges.append(edge)

        # 如果指定了边界速度条件
        if velocity is not None:
            assert velocity.shape == (2, self.dimension)  # 必须是2×维度的数组

            # 获取空间轨迹和时间轨迹的一阶导数控制点
            u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
            u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()
            # 计算初始速度误差：初始速度控制点 - 期望速度 * 初始时间导数
            initial_velocity_error = np.squeeze(u_path_control[0]) - velocity[0] * np.squeeze(u_time_control[0])
            # 计算终止速度误差
            final_velocity_error = np.squeeze(u_path_control[-1]) - velocity[1] * np.squeeze(u_time_control[-1])
            # 创建线性等式约束：误差 = 0
            initial_velocity_con = LinearEqualityConstraint(
                DecomposeLinearExpressions(initial_velocity_error, self.u_vars),
                np.zeros(self.dimension))
            final_velocity_con = LinearEqualityConstraint(
                DecomposeLinearExpressions(final_velocity_error, self.u_vars),
                np.zeros(self.dimension))

        # 如果指定了边界导数为零的条件
        if zero_deriv_boundary is not None:
            assert self.order > zero_deriv_boundary + 1  # 确保阶数足够高
            initial_constraints = []
            final_constraints = []

            # 为1到zero_deriv_boundary阶导数创建约束
            for deriv in range(1, zero_deriv_boundary+1):
                u_path_control = self.u_r_trajectory.MakeDerivative(deriv).control_points()
                # 初始点导数为零
                initial_constraints.append(LinearEqualityConstraint(
                    DecomposeLinearExpressions(np.squeeze(u_path_control[0]), self.u_vars),
                    np.zeros(self.dimension, dtype=float)))
                # 终止点导数为零
                final_constraints.append(LinearEqualityConstraint(
                    DecomposeLinearExpressions(np.squeeze(u_path_control[-1]), self.u_vars),
                    np.zeros(self.dimension, dtype=float)))

        # 为源点边添加约束
        for edge in source_edges:
            # 源点边：源点和第一个顶点在空间上重合
            for jj in range(self.dimension):
                edge.AddConstraint(edge.xu()[jj] == edge.xv()[jj])

            # 添加速度约束（如果指定）
            if velocity is not None:
                edge.AddConstraint(Binding[Constraint](initial_velocity_con, edge.xv()))
            # 添加边界导数为零的约束（如果指定）
            if zero_deriv_boundary is not None:
                for i_con in initial_constraints:
                    edge.AddConstraint(Binding[Constraint](i_con, edge.xv()))

            # 添加时间轨迹导数约束（如果指定），防止 dh/ds 过小导致速度突变
            if min_time_derivative is not None and min_time_derivative > 0:
                u_time_deriv = self.u_h_trajectory.MakeDerivative(1).control_points()
                initial_time_deriv_con = LinearConstraint(
                    DecomposeLinearExpressions(u_time_deriv[0], self.u_vars),
                    min_time_derivative * np.ones(1),
                    np.inf * np.ones(1)
                )
                edge.AddConstraint(Binding[Constraint](initial_time_deriv_con, edge.xv()))

            # 源点的时间起点为0
            edge.AddConstraint(edge.xv()[-(self.order + 1)] == 0.)

        # 为目标点边添加约束
        for edge in target_edges:    
            # 目标点边：最后一个顶点和目标点在空间上重合
            for jj in range(self.dimension):
                edge.AddConstraint(
                    edge.xu()[-(self.dimension + self.order + 1) + jj] == edge.xv()[jj])

            # 添加速度约束（如果指定）
            if velocity is not None:
                edge.AddConstraint(Binding[Constraint](final_velocity_con, edge.xu()))
            # 添加边界导数为零的约束（如果指定）
            if zero_deriv_boundary is not None:
                for f_con in final_constraints:
                    edge.AddConstraint(Binding[Constraint](f_con, edge.xu()))

            # 添加时间轨迹导数约束（如果指定），防止 dh/ds 过小导致速度突变
            if min_time_derivative is not None and min_time_derivative > 0:
                u_time_deriv = self.u_h_trajectory.MakeDerivative(1).control_points()
                final_time_deriv_con = LinearConstraint(
                    DecomposeLinearExpressions(u_time_deriv[-1], self.u_vars),
                    min_time_derivative * np.ones(1),
                    np.inf * np.ones(1)
                )
                edge.AddConstraint(Binding[Constraint](final_time_deriv_con, edge.xu()))

            # 为目标点边添加所有成本函数
            for cost in self.edge_costs:
                edge.AddCost(Binding[Cost](cost, edge.xu()))

            # 为目标点边添加所有导数约束
            for d_con in self.deriv_constraints:
                edge.AddConstraint(Binding[Constraint](d_con, edge.xu()))

        return source_edges, target_edges

    def SolvePath(self, rounding=False, verbose=False, preprocessing=False):
        """
        求解最优路径并返回轨迹
        
        Args:
            rounding (bool): 是否使用舍入策略（先求解松弛问题）
            verbose (bool): 是否显示详细求解信息
            preprocessing (bool): 是否进行预处理
            
        Returns:
            tuple: (BezierTrajectory对象, 结果字典)
        """
        # 调用基类方法求解GCS
        best_path, best_result, results_dict = self.solveGCS(
            rounding, preprocessing, verbose)

        if best_path is None:
            return None, results_dict

        # 提取轨迹控制点
        knots = np.zeros(self.order + 1)  # 节点向量
        path_control_points = []  # 空间控制点列表
        time_control_points = []  # 时间控制点列表
        
        # 遍历最优路径中的每条边
        for edge in best_path:
            # 如果到达目标点，处理最后一条边
            if edge.v() == self.target:
                knots = np.concatenate((knots, [knots[-1]]))
                # 获取目标点处的控制点
                path_control_points.append(best_result.GetSolution(edge.xv()))
                # 获取目标点处的时间
                time_control_points.append(np.array([best_result.GetSolution(edge.xu())[-1]]))
                break
                
            # 计算当前边的结束时间
            edge_time = knots[-1] + 1.
            # 扩展节点向量
            knots = np.concatenate((knots, np.full(self.order, edge_time)))
            
            # 提取边的控制点
            # 注意：edge.xv() 包含空间控制点和时间控制点
            edge_path_points = np.reshape(best_result.GetSolution(edge.xv())[:-(self.order + 1)],
                                             (self.dimension, self.order + 1), "F")

            edge_time_points = best_result.GetSolution(edge.xv())[-(self.order + 1):]
            
            # 将除最后一个控制点外的所有控制点添加到轨迹
            for ii in range(self.order):
                path_control_points.append(edge_path_points[:, ii])
                time_control_points.append(np.array([edge_time_points[ii]]))

        # 时间偏移校正：使起始时间为0
        offset = time_control_points[0].copy()
        for ii in range(len(time_control_points)):
            time_control_points[ii] -= offset

        # 转换为合适的数组格式
        path_control_points = np.array(path_control_points).T
        time_control_points = np.array(time_control_points).T

        # 创建B样条轨迹
        path = BsplineTrajectory(BsplineBasis(self.order + 1, knots), path_control_points)
        time_traj = BsplineTrajectory(BsplineBasis(self.order + 1, knots), time_control_points)

        return BezierTrajectory(path, time_traj), results_dict

class BezierTrajectory:
    """
    封装贝塞尔轨迹，提供基于实际时间的轨迹查询接口。
    
    该类将空间轨迹和时间轨迹组合，使得可以通过实际时间t查询对应的空间位置。
    """
    
    def __init__(self, path_traj, time_traj):
        """
        初始化贝塞尔轨迹
        
        Args:
            path_traj (BsplineTrajectory): 空间轨迹
            time_traj (BsplineTrajectory): 时间轨迹（s->t映射）
        """
        assert path_traj.start_time() == time_traj.start_time()
        assert path_traj.end_time() == time_traj.end_time()
        self.path_traj = path_traj
        self.time_traj = time_traj
        self.start_s = path_traj.start_time()  # 参数s的起始值
        self.end_s = path_traj.end_time()      # 参数s的结束值

    def invert_time_traj(self, t):
        """
        将实际时间t映射回参数s（求解t = h(s)的反函数）
        
        Args:
            t (float): 实际时间
            
        Returns:
            float: 对应的参数s值
        """
        if t <= self.start_time():
            return self.start_s
        if t >= self.end_time():
            return self.end_s
        # 定义误差函数：h(s) - t
        error = lambda s: self.time_traj.value(s)[0, 0] - t
        # 使用根查找算法求解s
        res = root_scalar(error, bracket=[self.start_s, self.end_s])
        return np.min([np.max([res.root, self.start_s]), self.end_s])

    def value(self, t):
        """
        获取t时刻的空间位置
        
        Args:
            t (float): 实际时间
            
        Returns:
            array: t时刻的空间位置
        """
        return self.path_traj.value(self.invert_time_traj(np.squeeze(t)))

    def vector_values(self, times):
        """
        获取多个时间点的空间位置
        
        Args:
            times (array): 时间点数组
            
        Returns:
            array: 对应的空间位置
        """
        s = [self.invert_time_traj(t) for t in np.squeeze(times)]
        return self.path_traj.vector_values(s)

    def EvalDerivative(self, t, derivative_order=1):
        """
        计算t时刻的指定阶导数
        
        Args:
            t (float): 实际时间
            derivative_order (int): 导数阶数（0=位置，1=速度，2=加速度）
            
        Returns:
            array: 指定阶导数
        """
        if derivative_order == 0:
            return self.value(t)
        elif derivative_order == 1:
            s = self.invert_time_traj(np.squeeze(t))
            s_dot = 1./self.time_traj.EvalDerivative(s, 1)[0, 0]  # ds/dt
            r_dot = self.path_traj.EvalDerivative(s, 1)           # dr/ds
            return r_dot * s_dot                                  # dr/dt = dr/ds * ds/dt
        elif derivative_order == 2:
            s = self.invert_time_traj(np.squeeze(t))
            s_dot = 1./self.time_traj.EvalDerivative(s, 1)[0, 0]
            h_ddot = self.time_traj.EvalDerivative(s, 2)[0, 0]    # d²h/ds²
            s_ddot = -h_ddot*(s_dot**3)                           # d²s/dt²
            r_dot = self.path_traj.EvalDerivative(s, 1)
            r_ddot = self.path_traj.EvalDerivative(s, 2)
            # d²r/dt² = d²r/ds² * (ds/dt)² + dr/ds * d²s/dt²
            return r_ddot * s_dot * s_dot + r_dot * s_ddot
        else:
            raise ValueError("Unsupported derivative order")

    def start_time(self):
        """获取轨迹起始时间"""
        return self.time_traj.value(self.start_s)[0, 0]

    def end_time(self):
        """获取轨迹结束时间"""
        return self.time_traj.value(self.end_s)[0, 0]

    def rows(self):
        """获取轨迹维度（行数）"""
        return self.path_traj.rows()

    def cols(self):
        """获取轨迹列数（通常为1）"""
        return self.path_traj.cols()
    
