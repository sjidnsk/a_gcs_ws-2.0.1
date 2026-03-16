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

    def addPathEnergyCost(self, weight):
        """
        添加路径能量成本：最小化动能（速度平方积分）
        
        Args:
            weight (float or array): 能量成本的权重
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

            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddCost(Binding[Cost](energy_cost, edge.xu()))

    def addDerivativeRegularization(self, weight_r, weight_h, order):
        """
        添加导数正则化：惩罚高阶导数，使轨迹更平滑
        
        Args:
            weight_r (float): 空间轨迹导数的正则化权重
            weight_h (float): 时间轨迹导数的正则化权重
            order (int): 要正则化的导数阶数
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

                for edge in self.gcs.Edges():
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

    def addTurningRadiusConstraint(self, min_radius, max_velocity):
        """
        添加转弯半径约束（通过限制向心加速度实现）
        公式: ||r''(t)|| <= (v^2 / min_radius)
        """
        # 计算加速度上限
        max_accel = (max_velocity**2) / min_radius
        
        # 获取空间轨迹的二阶导数控制点 r''(s)
        # 注意：这里的导数是相对于参数 s 的，需要转换到时间 t
        u_path_ddot = self.u_r_trajectory.MakeDerivative(2).control_points()
        u_time_dot = self.u_h_trajectory.MakeDerivative(1).control_points()
        
        # 近似处理：在 s 域限制加速度
        # 严谨做法需考虑 ds/dt 的平方，这里简化为对空间曲率控制点的约束
        for ii in range(len(u_path_ddot)):
            # 提取二阶导数的线性表达式矩阵
            A_ctrl = DecomposeLinearExpressions(u_path_ddot[ii], self.u_vars)
            
            # 创建 L2Norm 约束: ||A_ctrl * vars|| <= max_accel
            # 注意：实际中 s 和 t 的关系非线性，这里通常假设速度恒定进行预估
            accel_con = L2NormCost(A_ctrl, np.zeros(self.dimension)) 
            
            # 遍历所有边，添加这个“成本”作为软约束，或使用 AddConstraint 添加硬约束
            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                # 添加硬约束：加速度控制点必须在半径为 max_accel 的球体内
                edge.AddConstraint(Binding[Constraint](
                    LinearConstraint(A_ctrl, -max_accel * np.ones(self.dimension), 
                                    max_accel * np.ones(self.dimension)), 
                    edge.xu()))

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
    def addSourceTarget(self, source, target, edges=None, velocity=None, zero_deriv_boundary=None):
        """
        添加源点和目标点，并设置边界条件
        
        Args:
            source (array): 源点位置
            target (array): 目标点位置
            edges (list, optional): 指定的边连接关系
            velocity (array, optional): 边界速度条件 [初始速度, 终止速度]
            zero_deriv_boundary (int, optional): 边界处设为零的导数阶数
            
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
                    np.zeros(self.dimension)))
                # 终止点导数为零
                final_constraints.append(LinearEqualityConstraint(
                    DecomposeLinearExpressions(np.squeeze(u_path_control[-1]), self.u_vars),
                    np.zeros(self.dimension)))

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
    
