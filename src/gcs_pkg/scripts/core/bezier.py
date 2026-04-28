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
    RotatedLorentzConeConstraint,  # 旋转Lorentz锥约束（v2曲率约束）
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
    
    def __init__(self, regions, order, continuity, edges=None, hdot_min=0.01, full_dim_overlap=False, hyperellipsoid_num_samples_per_dim_factor=32, curvature_constraint_version="v1"):
        """
        初始化贝塞尔GCS
        
        Args:
            regions (list): 凸集列表，每个凸集表示一个可行区域
            order (int): B样条曲线的阶数（控制点数量 = order + 1）
            continuity (int): 路径连续性要求（0=C0连续，1=C1连续，2=C2连续等）
            edges (list, optional): 指定的边连接关系，默认为None时自动计算
            hdot_min (float): 时间导数的最小值，确保时间单调递增
            full_dim_overlap (bool): 是否要求区域交集维度为全维度
            curvature_constraint_version (str): 曲率约束版本，"v1"或"v2"。
                v2模式下顶点凸集预扩展+2维(σ_e, τ_e)。
        """
        # 调用基类初始化
        BaseGCS.__init__(self, regions, auto_add_vertices=False)

        self.order = order
        self.continuity = continuity
        self.hdot_min = hdot_min
        # 存储采样点数量参数（作为因子）
        self.hyperellipsoid_num_samples_per_dim_factor = hyperellipsoid_num_samples_per_dim_factor
        assert continuity < order  # 连续性要求必须小于曲线阶数

        # 曲率约束版本标记：v2模式下顶点凸集预扩展+2维(σ_e, τ_e)
        self._curvature_constraint_version = curvature_constraint_version
        self._vertex_extended = (curvature_constraint_version == "v2")

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
            # v2曲率约束：扩展顶点凸集维度+2 (σ_e, τ_e)
            if self._vertex_extended:
                free_R2 = HPolyhedron(np.zeros((0, 2)), np.zeros(0))
                vertex_set = vertex_set.CartesianProduct(free_R2)
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
        
        # v2曲率约束：扩展u_vars，追加σ_e和τ_e变量
        if self._vertex_extended:
            sigma_var = MakeVectorContinuousVariable(1, "sigma")[0]
            tau_var = MakeVectorContinuousVariable(1, "tau")[0]
            self.u_vars = np.concatenate([self.u_vars, [sigma_var, tau_var]])
        
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
        
        # v2曲率约束：扩展edge_vars，追加σ_e, τ_e (u侧) 和 σ_e, τ_e (v侧)
        if self._vertex_extended:
            u_sigma = MakeVectorContinuousVariable(1, "sigma_u")[0]
            u_tau = MakeVectorContinuousVariable(1, "tau_u")[0]
            v_sigma = MakeVectorContinuousVariable(1, "sigma_v")[0]
            v_tau = MakeVectorContinuousVariable(1, "tau_v")[0]
            edge_vars = np.concatenate([edge_vars, [u_sigma, u_tau, v_sigma, v_tau]])
        
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
            A_path = DecomposeLinearExpressions(path_continuity_error, edge_vars)
            self.contin_constraints.append(LinearEqualityConstraint(
                A_path, np.zeros(self.dimension)))

            # 时间轨迹的连续性约束（确保时间参数化连续）
            u_time_deriv = self.u_h_trajectory.MakeDerivative(deriv)
            v_time_deriv = v_h_trajectory.MakeDerivative(deriv)
            time_continuity_error = v_time_deriv.control_points()[0] - u_time_deriv.control_points()[-1]
            A_time_cont = DecomposeLinearExpressions(time_continuity_error, edge_vars)
            self.contin_constraints.append(LinearEqualityConstraint(
                A_time_cont, 0.0))

        # 存储导数约束和边成本
        self.deriv_constraints = []           # 仅速度约束
        self.curvature_constraints = []       # 仅曲率约束
        self._curvature_constraint_bindings = []  # (edge, binding) 对
        self.curvature_constraints_v2 = None  # v2曲率约束结果 (CurvatureV2Result)
        self._curvature_v2_bindings = []      # v2 (edge, binding, type) 三元组
        self._curvature_v2_added = False      # v2曲率约束幂等性标记
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

    def addTimeDerivativeRegularization(self, weight: float, h_ref: float = None):
        """
        添加时间导数正则化：惩罚过小的时间缩放导数h'(s)

        通过对时间轨迹h(s)的一阶导数控制点添加二次正则化成本，
        使h'(s)远离零值，避免ds/dt = 1/h'(s)过大导致加速度爆炸。

        数学推导：
            目标：惩罚 h'(s) 偏离参考值 h_ref 的程度
            成本形式：Σ_i weight * ||h'_i - h_ref||^2
            其中 h'_i 是第i个一阶导数控制点，h_ref 是参考时间导数

            展开为二次形式：
            weight * Σ_i ||h'_i||^2 - 2*weight*h_ref*Σ_i h'_i + const
            = weight * Σ_i (h'_i^T * h'_i) - 2*weight*h_ref*Σ_i h'_i + const

            第一项：QuadraticCost（凸）
            第二项：LinearCost（凸）
            常数项：可忽略

        凸性保证：
            QuadraticCost 和 LinearCost 均为凸成本，
            不破坏GCS优化问题的凸性。

        Args:
            weight (float): 正则化权重，必须为非负数。
                weight越大，h'(s)越远离零值，轨迹时间越长但数值更稳定。
                推荐范围：[0.1, 10.0]
                weight=0时等效于禁用正则化。
            h_ref (float, optional): h'(s)的参考目标值。如果为None，则使用hdot_min。
                增大h_ref可使h'(s)远离零值，从根本上避免加速度发散。
                推荐范围：[0.2, 0.5]，需大于hdot_min。

        Raises:
            AssertionError: 如果weight为负数

        Note:
            - 此方法与addDerivativeRegularization的区别：
              addDerivativeRegularization正则化二阶及以上导数（order>=2），
              此方法正则化一阶导数，专门用于防止h'(s)过小。
            - 参考值h_ref默认从hdot_min推导，也可通过参数指定更大的值。
        """
        assert weight >= 0, f"weight must be non-negative, got {weight}"
        if weight == 0:
            return

        # 获取时间轨迹的一阶导数控制点
        u_time_deriv_control = self.u_h_trajectory.MakeDerivative(1).control_points()

        # 参考值：使用 h_ref 参数或默认 hdot_min
        if h_ref is None:
            h_ref = self.hdot_min

        # 对每个一阶导数控制点添加正则化成本
        for c in u_time_deriv_control:
            # 将控制点表达式分解为线性系数
            A_ctrl = DecomposeLinearExpressions(c, self.u_vars)

            # 二次项：weight * ||h'_i||^2
            # QuadraticCost(H, b, c): 成本 = 0.5 * x^T * H * x + b^T * x + c
            # 我们需要 weight * ||A_ctrl @ x||^2 = weight * x^T * (A_ctrl^T * A_ctrl) * x
            H_quad = A_ctrl.T.dot(A_ctrl) * 2 * weight
            # 确保H_quad是正半定的（添加微小对角正则化避免浮点误差）
            H_quad += np.eye(H_quad.shape[0]) * 1e-12
            quad_cost = QuadraticCost(
                H_quad, np.zeros(H_quad.shape[0]), 0
            )

            # 线性项：-2 * weight * h_ref * h'_i
            # LinearCost(a, b): 成本 = a^T * x + b
            a_lin = -2 * weight * h_ref * A_ctrl[0, :]
            lin_cost = LinearCost(a_lin, 0.0)

            self.edge_costs.append(quad_cost)
            self.edge_costs.append(lin_cost)

            # 对所有GCS边添加成本（源点边除外）
            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddCost(Binding[Cost](quad_cost, edge.xu()))
                edge.AddCost(Binding[Cost](lin_cost, edge.xu()))

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
    def compute_h_bar_prime_from_trajectory(trajectory, num_samples=200):
        """从已求解轨迹精确计算 h̄' = (1/Δs)·∫h'(s)ds。

        使用梯形法则对 h'(s) = dh/ds 在 s ∈ [s_start, s_end] 上
        进行数值积分，返回均值。

        Args:
            trajectory: 已求解的 BezierTrajectory 对象，
                须包含有效的 time_traj。
            num_samples: 数值积分采样点数，必须 >= 10。默认 200。

        Returns:
            float: h̄' 值（正数）。

        Raises:
            ValueError: trajectory 无效（time_traj 为 None）
                或 num_samples < 10。

        Note:
            - h̄' ≈ L_path / T_total（路径长度 / 总物理时间）
            - 用于曲率硬约束阈值 C = κ_max · (v_min · h̄')²
        """
        if num_samples < 10:
            raise ValueError(
                f"num_samples must be >= 10, got {num_samples}"
            )
        if trajectory is None or not hasattr(trajectory, "time_traj"):
            raise ValueError(
                "trajectory must be a valid BezierTrajectory with "
                "time_traj attribute"
            )
        if trajectory.time_traj is None:
            raise ValueError(
                "trajectory.time_traj is None; cannot compute h_bar_prime"
            )

        s_start = trajectory.start_s
        s_end = trajectory.end_s
        if s_end <= s_start:
            raise ValueError(
                f"Invalid parameter range: s_end={s_end} <= "
                f"s_start={s_start}"
            )

        # 生成采样点
        s_points = np.linspace(s_start, s_end, num_samples + 1)

        # 获取 time_traj 的一阶导数 h'(s) = dh/ds
        h_deriv = trajectory.time_traj.MakeDerivative(1)

        # 批量计算 h'(s_i)
        h_prime_values = np.array(
            [h_deriv.value(s)[0, 0] for s in s_points]
        )

        # 梯形法则数值积分
        integral = np.trapezoid(h_prime_values, s_points)

        # 计算均值
        h_bar_prime = integral / (s_end - s_start)

        # 有限性检查
        if not np.isfinite(h_bar_prime) or h_bar_prime <= 0:
            raise ValueError(
                f"Computed h_bar_prime is not a finite positive number: "
                f"{h_bar_prime}. Check trajectory validity."
            )

        return float(h_bar_prime)

    @staticmethod
    def estimate_h_bar_prime(
        path_length_estimate,
        num_segments,
        w_time=1.0,
        w_energy=0.1,
        v_optimal=None,
        hdot_min=0.01,
    ):
        """静态估算 h̄' ≈ L_path / (N_segments · v_optimal)。

        在求解前基于路径几何和成本权重估算 h̄'，精度中等。

        Args:
            path_length_estimate: 路径长度估计值，必须 > 0。
            num_segments: 轨迹段数，必须 >= 1。
            w_time: 时间成本权重，必须 >= 0。默认 1.0。
            w_energy: 能量成本权重，必须 > 0。默认 0.1。
            v_optimal: 最优速度，若指定则忽略 w_time/w_energy。
            hdot_min: h̄' 下界，低于此值截断并警告。默认 0.01。

        Returns:
            float: h̄' 估算值（正数，>= hdot_min）。

        Raises:
            ValueError: 参数不合法。

        Note:
            - v_optimal = sqrt(w_time / w_energy)
            - 结果低于 hdot_min 时截断为 hdot_min 并发出 UserWarning
        """
        import warnings

        if path_length_estimate <= 0:
            raise ValueError(
                f"path_length_estimate must be positive, "
                f"got {path_length_estimate}"
            )
        if num_segments < 1:
            raise ValueError(
                f"num_segments must be >= 1, got {num_segments}"
            )
        if w_time < 0:
            raise ValueError(
                f"w_time must be non-negative, got {w_time}"
            )
        if w_energy <= 0:
            raise ValueError(
                f"w_energy must be positive, got {w_energy}"
            )

        # 计算最优速度
        if v_optimal is None:
            v_optimal = np.sqrt(w_time / w_energy)

        # 静态估算
        h_bar_prime = path_length_estimate / (num_segments * v_optimal)

        # 下界保护
        if h_bar_prime < hdot_min:
            warnings.warn(
                f"Estimated h_bar_prime={h_bar_prime:.6f} is below "
                f"hdot_min={hdot_min}. Clamping to hdot_min.",
                UserWarning,
                stacklevel=2,
            )
            h_bar_prime = hdot_min

        return float(h_bar_prime)

    def addCurvatureHardConstraint(
        self, max_curvature, min_velocity, h_bar_prime=None,
        h_bar_prime_safety_factor=1.0,
    ):
        """
        添加曲率硬约束：||Q_j||_2 <= C = kappa_max * rho_min^2

        利用贝塞尔曲线的凸包特性，通过限制二阶导数控制点的范数来
        间接、保守地保证曲率在安全范围内。

        数学推导：
            曲率公式：kappa(s) = ||r'(s) x r''(s)|| / ||r'(s)||^3
            Cauchy-Schwarz：|kappa(s)| <= ||r''(s)|| / ||r'(s)||^2
            凸包特性：||r''(s)|| <= max_j ||Q_j||
            充分条件：||Q_j||_2 <= C = kappa_max * rho_min^2

        其中 Q_j 是二阶导数贝塞尔曲线的控制点：
            Q_j = n*(n-1) * (P_{j+2} - 2*P_{j+1} + P_j),  j = 0, 1, ..., n-2

        Lorentz锥约束形式：
            z = H @ x + b, 其中 H = [0; A_ctrl], b = [C; 0]
            约束：z[0] >= ||z[1:]||_2  即  C >= ||Q_j||_2

        Args:
            max_curvature: 最大允许曲率 kappa_max (1/m)，必须为正数
            min_velocity: 最小速度 v_min (m/s)，用于计算 rho_min
            h_bar_prime: h'(s)的均值估计。None时使用默认值1.0。
            h_bar_prime_safety_factor: 保守修正因子，范围(0, 1.0]。
                默认1.0（不修正）。推荐0.7。

        Raises:
            ValueError: 如果 max_curvature <= 0 或 min_velocity < 0
            ValueError: 如果计算得到的 C 值为0（约束退化，仅允许直线）
            ValueError: 如果 h_bar_prime_safety_factor 不在 (0, 1.0]

        Note:
            - 此约束是凸约束（Lorentz锥），保持优化问题的凸性
            - 与 addScalarVelocityLimit 同构，仅导数阶数和阈值不同
            - 保守性来源：Cauchy-Schwarz不等式、凸包特性、速度下界估计
            - 当 min_velocity=0 时约束退化（C=0），需配合边界退化处理

        Example:
            >>> bezier_gcs = BezierGCS(regions, order=5, continuity=2)
            >>> bezier_gcs.addCurvatureHardConstraint(
            ...     max_curvature=0.5, min_velocity=0.7
            ... )
        """
        # 参数验证
        if max_curvature <= 0:
            raise ValueError(f"max_curvature must be positive, got {max_curvature}")
        if min_velocity < 0:
            raise ValueError(f"min_velocity must be non-negative, got {min_velocity}")

        # 步骤1：计算 h_bar_prime（自动估算或用户指定）
        if h_bar_prime is None:
            h_bar_prime = 1.0
            print(
                "Warning: h_bar_prime using default value 1.0 "
                "(fallback estimate, consider using iterative refinement "
                "or static estimation)"
            )

        # 步骤1.5：safety_factor 验证和应用
        if not (0 < h_bar_prime_safety_factor <= 1.0):
            raise ValueError(
                f"h_bar_prime_safety_factor must be in (0, 1.0], "
                f"got {h_bar_prime_safety_factor}"
            )
        effective_h_bar_prime = h_bar_prime * h_bar_prime_safety_factor

        # 步骤2：计算 rho_min 和 C（使用 effective_h_bar_prime）
        rho_min = min_velocity * effective_h_bar_prime
        C = max_curvature * rho_min ** 2

        if C <= 0:
            raise ValueError(
                f"Curvature constraint threshold C = kappa_max * rho_min^2 = {C:.6f} is non-positive. "
                f"This means the constraint degenerates (only straight lines allowed). "
                f"Parameters: kappa_max={max_curvature}, v_min={min_velocity}, "
                f"h_bar_prime={h_bar_prime}, rho_min={rho_min}. "
                f"Consider using a positive min_velocity or applying this constraint "
                f"only to interior segments (not boundary segments with v=0)."
            )

        # 步骤3：获取二阶导数控制点
        # r''(s) 的控制点 Q_j = n*(n-1) * (P_{j+2} - 2*P_{j+1} + P_j)
        u_path_ddot = self.u_r_trajectory.MakeDerivative(2).control_points()

        # 对每个二阶导数控制点添加 Lorentz 锥约束
        for ii in range(len(u_path_ddot)):
            # 将二阶导数表达式分解为线性系数
            # A_ctrl: 二阶导数的系数矩阵，形状 (dimension, num_vars)
            #         表示 Q_j = A_ctrl @ x
            A_ctrl = DecomposeLinearExpressions(u_path_ddot[ii], self.u_vars)

            # 构建 Lorentz 锥约束矩阵
            # 约束形式：C >= ||Q_j||_2
            # 即：C >= ||A_ctrl @ x||_2
            # 转化为 Lorentz 锥形式：
            #   z = H @ x + b, 其中 H = [0; A_ctrl], b = [C; 0]
            #   约束：z[0] >= ||z[1:]||_2
            #
            # H 的形状：(dimension + 1, num_vars)
            # H[0, :] = 0  (零行，因为 C 是常数项)
            # H[1:, :] = A_ctrl  (二阶导数系数)
            H = np.vstack([
                np.zeros((1, A_ctrl.shape[1])),  # 第一行：零（C在b中）
                A_ctrl                            # 后续行：二阶导数系数
            ])

            # b 向量：[C; 0, 0, ..., 0]
            b = np.zeros(A_ctrl.shape[0] + 1)
            b[0] = C

            # 创建 Lorentz 锥约束
            # Drake的LorentzConeConstraint定义：
            #   x ∈ LorentzCone <=> x[0] >= sqrt(x[1]^2 + ... + x[n]^2)
            # 构造函数：LorentzConeConstraint(A, b)
            # 约束：A @ x + b ∈ LorentzCone
            curvature_con = LorentzConeConstraint(H, b)

            # 存储到曲率约束专用列表（非deriv_constraints）
            self.curvature_constraints.append(curvature_con)

            # 对所有GCS边添加约束（跳过源点边），并保存Binding
            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                binding = edge.AddConstraint(
                    Binding[Constraint](curvature_con, edge.xu())
                )
                self._curvature_constraint_bindings.append((edge, binding))

    def addCurvatureHardConstraintForEdges(
        self, max_curvature, min_velocity,
        boundary_edge_ids=None, h_bar_prime=None,
        h_bar_prime_safety_factor=1.0,
    ):
        """
        添加曲率硬约束（支持跳过边界段）

        与 addCurvatureHardConstraint 相同的数学原理，但支持跳过
        起终点v=0附近的边界段，由航向角约束隐式保证曲率。

        Args:
            max_curvature: 最大允许曲率 kappa_max (1/m)
            min_velocity: 最小速度 v_min (m/s)
            boundary_edge_ids: 边界边的id集合，这些边不应用曲率硬约束
            h_bar_prime: h'(s)的均值估计。None时使用默认值1.0。
            h_bar_prime_safety_factor: 保守修正因子，范围(0, 1.0]。
                默认1.0（不修正）。推荐0.7。

        Raises:
            ValueError: 如果 max_curvature <= 0 或 min_velocity < 0
            ValueError: 如果计算得到的 C 值为0
            ValueError: 如果 h_bar_prime_safety_factor 不在 (0, 1.0]

        Note:
            - 边界段（起终点v=0附近）由航向角约束隐式保证曲率
            - 内部段使用 Lorentz 锥约束显式保证曲率
        """
        # 参数验证
        if max_curvature <= 0:
            raise ValueError(f"max_curvature must be positive, got {max_curvature}")
        if min_velocity < 0:
            raise ValueError(f"min_velocity must be non-negative, got {min_velocity}")

        # 计算 h_bar_prime
        if h_bar_prime is None:
            h_bar_prime = 1.0
            print(
                "Warning: h_bar_prime using default value 1.0 "
                "(fallback estimate, consider using iterative refinement "
                "or static estimation)"
            )

        # safety_factor 验证和应用
        if not (0 < h_bar_prime_safety_factor <= 1.0):
            raise ValueError(
                f"h_bar_prime_safety_factor must be in (0, 1.0], "
                f"got {h_bar_prime_safety_factor}"
            )
        effective_h_bar_prime = h_bar_prime * h_bar_prime_safety_factor

        # 计算 rho_min 和 C（使用 effective_h_bar_prime）
        rho_min = min_velocity * effective_h_bar_prime
        C = max_curvature * rho_min ** 2

        if C <= 0:
            raise ValueError(
                f"Curvature constraint threshold C = {C:.6f} is non-positive. "
                f"Parameters: kappa_max={max_curvature}, v_min={min_velocity}, "
                f"h_bar_prime={h_bar_prime}, rho_min={rho_min}. "
                f"Consider applying this constraint only to interior segments."
            )

        # 获取二阶导数控制点
        u_path_ddot = self.u_r_trajectory.MakeDerivative(2).control_points()

        # 对每个二阶导数控制点添加 Lorentz 锥约束
        for ii in range(len(u_path_ddot)):
            A_ctrl = DecomposeLinearExpressions(u_path_ddot[ii], self.u_vars)

            H = np.vstack([
                np.zeros((1, A_ctrl.shape[1])),
                A_ctrl
            ])
            b = np.zeros(A_ctrl.shape[0] + 1)
            b[0] = C

            curvature_con = LorentzConeConstraint(H, b)
            self.curvature_constraints.append(curvature_con)

            # 对所有GCS边添加约束（跳过源点边和边界边），并保存Binding
            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                # 跳过边界段（起终点v=0附近）
                if boundary_edge_ids is not None and id(edge) in boundary_edge_ids:
                    continue
                binding = edge.AddConstraint(
                    Binding[Constraint](curvature_con, edge.xu())
                )
                self._curvature_constraint_bindings.append((edge, binding))

    def addCurvatureHardConstraintV2(
        self, max_curvature, heading_directions=None,
        boundary_edge_ids=None, sigma_min="auto",
        ackermann_gcs=None,
        source_heading=None, target_heading=None,
    ):
        """添加曲率硬约束v2：旋转二阶锥 + 线性速度下界

        与v1的区别：
        - v1: ‖Qⱼ‖ ≤ κ_max · ρ_min²  (全局常数阈值)
        - v2: ‖Qⱼ‖ ≤ κ_max · σ_e²    (逐边变量阈值，由局部速度决定)

        保守性改善：消除 (v_max/v_min)² 因子，保守性降低1~2个数量级

        约束体系：
        - A1: qᵢ · d_θ ≥ σ_e          (线性速度下界, n+1个/边)
        - A2: τ_e · 1 ≥ σ_e²           (旋转二阶锥, 1个/边)
        - B:  κ_max · τ_e ≥ ‖Qⱼ‖₂     (Lorentz锥, n-1个/边)
        - C:  σ_e ≥ σ_min              (下界保证, 1个/边)

        Args:
            max_curvature: 最大允许曲率 κ_max (1/m)，必须为正数
            heading_directions: 逐边航向角方向映射 {edge_id: np.array([cos_θ, sin_θ])}
                若为None，从航向角约束自动提取
            boundary_edge_ids: 边界边ID集合，这些边不应用曲率硬约束
            sigma_min: σ_e的最小下界，"auto"自动推导，或用户显式指定正数

        Returns:
            CurvatureV2Result: 包含所有约束和binding的命名元组

        Raises:
            InvalidParameterError: 如果 max_curvature <= 0 或 sigma_min <= 0
            ConstraintConstructionError: 如果约束构建失败

        Note:
            - 此约束是凸约束（SOCP），保持优化问题的凸性
            - 需要MOSEK/SCS/Clarabel求解器（Gurobi不支持旋转锥）
            - 无需 min_velocity 和 h_bar_prime 参数（σ_e是优化变量）
            - 无需 h_bar_prime 迭代修正
            - 可行域严格包含v1的可行域
        """
        if self._curvature_v2_added:
            import warnings
            warnings.warn("v2曲率约束已添加，跳过重复调用", UserWarning, stacklevel=2)
            return self.curvature_constraints_v2

        from ackermann_gcs_pkg.curvature_constraint_v2.coordinator import CurvatureConstraintCoordinator

        # 构建配置对象
        config = type('CurvatureV2Config', (), {})()
        config.max_curvature = max_curvature
        config.curvature_constraint_version = "v2"
        config.enable_curvature_hard_constraint = True
        config.heading_method = 'rotation_matrix'
        config.sigma_min = sigma_min
        config.boundary_edge_ids = boundary_edge_ids or set()
        config.order = self.order
        config.hdot_min = self.hdot_min
        config.source_heading = source_heading
        config.target_heading = target_heading

        coordinator = CurvatureConstraintCoordinator(self, ackermann_gcs=ackermann_gcs)
        result = coordinator.add_curvature_constraint(config)

        return result

    def removeCurvatureHardConstraints(self, verbose=False):
        """移除GCS图上所有已添加的曲率硬约束。

        由于Drake Python binding不提供edge.RemoveConstraint()，
        采用边重建方案：收集所有受影响的普通边（非source/target边），
        移除后重建，重新添加连续性约束、速度约束和成本。

        source/target边上的曲率约束数量极少（仅order-1个），
        对性能影响可忽略，且重建这些边需要重新添加addSourceTarget
        中的大量专用约束，代价过高，因此跳过。

        Args:
            verbose: 是否输出移除过程的日志。

        Returns:
            int: 重建的边数量。
        """
        if not self._curvature_constraint_bindings:
            # 没有曲率约束需要移除
            return 0

        # 收集所有受曲率约束影响的普通边（去重，保持顺序）
        # 跳过source/target边，因为重建它们需要重新添加addSourceTarget
        # 中的专用约束，代价过高
        affected_edges = []
        seen_edge_ids = set()
        skipped_edges = []
        for edge, _binding in self._curvature_constraint_bindings:
            edge_id = id(edge)
            if edge_id not in seen_edge_ids:
                seen_edge_ids.add(edge_id)
                # 跳过source/target边
                if (self.target is not None and
                        (edge.u() == self.target or
                         edge.v() == self.target)):
                    skipped_edges.append(edge)
                    continue
                affected_edges.append(edge)

        # 记录每条受影响边的端点，然后移除并重建
        num_rebuilt = 0
        for old_edge in affected_edges:
            u_vertex = old_edge.u()
            v_vertex = old_edge.v()
            edge_name = old_edge.name()

            # 移除旧边
            self.gcs.RemoveEdge(old_edge)

            # 重建边
            new_edge = self.gcs.AddEdge(u_vertex, v_vertex, edge_name)

            # 重新添加连续性约束
            for c_con in self.contin_constraints:
                new_edge.AddConstraint(Binding[Constraint](
                    c_con, np.append(u_vertex.x(), v_vertex.x())
                ))

            # 重新添加速度约束（跳过源点边）
            if u_vertex != self.source:
                for d_con in self.deriv_constraints:
                    new_edge.AddConstraint(
                        Binding[Constraint](d_con, new_edge.xu())
                    )

                # 重新添加成本
                for cost in self.edge_costs:
                    new_edge.AddCost(Binding[Cost](cost, new_edge.xu()))

            num_rebuilt += 1

        if verbose and num_rebuilt > 0:
            msg = f"Removed curvature constraints by rebuilding {num_rebuilt} edges"
            if skipped_edges:
                msg += f" (skipped {len(skipped_edges)} target edges)"
            print(msg)

        # 清空记录（包括skipped_edges的binding）
        self._curvature_constraint_bindings.clear()
        self.curvature_constraints.clear()

        # 边重建后，_phi_updater的_prev_path_edges中的边引用已失效，必须reset
        if self._phi_updater is not None:
            self._phi_updater.reset()

        return num_rebuilt

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
            # v2模式下顶点变量末尾多了σ_e和τ_e，负索引需额外偏移2位
            time_offset = 2 if self._vertex_extended else 0
            edge.AddConstraint(edge.xv()[-(self.order + 1) - time_offset] == 0.)

        # 为目标点边添加约束
        # v2模式下顶点变量末尾多了σ_e和τ_e，负索引需额外偏移2位
        v2_offset = 2 if self._vertex_extended else 0
        for edge in target_edges:    
            # 目标点边：最后一个顶点和目标点在空间上重合
            for jj in range(self.dimension):
                edge.AddConstraint(
                    edge.xu()[-(self.dimension + self.order + 1) - v2_offset + jj] == edge.xv()[jj])

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

            # 为目标点边添加速度约束
            for d_con in self.deriv_constraints:
                edge.AddConstraint(Binding[Constraint](d_con, edge.xu()))

            # 为目标点边添加当前有效的曲率约束
            for d_con in self.curvature_constraints:
                binding = edge.AddConstraint(
                    Binding[Constraint](d_con, edge.xu())
                )
                self._curvature_constraint_bindings.append((edge, binding))

        return source_edges, target_edges

    def _build_trajectory_from_path(self, path_edges, result):
        """从路径边列表和求解结果构建 BezierTrajectory。

        Args:
            path_edges: 路径上的边列表。
            result: Drake 的 MathematicalProgramResult 对象。

        Returns:
            BezierTrajectory 对象，或 None（构建失败时）。
        """
        try:
            knots = np.zeros(self.order + 1)
            path_control_points = []
            time_control_points = []

            for edge in path_edges:
                if edge.v() == self.target:
                    knots = np.concatenate((knots, [knots[-1]]))
                    path_control_points.append(result.GetSolution(edge.xv()))
                    # v2模式下顶点变量末尾多了σ_e和τ_e，最后一个时间控制点需偏移
                    xu_sol = result.GetSolution(edge.xu())
                    last_time_idx = -1 - (2 if self._vertex_extended else 0)
                    time_control_points.append(
                        np.array([xu_sol[last_time_idx]]))
                    break

                edge_time = knots[-1] + 1.
                knots = np.concatenate(
                    (knots, np.full(self.order, edge_time)))

                # v2模式下顶点变量末尾多了σ_e和τ_e，需调整切片
                v2_trim = 2 if self._vertex_extended else 0
                xv_sol = result.GetSolution(edge.xv())
                # 空间控制点: 前 dim*(order+1) 个变量
                num_spatial = self.dimension * (self.order + 1)
                edge_path_points = np.reshape(
                    xv_sol[:num_spatial],
                    (self.dimension, self.order + 1), "F")
                # 时间控制点: 空间控制点之后、σ_e/τ_e之前
                time_start = num_spatial
                time_end = num_spatial + self.order + 1
                edge_time_points = xv_sol[time_start:time_end]

                # v2诊断：检查辅助变量值和控制点范围
                if self._vertex_extended and len(xv_sol) > time_end:
                    sigma_val = xv_sol[-2]
                    tau_val = xv_sol[-1]
                    path_range = np.ptp(edge_path_points, axis=1)
                    if np.any(path_range > 50) or abs(sigma_val) > 100 or abs(tau_val) > 100:
                        print(f"  [v2 diag] edge={edge.name()}: "
                              f"sigma={sigma_val:.4f}, tau={tau_val:.4f}, "
                              f"path_range={path_range}")

                for ii in range(self.order):
                    path_control_points.append(edge_path_points[:, ii])
                    time_control_points.append(
                        np.array([edge_time_points[ii]]))

            offset = time_control_points[0].copy()
            for ii in range(len(time_control_points)):
                time_control_points[ii] -= offset

            path_control_points = np.array(path_control_points).T
            time_control_points = np.array(time_control_points).T

            path = BsplineTrajectory(
                BsplineBasis(self.order + 1, knots), path_control_points)
            time_traj = BsplineTrajectory(
                BsplineBasis(self.order + 1, knots), time_control_points)

            return BezierTrajectory(path, time_traj)
        except Exception:
            return None

    def SolvePath(self, rounding=False, verbose=False, preprocessing=False):
        """
        求解最优路径并返回轨迹

        Args:
            rounding (bool): 是否使用舍入策略（先求解松弛问题）
            verbose (bool): 是否显示详细求解信息
            preprocessing (bool): 是否进行预处理

        Returns:
            tuple: (BezierTrajectory对象, 结果字典)
                   结果字典中包含 "all_candidate_trajectories" 列表，
                   存储所有候选舍入路径对应的轨迹，供上层按约束
                   违向量筛选。
        """
        # 调用基类方法求解GCS
        best_path, best_result, results_dict = self.solveGCS(
            rounding, preprocessing, verbose)

        if best_path is None:
            return None, results_dict

        # 构建最优轨迹
        best_trajectory = self._build_trajectory_from_path(
            best_path, best_result)

        # 构建所有候选轨迹（供上层按约束违反量筛选）
        all_candidate_trajectories = []
        all_paths_results = results_dict.get("all_rounded_paths_results", [])
        for path_edges, path_result in all_paths_results:
            if not path_result.is_success():
                continue
            traj = self._build_trajectory_from_path(path_edges, path_result)
            if traj is not None:
                all_candidate_trajectories.append(traj)

        # 确保最优轨迹在候选列表中
        if best_trajectory is not None:
            if best_trajectory not in all_candidate_trajectories:
                all_candidate_trajectories.insert(0, best_trajectory)

        results_dict["all_candidate_trajectories"] = (
            all_candidate_trajectories)

        return best_trajectory, results_dict

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
    
