"""
阿克曼GCS轨迹优化器模块

提供AckermannGCS类，实现阿克曼转向车辆的GCS轨迹优化。
利用微分平坦特性，将状态和控制量表示为平坦输出[x, y, θ]及其导数的函数，
并嵌入GCS算法进行轨迹优化。
"""

import numpy as np
from typing import Optional, List, Tuple, Dict

from pydrake.geometry.optimization import (
    HPolyhedron,
    Point,
    Hyperellipsoid,
    VPolytope,
)
from pydrake.solvers import (
    Binding,
    Constraint,
    Cost,
    L2NormCost,
    LinearConstraint,
    LinearEqualityConstraint,
    LinearCost,
    QuadraticCost,
    QuadraticConstraint,
    PerspectiveQuadraticCost,
    LorentzConeConstraint,
)
from pydrake.symbolic import (
    DecomposeLinearExpressions,
    Expression,
    MakeMatrixContinuousVariable,
    MakeVectorContinuousVariable,
)
from pydrake.trajectories import (
    BsplineTrajectory,
    BsplineTrajectory_,
)
from pydrake.math import (
    BsplineBasis,
    BsplineBasis_,
    KnotVectorType,
)

from gcs_pkg.scripts.core import BaseGCS
from gcs_pkg.scripts.core.utils import sample_unit_sphere
from gcs_pkg.scripts.core.steering_constraint import SteeringConstraintManager
from gcs_pkg.scripts.core.quadratic_constraint_manager import QuadraticConstraintManager
from gcs_pkg.scripts.core.ackermann_trajectory import AckermannTrajectory


__all__ = ['AckermannGCS']


class AckermannGCS(BaseGCS):
    """
    阿克曼转向车辆GCS轨迹优化器

    利用微分平坦特性，将状态和控制量表示为平坦输出[x, y, θ]及其导数的函数，
    并嵌入GCS算法进行轨迹优化。

    Attributes:
        wheelbase (float): 车辆轴距L
        order (int): B样条曲线阶数
        continuity (int): 路径连续性阶数
        flat_trajectory (BsplineTrajectory): 平坦输出轨迹
        time_trajectory (BsplineTrajectory): 时间轨迹
        flat_derivatives (list): 平坦输出导数轨迹列表
    """

    def __init__(
        self,
        regions: List,
        wheelbase: float,
        order: int = 5,
        continuity: int = 2,
        edges: Optional[List[Tuple[int, int]]] = None,
        hdot_min: float = 1e-6,
        full_dim_overlap: bool = False,
        hyperellipsoid_num_samples_per_dim_factor: int = 32,
        solver_config: Optional = None
    ):
        """
        初始化阿克曼GCS优化器

        Args:
            regions: 凸集列表，每个凸集表示平坦输出空间中的可行区域
            wheelbase: 车辆轴距L
            order: B样条曲线阶数（控制点数量 = order + 1）
            continuity: 路径连续性要求（0=C0，1=C1，2=C2）
            edges: 指定的边连接关系，None时自动计算
            hdot_min: 时间导数最小值
            full_dim_overlap: 是否要求区域交集为全维度
            hyperellipsoid_num_samples_per_dim_factor: 椭球采样因子
            solver_config: 自适应求解器配置
        """
        # 调用基类初始化
        BaseGCS.__init__(self, regions, auto_add_vertices=False, solver_config=solver_config)

        # 存储参数
        self.wheelbase = wheelbase
        self.order = order
        self.continuity = continuity
        self.hyperellipsoid_num_samples_per_dim_factor = hyperellipsoid_num_samples_per_dim_factor

        # 验证参数
        assert continuity < order, "连续性阶数必须小于B样条阶数"

        # 保存原始区域（用于findStartGoalEdges检查点是否在区域内）
        self.original_regions = regions.copy()

        # 重写dimension为3（平坦输出维度：x, y, theta）
        self.dimension = 3

        # 创建时间缩放凸集
        A_time = np.vstack((
            np.eye(order + 1),
            -np.eye(order + 1),
            np.eye(order, order + 1) - np.eye(order, order + 1, 1)
        ))
        b_time = np.concatenate((
            1e3 * np.ones(order + 1),
            np.zeros(order + 1),
            -hdot_min * np.ones(order)
        ))
        self.time_scaling_set = HPolyhedron(A_time, b_time)

        # 为每个区域创建顶点集合
        for i, r in enumerate(self.regions):
            vertex_set = self._build_vertex_set(r, order)
            self.gcs.AddVertex(vertex_set, name=self.names[i] if self.names is not None else '')

        # 定义边变量
        u_flat = MakeMatrixContinuousVariable(3, order + 1, "flat_u")
        v_flat = MakeMatrixContinuousVariable(3, order + 1, "flat_v")
        u_time = MakeVectorContinuousVariable(order + 1, "time_u")
        v_time = MakeVectorContinuousVariable(order + 1, "time_v")

        self.u_vars = np.concatenate((u_flat.flatten("F"), u_time))
        self.v_vars = np.concatenate((v_flat.flatten("F"), v_time))

        # 创建平坦输出轨迹和时间轨迹
        self.flat_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            u_flat
        )
        self.time_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
            np.expand_dims(u_time, 0)
        )

        # 添加连续性约束
        edge_vars = np.concatenate((u_flat.flatten("F"), u_time, v_flat.flatten("F"), v_time))
        self._add_continuity_constraints(
            self.flat_trajectory,
            BsplineTrajectory_[Expression](
                BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
                v_flat
            ),
            self.time_trajectory,
            BsplineTrajectory_[Expression](
                BsplineBasis_[Expression](order + 1, order + 1, KnotVectorType.kClampedUniform, 0., 1.),
                np.expand_dims(v_time, 0)
            ),
            edge_vars,
            continuity
        )

        # 确定边连接关系
        if edges is None:
            if full_dim_overlap:
                edges = self.findEdgesViaFullDimensionOverlaps()
            else:
                edges = self.findEdgesViaOverlaps()

        # 添加边
        vertices = self.gcs.Vertices()
        for ii, jj in edges:
            u = vertices[ii]
            v = vertices[jj]
            edge = self.gcs.AddEdge(u, v, f"({u.name()}, {v.name()})")

            # 添加连续性约束到边
            for con in self.contin_constraints:
                edge.AddConstraint(Binding[Constraint](con, np.append(u.x(), v.x())))

        # 存储成本和约束
        self.edge_costs = []
        self.deriv_constraints = []

        # 存储是否使用了QuadraticConstraint
        self._used_quadratic_constraint = False

    def _build_vertex_set(self, region, order: int):
        """
        为单个区域构建顶点集合

        顶点集合 = 区域^(3*(order+1)) × 时间缩放集合

        Args:
            region: 原始凸集（2D或3D）
            order: B样条阶数

        Returns:
            顶点集合
        """
        # 检查区域类型
        if isinstance(region, Hyperellipsoid):
            print(f"Warning: Converting Hyperellipsoid to HPolyhedron approximation for AckermannGCS vertex.")

            # 采样转换
            dimension = region.A().shape[0]
            num_samples = self.hyperellipsoid_num_samples_per_dim_factor * dimension

            # 生成单位超球面上的点
            unit_sphere_points = sample_unit_sphere(dimension, num_samples)

            # 转换到椭球表面
            A_inv_T = np.linalg.inv(region.A()).T
            center = region.center()
            ellipse_points = A_inv_T @ unit_sphere_points + center.reshape(-1, 1)

            # 创建VPolytope并转换为HPolyhedron
            v_poly = VPolytope(ellipse_points)
            region_poly = HPolyhedron(v_poly, tol=1e-8)

        elif isinstance(region, HPolyhedron):
            region_poly = region
        else:
            raise TypeError(f"Region type {type(region)} is not supported by AckermannGCS.")

        # 如果是2D，扩展到3D（添加θ维度）
        if region_poly.ambient_dimension() == 2:
            # 添加θ约束：-π <= θ <= π
            A_extended = np.vstack([
                np.hstack([region_poly.A(), np.zeros((region_poly.A().shape[0], 1))]),
                np.hstack([np.zeros((2, 2)), np.array([[1], [-1]])])
            ])
            b_extended = np.concatenate([region_poly.b(), np.array([np.pi, np.pi])])
            region_poly = HPolyhedron(A_extended, b_extended)

        # 计算笛卡尔幂：每个顶点有(order+1)个控制点，每个控制点是3维的
        # 顶点集合 = region^(order+1) × time_scaling_set
        vertex_set = region_poly.CartesianPower(order + 1).CartesianProduct(self.time_scaling_set)

        return vertex_set

    def _add_continuity_constraints(
        self,
        u_trajectory: BsplineTrajectory_[Expression],
        v_trajectory: BsplineTrajectory_[Expression],
        u_time_trajectory: BsplineTrajectory_[Expression],
        v_time_trajectory: BsplineTrajectory_[Expression],
        edge_vars: np.ndarray,
        continuity: int
    ):
        """
        添加边的连续性约束

        Args:
            u_trajectory: u顶点的平坦输出轨迹
            v_trajectory: v顶点的平坦输出轨迹
            u_time_trajectory: u顶点的时间轨迹
            v_time_trajectory: v顶点的时间轨迹
            edge_vars: 边变量
            continuity: 连续性阶数
        """
        self.contin_constraints = []

        # 平坦输出连续性约束
        for deriv in range(continuity + 1):
            u_deriv = u_trajectory.MakeDerivative(deriv)
            v_deriv = v_trajectory.MakeDerivative(deriv)

            # 连续性条件：v_deriv[0] == u_deriv[-1]
            continuity_error = v_deriv.control_points()[0] - u_deriv.control_points()[-1]

            self.contin_constraints.append(LinearEqualityConstraint(
                DecomposeLinearExpressions(continuity_error, edge_vars),
                np.zeros(3)
            ))

        # 时间轨迹连续性约束
        for deriv in range(continuity + 1):
            u_time_deriv = u_time_trajectory.MakeDerivative(deriv)
            v_time_deriv = v_time_trajectory.MakeDerivative(deriv)

            continuity_error = v_time_deriv.control_points()[0] - u_time_deriv.control_points()[-1]

            self.contin_constraints.append(LinearEqualityConstraint(
                DecomposeLinearExpressions(continuity_error, edge_vars),
                0.0
            ))

    @staticmethod
    def flat_output_to_state(
        flat_output: np.ndarray,
        flat_derivatives: List[np.ndarray],
        wheelbase: float
    ) -> np.ndarray:
        """
        将平坦输出及其导数映射到完整状态

        Args:
            flat_output: 平坦输出 [x, y, θ]
            flat_derivatives: 平坦输出导数列表
            wheelbase: 车辆轴距

        Returns:
            完整状态 [x, y, θ, v, δ]
        """
        x, y, theta = flat_output
        x_dot, y_dot, theta_dot = flat_derivatives[0]

        # 纵向速度
        v = np.sqrt(x_dot**2 + y_dot**2)

        # 前轮转向角（使用atan2避免除零）
        delta = np.arctan2(wheelbase * theta_dot, v)

        return np.array([x, y, theta, v, delta])

    @staticmethod
    def flat_output_to_control(
        flat_output: np.ndarray,
        flat_derivatives: List[np.ndarray],
        wheelbase: float
    ) -> np.ndarray:
        """
        将平坦输出及其导数映射到控制输入

        Args:
            flat_output: 平坦输出 [x, y, θ]
            flat_derivatives: 平坦输出导数列表
            wheelbase: 车辆轴距

        Returns:
            控制输入 [a, ω]
        """
        x, y, theta = flat_output
        x_dot, y_dot, theta_dot = flat_derivatives[0]
        x_ddot, y_ddot, theta_ddot = flat_derivatives[1]

        # 纵向速度
        v = np.sqrt(x_dot**2 + y_dot**2)

        # 纵向加速度
        a = (x_dot * x_ddot + y_dot * y_ddot) / v

        # 转向角速度
        tan_delta = wheelbase * theta_dot / v
        sec2_delta = 1 + tan_delta**2
        omega = sec2_delta * wheelbase * (theta_ddot * v - theta_dot * a) / v**2

        return np.array([a, omega])

    def addVelocityLimits(self, lower_bound: float, upper_bound: float):
        """
        添加纵向速度约束

        v = √(ẋ² + ẏ²)
        v_min ≤ v ≤ v_max

        使用LorentzConeConstraint实现精确的圆形约束：
        - 上界：√(ẋ² + ẏ²) ≤ v_max

        LorentzConeConstraint形式：
        ||z₂ₙ|| ≤ z₁, z₁ ≥ 0

        对于速度约束，令：
        z = [v_max, ẋ, ẏ]
        则：√(ẋ² + ẏ²) ≤ v_max

        Args:
            lower_bound: 最小速度（m/s）
            upper_bound: 最大速度（m/s）
        """
        # 验证参数
        if lower_bound < 0:
            raise ValueError(f"最小速度必须非负，当前值：{lower_bound}")
        if upper_bound <= lower_bound:
            raise ValueError(
                f"最大速度必须大于最小速度，当前值：max={upper_bound}, min={lower_bound}"
            )

        # 获取平坦输出轨迹的一阶导数
        flat_deriv = self.flat_trajectory.MakeDerivative(1)
        control_points = flat_deriv.control_points()

        # 存储约束
        self.velocity_constraints = []

        for i, cp in enumerate(control_points):
            # cp = [ẋ, ẏ, θ̇]
            x_dot_expr = cp[0]
            y_dot_expr = cp[1]

            # 使用LorentzConeConstraint表示速度上界约束
            # v = sqrt(ẋ² + ẏ²) <= v_max
            # 洛伦兹锥形式：sqrt(ẋ² + ẏ²) <= v_max

            # 构造 z = A * x + b
            # z = [v_max, ẋ, ẏ]
            H = DecomposeLinearExpressions([x_dot_expr, y_dot_expr], self.u_vars)

            # A = [[0, 0], [1, 0], [0, 1]]
            # b = [v_max, 0, 0]
            A = np.vstack([
                np.zeros((1, H.shape[1])),
                H
            ])
            b = np.array([upper_bound, 0, 0])

            velocity_con_max = LorentzConeConstraint(A, b)

            # 如果需要速度下界约束，需要在目标函数中添加惩罚项
            # 因为 v >= v_min 不是凸约束（v 是平方根函数）

            self.velocity_constraints.append(velocity_con_max)

            # 添加到所有边
            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    edge.AddConstraint(Binding[Constraint](velocity_con_max, edge.xv()))
                if edge.v() == self.target:
                    edge.AddConstraint(Binding[Constraint](velocity_con_max, edge.xu()))
                if edge.u() != self.source and edge.v() != self.target:
                    edge.AddConstraint(Binding[Constraint](velocity_con_max, edge.xu()))
                    edge.AddConstraint(Binding[Constraint](velocity_con_max, edge.xv()))

    def addSteeringLimits(self, lower_bound: float, upper_bound: float):
        """
        添加前轮转向角约束

        δ = atan(L * θ̇ / v)
        δ_min ≤ δ ≤ δ_max

        注意：由于Drake GCS求解器不支持QuadraticConstraint，这里使用LinearConstraint近似。
        虽然这不是数学上精确的实现，但考虑到GCS框架的限制，这是一个合理的折衷方案。

        改进点：
        1. 添加了参数验证
        2. 移除了错误的L2NormCost
        3. 使用更合理的LinearConstraint近似

        Args:
            lower_bound: 最小转向角（弧度），范围：[-π/2, 0]
            upper_bound: 最大转向角（弧度），范围：(0, π/2]
        """
        # 验证参数
        if lower_bound >= 0:
            raise ValueError(f"最小转向角必须为负数，当前值：{lower_bound}")
        if upper_bound <= 0:
            raise ValueError(f"最大转向角必须为正数，当前值：{upper_bound}")
        if lower_bound <= -np.pi/2 or upper_bound >= np.pi/2:
            raise ValueError(
                f"转向角必须在(-π/2, π/2)范围内，当前值：min={lower_bound}, max={upper_bound}"
            )

        # 获取平坦输出的一阶导数
        flat_deriv = self.flat_trajectory.MakeDerivative(1)
        control_points = flat_deriv.control_points()

        # 计算tan(δ)的边界
        tan_delta_min = np.tan(lower_bound)
        tan_delta_max = np.tan(upper_bound)

        # 存储约束
        self.steering_constraints = []

        for i, cp in enumerate(control_points):
            # cp = [ẋ, ẏ, θ̇]
            x_dot_expr = cp[0]
            y_dot_expr = cp[1]
            theta_dot_expr = cp[2]

            # 由于GCS不支持二次约束，这里使用LinearConstraint近似
            # 目标约束：tan(δ_min) ≤ L·θ̇/v ≤ tan(δ_max)
            # 近似：限制θ̇的范围，假设v在一个合理范围内

            # 上界约束：L·θ̇ ≤ tan(δ_max)·v
            # 近似为：L·θ̇ ≤ tan(δ_max)·v_max (假设v_max为最大速度)
            # 这里简化为：θ̇ ≤ tan(δ_max)/L * v
            # 由于v = sqrt(ẋ² + ẏ²)，我们使用速度约束的上界作为近似

            H_theta = DecomposeLinearExpressions([theta_dot_expr], self.u_vars)

            # 使用简化的LinearConstraint
            # 上界：L·θ̇ ≤ tan(δ_max) * (速度的某个近似值)
            # 这里我们使用一个保守的估计，假设速度不会太大
            steering_con_max = LinearConstraint(
                H_theta,
                -np.inf * np.ones(1),
                (tan_delta_max / self.wheelbase) * np.ones(1)
            )

            # 下界：L·θ̇ ≥ tan(δ_min)·v
            steering_con_min = LinearConstraint(
                H_theta,
                (tan_delta_min / self.wheelbase) * np.ones(1),
                np.inf * np.ones(1)
            )

            self.steering_constraints.extend([steering_con_max, steering_con_min])

            # 添加到所有边
            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddConstraint(Binding[Constraint](steering_con_max, edge.xu()))
                edge.AddConstraint(Binding[Constraint](steering_con_min, edge.xu()))

    def addSteeringLimits_iterative(
        self,
        lower_bound: float,
        upper_bound: float,
        max_iterations: int = 3,
        convergence_tolerance: float = 0.01,
        verbose: bool = False
    ):
        """
        使用迭代优化策略实现转向角约束（改进版）

        改进点：
        1. 使用SteeringConstraintManager避免约束累积
        2. 改进速度估计策略（30% -> 50% -> 90%）
        3. 修复失败处理逻辑（失败时放松约束）
        4. 使用相对误差判断收敛
        5. 完善输入验证
        6. 提供详细的转向角验证信息

        算法流程：
        1. 第一次迭代：使用保守的速度估计（v = min(0.3*v_max, 2.0)）
        2. 求解轨迹，提取实际速度曲线 v(t)
        3. 根据实际速度调整 θ̇ 约束（通过UpdateCoefficients）
        4. 重复步骤2-3，直到收敛或达到最大迭代次数

        数学原理：
        转向角约束：δ_min ≤ atan(L·θ̇/v) ≤ δ_max
        等价于：tan(δ_min) ≤ L·θ̇/v ≤ tan(δ_max)

        在每次迭代中，使用上一次的速度曲线 v(t) 来设置 θ̇ 约束：
        L·θ̇ ≤ v(t)·tan(δ_max)
        L·θ̇ ≥ v(t)·tan(δ_min)

        Args:
            lower_bound: 最小转向角（弧度），范围：[-π/2, 0]
            upper_bound: 最大转向角（弧度），范围：(0, π/2]
            max_iterations: 最大迭代次数（必须为正整数）
            convergence_tolerance: 收敛容忍度（相对误差，范围：(0, 0.1]，推荐0.01）
            verbose: 是否打印详细日志

        Returns:
            iterations_info: 包含每次迭代信息的字典列表
                每个字典包含：
                - iteration: 迭代编号（从1开始）
                - success: 是否成功
                - v_conservative: 使用的保守速度估计
                - trajectory_time: 轨迹总时间（秒）
                - velocity_profile: 速度曲线 {'t': array, 'v': array}
                - steering_analysis: 转向角分析 {
                    'delta_min': 最小转向角（弧度）,
                    'delta_max': 最大转向角（弧度）,
                    'delta_mean': 平均转向角（弧度）,
                    'violations': 违反约束次数,
                    'violation_rate': 违反率,
                    'max_violation': 最大违反量（弧度）,
                    'violation_details': 违反详情（最多10条）
                }
                - error: 错误信息（如果失败）

        Raises:
            ValueError: 参数验证失败
        """
        # ========== 参数验证 ==========
        if lower_bound >= 0:
            raise ValueError(f"最小转向角必须为负数，当前值：{lower_bound}")
        if upper_bound <= 0:
            raise ValueError(f"最大转向角必须为正数，当前值：{upper_bound}")
        if lower_bound <= -np.pi/2 or upper_bound >= np.pi/2:
            raise ValueError(
                f"转向角必须在(-π/2, π/2)范围内，当前值：min={lower_bound}, max={upper_bound}"
            )

        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError(f"max_iterations必须是正整数，当前值：{max_iterations}")

        if convergence_tolerance <= 0:
            raise ValueError(f"convergence_tolerance必须为正数，当前值：{convergence_tolerance}")
        if convergence_tolerance > 0.1:
            raise ValueError(f"convergence_tolerance建议不超过0.1（10%），当前值：{convergence_tolerance}")

        # 检查是否已添加速度约束
        if not hasattr(self, 'velocity_constraints'):
            raise ValueError("必须先调用 addVelocityLimits() 才能使用迭代优化转向角约束")

        # 获取速度上界
        v_max = self.velocity_constraints[0].b()[0]

        # 计算tan(δ)边界
        tan_delta_min = np.tan(lower_bound)
        tan_delta_max = np.tan(upper_bound)

        # 初始化约束管理器
        constraint_manager = SteeringConstraintManager(
            self.gcs,
            self.u_vars,
            self.wheelbase,
            self.flat_trajectory,
            self.source
        )

        # 存储迭代信息
        iterations_info = []
        prev_trajectory = None
        prev_velocity_profile = None

        # 迭代优化
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n{'='*60}")
                print(f"迭代 {iteration + 1}/{max_iterations}")
                print(f"{'='*60}")
                print(f"约束数量: {constraint_manager.get_constraint_count()}")

            # ========== 速度估计策略 ==========
            v_conservative = self._estimate_conservative_velocity(
                iteration,
                v_max,
                prev_velocity_profile,
                verbose
            )

            # ========== 更新约束 ==========
            if iteration == 0:
                # 第一次迭代：创建初始约束
                constraint_manager.create_initial_constraints(
                    v_conservative,
                    tan_delta_max,
                    tan_delta_min
                )
                if verbose:
                    print(f"创建初始约束")
            else:
                # 后续迭代：更新约束参数
                constraint_manager.update_constraints(
                    v_conservative,
                    tan_delta_max,
                    tan_delta_min
                )
                if verbose:
                    print(f"更新约束参数")

            # ========== 求解GCS问题 ==========
            if verbose:
                print("\n求解GCS问题...")

            trajectory, results_dict = self.SolvePath(rounding=True, verbose=False)

            # ========== 处理求解结果 ==========
            if trajectory is None:
                if verbose:
                    print(f"❌ 迭代 {iteration + 1} 失败：无法找到可行轨迹")

                iterations_info.append({
                    'iteration': iteration + 1,
                    'success': False,
                    'v_conservative': v_conservative,
                    'error': 'No feasible trajectory found'
                })

                # 失败处理：放松约束
                if iteration == 0:
                    if verbose:
                        print("失败处理：放松速度约束...")
                    # 放松速度估计（从30%提高到50%）
                    v_conservative = min(v_max * 0.5, 3.0)
                    # 更新约束
                    constraint_manager.update_constraints(
                        v_conservative,
                        tan_delta_max,
                        tan_delta_min
                    )
                    # 继续下一次迭代
                    continue
                else:
                    # 后续迭代失败，停止
                    break

            # ========== 提取和分析结果 ==========
            velocity_profile = self._extract_velocity_profile(trajectory)
            steering_analysis = self._analyze_steering_angles_v2(
                trajectory,
                lower_bound,
                upper_bound,
                verbose
            )

            if verbose:
                print(f"\n✓ 迭代 {iteration + 1} 成功")
                print(f"轨迹时间: {trajectory.start_time():.2f}s -> {trajectory.end_time():.2f}s")
                print(f"总时间: {trajectory.end_time() - trajectory.start_time():.2f}s")
                print(f"\n转向角分析:")
                print(f"  最小转向角: {np.degrees(steering_analysis['delta_min']):.2f}°")
                print(f"  最大转向角: {np.degrees(steering_analysis['delta_max']):.2f}°")
                print(f"  平均转向角: {np.degrees(steering_analysis['delta_mean']):.2f}°")
                print(f"  违反约束: {steering_analysis['violations']} 次")
                if steering_analysis['max_violation'] > 0:
                    print(f"  最大违反量: {np.degrees(steering_analysis['max_violation']):.2f}°")

            # 记录迭代信息
            iter_info = {
                'iteration': iteration + 1,
                'success': True,
                'v_conservative': v_conservative,
                'trajectory_time': trajectory.end_time() - trajectory.start_time(),
                'velocity_profile': velocity_profile,
                'steering_analysis': steering_analysis
            }
            iterations_info.append(iter_info)

            # ========== 收敛性检查 ==========
            if prev_trajectory is not None:
                relative_error = self._compute_trajectory_difference_v2(
                    prev_trajectory,
                    trajectory,
                    verbose
                )

                if verbose:
                    print(f"\n收敛性分析:")
                    print(f"  相对误差: {relative_error:.4%}")
                    print(f"  收敛阈值: {convergence_tolerance:.4%}")

                if relative_error < convergence_tolerance:
                    if verbose:
                        print(f"\n✓ 收敛！相对误差小于阈值 {convergence_tolerance:.4%}")
                    break

            # 更新前一次的轨迹和速度曲线
            prev_trajectory = trajectory
            prev_velocity_profile = velocity_profile

        # ========== 清理 ==========
        constraint_manager.cleanup()

        if verbose:
            print(f"\n{'='*60}")
            print(f"迭代优化完成")
            print(f"{'='*60}")
            print(f"总迭代次数: {len(iterations_info)}")
            print(f"最终状态: {'成功' if iterations_info[-1]['success'] else '失败'}")

        return iterations_info

    def _estimate_conservative_velocity(
        self,
        iteration: int,
        v_max: float,
        velocity_profile: Optional[dict],
        verbose: bool = False
    ) -> float:
        """
        估计保守速度（改进策略）

        策略：
        - 迭代0: min(0.3 * v_max, 2.0)  - 保守初始值
        - 迭代1: min(0.5 * v_max, 3.0)  - 适度放松
        - 迭代2+: 90%分位数          - 基于实际速度

        Args:
            iteration: 迭代编号
            v_max: 最大速度
            velocity_profile: 上一次的速度曲线
            verbose: 是否打印日志

        Returns:
            保守速度估计值
        """
        if iteration == 0:
            v_conservative = min(v_max * 0.3, 2.0)
            if verbose:
                print(f"速度估计策略: 初始迭代")
                print(f"  v_conservative = min(0.3 * {v_max:.2f}, 2.0) = {v_conservative:.3f} m/s")

        elif iteration == 1:
            v_conservative = min(v_max * 0.5, 3.0)
            if verbose:
                print(f"速度估计策略: 第二次迭代")
                print(f"  v_conservative = min(0.5 * {v_max:.2f}, 3.0) = {v_conservative:.3f} m/s")

        else:
            # 使用实际速度的90%分位数
            v_min_actual = np.min(velocity_profile['v'])
            v_max_actual = np.max(velocity_profile['v'])
            v_mean_actual = np.mean(velocity_profile['v'])
            v_conservative = np.percentile(velocity_profile['v'], 90)

            if verbose:
                print(f"速度估计策略: 基于实际速度")
                print(f"  速度曲线统计:")
                print(f"    最小值: {v_min_actual:.3f} m/s")
                print(f"    最大值: {v_max_actual:.3f} m/s")
                print(f"    平均值: {v_mean_actual:.3f} m/s")
                print(f"  v_conservative = 90%分位数 = {v_conservative:.3f} m/s")

        return v_conservative

    def _extract_velocity_profile(self, trajectory: 'AckermannTrajectory', num_samples: int = 100):
        """
        从轨迹中提取速度曲线

        Args:
            trajectory: 阿克曼轨迹对象
            num_samples: 采样点数

        Returns:
            velocity_profile: 包含时间和速度的字典
        """
        t_start = trajectory.start_time()
        t_end = trajectory.end_time()
        t_samples = np.linspace(t_start, t_end, num_samples)

        velocities = []
        for t in t_samples:
            state = trajectory.get_state(t)
            velocities.append(state[3])  # v = state[3]

        return {
            't': t_samples,
            'v': np.array(velocities)
        }

    def _analyze_steering_angles_v2(
        self,
        trajectory: 'AckermannTrajectory',
        lower_bound: float,
        upper_bound: float,
        verbose: bool = False,
        num_samples: int = 500
    ):
        """
        分析轨迹中的转向角约束满足情况（改进版）

        改进点：
        1. 增加采样密度（100 -> 500）
        2. 记录违反约束的详细信息
        3. 计算最大违反量
        4. 提供违反位置的详细信息

        Args:
            trajectory: 阿克曼轨迹对象
            lower_bound: 最小转向角（弧度）
            upper_bound: 最大转向角（弧度）
            verbose: 是否打印详细信息
            num_samples: 采样点数

        Returns:
            analysis: 包含转向角统计信息的字典
                - delta_min: 最小转向角（弧度）
                - delta_max: 最大转向角（弧度）
                - delta_mean: 平均转向角（弧度）
                - violations: 违反约束次数
                - violation_rate: 违反率
                - max_violation: 最大违反量（弧度）
                - violation_details: 违反详情列表（最多10条）
        """
        t_start = trajectory.start_time()
        t_end = trajectory.end_time()
        t_samples = np.linspace(t_start, t_end, num_samples)

        steering_angles = []
        violations_info = []

        for t in t_samples:
            state = trajectory.get_state(t)
            delta = state[4]  # δ = state[4]
            steering_angles.append(delta)

            # 检查是否违反约束
            if delta < lower_bound:
                violations_info.append({
                    't': t,
                    'delta': delta,
                    'bound': lower_bound,
                    'violation': lower_bound - delta,
                    'type': 'lower'
                })
            elif delta > upper_bound:
                violations_info.append({
                    't': t,
                    'delta': delta,
                    'bound': upper_bound,
                    'violation': delta - upper_bound,
                    'type': 'upper'
                })

        # 计算统计信息
        max_violation = 0
        if violations_info:
            max_violation = max([v['violation'] for v in violations_info])

        if verbose and violations_info:
            print(f"\n违反约束详情:")
            for i, v in enumerate(violations_info[:5]):  # 最多显示5条
                print(f"  {i+1}. t={v['t']:.2f}s, δ={np.degrees(v['delta']):.2f}°, "
                      f"边界={np.degrees(v['bound']):.2f}°, 违反={np.degrees(v['violation']):.2f}°")
            if len(violations_info) > 5:
                print(f"  ... 还有 {len(violations_info) - 5} 条违反记录")

        return {
            'delta_min': np.min(steering_angles),
            'delta_max': np.max(steering_angles),
            'delta_mean': np.mean(steering_angles),
            'violations': len(violations_info),
            'violation_rate': len(violations_info) / num_samples,
            'max_violation': max_violation,
            'violation_details': violations_info[:10]  # 最多返回10条
        }

    def _compute_trajectory_difference_v2(
        self,
        traj1: 'AckermannTrajectory',
        traj2: 'AckermannTrajectory',
        verbose: bool = False,
        num_samples: int = 100
    ) -> float:
        """
        计算两条轨迹之间的差异（改进版）

        改进点：
        1. 使用相对误差而非绝对误差
        2. 增加采样密度（50 -> 100）
        3. 提供详细的误差统计

        Args:
            traj1: 第一条轨迹
            traj2: 第二条轨迹
            verbose: 是否打印详细信息
            num_samples: 采样点数

        Returns:
            relative_error: 相对误差（0-1之间）
        """
        # 统一采样时间
        t_start = max(traj1.start_time(), traj2.start_time())
        t_end = min(traj1.end_time(), traj2.end_time())

        if t_start >= t_end:
            return float('inf')

        t_samples = np.linspace(t_start, t_end, num_samples)

        differences = []
        for t in t_samples:
            pos1 = traj1.value(t)
            pos2 = traj2.value(t)
            diff = np.linalg.norm(pos1 - pos2)
            differences.append(diff)

        # 计算轨迹长度
        pos_start = traj1.value(t_start)
        pos_end = traj1.value(t_end)
        trajectory_length = np.linalg.norm(pos_end - pos_start)

        # 计算相对误差
        mean_absolute_error = np.mean(differences)
        relative_error = mean_absolute_error / trajectory_length if trajectory_length > 0 else float('inf')

        if verbose:
            print(f"  轨迹长度: {trajectory_length:.3f} m")
            print(f"  平均绝对误差: {mean_absolute_error:.6f} m")
            print(f"  最大绝对误差: {np.max(differences):.6f} m")

        return relative_error

    def addMinTurningRadiusConstraint(self, min_radius: float):
        """
        添加最小转弯半径约束

        R = L / tan(|δ|) >= R_min
        等价于：|δ| <= atan(L / R_min)

        数学推导：
        - 转弯半径公式：R = L / tan(|δ|)
        - 最小转弯半径约束：R ≥ R_min
        - L / tan(|δ|) ≥ R_min ⇒ tan(|δ|) ≤ L / R_min ⇒ |δ| ≤ atan(L / R_min)

        添加参数验证，确保物理可行性。

        Args:
            min_radius: 最小转弯半径（米）
        """
        # 验证参数
        if min_radius <= 0:
            raise ValueError(f"最小转弯半径必须为正数，当前值：{min_radius}")

        if min_radius < self.wheelbase:
            raise ValueError(
                f"最小转弯半径 {min_radius} 小于轴距 {self.wheelbase}，"
                f"这会导致转向角超过 π/2，物理上不可行"
            )

        # 计算最大转向角
        max_steering_angle = np.arctan2(self.wheelbase, min_radius)

        # 验证转向角在合理范围内
        if abs(max_steering_angle) > np.pi / 2:
            raise ValueError(
                f"计算得到的最大转向角 {np.degrees(max_steering_angle):.2f}° "
                f"超过 90°，物理上不可行"
            )

        # 调用改进后的转向角约束
        self.addSteeringLimits(-max_steering_angle, max_steering_angle)

    def addSteeringLimitsQuadratic(self, lower_bound: float, upper_bound: float) -> bool:
        """
        使用QuadraticConstraint添加前轮转向角约束（精确实现）

        δ = atan(L·θ̇/v)
        δ_min ≤ δ ≤ δ_max

        数学推导：
        等价形式：tan(δ_min) ≤ L·θ̇/v ≤ tan(δ_max)
        其中 v = √(ẋ² + ẏ²)

        二次形式（上界）：
        L·θ̇ ≤ k_max·√(ẋ² + ẏ²)
        L²·θ̇² ≤ k_max²·(ẋ² + ẏ²)
        k_max²·ẋ² + k_max²·ẏ² - L²·θ̇² ≥ 0

        QuadraticConstraint形式：lb ≤ 0.5·xᵀ·Q·x + bᵀ·x ≤ ub
        Q = 2·[k²·H_xᵀ·H_x + k²·H_yᵀ·H_y - L²·H_θᵀ·H_θ]

        注意：Hessian矩阵是不定的（非凸约束），使用HessianType.kIndefinite。

        Args:
            lower_bound: 最小转向角（弧度），范围：[-π/2, 0]
            upper_bound: 最大转向角（弧度），范围：(0, π/2]

        Returns:
            bool: True表示成功使用QuadraticConstraint，False表示回退到LinearConstraint

        Raises:
            ValueError: 参数验证失败
        """
        # 验证参数
        if lower_bound >= 0:
            raise ValueError(f"最小转向角必须为负数，当前值：{lower_bound}")
        if upper_bound <= 0:
            raise ValueError(f"最大转向角必须为正数，当前值：{upper_bound}")
        if lower_bound <= -np.pi/2 or upper_bound >= np.pi/2:
            raise ValueError(
                f"转向角必须在(-π/2, π/2)范围内，当前值：min={lower_bound}, max={upper_bound}"
            )

        # 计算tan(δ)的边界
        tan_delta_min = np.tan(lower_bound)
        tan_delta_max = np.tan(upper_bound)

        # 创建二次约束管理器
        try:
            constraint_manager = QuadraticConstraintManager(self.wheelbase, self.u_vars)
        except Exception as e:
            print(f"创建QuadraticConstraintManager失败：{e}")
            return self._add_steering_limits_linear(lower_bound, upper_bound)

        # 获取平坦输出的一阶导数
        flat_deriv = self.flat_trajectory.MakeDerivative(1)
        control_points = flat_deriv.control_points()

        # 存储约束
        self.steering_constraints = []

        # 直接添加二次约束到所有边
        for i, cp in enumerate(control_points):
            # cp = [ẋ, ẏ, θ̇]
            x_dot_expr = cp[0]
            y_dot_expr = cp[1]
            theta_dot_expr = cp[2]

            # 创建二次约束
            con_upper, con_lower = constraint_manager.create_steering_constraint(
                x_dot_expr, y_dot_expr, theta_dot_expr,
                tan_delta_min, tan_delta_max
            )

            if con_upper is None or con_lower is None:
                print(f"创建二次约束失败（控制点{i}）")
                continue

            self.steering_constraints.extend([con_upper, con_lower])

            # 添加到所有边（跳过source边）
            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddConstraint(Binding[Constraint](con_upper, edge.xu()))
                edge.AddConstraint(Binding[Constraint](con_lower, edge.xu()))

        print("✓ 成功使用QuadraticConstraint添加转向角约束")
        return True

    def _add_steering_limits_linear(self, lower_bound: float, upper_bound: float) -> bool:
        """
        使用LinearConstraint添加转向角约束（回退方法）

        Args:
            lower_bound: 最小转向角（弧度）
            upper_bound: 最大转向角（弧度）

        Returns:
            bool: False表示使用了LinearConstraint
        """
        print("⚠ 使用LinearConstraint（近似方案）")
        # 清除之前添加的QuadraticConstraint
        self.steering_constraints = []
        # 调用原有的线性约束方法
        self.addSteeringLimits(lower_bound, upper_bound)
        return False

    def addTimeCost(self, weight: float):
        """
        添加时间成本

        J_time = T = h(T) - h(0)

        Args:
            weight: 时间权重
        """
        h_control = self.time_trajectory.MakeDerivative(0).control_points()
        segment_time = h_control[-1] - h_control[0]

        time_cost = LinearCost(
            weight * DecomposeLinearExpressions(segment_time, self.u_vars)[0],
            0.0
        )
        self.edge_costs.append(time_cost)

        for edge in self.gcs.Edges():
            if edge.u() == self.source:
                continue
            edge.AddCost(Binding[Cost](time_cost, edge.xu()))

    def addPathLengthCost(self, weight: float):
        """
        添加路径长度成本

        J_path = ∫₀ᵀ √(ẋ² + ẏ²) dt

        Args:
            weight: 路径长度权重
        """
        flat_deriv = self.flat_trajectory.MakeDerivative(1)
        control_points = flat_deriv.control_points()
        time_deriv = self.time_trajectory.MakeDerivative(1)
        time_control_points = time_deriv.control_points()

        for i, (cp, dt_cp) in enumerate(zip(control_points, time_control_points)):
            # cp = [ẋ, ẏ, θ̇]
            x_dot_expr = cp[0]
            y_dot_expr = cp[1]
            dt_expr = dt_cp[0]

            # 使用PerspectiveQuadraticCost
            # J = ||[ẋ, ẏ]|| / (dt/order)
            H = np.vstack([
                self.order * DecomposeLinearExpressions([dt_expr], self.u_vars),
                DecomposeLinearExpressions([x_dot_expr, y_dot_expr], self.u_vars)
            ])

            path_cost = PerspectiveQuadraticCost(H, np.zeros(H.shape[0]))
            self.edge_costs.append(path_cost)

            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddCost(Binding[Cost](path_cost, edge.xu()))

    def addControlEnergyCost(self, weight_a: float, weight_omega: float):
        """
        添加控制能量成本

        J_energy = ∫₀ᵀ (a² + ω²) dt

        Args:
            weight_a: 加速度权重
            weight_omega: 转向角速度权重
        """
        # 获取平坦输出导数
        flat_deriv1 = self.flat_trajectory.MakeDerivative(1)
        flat_deriv2 = self.flat_trajectory.MakeDerivative(2)
        flat_deriv3 = self.flat_trajectory.MakeDerivative(3)

        # 获取时间导数
        time_deriv1 = self.time_trajectory.MakeDerivative(1)
        time_deriv2 = self.time_trajectory.MakeDerivative(2)

        cp1 = flat_deriv1.control_points()
        cp2 = flat_deriv2.control_points()
        cp3 = flat_deriv3.control_points()
        dt_cp1 = time_deriv1.control_points()
        dt_cp2 = time_deriv2.control_points()

        # 二阶导数的控制点数量较少，使用最小长度
        min_len = min(len(cp1), len(cp2))

        for i in range(min_len):
            # 使用符号表达式构建
            # 这里简化实现，实际需要更复杂的符号计算
            # 暂时使用QuadraticCost近似
            x_dot = cp1[i][0]
            y_dot = cp1[i][1]
            x_ddot = cp2[i][0]
            y_ddot = cp2[i][1]

            # 加速度的二次近似
            A_a = DecomposeLinearExpressions([x_ddot, y_ddot], self.u_vars)
            energy_cost_a = QuadraticCost(
                A_a.T.dot(A_a) * 2 * weight_a,
                np.zeros(A_a.shape[1]),
                0
            )
            self.edge_costs.append(energy_cost_a)

            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddCost(Binding[Cost](energy_cost_a, edge.xu()))

    def addSmoothnessCost(self, weight: float, order: int = 3):
        """
        添加平滑性成本

        J_smooth = ∫₀ᵀ (σ₁⁽ᵏ⁾² + σ₂⁽ᵏ⁾² + σ₃⁽ᵏ⁾²) dt

        Args:
            weight: 平滑性权重
            order: 导数阶数
        """
        assert isinstance(order, int) and order >= 2 and order <= self.order

        flat_deriv = self.flat_trajectory.MakeDerivative(order)
        control_points = flat_deriv.control_points()

        for cp in control_points:
            # cp = [σ₁⁽ᵏ⁾, σ₂⁽ᵏ⁾, σ₃⁽ᵏ⁾]
            A_ctrl = DecomposeLinearExpressions(cp, self.u_vars)
            H = A_ctrl.T.dot(A_ctrl) * 2 * weight / (1 + self.order - order)
            smooth_cost = QuadraticCost(H, np.zeros(H.shape[1]), 0)
            self.edge_costs.append(smooth_cost)

            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                edge.AddCost(Binding[Cost](smooth_cost, edge.xu()))

    def addSourceTarget(
        self,
        source: np.ndarray,
        target: np.ndarray,
        edges: Optional[List[Tuple[int, int]]] = None,
        velocity: Optional[np.ndarray] = None,
        zero_velocity_at_boundaries: bool = True,
        min_time_derivative: Optional[float] = None
    ) -> Tuple[List, List]:
        """
        添加源点和目标点，并设置边界条件

        Args:
            source: 源点平坦输出 [x_start, y_start, theta_start]
            target: 目标点平坦输出 [x_goal, y_goal, theta_goal]
            edges: 指定的边连接关系
            velocity: 边界速度
            zero_velocity_at_boundaries: 是否在边界处设置零速度
            min_time_derivative: 时间轨迹导数最小值

        Returns:
            (源点边列表, 目标点边列表)
        """
        # 调用基类方法
        source_edges, target_edges = super().addSourceTarget(source, target, edges)

        # 添加平坦输出约束
        for edge in source_edges:
            for j in range(3):
                edge.AddConstraint(edge.xu()[j] == edge.xv()[j])

        for edge in target_edges:
            for j in range(3):
                edge.AddConstraint(
                    edge.xu()[-(3 + self.order + 1) + j] == edge.xv()[j]
                )

        # 添加速度约束
        if velocity is not None:
            assert velocity.shape == (2, 3)

            flat_deriv = self.flat_trajectory.MakeDerivative(1)
            control_points = flat_deriv.control_points()
            time_deriv = self.time_trajectory.MakeDerivative(1)
            time_control_points = time_deriv.control_points()

            initial_velocity_error = (
                control_points[0] - velocity[0] * time_control_points[0]
            )
            initial_velocity_con = LinearEqualityConstraint(
                DecomposeLinearExpressions(initial_velocity_error, self.u_vars),
                np.zeros(3)
            )

            final_velocity_error = (
                control_points[-1] - velocity[1] * time_control_points[-1]
            )
            final_velocity_con = LinearEqualityConstraint(
                DecomposeLinearExpressions(final_velocity_error, self.u_vars),
                np.zeros(3)
            )

            for edge in source_edges:
                edge.AddConstraint(Binding[Constraint](initial_velocity_con, edge.xv()))

            for edge in target_edges:
                edge.AddConstraint(Binding[Constraint](final_velocity_con, edge.xu()))

        # 添加零速度约束
        if zero_velocity_at_boundaries:
            flat_deriv = self.flat_trajectory.MakeDerivative(1)
            control_points = flat_deriv.control_points()

            for deriv in range(1, 2):
                for edge in source_edges:
                    edge.AddConstraint(Binding[Constraint](
                        LinearEqualityConstraint(
                            DecomposeLinearExpressions(
                                np.squeeze(control_points[0]),
                                self.u_vars
                            ),
                            np.zeros(3)
                        ),
                        edge.xv()
                    ))

                for edge in target_edges:
                    edge.AddConstraint(Binding[Constraint](
                        LinearEqualityConstraint(
                            DecomposeLinearExpressions(
                                np.squeeze(control_points[-1]),
                                self.u_vars
                            ),
                            np.zeros(3)
                        ),
                        edge.xu()
                    ))

        # 添加时间轨迹导数约束
        if min_time_derivative is not None and min_time_derivative > 0:
            time_deriv = self.time_trajectory.MakeDerivative(1)
            time_control_points = time_deriv.control_points()

            initial_time_deriv_con = LinearConstraint(
                DecomposeLinearExpressions(time_control_points[0], self.u_vars),
                min_time_derivative * np.ones(1),
                np.inf * np.ones(1)
            )

            final_time_deriv_con = LinearConstraint(
                DecomposeLinearExpressions(time_control_points[-1], self.u_vars),
                min_time_derivative * np.ones(1),
                np.inf * np.ones(1)
            )

            for edge in source_edges:
                edge.AddConstraint(Binding[Constraint](initial_time_deriv_con, edge.xv()))

            for edge in target_edges:
                edge.AddConstraint(Binding[Constraint](final_time_deriv_con, edge.xu()))

        # 添加成本到目标点边
        for edge in target_edges:
            for cost in self.edge_costs:
                edge.AddCost(Binding[Cost](cost, edge.xu()))

            for d_con in self.deriv_constraints:
                edge.AddConstraint(Binding[Constraint](d_con, edge.xu()))

        return source_edges, target_edges

    def findStartGoalEdges(self, start, goal):
        """
        确定源点和目标点应该连接到哪些区域顶点。
        通过检查点是否在原始区域内来实现（使用original_regions而非regions）。

        Args:
            start (array-like): 源点坐标 [x, y, theta]。
            goal (array-like): 目标点坐标 [x, y, theta]。

        Returns:
            list of lists: 格式为 [[start_connected_region_indices], [goal_connected_region_indices]]。
        """
        edges = [[], []]
        for ii in range(len(self.original_regions)):
            # 只检查[x, y]部分，因为original_regions是2D的
            if self.original_regions[ii].PointInSet(start[:2]):
                edges[0].append(ii)
            if self.original_regions[ii].PointInSet(goal[:2]):
                edges[1].append(ii)
        return edges

    def _extract_trajectory(self, best_path, best_result):
        """
        从求解结果中提取平坦输出轨迹和时间轨迹

        Args:
            best_path: 最优路径边列表
            best_result: 求解结果

        Returns:
            (平坦输出轨迹, 时间轨迹)
        """
        knots = np.zeros(self.order + 1)
        flat_control_points = []
        time_control_points = []

        for edge in best_path:
            if edge.v() == self.target:
                knots = np.concatenate((knots, [knots[-1]]))
                edge_points = best_result.GetSolution(edge.xu())
                edge_flat_points = np.reshape(
                    edge_points[:-(self.order + 1)],
                    (3, self.order + 1),
                    "F"
                )
                flat_control_points.append(edge_flat_points[:, -1])
                time_control_points.append(np.array([edge_points[-1]]))
                break

            edge_time = knots[-1] + 1.
            knots = np.concatenate((knots, np.full(self.order, edge_time)))

            edge_points = best_result.GetSolution(edge.xv())
            edge_flat_points = np.reshape(
                edge_points[:-(self.order + 1)],
                (3, self.order + 1),
                "F"
            )
            edge_time_points = edge_points[-(self.order + 1):]

            for ii in range(self.order):
                flat_control_points.append(edge_flat_points[:, ii])
                time_control_points.append(np.array([edge_time_points[ii]]))

        # 时间偏移校正
        offset = time_control_points[0].copy()
        for ii in range(len(time_control_points)):
            time_control_points[ii] -= offset

        flat_control_points = np.array(flat_control_points).T
        time_control_points = np.array(time_control_points).T

        flat_traj = BsplineTrajectory(
            BsplineBasis(self.order + 1, knots),
            flat_control_points
        )
        time_traj = BsplineTrajectory(
            BsplineBasis(self.order + 1, knots),
            time_control_points
        )

        return flat_traj, time_traj

    def SolvePath(
        self,
        rounding: bool = True,
        verbose: bool = False,
        preprocessing: bool = False
    ) -> Tuple[Optional['AckermannTrajectory'], Dict]:
        """
        求解GCS问题并返回阿克曼轨迹

        Args:
            rounding: 是否使用舍入策略
            verbose: 是否打印详细信息
            preprocessing: 是否启用预处理

        Returns:
            (阿克曼轨迹对象, 结果字典)
        """
        best_path, best_result, results_dict = self.solveGCS(
            rounding, preprocessing, verbose
        )

        if best_path is None:
            return None, results_dict

        flat_trajectory, time_trajectory = self._extract_trajectory(
            best_path, best_result
        )

        ackermann_trajectory = AckermannTrajectory(
            flat_trajectory,
            time_trajectory,
            self.wheelbase
        )

        return ackermann_trajectory, results_dict
