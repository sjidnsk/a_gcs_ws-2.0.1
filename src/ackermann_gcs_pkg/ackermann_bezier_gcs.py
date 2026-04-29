"""
阿克曼转向车辆贝塞尔GCS

本模块实现了阿克曼转向车辆的贝塞尔GCS类，继承自BezierGCS，添加航向角约束功能。
支持两种航向角约束方法：
1. 线性化方法（传统）：使用tan(θ)形式
2. 旋转矩阵法（推荐）：使用叉积形式，凸性约束，数值稳定
"""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass

from pydrake.geometry.optimization import HPolyhedron
from pydrake.solvers import (
    Binding,
    Constraint,
    LinearEqualityConstraint,
)

from gcs_pkg.scripts.core.bezier import BezierGCS

from .ackermann_data_structures import VehicleParams, BezierConfig
from .rotation_matrix_heading_constraint import (
    HeadingConstraintMethod,
    HeadingConstraintConfig,
    HeadingConstraintFactory,
    RotationMatrixHeadingConstraint,
    LinearizedHeadingConstraint,
    DirectionConstraint,
)


@dataclass
class TypedConstraint:
    """
    带类型标记的约束包装类

    用于区分起点航向角约束和终点航向角约束
    """
    constraint: Constraint  # Drake约束对象
    constraint_type: str    # 'source' 或 'target'


class AckermannBezierGCS(BezierGCS):
    """
    阿克曼转向车辆贝塞尔GCS

    继承自BezierGCS，添加航向角约束功能，通过几何约束保证起终点航向角。
    """

    def __init__(
        self,
        regions: List[HPolyhedron],
        vehicle_params: VehicleParams,
        bezier_config: Optional[BezierConfig] = None,
        curvature_constraint_version: str = "v1",
    ):
        """
        初始化阿克曼贝塞尔GCS

        Args:
            regions: 工作空间区域列表
            vehicle_params: 车辆参数
            bezier_config: 贝塞尔曲线配置，如果为None则使用默认配置
            curvature_constraint_version: 曲率约束版本，"v1"(Lorentz锥)或"v2"(旋转二阶锥)
        """
        # 使用默认配置
        if bezier_config is None:
            bezier_config = BezierConfig()

        # 调用父类初始化
        super().__init__(
            regions=regions,
            order=bezier_config.order,
            continuity=bezier_config.continuity,
            hdot_min=bezier_config.hdot_min,
            full_dim_overlap=bezier_config.full_dim_overlap,
            hyperellipsoid_num_samples_per_dim_factor=bezier_config.hyperellipsoid_num_samples_per_dim_factor,
            curvature_constraint_version=curvature_constraint_version,
        )

        # 存储车辆参数和贝塞尔配置
        self.vehicle_params = vehicle_params
        self.bezier_config = bezier_config

        # 初始化自定义约束列表
        self.custom_constraints: List[Constraint] = []
        
        # 航向角约束配置（默认使用旋转矩阵法）
        self.heading_constraint_config = HeadingConstraintConfig(
            method=HeadingConstraintMethod.ROTATION_MATRIX,
            num_control_points=3,  # 约束前2个控制点对
            enable_multi_point=True
        )

    def _add_heading_constraint(
        self,
        source_heading: float,
        target_heading: float,
        config: Optional[HeadingConstraintConfig] = None,
        verbose: bool = True,
        source_is_v0: bool = False,
        target_is_v0: bool = False,
    ) -> None:
        """
        添加航向角约束

        支持两种方法：
        1. 旋转矩阵法（推荐）：使用叉积形式，凸性约束，数值稳定
           约束：(P₁[x] - P₀[x])·sin(θ) - (P₁[y] - P₀[y])·cos(θ) = 0
        2. 线性化方法（传统）：使用tan(θ)形式
           约束：(P₁[y] - P₀[y]) = tan(θ)·(P₁[x] - P₀[x])

        当v=0时，使用逐对选择性禁用点积约束：
        - 仅对退化对（第一对P₁-P₀）禁用点积约束
        - 对非退化对保留完整约束（叉积+点积）

        Args:
            source_heading: 起点航向角（弧度）
            target_heading: 终点航向角（弧度）
            config: 航向角约束配置，如果为None则使用默认配置
            verbose: 是否输出调试信息
            source_is_v0: 起点速度是否为0（v=0时第一对控制点退化）
            target_is_v0: 终点速度是否为0（v=0时第一对控制点退化）
        """
        # 使用传入配置或默认配置
        if config is None:
            config = self.heading_constraint_config
        
        # 调试输出：约束配置信息
        if verbose:
            print("\n" + "=" * 60)
            print("航向角约束添加")
            print("=" * 60)
            print(f"约束方法: {config.method.value}")
            print(f"起点航向角: {np.degrees(source_heading):.2f}° ({source_heading:.4f} rad)")
            print(f"终点航向角: {np.degrees(target_heading):.2f}° ({target_heading:.4f} rad)")
            print(f"多控制点约束: {'启用' if config.enable_multi_point else '禁用'}")
            print(f"控制点数量: {config.num_control_points}")
            if source_is_v0:
                print(f"  起点v=0: 逐对禁用第一对点积约束")
            if target_is_v0:
                print(f"  终点v=0: 逐对禁用第一对点积约束")
        
        # 获取边上的变量
        u_control = self.u_r_trajectory.control_points()
        
        # 准备控制点列表
        # 起点控制点：P₀, P₁, P₂, ...
        source_control_points = []
        
        # 计算控制点数量
        available_control_points = len(u_control)  # 可用的控制点数
        configured_control_points = config.num_control_points  # 配置的控制点数
        needed_control_points = configured_control_points + 1 if config.enable_multi_point else 2  # 需要的控制点数
        
        # 实际使用的控制点数：取需要和可用的较小值
        actual_control_points = min(needed_control_points, available_control_points)
        
        # 调试输出：控制点信息
        if verbose:
            print(f"\n控制点配置:")
            print(f"  配置的控制点数: {configured_control_points}")
            print(f"  需要的控制点数: {needed_control_points} (配置数 + 1，用于约束)")
            print(f"  可用的控制点数: {available_control_points} (贝塞尔曲线 order + 1)")
            print(f"  实际使用: {actual_control_points}")
            
            if actual_control_points < needed_control_points:
                print(f"  ⚠️  警告: 可用控制点不足，使用 {actual_control_points} 个而非 {needed_control_points} 个")
            else:
                print(f"  ✓ 控制点充足，使用配置的 {actual_control_points} 个")
        
        for i in range(actual_control_points):
            # u_control[i] 是第i个控制点，u_control[i][0]是x，u_control[i][1]是y
            source_control_points.append((u_control[i][0], u_control[i][1]))
        
        # 终点控制点：Pₙ₋ₖ, ..., Pₙ₋₁, Pₙ（正序，从倒数第k个到最后一个）
        # 注意：必须正序排列，这样约束计算 (Pᵢ - Pᵢ₋₁) 才能得到正确的方向
        target_control_points = []

        # 收集控制点（从后往前）
        temp_target = []
        for i in range(actual_control_points):
            idx = -1 - i
            temp_target.append((u_control[idx][0], u_control[idx][1]))

        # 反转顺序，使其变为正序：Pₙ₋ₖ, ..., Pₙ₋₁, Pₙ
        target_control_points = temp_target[::-1]
        
        # 调试输出：控制点信息
        if verbose:
            print(f"\n起点控制点数量: {len(source_control_points)}")
            print(f"终点控制点数量: {len(target_control_points)}")

            # 输出约束数学形式
            if config.method == HeadingConstraintMethod.ROTATION_MATRIX:
                print("\n约束形式（旋转矩阵法 - 叉积形式）:")
                cos_s = np.cos(source_heading)
                sin_s = np.sin(source_heading)
                cos_t = np.cos(target_heading)
                sin_t = np.sin(target_heading)

                print(f"  起点: (P₁-P₀)·sin(θ) - (P₁-P₀)·cos(θ) = 0")
                print(f"        sin({np.degrees(source_heading):.1f}°) = {sin_s:.6f}")
                print(f"        cos({np.degrees(source_heading):.1f}°) = {cos_s:.6f}")
                print(f"  终点: (Pₙ-Pₙ₋₁)·sin(θ) - (Pₙ-Pₙ₋₁)·cos(θ) = 0")
                print(f"        sin({np.degrees(target_heading):.1f}°) = {sin_t:.6f}")
                print(f"        cos({np.degrees(target_heading):.1f}°) = {cos_t:.6f}")

                # 方向约束说明
                if config.enable_direction_constraint:
                    print("\n方向约束（点积形式）:")
                    if source_is_v0:
                        print(f"  起点: v=0，第一对(P₁-P₀)禁用点积约束（退化）")
                        print(f"        后续对保留点积约束: (Pᵢ-Pᵢ₋₁)·d_θ ≥ ε")
                    else:
                        print(f"  起点: (P₁-P₀)·d_θ ≥ ε")
                    if target_is_v0:
                        print(f"  终点: v=0，第一对(Pₙ-Pₙ₋₁)禁用点积约束（退化）")
                        print(f"        后续对保留点积约束: (Pᵢ-Pᵢ₋₁)·d_θ ≥ ε")
                    else:
                        print(f"  终点: (Pₙ-Pₙ₋₁)·d_θ ≥ ε")
                    print(f"        ε = {config.direction_epsilon}")
                    print("\n约束组合:")
                    print("  ✓ 叉积约束：保证方向共线（平行）")
                    if not source_is_v0 and not target_is_v0:
                        print("  ✓ 点积约束：保证方向朝向一致（同向）")
                    else:
                        print("  ✓ 点积约束：非退化对保证方向朝向一致（同向）")
                        print("  ✓ 退化对：由成本函数和C2连续性隐式保证同向")
                    print("  ✓ 两者结合：保证方向完全一致")
                else:
                    print("\n方向约束: 禁用（仅使用叉积约束）")

                # 凸性说明
                print("\n凸性分析:")
                print("  ✓ 叉积约束是线性等式约束（仿射约束）")
                if config.enable_direction_constraint:
                    print("  ✓ 点积约束是线性不等式约束")
                print("  ✓ 所有约束属于凸集")
                print("  ✓ 可嵌入GCS算法，保证全局最优")
            else:
                print("\n约束形式（线性化方法 - tan形式）:")
                if abs(np.cos(source_heading)) > 1e-6:
                    tan_s = np.tan(source_heading)
                    print(f"  起点: (P₁[y]-P₀[y]) = tan(θ)·(P₁[x]-P₀[x])")
                    print(f"        tan({np.degrees(source_heading):.1f}°) = {tan_s:.6f}")
                else:
                    print(f"  起点: P₁[x] = P₀[x] (垂直方向)")

                if abs(np.cos(target_heading)) > 1e-6:
                    tan_t = np.tan(target_heading)
                    print(f"  终点: (Pₙ[y]-Pₙ₋₁[y]) = tan(θ)·(Pₙ[x]-Pₙ₋₁[x])")
                    print(f"        tan({np.degrees(target_heading):.1f}°) = {tan_t:.6f}")
                else:
                    print(f"  终点: Pₙ[x] = Pₙ₋₁[x] (垂直方向)")

        # 使用逐对选择性禁用创建约束（v=0退化处理）
        if config.method == HeadingConstraintMethod.ROTATION_MATRIX and \
           config.enable_direction_constraint and \
           (source_is_v0 or target_is_v0):
            # 使用逐对选择性禁用点积约束
            source_constraints = HeadingConstraintFactory.create_heading_constraints_per_pair(
                source_heading,
                source_control_points,
                self.u_vars,
                config,
                constraint_type='source',
                is_first_pair_degenerate=source_is_v0
            )
            target_constraints = HeadingConstraintFactory.create_heading_constraints_per_pair(
                target_heading,
                target_control_points,
                self.u_vars,
                config,
                constraint_type='target',
                is_first_pair_degenerate=target_is_v0
            )
        else:
            # 标准路径：使用原有工厂方法
            source_constraints = HeadingConstraintFactory.create_heading_constraints(
                source_heading,
                source_control_points,
                self.u_vars,
                config,
                constraint_type='source'
            )
            target_constraints = HeadingConstraintFactory.create_heading_constraints(
                target_heading,
                target_control_points,
                self.u_vars,
                config,
                constraint_type='target'
            )

        # 包装约束为TypedConstraint
        typed_constraints = []
        for constraint in source_constraints:
            typed_constraints.append(TypedConstraint(constraint, 'source'))
        for constraint in target_constraints:
            typed_constraints.append(TypedConstraint(constraint, 'target'))

        # 调试输出：约束创建结果
        if verbose:
            print(f"\n约束创建结果:")
            print(f"  总约束数量: {len(typed_constraints)}")
            print(f"  起点约束: {len(source_constraints)} 个")
            print(f"  终点约束: {len(target_constraints)} 个")
            if source_constraints:
                print(f"  约束类型: {type(source_constraints[0]).__name__}")
            if config.enable_direction_constraint:
                if source_is_v0 or target_is_v0:
                    print(f"  方向约束: 逐对选择性禁用（退化对禁用点积）")
                else:
                    print(f"  方向约束: 启用")
            else:
                print(f"  方向约束: 禁用")

        # 添加到自定义约束列表
        self.custom_constraints.extend(typed_constraints)

        if verbose:
            print(f"\n当前总约束数: {len(self.custom_constraints)}")
            print("=" * 60)

    def setHeadingConstraintConfig(self, config: HeadingConstraintConfig) -> None:
        """
        设置航向角约束配置
        
        Args:
            config: 航向角约束配置
        """
        self.heading_constraint_config = config
    
    def addSourceTargetWithHeading(
        self,
        source_position: np.ndarray,
        source_heading: float,
        target_position: np.ndarray,
        target_heading: float,
        source_velocity: Optional[np.ndarray] = None,
        target_velocity: Optional[np.ndarray] = None,
        heading_config: Optional[HeadingConstraintConfig] = None,
        verbose: bool = True,
    ) -> None:
        """
        添加起终点约束（包含航向角）

        整合位置约束、航向角约束、速度约束。
        当起终点速度为0时，自动启用逐对选择性禁用点积约束。

        Args:
            source_position: 起点位置，形状为(2,)
            source_heading: 起点航向角（弧度）
            target_position: 终点位置，形状为(2,)
            target_heading: 终点航向角（弧度）
            source_velocity: 起点速度，形状为(2,)，可选
            target_velocity: 终点速度，形状为(2,)，可选
            heading_config: 航向角约束配置，可选
            verbose: 是否输出调试信息
        """
        # v=0退化判断阈值
        v_threshold = 1e-6

        # 判断起终点速度是否为0
        source_is_v0 = (source_velocity is not None and
                        np.linalg.norm(source_velocity) < v_threshold)
        target_is_v0 = (target_velocity is not None and
                        np.linalg.norm(target_velocity) < v_threshold)

        if verbose:
            print("\n" + "=" * 60)
            print("起终点约束添加")
            print("=" * 60)
            print(f"起点位置: ({source_position[0]:.2f}, {source_position[1]:.2f})")
            print(f"起点航向角: {np.degrees(source_heading):.2f}°")
            print(f"终点位置: ({target_position[0]:.2f}, {target_position[1]:.2f})")
            print(f"终点航向角: {np.degrees(target_heading):.2f}°")
            if source_velocity is not None:
                print(f"起点速度: ({source_velocity[0]:.2f}, {source_velocity[1]:.2f})")
                if source_is_v0:
                    print(f"  起点v≈0: 将禁用退化对的点积约束")
            if target_velocity is not None:
                print(f"终点速度: ({target_velocity[0]:.2f}, {target_velocity[1]:.2f})")
                if target_is_v0:
                    print(f"  终点v≈0: 将禁用退化对的点积约束")
        
        # 调用父类方法添加位置约束
        # 注意：velocity参数应该是形状为(2, 2)的numpy数组
        velocity = None
        if source_velocity is not None and target_velocity is not None:
            velocity = np.array([source_velocity, target_velocity])
        
        if verbose:
            print("\n添加位置约束...")
        
        self.addSourceTarget(
            source=source_position,
            target=target_position,
            velocity=velocity,
        )
        
        if verbose:
            print("✓ 位置约束添加完成")
            print("\n添加航向角约束...")
        
        # 添加航向角约束（传递v=0退化信息）
        self._add_heading_constraint(
            source_heading, target_heading, heading_config,
            verbose=verbose,
            source_is_v0=source_is_v0,
            target_is_v0=target_is_v0,
        )
        
        if verbose:
            print("✓ 航向角约束添加完成")
            print("=" * 60)

    def addCustomConstraint(self, constraint: Constraint) -> None:
        """
        添加自定义约束

        Args:
            constraint: 约束对象
        """
        self.custom_constraints.append(constraint)

    def _classify_edges(self, verbose: bool = False) -> dict:
        """
        识别并分类所有边的类型

        边的分类：
        - source_edges: 从source顶点出发的边
        - target_edges: 到达target顶点的边
        - first_real_edges: 从第一个实际区域出发的边（source邻居）
        - middle_edges: 中间边

        Args:
            verbose: 是否输出调试信息

        Returns:
            dict: {
                'source_edges': [edge1, edge2, ...],
                'target_edges': [edge1, edge2, ...],
                'first_real_edges': [edge1, edge2, ...],
                'middle_edges': [edge1, edge2, ...]
            }
        """
        classified_edges = {
            'source_edges': [],
            'target_edges': [],
            'first_real_edges': [],
            'middle_edges': []
        }

        # 获取source顶点的邻居顶点（第一个实际区域）
        source_neighbors = []
        for edge in self.gcs.Edges():
            if edge.u() == self.source:
                source_neighbors.append(edge.v())

        # 分类每条边
        for edge in self.gcs.Edges():
            if edge.u() == self.source:
                classified_edges['source_edges'].append(edge)
            elif edge.v() == self.target:
                classified_edges['target_edges'].append(edge)
            elif edge.u() in source_neighbors:
                classified_edges['first_real_edges'].append(edge)
            else:
                classified_edges['middle_edges'].append(edge)

        # 调试输出
        if verbose:
            print("\n边分类结果:")
            print(f"  Source边: {len(classified_edges['source_edges'])} 条")
            print(f"  Target边: {len(classified_edges['target_edges'])} 条")
            print(f"  第一条真实边: {len(classified_edges['first_real_edges'])} 条")
            print(f"  中间边: {len(classified_edges['middle_edges'])} 条")
            print(f"  总计: {sum(len(v) for v in classified_edges.values())} 条")

        return classified_edges

    def SolvePathWithConstraints(
        self,
        custom_constraints: Optional[List[Constraint]] = None,
        rounding: bool = True,
        preprocessing: bool = True,
        verbose: bool = True,
    ):
        """
        带自定义约束的路径求解

        修正后的约束应用逻辑：
        - 起点航向角约束仅应用到第一条真实边
        - 终点航向角约束仅应用到target边
        - 中间边不应用航向角约束

        Args:
            custom_constraints: 自定义约束列表，如果为None则使用self.custom_constraints
            rounding: 是否使用舍入
            preprocessing: 是否使用预处理
            verbose: 是否输出调试信息

        Returns:
            求解结果（与父类SolvePath返回值一致）
        """
        # 使用传入的自定义约束或存储的自定义约束
        if custom_constraints is None:
            custom_constraints = self.custom_constraints

        # 分类边
        classified_edges = self._classify_edges(verbose=verbose)

        # 分离起点和终点航向角约束
        source_heading_constraints = []
        target_heading_constraints = []
        other_constraints = []

        for item in custom_constraints:
            # 检查是否为TypedConstraint
            if isinstance(item, TypedConstraint):
                if item.constraint_type == 'source':
                    source_heading_constraints.append(item.constraint)
                elif item.constraint_type == 'target':
                    target_heading_constraints.append(item.constraint)
                else:
                    other_constraints.append(item.constraint)
            elif hasattr(item, 'constraint_type'):
                # 兼容旧版本：检查约束是否有constraint_type属性
                if item.constraint_type == 'source':
                    source_heading_constraints.append(item)
                elif item.constraint_type == 'target':
                    target_heading_constraints.append(item)
                else:
                    other_constraints.append(item)
            else:
                # 没有constraint_type属性的约束，视为其他约束
                other_constraints.append(item)

        # 调试输出
        if verbose:
            print("\n" + "=" * 60)
            print("约束应用与求解")
            print("=" * 60)
            print(f"约束分类:")
            print(f"  起点航向角约束: {len(source_heading_constraints)} 个")
            print(f"  终点航向角约束: {len(target_heading_constraints)} 个")
            print(f"  其他约束: {len(other_constraints)} 个")
            print(f"  总计: {len(custom_constraints)} 个")

        # 应用约束
        total_applied_constraints = 0

        # 应用起点航向角约束到第一条真实边
        if source_heading_constraints:
            for edge in classified_edges['first_real_edges']:
                for constraint in source_heading_constraints:
                    edge.AddConstraint(Binding[Constraint](constraint, edge.xu()))
                    total_applied_constraints += 1
            if verbose:
                print(f"\n✓ 起点航向角约束已应用到 {len(classified_edges['first_real_edges'])} 条第一条真实边")

        # 应用终点航向角约束到target边
        if target_heading_constraints:
            for edge in classified_edges['target_edges']:
                for constraint in target_heading_constraints:
                    edge.AddConstraint(Binding[Constraint](constraint, edge.xu()))
                    total_applied_constraints += 1
            if verbose:
                print(f"✓ 终点航向角约束已应用到 {len(classified_edges['target_edges'])} 条target边")

        # 应用其他约束到所有非source边（保持向后兼容）
        if other_constraints:
            for edge in self.gcs.Edges():
                if edge.u() == self.source:
                    continue
                for constraint in other_constraints:
                    edge.AddConstraint(Binding[Constraint](constraint, edge.xu()))
                    total_applied_constraints += 1
            if verbose:
                non_source_edges = len(self.gcs.Edges()) - len(classified_edges['source_edges'])
                print(f"✓ 其他约束已应用到 {non_source_edges} 条非source边")

        if verbose:
            print(f"\n总应用约束实例数: {total_applied_constraints}")
            print("\n开始GCS求解...")
            print("  求解器: MOSEK")
            print(f"  凸松弛: True")
            print(f"  预处理: {preprocessing}")
            print(f"  舍入: {rounding}")

        # 调用父类SolvePath方法
        result = self.SolvePath(rounding=rounding, preprocessing=preprocessing)

        if verbose:
            print("\n✓ GCS求解完成")
            if result is not None:
                print(f"  求解状态: 成功")
            else:
                print(f"  求解状态: 失败")
            print("=" * 60)

        return result
