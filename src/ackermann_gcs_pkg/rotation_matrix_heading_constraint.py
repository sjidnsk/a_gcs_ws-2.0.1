"""
旋转矩阵法航向角约束模块

实现基于旋转矩阵（叉积形式）的航向角约束，具有以下特点：
1. 凸性约束：仿射约束，可嵌入GCS算法
2. 数值稳定：避免tan(θ)在θ ≈ ±π/2时的奇异问题
3. 多控制点支持：可约束多个控制点，提高轨迹质量
4. 向后兼容：与传统线性化方法共存

数学原理：
- 使用叉积形式：ṗ × R(θ)·d_ref = 0
- 展开为：Δx·sin(θ) - Δy·cos(θ) = 0
- 这是线性等式约束，属于凸集

作者: Path Planning Team
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

# Drake导入
try:
    from pydrake.solvers import (
        LinearEqualityConstraint,
        LinearConstraint,
    )
    from pydrake.symbolic import (
        Expression,
        Variable,
        DecomposeLinearExpressions,
    )
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    warnings.warn("Drake未安装，部分功能不可用")


class HeadingConstraintMethod(Enum):
    """航向角约束方法枚举"""
    LINEARIZED = "linearized"  # 传统线性化方法（tan形式）
    ROTATION_MATRIX = "rotation_matrix"  # 旋转矩阵法（叉积形式）


@dataclass
class HeadingConstraintConfig:
    """航向角约束配置"""
    # 约束方法
    method: HeadingConstraintMethod = HeadingConstraintMethod.ROTATION_MATRIX

    # 多控制点数量（约束前k个/后k个控制点）
    num_control_points: int = 2

    # 是否启用多控制点约束
    enable_multi_point: bool = True

    # 【新增】是否启用方向约束（点积约束）
    # 启用后，将添加点积约束确保控制点差向量与期望方向朝向一致
    enable_direction_constraint: bool = True

    # 【新增】点积约束容差
    # 确保点积值大于此值，避免零解
    direction_epsilon: float = 0.01

    # 速度标量约束（可选）
    # 如果启用，将约束速度大小：v_min ≤ ||ṗ|| ≤ v_max
    enable_velocity_constraint: bool = False
    velocity_min: float = 0.1  # 最小速度
    velocity_max: float = 2.0  # 最大速度

    # 数值容差
    angle_tolerance: float = 1e-6  # 角度容差（用于判断特殊角度）


class RotationMatrixHeadingConstraint:
    """
    旋转矩阵法航向角约束

    使用叉积形式实现航向角约束：
    - 约束形式：(P1 - P0) × d_θ = 0
    - 展开：(P1[x] - P0[x])·sin(θ) - (P1[y] - P0[y])·cos(θ) = 0
    - 这是仿射约束，属于凸集

    优点：
    - 凸性约束，可嵌入GCS
    - 数值稳定，无奇异点
    - 可扩展多控制点
    """

    def __init__(self, heading_angle: float, constraint_type: Optional[str] = None):
        """
        初始化旋转矩阵航向角约束

        Args:
            heading_angle: 航向角（弧度）
            constraint_type: 约束类型，'source'或'target'，用于标识起点或终点约束
        """
        self.theta = heading_angle
        self.cos_theta = np.cos(heading_angle)
        self.sin_theta = np.sin(heading_angle)

        # 约束类型标记（'source'或'target'）
        self.constraint_type = constraint_type

        # 预计算旋转矩阵
        self.rotation_matrix = np.array([
            [self.cos_theta, -self.sin_theta],
            [self.sin_theta, self.cos_theta]
        ])

        # 方向向量
        self.direction_vector = np.array([self.cos_theta, self.sin_theta])
    
    def create_cross_product_constraint(
        self,
        p0_x: Union[Variable, Expression],
        p0_y: Union[Variable, Expression],
        p1_x: Union[Variable, Expression],
        p1_y: Union[Variable, Expression],
        variables: List[Variable]
    ) -> Optional['LinearEqualityConstraint']:
        """
        创建叉积形式的线性约束
        
        约束：(P1[x] - P0[x]) * sin(θ) - (P1[y] - P0[y]) * cos(θ) = 0
        
        Args:
            p0_x, p0_y: 控制点0的坐标
            p1_x, p1_y: 控制点1的坐标
            variables: 决策变量列表
        
        Returns:
            LinearEqualityConstraint对象（如果Drake可用）
        """
        if not DRAKE_AVAILABLE:
            warnings.warn("Drake未安装，无法创建约束")
            return None
        
        # 构造约束表达式
        # expr = (p1_x - p0_x) * sin(θ) - (p1_y - p0_y) * cos(θ)
        expr = (p1_x - p0_x) * self.sin_theta - (p1_y - p0_y) * self.cos_theta
        
        # 分解为线性表达式：A·x = b
        A = DecomposeLinearExpressions([expr], variables)
        b = np.array([0.0])
        
        return LinearEqualityConstraint(A, b)
    
    def create_multi_control_point_constraints(
        self,
        control_points: List[Tuple[Union[Variable, Expression], Union[Variable, Expression]]],
        num_points: Optional[int] = None,
        variables: List[Variable] = None
    ) -> List['LinearEqualityConstraint']:
        """
        创建多控制点航向角约束
        
        约束前k个控制点对的方向一致：
        (Pᵢ - Pᵢ₋₁) × d_θ = 0,  i = 1, 2, ..., k
        
        Args:
            control_points: 控制点列表，每个元素为(x, y)坐标
            num_points: 约束的控制点数量（默认约束所有）
            variables: 决策变量列表
        
        Returns:
            约束列表
        """
        if not DRAKE_AVAILABLE:
            warnings.warn("Drake未安装，无法创建约束")
            return []
        
        constraints = []
        
        # 确定约束数量
        k = num_points if num_points is not None else len(control_points) - 1
        k = min(k, len(control_points) - 1)  # 不能超过控制点数量
        
        for i in range(1, k + 1):
            # 获取控制点
            p_prev_x, p_prev_y = control_points[i - 1]
            p_curr_x, p_curr_y = control_points[i]
            
            # 创建约束
            constraint = self.create_cross_product_constraint(
                p_prev_x, p_prev_y,
                p_curr_x, p_curr_y,
                variables
            )
            
            if constraint is not None:
                constraints.append(constraint)
        
        return constraints
    
    @staticmethod
    def check_heading_satisfaction(
        p0: np.ndarray,
        p1: np.ndarray,
        heading_angle: float,
        tolerance: float = 1e-3
    ) -> Tuple[bool, float]:
        """
        检查两点是否满足航向角约束
        
        Args:
            p0: 起点 [x, y]
            p1: 终点 [x, y]
            heading_angle: 期望航向角
            tolerance: 容差
        
        Returns:
            (is_satisfied, error): 是否满足约束，误差值
        """
        # 计算实际方向
        delta = p1 - p0
        actual_heading = np.arctan2(delta[1], delta[0])
        
        # 计算角度差（考虑周期性）
        error = actual_heading - heading_angle
        while error > np.pi:
            error -= 2 * np.pi
        while error < -np.pi:
            error += 2 * np.pi
        
        is_satisfied = abs(error) < tolerance
        
        return is_satisfied, error


class DirectionConstraint:
    """
    点积方向约束

    使用点积形式确保控制点差向量与期望方向向量的朝向一致：
    - 约束形式：(P1 - P0) · d_θ ≥ ε
    - 展开：(P1[x] - P0[x])·cos(θ) + (P1[y] - P0[y])·sin(θ) ≥ ε
    - 这是线性不等式约束，属于凸集

    数学原理：
    - 点积 > 0 表示两向量夹角 < 90°（朝向一致）
    - 点积 < 0 表示两向量夹角 > 90°（朝向相反）
    - 点积 = 0 表示两向量垂直

    与叉积约束结合使用：
    - 叉积约束：保证方向共线（平行）
    - 点积约束：保证方向朝向一致（同向）
    - 两者结合：保证方向完全一致

    优点：
    - 凸性约束（线性不等式），可嵌入GCS
    - 数值稳定，无奇异点
    - 解决航向角约束的方向歧义问题
    """

    def __init__(
        self,
        heading_angle: float,
        epsilon: float = 0.01,
        constraint_type: Optional[str] = None
    ):
        """
        初始化点积方向约束

        Args:
            heading_angle: 航向角（弧度）
            epsilon: 最小点积值，确保严格正向（默认0.01）
            constraint_type: 约束类型，'source'或'target'，用于标识起点或终点约束
        """
        self.theta = heading_angle
        self.epsilon = epsilon
        self.cos_theta = np.cos(heading_angle)
        self.sin_theta = np.sin(heading_angle)

        # 约束类型标记（'source'或'target'）
        self.constraint_type = constraint_type

        # 方向向量
        self.direction_vector = np.array([self.cos_theta, self.sin_theta])

    def create_dot_product_constraint(
        self,
        p0_x: Union[Variable, Expression],
        p0_y: Union[Variable, Expression],
        p1_x: Union[Variable, Expression],
        p1_y: Union[Variable, Expression],
        variables: List[Variable]
    ) -> Optional['LinearConstraint']:
        """
        创建点积形式的线性不等式约束

        约束：(P1[x] - P0[x]) * cos(θ) + (P1[y] - P0[y]) * sin(θ) ≥ ε

        这确保控制点差向量与期望方向向量的点积为正，即夹角小于90°

        Args:
            p0_x, p0_y: 控制点0的坐标
            p1_x, p1_y: 控制点1的坐标
            variables: 决策变量列表

        Returns:
            LinearConstraint对象（如果Drake可用）
        """
        if not DRAKE_AVAILABLE:
            warnings.warn("Drake未安装，无法创建约束")
            return None

        # 构造约束表达式
        # expr = (p1_x - p0_x) * cos(θ) + (p1_y - p0_y) * sin(θ)
        expr = (p1_x - p0_x) * self.cos_theta + (p1_y - p0_y) * self.sin_theta

        # 分解为线性表达式：A·x >= b
        A = DecomposeLinearExpressions([expr], variables)

        # 线性不等式约束：lower_bound <= A·x <= upper_bound
        # 这里：epsilon <= expr <= inf
        lower_bound = np.array([self.epsilon])
        upper_bound = np.array([np.inf])

        return LinearConstraint(A, lower_bound, upper_bound)

    def create_multi_point_constraints(
        self,
        control_points: List[Tuple[Union[Variable, Expression], Union[Variable, Expression]]],
        num_points: Optional[int] = None,
        variables: List[Variable] = None
    ) -> List['LinearConstraint']:
        """
        创建多控制点方向约束

        约束前k个控制点对的方向一致：
        (Pᵢ - Pᵢ₋₁) · d_θ ≥ ε,  i = 1, 2, ..., k

        Args:
            control_points: 控制点列表，每个元素为(x, y)坐标
            num_points: 约束的控制点数量（默认约束所有）
            variables: 决策变量列表

        Returns:
            约束列表
        """
        if not DRAKE_AVAILABLE:
            warnings.warn("Drake未安装，无法创建约束")
            return []

        constraints = []

        # 确定约束数量
        k = num_points if num_points is not None else len(control_points) - 1
        k = min(k, len(control_points) - 1)  # 不能超过控制点数量

        for i in range(1, k + 1):
            # 获取控制点
            p_prev_x, p_prev_y = control_points[i - 1]
            p_curr_x, p_curr_y = control_points[i]

            # 创建约束
            constraint = self.create_dot_product_constraint(
                p_prev_x, p_prev_y,
                p_curr_x, p_curr_y,
                variables
            )

            if constraint is not None:
                constraints.append(constraint)

        return constraints

    @staticmethod
    def check_direction_satisfaction(
        p0: np.ndarray,
        p1: np.ndarray,
        heading_angle: float,
        tolerance: float = 1e-3
    ) -> Tuple[bool, float, float]:
        """
        检查两点是否满足方向约束

        Args:
            p0: 起点 [x, y]
            p1: 终点 [x, y]
            heading_angle: 期望航向角
            tolerance: 角度容差

        Returns:
            (is_satisfied, dot_product, angle_diff):
            - is_satisfied: 是否满足约束（点积>0且角度差小）
            - dot_product: 点积值
            - angle_diff: 角度差（归一化到[-π, π]）
        """
        # 计算控制点差向量
        delta = p1 - p0

        # 计算期望方向向量
        direction_vector = np.array([np.cos(heading_angle), np.sin(heading_angle)])

        # 计算点积
        dot_product = np.dot(delta, direction_vector)

        # 计算实际航向角
        actual_heading = np.arctan2(delta[1], delta[0])

        # 计算角度差
        angle_diff = actual_heading - heading_angle

        # 归一化角度差到[-π, π]
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        # 判断是否满足约束
        # 条件1：点积为正（朝向一致）
        # 条件2：角度差小（方向接近）
        is_satisfied = dot_product > 0 and abs(angle_diff) < tolerance

        return is_satisfied, dot_product, angle_diff


class LinearizedHeadingConstraint:
    """
    传统线性化航向角约束（用于对比）

    使用tan(θ)形式：
    - 约束形式：(P1[y] - P0[y]) = tan(θ) · (P1[x] - P0[x])
    - 特殊情况：θ ≈ ±π/2时，P1[x] = P0[x]

    问题：
    - tan(θ)在θ ≈ ±π/2时奇异
    - 只约束2个控制点
    """

    def __init__(self, heading_angle: float, angle_tolerance: float = 1e-6, constraint_type: Optional[str] = None):
        """
        初始化线性化航向角约束

        Args:
            heading_angle: 航向角（弧度）
            angle_tolerance: 角度容差（用于判断特殊角度）
            constraint_type: 约束类型，'source'或'target'，用于标识起点或终点约束
        """
        self.theta = heading_angle
        self.angle_tolerance = angle_tolerance

        # 约束类型标记（'source'或'target'）
        self.constraint_type = constraint_type

        # 预计算
        self.cos_theta = np.cos(heading_angle)
        self.sin_theta = np.sin(heading_angle)

        # 判断是否为特殊角度（θ ≈ ±π/2）
        self.is_vertical = abs(self.cos_theta) < angle_tolerance

        if not self.is_vertical:
            self.tan_theta = np.tan(heading_angle)
    
    def create_constraint(
        self,
        p0_x: Union[Variable, Expression],
        p0_y: Union[Variable, Expression],
        p1_x: Union[Variable, Expression],
        p1_y: Union[Variable, Expression],
        variables: List[Variable]
    ) -> Optional['LinearEqualityConstraint']:
        """
        创建线性化约束
        
        Args:
            p0_x, p0_y: 控制点0的坐标
            p1_x, p1_y: 控制点1的坐标
            variables: 决策变量列表
        
        Returns:
            LinearEqualityConstraint对象
        """
        if not DRAKE_AVAILABLE:
            warnings.warn("Drake未安装，无法创建约束")
            return None
        
        if self.is_vertical:
            # 特殊情况：θ ≈ ±π/2
            # 约束：P1[x] = P0[x]
            expr = p1_x - p0_x
        else:
            # 一般情况
            # 约束：(P1[y] - P0[y]) = tan(θ) · (P1[x] - P0[x])
            expr = (p1_y - p0_y) - self.tan_theta * (p1_x - p0_x)
        
        # 分解为线性表达式
        A = DecomposeLinearExpressions([expr], variables)
        b = np.array([0.0])
        
        return LinearEqualityConstraint(A, b)


class HeadingConstraintFactory:
    """
    航向角约束工厂

    根据配置创建相应的航向角约束
    """

    @staticmethod
    def create_constraint(
        heading_angle: float,
        method: HeadingConstraintMethod = HeadingConstraintMethod.ROTATION_MATRIX,
        config: Optional[HeadingConstraintConfig] = None
    ) -> Union[RotationMatrixHeadingConstraint, LinearizedHeadingConstraint]:
        """
        创建航向角约束对象

        Args:
            heading_angle: 航向角
            method: 约束方法
            config: 配置（可选）

        Returns:
            约束对象
        """
        if method == HeadingConstraintMethod.ROTATION_MATRIX:
            return RotationMatrixHeadingConstraint(heading_angle)
        elif method == HeadingConstraintMethod.LINEARIZED:
            tolerance = config.angle_tolerance if config else 1e-6
            return LinearizedHeadingConstraint(heading_angle, tolerance)
        else:
            raise ValueError(f"不支持的约束方法: {method}")

    @staticmethod
    def create_combined_constraints(
        heading_angle: float,
        control_points: List[Tuple[Union[Variable, Expression], Union[Variable, Expression]]],
        variables: List[Variable],
        config: HeadingConstraintConfig,
        constraint_type: Optional[str] = None
    ) -> Tuple[List['LinearEqualityConstraint'], List['LinearConstraint']]:
        """
        创建组合航向角约束（叉积约束 + 点积约束）

        叉积约束：保证方向共线（平行）
        点积约束：保证方向朝向一致（同向）
        两者结合：保证方向完全一致

        Args:
            heading_angle: 航向角
            control_points: 控制点列表
            variables: 决策变量
            config: 约束配置
            constraint_type: 约束类型，'source'或'target'

        Returns:
            (cross_product_constraints, dot_product_constraints):
            - cross_product_constraints: 叉积约束列表（等式约束）
            - dot_product_constraints: 点积约束列表（不等式约束）
        """
        # 创建叉积约束（共线约束）
        cross_constraint_obj = RotationMatrixHeadingConstraint(
            heading_angle, constraint_type=constraint_type
        )
        cross_constraints = cross_constraint_obj.create_multi_control_point_constraints(
            control_points,
            config.num_control_points if config.enable_multi_point else 1,
            variables
        )

        # 创建点积约束（方向约束）
        direction_constraint_obj = DirectionConstraint(
            heading_angle,
            epsilon=config.direction_epsilon,
            constraint_type=constraint_type
        )
        dot_constraints = direction_constraint_obj.create_multi_point_constraints(
            control_points,
            config.num_control_points if config.enable_multi_point else 1,
            variables
        )

        return cross_constraints, dot_constraints

    @staticmethod
    def create_heading_constraints(
        heading_angle: float,
        control_points: List[Tuple[Union[Variable, Expression], Union[Variable, Expression]]],
        variables: List[Variable],
        config: HeadingConstraintConfig,
        constraint_type: Optional[str] = None
    ) -> List[Union['LinearEqualityConstraint', 'LinearConstraint']]:
        """
        创建航向角约束（统一接口）

        根据配置决定是否添加方向约束：
        - 如果enable_direction_constraint=True: 返回叉积约束 + 点积约束
        - 如果enable_direction_constraint=False: 仅返回叉积约束

        Args:
            heading_angle: 航向角
            control_points: 控制点列表
            variables: 决策变量
            config: 约束配置
            constraint_type: 约束类型，'source'或'target'

        Returns:
            约束列表（可能包含等式约束和不等式约束）
        """
        cross_constraints, dot_constraints = \
            HeadingConstraintFactory.create_combined_constraints(
                heading_angle, control_points, variables, config, constraint_type
            )

        if config.enable_direction_constraint:
            # 启用方向约束：返回叉积约束 + 点积约束
            return cross_constraints + dot_constraints
        else:
            # 禁用方向约束：仅返回叉积约束（向后兼容）
            return cross_constraints

    @staticmethod
    def create_heading_constraints_per_pair(
        heading_angle: float,
        control_points: List[Tuple[Union[Variable, Expression], Union[Variable, Expression]]],
        variables: List[Variable],
        config: HeadingConstraintConfig,
        constraint_type: Optional[str] = None,
        is_first_pair_degenerate: bool = False
    ) -> List[Union['LinearEqualityConstraint', 'LinearConstraint']]:
        """
        创建航向角约束（逐对选择性禁用点积约束）

        当v=0时，第一对控制点(P1-P0)退化为零向量，点积约束
        (P1-P0)·d_theta >= epsilon 严格矛盾(0 >= epsilon)。
        此方法仅对退化对禁用点积约束，对非退化对保留完整约束。

        Args:
            heading_angle: 航向角
            control_points: 控制点列表
            variables: 决策变量
            config: 约束配置
            constraint_type: 约束类型，'source'或'target'
            is_first_pair_degenerate: 第一对控制点是否退化（v=0时P1=P0）

        Returns:
            约束列表（叉积约束始终添加，点积约束仅对非退化对添加）
        """
        # 创建叉积约束对象
        cross_constraint_obj = RotationMatrixHeadingConstraint(
            heading_angle, constraint_type=constraint_type
        )

        # 确定约束数量
        k = config.num_control_points if config.enable_multi_point else 1
        k = min(k, len(control_points) - 1)

        constraints = []

        # 创建点积约束对象（如果需要）
        if config.enable_direction_constraint:
            direction_constraint_obj = DirectionConstraint(
                heading_angle,
                epsilon=config.direction_epsilon,
                constraint_type=constraint_type
            )

        for i in range(1, k + 1):
            p_prev_x, p_prev_y = control_points[i - 1]
            p_curr_x, p_curr_y = control_points[i]

            # 叉积约束：始终添加（共线约束）
            cross_con = cross_constraint_obj.create_cross_product_constraint(
                p_prev_x, p_prev_y,
                p_curr_x, p_curr_y,
                variables
            )
            if cross_con is not None:
                constraints.append(cross_con)

            # 点积约束：仅对非退化对添加
            if config.enable_direction_constraint:
                is_degenerate = (i == 1 and is_first_pair_degenerate)
                if not is_degenerate:
                    dot_con = direction_constraint_obj.create_dot_product_constraint(
                        p_prev_x, p_prev_y,
                        p_curr_x, p_curr_y,
                        variables
                    )
                    if dot_con is not None:
                        constraints.append(dot_con)

        return constraints

    @staticmethod
    def create_source_and_target_constraints(
        source_heading: float,
        target_heading: float,
        source_control_points: List[Tuple[Union[Variable, Expression], Union[Variable, Expression]]],
        target_control_points: List[Tuple[Union[Variable, Expression], Union[Variable, Expression]]],
        variables: List[Variable],
        config: Optional[HeadingConstraintConfig] = None
    ) -> List['LinearEqualityConstraint']:
        """
        创建起点和终点航向角约束

        Args:
            source_heading: 起点航向角
            target_heading: 终点航向角
            source_control_points: 起点控制点列表（前k个）
            target_control_points: 终点控制点列表（后k个，逆序）
            variables: 决策变量
            config: 配置

        Returns:
            约束列表
        """
        if config is None:
            config = HeadingConstraintConfig()

        constraints = []

        # 创建起点约束（标记为'source'）
        if config.method == HeadingConstraintMethod.ROTATION_MATRIX:
            source_constraint_obj = RotationMatrixHeadingConstraint(source_heading, constraint_type='source')
        else:
            tolerance = config.angle_tolerance if config else 1e-6
            source_constraint_obj = LinearizedHeadingConstraint(source_heading, tolerance, constraint_type='source')

        if isinstance(source_constraint_obj, RotationMatrixHeadingConstraint):
            # 使用旋转矩阵法
            source_constraints = source_constraint_obj.create_multi_control_point_constraints(
                source_control_points,
                config.num_control_points if config.enable_multi_point else 1,
                variables
            )
            constraints.extend(source_constraints)
        else:
            # 使用线性化方法
            if len(source_control_points) >= 2:
                p0_x, p0_y = source_control_points[0]
                p1_x, p1_y = source_control_points[1]
                constraint = source_constraint_obj.create_constraint(
                    p0_x, p0_y, p1_x, p1_y, variables
                )
                if constraint is not None:
                    constraints.append(constraint)

        # 创建终点约束（标记为'target'）
        if config.method == HeadingConstraintMethod.ROTATION_MATRIX:
            target_constraint_obj = RotationMatrixHeadingConstraint(target_heading, constraint_type='target')
        else:
            tolerance = config.angle_tolerance if config else 1e-6
            target_constraint_obj = LinearizedHeadingConstraint(target_heading, tolerance, constraint_type='target')

        if isinstance(target_constraint_obj, RotationMatrixHeadingConstraint):
            # 使用旋转矩阵法
            target_constraints = target_constraint_obj.create_multi_control_point_constraints(
                target_control_points,
                config.num_control_points if config.enable_multi_point else 1,
                variables
            )
            constraints.extend(target_constraints)
        else:
            # 使用线性化方法
            if len(target_control_points) >= 2:
                p0_x, p0_y = target_control_points[0]
                p1_x, p1_y = target_control_points[1]
                constraint = target_constraint_obj.create_constraint(
                    p0_x, p0_y, p1_x, p1_y, variables
                )
                if constraint is not None:
                    constraints.append(constraint)

        return constraints


# ==================== 便捷函数 ====================

def create_rotation_matrix_heading_constraint(
    heading_angle: float,
    p0_x: Union[Variable, Expression],
    p0_y: Union[Variable, Expression],
    p1_x: Union[Variable, Expression],
    p1_y: Union[Variable, Expression],
    variables: List[Variable]
) -> Optional['LinearEqualityConstraint']:
    """
    便捷函数：创建旋转矩阵航向角约束
    
    Args:
        heading_angle: 航向角
        p0_x, p0_y: 控制点0
        p1_x, p1_y: 控制点1
        variables: 决策变量
    
    Returns:
        约束对象
    """
    constraint_obj = RotationMatrixHeadingConstraint(heading_angle)
    return constraint_obj.create_cross_product_constraint(
        p0_x, p0_y, p1_x, p1_y, variables
    )

