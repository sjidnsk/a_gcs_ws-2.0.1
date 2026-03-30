"""
转向角约束管理器模块

提供SteeringConstraintManager类，用于管理阿克曼转向车辆的转向角约束，
避免约束累积问题。
"""

import numpy as np
from typing import Optional, List, Tuple, Dict

from pydrake.solvers import (
    Binding,
    Constraint,
    LinearConstraint,
)
from pydrake.symbolic import (
    DecomposeLinearExpressions,
    Expression,
)


__all__ = ['SteeringConstraintManager']


class SteeringConstraintManager:
    """
    转向角约束管理器

    负责创建、更新和管理转向角约束，避免约束累积问题。

    使用策略：
    1. 为每条边创建可更新的约束对象
    2. 每次迭代时更新约束参数，而不是添加新约束
    3. 提供约束数量查询接口用于调试
    """

    def __init__(self, gcs, u_vars, wheelbase, flat_trajectory, source):
        """
        初始化约束管理器

        Args:
            gcs: GCS对象
            u_vars: 决策变量
            wheelbase: 车辆轴距
            flat_trajectory: 平坦输出轨迹
            source: 源点顶点
        """
        self.gcs = gcs
        self.u_vars = u_vars
        self.wheelbase = wheelbase
        self.flat_trajectory = flat_trajectory
        self.source = source
        self.constraint_bindings = {}  # {(edge_id, type): binding}
        self.constraint_objects = {}  # {(edge_id, type): constraint}
        self.edges = [e for e in gcs.Edges() if e.u() != source]

    def create_initial_constraints(self, v_conservative, tan_delta_max, tan_delta_min):
        """
        创建初始约束

        Args:
            v_conservative: 保守速度估计
            tan_delta_max: tan(δ_max)
            tan_delta_min: tan(δ_min)
        """
        flat_deriv = self.flat_trajectory.MakeDerivative(1)
        control_points = flat_deriv.control_points()

        for i, cp in enumerate(control_points):
            theta_dot_expr = cp[2]
            H_theta = DecomposeLinearExpressions([theta_dot_expr], self.u_vars)

            # 创建约束对象
            steering_con_max = LinearConstraint(
                H_theta,
                -np.inf * np.ones(1),
                (v_conservative * tan_delta_max / self.wheelbase) * np.ones(1)
            )

            steering_con_min = LinearConstraint(
                H_theta,
                (v_conservative * tan_delta_min / self.wheelbase) * np.ones(1),
                np.inf * np.ones(1)
            )

            # 添加到所有边
            for edge in self.edges:
                edge_id = id(edge)

                # 上界约束
                binding_max = Binding[Constraint](steering_con_max, edge.xu())
                edge.AddConstraint(binding_max)
                self.constraint_bindings[(edge_id, 'max')] = binding_max
                self.constraint_objects[(edge_id, 'max')] = steering_con_max

                # 下界约束
                binding_min = Binding[Constraint](steering_con_min, edge.xu())
                edge.AddConstraint(binding_min)
                self.constraint_bindings[(edge_id, 'min')] = binding_min
                self.constraint_objects[(edge_id, 'min')] = steering_con_min

    def update_constraints(self, v_conservative, tan_delta_max, tan_delta_min):
        """
        更新约束参数

        Args:
            v_conservative: 新的保守速度估计
            tan_delta_max: tan(δ_max)
            tan_delta_min: tan(δ_min)
        """
        flat_deriv = self.flat_trajectory.MakeDerivative(1)
        control_points = flat_deriv.control_points()

        for i, cp in enumerate(control_points):
            theta_dot_expr = cp[2]
            H_theta = DecomposeLinearExpressions([theta_dot_expr], self.u_vars)

            # 计算新的约束边界
            new_b_max = (v_conservative * tan_delta_max / self.wheelbase) * np.ones(1)
            new_b_min = (v_conservative * tan_delta_min / self.wheelbase) * np.ones(1)

            # 更新所有边的约束
            for edge in self.edges:
                edge_id = id(edge)

                # 更新上界约束: H_theta * x <= new_b_max
                if (edge_id, 'max') in self.constraint_objects:
                    self.constraint_objects[(edge_id, 'max')].UpdateCoefficients(
                        H_theta, -np.inf * np.ones(1), new_b_max
                    )

                # 更新下界约束: H_theta * x >= new_b_min
                if (edge_id, 'min') in self.constraint_objects:
                    self.constraint_objects[(edge_id, 'min')].UpdateCoefficients(
                        H_theta, new_b_min, np.inf * np.ones(1)
                    )

    def get_constraint_count(self):
        """获取当前约束数量"""
        return len(self.constraint_bindings)

    def cleanup(self):
        """清理约束（如果需要）"""
        # 当前实现中，约束由GCS对象管理，无需手动清理
        pass
