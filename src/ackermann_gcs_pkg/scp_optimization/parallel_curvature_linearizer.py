"""
并行曲率线性化器

本模块实现曲率约束的并行线性化计算。
"""

import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Optional
from functools import partial

from pydrake.solvers import LinearConstraint
from pydrake.symbolic import DecomposeLinearExpressions
from pydrake.trajectories import BsplineTrajectory

from ackermann_gcs_pkg.ackermann_data_structures import (
    ParallelConfig,
    VehicleParams
)


def _linearize_at_point(
    t: float,
    trajectory_derivatives: Tuple[np.ndarray, np.ndarray],
    max_curvature: float
) -> Tuple[float, float, float, float, float]:
    """
    在单个时间点线性化曲率约束（worker函数）
    
    Args:
        t: 时间点
        trajectory_derivatives: (first_deriv, second_deriv) 导数元组
        max_curvature: 最大曲率
    
    Returns:
        (kappa_0, grad_x_dot, grad_y_dot, grad_x_ddot, grad_y_ddot) 元组
    """
    first_deriv, second_deriv = trajectory_derivatives
    
    x_dot = first_deriv[0]
    y_dot = first_deriv[1]
    x_ddot = second_deriv[0]
    y_ddot = second_deriv[1]
    
    # 计算当前曲率κ₀
    denominator = (x_dot**2 + y_dot**2) ** 1.5
    epsilon = 1e-10
    if denominator < epsilon:
        denominator = 1.0
    numerator = x_dot * y_ddot - y_dot * x_ddot
    kappa_0 = numerator / denominator
    
    # 计算曲率梯度∇κ
    denom_1_5 = (x_dot**2 + y_dot**2) ** 2.5
    if denom_1_5 < epsilon:
        denom_1_5 = 1.0
    denom_0_5 = (x_dot**2 + y_dot**2) ** 0.5
    if denom_0_5 < epsilon:
        denom_0_5 = 1.0
    
    grad_x_dot = (3 * x_dot * y_dot * (y_dot * x_ddot - x_dot * y_ddot)) / denom_1_5
    grad_y_dot = (3 * x_dot * y_dot * (x_dot * y_ddot - y_dot * x_ddot)) / denom_1_5
    grad_x_ddot = -y_dot / denom_0_5
    grad_y_ddot = x_dot / denom_0_5
    
    return (kappa_0, grad_x_dot, grad_y_dot, grad_x_ddot, grad_y_ddot)


class ParallelCurvatureLinearizer:
    """
    并行曲率线性化器
    
    使用多进程并行计算多个采样点的曲率约束线性化，
    提高计算效率。
    
    Attributes:
        config: 并行计算配置
        vehicle_params: 车辆参数
        bezier_gcs: GCS求解器实例
        num_processes: 实际使用的进程数
    """
    
    def __init__(
        self,
        config: ParallelConfig,
        vehicle_params: VehicleParams,
        bezier_gcs
    ):
        """
        初始化并行线性化器
        
        Args:
            config: 并行计算配置
            vehicle_params: 车辆参数
            bezier_gcs: AckermannBezierGCS实例
        """
        self.config = config
        self.vehicle_params = vehicle_params
        self.bezier_gcs = bezier_gcs
        
        # 确定进程数
        cpu_count = mp.cpu_count()
        if config.num_processes is None:
            self.num_processes = cpu_count
        else:
            if config.num_processes > cpu_count:
                print(f"[Warning] num_processes ({config.num_processes}) > CPU cores ({cpu_count}), "
                      f"limiting to {cpu_count}")
                self.num_processes = cpu_count
            else:
                self.num_processes = config.num_processes
    
    def linearize_curvature_constraints(
        self,
        trajectory: BsplineTrajectory,
        num_samples: int = 50
    ) -> List[LinearConstraint]:
        """
        并行线性化曲率约束
        
        Args:
            trajectory: 当前轨迹
            num_samples: 采样点数
        
        Returns:
            线性约束列表
        
        Examples:
            >>> linearizer = ParallelCurvatureLinearizer(config, vehicle_params, bezier_gcs)
            >>> constraints = linearizer.linearize_curvature_constraints(trajectory, 50)
            >>> print(f"Generated {len(constraints)} constraints")  # 输出: 100 (50个点，每个2个约束)
        """
        # 生成采样时间点
        t_samples = np.linspace(
            trajectory.start_time(),
            trajectory.end_time(),
            num_samples
        )
        
        # 预计算所有采样点的导数
        derivatives_list = []
        for t in t_samples:
            first_deriv = trajectory.EvalDerivative(t, 1)
            second_deriv = trajectory.EvalDerivative(t, 2)
            derivatives_list.append((first_deriv, second_deriv))
        
        # 并行或串行计算线性化
        if self.config.enable_parallel and num_samples >= 10:
            # 并行计算
            results = self._parallel_linearize(t_samples, derivatives_list)
        else:
            # 串行计算（采样点太少时不值得并行）
            results = self._serial_linearize(t_samples, derivatives_list)
        
        # 组装约束
        constraints = self._assemble_constraints(t_samples, results)
        
        return constraints
    
    def _parallel_linearize(
        self,
        t_samples: np.ndarray,
        derivatives_list: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        并行线性化计算
        
        Args:
            t_samples: 采样时间点数组
            derivatives_list: 导数列表
        
        Returns:
            线性化结果列表
        """
        # 准备参数
        max_curvature = self.vehicle_params.max_curvature
        
        # 分批次
        if self.config.enable_batching and len(t_samples) > self.config.batch_size:
            num_batches = len(t_samples) // self.config.batch_size
            batches = np.array_split(t_samples, num_batches)
            derivatives_batches = np.array_split(derivatives_list, num_batches)
        else:
            batches = [t_samples]
            derivatives_batches = [derivatives_list]
        
        # 并行处理
        all_results = []
        with mp.Pool(self.num_processes) as pool:
            for t_batch, deriv_batch in zip(batches, derivatives_batches):
                # 准备参数
                args = [
                    (t, deriv, max_curvature)
                    for t, deriv in zip(t_batch, deriv_batch)
                ]
                
                # 并行计算
                batch_results = pool.starmap(_linearize_at_point, args)
                all_results.extend(batch_results)
        
        return all_results
    
    def _serial_linearize(
        self,
        t_samples: np.ndarray,
        derivatives_list: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Tuple[float, float, float, float, float]]:
        """
        串行线性化计算
        
        Args:
            t_samples: 采样时间点数组
            derivatives_list: 导数列表
        
        Returns:
            线性化结果列表
        """
        max_curvature = self.vehicle_params.max_curvature
        results = []
        
        for t, deriv in zip(t_samples, derivatives_list):
            result = _linearize_at_point(t, deriv, max_curvature)
            results.append(result)
        
        return results
    
    def _assemble_constraints(
        self,
        t_samples: np.ndarray,
        results: List[Tuple[float, float, float, float, float]]
    ) -> List[LinearConstraint]:
        """
        组装线性约束
        
        Args:
            t_samples: 采样时间点数组
            results: 线性化结果列表
        
        Returns:
            线性约束列表
        """
        constraints = []
        u_vars = self.bezier_gcs.u_vars
        u_r_trajectory = self.bezier_gcs.u_r_trajectory
        
        for i, (t, result) in enumerate(zip(t_samples, results)):
            kappa_0, grad_x_dot, grad_y_dot, grad_x_ddot, grad_y_ddot = result
            
            # 计算控制点对导数的贡献
            first_deriv_at_s = u_r_trajectory.MakeDerivative(1).value(t)
            second_deriv_at_s = u_r_trajectory.MakeDerivative(2).value(t)
            
            # 构建线性表达式
            delta_x_dot = first_deriv_at_s[0]
            delta_y_dot = first_deriv_at_s[1]
            delta_x_ddot = second_deriv_at_s[0]
            delta_y_ddot = second_deriv_at_s[1]
            
            linear_expr = (
                grad_x_dot * delta_x_dot
                + grad_y_dot * delta_y_dot
                + grad_x_ddot * delta_x_ddot
                + grad_y_ddot * delta_y_ddot
            )
            
            # 分解线性表达式
            A = DecomposeLinearExpressions(linear_expr, u_vars)
            
            # 上界约束
            constraints.append(
                LinearConstraint(
                    A,
                    np.array([-np.inf]),
                    np.array([self.vehicle_params.max_curvature - kappa_0]),
                )
            )
            
            # 下界约束
            constraints.append(
                LinearConstraint(
                    A,
                    np.array([-self.vehicle_params.max_curvature - kappa_0]),
                    np.array([np.inf]),
                )
            )
        
        return constraints
    
    def get_parallel_info(self) -> dict:
        """
        获取并行计算信息
        
        Returns:
            包含并行计算信息的字典
        """
        return {
            'num_processes': self.num_processes,
            'enable_parallel': self.config.enable_parallel,
            'enable_batching': self.config.enable_batching,
            'batch_size': self.config.batch_size
        }
