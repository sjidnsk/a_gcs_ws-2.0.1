"""
曲率平方成本计算器

本模块实现曲率平方积分成本的独立计算器，符合 CostCalculatorInterface 接口。
"""

import numpy as np
from typing import Tuple
from scipy.special import roots_legendre

from .ackermann_data_structures import CurvatureCostConfig
from .numerical_safety_utils import safe_curvature_calculation, DEFAULT_SMALL_VALUE
from .cost_calculator_interface import CostCalculatorInterface


class CurvatureSquaredCostCalculator:
    """
    曲率平方成本计算器
    
    实现曲率平方积分成本的计算和梯度计算，符合 CostCalculatorInterface 接口。
    
    J = ∫κ²(s) * ||r'(s)|| ds
    """
    
    def __init__(self, config: CurvatureCostConfig):
        """
        初始化曲率平方成本计算器
        
        Args:
            config: 曲率成本配置
        """
        self.config = config
        self.sampling_points = None
        self.weights = None
        self._init_integration_points()
    
    def _init_integration_points(self):
        """初始化数值积分采样点和权重"""
        if self.config.integration_method == "gauss_legendre":
            points, weights = roots_legendre(self.config.num_integration_points)
            self.sampling_points = (points + 1) / 2
            self.weights = weights / 2
        elif self.config.integration_method == "trapezoid":
            self.sampling_points = np.linspace(0, 1, self.config.num_integration_points)
            self.weights = np.ones(self.config.num_integration_points) / (self.config.num_integration_points - 1)
            self.weights[0] /= 2
            self.weights[-1] /= 2
        elif self.config.integration_method == "simpson":
            n = self.config.num_integration_points
            if n % 2 == 0:
                n += 1
            self.sampling_points = np.linspace(0, 1, n)
            h = 1.0 / (n - 1)
            self.weights = np.ones(n) * h / 3
            self.weights[1:-1:2] = 4 * h / 3
            self.weights[2:-1:2] = 2 * h / 3
        else:
            raise ValueError(f"Unknown integration method: {self.config.integration_method}")
    
    def _eval_bezier(self, control_points: np.ndarray, s: float) -> np.ndarray:
        """
        使用de Casteljau算法计算贝塞尔曲线值
        
        Args:
            control_points: 控制点数组，形状(n, 2)
            s: 参数值，范围[0, 1]
            
        Returns:
            曲线点，形状(2,)
        """
        points = control_points.copy()
        n = len(points)
        
        for k in range(1, n):
            for i in range(n - k):
                points[i] = (1 - s) * points[i] + s * points[i + 1]
        
        return points[0]
    
    def _compute_bezier_derivatives(
        self,
        control_points: np.ndarray,
        s: float,
        order: int = 2
    ) -> Tuple[np.ndarray, ...]:
        """
        计算贝塞尔曲线在参数s处的导数
        
        Args:
            control_points: 控制点数组，形状(n, 2)
            s: 参数值，范围[0, 1]
            order: 导数阶数
            
        Returns:
            导数元组
        """
        n = len(control_points) - 1
        
        if n < 1:
            return (np.zeros(2),)
        
        # 一阶导数控制点
        first_deriv_control = n * (control_points[1:] - control_points[:-1])
        first_deriv = self._eval_bezier(first_deriv_control, s)
        
        if order >= 2 and n >= 2:
            # 二阶导数控制点
            second_deriv_control = n * (n - 1) * (
                control_points[2:] - 2 * control_points[1:-1] + control_points[:-2]
            )
            second_deriv = self._eval_bezier(second_deriv_control, s)
            return first_deriv, second_deriv
        
        return (first_deriv,)
    
    def _compute_curvature_at_point(
        self,
        control_points: np.ndarray,
        s: float
    ) -> Tuple[float, float]:
        """
        计算贝塞尔曲线在参数s处的曲率和速度
        
        Args:
            control_points: 控制点数组
            s: 参数值
            
        Returns:
            (曲率, 速度)
        """
        first_deriv, second_deriv = self._compute_bezier_derivatives(
            control_points, s, order=2
        )
        
        # 使用数值安全工具计算曲率
        kappa, speed = safe_curvature_calculation(
            first_deriv[0], first_deriv[1],
            second_deriv[0], second_deriv[1],
            DEFAULT_SMALL_VALUE
        )
        
        return kappa, speed
    
    def compute_cost(self, control_points: np.ndarray) -> float:
        """
        计算曲率平方积分成本
        
        J = ∫κ²(s) * ||r'(s)|| ds
        
        Args:
            control_points: 贝塞尔曲线控制点，形状(n, 2)
            
        Returns:
            成本值
        """
        cost = 0.0
        
        for s_i, w_i in zip(self.sampling_points, self.weights):
            kappa, speed = self._compute_curvature_at_point(control_points, s_i)
            # 曲率平方积分：∫κ²(s) * ||r'(s)|| ds
            cost += w_i * kappa**2 * speed
        
        return cost
    
    def compute_cost_gradient(self, control_points: np.ndarray) -> np.ndarray:
        """
        计算成本对控制点的梯度（使用有限差分）
        
        Args:
            control_points: 控制点数组，形状(n, 2)
            
        Returns:
            梯度数组，形状(n * 2,)
        """
        n = len(control_points)
        gradient = np.zeros(n * 2)
        epsilon = 1e-6
        
        # 计算当前成本
        cost_current = self.compute_cost(control_points)
        
        # 对每个控制点分量计算偏导数
        for i in range(n):
            for j in range(2):
                # 扰动控制点
                control_points_perturbed = control_points.copy()
                control_points_perturbed[i, j] += epsilon
                
                # 计算扰动后的成本
                cost_perturbed = self.compute_cost(control_points_perturbed)
                
                # 有限差分梯度
                gradient[i * 2 + j] = (cost_perturbed - cost_current) / epsilon
        
        return gradient
