"""
成本计算器接口定义

本模块定义成本计算器的标准接口，使用 Protocol 实现结构化子类型。
"""

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class CostCalculatorInterface(Protocol):
    """
    成本计算器协议接口
    
    定义成本计算的标准接口，支持依赖注入和类型检查。
    任何实现了 compute_cost 和 compute_cost_gradient 方法的类
    都自动满足此接口（结构化子类型）。
    """
    
    def compute_cost(self, control_points: np.ndarray) -> float:
        """
        计算成本值
        
        Args:
            control_points: 控制点数组，形状 (n, 2)
            
        Returns:
            成本值（标量）
        """
        ...
    
    def compute_cost_gradient(self, control_points: np.ndarray) -> np.ndarray:
        """
        计算成本对控制点的梯度
        
        Args:
            control_points: 控制点数组，形状 (n, 2)
            
        Returns:
            梯度数组，形状 (n * 2,)，按 [x0, y0, x1, y1, ...] 顺序排列
        """
        ...
