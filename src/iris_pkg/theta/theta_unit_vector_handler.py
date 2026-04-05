"""
Theta单位向量处理模块

实现theta非凸性问题的解决方案：
1. 单位向量替代策略：使用 (u, w) = (cos(θ), sin(θ)) 替代 θ
2. SOCP松弛：将 u² + w² = 1 松弛为 u² + w² ≤ 1
3. Theta连续性处理：处理角度周期性和连续性约束

核心思想：
- GCS要求约束是凸的，但运动学约束 ẋ = v·cos(θ), ẏ = v·sin(θ) 是非凸的
- 通过引入单位向量 (u, w)，将非凸约束转化为凸约束
- 使用SOCP松弛确保问题的可解性

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

# Drake导入
try:
    from pydrake.geometry.optimization import HPolyhedron
    from pydrake.solvers import (
        Binding,
        Constraint,
        LorentzConeConstraint,
        LinearConstraint,
        LinearEqualityConstraint,
        QuadraticConstraint,
    )
    from pydrake.symbolic import (
        Expression,
        MakeVectorContinuousVariable,
        Variable,
    )
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    warnings.warn("Drake未安装，部分功能不可用")


@dataclass
class UnitVectorConfig:
    """单位向量配置"""
    # 是否使用SOCP松弛
    use_socp_relaxation: bool = True
    
    # 是否强制单位圆约束（不松弛）
    enforce_unit_circle: bool = False
    
    # Theta连续性参数
    max_theta_jump: float = np.pi / 4  # 相邻区域最大角度跳变
    theta_continuity_weight: float = 1.0  # 连续性权重
    
    # 周期性处理
    allow_wrap_around: bool = True  # 是否允许跨越0/2π边界
    normalize_theta: bool = True  # 是否归一化theta到[0, 2π)
    
    # SOCP松弛参数
    socp_tolerance: float = 1e-6  # SOCP约束容差
    min_radius: float = 0.99  # 最小半径（用于松弛）
    max_radius: float = 1.01  # 最大半径（用于松弛）


class ThetaUnitVectorHandler:
    """
    Theta单位向量处理器
    
    处理theta的非凸性问题，提供：
    1. Theta与单位向量的转换
    2. SOCP约束生成
    3. 连续性约束处理
    4. 周期性处理
    """
    
    def __init__(self, config: Optional[UnitVectorConfig] = None):
        """
        初始化处理器
        
        Args:
            config: 单位向量配置
        """
        self.config = config or UnitVectorConfig()
    
    # ==================== 基础转换函数 ====================
    
    @staticmethod
    def theta_to_unit_vector(theta: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        将theta转换为单位向量 (u, w) = (cos(θ), sin(θ))
        
        Args:
            theta: 角度值（标量或数组）
        
        Returns:
            (u, w): 单位向量分量
        """
        u = np.cos(theta)
        w = np.sin(theta)
        return u, w
    
    @staticmethod
    def unit_vector_to_theta(u: Union[float, np.ndarray], 
                            w: Union[float, np.ndarray],
                            normalize: bool = True) -> Union[float, np.ndarray]:
        """
        将单位向量 (u, w) 转换为theta
        
        Args:
            u: cos(θ)分量
            w: sin(θ)分量
            normalize: 是否归一化到[0, 2π)
        
        Returns:
            theta: 角度值
        """
        theta = np.arctan2(w, u)
        
        if normalize:
            # 归一化到[-π, π]（与A*算法保持一致）
            theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        
        return theta
    
    @staticmethod
    def normalize_angle(theta: Union[float, np.ndarray], 
                       range_type: str = '0_2pi') -> Union[float, np.ndarray]:
        """
        归一化角度到指定范围
        
        Args:
            theta: 角度值
            range_type: 归一化范围类型
                - '0_2pi': [0, 2π)
                - 'minus_pi_pi': [-π, π]
        
        Returns:
            归一化后的角度
        """
        if range_type == '0_2pi':
            return np.mod(theta, 2 * np.pi)
        elif range_type == 'minus_pi_pi':
            # 归一化到[-π, π]
            theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
            return theta
        else:
            raise ValueError(f"不支持的range_type: {range_type}")
    
    @staticmethod
    def angle_difference(theta1: float, theta2: float) -> float:
        """
        计算两个角度的最小差值（考虑周期性）
        
        Args:
            theta1: 第一个角度
            theta2: 第二个角度
        
        Returns:
            最小角度差（范围：[-π, π]）
        """
        diff = theta2 - theta1
        # 归一化到[-π, π]
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
    
    # ==================== SOCP约束生成 ====================
    
    def create_socp_constraint_for_unit_vector(self) -> Optional['LorentzConeConstraint']:
        """
        创建SOCP约束：u² + w² ≤ 1
        
        使用Drake的LorentzConeConstraint实现二阶锥约束
        
        Returns:
            LorentzConeConstraint对象（如果Drake可用）
        """
        if not DRAKE_AVAILABLE:
            warnings.warn("Drake未安装，无法创建SOCP约束")
            return None
        
        # LorentzConeConstraint的形式：||Ax + b||₂ ≤ c'x + d
        # 我们需要：sqrt(u² + w²) ≤ 1
        # 即：||[u, w]||₂ ≤ 1
        
        # Drake的LorentzConeConstraint构造函数：
        # LorentzConeConstraint(A, b, eval_type)
        # 约束形式：z₀ ≥ sqrt(z₁² + ... + zₙ²)
        # 其中 z = Ax + b
        
        # 我们需要构造约束：1 ≥ sqrt(u² + w²)
        # 即：[1, u, w] 满足 Lorentz 锥约束
        # z₀ = 1, z₁ = u, z₂ = w
        
        # 构造矩阵使得 z = Ax + b = [1, u, w]
        # x = [u, w]
        # A = [[0, 0], [1, 0], [0, 1]]
        # b = [1, 0, 0]
        
        A = np.array([
            [0, 0],  # z₀ = 1
            [1, 0],  # z₁ = u
            [0, 1]   # z₂ = w
        ])
        b = np.array([1.0, 0.0, 0.0])
        
        # 创建LorentzConeConstraint
        constraint = LorentzConeConstraint(A, b)
        
        return constraint
    
    def create_relaxed_unit_circle_constraints(
        self,
        min_radius: Optional[float] = None,
        max_radius: Optional[float] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        创建松弛的单位圆约束
        
        使用线性约束近似单位圆：
        - u² + w² ≤ max_radius²
        - u² + w² ≥ min_radius²
        
        Args:
            min_radius: 最小半径（默认使用配置）
            max_radius: 最大半径（默认使用配置）
        
        Returns:
            约束列表 [(A, b), ...]
        """
        min_r = min_radius or self.config.min_radius
        max_r = max_radius or self.config.max_radius
        
        constraints = []
        
        # 方法1：使用多边形近似单位圆
        # 生成正N边形来近似单位圆
        num_sides = 8  # 八边形近似
        
        # 外边界约束：u² + w² ≤ max_r²
        # 使用多边形近似
        for i in range(num_sides):
            angle = 2 * np.pi * i / num_sides
            # 法向量
            nx = np.cos(angle)
            ny = np.sin(angle)
            # 约束：nx*u + ny*w ≤ max_r
            A = np.array([[nx, ny]])
            b = np.array([max_r])
            constraints.append((A, b))
        
        # 内边界约束：u² + w² ≥ min_r²
        # 使用多边形近似（反向约束）
        for i in range(num_sides):
            angle = 2 * np.pi * i / num_sides
            # 法向量（反向）
            nx = -np.cos(angle)
            ny = -np.sin(angle)
            # 约束：-nx*u - ny*w ≤ -min_r
            A = np.array([[nx, ny]])
            b = np.array([-min_r])
            constraints.append((A, b))
        
        return constraints
    
    # ==================== 连续性约束 ====================
    
    def create_theta_continuity_constraint(
        self,
        theta1: float,
        theta2: float,
        max_jump: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建theta连续性约束
        
        确保相邻区域的theta差值不超过阈值
        
        Args:
            theta1: 第一个区域的theta
            theta2: 第二个区域的theta
            max_jump: 最大允许跳变（默认使用配置）
        
        Returns:
            (A, b): 线性约束矩阵和向量
        """
        max_j = max_jump or self.config.max_theta_jump
        
        # 计算角度差（考虑周期性）
        diff = self.angle_difference(theta1, theta2)
        
        # 约束：|theta2 - theta1| ≤ max_jump
        # 转化为两个线性约束：
        # theta2 - theta1 ≤ max_jump
        # theta1 - theta2 ≤ max_jump
        
        A = np.array([
            [1, -1],  # theta2 - theta1 ≤ max_jump
            [-1, 1]   # theta1 - theta2 ≤ max_jump
        ])
        b = np.array([max_j, max_j])
        
        return A, b
    
    def create_unit_vector_continuity_constraint(
        self,
        u1: float, w1: float,
        u2: float, w2: float,
        max_angle_diff: Optional[float] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        创建单位向量连续性约束
        
        确保相邻区域的单位向量变化平滑
        
        Args:
            u1, w1: 第一个区域的单位向量
            u2, w2: 第二个区域的单位向量
            max_angle_diff: 最大允许角度差（默认使用配置）
        
        Returns:
            约束列表 [(A, b), ...]
        """
        max_diff = max_angle_diff or self.config.max_theta_jump
        
        # 计算角度差
        theta1 = self.unit_vector_to_theta(u1, w1)
        theta2 = self.unit_vector_to_theta(u2, w2)
        angle_diff = self.angle_difference(theta1, theta2)
        
        constraints = []
        
        # 方法1：限制单位向量的欧氏距离
        # ||(u2, w2) - (u1, w1)||² ≤ 4 * sin²(max_diff/2)
        # 近似：||(u2, w2) - (u1, w1)|| ≤ max_diff（对于小角度）
        max_dist = 2 * np.sin(max_diff / 2)
        
        # 使用线性约束近似
        # 在4维空间 (u1, w1, u2, w2) 中
        # 约束：u2 - u1 ≤ max_dist, u1 - u2 ≤ max_dist
        #       w2 - w1 ≤ max_dist, w1 - w2 ≤ max_dist
        
        A = np.array([
            [0, 0, 1, -1],  # u2 - u1 ≤ max_dist
            [0, 0, -1, 1],  # u1 - u2 ≤ max_dist
            [0, 0, 0, 0],   # w2 - w1 ≤ max_dist
            [0, 0, 0, 0]    # w1 - w2 ≤ max_dist
        ])
        A[2, [0, 2]] = [0, 1]
        A[2, [1, 3]] = [-1, 0]
        A[3, [0, 2]] = [0, -1]
        A[3, [1, 3]] = [1, 0]
        
        b = np.array([max_dist, max_dist, max_dist, max_dist])
        
        constraints.append((A, b))
        
        return constraints
    
    # ==================== 区域扩展 ====================
    
    def expand_2d_region_with_unit_vector(
        self,
        A_2d: np.ndarray,
        b_2d: np.ndarray,
        theta_range: Tuple[float, float],
        use_socp: Optional[bool] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional['LorentzConeConstraint']]:
        """
        将2D区域 (x, y) 扩展为4D区域 (x, y, u, w)
        
        使用单位向量替代theta，并添加SOCP约束
        
        Args:
            A_2d: 2D约束矩阵 (M x 2)
            b_2d: 2D约束向量 (M,)
            theta_range: theta范围 [theta_min, theta_max]
            use_socp: 是否使用SOCP约束（默认使用配置）
        
        Returns:
            (A_4d, b_4d, socp_constraint): 4D约束矩阵、向量和SOCP约束
        """
        use_socp = use_socp if use_socp is not None else self.config.use_socp_relaxation
        
        M = A_2d.shape[0]
        theta_min, theta_max = theta_range
        
        # 扩展2D约束到4D
        # 原始：A_2d * [x, y]^T <= b_2d
        # 扩展：[A_2d, 0, 0] * [x, y, u, w]^T <= b_2d
        A_4d_xy = np.hstack([A_2d, np.zeros((M, 2))])
        b_4d_xy = b_2d
        
        # 添加theta范围约束（转换为u, w约束）
        # theta_min ≤ θ ≤ theta_max
        # 转换为扇形约束
        
        # 方法1：使用线性约束近似扇形
        # 在单位圆上，theta_min和theta_max对应两个点
        u_min, w_min = self.theta_to_unit_vector(theta_min)
        u_max, w_max = self.theta_to_unit_vector(theta_max)
        
        # 扇形约束：使用两条射线约束
        # 射线1：从原点经过(theta_min)的射线
        # 射线2：从原点经过(theta_max)的射线
        
        # 简化：使用theta范围的中点作为参考
        theta_mid = (theta_min + theta_max) / 2
        u_mid, w_mid = self.theta_to_unit_vector(theta_mid)
        
        # 添加约束：u和w在扇形范围内
        # 使用线性约束近似
        constraints_uv = []
        
        # 约束1：u在合理范围内
        # u_min ≤ u ≤ u_max（简化）
        u_range_min = min(u_min, u_max) - 0.1
        u_range_max = max(u_min, u_max) + 0.1
        
        A_u = np.array([
            [0, 0, 1, 0],   # u ≤ u_range_max
            [0, 0, -1, 0]   # -u ≤ -u_range_min
        ])
        b_u = np.array([u_range_max, -u_range_min])
        
        # 约束2：w在合理范围内
        w_range_min = min(w_min, w_max) - 0.1
        w_range_max = max(w_min, w_max) + 0.1
        
        A_w = np.array([
            [0, 0, 0, 1],   # w ≤ w_range_max
            [0, 0, 0, -1]   # -w ≤ -w_range_min
        ])
        b_w = np.array([w_range_max, -w_range_min])
        
        # 合并所有约束
        A_4d = np.vstack([A_4d_xy, A_u, A_w])
        b_4d = np.concatenate([b_4d_xy, b_u, b_w])
        
        # 创建SOCP约束
        socp_constraint = None
        if use_socp and DRAKE_AVAILABLE:
            socp_constraint = self.create_socp_constraint_for_unit_vector()
        
        return A_4d, b_4d, socp_constraint
    
    # ==================== 工具函数 ====================
    
    def check_unit_vector_validity(
        self,
        u: Union[float, np.ndarray],
        w: Union[float, np.ndarray],
        tolerance: float = 1e-3
    ) -> bool:
        """
        检查单位向量是否满足约束
        
        Args:
            u: cos(θ)分量
            w: sin(θ)分量
            tolerance: 容差
        
        Returns:
            True如果满足约束
        """
        norm = np.sqrt(u**2 + w**2)
        
        if self.config.use_socp_relaxation:
            # SOCP松弛：u² + w² ≤ 1
            return np.all(norm <= 1.0 + tolerance)
        else:
            # 严格约束：u² + w² = 1
            return np.all(np.abs(norm - 1.0) <= tolerance)
    
    def project_to_unit_circle(
        self,
        u: Union[float, np.ndarray],
        w: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        将点投影到单位圆上
        
        Args:
            u: u坐标
            w: w坐标
        
        Returns:
            (u_proj, w_proj): 投影后的坐标
        """
        norm = np.sqrt(u**2 + w**2)
        
        # 避免除零
        norm = np.where(norm < 1e-10, 1.0, norm)
        
        u_proj = u / norm
        w_proj = w / norm
        
        return u_proj, w_proj


# ==================== 便捷函数 ====================

def theta_to_unit_vector(theta: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """便捷函数：theta转单位向量"""
    return ThetaUnitVectorHandler.theta_to_unit_vector(theta)


def unit_vector_to_theta(u: Union[float, np.ndarray], 
                        w: Union[float, np.ndarray],
                        normalize: bool = True) -> Union[float, np.ndarray]:
    """便捷函数：单位向量转theta"""
    return ThetaUnitVectorHandler.unit_vector_to_theta(u, w, normalize)


def create_socp_unit_circle_constraint() -> Optional['LorentzConeConstraint']:
    """便捷函数：创建SOCP单位圆约束"""
    handler = ThetaUnitVectorHandler()
    return handler.create_socp_constraint_for_unit_vector()


# ==================== 测试代码 ====================

def test_theta_unit_vector_handler():
    """测试Theta单位向量处理器"""
    print("=" * 60)
    print("测试Theta单位向量处理器")
    print("=" * 60)
    
    handler = ThetaUnitVectorHandler()
    
    # 测试1：theta与单位向量转换
    print("\n测试1：theta与单位向量转换")
    test_thetas = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
    for theta in test_thetas:
        u, w = handler.theta_to_unit_vector(theta)
        theta_recovered = handler.unit_vector_to_theta(u, w)
        print(f"  θ={theta:.4f} -> (u,w)=({u:.4f},{w:.4f}) -> θ'={theta_recovered:.4f}")
    
    # 测试2：角度差计算
    print("\n测试2：角度差计算（考虑周期性）")
    test_pairs = [
        (0, np.pi/4),
        (0, 2*np.pi),
        (np.pi/4, 7*np.pi/4),
        (np.pi, -np.pi)
    ]
    for theta1, theta2 in test_pairs:
        diff = handler.angle_difference(theta1, theta2)
        print(f"  angle_diff({theta1:.4f}, {theta2:.4f}) = {diff:.4f}")
    
    # 测试3：SOCP约束创建
    print("\n测试3：SOCP约束创建")
    if DRAKE_AVAILABLE:
        socp_constraint = handler.create_socp_constraint_for_unit_vector()
        print(f"  SOCP约束创建成功: {socp_constraint is not None}")
    else:
        print("  Drake未安装，跳过SOCP约束测试")
    
    # 测试4：单位向量有效性检查
    print("\n测试4：单位向量有效性检查")
    test_vectors = [
        (1.0, 0.0),      # 有效
        (0.707, 0.707),  # 有效
        (1.5, 0.0),      # 无效（超出单位圆）
        (0.5, 0.5)       # 无效（在单位圆内）
    ]
    for u, w in test_vectors:
        is_valid = handler.check_unit_vector_validity(u, w)
        print(f"  (u,w)=({u:.3f},{w:.3f}), 有效={is_valid}")
    
    # 测试5：投影到单位圆
    print("\n测试5：投影到单位圆")
    test_points = [
        (1.5, 0.0),   # 在圆外
        (0.5, 0.5),   # 在圆内
        (2.0, 2.0)    # 在圆外
    ]
    for u, w in test_points:
        u_proj, w_proj = handler.project_to_unit_circle(u, w)
        print(f"  ({u:.3f},{w:.3f}) -> ({u_proj:.3f},{w_proj:.3f})")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_theta_unit_vector_handler()
