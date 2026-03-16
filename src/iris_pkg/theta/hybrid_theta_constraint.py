"""
混合Theta约束策略实现

结合SOCP约束和扇形约束的优势，解决theta非凸性问题：
1. SOCP约束：确保单位圆性质 (u² + w² ≤ 1)
2. 扇形约束：保留theta方向性
3. 自动处理周期性和多周期问题
4. 搜索空间最小化

核心优势：
- 避免周期性冲突
- 保留方向性引导
- 自动归一化多周期范围
- 保证凸性，GCS可解

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings

# Drake导入
try:
    from pydrake.geometry.optimization import HPolyhedron
    from pydrake.solvers import LorentzConeConstraint
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    warnings.warn("Drake未安装，部分功能不可用")


@dataclass
class HybridConstraintConfig:
    """混合约束配置"""
    # SOCP约束参数
    use_socp: bool = True                    # 是否使用SOCP约束
    socp_tolerance: float = 1e-6              # SOCP约束容差
    
    # 扇形约束参数
    use_sector_constraints: bool = True       # 是否使用扇形约束
    sector_approximation_sides: int = 8       # 扇形近似边数
    sector_tolerance: float = 1e-6            # 扇形约束容差
    
    # 多周期处理
    auto_normalize_multi_period: bool = True  # 自动归一化多周期
    
    # 连续性约束
    max_theta_jump: float = np.pi / 4         # 最大角度跳变（45度）
    enforce_continuity: bool = True           # 是否强制连续性
    
    # 边界处理
    allow_wrap_around: bool = True            # 是否允许跨越边界


class HybridThetaConstraintStrategy:
    """
    混合Theta约束策略
    
    结合SOCP约束和扇形约束的优势
    """
    
    def __init__(self, config: Optional[HybridConstraintConfig] = None):
        """
        初始化混合约束策略
        
        Args:
            config: 混合约束配置
        """
        self.config = config or HybridConstraintConfig()
    
    # ==================== 核心方法 ====================
    
    def create_hybrid_constraints(
        self,
        theta_min: float,
        theta_max: float,
        use_sector: Optional[bool] = None
    ) -> Tuple[Optional['LorentzConeConstraint'], List[Tuple[np.ndarray, np.ndarray]]]:
        """
        创建混合约束
        
        Args:
            theta_min: 最小theta值
            theta_max: 最大theta值
            use_sector: 是否使用扇形约束（None表示自动判断）
        
        Returns:
            (socp_constraint, sector_constraints): SOCP约束和扇形约束列表
        """
        # 1. 归一化theta范围
        theta_min_norm, theta_max_norm, num_periods = self._normalize_theta_range(
            theta_min, theta_max
        )
        
        # 2. 创建SOCP约束
        socp_constraint = None
        if self.config.use_socp and DRAKE_AVAILABLE:
            socp_constraint = self._create_socp_constraint()
        
        # 3. 判断是否需要扇形约束
        if use_sector is None:
            use_sector = self._should_use_sector_constraints(
                theta_min_norm, theta_max_norm, num_periods
            )
        
        # 4. 创建扇形约束
        sector_constraints = []
        if use_sector and self.config.use_sector_constraints:
            sector_constraints = self._create_sector_constraints(
                theta_min_norm, theta_max_norm
            )
        
        return socp_constraint, sector_constraints
    
    def create_hybrid_constraints_for_region(
        self,
        A_2d: np.ndarray,
        b_2d: np.ndarray,
        theta_min: float,
        theta_max: float
    ) -> Tuple[np.ndarray, np.ndarray, Optional['LorentzConeConstraint']]:
        """
        为区域创建完整的混合约束
        
        将2D约束扩展为4D约束，并添加混合theta约束
        
        Args:
            A_2d: 2D约束矩阵 (M x 2)
            b_2d: 2D约束向量 (M,)
            theta_min: 最小theta值
            theta_max: 最大theta值
        
        Returns:
            (A_4d, b_4d, socp_constraint): 4D约束矩阵、向量和SOCP约束
        """
        # 创建混合约束
        socp_constraint, sector_constraints = self.create_hybrid_constraints(
            theta_min, theta_max
        )
        
        # 扩展2D约束到4D
        M = A_2d.shape[0]
        A_4d_xy = np.hstack([A_2d, np.zeros((M, 2))])
        b_4d_xy = b_2d
        
        # 添加扇形约束
        A_sector_list = []
        b_sector_list = []
        
        for A_sector, b_sector in sector_constraints:
            A_sector_list.append(A_sector)
            b_sector_list.append(b_sector)
        
        # 合并所有约束
        if len(A_sector_list) > 0:
            A_sector_combined = np.vstack(A_sector_list)
            b_sector_combined = np.concatenate(b_sector_list)
            A_4d = np.vstack([A_4d_xy, A_sector_combined])
            b_4d = np.concatenate([b_4d_xy, b_sector_combined])
        else:
            A_4d = A_4d_xy
            b_4d = b_4d_xy
        
        return A_4d, b_4d, socp_constraint
    
    # ==================== SOCP约束 ====================
    
    def _create_socp_constraint(self) -> Optional['LorentzConeConstraint']:
        """
        创建SOCP约束：u² + w² ≤ 1
        
        使用Drake的LorentzConeConstraint实现
        
        Returns:
            LorentzConeConstraint对象
        """
        if not DRAKE_AVAILABLE:
            warnings.warn("Drake未安装，无法创建SOCP约束")
            return None
        
        # LorentzConeConstraint形式：z₀ ≥ sqrt(z₁² + z₂²)
        # 我们需要：1 ≥ sqrt(u² + w²)
        # 即：[1, u, w] 满足 Lorentz 锥约束
        
        # 构造矩阵使得 z = Ax + b = [1, u, w]
        # x = [u, w]
        A = np.array([
            [0, 0],  # z₀ = 1
            [1, 0],  # z₁ = u
            [0, 1]   # z₂ = w
        ])
        b = np.array([1.0, 0.0, 0.0])
        
        constraint = LorentzConeConstraint(A, b)
        return constraint
    
    # ==================== 扇形约束 ====================
    
    def _create_sector_constraints(
        self,
        theta_min: float,
        theta_max: float
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        创建扇形约束
        
        使用多边形近似单位圆的扇形部分
        
        Args:
            theta_min: 最小theta（归一化后）
            theta_max: 最大theta（归一化后）
        
        Returns:
            约束列表 [(A, b), ...]
        """
        constraints = []
        
        # 处理跨越边界的情况
        if theta_max < theta_min:
            # 跨越边界，分解为两个扇形
            # [theta_min, 2π) 和 [0, theta_max]
            constraints1 = self._create_sector_constraints_single(
                theta_min, 2 * np.pi
            )
            constraints2 = self._create_sector_constraints_single(
                0, theta_max
            )
            constraints = constraints1 + constraints2
        else:
            # 不跨越边界，直接创建
            constraints = self._create_sector_constraints_single(
                theta_min, theta_max
            )
        
        return constraints
    
    def _create_sector_constraints_single(
        self,
        theta_min: float,
        theta_max: float
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        创建单个扇形约束（不跨越边界）
        
        Args:
            theta_min: 最小theta
            theta_max: 最大theta
        
        Returns:
            约束列表 [(A, b), ...]
        """
        constraints = []
        num_sides = self.config.sector_approximation_sides
        
        # 生成扇形边界的角度
        angles = np.linspace(theta_min, theta_max, num_sides + 1)
        
        # 为每条边创建约束
        for i in range(len(angles) - 1):
            # 计算边的法向量（指向外部）
            angle1, angle2 = angles[i], angles[i + 1]
            mid_angle = (angle1 + angle2) / 2
            
            # 法向量
            nx = np.cos(mid_angle)
            ny = np.sin(mid_angle)
            
            # 约束：nx*u + ny*w ≤ 1
            # 在4D空间中：[0, 0, nx, ny] * [x, y, u, w]^T ≤ 1
            A = np.array([[0, 0, nx, ny]])
            b = np.array([1.0])
            
            constraints.append((A, b))
        
        # 添加两条射线约束（扇形的边界）
        # 射线1：从原点经过theta_min的射线
        # 射线2：从原点经过theta_max的射线
        
        # 射线约束：使用垂直于射线的法向量
        # 对于theta_min射线，法向量为 (sin(theta_min), -cos(theta_min))
        # 对于theta_max射线，法向量为 (-sin(theta_max), cos(theta_max))
        
        # 射线1约束
        nx1 = np.sin(theta_min)
        ny1 = -np.cos(theta_min)
        A1 = np.array([[0, 0, nx1, ny1]])
        b1 = np.array([0.0])  # 通过原点
        constraints.append((A1, b1))
        
        # 射线2约束
        nx2 = -np.sin(theta_max)
        ny2 = np.cos(theta_max)
        A2 = np.array([[0, 0, nx2, ny2]])
        b2 = np.array([0.0])  # 通过原点
        constraints.append((A2, b2))
        
        return constraints
    
    # ==================== 多周期处理 ====================
    
    def _normalize_theta_range(
        self,
        theta_min: float,
        theta_max: float
    ) -> Tuple[float, float, int]:
        """
        归一化theta范围到单周期
        
        Args:
            theta_min: 最小theta
            theta_max: 最大theta
        
        Returns:
            (theta_min_norm, theta_max_norm, num_periods): 归一化后的范围和周期数
        """
        if not self.config.auto_normalize_multi_period:
            return theta_min, theta_max, 0
        
        # 计算范围大小
        theta_range = theta_max - theta_min
        
        # 计算跨越的周期数
        num_periods = int(theta_range / (2 * np.pi))
        
        # 归一化到 [0, 2π)
        theta_min_norm = theta_min % (2 * np.pi)
        theta_max_norm = theta_max % (2 * np.pi)
        
        # 处理特殊情况
        if theta_min_norm < 0:
            theta_min_norm += 2 * np.pi
        if theta_max_norm < 0:
            theta_max_norm += 2 * np.pi
        
        return theta_min_norm, theta_max_norm, num_periods
    
    def _should_use_sector_constraints(
        self,
        theta_min: float,
        theta_max: float,
        num_periods: int
    ) -> bool:
        """
        判断是否应该使用扇形约束
        
        Args:
            theta_min: 最小theta（归一化后）
            theta_max: 最大theta（归一化后）
            num_periods: 周期数
        
        Returns:
            True如果应该使用扇形约束
        """
        # 如果跨越多个完整周期，不使用扇形约束
        if num_periods >= 1:
            return False
        
        # 如果范围覆盖整个圆，不使用扇形约束
        if theta_max - theta_min >= 2 * np.pi - self.config.sector_tolerance:
            return False
        
        # 如果跨越边界且范围很大，不使用扇形约束
        if theta_max < theta_min:
            range_size = (2 * np.pi - theta_min) + theta_max
            if range_size >= 2 * np.pi - self.config.sector_tolerance:
                return False
        
        return True
    
    # ==================== 工具方法 ====================
    
    @staticmethod
    def theta_to_unit_vector(theta: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        将theta转换为单位向量 (u, w) = (cos(θ), sin(θ))
        
        Args:
            theta: 角度值
        
        Returns:
            (u, w): 单位向量分量
        """
        u = np.cos(theta)
        w = np.sin(theta)
        return u, w
    
    @staticmethod
    def unit_vector_to_theta(
        u: Union[float, np.ndarray],
        w: Union[float, np.ndarray],
        normalize: bool = True
    ) -> Union[float, np.ndarray]:
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
            theta = np.mod(theta, 2 * np.pi)
        
        return theta
    
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


# ==================== 便捷函数 ====================

def create_hybrid_theta_constraints(
    theta_min: float,
    theta_max: float,
    config: Optional[HybridConstraintConfig] = None
) -> Tuple[Optional['LorentzConeConstraint'], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    便捷函数：创建混合theta约束
    
    Args:
        theta_min: 最小theta值
        theta_max: 最大theta值
        config: 混合约束配置
    
    Returns:
        (socp_constraint, sector_constraints): SOCP约束和扇形约束列表
    """
    strategy = HybridThetaConstraintStrategy(config)
    return strategy.create_hybrid_constraints(theta_min, theta_max)


# ==================== 测试代码 ====================

def test_hybrid_constraint_strategy():
    """测试混合约束策略"""
    print("=" * 70)
    print("测试混合Theta约束策略")
    print("=" * 70)
    
    strategy = HybridThetaConstraintStrategy()
    
    # 测试1: 正常范围 (90° 到 210°)
    print("\n测试1: 正常范围 (90° 到 210°)")
    theta_min = np.pi / 2      # 90°
    theta_max = 7 * np.pi / 6  # 210°
    
    socp, sectors = strategy.create_hybrid_constraints(theta_min, theta_max)
    print(f"  SOCP约束: {'已创建' if socp is not None else '未创建'}")
    print(f"  扇形约束数量: {len(sectors)}")
    
    # 测试2: 跨越边界 (350° 到 10°)
    print("\n测试2: 跨越边界 (350° 到 10°)")
    theta_min = 35 * np.pi / 18  # 350°
    theta_max = np.pi / 18       # 10°
    
    socp, sectors = strategy.create_hybrid_constraints(theta_min, theta_max)
    print(f"  SOCP约束: {'已创建' if socp is not None else '未创建'}")
    print(f"  扇形约束数量: {len(sectors)}")
    
    # 测试3: 多周期 (-2π 到 9π)
    print("\n测试3: 多周期 (-2π 到 9π)")
    theta_min = -2 * np.pi
    theta_max = 9 * np.pi
    
    socp, sectors = strategy.create_hybrid_constraints(theta_min, theta_max)
    print(f"  SOCP约束: {'已创建' if socp is not None else '未创建'}")
    print(f"  扇形约束数量: {len(sectors)}")
    
    # 测试4: 完整圆周 (0 到 2π)
    print("\n测试4: 完整圆周 (0 到 2π)")
    theta_min = 0
    theta_max = 2 * np.pi
    
    socp, sectors = strategy.create_hybrid_constraints(theta_min, theta_max)
    print(f"  SOCP约束: {'已创建' if socp is not None else '未创建'}")
    print(f"  扇形约束数量: {len(sectors)}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_hybrid_constraint_strategy()
