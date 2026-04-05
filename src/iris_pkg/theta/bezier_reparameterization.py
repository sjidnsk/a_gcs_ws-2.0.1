"""
贝塞尔曲线重新参数化模块

将 GCS 优化后的贝塞尔曲线重新参数化，使得 (u, w) 始终在单位圆上。

核心思路：
1. 提取 GCS 优化后的贝塞尔控制点
2. 对每个控制点，投影 (u, w) 到单位圆
3. 重新构建贝塞尔曲线
4. 检查连续性，必要时调整

优点：
- 保持贝塞尔曲线的结构
- 轨迹相对平滑
- 保证 u² + w² = 1

缺点：
- 实现复杂
- 可能需要迭代调整

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings

try:
    from pydrake.trajectories import (
    BsplineTrajectory,      # B样条轨迹
    BsplineTrajectory_,     # 带类型的B样条轨迹
    Trajectory,             # 轨迹基类
)
    DRAKE_AVAILABLE = True
except ImportError:
    DRAKE_AVAILABLE = False
    warnings.warn("Drake未安装，部分功能不可用")


@dataclass
class ReparameterizationConfig:
    """重新参数化配置"""
    # 投影方法
    projection_method: str = "radial"  # "radial" (径向投影) 或 "tangent" (切线投影)
    
    # 连续性检查
    check_continuity: bool = True
    continuity_order: int = 2  # 检查 C0, C1, C2 连续性
    continuity_tolerance: float = 1e-3
    
    # 迭代调整
    enable_iterative_refinement: bool = True
    max_iterations: int = 10
    refinement_tolerance: float = 1e-6
    
    # 平滑参数
    enable_smoothing: bool = True
    smoothing_window: int = 5
    smoothing_iterations: int = 3
    
    # 边界保护参数
    preserve_boundary: bool = True  # 是否保护边界控制点（保持起点和终点速度为零）
    boundary_padding: int = 2  # 边界保护范围（从起点和终点各保留多少个控制点不变）


class BezierReparameterizer:
    """
    贝塞尔曲线重新参数化器
    
    将 GCS 优化后的贝塞尔曲线重新参数化，使得 (u, w) 始终在单位圆上。
    """
    
    def __init__(self, config: Optional[ReparameterizationConfig] = None):
        """
        初始化重新参数化器
        
        Args:
            config: 重新参数化配置
        """
        self.config = config or ReparameterizationConfig()
    
    def reparameterize_trajectory(
        self,
        trajectory: Any,
        dimension: int = 4
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        重新参数化轨迹
        
        Args:
            trajectory: GCS 优化后的轨迹（BsplineTrajectory）
            dimension: 轨迹维度（默认为 4，即 x, y, u, w）
        
        Returns:
            (reparameterized_trajectory, metrics): 重新参数化后的轨迹和性能指标
        """
        if not DRAKE_AVAILABLE:
            raise RuntimeError("Drake未安装，无法重新参数化轨迹")
        
        # 提取贝塞尔曲线参数
        basis = trajectory.basis()
        control_points = trajectory.control_points()
        
        # 确保 control_points 是 numpy 数组
        if isinstance(control_points, list):
            control_points = np.array(control_points)
        
        # 检查控制点形状并标准化为 (num_dims, num_points)
        control_points = self._normalize_control_points_shape(control_points)
        
        # 检查维度
        if control_points.shape[0] > dimension:
            # 提取空间控制点（前 dimension 行）
            control_points = control_points[:dimension, :]
        
        if control_points.shape[0] != dimension:
            raise ValueError(f"控制点维度不匹配：期望 {dimension}，实际 {control_points.shape[0]}")
        
        # 验证控制点数量与基函数数量匹配
        if control_points.shape[1] != basis.num_basis_functions():
            print(f"  [WARNING] 控制点数量 {control_points.shape[1]} != 基函数数量 {basis.num_basis_functions()}")
            print(f"  [WARNING] 这可能导致 B 样条轨迹创建失败")
        
        # 提取 (u, w) 控制点（假设是 4D 轨迹，u 和 w 在最后两个维度）
        u_control = control_points[2, :].flatten()
        w_control = control_points[3, :].flatten()
        
        # 记录原始指标
        original_metrics = self._compute_metrics(u_control, w_control)
        
        # 步骤 1：投影控制点到单位圆
        u_proj, w_proj = self._project_control_points(
            u_control, w_control, 
            method=self.config.projection_method
        )
        
        # 步骤 2：重新构建控制点（使用原地操作避免拷贝）
        control_points_proj = control_points.copy()
        control_points_proj[2, :] = u_proj
        control_points_proj[3, :] = w_proj
        
        # 步骤 3：重新构建轨迹
        trajectory_proj = BsplineTrajectory(basis, control_points_proj)
        
        # 步骤 4：检查连续性
        if self.config.check_continuity:
            continuity_ok, continuity_metrics = self._check_continuity(
                trajectory_proj, 
                order=self.config.continuity_order,
                tolerance=self.config.continuity_tolerance
            )
            
            if not continuity_ok and self.config.enable_iterative_refinement:
                # 步骤 5：迭代调整
                trajectory_proj, metrics = self._iterative_refinement(
                    trajectory_proj,
                    control_points_proj,
                    original_metrics,
                    u_proj,
                    w_proj,
                    continuity_metrics
                )
            else:
                # 避免重复计算投影指标
                projected_metrics = self._compute_metrics(u_proj, w_proj)
                metrics = {
                    'original': original_metrics,
                    'projected': projected_metrics,
                    'continuity': continuity_metrics
                }
        else:
            # 避免重复计算投影指标
            projected_metrics = self._compute_metrics(u_proj, w_proj)
            metrics = {
                'original': original_metrics,
                'projected': projected_metrics
            }
        
        # 步骤 6：平滑（如果启用）
        if self.config.enable_smoothing:
            trajectory_proj, smoothing_metrics = self._smooth_trajectory(
                trajectory_proj,
                window_size=self.config.smoothing_window,
                iterations=self.config.smoothing_iterations
            )
            metrics['smoothing'] = smoothing_metrics
        
        return trajectory_proj, metrics
    
    def _normalize_control_points_shape(self, control_points: np.ndarray) -> np.ndarray:
        """
        标准化控制点形状为 (num_dims, num_points)
        
        Args:
            control_points: 控制点数组，可能是 (num_dims, num_points) 或 (num_points, num_dims, batch)
        
        Returns:
            标准化后的控制点数组
        """
        # 检查控制点形状
        if len(control_points.shape) == 3:
            # 形状是 (num_points, num_dims, batch)，需要转置
            control_points = control_points.squeeze()  # 移除批次维度
            if control_points.shape[0] > control_points.shape[1]:
                # 如果第一个维度更大，说明是 (num_points, num_dims)，需要转置
                control_points = control_points.T
        elif control_points.shape[0] > control_points.shape[1]:
            # 如果第一个维度更大，说明是 (num_points, num_dims)，需要转置
            control_points = control_points.T
        
        return control_points
    
    def _project_control_points(
        self,
        u_control: np.ndarray,
        w_control: np.ndarray,
        method: str = "radial"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        投影控制点到单位圆
        
        Args:
            u_control: u 控制点 (N,)
            w_control: w 控制点 (N,)
            method: 投影方法 ("radial" 或 "tangent")
        
        Returns:
            (u_proj, w_proj): 投影后的控制点
        """
        if method == "radial":
            # 径向投影：直接投影到单位圆（使用向量化操作）
            norm = np.sqrt(u_control**2 + w_control**2)
            norm = np.where(norm < 1e-10, 1.0, norm)
            u_proj = u_control / norm
            w_proj = w_control / norm
        
        elif method == "tangent":
            # 切线投影：保持切线方向，投影到圆弧
            # 计算每个点到原点的角度
            angles = np.arctan2(w_control, u_control)
            u_proj = np.cos(angles)
            w_proj = np.sin(angles)
        
        else:
            raise ValueError(f"不支持的投影方法: {method}")
        
        return u_proj, w_proj
    
    def _compute_metrics(
        self,
        u_control: np.ndarray,
        w_control: np.ndarray
    ) -> Dict[str, float]:
        """
        计算控制点的性能指标
        
        Args:
            u_control: u 控制点
            w_control: w 控制点
        
        Returns:
            性能指标字典
        """
        metrics = {}
        
        # 单位圆偏差
        norm = np.sqrt(u_control**2 + w_control**2)
        metrics['unit_circle_deviation_max'] = np.max(np.abs(norm - 1.0))
        metrics['unit_circle_deviation_mean'] = np.mean(np.abs(norm - 1.0))
        metrics['unit_circle_deviation_std'] = np.std(np.abs(norm - 1.0))
        
        # 角度变化
        angles = np.arctan2(w_control, u_control)
        angles_unwrapped = np.unwrap(angles)
        angle_diff = np.diff(angles_unwrapped)
        
        if len(angle_diff) > 0:
            metrics['angle_change_max'] = np.max(np.abs(angle_diff))
            metrics['angle_change_mean'] = np.mean(np.abs(angle_diff))
            metrics['angle_change_std'] = np.std(angle_diff)
        else:
            metrics['angle_change_max'] = 0.0
            metrics['angle_change_mean'] = 0.0
            metrics['angle_change_std'] = 0.0
        
        return metrics
    
    def _check_continuity(
        self,
        trajectory: BsplineTrajectory,
        order: int = 2,
        tolerance: float = 1e-3
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        检查轨迹的连续性
        
        Args:
            trajectory: 贝塞尔轨迹
            order: 检查的连续性阶数（0=C0, 1=C1, 2=C2）
            tolerance: 容差
        
        Returns:
            (is_continuous, metrics): 是否连续和指标
        """
        if not DRAKE_AVAILABLE:
            return False, {'error': 'Drake未安装'}
        
        metrics = {}
        is_continuous = True
        
        # 在连接点处采样
        t_samples = np.linspace(0.0, 1.0, 11)
        
        for deriv in range(order + 1):
            # 计算导数
            deriv_traj = trajectory.MakeDerivative(deriv)
            deriv_values = deriv_traj.vector_values(t_samples)
            
            # 检查导数的变化是否平滑（使用向量化操作）
            # 计算相邻点之间的差异
            diff = np.diff(deriv_values, axis=1)
            abs_diff = np.abs(diff)
            
            # 向量化计算所有维度的最大值和平均值
            max_diffs = np.max(abs_diff, axis=1)
            mean_diffs = np.mean(abs_diff, axis=1)
            
            # 检查是否有突变
            for dim in range(deriv_values.shape[0]):
                metrics[f'deriv{deriv}_dim{dim}_max_diff'] = max_diffs[dim]
                metrics[f'deriv{deriv}_dim{dim}_mean_diff'] = mean_diffs[dim]
                
                # 如果突变过大，认为不连续
                if max_diffs[dim] > tolerance:
                    is_continuous = False
        
        return is_continuous, metrics
    
    def _iterative_refinement(
        self,
        trajectory: BsplineTrajectory,
        control_points: np.ndarray,
        original_metrics: Dict[str, float],
        u_proj: np.ndarray,
        w_proj: np.ndarray,
        continuity_metrics: Dict[str, Any],
    ) -> Tuple[BsplineTrajectory, Dict[str, Any]]:
        """
        迭代调整控制点，改善连续性
        
        Args:
            trajectory: 当前轨迹
            control_points: 当前控制点
            original_metrics: 原始指标
            u_proj: 投影后的 u 控制点
            w_proj: 投影后的 w 控制点
            continuity_metrics: 初始连续性指标
        
        Returns:
            (refined_trajectory, metrics): 精炼后的轨迹和指标
        """
        metrics = {'iterations': [], 'original': original_metrics}
        
        # 预计算初始投影指标，避免重复计算
        initial_projected_metrics = self._compute_metrics(u_proj, w_proj)
        prev_deviation_max = initial_projected_metrics['unit_circle_deviation_max']
        
        for iteration in range(self.config.max_iterations):
            # 调整控制点
            control_points = self._adjust_control_points(
                control_points,
                continuity_metrics
            )
            
            # 重新构建轨迹
            trajectory = BsplineTrajectory(trajectory.basis(), control_points)
            
            # 计算指标
            u_control = control_points[2, :].flatten()
            w_control = control_points[3, :].flatten()
            current_metrics = self._compute_metrics(u_control, w_control)
            
            metrics['iterations'].append({
                'iteration': iteration,
                'metrics': current_metrics,
                'continuity': continuity_metrics
            })
            
            # 检查收敛
            current_deviation_max = current_metrics['unit_circle_deviation_max']
            improvement = abs(current_deviation_max - prev_deviation_max)
            if improvement < self.config.refinement_tolerance:
                break
            
            prev_deviation_max = current_deviation_max
            
            # 检查连续性（只在需要时检查，减少计算）
            is_continuous, continuity_metrics = self._check_continuity(
                trajectory,
                order=self.config.continuity_order,
                tolerance=self.config.continuity_tolerance
            )
            
            if is_continuous:
                break
        
        # 使用最终的投影指标
        metrics['projected'] = initial_projected_metrics
        
        return trajectory, metrics
    
    def _adjust_control_points(
        self,
        control_points: np.ndarray,
        continuity_metrics: Dict[str, Any]
    ) -> np.ndarray:
        """
        调整控制点以改善连续性
        
        Args:
            control_points: 当前控制点
            continuity_metrics: 连续性指标
        
        Returns:
            调整后的控制点
        """
        adjusted = control_points.copy()
        
        # 简单策略：对 (u, w) 进行平滑
        u_control = control_points[2, :]
        w_control = control_points[3, :]
        
        window = 3
        # 预计算卷积核
        kernel = np.ones(window) / window
        u_smooth = np.convolve(u_control, kernel, mode='same')
        w_smooth = np.convolve(w_control, kernel, mode='same')
        
        # 边界保护：如果启用，保持边界控制点不变
        if self.config.preserve_boundary and len(u_smooth) > 2 * self.config.boundary_padding:
            padding = self.config.boundary_padding
            u_smooth[:padding] = u_control[:padding]
            u_smooth[-padding:] = u_control[-padding:]
            w_smooth[:padding] = w_control[:padding]
            w_smooth[-padding:] = w_control[-padding:]
        
        # 重新投影到单位圆（使用向量化操作）
        norm = np.sqrt(u_smooth**2 + w_smooth**2)
        norm = np.where(norm < 1e-10, 1.0, norm)
        u_smooth = u_smooth / norm
        w_smooth = w_smooth / norm
        
        adjusted[2, :] = u_smooth
        adjusted[3, :] = w_smooth
        
        return adjusted
    
    def _smooth_trajectory(
        self,
        trajectory: BsplineTrajectory,
        window_size: int = 5,
        iterations: int = 3
    ) -> Tuple[BsplineTrajectory, Dict[str, Any]]:
        """
        平滑轨迹
        
        Args:
            trajectory: 轨迹
            window_size: 平滑窗口大小
            iterations: 迭代次数
        
        Returns:
            (smoothed_trajectory, metrics): 平滑后的轨迹和指标
        """
        metrics = {}
        
        control_points = trajectory.control_points()
        if isinstance(control_points, list):
            control_points = np.array(control_points)
        # 不使用 squeeze，保持原始形状
        control_points = np.array(control_points, copy=True)
        basis = trajectory.basis()
        
        # 预计算卷积核
        kernel = np.ones(window_size) / window_size
        
        # 确定控制点形状和索引方式
        shape_info = self._get_control_point_shape_info(control_points)
        
        for iteration in range(iterations):
            # 提取 (u, w) 控制点
            u_control, w_control = self._extract_uw_control_points(control_points, shape_info)
            
            # 平滑
            u_smooth = np.convolve(u_control, kernel, mode='same')
            w_smooth = np.convolve(w_control, kernel, mode='same')
            
            # 边界保护：如果启用，保持边界控制点不变
            if self.config.preserve_boundary and len(u_smooth) > 2 * self.config.boundary_padding:
                padding = self.config.boundary_padding
                u_smooth[:padding] = u_control[:padding]
                u_smooth[-padding:] = u_control[-padding:]
                w_smooth[:padding] = w_control[:padding]
                w_smooth[-padding:] = w_control[-padding:]
            
            # 重新投影到单位圆（使用向量化操作）
            norm = np.sqrt(u_smooth**2 + w_smooth**2)
            norm = np.where(norm < 1e-10, 1.0, norm)
            u_smooth = u_smooth / norm
            w_smooth = w_smooth / norm
            
            # 重新赋值
            self._assign_uw_control_points(control_points, u_smooth, w_smooth, shape_info)
            
            # 重新构建轨迹
            trajectory_smooth = BsplineTrajectory(basis, control_points)
            
            # 计算指标
            u_control_final = control_points[2, :].flatten()
            w_control_final = control_points[3, :].flatten()
            current_metrics = self._compute_metrics(u_control_final, w_control_final)
            metrics[f'iteration_{iteration}'] = current_metrics
        
        return trajectory_smooth, metrics
    
    def _get_control_point_shape_info(self, control_points: np.ndarray) -> Dict[str, Any]:
        """
        获取控制点形状信息
        
        Args:
            control_points: 控制点数组
        
        Returns:
            形状信息字典
        """
        if len(control_points.shape) == 3:
            return {'shape': '3d', 'is_transposed': False}
        elif len(control_points.shape) == 2:
            is_transposed = control_points.shape[0] > control_points.shape[1]
            return {'shape': '2d', 'is_transposed': is_transposed}
        else:
            raise ValueError(f"不支持的 control_points 形状: {control_points.shape}")
    
    def _extract_uw_control_points(
        self,
        control_points: np.ndarray,
        shape_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取 (u, w) 控制点
        
        Args:
            control_points: 控制点数组
            shape_info: 形状信息
        
        Returns:
            (u_control, w_control): u 和 w 控制点
        """
        if shape_info['shape'] == '3d':
            # 形状是 (num_points, num_dims, batch)
            u_control = control_points[:, 2, 0]
            w_control = control_points[:, 3, 0]
        elif shape_info['shape'] == '2d':
            if shape_info['is_transposed']:
                # (num_points, num_dims)
                u_control = control_points[:, 2]
                w_control = control_points[:, 3]
            else:
                # (num_dims, num_points)
                u_control = control_points[2, :]
                w_control = control_points[3, :]
        
        return u_control, w_control
    
    def _assign_uw_control_points(
        self,
        control_points: np.ndarray,
        u_smooth: np.ndarray,
        w_smooth: np.ndarray,
        shape_info: Dict[str, Any]
    ) -> None:
        """
        赋值 (u, w) 控制点
        
        Args:
            control_points: 控制点数组
            u_smooth: 平滑后的 u 控制点
            w_smooth: 平滑后的 w 控制点
            shape_info: 形状信息
        """
        if shape_info['shape'] == '3d':
            control_points[:, 2, 0] = u_smooth
            control_points[:, 3, 0] = w_smooth
        elif shape_info['shape'] == '2d':
            if shape_info['is_transposed']:
                control_points[:, 2] = u_smooth
                control_points[:, 3] = w_smooth
            else:
                control_points[2, :] = u_smooth
                control_points[3, :] = w_smooth


# ==================== 便捷函数 ====================

def reparameterize_bezier_trajectory(
    trajectory: Any,
    dimension: int = 4,
    config: Optional[ReparameterizationConfig] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    便捷函数：重新参数化贝塞尔轨迹
    
    Args:
        trajectory: GCS 优化后的轨迹
        dimension: 轨迹维度
        config: 重新参数化配置
    
    Returns:
        (reparameterized_trajectory, metrics): 重新参数化后的轨迹和性能指标
    """
    reparameterizer = BezierReparameterizer(config)
    return reparameterizer.reparameterize_trajectory(trajectory, dimension)


# ==================== 测试代码 ====================

def test_bezier_reparameterization():
    """测试贝塞尔曲线重新参数化"""
    print("=" * 70)
    print("测试贝塞尔曲线重新参数化")
    print("=" * 70)
    
    if not DRAKE_AVAILABLE:
        print("Drake未安装，跳过测试")
        return
    
    # 创建模拟的贝塞尔轨迹
    from pydrake.math import BsplineBasis
    
    # GCS 使用 order=3，所以 BsplineBasis(4, knots)，有 4 个基函数
    # knots 需要 4 + 4 = 8 个节点
    knots = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
    basis = BsplineBasis(4, knots)
    
    # 创建控制点（模拟 GCS 优化结果，(u, w) 在单位圆内部）
    # 控制点形状：(4, 4) = (维度, 控制点数量)
    control_points = np.array([
        [0.0, 0.0, 1.0, 0.0],    # p0：起点，theta = 0°
        [2.5, 0.5, 0.9, 0.2],    # p1：控制点1，(u, w) 在圆内
        [5.0, 1.0, 0.6, 0.5],    # p2：控制点2，(u, w) 在圆内
        [10.0, 2.0, 0.0, 1.0]   # p3：终点，theta = 90°
    ])
    
    print(f"\n基函数阶数: {basis.order()}")
    print(f"基函数数量: {basis.num_basis_functions()}")
    print(f"控制点形状: {control_points.shape}")
    print(f"控制点(0): {control_points[:, 0]}")
    print(f"控制点(1): {control_points[:, 1]}")
    print(f"控制点(2): {control_points[:, 2]}")
    print(f"控制点(3): {control_points[:, 3]}")
    
    # 创建原始轨迹
    trajectory = BsplineTrajectory(basis, control_points)
    
    # 创建重新参数化器
    config = ReparameterizationConfig(
        projection_method="radial",
        check_continuity=True,
        continuity_order=2,
        enable_iterative_refinement=True,
        enable_smoothing=True,
        smoothing_window=3
    )
    
    reparameterizer = BezierReparameterizer(config)
    
    # 重新参数化
    print("\n步骤 1：重新参数化轨迹")
    trajectory_proj, metrics = reparameterizer.reparameterize_trajectory(trajectory)
    
    # 打印结果
    print("\n" + "=" * 70)
    print("重新参数化结果")
    print("=" * 70)
    
    print("\n原始指标:")
    for key, value in metrics['original'].items():
        print(f"  {key}: {value:.6f}")
    
    print("\n投影后指标:")
    for key, value in metrics['projected'].items():
        print(f"  {key}: {value:.6f}")
    
    if 'continuity' in metrics:
        print("\n连续性指标:")
        for key, value in metrics['continuity'].items():
            print(f"  {key}: {value:.6f}")
    
    if 'iterations' in metrics:
        print(f"\n迭代调整次数: {len(metrics['iterations'])}")
        for i, iter_metrics in enumerate(metrics['iterations']):
            print(f"\n迭代 {i}:")
            for key, value in iter_metrics['metrics'].items():
                print(f"  {key}: {value:.6f}")
    
    if 'smoothing' in metrics:
        print(f"\n平滑迭代次数: {len([k for k in metrics['smoothing'].keys() if k.startswith('iteration_')])}")
    
    # 验证单位圆约束
    control_points_proj = trajectory_proj.control_points()
    if isinstance(control_points_proj, list):
        control_points_proj = np.array(control_points_proj)
    u_proj = control_points_proj[2, :].flatten()
    w_proj = control_points_proj[3, :].flatten()
    norm_proj = np.sqrt(u_proj**2 + w_proj**2)
    
    # 提取原始控制点
    u_control = control_points[2, :].flatten()
    w_control = control_points[3, :].flatten()
    
    print("\n" + "=" * 70)
    print("单位圆约束验证")
    print("=" * 70)
    print(f"最大偏差: {np.max(np.abs(norm_proj - 1.0)):.10f}")
    print(f"平均偏差: {np.mean(np.abs(norm_proj - 1.0)):.10f}")
    print(f"标准差: {np.std(np.abs(norm_proj - 1.0)):.10f}")
    
    if np.allclose(norm_proj, 1.0, atol=1e-6):
        print("\n✓ 单位圆约束满足（误差 < 1e-6）")
    else:
        print("\n✗ 单位圆约束不满足")
    
    # 可视化
    print("\n" + "=" * 70)
    print("可视化")
    print("=" * 70)
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # 创建图形
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig)
        
        # 1. 2D 轨迹 (x, y)
        ax1 = fig.add_subplot(gs[0, 0])
        x_traj = trajectory.vector_values(np.linspace(0, 1, 100))[0, :]
        y_traj = trajectory.vector_values(np.linspace(0, 1, 100))[1, :]
        x_proj = trajectory_proj.vector_values(np.linspace(0, 1, 100))[0, :]
        y_proj = trajectory_proj.vector_values(np.linspace(0, 1, 100))[1, :]
        
        ax1.plot(x_traj, y_traj, 'b-', label='Original Trajectory', linewidth=2, alpha=0.7)
        ax1.plot(x_proj, y_proj, 'r--', label='Reparameterized', linewidth=2, alpha=0.7)
        ax1.scatter(control_points[0, :], control_points[1, :], c='b', s=100, label='Original Control Points', zorder=5)
        ax1.scatter(control_points_proj[0, :], control_points_proj[1, :], c='r', s=100, marker='s', label='Projected Control Points', zorder=5)
        ax1.plot(control_points[0, :], control_points[1, :], 'b:', alpha=0.5)
        ax1.plot(control_points_proj[0, :], control_points_proj[1, :], 'r:', alpha=0.5)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('2D Trajectory (x, y)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 2. 单位圆 (u, w) - 原始
        ax2 = fig.add_subplot(gs[0, 1])
        u_traj = trajectory.vector_values(np.linspace(0, 1, 100))[2, :]
        w_traj = trajectory.vector_values(np.linspace(0, 1, 100))[3, :]
        
        # 绘制单位圆
        theta_circle = np.linspace(0, 2*np.pi, 100)
        ax2.plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', label='Unit Circle', linewidth=2, alpha=0.5)
        
        # 绘制轨迹
        ax2.plot(u_traj, w_traj, 'b-', label='Original Trajectory', linewidth=2, alpha=0.7)
        ax2.scatter(u_control, w_control, c='b', s=100, label='Original Control Points', zorder=5)
        ax2.plot(u_control, w_control, 'b:', alpha=0.5)
        
        # 标记起点和终点
        ax2.scatter([u_control[0]], [w_control[0]], c='green', s=150, marker='o', label='Start', zorder=6)
        ax2.scatter([u_control[-1]], [w_control[-1]], c='red', s=150, marker='s', label='End', zorder=6)
        
        ax2.set_xlabel('u')
        ax2.set_ylabel('w')
        ax2.set_title('Unit Circle (u, w) - Original')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # 3. 单位圆 (u, w) - 投影后
        ax3 = fig.add_subplot(gs[0, 2])
        u_proj_traj = trajectory_proj.vector_values(np.linspace(0, 1, 100))[2, :]
        w_proj_traj = trajectory_proj.vector_values(np.linspace(0, 1, 100))[3, :]
        
        # 绘制单位圆
        ax3.plot(np.cos(theta_circle), np.sin(theta_circle), 'k--', label='Unit Circle', linewidth=2, alpha=0.5)
        
        # 绘制轨迹
        ax3.plot(u_proj_traj, w_proj_traj, 'r-', label='Projected Trajectory', linewidth=2, alpha=0.7)
        ax3.scatter(u_proj, w_proj, c='r', s=100, marker='s', label='Projected Control Points', zorder=5)
        ax3.plot(u_proj, w_proj, 'r:', alpha=0.5)
        
        # 标记起点和终点
        ax3.scatter([u_proj[0]], [w_proj[0]], c='green', s=150, marker='o', label='Start', zorder=6)
        ax3.scatter([u_proj[-1]], [w_proj[-1]], c='red', s=150, marker='s', label='End', zorder=6)
        
        ax3.set_xlabel('u')
        ax3.set_ylabel('w')
        ax3.set_title('Unit Circle (u, w) - Projected')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        # 4. 单位圆偏差
        ax4 = fig.add_subplot(gs[1, 0])
        norm_traj = np.sqrt(u_traj**2 + w_traj**2)
        norm_proj_traj = np.sqrt(u_proj_traj**2 + w_proj_traj**2)
        t = np.linspace(0, 1, 100)
        ax4.plot(t, norm_traj, 'b-', label='Original Trajectory', linewidth=2, alpha=0.7)
        ax4.plot(t, norm_proj_traj, 'r-', label='Projected Trajectory', linewidth=2, alpha=0.7)
        ax4.axhline(y=1.0, color='k', linestyle='--', label='Unit Circle', linewidth=2, alpha=0.5)
        ax4.set_xlabel('Normalized Time t')
        ax4.set_ylabel('||(u, w)||')
        ax4.set_title('Unit Circle Deviation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. 角度变化
        ax5 = fig.add_subplot(gs[1, 1])
        angles_traj = np.arctan2(w_traj, u_traj)
        angles_proj_traj = np.arctan2(w_proj_traj, u_proj_traj)
        angles_traj_unwrapped = np.unwrap(angles_traj)
        angles_proj_traj_unwrapped = np.unwrap(angles_proj_traj)
        
        ax5.plot(t, angles_traj_unwrapped, 'b-', label='Original Trajectory', linewidth=2, alpha=0.7)
        ax5.plot(t, angles_proj_traj_unwrapped, 'r-', label='Projected Trajectory', linewidth=2, alpha=0.7)
        ax5.scatter(np.linspace(0, 1, len(u_control)), np.unwrap(np.arctan2(w_control, u_control)), 
                   c='b', s=100, label='Original Control Points', zorder=5)
        ax5.scatter(np.linspace(0, 1, len(u_proj)), np.unwrap(np.arctan2(w_proj, u_proj)), 
                   c='r', s=100, marker='s', label='Projected Control Points', zorder=5)
        ax5.set_xlabel('Normalized Time t')
        ax5.set_ylabel('theta (rad)')
        ax5.set_title('Angle Change')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 角度变化率
        ax6 = fig.add_subplot(gs[1, 2])
        dtheta_traj = np.diff(angles_traj_unwrapped)
        dtheta_proj = np.diff(angles_proj_traj_unwrapped)
        t_diff = t[:-1]
        
        ax6.plot(t_diff, dtheta_traj, 'b-', label='Original Trajectory', linewidth=2, alpha=0.7)
        ax6.plot(t_diff, dtheta_proj, 'r-', label='Projected Trajectory', linewidth=2, alpha=0.7)
        ax6.set_xlabel('Normalized Time t')
        ax6.set_ylabel('dtheta/dt (rad)')
        ax6.set_title('Angle Change Rate')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. x(t)
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(t, x_traj, 'b-', label='Original Trajectory', linewidth=2, alpha=0.7)
        ax7.plot(t, x_proj, 'r-', label='Projected Trajectory', linewidth=2, alpha=0.7)
        ax7.set_xlabel('Normalized Time t')
        ax7.set_ylabel('x')
        ax7.set_title('x(t)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. y(t)
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(t, y_traj, 'b-', label='Original Trajectory', linewidth=2, alpha=0.7)
        ax8.plot(t, y_proj, 'r-', label='Projected Trajectory', linewidth=2, alpha=0.7)
        ax8.set_xlabel('Normalized Time t')
        ax8.set_ylabel('y')
        ax8.set_title('y(t)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. u(t) 和 w(t)
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(t, u_traj, 'b-', label='u (Original)', linewidth=2, alpha=0.7)
        ax9.plot(t, w_traj, 'b--', label='w (Original)', linewidth=2, alpha=0.7)
        ax9.plot(t, u_proj_traj, 'r-', label='u (Projected)', linewidth=2, alpha=0.7)
        ax9.plot(t, w_proj_traj, 'r--', label='w (Projected)', linewidth=2, alpha=0.7)
        ax9.set_xlabel('Normalized Time t')
        ax9.set_ylabel('u, w')
        ax9.set_title('u(t) and w(t)')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = '/home/kai/WS/a_gcs_ws 2.0.1/bezier_reparameterization_visualization.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 可视化图像已保存到: {output_path}")
        
        plt.close()
        
    except ImportError:
        print("\n⚠ matplotlib not installed, skipping visualization")
        print("  Install with: pip install matplotlib")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_bezier_reparameterization()
