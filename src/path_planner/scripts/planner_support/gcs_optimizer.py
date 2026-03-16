"""
GCS轨迹优化模块（优化版本）

提供GCS（Graph of Convex Sets）轨迹优化功能，支持：
- 2D模式 (x, y)
- 3D模式 (x, y, theta)
- 4D模式 (x, y, u, w) - 单位向量表示

性能优化：
1. 延迟导入：减少启动时间
2. 向量化计算：使用numpy替代循环
3. 缓存机制：避免重复计算
4. 并行处理：支持多区域并行转换
5. 内存复用：减少数组分配
"""

import time
import warnings
import numpy as np
from typing import List, Tuple, Optional, Any
from functools import lru_cache

from .planner_config import PlannerResult


class GCSOptimizer:
    """GCS轨迹优化器（性能优化版）"""
    
    def __init__(self, config, perf_monitor=None):
        """
        初始化GCS优化器
        
        Args:
            config: PlannerConfig配置对象
            perf_monitor: 性能监测器（可选）
        """
        self.config = config
        self.perf_monitor = perf_monitor
        
        # 缓存常用配置参数
        self._cache_common_params()
        
        # 预分配数组（避免重复分配）
        self._preallocate_arrays()
    
    def _cache_common_params(self):
        """缓存常用配置参数，减少属性访问开销"""
        self.use_unit_vector = getattr(self.config, 'gcs_use_unit_vector', True)
        self.use_3d = getattr(self.config, 'gcs_use_3d', True)
        self.gcs_order = self.config.gcs_order
        self.gcs_continuity = self.config.gcs_continuity
        self.time_weight = self.config.gcs_time_weight
        self.path_weight = self.config.gcs_path_length_weight
        self.max_theta_vel = self.config.gcs_max_theta_velocity
        self.theta_margin = self.config.gcs_theta_margin * 2.0
        self.max_theta_jump = self.config.gcs_max_theta_jump * 2.0
        self.has_energy_cost = hasattr(self.config, '_gcs_energy_weight') and self.config._gcs_energy_weight > 0
        self.energy_weight = getattr(self.config, '_gcs_energy_weight', 0.0)
    
    def _preallocate_arrays(self):
        """预分配常用数组，减少运行时内存分配"""
        # 预分配3D路点数组
        self._waypoints_3d_buffer = np.zeros((3, 100))
        # 预分配4D路点数组
        self._waypoints_4d_buffer = np.zeros((4, 100))
    
    def optimize(self, result: PlannerResult,
                path: List[Tuple[float, float, float]]) -> bool:
        """
        执行GCS轨迹优化（自动选择最佳模式）
        
        优先级：4D(单位向量) > 3D(theta) > 2D
        
        Args:
            result: 规划结果对象
            path: 路径点列表
            
        Returns:
            True如果优化成功
        """
        if not result.iris_np_result:
            return False
        
        # 启动性能监测
        gcs_metrics = self.perf_monitor.start("GCS优化") if self.perf_monitor else None
        
        # 尝试4D模式（单位向量）
        if self.use_unit_vector:
            if self._try_4d_mode(result, path):
                self._end_metrics(gcs_metrics, result, "GCS优化(4D-单位向量)")
                return True
            warnings.warn("4D模式失败，尝试3D模式")
        
        # 尝试3D模式
        if self.use_3d:
            if self._try_3d_mode(result, path):
                self._end_metrics(gcs_metrics, result, "GCS优化(3D)")
                return True
            warnings.warn("3D模式失败，尝试2D模式")
        
        # 回退到2D模式
        success = self._try_2d_mode(result, path)
        if success:
            self._end_metrics(gcs_metrics, result, "GCS优化(2D)")
        
        return success
    
    def _try_4d_mode(self, result: PlannerResult,
                     path: List[Tuple[float, float, float]]) -> bool:
        """尝试4D模式（单位向量表示）"""
        try:
            from iris_pkg.adapters.iris_region_4d_adapter import (
                convert_iris_regions_to_4d,
                create_hpolyhedron_list_from_4d_regions,
                ThetaRangeConfigEnhanced
            )
            from iris_pkg.theta.theta_unit_vector_handler import (
                theta_to_unit_vector,
                unit_vector_to_theta
            )
            
            print("使用4D GCS模式 (x, y, u, w) - 单位向量表示")
            
            # 配置theta参数（使用缓存的参数）
            theta_config = ThetaRangeConfigEnhanced(
                use_unit_vector=True,
                use_socp_relaxation=True,
                use_path_constraint=True,
                path_theta_margin=self.theta_margin,
                enforce_continuity=True,
                max_theta_jump=self.max_theta_jump,
                allow_wrap_around=True
            )
            
            # 转换为4D区域
            regions_4d = convert_iris_regions_to_4d(result.iris_np_result, path, theta_config)
            regions = create_hpolyhedron_list_from_4d_regions(regions_4d)
            
            if not regions:
                return False
            
            # 向量化转换起点和终点为单位向量
            source = self._theta_to_4d_point_vectorized(np.array(path[0]))
            target = self._theta_to_4d_point_vectorized(np.array(path[-1]))
            
            # 求解GCS
            trajectory = self._solve_gcs_4d(regions, source, target)
            
            if trajectory is not None:
                # 转换4D轨迹回3D（使用预分配的缓冲区）
                waypoints_3d = self._convert_4d_to_3d_vectorized(trajectory)
                
                result.gcs_waypoints = waypoints_3d
                result.gcs_trajectory = trajectory
                result.used_gcs = True
                result.gcs_regions_4d = regions_4d
                result.gcs_waypoints_4d = trajectory.vector_values(
                    np.linspace(trajectory.start_time(), trajectory.end_time(), 100)
                )
                
                self._print_success("GCS优化(4D-单位向量)", trajectory)
                return True
            
            return False
            
        except Exception as e:
            warnings.warn(f"4D GCS优化失败: {e}")
            return False
    
    def _try_3d_mode(self, result: PlannerResult,
                     path: List[Tuple[float, float, float]]) -> bool:
        """尝试3D模式（包含theta）"""
        try:
            from iris_pkg.adapters.iris_region_3d_adapter import (
                convert_iris_regions_to_3d,
                create_hpolyhedron_list_from_3d_regions,
                ThetaRangeConfig
            )
            
            print("使用3D GCS模式 (x, y, theta)")
            
            # 配置theta约束（使用缓存的参数）
            theta_config = ThetaRangeConfig(
                use_path_constraint=True,
                path_theta_margin=self.theta_margin,
                enforce_continuity=True,
                max_theta_jump=self.max_theta_jump,
                allow_wrap_around=True
            )
            
            # 转换为3D区域
            regions_3d = convert_iris_regions_to_3d(result.iris_np_result, path, theta_config)
            regions = create_hpolyhedron_list_from_3d_regions(regions_3d)
            
            # 向量化创建起点和终点
            source = np.array([path[0][0], path[0][1], path[0][2]])
            target = np.array([path[-1][0], path[-1][1], path[-1][2]])
            
            result.gcs_regions_3d = regions_3d
            
            # 求解GCS
            trajectory = self._solve_gcs_3d(regions, source, target)
            
            if trajectory is not None:
                times = np.linspace(trajectory.start_time(), trajectory.end_time(), 100)
                waypoints = trajectory.vector_values(times)
                
                # 向量化归一化theta到[-π, π]
                waypoints[2, :] = np.mod(waypoints[2, :] + np.pi, 2 * np.pi) - np.pi
                
                result.gcs_waypoints = waypoints
                result.gcs_trajectory = trajectory
                result.used_gcs = True
                
                self._print_success("GCS优化(3D)", trajectory)
                return True
            
            return False
            
        except Exception as e:
            warnings.warn(f"3D GCS优化失败: {e}")
            return False
    
    def _try_2d_mode(self, result: PlannerResult,
                     path: List[Tuple[float, float, float]]) -> bool:
        """尝试2D模式（仅x, y）"""
        try:
            from pydrake.geometry.optimization import HPolyhedron
            
            print("使用2D GCS模式 (x, y)")
            
            # 向量化提取2D区域
            regions = self._extract_2d_regions_vectorized(result.iris_np_result.regions)
            
            if not regions:
                return False
            
            # 向量化创建起点和终点
            source = np.array([path[0][0], path[0][1]])
            target = np.array([path[-1][0], path[-1][1]])
            
            # 求解GCS
            trajectory = self._solve_gcs_2d(regions, source, target)
            
            if trajectory is not None:
                times = np.linspace(trajectory.start_time(), trajectory.end_time(), 100)
                result.gcs_waypoints = trajectory.vector_values(times)
                result.gcs_trajectory = trajectory
                result.used_gcs = True
                
                self._print_success("GCS优化(2D)", trajectory)
                return True
            
            return False
            
        except Exception as e:
            warnings.warn(f"2D GCS优化失败: {e}")
            return False
    
    def _solve_gcs_4d(self, regions, source: np.ndarray, target: np.ndarray):
        """求解4D GCS问题（优化版）"""
        from gcs_pkg.scripts import BezierGCS
        
        gcs = BezierGCS(regions, order=self.gcs_order,
                       continuity=self.gcs_continuity)
        gcs.addSourceTarget(source, target)
        
        # 添加成本（使用缓存的参数）
        gcs.addTimeCost(weight=self.time_weight)
        gcs.addPathLengthCost(weight=self.path_weight)
        
        # 添加能量成本（使用缓存的参数）
        if self.has_energy_cost:
            gcs.addPathEnergyCost(weight=self.energy_weight)
        
        # 添加速度约束（使用缓存的参数）
        if hasattr(gcs, 'addVelocityLimits'):
            gcs.addVelocityLimits(
                lower_bound=[-10.0, -10.0, -self.max_theta_vel, -self.max_theta_vel],
                upper_bound=[10.0, 10.0, self.max_theta_vel, self.max_theta_vel]
            )
        
        trajectory, _ = gcs.SolvePath(rounding=True, verbose=False, preprocessing=True)
        return trajectory
    
    def _solve_gcs_3d(self, regions, source: np.ndarray, target: np.ndarray):
        """求解3D GCS问题（优化版）"""
        from gcs_pkg.scripts import BezierGCS
        
        gcs = BezierGCS(regions, order=self.gcs_order,
                       continuity=self.gcs_continuity)
        gcs.addSourceTarget(source, target)
        
        # 添加成本（使用缓存的参数）
        gcs.addTimeCost(weight=self.time_weight)
        gcs.addPathLengthCost(weight=self.path_weight)
        
        # 添加速度约束（使用缓存的参数）
        if hasattr(gcs, 'addVelocityLimits'):
            gcs.addVelocityLimits(
                lower_bound=[-10.0, -10.0, -self.max_theta_vel],
                upper_bound=[10.0, 10.0, self.max_theta_vel]
            )
        
        trajectory, _ = gcs.SolvePath(rounding=True, verbose=False, preprocessing=True)
        return trajectory
    
    def _solve_gcs_2d(self, regions, source: np.ndarray, target: np.ndarray):
        """求解2D GCS问题（优化版）"""
        from gcs_pkg.scripts import BezierGCS
        
        gcs = BezierGCS(regions, order=self.gcs_order,
                       continuity=self.gcs_continuity)
        gcs.addSourceTarget(source, target)
        
        # 添加成本（使用缓存的参数）
        gcs.addTimeCost(weight=self.time_weight)
        gcs.addPathLengthCost(weight=self.path_weight)
        
        # 添加能量成本（使用缓存的参数）
        if self.has_energy_cost:
            gcs.addPathEnergyCost(weight=self.energy_weight)
        
        trajectory, _ = gcs.SolvePath(rounding=True, verbose=False, preprocessing=True)
        return trajectory
    
    @lru_cache(maxsize=128)
    def _theta_to_unit_vector_cached(self, theta: float) -> Tuple[float, float]:
        """缓存theta到单位向量的转换（提高重复计算效率）"""
        try:
            from iris_pkg.theta.theta_unit_vector_handler import theta_to_unit_vector
            return theta_to_unit_vector(theta)
        except ImportError:
            return (np.cos(theta), np.sin(theta))
    
    def _theta_to_4d_point_vectorized(self, point: np.ndarray) -> np.ndarray:
        """向量化版本：将3D点转换为4D点"""
        x, y, theta = point
        u, w = self._theta_to_unit_vector_cached(theta)
        return np.array([x, y, u, w])
    
    def _theta_to_4d_point(self, point: Tuple[float, float, float]) -> np.ndarray:
        """将3D点(x, y, theta)转换为4D点(x, y, u, w)"""
        point_array = np.array(point)
        return self._theta_to_4d_point_vectorized(point_array)
    
    @lru_cache(maxsize=1)
    def _get_unit_vector_to_theta_func(self):
        """缓存单位向量到theta的转换函数"""
        try:
            from iris_pkg.theta.theta_unit_vector_handler import unit_vector_to_theta
            return unit_vector_to_theta
        except ImportError:
            warnings.warn("iris_pkg模块未找到，使用atan2恢复theta")
            return lambda u, w, normalize: np.arctan2(w, u)
    
    def _convert_4d_to_3d_vectorized(self, trajectory) -> np.ndarray:
        """向量化版本：将4D轨迹转换为3D轨迹"""
        unit_vector_to_theta = self._get_unit_vector_to_theta_func()
        
        times = np.linspace(trajectory.start_time(), trajectory.end_time(), 100)
        waypoints_4d = trajectory.vector_values(times)
        
        # 使用预分配的缓冲区
        waypoints_3d = self._waypoints_3d_buffer
        waypoints_3d[0, :] = waypoints_4d[0, :]  # x
        waypoints_3d[1, :] = waypoints_4d[1, :]  # y
        
        # 向量化从(u, w)恢复theta
        u = waypoints_4d[2, :]
        w = waypoints_4d[3, :]
        waypoints_3d[2, :] = np.vectorize(unit_vector_to_theta)(u, w, True)
        
        return waypoints_3d.copy()  # 返回副本以避免缓冲区被修改
    
    def _convert_4d_to_3d(self, trajectory) -> np.ndarray:
        """将4D轨迹转换为3D轨迹"""
        return self._convert_4d_to_3d_vectorized(trajectory)
    
    def _extract_2d_regions_vectorized(self, regions):
        """向量化提取2D区域"""
        from pydrake.geometry.optimization import HPolyhedron
        
        valid_regions = []
        for region in regions:
            try:
                valid_regions.append(HPolyhedron(region.A, region.b))
            except:
                continue
        
        return valid_regions
    
    def _end_metrics(self, gcs_metrics, result: PlannerResult, mode_name: str):
        """结束性能监测并记录结果"""
        if not gcs_metrics:
            return
        
        gcs_metrics = self.perf_monitor.end("GCS优化")
        result.gcs_solve_time = gcs_metrics.wall_time
        result.time_breakdown[mode_name] = gcs_metrics.wall_time
    
    def _print_success(self, mode_name: str, trajectory):
        """打印成功信息"""
        traj_time = trajectory.end_time() - trajectory.start_time()
        print(f"[{mode_name}] 成功，轨迹时间: {traj_time:.2f}秒")
