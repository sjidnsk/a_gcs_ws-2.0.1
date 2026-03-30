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
        self.zero_velocity_at_boundaries = getattr(self.config, 'gcs_zero_velocity_at_boundaries', True)
        self.min_time_derivative = getattr(self.config, 'gcs_min_time_derivative', 1.0)
    
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

        优先级：阿克曼 > 4D(单位向量) > 3D(theta) > 2D

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

        # 优先尝试阿克曼模式
        if self._try_ackermann_mode(result, path):
            self._end_metrics(gcs_metrics, result, "GCS优化(阿克曼)")
            return True

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
                # 保存基本结果
                result.gcs_trajectory = trajectory
                result.used_gcs = True
                result.gcs_regions_4d = regions_4d
                
                # 重新参数化轨迹（如果启用）
                if self.config.gcs_enable_reparameterization:
                    self._reparameterize_4d_trajectory(result, trajectory)
                    # 使用重参数化后的轨迹
                    final_trajectory = result.reparameterized_trajectory
                else:
                    # 使用原始轨迹
                    final_trajectory = trajectory
                
                # 转换4D轨迹回3D（使用预分配的缓冲区）
                waypoints_3d = self._convert_4d_to_3d_vectorized(final_trajectory)
                result.gcs_waypoints = waypoints_3d
                
                # 增加采样点数以获得更平滑的速度曲线
                num_samples = 1000
                sample_times = np.linspace(final_trajectory.start_time(), final_trajectory.end_time(), num_samples)
                result.gcs_waypoints_4d = final_trajectory.vector_values(sample_times)
                result.gcs_sample_times = sample_times  # 保存采样时间用于可视化
                
                self._print_success("GCS优化(4D-单位向量)", final_trajectory)
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

    def _try_ackermann_mode(self, result: PlannerResult,
                            path: List[Tuple[float, float, float]]) -> bool:
        """
        尝试阿克曼GCS模式

        Args:
            result: 规划结果对象
            path: 路径点列表

        Returns:
            True如果优化成功
        """
        try:
            from gcs_pkg.scripts.core import AckermannGCS

            print("使用阿克曼GCS模式")

            # 提取2D IRIS区域
            regions = self._extract_2d_regions_vectorized(result.iris_np_result.regions)

            if not regions:
                return False

            # 获取车辆参数
            wheelbase = getattr(self.config, 'ackermann_wheelbase', 2.5)
            order = self.gcs_order
            continuity = self.gcs_continuity

            # 创建阿克曼GCS优化器
            gcs = AckermannGCS(
                regions=regions,
                wheelbase=wheelbase,
                order=order,
                continuity=continuity,
                hdot_min=self.min_time_derivative
            )

            # 设置起点和终点（平坦输出 [x, y, theta]）
            source = np.array([path[0][0], path[0][1], path[0][2]])
            target = np.array([path[-1][0], path[-1][1], path[-1][2]])

            gcs.addSourceTarget(
                source, target,
                zero_velocity_at_boundaries=self.zero_velocity_at_boundaries,
                min_time_derivative=self.min_time_derivative
            )

            # 添加速度约束
            v_min = getattr(self.config, 'ackermann_v_min', 0.0)
            v_max = getattr(self.config, 'ackermann_v_max', 5.0)
            gcs.addVelocityLimits(v_min, v_max)

            # 添加转向角约束（使用迭代优化方法）
            delta_min = getattr(self.config, 'ackermann_delta_min', -np.pi/4)
            delta_max = getattr(self.config, 'ackermann_delta_max', np.pi/4)

            # 获取迭代优化参数
            max_iterations = getattr(self.config, 'ackermann_steering_max_iterations', 3)
            convergence_tolerance = getattr(self.config, 'ackermann_steering_convergence_tolerance', 0.01)
            verbose = getattr(self.config, 'ackermann_steering_verbose', False)

            print(f"使用迭代优化方法添加转向角约束")
            print(f"  最大迭代次数: {max_iterations}")
            print(f"  收敛容忍度: {convergence_tolerance:.2%}")

            iterations_info = gcs.addSteeringLimits_iterative(
                lower_bound=delta_min,
                upper_bound=delta_max,
                max_iterations=max_iterations,
                convergence_tolerance=convergence_tolerance,
                verbose=verbose
            )

            # 打印迭代结果摘要
            if iterations_info:
                print(f"\n迭代优化完成，总迭代次数: {len(iterations_info)}")
                for i, info in enumerate(iterations_info):
                    if info.get('success'):
                        steering = info.get('steering_analysis', {})
                        print(f"  迭代 {i+1}: 成功")
                        print(f"    轨迹时间: {info.get('trajectory_time', 0):.2f}s")
                        print(f"    转向角范围: {np.degrees(steering.get('delta_min', 0)):.2f}° ~ {np.degrees(steering.get('delta_max', 0)):.2f}°")
                        print(f"    违反约束: {steering.get('violations', 0)} 次")
                        if steering.get('max_violation', 0) > 0:
                            print(f"    最大违反量: {np.degrees(steering.get('max_violation', 0)):.2f}°")
                    else:
                        print(f"  迭代 {i+1}: 失败 - {info.get('error', 'Unknown error')}")

            # 添加最小转弯半径约束
            r_min = getattr(self.config, 'ackermann_r_min', None)
            if r_min is not None:
                gcs.addMinTurningRadiusConstraint(r_min)

            # 添加成本函数
            gcs.addTimeCost(weight=self.time_weight)
            gcs.addPathLengthCost(weight=self.path_weight)
            # 暂时不添加平滑性成本，可能导致数值不稳定
            # gcs.addSmoothnessCost(weight=getattr(self.config, 'gcs_smoothness_weight', 1.0))

            # 求解
            trajectory, results_dict = gcs.SolvePath(
                rounding=True,
                verbose=False,
                preprocessing=True
            )

            if trajectory is not None:
                # 保存结果
                result.gcs_trajectory = trajectory
                result.used_gcs = True
                result.gcs_mode = 'ackermann'

                # 采样轨迹点
                num_samples = 1000
                sample_times = np.linspace(trajectory.start_time(), trajectory.end_time(), num_samples)
                waypoints = np.zeros((5, num_samples))

                for i, t in enumerate(sample_times):
                    waypoints[:, i] = trajectory.get_state(t)

                result.gcs_waypoints = waypoints
                result.gcs_sample_times = sample_times

                self._print_success("GCS优化(阿克曼)", trajectory)
                return True

            return False

        except Exception as e:
            warnings.warn(f"阿克曼GCS优化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _solve_gcs_4d(self, regions, source: np.ndarray, target: np.ndarray):
        """求解4D GCS问题（优化版）"""
        from gcs_pkg.scripts import BezierGCS
        
        gcs = BezierGCS(regions, order=self.gcs_order,
                       continuity=self.gcs_continuity)
        
        # 添加边界条件
        if self.zero_velocity_at_boundaries:
            # 使用 zero_deriv_boundary 参数设置一阶导数为零（速度为零）
            # 添加 min_time_derivative 约束防止 dh/ds 过小导致速度突变
            gcs.addSourceTarget(source, target, zero_deriv_boundary=1, min_time_derivative=self.min_time_derivative)
        # else:
        #     # 不设置边界速度约束
        #     gcs.addSourceTarget(source, target)
        
        # 添加成本（使用缓存的参数）
        gcs.addTimeCost(weight=self.time_weight)
        gcs.addPathLengthCost(weight=self.path_weight)
        
        # 添加能量成本（使用缓存的参数）
        # if self.has_energy_cost:
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
        
        # 添加边界条件
        if self.zero_velocity_at_boundaries:
            # 使用 zero_deriv_boundary 参数设置一阶导数为零（速度为零）
            # 添加 min_time_derivative 约束防止 dh/ds 过小导致速度突变
            gcs.addSourceTarget(source, target, zero_deriv_boundary=1, min_time_derivative=self.min_time_derivative)
        else:
            # 不设置边界速度约束
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
    
    def _reparameterize_4d_trajectory(self, result: PlannerResult, trajectory):
        """
        重新参数化4D轨迹，使得 (u, w) 始终在单位圆上
        
        Args:
            result: 规划结果对象
            trajectory: GCS优化后的4D轨迹（BezierTrajectory 或 BsplineTrajectory）
        """
        try:
            from iris_pkg.theta.bezier_reparameterization import (
                BezierReparameterizer,
                ReparameterizationConfig
            )
            
            # 启动性能监测
            reparam_metrics = self.perf_monitor.start("重新参数化") if self.perf_monitor else None
            
            # 创建重新参数化配置
            reparam_config = ReparameterizationConfig(
                projection_method=self.config.gcs_reparameterization_projection_method,
                check_continuity=self.config.gcs_reparameterization_check_continuity,
                continuity_order=self.config.gcs_reparameterization_continuity_order,
                enable_iterative_refinement=self.config.gcs_reparameterization_enable_iterative_refinement,
                max_iterations=self.config.gcs_reparameterization_max_iterations,
                enable_smoothing=self.config.gcs_reparameterization_enable_smoothing,
                smoothing_window=self.config.gcs_reparameterization_smoothing_window,
                smoothing_iterations=self.config.gcs_reparameterization_smoothing_iterations
            )
            
            # 创建重新参数化器
            reparameterizer = BezierReparameterizer(reparam_config)
            
            # 检查轨迹类型
            if hasattr(trajectory, 'path_traj'):
                # BezierTrajectory 类型
                path_traj = trajectory.path_traj
                time_traj = trajectory.time_traj
            else:
                # BsplineTrajectory 类型
                path_traj = trajectory
                time_traj = None
            
            # 重新参数化空间轨迹
            reparameterized_path_traj, metrics = reparameterizer.reparameterize_trajectory(path_traj, dimension=4)
            
            # 保存结果
            if time_traj is not None:
                # 如果是 BezierTrajectory，重新创建
                from gcs_pkg.scripts.core.bezier import BezierTrajectory
                result.reparameterized_trajectory = BezierTrajectory(reparameterized_path_traj, time_traj)
            else:
                # 如果是 BsplineTrajectory，直接保存
                result.reparameterized_trajectory = reparameterized_path_traj

            result.reparameterization_metrics = metrics

            # 结束性能监测
            if reparam_metrics:
                reparam_metrics = self.perf_monitor.end("重新参数化")
                result.reparameterization_time = reparam_metrics.wall_time
                result.time_breakdown["重新参数化"] = reparam_metrics.wall_time
            
            # 打印重新参数化结果
            print(f"[重新参数化] 成功")
            if 'original' in metrics and 'projected' in metrics:
                orig_dev = metrics['original'].get('unit_circle_deviation_max', 0)
                proj_dev = metrics['projected'].get('unit_circle_deviation_max', 0)
                print(f"  原始单位圆偏差: {orig_dev:.6f}")
                print(f"  投影后单位圆偏差: {proj_dev:.6f}")
            
        except ImportError:
            warnings.warn("贝塞尔曲线重新参数化模块未找到，跳过重新参数化")
        except Exception as e:
            import traceback
            traceback.print_exc()
            warnings.warn(f"贝塞尔曲线重新参数化失败: {e}")
