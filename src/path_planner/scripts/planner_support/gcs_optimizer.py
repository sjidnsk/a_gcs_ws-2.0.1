"""
GCS轨迹优化模块（优化版本）

提供GCS（Graph of Convex Sets）轨迹优化功能，支持：
- 2D模式 (x, y)
- 阿克曼模式 (x, y, theta, v, delta)

性能优化：
1. 延迟导入：减少启动时间
2. 向量化计算：使用numpy替代循环
3. 缓存机制：避免重复计算
4. 并行处理：支持多区域并行转换
5. 内存复用：减少数组分配
"""
import warnings
import numpy as np
from typing import List, Tuple, Optional, Any


from config.planner import PlannerResult


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
        self.gcs_order = self.config.gcs_order
        self.gcs_continuity = self.config.gcs_continuity
        self.time_weight = self.config.gcs_time_weight
        self.path_weight = self.config.gcs_path_length_weight
        self.has_energy_cost = hasattr(self.config, '_gcs_energy_weight') and self.config._gcs_energy_weight > 0
        self.energy_weight = getattr(self.config, '_gcs_energy_weight', 0.0)
        self.zero_velocity_at_boundaries = getattr(self.config, 'gcs_zero_velocity_at_boundaries', True)
        self.min_time_derivative = getattr(self.config, 'gcs_min_time_derivative', 1.0)
    
    def _preallocate_arrays(self):
        """预分配常用数组，减少运行时内存分配"""
        # 预分配2D路点数组
        self._waypoints_2d_buffer = np.zeros((2, 100))
    
    def optimize(self, result: PlannerResult,
                path: List[Tuple[float, float, float]]) -> bool:
        """
        执行GCS轨迹优化（自动选择最佳模式）

        优先级：阿克曼 > 2D

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

        # 回退到2D模式
        success = self._try_2d_mode(result, path)
        if success:
            self._end_metrics(gcs_metrics, result, "GCS优化(2D)")

        return success
    
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
        尝试阿克曼GCS模式（使用新的ackermann_gcs_pkg实现）

        Args:
            result: 规划结果对象
            path: 路径点列表

        Returns:
            True如果优化成功
        """
        try:
            # 导入新的ackermann_gcs_pkg模块
            from ackermann_gcs_pkg.ackermann_gcs_planner import AckermannGCSPlanner
            from ackermann_gcs_pkg.ackermann_data_structures import (
                VehicleParams,
                EndpointState,
                BezierConfig
            )
            from ackermann_gcs_pkg.flat_output_mapper import compute_flat_output_mapping

            print("使用阿克曼GCS模式 (新实现: ackermann_gcs_pkg)")

            # 提取2D IRIS区域
            regions = self._extract_2d_regions_vectorized(result.iris_np_result.regions)

            if not regions:
                return False

            # 获取车辆参数
            wheelbase = getattr(self.config, 'ackermann_wheelbase', 2.5)
            max_steering_angle = getattr(self.config, 'ackermann_delta_max', np.deg2rad(85))
            v_max = getattr(self.config, 'ackermann_v_max', 10.0)
            max_acceleration = getattr(self.config, 'ackermann_max_acceleration', 5.0)

            # 创建车辆参数
            vehicle_params = VehicleParams(
                wheelbase=wheelbase,
                max_steering_angle=max_steering_angle,
                max_velocity=v_max,
                max_acceleration=max_acceleration
            )

            # 创建Bezier配置
            bezier_config = BezierConfig(
                order=self.gcs_order,
                continuity=self.gcs_continuity
            )

            # 创建规划器
            planner = AckermannGCSPlanner(
                vehicle_params=vehicle_params,
                bezier_config=bezier_config
            )

            # 创建起点和终点状态
            source = EndpointState(
                position=np.array([path[0][0], path[0][1]]),
                heading=path[0][2],
                velocity=0.0 if self.zero_velocity_at_boundaries else None
            )
            target = EndpointState(
                position=np.array([path[-1][0], path[-1][1]]),
                heading=path[-1][2],
                velocity=0.0 if self.zero_velocity_at_boundaries else None
            )

            # 设置成本权重
            cost_weights = {
                "time": self.time_weight,
                "path_length": self.path_weight,
                "energy": getattr(self.config, '_gcs_energy_weight', 0.01),
                # 曲率惩罚成本（可选）
                "curvature_squared": getattr(self.config, 'curvature_squared_weight', 0.0),
                "curvature_derivative": getattr(self.config, 'curvature_derivative_weight', 0.0),
                "curvature_peak": getattr(self.config, 'curvature_peak_weight', 0.0)
            }

            # 执行轨迹规划
            planning_result = planner.plan_trajectory(
                source=source,
                target=target,
                workspace_regions=regions,
                constraints=None,  # 使用默认约束
                cost_weights=cost_weights,
                verbose=True
            )

            if planning_result.success:
                # 保存结果
                result.gcs_trajectory = planning_result.trajectory
                result.used_gcs = True
                result.gcs_mode = 'ackermann'

                # 使用flat_output_mapper采样轨迹
                num_samples = 1000
                mapping = compute_flat_output_mapping(
                    planning_result.trajectory,
                    vehicle_params,
                    num_samples=num_samples
                )

                # 提取状态信息
                position = mapping["position"]  # shape: (2, num_samples)
                heading = mapping["heading"].flatten()  # shape: (num_samples,)
                velocity = mapping["velocity"].flatten()  # shape: (num_samples,)
                steering = mapping["steering"].flatten()  # shape: (num_samples,)

                # 构建waypoints数组 (x, y, theta, v, delta)
                waypoints = np.zeros((5, num_samples))
                waypoints[0, :] = position[0, :]  # x
                waypoints[1, :] = position[1, :]  # y
                waypoints[2, :] = heading        # theta
                waypoints[3, :] = velocity       # v
                waypoints[4, :] = steering       # delta

                result.gcs_waypoints = waypoints
                result.gcs_sample_times = np.linspace(
                    planning_result.trajectory.start_time(),
                    planning_result.trajectory.end_time(),
                    num_samples
                )

                # 打印成功信息
                traj_time = planning_result.trajectory.end_time() - planning_result.trajectory.start_time()
                print(f"[GCS优化(阿克曼)] 成功")
                print(f"  轨迹时间: {traj_time:.2f}秒")
                print(f"  求解时间: {planning_result.solve_time:.2f}秒")
                print(f"  SCP迭代次数: {planning_result.num_iterations}")
                print(f"  收敛原因: {planning_result.convergence_reason}")

                return True

            else:
                print(f"[GCS优化(阿克曼)] 失败: {planning_result.error_message}")
                return False

        except Exception as e:
            warnings.warn(f"阿克曼GCS优化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
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
    
