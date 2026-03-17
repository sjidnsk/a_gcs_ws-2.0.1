"""
A*与GCS分层轨迹规划器

实现A*算法和凸集图（GCS）轨迹优化的分层架构：
1. A*路径规划层：全局路径搜索
2. 走廊生成层：基于A*路径生成局部走廊
3. 凸分解层：对走廊配置空间进行IRIS或传统凸分解
4. GCS轨迹优化层：使用凸集图进行轨迹优化

支持两种凸分解模式:
- 传统模式: 分解障碍物为凸块
- IRIS模式: 使用 Drake IRIS 生成凸可行区域（推荐）

配置说明：
- 推荐使用优化配置：IrisNpConfigOptimized
- 提供预定义模板：高安全、快速处理、平衡配置
- 详细说明见：iris_np_config_optimized.py 和 IRIS_NP_CONFIG_README.md
"""

import os
import sys
import warnings
import numpy as np
from typing import List, Tuple, Optional, Any, Dict

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 确保当前目录在路径中（用于导入decomposition子模块）
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from C_space_pkg.partial_corridor import CorridorGenerator, CorridorConfig, CorridorResult
from C_space_pkg.se2 import SE2ConfigurationSpace, RobotShape
from C_space_pkg.obstacles_optimized import binary_map_to_convex_obstacles_optimized
from A_pkg.A_star_fast_optimized import FastSE2AStarPlanner, PlannerConfig as AStarPlannerConfig

# 导入规划器支持模块（使用绝对导入）
from planner_support import (
    PerformanceMonitor,
    PlannerConfig,
    PlannerResult,
    TrajectoryVisualizer,
    GCSOptimizer
)

# IRIS模块导入（仅使用np模式）
IRIS_MODULES = {}
try:
    from iris_pkg import IrisNpRegionGenerator, IrisNpConfig, check_drake_availability
    IRIS_MODULES['np'] = {
        'available': True,
        'Generator': IrisNpRegionGenerator,
        'Config': IrisNpConfig,
        'check': check_drake_availability
    }
except ImportError:
    IRIS_MODULES['np'] = {'available': False}


def check_drake_availability():
    """检查 Drake 是否可用"""
    return any(m['available'] and m['check']() for m in IRIS_MODULES.values())


class HybridAStarGCSPlanner:
    """A*与GCS分层轨迹规划器"""
    
    def __init__(self, c_space: SE2ConfigurationSpace, 
                 config: Optional[PlannerConfig] = None):
        """
        初始化规划器
        
        Args:
            c_space: SE2配置空间
            config: 规划配置（可选）
        """
        self.c_space = c_space
        self.config = config or PlannerConfig()
        self.iris_available = check_drake_availability()
        
        # 创建性能监测器
        self.perf_monitor = PerformanceMonitor(
            enabled=self.config.enable_performance_monitoring,
            verbose=self.config.performance_verbose
        )
        
        # 创建默认机器人形状
        self.default_robot = RobotShape(
            shape_type='rectangle',
            length=1.0,
            width=0.6
        )
        
        # 检查各IRIS模式可用性
        self.iris_mode_available = {
            name: m['available'] and m['check']() 
            for name, m in IRIS_MODULES.items()
        }
        
        # 自动选择可用的IRIS模式
        self._select_iris_mode()
        
        # 创建走廊生成器
        corridor_config = CorridorConfig(
            corridor_width=self.config.corridor_width,
            smooth_path=self.config.smooth_path,
            smooth_window=self.config.smooth_window,
            boundary_margin=self.config.boundary_margin
        )
        self.corridor_generator = CorridorGenerator(c_space, corridor_config)
        
        # 创建IRIS生成器
        self.iris_generators = {}
        if self.config.use_iris:
            self._create_iris_generators()
        
        # 创建可视化器
        self.visualizer = TrajectoryVisualizer(c_space, self.config)
        
        # 创建GCS优化器
        self.gcs_optimizer = GCSOptimizer(self.config, self.perf_monitor)
    
    def _select_iris_mode(self):
        """选择IRIS模式（仅支持np模式）"""
        if not self.config.use_iris or not self.iris_available:
            self.config.use_iris = False
            return
        
        # 检查np模式是否可用
        if not self.iris_mode_available.get('np', False):
            warnings.warn("IRIS np模式不可用，回退到传统方法")
            self.config.use_iris = False
            return
        
        # 强制使用np模式
        self.config.iris_mode = 'np'
    
    def _create_iris_generators(self):
        """创建IRIS生成器 - 仅支持np模式"""
        mode = 'np'  # 强制使用np模式
        if not self.iris_mode_available.get(mode, False):
            return
            
        mod = IRIS_MODULES[mode]
        ConfigClass = mod['Config']
        GeneratorClass = mod['Generator']
        
        # 优先使用优化配置
        if self.config.iris_config is not None:
            iris_config = self.config.iris_config
            print(f"使用优化配置创建IRIS生成器（{mode}模式）")
            
            # 从优化配置中提取参数
            common_params = {
                'iteration_limit': iris_config.iteration_limit,
                'termination_threshold': iris_config.termination_threshold,
                'configuration_space_margin': iris_config.configuration_space_margin,
                'min_seed_distance': iris_config.min_seed_distance,
                'max_seed_points': iris_config.max_seed_points,
                'merge_overlapping_regions': iris_config.merge_overlapping_regions,
                'num_collision_infeasible_samples': iris_config.num_collision_infeasible_samples,
                'num_additional_constraints_infeasible_samples': iris_config.num_additional_constraints_infeasible_samples,
                'enable_collision_cache': iris_config.enable_collision_cache,
                'collision_cache_size': iris_config.collision_cache_size,
                'use_batch_collision_check': iris_config.use_batch_collision_check,
                'enable_parallel_processing': iris_config.enable_parallel_processing,
                'num_parallel_workers': iris_config.num_parallel_workers,
                'adaptive_initial_step': iris_config.adaptive_initial_step,
                'adaptive_min_step': iris_config.adaptive_min_step,
                'adaptive_step_reduction': iris_config.adaptive_step_reduction,
                'num_expansion_directions': iris_config.num_expansion_directions,
                'enable_two_batch_expansion': iris_config.enable_two_batch_expansion,
                'first_batch_seed_interval': iris_config.first_batch_seed_interval,
                'tangent_normal_ratio': iris_config.tangent_normal_ratio,
                'strict_coverage_check': iris_config.strict_coverage_check,
                'enable_visualization': iris_config.enable_visualization,
                'verbose': iris_config.verbose
            }
        else:
            # 向后兼容：使用旧参数
            print(f"使用传统参数创建IRIS生成器（{mode}模式）")
            common_params = {
                'iteration_limit': self.config.iris_iteration_limit,
                'termination_threshold': self.config.iris_termination_threshold,
                'configuration_space_margin': self.config.iris_configuration_space_margin,
                'min_seed_distance': self.config.iris_min_seed_distance,
                'max_seed_points': self.config.iris_max_seed_points,
                'merge_overlapping_regions': self.config.iris_merge_overlapping,
                'num_collision_infeasible_samples': self.config.iris_num_collision_infeasible_samples
            }
        
        self.iris_generators[mode] = GeneratorClass(ConfigClass(**common_params))
    
    def process(self, path: List[Tuple[float, float, float]], 
                robot: Optional[RobotShape] = None) -> PlannerResult:
        """
        执行A*与GCS分层轨迹规划流程
        
        Args:
            path: A*算法规划的路径点列表
            robot: 机器人形状（可选）
            
        Returns:
            PlannerResult: 规划结果
        """
        # 重置性能监测器
        self.perf_monitor.reset()
        
        # 开始总体性能监测
        self.perf_monitor.start("总流程")
        
        result = PlannerResult(config=self.config)
        
        # 使用传入的机器人或默认机器人
        robot = robot if robot is not None else self.default_robot
        
        # Step 1: 生成走廊
        corridor_metrics = self.perf_monitor.start("走廊生成")
        result.corridor_result = self.corridor_generator.generate_corridor(path, robot)
        corridor_metrics = self.perf_monitor.end("走廊生成")
        
        # 获取走廊生成的性能数据
        result.corridor_generation_time = corridor_metrics.wall_time
        result.time_breakdown["走廊生成"] = corridor_metrics.wall_time
        result.memory_usage["走廊生成"] = corridor_metrics.memory_delta
        result.cpu_usage["走廊生成"] = corridor_metrics.cpu_percent_avg
        
        self._print_step("走廊生成", result.corridor_generation_time, 
                        f"面积: {result.corridor_result.corridor_area:.2f}m²")
        
        # Step 2: 凸分解
        if self.config.use_iris and self.iris_available:
            self._run_iris_decomposition(path, result)
        else:
            self._run_traditional_decomposition(result)
        
        result.corridor_area = result.corridor_result.corridor_area
        
        # Step 3: GCS优化（可选）
        if self.config.enable_gcs_optimization and result.used_iris:
            self.gcs_optimizer.optimize(result, path)
        
        # Step 4: 可视化
        if self.config.enable_visualization:
            vis_metrics = self.perf_monitor.start("可视化")
            self.visualizer.visualize(result, path)
            vis_metrics = self.perf_monitor.end("可视化")
            
            result.time_breakdown["可视化"] = vis_metrics.wall_time
            result.memory_usage["可视化"] = vis_metrics.memory_delta
        
        # 结束总体性能监测
        total_metrics = self.perf_monitor.end("总流程")
        
        # 保存性能数据到结果
        result.performance_metrics = total_metrics
        result.total_time = total_metrics.wall_time
        result.performance_summary = self.perf_monitor.get_summary()
        
        # 打印性能报告
        if self.config.enable_performance_monitoring:
            print("\n" + "="*60)
            print(result.get_performance_report())
            print("="*60)
        
        # 保存性能报告
        if self.config.save_performance_report:
            os.makedirs(self.config.output_dir, exist_ok=True)
            report_file = os.path.join(self.config.output_dir, "performance_report.txt")
            self.perf_monitor.generate_report(report_file)
        
        # 保存性能JSON
        if self.config.save_performance_json:
            os.makedirs(self.config.output_dir, exist_ok=True)
            json_file = os.path.join(self.config.output_dir, "performance_data.json")
            self.perf_monitor.export_json(json_file)
        
        return result
    
    def _run_iris_decomposition(self, path: List[Tuple[float, float, float]], 
                                 result: PlannerResult):
        """运行IRIS分解（仅使用np模式）"""
        mode = 'np'  # 强制使用np模式
        
        # 使用生成器方式
        generator = self.iris_generators.get(mode)
        if not generator:
            warnings.warn(f"IRIS {mode} 生成器未初始化")
            return
        
        # 使用性能监测
        stage_name = f"IRIS分解-{mode}"
        decomp_metrics = self.perf_monitor.start(stage_name)
        
        try:
            iris_result = generator.generate_from_path(
                path=path,
                obstacle_map=result.corridor_result.adjusted_c_space_2d,
                resolution=self.c_space.resolution,
                origin=self.c_space.origin
            )
            
            # 保存结果
            result.iris_np_result = iris_result
            
            result.used_iris = True
            result.iris_mode_used = mode
            
            # 统计
            if iris_result:
                result.num_obstacles = iris_result.num_regions
                result.total_vertices = sum(len(r.vertices) for r in iris_result.regions)
                result.obstacle_area = iris_result.total_area
            
        except Exception as e:
            warnings.warn(f"IRIS {mode} 执行失败: {e}")
        
        decomp_metrics = self.perf_monitor.end(stage_name)
        
        # 获取性能数据
        result.decomposition_time = decomp_metrics.wall_time
        result.time_breakdown[f"IRIS-{mode.upper()}"] = decomp_metrics.wall_time
        result.memory_usage[f"IRIS-{mode.upper()}"] = decomp_metrics.memory_delta
        result.cpu_usage[f"IRIS-{mode.upper()}"] = decomp_metrics.cpu_percent_avg
        
        self._print_step(f"IRIS-{mode.upper()}", result.decomposition_time,
                        f"区域数: {result.num_obstacles}, 面积: {result.obstacle_area:.2f}m²")
    
    def _run_traditional_decomposition(self, result: PlannerResult):
        """运行传统分解"""
        # 使用性能监测
        decomp_metrics = self.perf_monitor.start("传统分解")
        
        adjusted_c_space = result.corridor_result.adjusted_c_space_2d
        if adjusted_c_space.dtype != np.uint8:
            adjusted_c_space = adjusted_c_space.astype(np.uint8)
        
        obstacles = binary_map_to_convex_obstacles_optimized(
            adjusted_c_space,
            min_area=self.config.min_obstacle_area,
            simplify_tolerance=self.config.simplify_tolerance,
            decompose_non_convex=self.config.decompose_non_convex,
            decomposition_threshold=self.config.decomposition_threshold
        )
        
        # 转换为世界坐标
        result.convex_obstacles = [
            [self.c_space.grid_to_world(int(px), int(py)) for px, py in obstacle]
            for obstacle in obstacles
        ]
        
        result.used_iris = False
        result.iris_mode_used = "traditional"
        result.num_obstacles = len(result.convex_obstacles)
        result.total_vertices = sum(len(obs) for obs in result.convex_obstacles)
        result.obstacle_area = self._calculate_total_area(result.convex_obstacles)
        
        decomp_metrics = self.perf_monitor.end("传统分解")
        
        # 获取性能数据
        result.decomposition_time = decomp_metrics.wall_time
        result.time_breakdown["传统分解"] = decomp_metrics.wall_time
        result.memory_usage["传统分解"] = decomp_metrics.memory_delta
        result.cpu_usage["传统分解"] = decomp_metrics.cpu_percent_avg
        
        self._print_step("传统分解", result.decomposition_time,
                        f"障碍物数: {result.num_obstacles}, 面积: {result.obstacle_area:.2f}m²")
    
    def _calculate_total_area(self, obstacles: List[List[Tuple[float, float]]]) -> float:
        """计算总面积"""
        total = 0.0
        for obs in obstacles:
            if len(obs) >= 3:
                n = len(obs)
                area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    area += obs[i][0] * obs[j][1] - obs[j][0] * obs[i][1]
                total += abs(area) / 2.0
        return total
    
    def _print_step(self, name: str, elapsed: float, info: str = ""):
        """打印步骤信息"""
        print(f"[{name}] 耗时: {elapsed:.4f}秒 {info}")
