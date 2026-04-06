"""
A*与GCS分层轨迹规划配置模块

提供配置类和结果类，支持：
- IRIS配置管理
- GCS策略预设
- GCS成本预设
- 性能监测配置
"""

import numpy as np
import warnings
from typing import Any, Dict, Optional, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

# 仅在类型检查时导入，避免循环依赖
if TYPE_CHECKING:
    from path_planner.scripts.planner_support.performance_monitor import PerformanceMetrics

# 导入优化配置
try:
    from config.iris import (
        IrisNpConfigOptimized,
        get_high_safety_config,
        get_fast_processing_config,
        get_balanced_config
    )
    OPTIMIZED_CONFIG_AVAILABLE = True
except ImportError:
    OPTIMIZED_CONFIG_AVAILABLE = False
    warnings.warn("优化配置模块未找到，使用默认配置")


@dataclass
class PlannerConfig:
    """
    A*与GCS分层轨迹规划配置
    
    推荐使用方式：
    1. 使用预定义IRIS配置模板：
       from path_planner.util_map.iris_np_config_optimized import get_high_safety_config
       iris_config = get_high_safety_config()
       config = PlannerConfig(iris_config=iris_config)
    
    2. 自定义IRIS配置：
       from path_planner.util_map.iris_np_config_optimized import IrisNpConfigOptimized
       iris_config = IrisNpConfigOptimized(
           num_collision_infeasible_samples=100,
           configuration_space_margin=0.3
       )
       config = PlannerConfig(iris_config=iris_config)
    
    3. 使用默认配置（平衡配置）：
       config = PlannerConfig()
    """
    # 走廊参数
    corridor_width: float = 5.0
    smooth_path: bool = False
    smooth_window: int = 3
    boundary_margin: float = 0.5
    
    # 凸分解模式
    use_iris: bool = True
    iris_mode: str = "np"  # 仅支持 "np" 模式
    
    # 传统凸分解参数
    min_obstacle_area: int = 10
    simplify_tolerance: float = 1.0
    decompose_non_convex: bool = True
    decomposition_threshold: float = 0.8
    
    # IRIS配置 - 推荐使用 IrisNpConfigOptimized
    iris_config: Optional[Any] = None  # IrisNpConfigOptimized 实例
    
    # IRIS参数（向后兼容，如果 iris_config 未提供则使用这些参数）
    iris_iteration_limit: int = 100
    iris_termination_threshold: float = 0.01
    iris_configuration_space_margin: float = 0.2
    iris_min_seed_distance: float = 1.0
    iris_max_seed_points: int = 50
    iris_merge_overlapping: bool = False
    iris_num_collision_infeasible_samples: int = 50
    iris_requires_sample_as_member: bool = True
    
    # GCS参数
    enable_gcs_optimization: bool = True
    gcs_order: int = 4  # 提高到8以获得更平滑的曲线
    gcs_continuity: int = 2  # 提高到C3以获得更平滑的曲线
    gcs_time_weight: float = 0.1
    gcs_path_length_weight: float = 1.0

    # GCS策略配置（快速选择）
    # 可选: "standard", "high_risk", "emergency", "complex", "custom"
    gcs_strategy_preset: str = "standard"
    
    # GCS成本配置（快速选择）
    # 可选: "time_optimal", "path_optimal", "energy_optimal", "balanced", 
    #       "smooth", "lunar_standard", "lunar_high_risk", "lunar_emergency", "lunar_complex", "custom"
    gcs_cost_preset: str = "lunar_standard"
    
    # 自定义成本权重（当gcs_cost_preset="custom"时使用）
    gcs_custom_time_weight: float = 0.2
    gcs_custom_path_weight: float = 1.0
    gcs_custom_energy_weight: float = 100.0
    
    # GCS边界条件配置
    gcs_zero_velocity_at_boundaries: bool = True  # 是否在起点和终点设置零速度约束（改为False以减少突变）
    gcs_min_time_derivative: float = 1.0  # 时间轨迹导数的最小值，防止 dh/ds 过小导致速度突变（降低以允许更平滑的过渡）

    # 阿克曼车辆参数
    ackermann_wheelbase: float = 2.5  # 轴距（米）
    ackermann_v_min: float = 0.0      # 最小速度（米/秒）
    ackermann_v_max: float = 5.0      # 最大速度（米/秒）
    ackermann_delta_min: float = -np.pi/4  # 最小转向角（弧度）
    ackermann_delta_max: float = np.pi/4   # 最大转向角（弧度）
    ackermann_r_min: Optional[float] = None  # 最小转弯半径（米）

    # 可视化
    enable_visualization: bool = True
    save_visualization: bool = False
    output_dir: str = "./output"
    
    # 性能监测
    enable_performance_monitoring: bool = True
    performance_verbose: bool = False
    save_performance_report: bool = False
    save_performance_json: bool = False
    
    def __post_init__(self):
        """初始化后处理：加载预设配置"""
        # 加载IRIS配置
        if self.iris_config is None and OPTIMIZED_CONFIG_AVAILABLE:
            self.iris_config = get_high_safety_config()
            if self.iris_config is not None:
                print("使用默认IRIS配置（IrisNpConfigOptimized）")
        
        # 应用GCS策略预设
        self._apply_gcs_strategy_preset()
        
        # 应用GCS成本预设
        self._apply_gcs_cost_preset()
    
    def _apply_gcs_strategy_preset(self):
        """应用GCS策略预设"""
        try:
            from config.gcs import (
                get_standard_lunar_config,
                get_high_risk_lunar_config,
                get_emergency_lunar_config,
                get_complex_terrain_config,
            )
            
            preset_map = {
                "standard": get_standard_lunar_config,
                "high_risk": get_high_risk_lunar_config,
                "emergency": get_emergency_lunar_config,
                "complex": get_complex_terrain_config,
            }
            
            if self.gcs_strategy_preset in preset_map:
                lunar_config = preset_map[self.gcs_strategy_preset]()
                
                # 应用策略配置
                self.gcs_max_theta_velocity = lunar_config.max_theta_velocity
                self.gcs_max_theta_jump = lunar_config.max_theta_jump
                self.corridor_width = lunar_config.corridor_width
                self.boundary_margin = lunar_config.boundary_margin
                
                # 应用求解器配置
                if hasattr(lunar_config, 'solver_profile'):
                    self.gcs_solver_relaxation_tol = lunar_config.solver_profile.relaxation_tol
                    self.gcs_solver_max_time = lunar_config.solver_profile.max_time
                
                print(f"应用GCS策略预设: {self.gcs_strategy_preset}")
                
        except ImportError:
            warnings.warn("GCS策略配置模块未找到，使用默认配置")
    
    def _apply_gcs_cost_preset(self):
        """应用GCS成本预设"""
        try:
            from config.gcs import CostConfigurator, CostWeights
            
            configurator = CostConfigurator()
            
            # 预设映射
            preset_map = {
                "time_optimal": "time_optimal",
                "path_optimal": "path_optimal",
                "energy_optimal": "energy_optimal",
                "balanced": "balanced",
                "smooth": "smooth",
                "lunar_standard": "lunar_standard",
                "lunar_high_risk": "lunar_high_risk",
                "lunar_emergency": "lunar_emergency",
                "lunar_complex": "lunar_complex",
            }
            
            if self.gcs_cost_preset == "custom":
                # 使用自定义权重
                weights = CostWeights(
                    time=self.gcs_custom_time_weight,
                    path_length=self.gcs_custom_path_weight,
                    energy=self.gcs_custom_energy_weight,
                    regularization_r=self.gcs_custom_regularization_weight,
                    regularization_h=self.gcs_custom_regularization_weight,
                )
                configurator.set_weights(weights)
                print(f"应用自定义GCS成本配置")
                
            elif self.gcs_cost_preset in preset_map:
                # 使用预设
                configurator.set_preset(preset_map[self.gcs_cost_preset])
                print(f"应用GCS成本预设: {self.gcs_cost_preset}")
            
            # 应用权重到配置
            weights = configurator.weights
            self.gcs_time_weight = weights.time
            self.gcs_path_length_weight = weights.path_length
            
            # 存储能量权重（供后续使用）
            self._gcs_energy_weight = weights.energy
            
        except ImportError:
            warnings.warn("GCS成本配置模块未找到，使用默认配置")


@dataclass
class PlannerResult:
    """A*与GCS分层轨迹规划结果"""
    corridor_result: Any = None  # CorridorResult
    convex_obstacles: List[List[Tuple[float, float]]] = field(default_factory=list)
    iris_np_result: Optional[Any] = None  # IRIS np模式结果
    gcs_trajectory: Optional[Any] = None
    gcs_waypoints: Optional[np.ndarray] = None
    gcs_sample_times: Optional[np.ndarray] = None  # 采样时间点
    gcs_mode: str = ""  # GCS模式：'ackermann', '2d'
    gcs_solve_time: float = 0.0
    used_gcs: bool = False
    num_obstacles: int = 0
    total_vertices: int = 0
    corridor_area: float = 0.0
    obstacle_area: float = 0.0
    corridor_generation_time: float = 0.0
    decomposition_time: float = 0.0
    total_time: float = 0.0
    config: PlannerConfig = None
    used_iris: bool = False
    iris_mode_used: str = ""

    # 详细性能指标
    performance_metrics: Optional['PerformanceMetrics'] = None
    performance_summary: Dict = field(default_factory=dict)
    
    # 详细时间分解
    time_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # 内存使用情况
    memory_usage: Dict[str, float] = field(default_factory=dict)
    
    # CPU使用情况
    cpu_usage: Dict[str, float] = field(default_factory=dict)
    
    def get_performance_report(self) -> str:
        """获取性能报告"""
        if not self.performance_metrics:
            return "无性能数据"
        
        lines = [
            "性能分析报告:",
            f"  总耗时: {self.total_time:.4f}s",
            f"  CPU时间: {self.performance_metrics.cpu_time:.4f}s",
            f"  CPU效率: {self.performance_metrics.cpu_percent_avg:.1f}%",
            f"  内存变化: {self.performance_metrics.memory_delta:+.2f}MB",
            "",
            "时间分解:"
        ]
        
        for stage, t in self.time_breakdown.items():
            percentage = (t / self.total_time * 100) if self.total_time > 0 else 0
            lines.append(f"  {stage}: {t:.4f}s ({percentage:.1f}%)")
        
        return "\n".join(lines)
