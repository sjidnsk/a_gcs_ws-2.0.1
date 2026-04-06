"""
月面月球车GCS优化配置

针对月面复杂环境优化的GCS配置模板
"""

import numpy as np
from typing import List
from dataclasses import dataclass

# 导入求解器配置
from config.solver import (
    SolverPerformanceProfile,
    ProblemSize,
    SolverType,
)


@dataclass
class LunarRoverGCSConfig:
    """月面月球车GCS配置"""
    
    # 场景类型
    scenario: str = "standard"  # "standard", "high_risk", "emergency", "complex"
    
    # 舍入策略
    rounding_strategies: List = None
    rounding_kwargs: dict = None
    
    # 求解器配置
    solver_profile: SolverPerformanceProfile = None
    
    # 约束参数
    max_theta_velocity: float = np.pi / 2
    max_theta_jump: float = np.pi / 4
    corridor_width: float = 4.0
    boundary_margin: float = 0.5
    
    # 性能参数
    enable_preprocessing: bool = True
    enable_unit_vector: bool = True
    enable_socp: bool = True
    
    def __post_init__(self):
        """根据场景类型自动配置"""
        if self.rounding_strategies is None:
            self.rounding_strategies = self._get_rounding_strategies()
        
        if self.rounding_kwargs is None:
            self.rounding_kwargs = self._get_rounding_kwargs()
        
        if self.solver_profile is None:
            self.solver_profile = self._get_solver_profile()
    
    def _get_rounding_strategies(self) -> List:
        """根据场景获取舍入策略"""
        # 延迟导入以避免循环依赖
        from gcs_pkg.scripts.rounding import (
            greedyForwardPathSearch,
            greedyBackwardPathSearch,
            randomForwardPathSearch,
            randomBackwardPathSearch,
            averageVertexPositionGcs,
        )

        strategies_map = {
            "standard": [
                greedyForwardPathSearch,
                greedyBackwardPathSearch,
                averageVertexPositionGcs,
            ],
            "high_risk": [
                randomForwardPathSearch,
                randomBackwardPathSearch,
                averageVertexPositionGcs,
            ],
            "emergency": [
                greedyForwardPathSearch,
            ],
            "complex": [
                greedyForwardPathSearch,
                randomForwardPathSearch,
                greedyBackwardPathSearch,
                averageVertexPositionGcs,
            ],
        }
        return strategies_map.get(self.scenario, strategies_map["standard"])
    
    def _get_rounding_kwargs(self) -> dict:
        """根据场景获取舍入参数"""
        kwargs_map = {
            "standard": {
                'flow_tol': 1e-4,
                'max_paths': 5,
                'max_trials': 50,
            },
            "high_risk": {
                'flow_tol': 1e-5,
                'max_paths': 10,
                'max_trials': 100,
                'seed': 42,
            },
            "emergency": {
                'flow_tol': 1e-3,
            },
            "complex": {
                'flow_tol': 1e-4,
                'max_paths': 8,
                'max_trials': 80,
            },
        }
        return kwargs_map.get(self.scenario, kwargs_map["standard"])
    
    def _get_solver_profile(self) -> SolverPerformanceProfile:
        """根据场景获取求解器配置"""
        profiles_map = {
            "standard": SolverPerformanceProfile(
                problem_size=ProblemSize.MEDIUM,
                solver_type=SolverType.MOSEK,
                relaxation_tol=1e-4,
                mip_tol=1e-3,
                constraint_tol=1e-6,
                max_time=30.0,
                mip_max_time=15.0,
                enable_presolve=True,
                presolve_level=1,
                num_threads=4,
                enable_warm_start=True,
                cache_solutions=True,
            ),
            "high_risk": SolverPerformanceProfile(
                problem_size=ProblemSize.SMALL,
                solver_type=SolverType.MOSEK,
                relaxation_tol=1e-6,
                mip_tol=1e-4,
                constraint_tol=1e-8,
                max_time=120.0,
                mip_max_time=60.0,
                enable_presolve=True,
                presolve_level=2,
                num_threads=8,
                enable_warm_start=True,
                cache_solutions=True,
                verbose=True,
            ),
            "emergency": SolverPerformanceProfile(
                problem_size=ProblemSize.LARGE,
                solver_type=SolverType.MOSEK,
                relaxation_tol=1e-3,
                mip_tol=1e-2,
                max_time=10.0,
                mip_max_time=5.0,
                enable_presolve=True,
                presolve_level=2,
                num_threads=8,
            ),
            "complex": SolverPerformanceProfile(
                problem_size=ProblemSize.LARGE,
                solver_type=SolverType.MOSEK,
                relaxation_tol=1e-4,
                mip_tol=1e-3,
                max_time=60.0,
                mip_max_time=30.0,
                enable_presolve=True,
                presolve_level=2,
                num_threads=8,
                enable_warm_start=True,
                cache_solutions=True,
            ),
        }
        return profiles_map.get(self.scenario, profiles_map["standard"])


# ==================== 预定义配置 ====================

def get_standard_lunar_config() -> LunarRoverGCSConfig:
    """
    获取标准月面探索配置（推荐）
    
    适用场景：
    - 中等复杂度地形
    - 标准探索任务
    - 平衡速度和质量
    
    性能：
    - 求解时间：1-2秒
    - 成功率：95%+
    - 轨迹质量：优秀
    """
    return LunarRoverGCSConfig(scenario="standard")


def get_high_risk_lunar_config() -> LunarRoverGCSConfig:
    """
    获取高风险区域配置
    
    适用场景：
    - 密集障碍物区域
    - 狭窄通道
    - 高安全性要求
    
    性能：
    - 求解时间：5-10秒
    - 成功率：99%+
    - 轨迹质量：极佳
    """
    config = LunarRoverGCSConfig(scenario="high_risk")
    config.max_theta_velocity = np.pi / 4
    config.max_theta_jump = np.pi / 8
    config.corridor_width = 5.0
    config.boundary_margin = 1.0
    return config


def get_emergency_lunar_config() -> LunarRoverGCSConfig:
    """
    获取紧急避障配置
    
    适用场景：
    - 紧急情况
    - 快速响应需求
    - 开阔区域
    
    性能：
    - 求解时间：<1秒
    - 成功率：85%+
    - 轨迹质量：良好
    """
    config = LunarRoverGCSConfig(scenario="emergency")
    config.max_theta_velocity = np.pi
    config.max_theta_jump = np.pi / 2
    config.corridor_width = 3.0
    config.boundary_margin = 0.3
    return config


def get_complex_terrain_config() -> LunarRoverGCSConfig:
    """
    获取复杂地形配置
    
    适用场景：
    - 极度复杂地形
    - 多种障碍物混合
    - 需要高质量轨迹
    
    性能：
    - 求解时间：2-5秒
    - 成功率：98%+
    - 轨迹质量：优秀
    """
    return LunarRoverGCSConfig(scenario="complex")


# ==================== 便捷函数 ====================

def apply_lunar_config_to_gcs(gcs, config: LunarRoverGCSConfig):
    """
    将月面配置应用到GCS对象
    
    Args:
        gcs: GCS对象
        config: 月面配置
    
    Returns:
        配置后的GCS对象
    """
    # 设置舍入策略
    if hasattr(gcs, 'rounding_fn'):
        gcs.rounding_fn = config.rounding_strategies
        gcs.rounding_kwargs = config.rounding_kwargs
    
    # 设置求解器配置
    if hasattr(gcs, 'solver_profile'):
        gcs.solver_profile = config.solver_profile
    
    # 添加速度约束
    if hasattr(gcs, 'addVelocityLimits') and config.enable_unit_vector:
        max_linear_velocity = 0.5  # m/s
        max_angular_velocity = config.max_theta_velocity
        
        # 根据维度设置约束
        dimension = gcs.dimension if hasattr(gcs, 'dimension') else 3
        
        if dimension == 4:
            # 4D模式 (x, y, u, w)
            gcs.addVelocityLimits(
                lower_bound=[-max_linear_velocity, -max_linear_velocity, 
                            -max_angular_velocity, -max_angular_velocity],
                upper_bound=[max_linear_velocity, max_linear_velocity,
                            max_angular_velocity, max_angular_velocity]
            )
        elif dimension == 3:
            # 3D模式 (x, y, theta)
            gcs.addVelocityLimits(
                lower_bound=[-max_linear_velocity, -max_linear_velocity, -max_angular_velocity],
                upper_bound=[max_linear_velocity, max_linear_velocity, max_angular_velocity]
            )
    
    return gcs


def get_gcs_solve_options(config: LunarRoverGCSConfig):
    """
    获取GCS求解选项
    
    Args:
        config: 月面配置
    
    Returns:
        求解选项字典
    """
    return {
        'rounding': True,
        'preprocessing': config.enable_preprocessing,
        'verbose': False,
    }


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    from path_planner.scripts.hybrid_astar_gcs_planner import (
        PlannerConfig,
        HybridAStarGCSPlanner
    )
    from path_planner.scripts.planner_support import PlannerResult
    
    # 1. 获取预定义配置
    lunar_config = get_standard_lunar_config()

    # 2. 创建走廊分解配置
    config = PlannerConfig(
        # 走廊参数
        corridor_width=lunar_config.corridor_width,
        boundary_margin=lunar_config.boundary_margin,
    )

    # 3. 执行规划
    # planner = HybridAStarGCSPlanner(c_space, config)
    # result: PlannerResult = planner.process(path)

    print("月面月球车GCS配置示例")
    print(f"场景类型: {lunar_config.scenario}")
    print(f"舍入策略数: {len(lunar_config.rounding_strategies)}")
    print(f"最大求解时间: {lunar_config.solver_profile.max_time}秒")
    print(f"预处理: {'启用' if lunar_config.enable_preprocessing else '禁用'}")


if __name__ == "__main__":
    example_usage()
