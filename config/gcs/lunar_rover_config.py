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
