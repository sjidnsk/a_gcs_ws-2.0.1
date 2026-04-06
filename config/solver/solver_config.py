"""
求解器配置优化模块

提供自适应求解器配置,根据问题规模自动调整求解器参数,
提高求解速度和稳定性。

主要功能:
- 自适应求解器配置
- 求解器预热
- 求解历史缓存
- 性能监控
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import hashlib
import pickle

import warnings
from enum import Enum

from pydrake.solvers import (
    SolverOptions,
    CommonSolverOption,
    MosekSolver,
    GurobiSolver,
    ClpSolver,
    ScsSolver,
)
from pydrake.geometry.optimization import GraphOfConvexSetsOptions


class ProblemSize(Enum):
    """问题规模枚举"""
    SMALL = "small"      # < 50条边
    MEDIUM = "medium"    # 50-200条边
    LARGE = "large"      # > 200条边


class SolverType(Enum):
    """求解器类型枚举"""
    MOSEK = "mosek"
    GUROBI = "gurobi"
    CLP = "clp"
    SCS = "scs"


@dataclass
class SolverPerformanceProfile:
    """求解器性能配置"""
    # 问题规模
    problem_size: ProblemSize = ProblemSize.MEDIUM
    
    # 求解器类型
    solver_type: SolverType = SolverType.MOSEK
    
    # 容差设置
    relaxation_tol: float = 1e-4      # 松弛问题容差
    mip_tol: float = 1e-3              # MIP问题容差
    constraint_tol: float = 1e-6       # 约束容差
    
    # 时间限制
    max_time: float = 60.0             # 最大求解时间(秒)
    mip_max_time: float = 30.0         # MIP最大求解时间
    
    # 预处理
    enable_presolve: bool = True       # 启用预处理
    presolve_level: int = 1            # 预处理级别(0-2)
    
    # 并行化
    num_threads: int = 4               # 线程数
    
    # 输出控制
    verbose: bool = False              # 详细输出
    print_to_console: bool = False     # 打印到控制台
    
    # 高级选项
    enable_warm_start: bool = True     # 启用预热
    cache_solutions: bool = True       # 缓存解
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'problem_size': self.problem_size.value,
            'solver_type': self.solver_type.value,
            'relaxation_tol': self.relaxation_tol,
            'mip_tol': self.mip_tol,
            'constraint_tol': self.constraint_tol,
            'max_time': self.max_time,
            'mip_max_time': self.mip_max_time,
            'enable_presolve': self.enable_presolve,
            'presolve_level': self.presolve_level,
            'num_threads': self.num_threads,
            'verbose': self.verbose,
            'print_to_console': self.print_to_console,
            'enable_warm_start': self.enable_warm_start,
            'cache_solutions': self.cache_solutions,
        }


class AdaptiveSolverConfig:
    """自适应求解器配置"""
    
    # 预定义配置模板
    PROFILES = {
        ProblemSize.SMALL: SolverPerformanceProfile(
            problem_size=ProblemSize.SMALL,
            relaxation_tol=1e-6,
            mip_tol=1e-4,
            max_time=120.0,
            mip_max_time=60.0,
            num_threads=2,
        ),
        ProblemSize.MEDIUM: SolverPerformanceProfile(
            problem_size=ProblemSize.MEDIUM,
            relaxation_tol=1e-4,
            mip_tol=1e-3,
            max_time=60.0,
            mip_max_time=30.0,
            num_threads=4,
        ),
        ProblemSize.LARGE: SolverPerformanceProfile(
            problem_size=ProblemSize.LARGE,
            relaxation_tol=1e-3,
            mip_tol=1e-2,
            max_time=30.0,
            mip_max_time=15.0,
            enable_presolve=True,
            presolve_level=2,
            num_threads=8,
        ),
    }
    
    def __init__(self, 
                 problem_size: str = 'auto',
                 solver_type: str = 'mosek',
                 custom_profile: Optional[SolverPerformanceProfile] = None):
        """
        初始化自适应求解器配置
        
        Args:
            problem_size: 问题规模 ('small', 'medium', 'large', 'auto')
            solver_type: 求解器类型 ('mosek', 'gurobi', 'clp', 'scs')
            custom_profile: 自定义性能配置
        """
        self.problem_size = ProblemSize(problem_size) if problem_size != 'auto' else None
        self.solver_type = SolverType(solver_type)
        self.custom_profile = custom_profile
        
        # 求解历史缓存
        self.solution_cache: Dict[str, Any] = {}
        self.cache_max_size = 50
        
        # 性能统计
        self.solve_stats: Dict[str, Dict] = {}
    
    def configure(self, 
                  num_vertices: int, 
                  num_edges: int,
                  dimension: int) -> Tuple[GraphOfConvexSetsOptions, SolverPerformanceProfile]:
        """
        根据问题规模配置求解器
        
        Args:
            num_vertices: 顶点数量
            num_edges: 边数量
            dimension: 空间维度
            
        Returns:
            (GCS选项, 性能配置)
        """
        # 确定问题规模
        if self.problem_size is None:
            problem_size = self._estimate_problem_size(num_vertices, num_edges, dimension)
        else:
            problem_size = self.problem_size
        
        # 获取性能配置
        if self.custom_profile is not None:
            profile = self.custom_profile
        else:
            profile = self.PROFILES[problem_size].__class__(**self.PROFILES[problem_size].to_dict())
            profile.solver_type = self.solver_type
        
        # 创建求解器选项
        options = self._create_solver_options(profile)
        
        # 创建GCS选项
        gcs_options = GraphOfConvexSetsOptions()
        gcs_options.solver = self._get_solver(profile.solver_type)
        gcs_options.solver_options = options
        gcs_options.preprocessing = profile.enable_presolve
        
        return gcs_options, profile
    
    def _estimate_problem_size(self, 
                               num_vertices: int, 
                               num_edges: int,
                               dimension: int) -> ProblemSize:
        """
        估算问题规模
        
        Args:
            num_vertices: 顶点数量
            num_edges: 边数量
            dimension: 空间维度
            
        Returns:
            问题规模枚举
        """
        # 综合考虑顶点数、边数和维度
        complexity_score = num_edges * dimension
        
        if complexity_score < 100:  # < 50条边 * 2维
            return ProblemSize.SMALL
        elif complexity_score < 600:  # < 200条边 * 3维
            return ProblemSize.MEDIUM
        else:
            return ProblemSize.LARGE
    
    def _create_solver_options(self, profile: SolverPerformanceProfile) -> SolverOptions:
        """
        创建求解器选项
        
        Args:
            profile: 性能配置
            
        Returns:
            求解器选项
        """
        options = SolverOptions()
        
        # 通用选项
        if profile.print_to_console:
            options.SetOption(CommonSolverOption.kPrintToConsole, 1)
        else:
            options.SetOption(CommonSolverOption.kPrintToConsole, 0)
        
        # 根据求解器类型设置特定选项
        if profile.solver_type == SolverType.MOSEK:
            self._configure_mosek(options, profile)
        elif profile.solver_type == SolverType.GUROBI:
            self._configure_gurobi(options, profile)
        elif profile.solver_type == SolverType.CLP:
            self._configure_clp(options, profile)
        elif profile.solver_type == SolverType.SCS:
            self._configure_scs(options, profile)
        
        return options
    
    def _configure_mosek(self, options: SolverOptions, profile: SolverPerformanceProfile):
        """配置MOSEK求解器"""
        solver_id = MosekSolver.id()
        
        # 松弛问题容差
        options.SetOption(solver_id, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 
                         profile.relaxation_tol)
        options.SetOption(solver_id, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 
                         profile.relaxation_tol * 10)
        options.SetOption(solver_id, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 
                         profile.relaxation_tol * 10)
        
        # MIP容差
        options.SetOption(solver_id, "MSK_DPAR_MIO_TOL_REL_GAP", 
                         profile.mip_tol)
        options.SetOption(solver_id, "MSK_DPAR_MIO_TOL_ABS_GAP", 
                         profile.mip_tol * 0.1)
        
        # 时间限制
        options.SetOption(solver_id, "MSK_DPAR_OPTIMIZER_MAX_TIME", 
                         profile.max_time)
        options.SetOption(solver_id, "MSK_DPAR_MIO_MAX_TIME", 
                         profile.mip_max_time)
        
        # 预处理
        if profile.enable_presolve:
            options.SetOption(solver_id, "MSK_IPAR_PRESOLVE_USE", 
                             profile.presolve_level)
        
        # 线程数
        options.SetOption(solver_id, "MSK_IPAR_NUM_THREADS", 
                         profile.num_threads)
        
        # 求解形式(对偶问题通常更快)
        options.SetOption(solver_id, "MSK_IPAR_INTPNT_SOLVE_FORM", 1)
        
        # 详细输出
        if profile.verbose:
            options.SetOption(solver_id, "MSK_IPAR_LOG", 1)
            options.SetOption(solver_id, "MSK_IPAR_LOG_INTPNT", 1)
    
    def _configure_gurobi(self, options: SolverOptions, profile: SolverPerformanceProfile):
        """配置Gurobi求解器"""
        solver_id = GurobiSolver.id()
        
        # 容差
        options.SetOption(solver_id, "OptimalityTol", profile.relaxation_tol)
        options.SetOption(solver_id, "MIPGap", profile.mip_tol)
        
        # 时间限制
        options.SetOption(solver_id, "TimeLimit", profile.max_time)
        
        # 线程数
        options.SetOption(solver_id, "Threads", profile.num_threads)
        
        # 预处理
        if profile.enable_presolve:
            options.SetOption(solver_id, "Presolve", profile.presolve_level)
        
        # 详细输出
        if profile.verbose:
            options.SetOption(solver_id, "OutputFlag", 1)
    
    def _configure_clp(self, options: SolverOptions, profile: SolverPerformanceProfile):
        """配置CLP求解器"""
        solver_id = ClpSolver.id()
        
        # 容差
        options.SetOption(solver_id, "primal_tolerance", profile.relaxation_tol)
        options.SetOption(solver_id, "dual_tolerance", profile.relaxation_tol)
        
        # 时间限制
        options.SetOption(solver_id, "max_iterations", 
                         int(profile.max_time * 1000))  # 近似迭代次数
    
    def _configure_scs(self, options: SolverOptions, profile: SolverPerformanceProfile):
        """配置SCS求解器"""
        solver_id = ScsSolver.id()
        
        # 容差
        options.SetOption(solver_id, "eps", profile.relaxation_tol)
        
        # 迭代限制
        options.SetOption(solver_id, "max_iters", 
                         int(profile.max_time * 1000))
        
        # 详细输出
        if profile.verbose:
            options.SetOption(solver_id, "verbose", 1)
    
    def _get_solver(self, solver_type: SolverType):
        """获取求解器实例"""
        if solver_type == SolverType.MOSEK:
            return MosekSolver()
        elif solver_type == SolverType.GUROBI:
            return GurobiSolver()
        elif solver_type == SolverType.CLP:
            return ClpSolver()
        elif solver_type == SolverType.SCS:
            return ScsSolver()
        else:
            warnings.warn(f"未知求解器类型: {solver_type}, 使用MOSEK")
            return MosekSolver()
    
    def compute_problem_hash(self, 
                            gcs,
                            source,
                            target) -> str:
        """
        计算问题的哈希值(用于缓存)
        
        Args:
            gcs: GCS对象
            source: 源点
            target: 目标点
            
        Returns:
            哈希字符串
        """
        # 收集问题特征
        features = {
            'num_vertices': len(gcs.Vertices()),
            'num_edges': len(gcs.Edges()),
            'source': tuple(source) if hasattr(source, '__iter__') else source,
            'target': tuple(target) if hasattr(target, '__iter__') else target,
        }
        
        # 计算哈希
        hash_str = hashlib.md5(pickle.dumps(features)).hexdigest()
        return hash_str
    
    def get_cached_solution(self, problem_hash: str) -> Optional[Any]:
        """
        获取缓存的解
        
        Args:
            problem_hash: 问题哈希
            
        Returns:
            缓存的解,如果不存在则返回None
        """
        return self.solution_cache.get(problem_hash)
    
    def cache_solution(self, problem_hash: str, solution: Any):
        """
        缓存解
        
        Args:
            problem_hash: 问题哈希
            solution: 解对象
        """
        # LRU缓存淘汰
        if len(self.solution_cache) >= self.cache_max_size:
            # 移除最旧的缓存
            oldest_key = next(iter(self.solution_cache))
            del self.solution_cache[oldest_key]
        
        self.solution_cache[problem_hash] = solution
    
    def record_solve_stats(self, 
                          problem_hash: str,
                          solve_time: float,
                          cost: float,
                          success: bool):
        """
        记录求解统计信息
        
        Args:
            problem_hash: 问题哈希
            solve_time: 求解时间
            cost: 最优成本
            success: 是否成功
        """
        if problem_hash not in self.solve_stats:
            self.solve_stats[problem_hash] = {
                'solve_times': [],
                'costs': [],
                'successes': [],
            }
        
        stats = self.solve_stats[problem_hash]
        stats['solve_times'].append(solve_time)
        stats['costs'].append(cost)
        stats['successes'].append(success)
    
    def get_performance_summary(self) -> Dict:
        """
        获取性能摘要
        
        Returns:
            性能统计摘要
        """
        summary = {
            'total_problems': len(self.solve_stats),
            'cache_size': len(self.solution_cache),
            'problems': {},
        }
        
        for problem_hash, stats in self.solve_stats.items():
            times = stats['solve_times']
            successes = stats['successes']
            
            summary['problems'][problem_hash] = {
                'num_solves': len(times),
                'avg_time': np.mean(times) if times else 0,
                'min_time': np.min(times) if times else 0,
                'max_time': np.max(times) if times else 0,
                'success_rate': np.mean(successes) if successes else 0,
            }
        
        return summary
    
    def clear_cache(self):
        """清空缓存"""
        self.solution_cache.clear()
        self.solve_stats.clear()


# ==================== 便捷函数 ====================

def create_optimized_gcs_options(num_vertices: int,
                                 num_edges: int,
                                 dimension: int,
                                 solver_type: str = 'mosek',
                                 problem_size: str = 'auto') -> GraphOfConvexSetsOptions:
    """
    创建优化的GCS选项(便捷函数)
    
    Args:
        num_vertices: 顶点数量
        num_edges: 边数量
        dimension: 空间维度
        solver_type: 求解器类型
        problem_size: 问题规模
        
    Returns:
        GCS选项
    """
    config = AdaptiveSolverConfig(
        problem_size=problem_size,
        solver_type=solver_type
    )
    options, _ = config.configure(num_vertices, num_edges, dimension)
    return options


def get_fast_solver_config() -> SolverPerformanceProfile:
    """
    获取快速求解配置(牺牲精度)
    
    Returns:
        性能配置
    """
    return SolverPerformanceProfile(
        problem_size=ProblemSize.LARGE,
        relaxation_tol=1e-2,
        mip_tol=1e-1,
        max_time=10.0,
        mip_max_time=5.0,
        enable_presolve=True,
        presolve_level=2,
        num_threads=8,
    )


def get_accurate_solver_config() -> SolverPerformanceProfile:
    """
    获取高精度求解配置(较慢)
    
    Returns:
        性能配置
    """
    return SolverPerformanceProfile(
        problem_size=ProblemSize.SMALL,
        relaxation_tol=1e-8,
        mip_tol=1e-6,
        max_time=300.0,
        mip_max_time=120.0,
        enable_presolve=True,
        presolve_level=1,
        num_threads=4,
    )


def get_balanced_solver_config() -> SolverPerformanceProfile:
    """
    获取平衡配置(推荐)
    
    Returns:
        性能配置
    """
    return SolverPerformanceProfile(
        problem_size=ProblemSize.MEDIUM,
        relaxation_tol=1e-4,
        mip_tol=1e-3,
        max_time=60.0,
        mip_max_time=30.0,
        enable_presolve=True,
        presolve_level=1,
        num_threads=4,
    )
