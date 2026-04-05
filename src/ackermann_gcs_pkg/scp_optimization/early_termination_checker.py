"""
提前终止检查器

本模块实现多层次提前终止机制，避免不必要的迭代。
"""

from typing import Tuple, List, Optional
from ..ackermann_data_structures import (
    TerminationConfig,
    TerminationReason,
    ViolationReport
)


class EarlyTerminationChecker:
    """
    提前终止检查器
    
    提供多层次提前终止判断，包括：
    1. 收敛性检查：曲率违反量小于阈值
    2. 改进停滞检查：连续多次低改进率
    3. 约束满足度检查：达到工程可行解
    4. 信任区域耗尽检查：信任区域过小且无改进
    
    Attributes:
        config: 终止条件配置
        improvement_history: 改进率历史列表
        shrink_count: 连续缩小计数
        delta_history: 信任区域半径历史
    """
    
    def __init__(self, config: TerminationConfig):
        """
        初始化提前终止检查器
        
        Args:
            config: 终止条件配置参数
        """
        self.config = config
        self.improvement_history: List[float] = []
        self.shrink_count: int = 0
        self.delta_history: List[float] = []
    
    def check_convergence(self, violation: float) -> bool:
        """
        检查收敛性（向后兼容版本）

        当曲率违反量小于收敛阈值时，认为已收敛

        Args:
            violation: 当前曲率违反量

        Returns:
            如果收敛则返回True，否则返回False
        """
        return violation < self.config.convergence_tolerance

    def check_convergence_with_report(self, violation_report: ViolationReport) -> bool:
        """
        检查收敛性（改进版）

        当所有约束违反量都小于收敛阈值时，认为已收敛

        Args:
            violation_report: 约束违反量报告

        Returns:
            如果收敛则返回True，否则返回False
        """
        return (
            violation_report.velocity_violation < self.config.velocity_threshold and
            violation_report.acceleration_violation < self.config.acceleration_threshold and
            violation_report.curvature_violation < self.config.convergence_tolerance
        )

    def check_severe_violation(
        self,
        violation_report: ViolationReport
    ) -> Tuple[bool, Optional[str]]:
        """
        检查严重违反（新增）

        检测是否有约束严重违反，阻止过早收敛

        Args:
            violation_report: 约束违反量报告

        Returns:
            (is_severe, violation_type): 是否严重违反及违反类型
        """
        # 检查加速度严重违反（最高优先级）
        if violation_report.acceleration_violation > self.config.acceleration_severe_threshold:
            return True, "acceleration"

        # 检查速度严重违反
        if violation_report.velocity_violation > self.config.velocity_severe_threshold:
            return True, "velocity"

        # 检查曲率严重违反
        if violation_report.curvature_violation > self.config.curvature_severe_threshold:
            return True, "curvature"

        return False, None
    
    def check_improvement_stagnation(self, improvement_ratio: float) -> bool:
        """
        检查改进停滞
        
        当连续N次迭代改进率低于停滞阈值时，认为改进停滞。
        同时检测震荡模式（改进率正负交替）。
        
        Args:
            improvement_ratio: 当前改进率
        
        Returns:
            如果改进停滞则返回True，否则返回False
        """
        # 添加到历史记录
        self.improvement_history.append(improvement_ratio)
        
        # 保持固定窗口大小
        if len(self.improvement_history) > self.config.stagnation_window:
            self.improvement_history.pop(0)
        
        # 检查是否所有改进率都低于阈值
        if len(self.improvement_history) == self.config.stagnation_window:
            if all(r < self.config.stagnation_threshold for r in self.improvement_history):
                return True
        
        # 检查震荡模式（改进率正负交替）
        if len(self.improvement_history) >= 4:
            recent = self.improvement_history[-4:]
            # 检查是否正负交替
            oscillating = (
                (recent[0] * recent[1] < 0) and
                (recent[1] * recent[2] < 0) and
                (recent[2] * recent[3] < 0)
            )
            if oscillating:
                # 检查震荡幅度是否较大
                if any(abs(r) > self.config.oscillation_threshold for r in recent):
                    return True
        
        return False
    
    def check_constraint_satisfaction(
        self,
        violation: float,
        iteration: int,
        max_iterations: int
    ) -> bool:
        """
        检查约束满足度（向后兼容版本）

        当曲率违反量已达到工程可接受范围，且已迭代足够次数时，
        认为已找到工程可行解。

        Args:
            violation: 当前曲率违反量
            iteration: 当前迭代次数
            max_iterations: 最大迭代次数

        Returns:
            如果约束满足度足够则返回True，否则返回False
        """
        # 计算工程阈值
        engineering_tolerance = (
            self.config.convergence_tolerance * self.config.engineering_tolerance_factor
        )

        # 检查是否达到工程可行解
        if violation < engineering_tolerance:
            # 检查是否已达到最小迭代比例
            min_iterations = int(max_iterations * self.config.min_iteration_ratio)
            if iteration >= min_iterations:
                return True

        return False

    def check_constraint_satisfaction_with_report(
        self,
        violation_report: ViolationReport,
        iteration: int,
        max_iterations: int
    ) -> bool:
        """
        检查约束满足度（改进版）

        当所有约束违反量都达到工程可接受范围时，认为已找到工程可行解

        Args:
            violation_report: 约束违反量报告
            iteration: 当前迭代次数
            max_iterations: 最大迭代次数

        Returns:
            如果约束满足度足够则返回True，否则返回False
        """
        # 计算工程阈值
        engineering_tolerance = (
            self.config.convergence_tolerance * self.config.engineering_tolerance_factor
        )

        # 检查是否达到工程可行解（所有约束）
        all_constraints_satisfied = (
            violation_report.velocity_violation < engineering_tolerance and
            violation_report.acceleration_violation < engineering_tolerance and
            violation_report.curvature_violation < engineering_tolerance
        )

        if all_constraints_satisfied:
            # 检查是否已达到最小迭代比例
            min_iterations = int(max_iterations * self.config.min_iteration_ratio)
            if iteration >= min_iterations:
                return True

        return False
    
    def check_trust_region_exhaustion(
        self,
        delta: float,
        improved: bool
    ) -> Tuple[bool, TerminationReason]:
        """
        检查信任区域耗尽
        
        当信任区域半径过小，或连续缩小多次且无改进时，
        认为信任区域已耗尽或陷入局部最优。
        
        Args:
            delta: 当前信任区域半径
            improved: 当前迭代是否有改进（improvement_ratio > 0）
        
        Returns:
            (should_terminate, reason) 元组
        """
        # 记录delta历史
        self.delta_history.append(delta)
        if len(self.delta_history) > 5:
            self.delta_history.pop(0)
        
        # 检查信任区域是否过小
        if delta < self.config.min_delta:
            return True, TerminationReason.TRUST_REGION_EXHAUSTED
        
        # 检查是否在缩小
        if len(self.delta_history) >= 2:
            if delta < self.delta_history[-2]:
                # 信任区域在缩小
                if not improved:
                    # 且无改进
                    self.shrink_count += 1
                else:
                    # 有改进，重置计数
                    self.shrink_count = 0
            else:
                # 信任区域在扩大或保持，重置计数
                self.shrink_count = 0
        
        # 检查连续缩小次数
        if self.shrink_count >= self.config.max_shrink_count:
            return True, TerminationReason.LOCAL_OPTIMUM
        
        return False, TerminationReason.CONTINUE
    
    def check_all(
        self,
        violation: float,
        improvement_ratio: float,
        delta: float,
        iteration: int,
        max_iterations: int
    ) -> Tuple[bool, TerminationReason]:
        """
        检查所有终止条件（向后兼容版本）

        按优先级依次检查各个终止条件：
        1. 收敛性检查（最高优先级）
        2. 改进停滞检查
        3. 约束满足度检查
        4. 信任区域耗尽检查

        Args:
            violation: 当前曲率违反量
            improvement_ratio: 当前改进率
            delta: 当前信任区域半径
            iteration: 当前迭代次数
            max_iterations: 最大迭代次数

        Returns:
            (should_terminate, reason) 元组

        Examples:
            >>> checker = EarlyTerminationChecker(TerminationConfig())
            >>> should_stop, reason = checker.check_all(
            ...     violation=1e-4,  # 小于收敛阈值
            ...     improvement_ratio=0.1,
            ...     delta=0.1,
            ...     iteration=5,
            ...     max_iterations=10
            ... )
            >>> print(f"Should stop: {should_stop}, Reason: {reason.value}")
        """
        # 如果未启用提前终止，继续迭代
        if not self.config.enable_early_termination:
            return False, TerminationReason.CONTINUE

        # 1. 收敛性检查（最高优先级）
        if self.check_convergence(violation):
            return True, TerminationReason.CONVERGED

        # 2. 改进停滞检查
        if self.check_improvement_stagnation(improvement_ratio):
            return True, TerminationReason.IMPROVEMENT_STAGNATION

        # 3. 约束满足度检查
        if self.check_constraint_satisfaction(violation, iteration, max_iterations):
            return True, TerminationReason.CONSTRAINT_SATISFIED

        # 4. 信任区域耗尽检查
        improved = improvement_ratio > 0
        should_stop, reason = self.check_trust_region_exhaustion(delta, improved)
        if should_stop:
            return True, reason

        # 5. 继续迭代
        return False, TerminationReason.CONTINUE

    def check_all_with_report(
        self,
        violation_report: ViolationReport,
        improvement_ratio: float,
        delta: float,
        iteration: int,
        max_iterations: int
    ) -> Tuple[bool, TerminationReason]:
        """
        检查所有终止条件（改进版）

        新增优先级0：严重违反检查（最高优先级）
        调整优先级：收敛检查使用violation_report

        按优先级依次检查各个终止条件：
        0. 严重违反检查（新增，最高优先级）
        1. 收敛性检查
        2. 改进停滞检查
        3. 约束满足度检查
        4. 信任区域耗尽检查

        Args:
            violation_report: 约束违反量报告
            improvement_ratio: 当前改进率
            delta: 当前信任区域半径
            iteration: 当前迭代次数
            max_iterations: 最大迭代次数

        Returns:
            (should_terminate, reason) 元组
        """
        # 如果未启用提前终止，继续迭代
        if not self.config.enable_early_termination:
            return False, TerminationReason.CONTINUE

        # 0. 严重违反检查（新增，最高优先级）
        is_severe, violation_type = self.check_severe_violation(violation_report)
        if is_severe:
            # 严重违反，强制继续迭代，不允许收敛
            return False, TerminationReason.CONTINUE

        # 1. 收敛性检查（改进）
        if self.check_convergence_with_report(violation_report):
            return True, TerminationReason.CONVERGED

        # 2. 改进停滞检查
        if self.check_improvement_stagnation(improvement_ratio):
            return True, TerminationReason.IMPROVEMENT_STAGNATION

        # 3. 约束满足度检查（改进）
        if self.check_constraint_satisfaction_with_report(violation_report, iteration, max_iterations):
            return True, TerminationReason.CONSTRAINT_SATISFIED

        # 4. 信任区域耗尽检查
        improved = improvement_ratio > 0
        should_stop, reason = self.check_trust_region_exhaustion(delta, improved)
        if should_stop:
            return True, reason

        # 5. 继续迭代
        return False, TerminationReason.CONTINUE
    
    def reset(self) -> None:
        """重置检查器状态"""
        self.improvement_history.clear()
        self.shrink_count = 0
        self.delta_history.clear()
    
    def get_stagnation_info(self) -> dict:
        """
        获取停滞信息
        
        Returns:
            包含停滞相关信息的字典
        """
        return {
            'improvement_history': self.improvement_history.copy(),
            'shrink_count': self.shrink_count,
            'delta_history': self.delta_history.copy(),
            'avg_improvement_ratio': (
                sum(self.improvement_history) / len(self.improvement_history)
                if self.improvement_history else 0.0
            )
        }
