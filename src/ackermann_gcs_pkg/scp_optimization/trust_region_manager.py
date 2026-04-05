"""
信任区域管理器

本模块实现基于改进率的动态信任区域调整策略。
"""

from typing import List
from ackermann_data_structures import (
    TrustRegionConfig,
    ImprovementRecord
)


class TrustRegionManager:
    """
    信任区域管理器
    
    负责基于改进率动态调整信任区域半径，加速SCP收敛
    
    核心功能：
    1. 计算改进率（improvement ratio）
    2. 根据改进率自适应调整信任区域
    3. 记录改进历史用于趋势分析
    
    Attributes:
        config: 信任区域配置
        history: 改进历史记录列表
        current_delta: 当前信任区域半径
        iteration_count: 当前迭代次数
    """
    
    def __init__(self, config: TrustRegionConfig):
        """
        初始化信任区域管理器
        
        Args:
            config: 信任区域配置参数
        """
        self.config = config
        self.history: List[ImprovementRecord] = []
        self.current_delta: float = 0.0
        self.iteration_count: int = 0
    
    def compute_improvement_ratio(
        self,
        improvement: float,
        previous_violation: float
    ) -> float:
        """
        计算改进率
        
        改进率 = improvement / previous_violation
        其中 improvement = previous_violation - current_violation
        
        Args:
            improvement: 改进量（previous_violation - current_violation）
            previous_violation: 上一迭代的曲率违反量
        
        Returns:
            改进率，范围可以是负数（解变差）到正数（解改进）
        
        Examples:
            >>> manager = TrustRegionManager(TrustRegionConfig())
            >>> # 改进了20%
            >>> ratio = manager.compute_improvement_ratio(0.01, 0.05)
            >>> print(f"改进率: {ratio:.2f}")  # 输出: 改进率: 0.20
        """
        # 处理除零情况
        if previous_violation == 0.0:
            return 0.0
        
        return improvement / previous_violation
    
    def adjust_trust_region(self, improvement_ratio: float) -> float:
        """
        根据改进率调整信任区域半径
        
        调整策略：
        - 高改进（> high_threshold）：扩大信任区域，允许更大步长
        - 中等改进（medium_threshold ~ high_threshold）：保持不变
        - 低改进（low_threshold ~ medium_threshold）：缩小信任区域
        - 负改进或微小改进（< low_threshold）：激进缩小
        
        Args:
            improvement_ratio: 改进率
        
        Returns:
            调整因子，新delta = 当前delta * 调整因子
        
        Examples:
            >>> manager = TrustRegionManager(TrustRegionConfig())
            >>> # 高改进，扩大信任区域
            >>> factor = manager.adjust_trust_region(0.5)
            >>> print(f"调整因子: {factor}")  # 输出: 1.5
            >>> 
            >>> # 负改进，激进缩小
            >>> factor = manager.adjust_trust_region(-0.1)
            >>> print(f"调整因子: {factor}")  # 输出: 0.25
        """
        # 如果未启用动态调整，返回1.0（不调整）
        if not self.config.enable_dynamic_adjustment:
            return 1.0
        
        # 根据改进率选择调整策略
        if improvement_ratio > self.config.high_improvement_threshold:
            # 高改进：扩大信任区域
            return self.config.expand_factor
        elif improvement_ratio >= self.config.medium_improvement_threshold:
            # 中等改进：保持不变
            return self.config.maintain_factor
        elif improvement_ratio >= self.config.low_improvement_threshold:
            # 低改进：缩小信任区域
            return self.config.shrink_factor
        else:
            # 负改进或微小改进：激进缩小
            return self.config.aggressive_shrink_factor
    
    def record_improvement(
        self,
        improvement: float,
        violation: float,
        delta: float
    ) -> None:
        """
        记录改进信息
        
        将当前迭代的改进信息添加到历史记录，用于趋势分析。
        如果历史记录超过最大长度，移除最旧的记录。
        
        Args:
            improvement: 改进量
            violation: 当前曲率违反量
            delta: 当前信任区域半径
        """
        self.iteration_count += 1
        
        # 计算改进率
        if len(self.history) > 0:
            previous_violation = self.history[-1].violation
            improvement_ratio = self.compute_improvement_ratio(
                improvement, previous_violation
            )
        else:
            # 第一次迭代，无法计算改进率
            improvement_ratio = 0.0
        
        # 创建改进记录
        record = ImprovementRecord(
            iteration=self.iteration_count,
            improvement=improvement,
            improvement_ratio=improvement_ratio,
            violation=violation,
            delta=delta
        )
        
        # 添加到历史记录
        self.history.append(record)
        
        # 如果超过最大长度，移除最旧的记录
        if len(self.history) > self.config.max_history_length:
            self.history.pop(0)
        
        # 更新当前delta
        self.current_delta = delta
    
    def get_history(self) -> List[ImprovementRecord]:
        """
        获取改进历史记录
        
        Returns:
            改进历史记录列表的副本
        """
        return self.history.copy()
    
    def get_latest_record(self) -> ImprovementRecord:
        """
        获取最新的改进记录
        
        Returns:
            最新的改进记录，如果历史为空则返回None
        """
        if len(self.history) == 0:
            return None
        return self.history[-1]
    
    def clear_history(self) -> None:
        """清空历史记录"""
        self.history.clear()
        self.iteration_count = 0
    
    def get_average_improvement_ratio(self, window: int = 5) -> float:
        """
        计算最近N次迭代的平均改进率
        
        Args:
            window: 窗口大小，默认为5
        
        Returns:
            平均改进率，如果历史记录不足则返回0.0
        """
        if len(self.history) == 0:
            return 0.0
        
        # 取最近window次记录
        recent_records = self.history[-window:]
        
        # 计算平均改进率
        avg_ratio = sum(r.improvement_ratio for r in recent_records) / len(recent_records)
        
        return avg_ratio
    
    def is_improving(self) -> bool:
        """
        判断是否在持续改进
        
        检查最近3次迭代的改进率是否都为正
        
        Returns:
            如果最近3次迭代都在改进则返回True，否则返回False
        """
        if len(self.history) < 3:
            return True  # 历史不足，假设在改进
        
        recent_ratios = [r.improvement_ratio for r in self.history[-3:]]
        return all(ratio > 0 for ratio in recent_ratios)
