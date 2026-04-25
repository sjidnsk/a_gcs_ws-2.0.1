"""
MOSEK求解器优化统一配置模块

提供MOSEK求解器优化参数的统一配置入口，包括：
- Rounding路径搜索数量优化
- MOSEK线程数配置
- MIO时间限制优化
- 增量Phi约束更新开关
- 各优化方案的独立启停控制
"""

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class MosekOptimizationConfig:
    """MOSEK求解器优化统一配置

    集中管理所有MOSEK求解器优化相关参数，支持：
    - 优化参数值的配置化（替代硬编码）
    - 各优化方案的独立启停（便于A/B对比测试和问题定位）
    - 参数范围校验（防止无效配置）
    - 配置摘要输出（便于日志记录和问题排查）

    Attributes:
        max_paths: Rounding阶段每个策略搜索的最大路径数
        num_threads: MOSEK求解器使用的线程数
        mio_max_time: MIO最大求解时间(秒)
        max_rounding_attempts: Rounding阶段最大重试次数
        max_rounded_paths: 每次Rounding的路径数上限
        enable_reduced_paths: 启用减少Rounding路径数优化
        enable_thread_limit: 启用MOSEK线程数限制优化
        enable_mio_time_limit: 启用MIO时间限制优化
        enable_incremental_phi: 启用增量Phi约束更新优化
    """

    # --- 优化参数值 ---
    max_paths: int = 5                    # Rounding最大路径数 (原值10)
    num_threads: int = 8                  # MOSEK线程数 (保持8，单次大规模SOCP/SDP需要并行度)
    mio_max_time: float = 30.0            # MIO最大时间(秒) (原值3600.0，与SolverPerformanceProfile.mip_max_time对齐)
    max_rounding_attempts: int = 3        # 最大Rounding重试次数 (不变)
    max_rounded_paths: int = 5            # 每次Rounding的路径数 (原值10)

    # --- 优化方案独立开关 ---
    enable_reduced_paths: bool = True     # 启用减少Rounding路径数
    enable_thread_limit: bool = True      # 启用线程数限制
    enable_mio_time_limit: bool = True    # 启用MIO时间限制
    enable_incremental_phi: bool = True   # 启用增量Phi约束更新

    def __post_init__(self):
        """参数范围校验"""
        # max_paths范围校验
        if not (1 <= self.max_paths <= 20):
            raise ValueError(
                f"max_paths={self.max_paths} 超出有效范围[1, 20]"
            )

        # num_threads范围校验
        cpu_count = os.cpu_count() or 1
        if not (1 <= self.num_threads <= cpu_count):
            raise ValueError(
                f"num_threads={self.num_threads} 超出有效范围[1, {cpu_count}]"
            )

        # mio_max_time范围校验
        if not (1.0 <= self.mio_max_time <= 3600.0):
            raise ValueError(
                f"mio_max_time={self.mio_max_time} 超出有效范围[1.0, 3600.0]"
            )

        # max_rounding_attempts范围校验
        if not (1 <= self.max_rounding_attempts <= 10):
            raise ValueError(
                f"max_rounding_attempts={self.max_rounding_attempts} 超出有效范围[1, 10]"
            )

        # max_rounded_paths范围校验
        if not (1 <= self.max_rounded_paths <= 20):
            raise ValueError(
                f"max_rounded_paths={self.max_rounded_paths} 超出有效范围[1, 20]"
            )

    def effective_max_paths(self) -> int:
        """返回实际生效的max_paths（考虑开关）

        Returns:
            enable_reduced_paths=True时返回max_paths，否则返回原始值10
        """
        if self.enable_reduced_paths:
            return self.max_paths
        return 10

    def effective_num_threads(self) -> int:
        """返回实际生效的num_threads（考虑开关）

        Returns:
            enable_thread_limit=True时返回num_threads，否则返回原始值8
        """
        if self.enable_thread_limit:
            return self.num_threads
        return 8

    def effective_mio_max_time(self) -> float:
        """返回实际生效的mio_max_time（考虑开关）

        Returns:
            enable_mio_time_limit=True时返回mio_max_time，否则返回原始值3600.0
        """
        if self.enable_mio_time_limit:
            return self.mio_max_time
        return 3600.0

    def to_dict(self) -> Dict:
        """导出配置为字典

        Returns:
            包含所有配置参数和开关状态的字典
        """
        return {
            'max_paths': self.max_paths,
            'num_threads': self.num_threads,
            'mio_max_time': self.mio_max_time,
            'max_rounding_attempts': self.max_rounding_attempts,
            'max_rounded_paths': self.max_rounded_paths,
            'enable_reduced_paths': self.enable_reduced_paths,
            'enable_thread_limit': self.enable_thread_limit,
            'enable_mio_time_limit': self.enable_mio_time_limit,
            'enable_incremental_phi': self.enable_incremental_phi,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MosekOptimizationConfig':
        """从字典创建配置实例

        Args:
            data: 包含配置参数的字典

        Returns:
            MosekOptimizationConfig实例
        """
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})

    def summary(self) -> str:
        """返回配置摘要字符串，用于日志输出

        Returns:
            格式化的多行配置摘要
        """
        lines = [
            "MOSEK优化配置:",
            f"  max_paths: {self.effective_max_paths()} (配置值={self.max_paths}, 开关={self.enable_reduced_paths})",
            f"  num_threads: {self.effective_num_threads()} (配置值={self.num_threads}, 开关={self.enable_thread_limit})",
            f"  mio_max_time: {self.effective_mio_max_time()}s (配置值={self.mio_max_time}s, 开关={self.enable_mio_time_limit})",
            f"  max_rounding_attempts: {self.max_rounding_attempts}",
            f"  max_rounded_paths: {self.max_rounded_paths}",
            f"  enable_incremental_phi: {self.enable_incremental_phi}",
        ]
        return "\n".join(lines)
