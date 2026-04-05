"""
性能统计收集器

本模块实现SCP求解过程的性能统计和报告生成。
"""

import time
from typing import List, Optional

from ..ackermann_data_structures import (
    IterationStats,
    PerformanceMetrics
)


class PerformanceStats:
    """
    性能统计收集器
    
    收集和报告SCP求解过程的性能指标，包括：
    - 每次迭代的详细统计
    - 总体性能指标
    - 并行性能指标（如果启用）
    
    Attributes:
        iteration_stats: 迭代统计列表
        start_time: 开始时间
        parallel_time: 并行计算累计时间
        sequential_time: 串行计算累计时间
        num_processes: 并行进程数
    """
    
    def __init__(self, num_processes: Optional[int] = None):
        """
        初始化性能统计器
        
        Args:
            num_processes: 并行进程数（可选）
        """
        self.iteration_stats: List[IterationStats] = []
        self.start_time: float = 0.0
        self.parallel_time: float = 0.0
        self.sequential_time: float = 0.0
        self.num_processes = num_processes
    
    def start(self) -> None:
        """开始计时"""
        self.start_time = time.time()
    
    def record_iteration(
        self,
        iteration: int,
        violation: float,
        improvement: float,
        improvement_ratio: float,
        delta: float,
        solve_time: float,
        # 新增参数（带默认值，向后兼容）
        velocity_violation: float = 0.0,
        acceleration_violation: float = 0.0,
        curvature_violation: float = 0.0,
        workspace_violation: float = 0.0
    ) -> None:
        """
        记录迭代统计（改进版）

        Args:
            iteration: 迭代次数
            violation: 最大约束违反量
            improvement: 改进量
            improvement_ratio: 改进率
            delta: 信任区域半径
            solve_time: 求解时间（秒）
            velocity_violation: 速度违反量（新增）
            acceleration_violation: 加速度违反量（新增）
            curvature_violation: 曲率违反量（新增）
            workspace_violation: 工作空间违反量（新增）
        """
        stats = IterationStats(
            iteration=iteration,
            violation=violation,
            improvement=improvement,
            improvement_ratio=improvement_ratio,
            delta=delta,
            solve_time=solve_time,
            velocity_violation=velocity_violation,
            acceleration_violation=acceleration_violation,
            curvature_violation=curvature_violation,
            workspace_violation=workspace_violation
        )
        self.iteration_stats.append(stats)
    
    def record_parallel_time(self, parallel_time: float) -> None:
        """
        记录并行计算时间
        
        Args:
            parallel_time: 并行计算时间（秒）
        """
        self.parallel_time += parallel_time
    
    def record_sequential_time(self, sequential_time: float) -> None:
        """
        记录串行计算时间
        
        Args:
            sequential_time: 串行计算时间（秒）
        """
        self.sequential_time += sequential_time
    
    def compute_performance_metrics(self) -> PerformanceMetrics:
        """
        计算性能指标
        
        Returns:
            PerformanceMetrics对象
        
        Examples:
            >>> stats = PerformanceStats()
            >>> stats.start()
            >>> # ... 记录迭代 ...
            >>> metrics = stats.compute_performance_metrics()
            >>> print(f"Total time: {metrics.total_solve_time:.2f}s")
        """
        if not self.iteration_stats:
            return PerformanceMetrics(
                total_solve_time=0.0,
                average_iteration_time=0.0,
                num_iterations=0,
                final_violation=0.0,
                convergence_rate=0.0
            )
        
        # 计算总时间
        total_time = time.time() - self.start_time
        
        # 计算平均迭代时间
        avg_iteration_time = (
            sum(s.solve_time for s in self.iteration_stats) / len(self.iteration_stats)
        )
        
        # 计算收敛率
        initial_violation = self.iteration_stats[0].violation
        final_violation = self.iteration_stats[-1].violation
        convergence_rate = (
            final_violation / initial_violation if initial_violation > 0 else 0.0
        )
        
        # 创建性能指标
        metrics = PerformanceMetrics(
            total_solve_time=total_time,
            average_iteration_time=avg_iteration_time,
            num_iterations=len(self.iteration_stats),
            final_violation=final_violation,
            convergence_rate=convergence_rate
        )
        
        # 计算并行性能指标
        if self.parallel_time > 0 and self.sequential_time > 0 and self.num_processes:
            metrics.parallel_speedup = self.sequential_time / self.parallel_time
            metrics.parallel_efficiency = metrics.parallel_speedup / self.num_processes
        
        return metrics
    
    def generate_report(self) -> str:
        """
        生成性能报告（改进版）

        Returns:
            格式化的性能报告字符串
        """
        metrics = self.compute_performance_metrics()

        report_lines = [
            "=" * 70,
            "SCP求解器性能报告",
            "=" * 70,
            "",
            "总体性能:",
            f"  总求解时间: {metrics.total_solve_time:.3f} 秒",
            f"  迭代次数: {metrics.num_iterations}",
            f"  平均迭代时间: {metrics.average_iteration_time:.3f} 秒",
            f"  最终曲率违反量: {metrics.final_violation:.6f}",
            f"  收敛率: {metrics.convergence_rate:.6f}",
            "",
        ]

        # 并行性能
        if metrics.parallel_speedup is not None:
            report_lines.extend([
                "并行性能:",
                f"  并行进程数: {self.num_processes}",
                f"  并行加速比: {metrics.parallel_speedup:.2f}x",
                f"  并行效率: {metrics.parallel_efficiency:.2%}",
                "",
            ])

        # 迭代历史（改进：显示所有约束违反量）
        if self.iteration_stats:
            report_lines.extend([
                "约束违反量历史:",
                "-" * 70,
            ])

            # 检查是否有约束违反量数据
            has_violation_data = any(
                s.velocity_violation > 0 or s.acceleration_violation > 0
                for s in self.iteration_stats
            )

            if has_violation_data:
                # 新格式：显示所有约束违反量
                report_lines.append(
                    f"{'Iter':>4} {'Vel':>10} {'Accel':>10} {'Curv':>10} {'Max':>10} {'Time':>8}"
                )
                report_lines.append("-" * 70)

                for stats in self.iteration_stats:
                    report_lines.append(
                        f"{stats.iteration:>4} "
                        f"{stats.velocity_violation:>10.6f} "
                        f"{stats.acceleration_violation:>10.6f} "
                        f"{stats.curvature_violation:>10.6f} "
                        f"{stats.violation:>10.6f} "
                        f"{stats.solve_time:>8.3f}s"
                    )
            else:
                # 旧格式：仅显示曲率违反量（向后兼容）
                report_lines.extend([
                    "  Iter | Violation | Improvement | Ratio  | Delta    | Time(s)",
                    "  " + "-" * 56,
                ])

                for stats in self.iteration_stats:
                    report_lines.append(
                        f"  {stats.iteration:4d} | {stats.violation:9.6f} | "
                        f"{stats.improvement:11.6f} | {stats.improvement_ratio:6.3f} | "
                        f"{stats.delta:8.6f} | {stats.solve_time:7.4f}"
                    )

        report_lines.extend(["", "=" * 70])

        return "\n".join(report_lines)
    
    def get_iteration_stats(self) -> List[IterationStats]:
        """
        获取迭代统计列表
        
        Returns:
            迭代统计列表的副本
        """
        return self.iteration_stats.copy()
    
    def get_summary(self) -> dict:
        """
        获取性能摘要
        
        Returns:
            包含关键性能指标的字典
        """
        metrics = self.compute_performance_metrics()
        
        summary = {
            'total_time': metrics.total_solve_time,
            'num_iterations': metrics.num_iterations,
            'avg_iteration_time': metrics.average_iteration_time,
            'final_violation': metrics.final_violation,
            'convergence_rate': metrics.convergence_rate,
        }
        
        if metrics.parallel_speedup is not None:
            summary['parallel_speedup'] = metrics.parallel_speedup
            summary['parallel_efficiency'] = metrics.parallel_efficiency
        
        return summary
    
    def clear(self) -> None:
        """清空统计数据"""
        self.iteration_stats.clear()
        self.start_time = 0.0
        self.parallel_time = 0.0
        self.sequential_time = 0.0
    
    def export_to_csv(self, filepath: str) -> None:
        """
        导出迭代统计到CSV文件
        
        Args:
            filepath: CSV文件路径
        """
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow([
                'iteration', 'violation', 'improvement',
                'improvement_ratio', 'delta', 'solve_time', 'timestamp'
            ])
            
            # 写入数据
            for stats in self.iteration_stats:
                writer.writerow([
                    stats.iteration,
                    stats.violation,
                    stats.improvement,
                    stats.improvement_ratio,
                    stats.delta,
                    stats.solve_time,
                    stats.timestamp
                ])
