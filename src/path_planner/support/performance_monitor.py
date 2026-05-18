"""
性能监测模块

提供全面的性能监测功能，包括：
- 时间监测（墙钟时间、CPU时间）
- 内存使用监测
- CPU使用率监测
- 多层级性能追踪
- 性能报告生成
"""

import time
import json
import psutil
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager


@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    # 时间指标
    start_time: float = 0.0
    end_time: float = 0.0
    wall_time: float = 0.0
    cpu_time: float = 0.0
    user_time: float = 0.0
    system_time: float = 0.0
    
    # 内存指标 (MB)
    memory_start: float = 0.0
    memory_end: float = 0.0
    memory_peak: float = 0.0
    memory_delta: float = 0.0
    
    # CPU指标
    cpu_percent_start: float = 0.0
    cpu_percent_end: float = 0.0
    cpu_percent_avg: float = 0.0
    
    # 调用次数
    call_count: int = 0
    
    # 子阶段指标
    sub_stages: Dict[str, 'PerformanceMetrics'] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'wall_time': self.wall_time,
            'cpu_time': self.cpu_time,
            'user_time': self.user_time,
            'system_time': self.system_time,
            'memory_start_mb': self.memory_start,
            'memory_end_mb': self.memory_end,
            'memory_peak_mb': self.memory_peak,
            'memory_delta_mb': self.memory_delta,
            'cpu_percent_avg': self.cpu_percent_avg,
            'call_count': self.call_count,
            'sub_stages': {k: v.to_dict() for k, v in self.sub_stages.items()}
        }


class PerformanceMonitor:
    """性能监测器
    
    提供全面的性能监测功能，包括：
    - 时间监测（墙钟时间、CPU时间）
    - 内存使用监测
    - CPU使用率监测
    - 多层级性能追踪
    - 性能报告生成
    """
    
    def __init__(self, enabled: bool = True, verbose: bool = False):
        self.enabled = enabled
        self.verbose = verbose
        self.metrics_stack: List[PerformanceMetrics] = []
        self.root_metrics: Optional[PerformanceMetrics] = None
        self._lock = threading.Lock()
        self._memory_monitor_thread: Optional[threading.Thread] = None
        self._monitoring = False
        
    def start(self, stage_name: str = "root") -> PerformanceMetrics:
        """开始监测一个阶段"""
        if not self.enabled:
            return PerformanceMetrics()
        
        with self._lock:
            metrics = PerformanceMetrics()
            
            # 记录开始时间和资源
            process = psutil.Process()
            metrics.start_time = time.time()
            metrics.memory_start = process.memory_info().rss / 1024 / 1024  # MB
            metrics.cpu_percent_start = process.cpu_percent()
            
            # 记录CPU时间
            cpu_times = process.cpu_times()
            metrics.user_time = cpu_times.user
            metrics.system_time = cpu_times.system
            
            self.metrics_stack.append(metrics)
            
            if self.verbose:
                indent = "  " * (len(self.metrics_stack) - 1)
                print(f"{indent}[开始] {stage_name}")
            
            return metrics
    
    def end(self, stage_name: str = "unknown") -> PerformanceMetrics:
        """结束当前阶段的监测"""
        if not self.enabled or not self.metrics_stack:
            return PerformanceMetrics()
        
        with self._lock:
            metrics = self.metrics_stack.pop()
            
            # 记录结束时间和资源
            process = psutil.Process()
            metrics.end_time = time.time()
            metrics.memory_end = process.memory_info().rss / 1024 / 1024  # MB
            metrics.cpu_percent_end = process.cpu_percent()
            
            # 计算时间差
            metrics.wall_time = metrics.end_time - metrics.start_time
            metrics.memory_delta = metrics.memory_end - metrics.memory_start
            
            # 计算CPU时间
            cpu_times = process.cpu_times()
            metrics.cpu_time = (cpu_times.user - metrics.user_time + 
                               cpu_times.system - metrics.system_time)
            
            # 计算平均CPU使用率
            if metrics.wall_time > 0:
                metrics.cpu_percent_avg = (metrics.cpu_time / metrics.wall_time) * 100
            
            # 如果有父阶段，添加为子阶段
            if self.metrics_stack:
                parent = self.metrics_stack[-1]
                parent.sub_stages[stage_name] = metrics
            else:
                self.root_metrics = metrics
            
            if self.verbose:
                indent = "  " * len(self.metrics_stack)
                print(f"{indent}[结束] {stage_name}: "
                      f"耗时={metrics.wall_time:.4f}s, "
                      f"内存={metrics.memory_delta:+.2f}MB, "
                      f"CPU={metrics.cpu_percent_avg:.1f}%")
            
            return metrics
    
    @contextmanager
    def track(self, stage_name: str):
        """上下文管理器方式追踪性能"""
        metrics = self.start(stage_name)
        try:
            yield metrics
        finally:
            self.end(stage_name)
    
    def get_summary(self) -> Dict:
        """获取性能摘要"""
        if not self.root_metrics:
            return {}
        
        return {
            'total_time': self.root_metrics.wall_time,
            'cpu_time': self.root_metrics.cpu_time,
            'memory_usage': self.root_metrics.memory_delta,
            'memory_peak': self.root_metrics.memory_peak,
            'cpu_efficiency': self.root_metrics.cpu_percent_avg,
            'breakdown': self._get_stage_breakdown(self.root_metrics)
        }
    
    def _get_stage_breakdown(self, metrics: PerformanceMetrics, 
                            parent_time: float = None) -> Dict:
        """获取阶段分解"""
        if parent_time is None:
            parent_time = metrics.wall_time
        
        breakdown = {}
        for stage_name, stage_metrics in metrics.sub_stages.items():
            percentage = (stage_metrics.wall_time / parent_time * 100 
                         if parent_time > 0 else 0)
            breakdown[stage_name] = {
                'time': stage_metrics.wall_time,
                'percentage': percentage,
                'memory': stage_metrics.memory_delta,
                'sub_stages': self._get_stage_breakdown(
                    stage_metrics, stage_metrics.wall_time
                )
            }
        return breakdown
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """生成性能报告"""
        if not self.root_metrics:
            return "无性能数据"
        
        report_lines = [
            "=" * 80,
            "性能监测报告",
            "=" * 80,
            "",
            "总体性能:",
            f"  总耗时: {self.root_metrics.wall_time:.4f} 秒",
            f"  CPU时间: {self.root_metrics.cpu_time:.4f} 秒",
            f"  CPU效率: {self.root_metrics.cpu_percent_avg:.1f}%",
            f"  内存变化: {self.root_metrics.memory_delta:+.2f} MB",
            f"  内存峰值: {self.root_metrics.memory_peak:.2f} MB",
            "",
            "阶段分解:",
        ]
        
        # 添加阶段分解
        self._add_stage_report(report_lines, self.root_metrics, indent=2)
        
        report_lines.extend([
            "",
            "=" * 80
        ])
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            if self.verbose:
                print(f"性能报告已保存至: {output_file}")
        
        return report
    
    def _add_stage_report(self, lines: List[str], metrics: PerformanceMetrics, 
                         indent: int = 0):
        """添加阶段报告"""
        prefix = " " * indent
        
        for stage_name, stage_metrics in metrics.sub_stages.items():
            percentage = (stage_metrics.wall_time / self.root_metrics.wall_time * 100 
                         if self.root_metrics.wall_time > 0 else 0)
            
            lines.append(
                f"{prefix}- {stage_name}: "
                f"{stage_metrics.wall_time:.4f}s ({percentage:.1f}%), "
                f"内存: {stage_metrics.memory_delta:+.2f}MB"
            )
            
            if stage_metrics.sub_stages:
                self._add_stage_report(lines, stage_metrics, indent + 4)
    
    def export_json(self, output_file: str):
        """导出性能数据为JSON"""
        if not self.root_metrics:
            return
        
        data = {
            'summary': self.get_summary(),
            'detailed_metrics': self.root_metrics.to_dict()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        if self.verbose:
            print(f"性能数据已导出至: {output_file}")
    
    def reset(self):
        """重置监测器"""
        with self._lock:
            self.metrics_stack.clear()
            self.root_metrics = None
