"""
GCS成本配置工具

提供便捷的成本权重设置和优化功能
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class OptimizationPriority(Enum):
    """优化优先级枚举"""
    TIME = "time"                    # 时间优先
    PATH = "path"                    # 路径优先
    ENERGY = "energy"                # 能量优先
    BALANCED = "balanced"            # 平衡
    SMOOTHNESS = "smoothness"        # 平滑性优先
    CUSTOM = "custom"                # 自定义


@dataclass
class CostWeights:
    """成本权重配置"""
    time: float = 1.0
    path_length: float = 1.0
    energy: float = 0.0
    regularization_r: float = 0.0
    regularization_h: float = 0.0
    regularization_order: int = 2
    
    def normalize(self) -> 'CostWeights':
        """归一化权重"""
        total = (self.time + self.path_length + self.energy + 
                self.regularization_r + self.regularization_h)
        if total > 0:
            return CostWeights(
                time=self.time / total,
                path_length=self.path_length / total,
                energy=self.energy / total,
                regularization_r=self.regularization_r / total,
                regularization_h=self.regularization_h / total,
                regularization_order=self.regularization_order
            )
        return self
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'time': self.time,
            'path_length': self.path_length,
            'energy': self.energy,
            'regularization_r': self.regularization_r,
            'regularization_h': self.regularization_h,
            'regularization_order': self.regularization_order
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'CostWeights':
        """从字典创建"""
        return cls(
            time=d.get('time', 1.0),
            path_length=d.get('path_length', 1.0),
            energy=d.get('energy', 0.0),
            regularization_r=d.get('regularization_r', 0.0),
            regularization_h=d.get('regularization_h', 0.0),
            regularization_order=d.get('regularization_order', 2)
        )


class CostConfigurator:
    """成本配置器"""
    
    # 预定义配置模板
    PRESETS = {
        'time_optimal': CostWeights(
            time=10.0,
            path_length=0.5,
            energy=0.0,
            regularization_r=0.0,
            regularization_h=0.0
        ),
        'path_optimal': CostWeights(
            time=0.5,
            path_length=10.0,
            energy=0.0,
            regularization_r=0.0,
            regularization_h=0.0
        ),
        'energy_optimal': CostWeights(
            time=0.5,
            path_length=1.0,
            energy=5.0,
            regularization_r=0.0,
            regularization_h=0.0
        ),
        'balanced': CostWeights(
            time=1.0,
            path_length=1.0,
            energy=0.0,
            regularization_r=0.0,
            regularization_h=0.0
        ),
        'smooth': CostWeights(
            time=1.0,
            path_length=1.0,
            energy=2.0,
            regularization_r=0.5,
            regularization_h=0.5,
            regularization_order=2
        ),
        'lunar_standard': CostWeights(
            time=1.0,
            path_length=1.5,
            energy=20.0,
            regularization_r=0.3,
            regularization_h=0.3,
            regularization_order=2
        ),
        'lunar_high_risk': CostWeights(
            time=0.5,
            path_length=2.0,
            energy=3.0,
            regularization_r=1.0,
            regularization_h=1.0,
            regularization_order=2
        ),
        'lunar_emergency': CostWeights(
            time=10.0,
            path_length=0.5,
            energy=0.5,
            regularization_r=0.0,
            regularization_h=0.0
        ),
        'lunar_complex': CostWeights(
            time=1.5,
            path_length=2.0,
            energy=2.5,
            regularization_r=0.5,
            regularization_h=0.5,
            regularization_order=2
        ),
    }
    
    def __init__(self, priority: OptimizationPriority = OptimizationPriority.BALANCED):
        """
        初始化成本配置器
        
        Args:
            priority: 优化优先级
        """
        self.priority = priority
        self.weights = self._get_default_weights()
        self.characteristic_values = {
            'time': 10.0,      # 特征时间（秒）
            'length': 20.0,    # 特征长度（米）
            'velocity': 2.0,   # 特征速度（米/秒）
        }
    
    def _get_default_weights(self) -> CostWeights:
        """根据优先级获取默认权重"""
        priority_map = {
            OptimizationPriority.TIME: self.PRESETS['time_optimal'],
            OptimizationPriority.PATH: self.PRESETS['path_optimal'],
            OptimizationPriority.ENERGY: self.PRESETS['energy_optimal'],
            OptimizationPriority.BALANCED: self.PRESETS['balanced'],
            OptimizationPriority.SMOOTHNESS: self.PRESETS['smooth'],
            OptimizationPriority.CUSTOM: self.PRESETS['balanced'],
        }
        return priority_map[self.priority]
    
    def set_preset(self, preset_name: str) -> 'CostConfigurator':
        """
        设置预定义配置
        
        Args:
            preset_name: 预定义配置名称
            
        Returns:
            self（支持链式调用）
        """
        if preset_name in self.PRESETS:
            self.weights = self.PRESETS[preset_name]
        else:
            raise ValueError(f"未知的预定义配置: {preset_name}. "
                           f"可用配置: {list(self.PRESETS.keys())}")
        return self
    
    def set_weights(self, weights: CostWeights) -> 'CostConfigurator':
        """
        设置自定义权重
        
        Args:
            weights: 成本权重配置
            
        Returns:
            self（支持链式调用）
        """
        self.weights = weights
        self.priority = OptimizationPriority.CUSTOM
        return self
    
    def set_characteristic_values(self, time: float = None, length: float = None,
                                  velocity: float = None) -> 'CostConfigurator':
        """
        设置特征值（用于量纲归一化）
        
        Args:
            time: 特征时间（秒）
            length: 特征长度（米）
            velocity: 特征速度（米/秒）
            
        Returns:
            self（支持链式调用）
        """
        if time is not None:
            self.characteristic_values['time'] = time
        if length is not None:
            self.characteristic_values['length'] = length
        if velocity is not None:
            self.characteristic_values['velocity'] = velocity
        return self
    
    def get_normalized_weights(self) -> CostWeights:
        """
        获取归一化后的权重（考虑量纲）
        
        Returns:
            归一化后的成本权重
        """
        # 量纲归一化
        time_scale = 1.0 / self.characteristic_values['time']
        length_scale = 1.0 / self.characteristic_values['length']
        velocity_scale = 1.0 / (self.characteristic_values['velocity'] ** 2)
        
        normalized = CostWeights(
            time=self.weights.time * time_scale,
            path_length=self.weights.path_length * length_scale,
            energy=self.weights.energy * velocity_scale,
            regularization_r=self.weights.regularization_r,
            regularization_h=self.weights.regularization_h,
            regularization_order=self.weights.regularization_order
        )
        
        return normalized.normalize()
    
    def apply_to_gcs(self, gcs) -> None:
        """
        将成本配置应用到GCS实例
        
        Args:
            gcs: BezierGCS实例
        """
        # 添加时间成本
        if self.weights.time > 0:
            gcs.addTimeCost(weight=self.weights.time)
        
        # 添加路径长度成本
        if self.weights.path_length > 0:
            gcs.addPathLengthCost(weight=self.weights.path_length)
        
        # 添加能量成本
        if self.weights.energy > 0:
            gcs.addPathEnergyCost(weight=self.weights.energy)
        
        # 添加导数正则化
        if self.weights.regularization_r > 0 or self.weights.regularization_h > 0:
            gcs.addDerivativeRegularization(
                weight_r=self.weights.regularization_r,
                weight_h=self.weights.regularization_h,
                order=self.weights.regularization_order
            )
    
    def get_config_summary(self) -> str:
        """
        获取配置摘要
        
        Returns:
            配置摘要字符串
        """
        lines = [
            "=" * 60,
            "GCS成本配置摘要",
            "=" * 60,
            f"优化优先级: {self.priority.value}",
            "",
            "成本权重:",
            f"  时间成本: {self.weights.time:.2f}",
            f"  路径长度成本: {self.weights.path_length:.2f}",
            f"  能量成本: {self.weights.energy:.2f}",
            f"  空间正则化: {self.weights.regularization_r:.2f}",
            f"  时间正则化: {self.weights.regularization_h:.2f}",
            f"  正则化阶数: {self.weights.regularization_order}",
            "",
            "特征值:",
            f"  特征时间: {self.characteristic_values['time']:.1f}秒",
            f"  特征长度: {self.characteristic_values['length']:.1f}米",
            f"  特征速度: {self.characteristic_values['velocity']:.1f}米/秒",
            "",
            "预期性能:",
        ]
        
        # 根据权重预测性能
        total = self.weights.time + self.weights.path_length + self.weights.energy
        if total > 0:
            time_ratio = self.weights.time / total
            path_ratio = self.weights.path_length / total
            energy_ratio = self.weights.energy / total
            
            if time_ratio > 0.5:
                lines.append("时间优先：快速到达，可能路径较长")
            elif path_ratio > 0.5:
                lines.append("路径优先：最短路径，可能时间较长")
            elif energy_ratio > 0.5:
                lines.append("能量优先：平滑运动，节能效果显著")
            else:
                lines.append("平衡策略：综合性能良好")
        
        if self.weights.regularization_r > 0.5:
            lines.append("高平滑性：轨迹极度平滑")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def estimate_solve_time(self) -> Tuple[float, str]:
        """
        估计求解时间
        
        Returns:
            (估计时间, 难度描述)
        """
        # 基础时间（仅时间和路径成本）
        base_time = 0.5
        
        # 能量成本增加复杂度
        if self.weights.energy > 0:
            base_time += 0.5 * (1 + self.weights.energy / 5.0)
        
        # 正则化增加复杂度
        if self.weights.regularization_r > 0 or self.weights.regularization_h > 0:
            base_time += 0.3 * (1 + self.weights.regularization_order / 2.0)
        
        # 难度描述
        if base_time < 1.0:
            difficulty = "简单"
        elif base_time < 2.0:
            difficulty = "中等"
        elif base_time < 5.0:
            difficulty = "较难"
        else:
            difficulty = "困难"
        
        return base_time, difficulty


class CostOptimizer:
    """成本优化器 - 用于自动调优权重"""
    
    def __init__(self, gcs_factory, test_scenarios: List[Dict]):
        """
        初始化成本优化器
        
        Args:
            gcs_factory: GCS工厂函数
            test_scenarios: 测试场景列表
        """
        self.gcs_factory = gcs_factory
        self.test_scenarios = test_scenarios
        self.history = []
    
    def evaluate_weights(self, weights: CostWeights) -> Dict:
        """
        评估给定权重的性能
        
        Args:
            weights: 成本权重
            
        Returns:
            性能评估结果
        """
        results = []
        
        for scenario in self.test_scenarios:
            # 创建GCS
            gcs = self.gcs_factory(scenario)
            
            # 应用权重
            configurator = CostConfigurator()
            configurator.set_weights(weights)
            configurator.apply_to_gcs(gcs)
            
            # 求解
            import time
            start = time.time()
            trajectory, _ = gcs.SolvePath(rounding=True, verbose=False)
            solve_time = time.time() - start
            
            # 计算指标
            if trajectory is not None:
                traj_time = trajectory.end_time() - trajectory.start_time()
                # 计算路径长度和能量（简化）
                times = np.linspace(trajectory.start_time(), trajectory.end_time(), 100)
                waypoints = trajectory.vector_values(times)
                path_length = np.sum(np.linalg.norm(np.diff(waypoints, axis=1), axis=0))
                
                results.append({
                    'solve_time': solve_time,
                    'trajectory_time': traj_time,
                    'path_length': path_length,
                    'success': True
                })
            else:
                results.append({
                    'solve_time': solve_time,
                    'trajectory_time': float('inf'),
                    'path_length': float('inf'),
                    'success': False
                })
        
        # 汇总结果
        success_rate = sum(r['success'] for r in results) / len(results)
        avg_solve_time = np.mean([r['solve_time'] for r in results])
        avg_traj_time = np.mean([r['trajectory_time'] for r in results if r['success']])
        avg_path_length = np.mean([r['path_length'] for r in results if r['success']])
        
        return {
            'success_rate': success_rate,
            'avg_solve_time': avg_solve_time,
            'avg_trajectory_time': avg_traj_time,
            'avg_path_length': avg_path_length,
            'detailed_results': results
        }
    
    def optimize(self, target_metrics: Dict, max_iterations: int = 10) -> CostWeights:
        """
        优化权重以达到目标性能
        
        Args:
            target_metrics: 目标性能指标
            max_iterations: 最大迭代次数
            
        Returns:
            优化后的权重
        """
        # 初始权重
        current_weights = CostWeights()
        
        for iteration in range(max_iterations):
            # 评估当前权重
            performance = self.evaluate_weights(current_weights)
            self.history.append({
                'iteration': iteration,
                'weights': current_weights.to_dict(),
                'performance': performance
            })
            
            # 检查是否达到目标
            if self._check_target(performance, target_metrics):
                print(f"迭代{iteration}: 达到目标性能")
                break
            
            # 调整权重
            current_weights = self._adjust_weights(current_weights, performance, target_metrics)
            print(f"迭代{iteration}: 成功率={performance['success_rate']:.1%}, "
                  f"求解时间={performance['avg_solve_time']:.2f}s")
        
        return current_weights
    
    def _check_target(self, performance: Dict, target: Dict) -> bool:
        """检查是否达到目标"""
        if 'success_rate' in target and performance['success_rate'] < target['success_rate']:
            return False
        if 'max_solve_time' in target and performance['avg_solve_time'] > target['max_solve_time']:
            return False
        if 'max_trajectory_time' in target and performance['avg_trajectory_time'] > target['max_trajectory_time']:
            return False
        return True
    
    def _adjust_weights(self, weights: CostWeights, performance: Dict, 
                       target: Dict) -> CostWeights:
        """调整权重"""
        new_weights = CostWeights(
            time=weights.time,
            path_length=weights.path_length,
            energy=weights.energy,
            regularization_r=weights.regularization_r,
            regularization_h=weights.regularization_h,
            regularization_order=weights.regularization_order
        )
        
        # 根据性能调整
        if 'max_solve_time' in target and performance['avg_solve_time'] > target['max_solve_time']:
            # 求解时间过长，降低复杂度
            new_weights.energy *= 0.8
            new_weights.regularization_r *= 0.8
            new_weights.regularization_h *= 0.8
        
        if 'max_trajectory_time' in target and performance['avg_trajectory_time'] > target['max_trajectory_time']:
            # 轨迹时间过长，增加时间权重
            new_weights.time *= 1.5
        
        if 'max_path_length' in target and performance['avg_path_length'] > target['max_path_length']:
            # 路径过长，增加路径权重
            new_weights.path_length *= 1.5
        
        return new_weights


# 便捷函数
def get_lunar_standard_config() -> CostConfigurator:
    """获取月面标准配置"""
    return CostConfigurator().set_preset('lunar_standard')


def get_lunar_high_risk_config() -> CostConfigurator:
    """获取月面高风险配置"""
    return CostConfigurator().set_preset('lunar_high_risk')


def get_lunar_emergency_config() -> CostConfigurator:
    """获取月面紧急配置"""
    return CostConfigurator().set_preset('lunar_emergency')


def get_lunar_complex_config() -> CostConfigurator:
    """获取月面复杂地形配置"""
    return CostConfigurator().set_preset('lunar_complex')


# 使用示例
if __name__ == "__main__":
    # 示例1：使用预定义配置
    configurator = get_lunar_standard_config()
    print(configurator.get_config_summary())
    
    # 示例2：自定义权重
    custom_weights = CostWeights(
        time=2.0,
        path_length=3.0,
        energy=1.5,
        regularization_r=0.5,
        regularization_h=0.5
    )
    configurator = CostConfigurator().set_weights(custom_weights)
    print(configurator.get_config_summary())
    
    # 示例3：估计求解时间
    solve_time, difficulty = configurator.estimate_solve_time()
    print(f"\n估计求解时间: {solve_time:.2f}秒 ({difficulty})")
