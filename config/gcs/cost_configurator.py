"""
GCS成本配置工具

提供便捷的成本权重设置和优化功能
"""

from typing import Dict, Tuple
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
        # 曲率硬约束场景预设
        # w_time/w_energy比值控制速度水平，影响保守性
        # regularization_r限制||r''||上界，协同曲率约束
        'curvature_constrained': CostWeights(
            time=3.0,
            path_length=1.5,
            energy=3.0,
            regularization_r=5.0,
            regularization_h=0.5,
            regularization_order=2
        ),
        'curvature_constrained_high_speed': CostWeights(
            time=5.0,
            path_length=1.0,
            energy=2.0,
            regularization_r=8.0,
            regularization_h=0.5,
            regularization_order=2
        ),
        'curvature_constrained_parking': CostWeights(
            time=2.0,
            path_length=2.0,
            energy=4.0,
            regularization_r=10.0,
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
