"""
曲率约束v2数据结构定义
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


@dataclass
class CurvatureV2Result:
    """曲率约束v2的构建结果

    Attributes:
        constraints_A1: 线性速度下界约束列表 (qᵢ·d_θ ≥ σ_e)
        constraints_A2: 旋转二阶锥约束列表 (τ_e·1 ≥ σ_e²)
        constraints_B: Lorentz锥曲率-速度耦合约束列表 (κ_max·τ_e ≥ ‖Qⱼ‖₂)
        constraints_C: σ_e下界约束列表 (σ_e ≥ σ_min)
        all_bindings: 所有(edge, binding, type)三元组
        num_interior_edges: 应用约束的内部边数量
        sigma_min: 使用的σ_min值
        max_curvature: 使用的κ_max值
    """
    constraints_A1: List = field(default_factory=list)
    constraints_A2: List = field(default_factory=list)
    constraints_B: List = field(default_factory=list)
    constraints_C: List = field(default_factory=list)
    all_bindings: List[Tuple] = field(default_factory=list)
    num_interior_edges: int = 0
    sigma_min: float = 0.0
    max_curvature: float = 0.0

    @property
    def total_constraints(self) -> int:
        """总约束数量"""
        return (len(self.constraints_A1) + len(self.constraints_A2) +
                len(self.constraints_B) + len(self.constraints_C))

    @property
    def constraints_per_edge(self) -> Dict[str, int]:
        """每类约束数量统计"""
        return {
            'A1': len(self.constraints_A1),
            'A2': len(self.constraints_A2),
            'B': len(self.constraints_B),
            'C': len(self.constraints_C),
        }


@dataclass
class ValidationReport:
    """约束验证报告

    Attributes:
        checks: 检查项名称到结果的映射
        failures: 失败项列表
        curvature_violation: 最大曲率违反量（若有轨迹）
        all_passed: 是否全部通过
    """
    checks: Dict[str, bool] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    curvature_violation: float = 0.0
    all_passed: bool = True

    def check(self, name: str, passed: bool):
        """添加一个检查项

        Args:
            name: 检查项名称
            passed: 是否通过
        """
        self.checks[name] = passed
        if not passed:
            self.failures.append(name)
            self.all_passed = False

    def __repr__(self) -> str:
        n_total = len(self.checks)
        n_passed = sum(1 for v in self.checks.values() if v)
        return (f"ValidationReport(passed={n_passed}/{n_total}, "
                f"failures={self.failures}, "
                f"curvature_violation={self.curvature_violation:.6f})")
