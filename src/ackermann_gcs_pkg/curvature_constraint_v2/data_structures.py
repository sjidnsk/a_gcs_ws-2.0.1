"""
曲率约束v2数据结构定义

包含：
- CurvatureV2Result: 曲率约束v2构建结果
- ValidationReport: 约束验证报告
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any


@dataclass
class CurvatureV2Result:
    """曲率约束v2的构建结果

    Attributes:
        constraints_A1: 线性速度下界约束列表 (q_i · d_θ ≥ σ_e)
        constraints_A2: 旋转二阶锥约束列表 (τ_e · 1 ≥ σ_e²)
        constraints_B: Lorentz锥曲率-速度耦合约束列表 (κ_max · τ_e ≥ ‖Q_j‖)
        constraints_C: σ_e下界约束列表 (σ_e ≥ σ_min)
        all_bindings: 所有(edge, binding, type)三元组列表
        num_interior_edges: 应用约束的内部边数量
        sigma_min: 使用的σ_min值
        max_curvature: 使用的κ_max值
    """
    constraints_A1: List[Any] = field(default_factory=list)
    constraints_A2: List[Any] = field(default_factory=list)
    constraints_B: List[Any] = field(default_factory=list)
    constraints_C: List[Any] = field(default_factory=list)
    all_bindings: List[Tuple[Any, Any, str]] = field(default_factory=list)
    num_interior_edges: int = 0
    sigma_min: float = 0.0
    max_curvature: float = 0.0

    @property
    def total_constraints(self) -> int:
        """总约束数量"""
        return (len(self.constraints_A1) + len(self.constraints_A2) +
                len(self.constraints_B) + len(self.constraints_C))

    def summary(self) -> str:
        """返回约束统计摘要"""
        return (
            f"CurvatureV2Result: {self.num_interior_edges} interior edges, "
            f"A1={len(self.constraints_A1)}, A2={len(self.constraints_A2)}, "
            f"B={len(self.constraints_B)}, C={len(self.constraints_C)}, "
            f"total={self.total_constraints}, "
            f"σ_min={self.sigma_min:.6f}, κ_max={self.max_curvature:.4f}"
        )


@dataclass
class ValidationReport:
    """约束验证报告

    Attributes:
        checks: 检查项名称到结果的映射
        failures: 失败项列表
        curvature_violation: 最大曲率违反量（若有轨迹），0表示无违反
        all_passed: 是否全部通过
    """
    checks: Dict[str, bool] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    curvature_violation: float = 0.0
    all_passed: bool = True

    def check(self, name: str, passed: bool) -> None:
        """添加一个检查项

        Args:
            name: 检查项名称
            passed: 是否通过
        """
        self.checks[name] = passed
        if not passed:
            self.failures.append(name)
            self.all_passed = False

    def summary(self) -> str:
        """返回验证摘要"""
        status = "PASSED" if self.all_passed else "FAILED"
        n_passed = sum(1 for v in self.checks.values() if v)
        n_total = len(self.checks)
        result = f"ValidationReport: {status} ({n_passed}/{n_total} checks passed)"
        if self.failures:
            result += f", failures: {self.failures}"
        if self.curvature_violation > 0:
            result += f", curvature_violation={self.curvature_violation:.6f}"
        return result
