"""
h_bar_prime 迭代修正模块

实现 h̄' 的自动迭代收敛流程，使曲率硬约束逐步精确化。

流程：
1. 迭代1：求解无曲率约束的 GCS → compute_h_bar_prime_from_trajectory
2. 迭代2+：用 h̄' 添加曲率约束 → 求解 → 计算新 h̄'
3. 收敛判定 / 放宽重试
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .ackermann_data_structures import TrajectoryConstraints


@dataclass
class HBarPrimeIterationResult:
    """h̄' 迭代修正结果。

    Attributes:
        h_bar_prime: 最终 h̄' 值。
        effective_h_bar_prime: 经 safety_factor 修正后的值。
        converged: 是否收敛。
        num_iterations: 实际迭代次数。
        iteration_history: 每次迭代的 h̄' 值。
        convergence_reason: 收敛/终止原因。
        relax_attempts: 放宽重试次数。
        final_safety_factor: 最终使用的 safety_factor（可能因
            h̄' 下降而动态收紧）。
    """

    h_bar_prime: float
    effective_h_bar_prime: float
    converged: bool
    num_iterations: int
    iteration_history: List[float] = field(default_factory=list)
    convergence_reason: str = ""
    relax_attempts: int = 0
    final_safety_factor: float = 1.0


def _select_trajectory_with_max_h_bar_prime(
    trajectory, result, BezierGCS, verbose
):
    """在候选轨迹中选择 h̄' 最大的（约束最保守）。

    Args:
        trajectory: 默认最优轨迹。
        result: SolvePathWithConstraints 返回值。
        BezierGCS: BezierGCS 类（用于 compute_h_bar_prime_from_trajectory）。
        verbose: 是否输出日志。

    Returns:
        选择后的轨迹。
    """
    if trajectory is None:
        return None
    results_dict = result[1] if len(result) > 1 else {}
    all_candidates = results_dict.get("all_candidate_trajectories", [])
    if len(all_candidates) <= 1:
        return trajectory
    best_h = -np.inf
    best_traj = trajectory
    for cand_traj in all_candidates:
        try:
            cand_h = BezierGCS.compute_h_bar_prime_from_trajectory(
                cand_traj)
            if cand_h > best_h:
                best_h = cand_h
                best_traj = cand_traj
        except (ValueError, AttributeError):
            continue
    if best_traj is not trajectory:
        if verbose:
            print(
                f"[h̄' Iteration] Selected candidate with "
                f"higher h̄'={best_h:.6f}"
            )
    return best_traj


def iterate_h_bar_prime(
    bezier_gcs,
    constraints: TrajectoryConstraints,
    cost_weights: dict,
    source,
    target,
    workspace_regions,
    verbose: bool = True,
) -> Tuple[Optional[object], HBarPrimeIterationResult]:
    """执行 h̄' 迭代修正流程。

    流程：
    1. 迭代1：求解无曲率约束的 GCS →
       compute_h_bar_prime_from_trajectory
    2. 迭代2+：用 h̄' 添加曲率约束 → 求解 → 计算新 h̄'
    3. 收敛判定：|h_curr - h_prev| / h_prev < threshold
    4. 求解失败：h̄' *= relax_factor 重试

    Args:
        bezier_gcs: 已初始化的 AckermannBezierGCS 实例。
        constraints: 轨迹约束（含 h̄' 迭代参数）。
        cost_weights: 成本权重字典。
        source: 起点状态。
        target: 终点状态。
        workspace_regions: 工作空间区域。
        verbose: 是否输出详细日志。

    Returns:
        (最优轨迹, 迭代结果)。轨迹可能为 None（求解失败）。
    """
    from gcs_pkg.scripts.core.bezier import BezierGCS

    max_iterations = constraints.max_h_bar_prime_iterations
    convergence_threshold = constraints.h_bar_prime_convergence_threshold
    safety_factor = constraints.h_bar_prime_safety_factor
    safety_factor_decay = constraints.h_bar_prime_safety_factor_decay
    relax_factor = constraints.h_bar_prime_relax_factor
    max_relax_attempts = constraints.max_h_bar_prime_relax_attempts

    h_bar_prime_history: List[float] = []
    best_trajectory = None
    h_bar_prime = 1.0  # 初始默认值
    total_relax_attempts = 0
    converged = False
    convergence_reason = "max_iterations"
    # 动态 safety_factor：初始为用户指定值，可能在迭代中收紧
    dynamic_safety_factor = safety_factor

    for iter_num in range(1, max_iterations + 1):
        if verbose:
            print(f"\n[h̄' Iteration] Iteration {iter_num}/{max_iterations}")

        # 迭代1：无曲率约束求解
        # 迭代2+：带曲率约束求解
        if iter_num == 1:
            # 第一次迭代：不添加曲率约束，直接求解
            if verbose:
                print("[h̄' Iteration] Solving without curvature constraint...")
        else:
            # 后续迭代：先移除旧曲率约束，再添加新约束
            if bezier_gcs.curvature_constraints:
                if verbose:
                    print(
                        "[h̄' Iteration] Removing previous curvature "
                        "constraints..."
                    )
                bezier_gcs.removeCurvatureHardConstraints(verbose=verbose)

            effective_hbp = h_bar_prime * dynamic_safety_factor
            if verbose:
                print(
                    f"[h̄' Iteration] Adding curvature constraint: "
                    f"h̄'={h_bar_prime:.6f}, "
                    f"safety_factor={dynamic_safety_factor:.6f}, "
                    f"effective={effective_hbp:.6f}"
                )
            try:
                bezier_gcs.addCurvatureHardConstraint(
                    max_curvature=constraints.max_curvature,
                    min_velocity=constraints.min_velocity,
                    h_bar_prime=h_bar_prime,
                    h_bar_prime_safety_factor=dynamic_safety_factor,
                )
            except ValueError as e:
                if verbose:
                    print(
                        f"[h̄' Iteration] Failed to add curvature "
                        f"constraint: {e}"
                    )
                convergence_reason = "constraint_failed"
                break

        # 求解
        result = bezier_gcs.SolvePathWithConstraints(
            rounding=True, preprocessing=True, verbose=False
        )
        trajectory = result[0] if result is not None else None

        # 在候选轨迹中选择 h̄' 最大的（约束最保守）
        trajectory = _select_trajectory_with_max_h_bar_prime(
            trajectory, result, BezierGCS, verbose
        )

        # 求解失败处理：放宽重试
        if trajectory is None and iter_num > 1:
            if verbose:
                print("[h̄' Iteration] Solve failed, attempting relax...")
            relax_success = False
            for relax_num in range(1, max_relax_attempts + 1):
                h_bar_prime *= relax_factor
                total_relax_attempts += 1
                if verbose:
                    print(
                        f"[h̄' Iteration] Relax attempt {relax_num}: "
                        f"h̄' *= {relax_factor} -> {h_bar_prime:.6f}"
                    )
                try:
                    # 先移除当前曲率约束，再添加放宽后的约束
                    bezier_gcs.removeCurvatureHardConstraints(
                        verbose=verbose
                    )
                    bezier_gcs.addCurvatureHardConstraint(
                        max_curvature=constraints.max_curvature,
                        min_velocity=constraints.min_velocity,
                        h_bar_prime=h_bar_prime,
                        h_bar_prime_safety_factor=dynamic_safety_factor,
                    )
                except ValueError:
                    continue
                result = bezier_gcs.SolvePathWithConstraints(
                    rounding=True, preprocessing=True, verbose=False
                )
                trajectory = result[0] if result is not None else None
                # 在候选轨迹中选择 h̄' 最大的
                trajectory = _select_trajectory_with_max_h_bar_prime(
                    trajectory, result, BezierGCS, verbose
                )
                if trajectory is not None:
                    relax_success = True
                    break

            if not relax_success:
                if verbose:
                    print(
                        "[h̄' Iteration] All relax attempts failed. "
                        "Stopping iteration."
                    )
                convergence_reason = "solve_failed"
                break

        elif trajectory is None and iter_num == 1:
            if verbose:
                print(
                    "[h̄' Iteration] First solve failed (no curvature "
                    "constraint). Stopping iteration."
                )
            convergence_reason = "solve_failed"
            break

        # 从轨迹计算新 h̄'
        try:
            h_new = BezierGCS.compute_h_bar_prime_from_trajectory(
                trajectory
            )
        except (ValueError, AttributeError) as e:
            if verbose:
                print(
                    f"[h̄' Iteration] Failed to compute h̄' from "
                    f"trajectory: {e}"
                )
            convergence_reason = "compute_failed"
            break

        h_bar_prime_history.append(h_new)
        best_trajectory = trajectory

        if verbose:
            print(f"[h̄' Iteration] Computed h̄' = {h_new:.6f}")

        # 收敛判定（iter >= 2）
        if iter_num >= 2:
            h_prev = h_bar_prime_history[-2]
            relative_change = abs(h_new - h_prev) / h_prev
            if verbose:
                print(
                    f"[h̄' Iteration] Relative change: "
                    f"{relative_change:.6f} "
                    f"(threshold: {convergence_threshold})"
                )
            if relative_change < convergence_threshold:
                converged = True
                convergence_reason = "converged"
                if verbose:
                    print("[h̄' Iteration] Converged!")
                break

            # 单调性检查：h̄' 显著下降时动态收紧 safety_factor
            if h_new < h_prev * (1 - convergence_threshold):
                old_sf = dynamic_safety_factor
                dynamic_safety_factor *= safety_factor_decay
                warnings.warn(
                    f"h̄' decreased significantly: "
                    f"{h_prev:.6f} -> {h_new:.6f}. "
                    f"Tightening safety_factor: "
                    f"{old_sf:.6f} -> {dynamic_safety_factor:.6f} "
                    f"(decay={safety_factor_decay}).",
                    UserWarning,
                    stacklevel=2,
                )

        h_bar_prime = h_new

    # 构造结果
    final_h_bar_prime = h_bar_prime_history[-1] if h_bar_prime_history else h_bar_prime
    effective_h_bar_prime = final_h_bar_prime * dynamic_safety_factor

    result = HBarPrimeIterationResult(
        h_bar_prime=final_h_bar_prime,
        effective_h_bar_prime=effective_h_bar_prime,
        converged=converged,
        num_iterations=len(h_bar_prime_history),
        iteration_history=list(h_bar_prime_history),
        convergence_reason=convergence_reason,
        relax_attempts=total_relax_attempts,
        final_safety_factor=dynamic_safety_factor,
    )

    if verbose:
        print(f"\n[h̄' Iteration] Final result:")
        print(f"  h̄' = {result.h_bar_prime:.6f}")
        print(f"  effective h̄' = {result.effective_h_bar_prime:.6f}")
        print(f"  final safety_factor = {result.final_safety_factor:.6f}")
        if result.final_safety_factor != safety_factor:
            print(
                f"  (tightened from initial {safety_factor:.6f})"
            )
        print(f"  converged = {result.converged}")
        print(f"  iterations = {result.num_iterations}")
        print(f"  reason = {result.convergence_reason}")
        if result.relax_attempts > 0:
            print(f"  relax attempts = {result.relax_attempts}")

    return best_trajectory, result
