# h̄' 迭代修正完整报告

> **版本**: v1.0  
> **日期**: 2026-05-13  
> **源码**: `h_bar_prime_iteration.py`, `bezier.py`, `ackermann_gcs_planner.py`

---

## 目录

1. [问题背景](#1-问题背景)
2. [数学推导](#2-数学推导)
3. [h̄' 的物理意义与计算](#3-h̄-的物理意义与计算)
4. [迭代修正算法](#4-迭代修正算法)
5. [收敛机制](#5-收敛机制)
6. [求解失败处理：放宽重试](#6-求解失败处理放宽重试)
7. [动态 safety_factor 收紧](#7-动态-safety_factor-收紧)
8. [候选轨迹选择策略](#8-候选轨迹选择策略)
9. [约束移除与重建](#9-约束移除与重建)
10. [两种使用模式](#10-两种使用模式)
11. [参数配置指南](#11-参数配置指南)
12. [完整伪代码](#12-完整伪代码)
13. [与规划器的集成](#13-与规划器的集成)
14. [设计决策与权衡](#14-设计决策与权衡)

---

## 1. 问题背景

### 1.1 曲率硬约束的保守性困境

AckermannGCS规划器使用 **Lorentz锥凸约束** 保证曲率不超限：

```
||Q_j||_2 ≤ C = κ_max · ρ_min²
```

其中阈值 C 依赖于 **ρ_min**（弧长参数下的最小速度下界）：

```
ρ_min = v_min · h̄' · safety_factor
```

这里出现一个关键问题：**h̄' = mean(dh/ds) 在求解前未知**。

### 1.2 h̄' 未知时的困境

| h̄' 取值 | 后果 |
|----------|------|
| **过小** (如 0.1) | ρ_min 小 → C 小 → 约束过紧 → **可能无可行解** |
| **过大** (如 10.0) | ρ_min 大 → C 大 → 约束过松 → **曲率可能违反** |
| **默认 1.0** | 不一定合理，取决于场景 → **结果不可预测** |

### 1.3 迭代修正的核心思想

**h̄' 由轨迹决定，而轨迹又受 h̄' 影响的约束决定**——这是一个循环依赖。

迭代修正打破循环依赖：

```
无约束求解 → 得到轨迹 → 从轨迹计算 h̄' → 添加约束 → 再求解 → 更新 h̄' → 收敛
```

---

## 2. 数学推导

### 2.1 曲率硬约束的充分条件推导

**原始曲率约束**:

```
|κ(s)| ≤ κ_max
```

**Step 1 — Cauchy-Schwarz 不等式**:

```
|κ(s)| = |ẋ·ÿ - ẏ·ẍ| / (ẋ² + ẏ²)^(3/2)
       ≤ ||r''(s)|| / ||r'(s)||²
```

**Step 2 — 速度下界**:

在弧长参数 σ 下，||r'(σ)|| = ρ(σ) ≥ ρ_min，其中：

```
ρ(σ) = ||dr/dσ|| = ||r'(s)|| / h'(s)
ρ_min = v_min · h̄' · safety_factor
```

因此：

```
|κ(s)| ≤ ||r''(s)|| / ρ_min²
```

**Step 3 — 凸包特性**:

贝塞尔曲线的二阶导数在控制点凸包内：

```
||r''(s)|| ≤ max_j ||Q_j||
```

其中 Q_j = n(n-1)(P_{j+2} - 2P_{j+1} + P_j)。

**Step 4 — 充分条件**:

```
若 ||Q_j||_2 ≤ C = κ_max · ρ_min²，则 |κ(s)| ≤ κ_max
```

### 2.2 ρ_min 对 h̄' 的依赖

展开 ρ_min：

```
ρ_min = v_min · h̄'_effective
h̄'_effective = h̄' · safety_factor
```

因此阈值 C 对 h̄' 的敏感度为 **二次**：

```
C = κ_max · (v_min · h̄' · safety_factor)²
  = κ_max · v_min² · h̄'² · safety_factor²
```

**h̄' 变化 2 倍 → C 变化 4 倍**。这就是为什么 h̄' 的精确估计至关重要。

### 2.3 保守因子

保守因子 α 衡量约束的保守程度：

```
α = (v_max / v_min)²
```

| v_min (m/s) | v_max (m/s) | α | 保守程度 |
|-------------|-------------|---|---------|
| 1.0 | 10.0 | 100 | 极度保守 |
| 1.58 | 10.0 | 40.0 | 较保守 |
| 3.0 | 10.0 | 11.1 | 适中 |
| 5.0 | 10.0 | 4.0 | 较精确 |

---

## 3. h̄' 的物理意义与计算

### 3.1 物理意义

h̄' 是时间缩放导数 dh/ds 在参数区间上的均值：

```
h̄' = (1/Δs) · ∫_{s_start}^{s_end} h'(s) ds
```

其物理含义是 **弧长参数 s 与物理时间 t 之间的平均转换率**：

```
h̄' ≈ L_path / T_total
```

- L_path: 轨迹空间长度（米）
- T_total: 轨迹总物理时间（秒）
- h̄' 单位: 米/秒（即平均速度的倒数在参数空间的表示）

### 3.2 精确计算：梯形法则数值积分

**方法**: `BezierGCS.compute_h_bar_prime_from_trajectory(trajectory, num_samples=200)`

**算法**:

```python
# 1. 生成采样点
s_points = linspace(s_start, s_end, num_samples + 1)  # 201个点

# 2. 获取时间轨迹的一阶导数 h'(s) = dh/ds
h_deriv = trajectory.time_traj.MakeDerivative(1)

# 3. 批量计算 h'(s_i)
h_prime_values = [h_deriv.value(s)[0,0] for s in s_points]

# 4. 梯形法则数值积分
integral = trapezoid(h_prime_values, s_points)

# 5. 计算均值
h_bar_prime = integral / (s_end - s_start)
```

**验证**:
- 有限性检查: `isfinite(h_bar_prime) and h_bar_prime > 0`
- 否则抛出 `ValueError`

**精度**: num_samples=200 提供足够精度，梯形法则对光滑函数误差 O(1/N²)。

### 3.3 静态估算（求解前使用）

**方法**: `BezierGCS.estimate_h_bar_prime(path_length_estimate, num_segments, ...)`

```
h̄' ≈ L_path / (N_segments · v_optimal)
```

其中 `v_optimal = sqrt(w_time / w_energy)`。

**用途**: 在迭代修正之前，提供一个初始估计。有下界保护 `hdot_min = 0.01`。

**精度**: 中等，依赖于路径长度估计的准确性。

---

## 4. 迭代修正算法

### 4.1 算法总览

```
输入: bezier_gcs (已初始化，含速度约束+成本函数)
      constraints (含 h̄' 迭代参数)
输出: (best_trajectory, HBarPrimeIterationResult)

初始化:
  h_bar_prime ← 1.0                    # 初始默认值
  dynamic_safety_factor ← safety_factor  # 初始为0.7
  h_bar_prime_history ← []
  converged ← False
  convergence_reason ← "max_iterations"

for iter_num = 1, 2, ..., max_iterations:

    ┌─ 迭代1: 不添加曲率约束，直接求解
    │  (获取无约束基线轨迹，计算初始h̄')
    │
    ├─ 迭代2+:
    │  1. 移除旧曲率约束: removeCurvatureHardConstraints()
    │  2. 计算有效h̄': effective = h̄' × dynamic_safety_factor
    │  3. 添加新曲率约束: addCurvatureHardConstraint(
    │       max_curvature, min_velocity, h̄', dynamic_safety_factor)
    │
    ├─ 求解: SolvePathWithConstraints(rounding=True)
    │
    ├─ 候选选择: 选择h̄'最大的候选轨迹
    │
    ├─ 求解失败处理 (iter > 1):
    │  └─ 放宽重试: h̄' *= relax_factor, 最多 max_relax_attempts 次
    │
    ├─ 从轨迹计算新h̄': compute_h_bar_prime_from_trajectory()
    │
    ├─ 收敛判定 (iter ≥ 2):
    │  └─ |h_new - h_prev| / h_prev < convergence_threshold → 收敛
    │
    └─ 单调性检查 (iter ≥ 2):
       └─ h̄' 显著下降 → dynamic_safety_factor *= decay

返回: (best_trajectory, HBarPrimeIterationResult)
```

### 4.2 迭代1详解：无约束基线

**目的**: 获取一个不受曲率约束影响的"自由"轨迹，从中计算真实的 h̄' 初始值。

```
迭代1:
  不添加曲率约束
  result = SolvePathWithConstraints(rounding=True)
  trajectory = 选择h̄'最大的候选
  h_new = compute_h_bar_prime_from_trajectory(trajectory)
  h_bar_prime_history.append(h_new)
```

**关键点**: 
- 此时 GCS 中已有速度约束、航向角约束、成本函数，但无曲率约束
- 求解得到的轨迹是"忽略曲率的最优轨迹"
- 从此轨迹计算的 h̄' 反映了无曲率约束时的真实时间-空间关系

### 4.3 迭代2+详解：约束-求解-更新循环

```
迭代k (k ≥ 2):
  1. 移除旧约束: removeCurvatureHardConstraints()
  2. 添加新约束: addCurvatureHardConstraint(
       κ_max, v_min, h̄'=h_bar_prime_history[-1], 
       safety_factor=dynamic_safety_factor)
  3. 求解: result = SolvePathWithConstraints(rounding=True)
  4. 候选选择: 选择h̄'最大的候选
  5. 计算新h̄': h_new = compute_h_bar_prime_from_trajectory(trajectory)
  6. 收敛判定: |h_new - h_prev| / h_prev < threshold?
  7. 单调性检查: h̄' 显著下降?
  8. 更新: h_bar_prime ← h_new
```

---

## 5. 收敛机制

### 5.1 收敛判定

**条件** (仅在 iter ≥ 2 时判定):

```
relative_change = |h_new - h_prev| / h_prev < convergence_threshold
```

**默认参数**: convergence_threshold = 0.15 (15%)

**含义**: 连续两次迭代的 h̄' 相对变化小于 15%，认为已收敛。

### 5.2 收敛原因枚举

| convergence_reason | 含义 |
|-------------------|------|
| `"converged"` | 相对变化小于阈值，成功收敛 |
| `"max_iterations"` | 达到最大迭代次数仍未收敛 |
| `"solve_failed"` | 第1次迭代求解失败，或放宽重试全部失败 |
| `"constraint_failed"` | 添加曲率约束时参数验证失败 (ValueError) |
| `"compute_failed"` | 从轨迹计算 h̄' 失败 |

### 5.3 收敛性分析

**为什么期望收敛？**

h̄' 和曲率约束之间存在负反馈：

1. h̄' 大 → C 大 → 约束宽松 → 轨迹更自由 → 速度可能更高 → h̄' 可能减小
2. h̄' 小 → C 小 → 约束严格 → 轨迹受限 → 速度可能更低 → h̄' 可能增大

这种负反馈使得 h̄' 趋向于某个固定点。

**不收敛的情况**:
- 问题本身无可行解（约束过严）
- 成本权重导致极端行为（如 w_energy 极大导致速度极低）
- 几何场景导致 h̄' 振荡

---

## 6. 求解失败处理：放宽重试

### 6.1 触发条件

仅在 **iter > 1**（已添加曲率约束）且求解失败时触发。

> 第1次迭代求解失败直接终止，因为此时无曲率约束，求解失败说明问题本身无解。

### 6.2 放宽策略

```
for relax_num = 1, 2, ..., max_relax_attempts:
    h̄' *= relax_factor          # 默认 1.3
    移除当前曲率约束
    添加放宽后的曲率约束 (更大的h̄' → 更大的C → 更宽松)
    重新求解
    if 求解成功: break
```

### 6.3 放宽的数学含义

```
h̄' 增大 → ρ_min = v_min · h̄' · sf 增大 → C = κ_max · ρ_min² 增大 → 约束更宽松
```

| 放宽次数 | h̄' 倍数 | C 倍数 | 约束宽松度 |
|---------|---------|--------|-----------|
| 0 | 1.0 | 1.0 | 基准 |
| 1 | 1.3 | 1.69 | +69% |
| 2 | 1.69 | 2.86 | +186% |
| 3 | 2.20 | 4.83 | +383% |

> 注意: C 与 h̄' 是**二次关系**，每次放宽 C 增长为 relax_factor² 倍。

### 6.4 权衡

放宽后约束更宽松，曲率可能违反，但至少能得到一个可行轨迹。后续通过后验评估量化违反程度。

---

## 7. 动态 safety_factor 收紧

### 7.1 触发条件

在 iter ≥ 2 且 h̄' **显著下降** 时触发：

```
h_new < h_prev × (1 - convergence_threshold)
```

默认阈值: h_new < h_prev × 0.85，即 h̄' 下降超过 15%。

### 7.2 收紧操作

```
dynamic_safety_factor *= safety_factor_decay   # 默认 0.8
```

### 7.3 收紧的数学含义

h̄' 显著下降意味着轨迹的时间缩放变慢（速度降低），这可能导致：

```
h̄' 下降 → ρ_min = v_min × h̄' × sf 下降 → C 下降 → 约束变严
```

但 h̄' 下降本身说明上一次约束可能**不够保守**（轨迹被过度约束后速度降低），因此需要收紧 safety_factor 来补偿：

```
sf 收紧 (0.7 → 0.56) → effective h̄' = h̄' × sf 更小 → ρ_min 更小 → C 更小
```

**直觉**: h̄' 下降是"约束过紧"的信号，但同时也说明"实际 h̄' 比估计的小"，所以用更小的 safety_factor 来使估计更接近实际。

### 7.4 示例

```
迭代1: h̄' = 2.0, sf = 0.7, effective = 1.4
迭代2: h̄' = 1.5 (下降25% > 15%)
  → sf 收紧: 0.7 × 0.8 = 0.56
  → effective = 1.5 × 0.56 = 0.84
迭代3: h̄' = 1.3 (下降13% < 15%, 不触发)
  → effective = 1.3 × 0.56 = 0.728
```

### 7.5 发出警告

收紧时发出 `UserWarning`:

```
h̄' decreased significantly: 2.000000 -> 1.500000.
Tightening safety_factor: 0.700000 -> 0.560000 (decay=0.8).
```

---

## 8. 候选轨迹选择策略

### 8.1 问题

GCS 舍入(Rounding) 可能产生多条候选轨迹。默认选择成本最低的，但**成本最低的轨迹 h̄' 不一定最大**。

### 8.2 策略：选择 h̄' 最大的候选

```python
def _select_trajectory_with_max_h_bar_prime(trajectory, result, BezierGCS, verbose):
    """在候选轨迹中选择 h̄' 最大的（约束最保守）。"""
    all_candidates = results_dict.get("all_candidate_trajectories", [])
    best_h = -inf
    best_traj = trajectory
    for cand_traj in all_candidates:
        cand_h = BezierGCS.compute_h_bar_prime_from_trajectory(cand_traj)
        if cand_h > best_h:
            best_h = cand_h
            best_traj = cand_traj
    return best_traj
```

### 8.3 理由

h̄' 大意味着轨迹的"速度水平"高，对应的 ρ_min 大，曲率约束阈值 C 大。

选择 h̄' 最大的候选轨迹：
- 提供更保守的 h̄' 估计（不容易低估）
- 使后续迭代从更安全的起点开始
- 降低因 h̄' 过低导致无可行解的风险

---

## 9. 约束移除与重建

### 9.1 问题

迭代2+需要先移除旧曲率约束再添加新的。但 **Drake Python binding 不提供 `edge.RemoveConstraint()` 方法**。

### 9.2 解决方案：边重建

`removeCurvatureHardConstraints()` 采用**边重建方案**：

1. 收集所有受曲率约束影响的**普通边**（非 source/target 边）
2. 移除这些边
3. 重建这些边，重新添加连续性约束、速度约束和成本

### 9.3 跳过 source/target 边

source/target 边上的曲率约束数量极少（仅 order-1 个），对性能影响可忽略。且重建这些边需要重新添加 `addSourceTargetWithHeading()` 中的大量专用约束（航向角约束、方向约束等），代价过高，因此跳过。

---

## 10. 两种使用模式

### 10.1 迭代修正模式

**触发条件**: `h_bar_prime is None` 且 `max_h_bar_prime_iterations > 1`

**流程**:
1. 先添加成本函数（迭代修正需要）
2. 调用 `iterate_h_bar_prime()`
3. 返回迭代修正后的轨迹和结果

**优点**: 自动确定 h̄'，无需手动调参
**缺点**: 额外求解开销（至少多求解1次无约束问题）

### 10.2 直接模式

**触发条件**: `h_bar_prime` 指定具体值，或 `max_h_bar_prime_iterations = 1`

**流程**:
1. 直接调用 `addCurvatureHardConstraint(h_bar_prime=指定值或1.0)`
2. 继续正常的求解流程

**优点**: 无额外迭代开销
**缺点**: h̄' 值可能不准确，约束过紧或过松

### 10.3 模式选择建议

| 场景 | 推荐模式 | h_bar_prime | max_iterations |
|------|---------|-------------|----------------|
| 首次求解/不确定 | 迭代修正 | None | 3 |
| 已知大致范围 | 直接 | 估计值 | 1 |
| 追求速度 | 直接 | 1.0 | 1 |
| 追求精确 | 迭代修正 | None | 5 |

---

## 11. 参数配置指南

### 11.1 完整参数表

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `h_bar_prime` | None | None 或 >0 | h̄' 均值估计。None=迭代修正 |
| `h_bar_prime_safety_factor` | 0.7 | (0, 1.0] | 保守修正因子 |
| `max_h_bar_prime_iterations` | 3 | ≥1 | 最大迭代次数。1=禁用迭代 |
| `h_bar_prime_convergence_threshold` | 0.15 | (0, 1.0) | 收敛相对变化阈值 |
| `h_bar_prime_relax_factor` | 1.3 | >1 | 求解失败放宽因子 |
| `max_h_bar_prime_relax_attempts` | 3 | ≥1 | 放宽重试最大次数 |
| `h_bar_prime_safety_factor_decay` | 0.8 | (0, 1.0] | 动态收紧衰减因子。1.0=禁用 |

### 11.2 参数调优建议

**safety_factor (0.7)**:
- 越小越保守（约束更严），但可能无可行解
- 越大越宽松（约束更松），但曲率可能违反
- 推荐: 0.5-0.8

**max_iterations (3)**:
- 3次通常足够收敛
- 复杂场景可设为5
- 设为1禁用迭代（使用直接模式）

**convergence_threshold (0.15)**:
- 0.15 = 允许15%的相对变化
- 追求精确可降到0.05
- 追求速度可升到0.3

**relax_factor (1.3)**:
- 每次放宽使 h̄' 增大30%，C增大69%
- 过大(如2.0)可能导致约束过松
- 过小(如1.1)可能需要更多次重试

**safety_factor_decay (0.8)**:
- 0.8 = 每次收紧20%
- 1.0 = 禁用动态收紧
- 更小(如0.5)收紧更激进

---

## 12. 完整伪代码

```
function iterate_h_bar_prime(bezier_gcs, constraints, cost_weights, ...):
    
    # 初始化
    max_iter ← constraints.max_h_bar_prime_iterations       # 默认3
    threshold ← constraints.h_bar_prime_convergence_threshold # 默认0.15
    sf ← constraints.h_bar_prime_safety_factor               # 默认0.7
    decay ← constraints.h_bar_prime_safety_factor_decay      # 默认0.8
    relax ← constraints.h_bar_prime_relax_factor             # 默认1.3
    max_relax ← constraints.max_h_bar_prime_relax_attempts   # 默认3
    
    h_bar_prime ← 1.0
    dynamic_sf ← sf
    history ← []
    best_traj ← None
    converged ← False
    reason ← "max_iterations"
    total_relax ← 0
    
    for iter = 1 to max_iter:
        
        # ─── 约束管理 ───
        if iter == 1:
            # 不添加曲率约束
            pass
        else:
            # 移除旧约束
            if bezier_gcs.curvature_constraints:
                bezier_gcs.removeCurvatureHardConstraints()
            
            # 添加新约束
            effective ← h_bar_prime × dynamic_sf
            try:
                bezier_gcs.addCurvatureHardConstraint(
                    κ_max, v_min, h_bar_prime, dynamic_sf)
            except ValueError:
                reason ← "constraint_failed"
                break
        
        # ─── 求解 ───
        result ← bezier_gcs.SolvePathWithConstraints(rounding=True)
        trajectory ← result[0]
        
        # ─── 候选选择 ───
        trajectory ← select_trajectory_with_max_h_bar_prime(
            trajectory, result)
        
        # ─── 求解失败处理 ───
        if trajectory is None and iter > 1:
            relax_ok ← False
            for r = 1 to max_relax:
                h_bar_prime *= relax
                total_relax += 1
                bezier_gcs.removeCurvatureHardConstraints()
                bezier_gcs.addCurvatureHardConstraint(...)
                result ← bezier_gcs.SolvePathWithConstraints(...)
                trajectory ← select_max_h_bar_prime(result)
                if trajectory is not None:
                    relax_ok ← True
                    break
            if not relax_ok:
                reason ← "solve_failed"
                break
        
        elif trajectory is None and iter == 1:
            reason ← "solve_failed"
            break
        
        # ─── 计算新 h̄' ───
        try:
            h_new ← compute_h_bar_prime_from_trajectory(trajectory)
        except:
            reason ← "compute_failed"
            break
        
        history.append(h_new)
        best_traj ← trajectory
        
        # ─── 收敛判定 (iter ≥ 2) ───
        if iter ≥ 2:
            h_prev ← history[-2]
            rel_change ← |h_new - h_prev| / h_prev
            
            if rel_change < threshold:
                converged ← True
                reason ← "converged"
                break
            
            # ─── 单调性检查 ───
            if h_new < h_prev × (1 - threshold):
                dynamic_sf *= decay
                warn("h̄' decreased, tightening safety_factor")
        
        h_bar_prime ← h_new
    
    # ─── 构造结果 ───
    final_h ← history[-1] if history else h_bar_prime
    effective_h ← final_h × dynamic_sf
    
    return best_traj, HBarPrimeIterationResult(
        h_bar_prime=final_h,
        effective_h_bar_prime=effective_h,
        converged=converged,
        num_iterations=len(history),
        iteration_history=history,
        convergence_reason=reason,
        relax_attempts=total_relax,
        final_safety_factor=dynamic_sf
    )
```

---

## 13. 与规划器的集成

### 13.1 调用位置

在 `AckermannGCSPlanner.plan_trajectory()` 中，步骤4.5（`ackermann_gcs_planner.py:172-270`）：

```python
# 判断是否使用迭代修正模式
use_iteration = (
    constraints.h_bar_prime is None
    and constraints.max_h_bar_prime_iterations > 1
)

if use_iteration:
    # 迭代修正模式
    # 1. 先添加成本函数（迭代修正需要）
    bezier_gcs.addTimeCost(...)
    bezier_gcs.addPathLengthCost(...)
    bezier_gcs.addPathEnergyCost(...)
    bezier_gcs.addDerivativeRegularization(...)
    
    # 2. 执行迭代修正
    iter_traj, h_bar_prime_iteration_result = iterate_h_bar_prime(
        bezier_gcs, constraints, cost_weights, ...)
    
else:
    # 直接模式
    bezier_gcs.addCurvatureHardConstraint(
        max_curvature, min_velocity, h_bar_prime, safety_factor)
```

### 13.2 迭代修正模式的成本添加顺序

在迭代修正模式下，成本函数在曲率约束**之前**添加（步骤5→步骤4.5），因为：

1. 迭代1需要成本函数来求解无曲率约束的GCS
2. 迭代2+需要成本函数来求解带曲率约束的GCS
3. `iterate_h_bar_prime()` 内部直接使用已配置好的 `bezier_gcs` 实例

### 13.3 迭代完成后的处理

```python
if iter_traj is not None:
    # 迭代修正成功
    rho_min = constraints.min_velocity * result.effective_h_bar_prime
    C = constraints.max_curvature * rho_min ** 2
    print(f"✓ h̄'={result.h_bar_prime:.6f}, effective={result.effective_h_bar_prime:.6f}, C={C:.6f}")
else:
    # 迭代修正失败，跳过曲率硬约束
    print("⚠️  迭代修正求解失败，将跳过曲率硬约束")
```

---

## 14. 设计决策与权衡

### 14.1 为什么选择 Lorentz 锥约束而非 SCP？

| 方面 | Lorentz锥 (当前) | SCP (旧方案) |
|------|-----------------|-------------|
| 凸性 | ✓ 始终凸 | ✗ 需迭代线性化 |
| 全局最优 | ✓ 凸松弛后全局最优 | ✗ 可能局部最优 |
| 收敛性 | ✓ h̄'迭代有负反馈 | ✗ 依赖初始点 |
| 保守性 | 保守(Cauchy-Schwarz) | 精确但可能发散 |
| 求解次数 | 少(2-3次) | 多(10-50次) |

### 14.2 为什么 h̄' 用均值而非逐点？

逐点使用 h'(s) 需要对每个控制点添加不同的约束阈值，导致约束数量大幅增加。使用均值 h̄' 将所有控制点统一为同一阈值 C，保持约束的简洁性和求解效率。

### 14.3 为什么放宽而非缩小信任区域？

SCP方法使用信任区域限制变量变化，而h̄'迭代使用放宽因子。原因：

1. h̄'迭代不是对非凸约束的线性化，而是对参数估计的修正
2. 放宽直接作用于阈值C，物理含义清晰
3. 放宽后的约束仍是凸约束，保持凸性

### 14.4 为什么选择 h̄' 最大的候选？

成本最低的轨迹可能速度偏低→h̄'偏小→约束过紧→后续迭代可能无可行解。选择h̄'最大的候选确保从"最安全"的起点继续迭代。

### 14.5 安全因子的双重角色

safety_factor 同时扮演两个角色：
1. **保守修正**: 防止h̄'过估导致约束过松 (初始0.7 = 30%保守余量)
2. **动态适应**: 通过decay因子响应h̄'的变化趋势

---

## 附录: 典型迭代过程示例

### 示例1: 正常收敛 (3次迭代)

```
[h̄' Iteration] Iteration 1/3
[h̄' Iteration] Solving without curvature constraint...
[h̄' Iteration] Computed h̄' = 2.345678

[h̄' Iteration] Iteration 2/3
[h̄' Iteration] Adding curvature constraint: h̄'=2.345678, safety_factor=0.700000, effective=1.641975
[h̄' Iteration] Computed h̄' = 2.189234
[h̄' Iteration] Relative change: 0.066733 (threshold: 0.15)

[h̄' Iteration] Iteration 3/3
[h̄' Iteration] Removing previous curvature constraints...
[h̄' Iteration] Adding curvature constraint: h̄'=2.189234, safety_factor=0.700000, effective=1.532464
[h̄' Iteration] Computed h̄' = 2.154321
[h̄' Iteration] Relative change: 0.016033 (threshold: 0.15)
[h̄' Iteration] Converged!

[h̄' Iteration] Final result:
  h̄' = 2.154321
  effective h̄' = 1.508025
  final safety_factor = 0.700000
  converged = True
  iterations = 3
  reason = converged
```

### 示例2: 含放宽重试

```
[h̄' Iteration] Iteration 1/3
[h̄' Iteration] Solving without curvature constraint...
[h̄' Iteration] Computed h̄' = 0.856234

[h̄' Iteration] Iteration 2/3
[h̄' Iteration] Adding curvature constraint: h̄'=0.856234, safety_factor=0.700000, effective=0.599364
[h̄' Iteration] Solve failed, attempting relax...
[h̄' Iteration] Relax attempt 1: h̄' *= 1.3 -> 1.113104
[h̄' Iteration] Computed h̄' = 0.923456
[h̄' Iteration] Relative change: 0.078527 (threshold: 0.15)

[h̄' Iteration] Iteration 3/3
[h̄' Iteration] Removing previous curvature constraints...
[h̄' Iteration] Adding curvature constraint: h̄'=0.923456, safety_factor=0.700000, effective=0.646419
[h̄' Iteration] Computed h̄' = 0.901234
[h̄' Iteration] Relative change: 0.024069 (threshold: 0.15)
[h̄' Iteration] Converged!

[h̄' Iteration] Final result:
  h̄' = 0.901234
  effective h̄' = 0.630864
  final safety_factor = 0.700000
  converged = True
  iterations = 3
  reason = converged
  relax attempts = 1
```

### 示例3: 含动态 safety_factor 收紧

```
[h̄' Iteration] Iteration 1/5
[h̄' Iteration] Solving without curvature constraint...
[h̄' Iteration] Computed h̄' = 3.200000

[h̄' Iteration] Iteration 2/5
[h̄' Iteration] Adding curvature constraint: h̄'=3.200000, safety_factor=0.700000, effective=2.240000
[h̄' Iteration] Computed h̄' = 2.100000
[h̄' Iteration] Relative change: 0.343750 (threshold: 0.15)
  UserWarning: h̄' decreased significantly: 3.200000 -> 2.100000.
  Tightening safety_factor: 0.700000 -> 0.560000 (decay=0.8).

[h̄' Iteration] Iteration 3/5
[h̄' Iteration] Removing previous curvature constraints...
[h̄' Iteration] Adding curvature constraint: h̄'=2.100000, safety_factor=0.560000, effective=1.176000
[h̄' Iteration] Computed h̄' = 1.900000
[h̄' Iteration] Relative change: 0.095238 (threshold: 0.15)
[h̄' Iteration] Converged!

[h̄' Iteration] Final result:
  h̄' = 1.900000
  effective h̄' = 1.064000
  final safety_factor = 0.560000
  (tightened from initial 0.700000)
  converged = True
  iterations = 3
  reason = converged
```
