# AckermannGCS规划器约束与成本设计详解

> **文档版本**: v2.0  
> **更新日期**: 2026-05-13  
> **适用范围**: 阿克曼转向车辆轨迹规划  
> **变更说明**: 基于代码实现全面更新，修正曲率约束设计(Lorentz锥硬约束+h̄'迭代)、成本设计(移除不存在的曲率惩罚成本，添加时间导数正则化)、新增数据结构与求解配置章节

---

## 目录

1. [概述](#1-概述)
2. [数据结构](#2-数据结构)
3. [约束设计](#3-约束设计)
   - 3.1 [航向角约束](#31-航向角约束)
   - 3.2 [速度约束](#32-速度约束)
   - 3.3 [曲率约束](#33-曲率约束)
   - 3.4 [工作空间约束](#34-工作空间约束)
   - 3.5 [连续性约束](#35-连续性约束)
   - 3.6 [时间缩放约束](#36-时间缩放约束)
4. [成本设计](#4-成本设计)
   - 4.1 [时间成本](#41-时间成本)
   - 4.2 [路径长度成本](#42-路径长度成本)
   - 4.3 [能量成本](#43-能量成本)
   - 4.4 [时间导数正则化成本](#44-时间导数正则化成本)
   - 4.5 [导数正则化成本](#45-导数正则化成本)
5. [成本配置器与预设模板](#5-成本配置器与预设模板)
6. [约束违反检查](#6-约束违反检查)
7. [约束与成本的协同设计](#7-约束与成本的协同设计)
8. [求解流程](#8-求解流程)
9. [求解器配置](#9-求解器配置)
10. [关键设计特点](#10-关键设计特点)
11. [参数配置指南](#11-参数配置指南)
12. [性能优化建议](#12-性能优化建议)

---

## 1. 概述

### 1.1 规划器简介

AckermannGCS规划器是一个基于图凸集(Graph of Convex Sets, GCS)方法的阿克曼转向车辆轨迹规划器。该规划器通过以下核心技术实现高质量轨迹规划:

- **凸优化基础**: 使用GCS方法处理凸约束,保证全局最优性
- **曲率硬约束**: 使用Lorentz锥凸约束保证曲率约束满足,支持h̄'迭代修正
- **贝塞尔曲线**: 使用B样条/贝塞尔曲线表示轨迹,保证平滑性
- **数值稳定性**: 采用旋转矩阵法等数值稳定技术
- **平坦输出映射**: 将位置轨迹(x,y)映射到完整车辆状态空间

### 1.2 核心优势

| 优势 | 说明 |
|------|------|
| **凸性保证** | 航向角、速度、工作空间、曲率硬约束均保持凸性 |
| **全局最优** | GCS方法可在凸松弛后找到全局最优解 |
| **数值稳定** | 旋转矩阵法避免奇异点,h̄'迭代保证约束可靠性 |
| **物理准确** | 标量速度约束更符合阿克曼车辆模型 |
| **可扩展性** | 模块化设计,13种预设成本模板,易于添加新约束和成本 |

### 1.3 核心数据流

```
A*搜索(SE2) → 局部走廊 → IRIS区域分解 → HPolyhedron区域列表
    ↓
AckermannGCSPlanner.plan_trajectory()
    ├─ 1. 构建TrajectoryConstraints (速度/加速度/曲率限制)
    ├─ 2. 初始化AckermannBezierGCS (继承BezierGCS)
    ├─ 3. addSourceTargetWithHeading() (航向角约束: 旋转矩阵法)
    ├─ 4. addScalarVelocityLimit() (SOCP: ||v||_2 <= v_max)
    ├─ 5. addCurvatureHardConstraint() (Lorentz锥: ||Q_j||_2 <= C)
    │     └─ 可选: iterate_h_bar_prime() 迭代修正
    ├─ 6. 添加成本函数 (时间+路径+能量+正则化)
    ├─ 7. SolvePathWithConstraints() (MOSEK求解+Rounding)
    │     └─ 多次舍入尝试 → 选择综合违反量最小的轨迹
    └─ 8. TrajectoryEvaluator.evaluate_trajectory()
          └─ 速度/加速度/曲率/工作空间约束验证 → TrajectoryReport
```

### 1.4 适用场景

- 自动驾驶车辆轨迹规划
- 月面漫游车轨迹规划
- 移动机器人路径规划
- 需要考虑运动学约束的规划问题

---

## 2. 数据结构

### 2.1 VehicleParams

车辆参数数据类，包含物理约束和推导量:

| 属性 | 类型 | 说明 |
|------|------|------|
| `wheelbase` | float | 车辆轴距（米） |
| `max_steering_angle` | float | 最大转向角（弧度） |
| `max_velocity` | float | 最大速度（m/s） |
| `max_acceleration` | float | 最大加速度（m/s²） |
| `max_curvature` | float | 最大曲率（1/m），自动计算: κ_max = tan(δ_max) / L |
| `min_turning_radius` | float | 最小转弯半径（m），自动计算: R_min = L / tan(δ_max) |

**自动推导**: `__post_init__` 中自动计算 `max_curvature` 和 `min_turning_radius`，并验证 `max_steering_angle` 不接近π/2（避免tan发散）。

### 2.2 EndpointState

起终点状态:

| 属性 | 类型 | 说明 |
|------|------|------|
| `position` | np.ndarray(2,) | 位置坐标（米） |
| `heading` | float | 航向角（弧度），范围[-π, π] |
| `velocity` | Optional[float] | 速度（m/s），可选 |

### 2.3 TrajectoryConstraints

轨迹约束总控，是约束系统的核心数据结构:

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_velocity` | float | - | 最大速度（m/s） |
| `max_acceleration` | float | - | 最大加速度（m/s²） |
| `max_curvature` | float | - | 最大曲率（1/m） |
| `workspace_regions` | Optional[List] | None | 工作空间区域（HPolyhedron列表） |
| `enable_curvature_hard_constraint` | bool | False | 是否启用曲率硬约束 |
| `min_velocity` | float | 1.58 | 最小速度，用于曲率硬约束计算ρ_min |
| `curvature_constraint_mode` | str | "none" | 曲率约束模式: "none"/"hard"/"turning_radius" |
| `h_bar_prime` | Optional[float] | None | h̄'均值估计，None表示使用迭代修正 |
| `h_bar_prime_safety_factor` | float | 0.7 | 保守修正因子，范围(0, 1.0] |
| `max_h_bar_prime_iterations` | int | 3 | h̄'迭代修正最大次数，1=禁用迭代 |
| `h_bar_prime_convergence_threshold` | float | 0.15 | 迭代收敛判定阈值（相对变化15%） |
| `h_bar_prime_relax_factor` | float | 1.3 | 求解失败时h̄'放宽因子 |
| `max_h_bar_prime_relax_attempts` | int | 3 | 求解失败放宽重试最大次数 |
| `h_bar_prime_safety_factor_decay` | float | 0.8 | h̄'显著下降时safety_factor动态衰减因子 |

**最小速度推导**: `compute_min_velocity_from_weights(w_time, w_energy, safety_factor=0.5)`
- 最优速度: `v_optimal = sqrt(w_time / w_energy)`
- 最小速度: `min_velocity = v_optimal * safety_factor`
- 默认值1.58 m/s 对应 w_time=1.0, w_energy=0.1

### 2.4 CurvatureConstraintMode

曲率约束模式枚举:

| 模式 | 值 | 说明 |
|------|-----|------|
| `NONE` | "none" | 无曲率硬约束（仅评估检查） |
| `HARD` | "hard" | 凸硬约束（Lorentz锥，保守但可靠） |
| `TURNING_RADIUS` | "turning_radius" | 旧转弯半径约束（盒约束，已弃用） |

### 2.5 BezierConfig

贝塞尔曲线配置:

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `order` | int | 5 | 贝塞尔曲线阶数 |
| `continuity` | int | 1 | 连续性阶数 |
| `hdot_min` | float | 0.01 | 时间导数最小值 |
| `full_dim_overlap` | bool | False | 是否使用全维重叠 |
| `hyperellipsoid_num_samples_per_dim_factor` | int | 32 | 超椭球采样因子 |
| `max_rounding_attempts` | int | 3 | 舍入验证重试次数 |
| `max_rounded_paths` | int | 5 | 每次舍入尝试的路径数 |

### 2.6 评估报告数据结构

**ConstraintViolation**: 约束违反报告
- `constraint_name`: 约束名称
- `is_violated`: 是否违反
- `max_violation`: 最大违反量
- `violation_points`: 违反点列表

**TrajectoryReport**: 轨迹评估报告
- `is_feasible`: 轨迹是否可行
- `velocity_violation`, `acceleration_violation`, `curvature_violation`, `workspace_violation`: 各约束违反报告
- `c0_continuity`, `c1_continuity`, `c2_continuity`: 连续性报告
- `curvature_stats`: 曲率统计信息（CurvatureStats）
- `improvement_metrics`: 平滑度改善指标（ImprovementMetrics）

**PlanningResult**: 规划结果
- `success`: 规划是否成功
- `trajectory`: 轨迹对象
- `trajectory_report`: 轨迹评估报告
- `solve_time`: 求解时间（秒）
- `num_iterations`: 求解尝试次数
- `convergence_reason`: 收敛原因
- `error_message`: 错误消息

---

## 3. 约束设计

约束设计是轨迹规划的核心,决定了轨迹的可行性和物理合理性。AckermannGCS规划器实现了多种约束,**所有约束均保持凸性**。

### 3.1 航向角约束

航向角约束是阿克曼转向车辆特有的约束,用于保证轨迹在起点和终点的朝向符合要求。

#### 3.1.1 约束方法对比

规划器支持三种航向角约束方法:

| 方法 | 数学形式 | 凸性 | 数值稳定性 | 适用场景 | 推荐度 |
|------|---------|------|-----------|---------|--------|
| **旋转矩阵法** | (P₁-P₀)×d_θ = 0 | ✓ 凸约束 | ✓ 无奇异点 | 所有角度 | ⭐⭐⭐⭐⭐ |
| **方向约束** | (P₁-P₀)·d_θ ≥ ε | ✓ 凸约束 | ✓ 稳定 | 确保朝向一致 | ⭐⭐⭐⭐ |
| **线性化方法** | Δy = tan(θ)·Δx | ✓ 凸约束 | ✗ θ≈±π/2时奇异 | 非垂直方向 | ⭐⭐ |

#### 3.1.2 旋转矩阵法详解

**数学原理**

旋转矩阵法使用叉积形式表示航向角约束:

```
约束形式: ṗ × R(θ)·d_ref = 0
展开形式: (P₁[x] - P₀[x])·sin(θ) - (P₁[y] - P₀[y])·cos(θ) = 0
```

其中:
- P₀, P₁: 相邻控制点
- θ: 期望航向角
- R(θ): 旋转矩阵
- d_ref: 参考方向向量

**凸性分析**: 该约束是线性等式约束(A·x = b)，属于仿射约束，是凸集，可以嵌入GCS算法。

**数值稳定性**: 使用sin和cos函数，避免tan函数在θ ≈ ±π/2时的奇异问题。

| 角度 | tan(θ) | sin(θ) | cos(θ) | 稳定性 |
|------|--------|--------|--------|--------|
| 0° | 0 | 0 | 1 | ✓ |
| 45° | 1 | 0.707 | 0.707 | ✓ |
| 90° | ∞ (奇异) | 1 | 0 | ✓ |
| 180° | 0 | 0 | -1 | ✓ |
| 270° | -∞ (奇异) | -1 | 0 | ✓ |

**实现细节**:

1. 计算旋转矩阵系数: `cos_θ = cos(θ)`, `sin_θ = sin(θ)`
2. 构建约束表达式: `expr = (P₁[x] - P₀[x])·sin_θ - (P₁[y] - P₀[y])·cos_θ`
3. 分解线性表达式: `A, b = DecomposeLinearExpressions(expr, variables)`
4. 创建线性等式约束: `constraint = LinearEqualityConstraint(A, b)`

#### 3.1.3 方向约束详解

**数学原理**

方向约束使用点积形式确保控制点差向量与期望方向向量的朝向一致:

```
约束形式: (P₁ - P₀)·d_θ ≥ ε
展开形式: (P₁[x] - P₀[x])·cos(θ) + (P₁[y] - P₀[y])·sin(θ) ≥ ε
```

其中 d_θ = [cos(θ), sin(θ)] 为期望方向向量，ε 为最小点积值。

**与叉积约束的结合**:

| 约束类型 | 约束形式 | 作用 |
|---------|---------|------|
| 叉积约束 | (P₁-P₀)×d_θ = 0 | 保证方向共线(平行) |
| 点积约束 | (P₁-P₀)·d_θ ≥ ε | 保证方向朝向一致(同向) |
| **结合** | 两者同时满足 | 保证方向完全一致 |

**凸性**: 点积约束是线性不等式约束(A·x ≥ b)，定义的半空间是凸集。

#### 3.1.4 多控制点约束

仅约束前两个控制点(P₀, P₁)可能不足以保证轨迹质量。多控制点约束通过约束前k个控制点对来提高轨迹质量:

```
约束: (Pᵢ - Pᵢ₋₁) × d_θ = 0,  i = 1, 2, ..., k
```

| 贝塞尔阶数 | 推荐控制点数 | 说明 |
|-----------|-------------|------|
| 3 (三次) | 2 | 约束P₀-P₁, P₁-P₂ |
| 5 (五次) | 3 | 约束P₀-P₁, P₁-P₂, P₂-P₃ |
| 7 (七次) | 4 | 约束P₀-P₁, P₁-P₂, P₂-P₃, P₃-P₄ |

**v=0退化处理**: 当起/终点速度为0时，逐对选择性禁用第一对点积约束。

#### 3.1.5 线性化方法(传统，已弃用)

```
一般情况: (P₁[y] - P₀[y]) = tan(θ)·(P₁[x] - P₀[x])
特殊情况(θ ≈ ±π/2): P₁[x] = P₀[x]
```

问题: tan(θ)在θ ≈ ±π/2时趋向无穷，导致数值不稳定。仅适用于航向角远离±π/2的情况。

---

### 3.2 速度约束

速度约束确保轨迹的速度在物理允许范围内。

#### 3.2.1 两种实现方式对比

| 类型 | 约束形式 | 几何意义 | 凸性 | 可行域 | 推荐度 |
|------|---------|---------|------|--------|--------|
| **矢量约束** | v_lower ≤ dr/dt ≤ v_upper | 矩形区域 | ✓ 线性约束 | 小 | ⭐⭐ (已弃用) |
| **标量约束** | ‖v‖₂ ≤ v_max | 圆形区域 | ✓ 二阶锥约束 | 大27% | ⭐⭐⭐⭐⭐ |

#### 3.2.2 标量约束详解

**数学推导**

速度的定义:

```
v = dr/dt = dr/ds · ds/dt = r'(s) / h'(s)
```

标量速度约束:

```
‖v‖₂ ≤ v_max
代入: ‖r'(s)‖₂ ≤ v_max · h'(s)
```

**二阶锥规划(SOCP)形式**

```
‖A_ctrl·x‖₂ ≤ v_max · (b_ctrl·x)
```

其中:
- A_ctrl: 空间导数r'(s)的系数矩阵
- b_ctrl: 时间导数h'(s)的系数矩阵
- x: 决策变量(控制点)

**Lorentz锥约束**

构造变量z:

```
z = H·x = [v_max · b_ctrl·x]
           [    A_ctrl·x    ]
```

约束z ∈ Lorentz锥等价于: `v_max · b_ctrl·x ≥ ‖A_ctrl·x‖₂`

**实现**: `bezier_gcs.addScalarVelocityLimit(max_velocity)`，对应代码 `ackermann_gcs_planner.py:167`

**优势**: 圆形区域准确表示速度限制，可行域比矢量约束大27%，更符合阿克曼车辆物理模型。

#### 3.2.3 性能对比

| 指标 | 矢量约束 | 标量约束 | 改进 |
|------|---------|---------|------|
| 可行域大小 | 1.0 | 1.27 | +27% |
| 求解时间 | 1.0 | 1.08 | +8% |
| 解的质量 | 保守 | 精确 | 显著提升 |
| 物理准确性 | 低 | 高 | 显著提升 |

---

### 3.3 曲率约束

曲率约束是阿克曼转向车辆的核心约束，限制了轨迹的最小转弯半径。

#### 3.3.1 曲率定义

**数学公式**

```
κ = (ẋ·ÿ - ẏ·ẍ) / (ẋ² + ẏ²)^(3/2)
```

**阿克曼车辆约束**

```
|κ| ≤ κ_max = tan(δ_max) / L
```

其中 L 为轴距，δ_max 为最大转向角。

#### 3.3.2 曲率硬约束（Lorentz锥）— 核心方法

**设计思想**

曲率约束 `|κ| ≤ κ_max` 本身是非凸的。规划器采用**Lorentz锥凸约束**作为保守但可靠的凸内近似，保证优化问题的凸性。

**约束形式**

```
||Q_j||_2 ≤ C = κ_max · ρ_min²
```

其中:
- Q_j: 贝塞尔曲线二阶导数控制点
- C: 约束阈值
- ρ_min = min_velocity · h̄' · safety_factor: 弧长参数下的最小速度下界

**凸性**: Lorentz锥约束 `{z | z[0] ≥ √(z[1]² + ... + z[n]²)}` 是凸集，保持优化问题的凸性。

**保守性分析**

保守性来源于三个层次的近似:

1. **Cauchy-Schwarz不等式**: ‖r''(s)‖ ≥ |ẋ·ÿ - ẏ·ẍ| / ‖r'(s)‖
2. **凸包性质**: 贝塞尔曲线的导数在凸包内，控制点约束是充分条件
3. **速度下界**: 使用 ρ_min ≤ ‖r'(s)‖ 得到保守阈值

**保守因子**: α = (v_max / v_min)²，当 min_velocity 越小，α 越大，约束越保守。

**实现**: `bezier_gcs.addCurvatureHardConstraint(max_curvature, min_velocity, h_bar_prime, h_bar_prime_safety_factor)`

#### 3.3.3 h̄' 迭代修正

**动机**

曲率硬约束的阈值 C 依赖于 h̄' = dh/ds 的均值估计。h̄' 未知时，使用默认值1.0会导致约束过紧（可能无可行解）或过松（曲率违反）。h̄' 迭代修正通过自动求解-计算-修正循环使约束逐步精确化。

**迭代流程**

```
输入: AckermannBezierGCS实例, TrajectoryConstraints, 成本权重
输出: 最优轨迹, HBarPrimeIterationResult

1. 迭代1: 求解无曲率约束的GCS → compute_h_bar_prime_from_trajectory
2. 迭代2+: 移除旧曲率约束 → 用新h̄'添加曲率约束 → 求解 → 计算新h̄'
3. 收敛判定: |h_curr - h_prev| / h_prev < threshold (默认0.15)
4. 求解失败: h̄' *= relax_factor (默认1.3) 重试
5. h̄'显著下降: safety_factor *= decay (默认0.8) 动态收紧
```

**HBarPrimeIterationResult 数据结构**:

| 属性 | 类型 | 说明 |
|------|------|------|
| `h_bar_prime` | float | 最终h̄'值 |
| `effective_h_bar_prime` | float | 经safety_factor修正后的值 |
| `converged` | bool | 是否收敛 |
| `num_iterations` | int | 实际迭代次数 |
| `iteration_history` | List[float] | 每次迭代的h̄'值 |
| `convergence_reason` | str | 收敛/终止原因 |
| `relax_attempts` | int | 放宽重试次数 |
| `final_safety_factor` | float | 最终使用的safety_factor |

**收敛判定**:
- 相对变化: `|h_new - h_prev| / h_prev < convergence_threshold`
- 收敛原因: "converged" / "max_iterations" / "solve_failed" / "constraint_failed" / "compute_failed"

**求解失败处理**: 当添加曲率约束后求解失败，通过 `h̄' *= relax_factor` 放宽约束，最多重试 `max_h_bar_prime_relax_attempts` 次。

**动态safety_factor收紧**: 当h̄'显著下降时（`h_new < h_prev * (1 - threshold)`），自动将 `safety_factor *= decay`，收紧约束以补偿h̄'下降带来的保守性不足。

**两种使用模式**:

| 模式 | 条件 | 说明 |
|------|------|------|
| **迭代修正模式** | h_bar_prime=None 且 max_h_bar_prime_iterations > 1 | 先无曲率约束求解，再迭代修正h̄' |
| **直接模式** | h_bar_prime指定 或 max_h_bar_prime_iterations = 1 | 使用指定h̄'或默认值1.0，直接添加约束 |

**实现**: `h_bar_prime_iteration.py:iterate_h_bar_prime()`，对应代码 `ackermann_gcs_planner.py:194-270`

#### 3.3.4 曲率约束模式

| 模式 | 值 | 约束类型 | 说明 |
|------|-----|---------|------|
| `NONE` | "none" | 无硬约束 | 仅后验评估检查曲率 |
| `HARD` | "hard" | Lorentz锥凸约束 | 保守但可靠，推荐 |
| `TURNING_RADIUS` | "turning_radius" | 盒约束 | 旧实现，已弃用 |

---

### 3.4 工作空间约束

工作空间约束确保轨迹始终在可行区域内,避免碰撞障碍物。

#### 3.4.1 约束形式

```
P_i ∈ Region,  i = 0, 1, ..., n
```

利用贝塞尔曲线的凸包性质: 如果所有控制点都在凸集C内,则整条曲线都在C内。

#### 3.4.2 凸集表示

| 凸集类型 | 表示形式 | 适用场景 |
|---------|---------|---------|
| HPolyhedron | A·x ≤ b | 多面体区域,最常用 |
| Hyperellipsoid | (x-c)ᵀAᵀA(x-c) ≤ 1 | 椭球区域 |
| Point | x = x₀ | 单点区域 |

#### 3.4.3 GCS顶点构建

每个GCS顶点对应一个区域,顶点的凸集为:

```
Vertex_Set = Region^(order+1) × TimeScalingSet
```

---

### 3.5 连续性约束

连续性约束确保轨迹在连接点处平滑。

#### 3.5.1 连续性阶数

| 阶数 | 名称 | 物理意义 | 约束内容 |
|------|------|---------|---------|
| C0 | 位置连续 | 轨迹不断裂 | 位置相等 |
| C1 | 速度连续 | 速度不突变 | 位置、速度相等 |
| C2 | 加速度连续 | 加速度不突变 | 位置、速度、加速度相等 |

#### 3.5.2 约束形式

```
d^k r_v/ds^k (s=0) = d^k r_u/ds^k (s=1),  k = 0, 1, ..., continuity
```

连续性约束是线性等式约束(A·x = b)，保持凸性。

---

### 3.6 时间缩放约束

时间缩放约束确保时间参数化单调递增。

#### 3.6.1 约束形式

```
dh/ds ≥ hdot_min > 0
```

实现为控制点约束: `h_{i+1} - h_i ≥ hdot_min`，是线性不等式约束。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| hdot_min | 0.01 | 时间导数最小值(物理合理默认值) |
| hdot_min警告阈值 | 0.001 | 低于此值可能引起数值不稳定 |

---

## 4. 成本设计

成本设计决定了轨迹的优化目标。AckermannGCS规划器提供5种成本项，可根据应用场景灵活配置。

### 4.1 时间成本

**数学形式**

```
J_time = w_time · T_total = w_time · (h(1) - h(0))
```

**成本类型**: 线性成本 `J = cᵀ·x + d`，凸成本。

**物理意义**: 最小化总时间，使车辆尽快到达目标。

**实现**: `bezier_gcs.addTimeCost(weight)`

---

### 4.2 路径长度成本

**数学形式**

```
J_length = w_length · ∫‖dr/ds‖ds
```

**成本类型**: L2范数成本 `J = ‖A·x + b‖₂`，凸成本。

**物理意义**: 最小化轨迹空间长度，避免绕路。

**实现**: `bezier_gcs.addPathLengthCost(weight)`

---

### 4.3 能量成本

**数学形式**

```
J_energy = w_energy · ∫‖r'(s)‖² / h'(s) ds
```

**成本类型**: 透视二次成本 `J = ‖A·x‖² / (b·x)`，凸成本。

**物理意义**: 最小化动能积分，减少速度波动，保持匀速运动。

**与路径长度成本的区别**: 路径长度成本优化‖v‖的一次方，能量成本优化‖v‖²，对速度变化更敏感。

**实现**: `bezier_gcs.addPathEnergyCost(weight)`

---

### 4.4 时间导数正则化成本

**数学形式**

```
J_time_reg = w_time_reg · ∫|h'(s) - h_ref|² ds
```

**物理意义**: 惩罚时间导数h'(s)偏离参考值h_ref，使速度分布更均匀。

**参数**:
- `weight`: 正则化权重
- `h_ref`: 参考时间导数值（可选，None表示自动计算）

**实现**: `bezier_gcs.addTimeDerivativeRegularization(weight, h_ref=h_ref)`

---

### 4.5 导数正则化成本

**数学形式**

```
J_reg = w_r·∫‖d^k r/ds^k‖²ds + w_h·∫‖d^k h/ds^k‖²ds
```

**成本类型**: 二次成本 `J = xᵀ·Q·x + cᵀ·x + d`（Q正半定），凸成本。

**物理意义**:

| 导数阶数 | 物理意义 | 作用 |
|---------|---------|------|
| k=2 | 加速度 | 减少加速度波动，提高舒适性 |
| k=3 | 加加速度 | 减少加加速度波动，提高超舒适性 |

**参数**:
- `weight_r`: 空间轨迹导数的正则化权重
- `weight_h`: 时间轨迹导数的正则化权重
- `order`: 导数阶数（默认2）

**实现**: `bezier_gcs.addDerivativeRegularization(weight_r, weight_h, order)`

**与曲率硬约束的协同**: `regularization_r` 限制 ‖r''(s)‖ 上界，与曲率硬约束形成双重保障——硬约束保证不超限，正则化成本引导轨迹远离约束边界。

---

## 5. 成本配置器与预设模板

### 5.1 CostWeights 数据类

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `time` | float | 1.0 | 时间成本权重 |
| `path_length` | float | 1.0 | 路径长度成本权重 |
| `energy` | float | 0.0 | 能量成本权重 |
| `regularization_r` | float | 0.0 | 空间导数正则化权重 |
| `regularization_h` | float | 0.0 | 时间导数正则化权重 |
| `regularization_order` | int | 2 | 正则化导数阶数 |

支持 `normalize()` 归一化、`to_dict()` / `from_dict()` 序列化。

### 5.2 CostConfigurator — 13种预设模板

| 预设名 | time | path_length | energy | reg_r | reg_h | 场景 |
|--------|------|-------------|--------|-------|-------|------|
| `time_optimal` | 10 | 0.5 | 0 | 0 | 0 | 时间优先 |
| `path_optimal` | 0.5 | 10 | 0 | 0 | 0 | 路径优先 |
| `energy_optimal` | 0.5 | 1 | 5 | 0 | 0 | 能量优先 |
| `balanced` | 1 | 1 | 0 | 0 | 0 | 平衡 |
| `smooth` | 1 | 1 | 2 | 0.5 | 0.5 | 平滑性优先 |
| `lunar_standard` | 1 | 1.5 | 20 | 0.3 | 0.3 | 月面标准 |
| `lunar_high_risk` | 0.5 | 2 | 3 | 1 | 1 | 月面高风险 |
| `lunar_emergency` | 10 | 0.5 | 0.5 | 0 | 0 | 月面紧急 |
| `lunar_complex` | 1.5 | 2 | 2.5 | 0.5 | 0.5 | 月面复杂地形 |
| `curvature_constrained` | 3 | 1.5 | 3 | 5 | 0.5 | 曲率硬约束场景 |
| `curvature_constrained_high_speed` | 5 | 1 | 2 | 8 | 0.5 | 曲率约束+高速 |
| `curvature_constrained_parking` | 2 | 2 | 4 | 10 | 0.5 | 曲率约束+泊车 |

**曲率约束场景预设说明**:
- `w_time/w_energy` 比值控制速度水平，影响保守因子α
- `regularization_r` 限制 ‖r''‖ 上界，协同曲率硬约束
- 高 `regularization_r` 值使轨迹远离曲率约束边界

### 5.3 量纲归一化

**特征值**:
- characteristic_time = 10s
- characteristic_length = 20m
- characteristic_velocity = 2m/s

**归一化**:
- time_scale = 1 / T_char
- length_scale = 1 / L_char
- velocity_scale = 1 / V_char²

**实现**: `CostConfigurator.get_normalized_weights()` 返回量纲归一化+L1归一化后的权重。

### 5.4 CostOptimizer — 自动调优

`CostOptimizer` 类提供基于目标性能指标的权重自动调优:
- `evaluate_weights(weights)`: 评估给定权重的性能（成功率、求解时间、轨迹时间、路径长度）
- `optimize(target_metrics, max_iterations)`: 迭代优化权重以达到目标性能指标

### 5.5 便捷函数

```python
get_lunar_standard_config()     # 月面标准配置
get_lunar_high_risk_config()    # 月面高风险配置
get_lunar_emergency_config()    # 月面紧急配置
get_lunar_complex_config()      # 月面复杂地形配置
```

---

## 6. 约束违反检查

### 6.1 核心函数

**`compute_constraint_violation(value, limit, use_absolute=True)`**

计算约束违反量:
- 绝对值模式(双边): `max(0, |value| - limit)` — 适用于加速度、曲率
- 非绝对值模式(单边): `max(0, value - limit)` — 适用于速度

**`identify_violation_points(violations, threshold, num_samples)`**

识别违反点，返回违反点归一化位置(0到1)和索引。

**`check_constraint_satisfaction(value, limit, threshold, use_absolute=True)`**

返回 `ViolationResult` 数据类: `is_violated`, `max_violation`, `violation_points`, `violation_indices`

### 6.2 违反阈值常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `VELOCITY_VIOLATION_THRESHOLD` | 1e-2 | 速度违反阈值 (m/s) |
| `ACCELERATION_VIOLATION_THRESHOLD` | 1e-4 | 加速度违反阈值 (m/s²) |
| `CURVATURE_VIOLATION_THRESHOLD` | 1e-4 | 曲率违反阈值 (1/m) |

### 6.3 综合违反量评估

在多次舍入求解中，使用加权综合违反量选择最优轨迹:

```
combined_violation = vel_viol × 10.0 + curv_viol + acc_viol × 5.0
```

速度违反权重最高(10.0)，因为速度约束是硬约束；曲率违反权重为1.0；加速度违反权重为5.0。

---

## 7. 约束与成本的协同设计

### 7.1 凸性保证

| 组件 | 凸性 | 处理方法 | 说明 |
|------|------|---------|------|
| 航向角约束 | ✓ 凸 | 旋转矩阵法 | 线性等式约束 |
| 速度约束 | ✓ 凸 | Lorentz锥约束 | 二阶锥约束 |
| 曲率硬约束 | ✓ 凸 | Lorentz锥约束 | 保守凸内近似 |
| 工作空间约束 | ✓ 凸 | HPolyhedron | 多面体约束 |
| 连续性约束 | ✓ 凸 | 线性等式约束 | 仿射约束 |
| 时间缩放约束 | ✓ 凸 | 线性不等式约束 | 半空间约束 |
| 时间成本 | ✓ 凸 | 线性成本 | 仿射成本 |
| 路径长度成本 | ✓ 凸 | L2范数成本 | 范数成本 |
| 能量成本 | ✓ 凸 | 透视二次成本 | 透视成本 |
| 导数正则化成本 | ✓ 凸 | 二次成本 | 二次形式 |

### 7.2 约束层次

| 层次 | 约束类型 | 处理方式 | 说明 |
|------|---------|---------|------|
| **硬约束** | 工作空间、连续性、时间缩放 | 必须满足 | 物理可行性 |
| **凸硬约束** | 速度(SOCP)、曲率(Lorentz锥) | 凸约束嵌入GCS | 物理约束 |
| **航向角约束** | 叉积+点积 | 仿射等式+不等式 | 起终点朝向 |
| **软引导** | 导数正则化 | 成本项 | 远离约束边界 |

### 7.3 曲率约束与正则化的协同

曲率硬约束和 `regularization_r` 形成双重保障:

1. **硬约束**: 保证 ‖r''(s)‖ 不超过阈值 → 曲率不超限（保守）
2. **正则化成本**: 惩罚大的 ‖r''(s)‖² → 引导轨迹远离约束边界
3. **协同效果**: 硬约束兜底，正则化优化，两者互补

**配置建议**: 启用曲率硬约束时，设置 `regularization_r ≥ 5.0`（如 `curvature_constrained` 预设）。

### 7.4 成本权重与最小速度的协同

成本权重通过影响最优速度间接影响曲率硬约束的保守性:

```
v_optimal = sqrt(w_time / w_energy)
min_velocity = v_optimal * safety_factor
ρ_min = min_velocity * h̄' * safety_factor
C = κ_max * ρ_min²
```

- `w_time/w_energy` 比值大 → v_optimal大 → min_velocity大 → C大 → 约束宽松 → 更容易求解但保守性降低
- `w_time/w_energy` 比值小 → v_optimal小 → min_velocity小 → C小 → 约束严格 → 更难求解但精确性高

---

## 8. 求解流程

### 8.1 整体流程

```
输入: 起点、终点、工作空间、车辆参数、约束、成本权重
输出: PlanningResult

1. 构建TrajectoryConstraints（如果未提供）
   └── 从成本权重推导min_velocity

2. 初始化AckermannBezierGCS
   └── 构建GCS图 + 工作空间约束 + 连续性约束 + 时间缩放约束

3. 添加起终点约束
   └── addSourceTargetWithHeading() [航向角: 旋转矩阵法 + 方向约束]

4. 添加速度约束
   └── addScalarVelocityLimit() [SOCP: ||v||_2 <= v_max]

5. 添加曲率硬约束（可选）
   ├── 迭代修正模式: iterate_h_bar_prime()
   │   ├── 迭代1: 无曲率约束求解 → 计算h̄'
   │   └── 迭代2+: 添加约束 → 求解 → 计算h̄' → 收敛判定
   └── 直接模式: addCurvatureHardConstraint() [Lorentz锥]

6. 添加成本函数
   ├── addTimeCost()
   ├── addPathLengthCost()
   ├── addPathEnergyCost()
   ├── addTimeDerivativeRegularization()
   └── addDerivativeRegularization()

7. GCS求解（多次舍入尝试）
   ├── 设置舍入策略: 随机前向+随机后向
   ├── for attempt in range(max_rounding_attempts):
   │   ├── SolvePathWithConstraints(rounding=True)
   │   ├── 评估综合违反量
   │   ├── 筛选所有候选轨迹
   │   └── 选择综合违反量最小的轨迹
   └── 提前退出: 速度+曲率+加速度均可行

8. 评估轨迹
   └── TrajectoryEvaluator.evaluate_trajectory()
       ├── 速度/加速度/曲率/工作空间约束验证
       ├── C0/C1/C2连续性检查
       ├── 曲率统计信息
       └── 曲率后验验证（启用硬约束时）

9. 返回PlanningResult
```

### 8.2 GCS求解细节

```
1. 凸松弛: 将整数变量松弛为连续变量
2. 求解凸优化: 使用MOSEK求解器
3. 舍入(Rounding): 将松弛解舍入为整数解
   ├── 随机前向搜索
   └── 随机后向搜索
4. 提取轨迹: 从GCS解中提取控制点
5. 多候选筛选: 从所有舍入路径中选择最优
```

### 8.3 曲率后验验证

启用曲率硬约束后，即使添加了Lorentz锥约束，仍可能存在曲率违反（主要在边界段v≈0附近）。规划器在后验验证中检查曲率违反并报告可能原因:

1. 边界段（v=0附近）曲率超限
2. 保守性不足: min_velocity设置过低
3. 成本权重不当: w_time/w_energy比值过小导致速度过低

---

## 9. 求解器配置

### 9.1 自适应求解器配置 (SolverPerformanceProfile)

根据问题规模自动调整求解参数:

- `_estimate_problem_size()`: `complexity_score = num_edges × dimension`
- 三档预定义:
  - SMALL: relaxation_tol=1e-6
  - MEDIUM: relaxation_tol=1e-4
  - LARGE: relaxation_tol=1e-3
- 支持4种求解器: MOSEK / GUROBI / CLP / SCS

### 9.2 MOSEK优化统一配置 (MosekOptimizationConfig)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_paths` | 5 | 最大舍入路径数 |
| `num_threads` | 8 | 求解线程数 |
| `mio_max_time` | 30.0 | MIP求解时间限制(秒) |
| `max_rounding_attempts` | 3 | 最大舍入尝试次数 |
| `max_rounded_paths` | 5 | 每次舍入最大路径数 |

**独立开关**:
- `enable_reduced_paths`: 减少路径数
- `enable_thread_limit`: 线程数限制
- `enable_mio_time_limit`: MIP时间限制
- `enable_incremental_phi`: 增量Phi约束更新

`effective_*()` 方法: 根据开关返回实际生效值。

### 9.3 增量Phi约束更新 (IncrementalPhiUpdater)

跟踪上一条Rounding路径的边集合，仅对差异边执行Phi约束修改，开销 O(|E_prev △ E_curr|)。

---

## 10. 关键设计特点

### 10.1 约束设计特点

**航向角约束创新**:
- 旋转矩阵法使用叉积形式避免奇异，数值稳定，凸性约束
- 方向约束使用点积形式确保朝向一致，解决方向歧义
- 多控制点约束提高轨迹质量，减少振荡

**速度约束优化**:
- 标量约束比矢量约束精确，可行域大27%
- SOCP/Lorentz锥保持凸性，符合阿克曼车辆物理模型

**曲率约束设计**:
- Lorentz锥凸约束保证凸性，保守但可靠
- h̄'迭代修正使约束逐步精确化，自动收敛
- 动态safety_factor收紧补偿h̄'下降
- 求解失败放宽重试机制
- 与regularization_r形成双重保障

### 10.2 成本设计特点

**5种成本类型**:
- 时间成本(线性)、路径长度成本(L2范数)、能量成本(透视二次)
- 时间导数正则化、导数正则化(二次)

**13种预设模板**: 覆盖时间优先、路径优先、能量优先、平滑、月面场景、曲率约束场景

**量纲归一化**: 基于特征值的量纲归一化，确保各成本项量级一致

**成本优化器**: CostOptimizer自动调优权重以达到目标性能指标

### 10.3 数值稳定性

**旋转矩阵法**: 避免tan函数奇异，使用sin/cos，适用于所有角度

**数值安全工具** (numerical_safety_utils.py):
- `safe_divide()`: 安全除法，避免除零
- `safe_power_divide()`: 安全幂除法
- `check_tan_safety()`: 检查tan函数安全性
- `safe_norm()`: 安全范数计算
- `clamp_value()`: 值域截断

**数值常量** (constants.py):
- NUMERICAL_TOLERANCE = 1e-10
- SMALL_VALUE_THRESHOLD = 1e-6
- HDOT_MIN_DEFAULT = 0.01
- HDOT_MIN_WARNING_THRESHOLD = 0.001

---

## 11. 参数配置指南

### 11.1 车辆参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| wheelbase | 2.5 | 轴距（米） |
| max_steering_angle | 85° (1.484 rad) | 最大转向角 |
| max_velocity | 2.0-10.0 | 最大速度（m/s） |
| max_acceleration | 根据车辆 | 最大加速度（m/s²） |

默认推导: κ_max = tan(85°) / 2.5 ≈ 4.587 1/m

### 11.2 贝塞尔曲线参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| order | 5-7 | 贝塞尔曲线阶数,阶数越高越灵活 |
| continuity | 1-2 | 连续性阶数,C2适合自动驾驶 |
| hdot_min | 0.01 | 时间导数最小值 |

### 11.3 航向角约束参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| method | rotation_matrix | 约束方法,推荐旋转矩阵法 |
| num_control_points | 2-3 | 多控制点约束数量 |
| enable_direction_constraint | True | 是否启用方向约束 |

### 11.4 速度约束参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| max_velocity | 2.0-10.0 | 最大速度(m/s) |
| 约束类型 | 标量约束(SOCP) | 推荐标量约束 |

### 11.5 曲率约束参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| curvature_constraint_mode | "hard" | 曲率约束模式 |
| min_velocity | 1.58 | 最小速度（从成本权重推导） |
| h_bar_prime | None（自动迭代） | h̄'均值估计 |
| h_bar_prime_safety_factor | 0.7 | 保守修正因子 |
| max_h_bar_prime_iterations | 3 | h̄'迭代最大次数 |
| h_bar_prime_convergence_threshold | 0.15 | 收敛阈值（15%） |
| h_bar_prime_relax_factor | 1.3 | 求解失败放宽因子 |
| h_bar_prime_safety_factor_decay | 0.8 | 动态收紧衰减因子 |

### 11.6 成本权重参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| time | 0.5-10.0 | 时间成本权重 |
| path_length | 0.5-10.0 | 路径长度成本权重 |
| energy | 0-20.0 | 能量成本权重 |
| time_derivative_reg | 0-5.0 | 时间导数正则化权重 |
| regularization_r | 0-10.0 | 空间导数正则化权重 |
| regularization_h | 0-1.0 | 时间导数正则化权重 |
| regularization_order | 2 | 正则化导数阶数 |

**推荐使用预设模板**: `CostConfigurator().set_preset('curvature_constrained')` 等。

---

## 12. 性能优化建议

### 12.1 求解效率优化

- **减少约束数量**: 使用必要约束，避免冗余，使用稀疏约束表示
- **调整h̄'迭代参数**: 增大收敛阈值减少迭代次数，降低max_h_bar_prime_iterations
- **调整MOSEK参数**: 增加线程数、设置MIP时间限制
- **增量Phi更新**: 启用enable_incremental_phi减少约束修改开销
- **并行化**: 多次舍入尝试间独立，可并行执行

### 12.2 解的质量优化

- **提高贝塞尔阶数**: 增加灵活性，但求解时间增加
- **使用曲率硬约束+高regularization_r**: 双重保障，如 `curvature_constrained` 预设
- **增加舍入尝试次数**: max_rounding_attempts=3-5，探索更多候选路径
- **调整成本权重**: 使用预设模板或CostOptimizer自动调优

### 12.3 数值稳定性优化

- **使用旋转矩阵法**: 处理航向角约束，避免tan奇异
- **使用标量约束**: 处理速度约束，更符合物理实际
- **合理设置hdot_min**: 默认0.01，不低于0.001
- **合理设置min_velocity**: 从成本权重推导，避免过低
- **使用h̄'迭代修正**: 自动确定h̄'，避免手动设置不当

---

## 附录A: 数学符号说明

| 符号 | 说明 |
|------|------|
| r(s) | 空间轨迹(关于参数s) |
| h(s) | 时间轨迹(关于参数s) |
| h̄' | h'(s)的均值估计 |
| P_i | 第i个控制点 |
| Q_j | 第j个二阶导数控制点 |
| θ | 航向角 |
| κ | 曲率 |
| κ_max | 最大曲率 |
| v | 速度向量 |
| ρ_min | 弧长参数下最小速度下界 |
| ‖·‖₂ | L2范数 |
| ∇ | 梯度算子 |
| d/ds | 对参数s的导数 |
| d/dt | 对时间t的导数 |
| R(θ) | 旋转矩阵 |
| d_θ | 期望方向向量 [cos(θ), sin(θ)] |

## 附录B: 关键源码文件索引

| 文件 | 核心类/函数 |
|------|------------|
| `src/ackermann_gcs_pkg/ackermann_gcs_planner.py` | `AckermannGCSPlanner` |
| `src/ackermann_gcs_pkg/ackermann_bezier_gcs.py` | `AckermannBezierGCS` |
| `src/ackermann_gcs_pkg/ackermann_data_structures.py` | `VehicleParams`, `EndpointState`, `TrajectoryConstraints`, `BezierConfig` |
| `src/ackermann_gcs_pkg/h_bar_prime_iteration.py` | `iterate_h_bar_prime()`, `HBarPrimeIterationResult` |
| `src/ackermann_gcs_pkg/constraint_utils.py` | `compute_constraint_violation()`, `check_constraint_satisfaction()` |
| `src/ackermann_gcs_pkg/rotation_matrix_heading_constraint.py` | `RotationMatrixHeadingConstraint`, `DirectionConstraint` |
| `src/ackermann_gcs_pkg/curvature_utils.py` | `compute_curvature()`, `compute_curvature_gradient()` |
| `src/ackermann_gcs_pkg/flat_output_mapper.py` | `FlatOutputMapper` |
| `src/ackermann_gcs_pkg/trajectory_evaluator.py` | `TrajectoryEvaluator` |
| `src/ackermann_gcs_pkg/constants.py` | 数值常量定义 |
| `src/ackermann_gcs_pkg/numerical_safety_utils.py` | `safe_divide()`, `check_tan_safety()` |
| `config/gcs/cost_configurator.py` | `CostWeights`, `CostConfigurator`, `CostOptimizer` |
| `config/solver/solver_config.py` | `AdaptiveSolverConfig`, `SolverPerformanceProfile` |
| `config/solver/mosek_opt_config.py` | `MosekOptimizationConfig` |
