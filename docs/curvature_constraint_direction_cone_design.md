# 无迭代方向锥曲率约束设计方案

日期：2026-05-16

## 1. 目标

当前项目的曲率硬约束依赖 `h_bar_prime` 估计，并通过“先求解、估计、重加约束、再求解”的方式修正参数。该流程会显著增加 GCS 求解次数，导致计算时间过长。

本方案目标是设计一种新的曲率约束：

- 数学上可证明成立；
- 保持凸性；
- 不需要从 GCS 求解结果中迭代估计参数；
- 不与现有速度、时间、能量、工作空间约束冲突；
- 可嵌入当前 Bezier/GCS 框架；
- 尽可能降低计算成本；
- 满足 Ackermann 车辆最小转弯半径约束。

核心思路是：不用时间参数估计 `h_bar_prime`，而是从求解前已知的粗路径、凸走廊和车辆几何参数中构造每段的方向锥约束与参数速度下界。

## 2. 当前方案问题

当前曲率硬约束位于：

- `src/gcs_pkg/scripts/core/bezier.py` 的 `addCurvatureHardConstraint`
- `src/ackermann_gcs_pkg/h_bar_prime_iteration.py` 的 `iterate_h_bar_prime`

当前约束形式为：

```text
||Q_j||_2 <= kappa_max * rho_min^2
rho_min = min_velocity * h_bar_prime
```

其中 `Q_j` 是空间轨迹二阶导数 Bezier 控制点。该约束是凸的，但 `rho_min` 依赖时间轨迹 `h(s)` 的平均导数估计。若 `h_bar_prime` 未知，就需要先求解轨迹，再根据结果估计，再重新添加曲率约束求解。

主要问题：

- 需要多次 GCS 求解；
- `h_bar_prime` 是平均量，不是点态下界；
- 迭代修正与最终求解流程存在额外计算放大；
- 保守性由全局速度/时间下界控制，难以针对局部几何调整。

## 3. 为什么不采用“曲率平方精确凸化”

二维参数曲线曲率为：

```text
kappa(s) = det(r'(s), r''(s)) / ||r'(s)||^3
```

精确曲率约束为：

```text
det(r'(s), r''(s))^2 <= kappa_max^2 * ||r'(s)||^6
```

该约束包含 `r'(s)` 与 `r''(s)` 的双线性项，以及变量范数的高次幂，一般不是凸约束，也不能无损转化为 SOCP。

常见的替代式：

```text
||r''(s)|| <= kappa_max * ||r'(s)||^2
```

不是精确曲率约束，而是保守充分条件。若把右侧引入辅助变量并写成 SOC，也会遇到安全方向的非凸约束。因此，本方案不追求不存在的精确凸重写，而采用可证明安全的凸充分约束。

## 4. 推荐方案概述

对每个 GCS/Bezier 段 `k`，预先给定：

```text
t_k              名义几何前进方向，单位向量
n_k              t_k 的法向量，n_k = [-t_y, t_x]
rho_k            沿 t_k 的参数速度下界
theta_max,k      方向锥半角
kappa_max        最大曲率，kappa_max = 1 / R_min
```

对该段 Bezier 曲线：

```text
r_k(s), s in [0, 1]
```

一阶导控制点：

```text
D_i = n * (P_{i+1} - P_i)
```

二阶导控制点：

```text
A_j = n * (n - 1) * (P_{j+2} - 2P_{j+1} + P_j)
```

添加三组约束。

### 4.1 参数速度下界

```text
t_k^T D_i >= rho_k
```

由 Bezier 凸包性质可知：

```text
t_k^T r'_k(s) >= rho_k
```

因此：

```text
||r'_k(s)|| >= rho_k
```

### 4.2 方向锥约束

```text
|n_k^T D_i| <= tan(theta_max,k) * t_k^T D_i
```

等价线性形式：

```text
 n_k^T D_i <= eta_k * t_k^T D_i
-n_k^T D_i <= eta_k * t_k^T D_i
eta_k = tan(theta_max,k)
```

该约束保证轨迹切向不偏离名义方向过多。

### 4.3 曲率安全约束

```text
|n_k^T A_j| + eta_k * |t_k^T A_j| <= kappa_max * rho_k^2
```

二维下可展开为 4 条线性约束：

```text
 sigma * n_k^T A_j + eta_k * tau * t_k^T A_j <= kappa_max * rho_k^2
 sigma, tau in {-1, +1}
```

因此该方案不需要二阶锥，直接是线性约束，计算上比当前 Lorentz 锥曲率约束更轻。

## 5. 正确性说明

令：

```text
r'  = alpha * t_k + q * n_k
r'' = b * t_k + c * n_k
```

其中：

```text
alpha = t_k^T r'
q     = n_k^T r'
b     = t_k^T r''
c     = n_k^T r''
```

方向锥给出：

```text
|q| <= eta_k * alpha
alpha >= rho_k
```

曲率分子满足：

```text
|det(r', r'')|
= |alpha * c - q * b|
<= alpha * |c| + |q| * |b|
<= alpha * (|c| + eta_k * |b|)
```

若对所有二阶导控制点约束：

```text
|c| + eta_k * |b| <= kappa_max * rho_k^2
```

则对整段曲线有：

```text
|det(r', r'')| <= alpha * kappa_max * rho_k^2
```

又由于：

```text
||r'|| >= alpha >= rho_k
```

因此：

```text
|kappa|
= |det(r', r'')| / ||r'||^3
<= alpha * kappa_max * rho_k^2 / alpha^3
<= kappa_max
```

所以该约束是曲率约束的凸充分条件，可保证最小转弯半径：

```text
R >= R_min
```

## 6. 参数来源总览

计算 `t_k / rho_k / theta_max,k` 需要以下输入量。

| 类别 | 输入量 | 用途 |
| --- | --- | --- |
| 粗路径 | `(x_i, y_i, theta_i)` | 生成局部切线和局部转角 |
| 粗路径 | `gear_i` | 判断前进/倒车与车头方向关系 |
| 粗路径 | 局部路径弧长 | 生成 `rho_k` |
| 粗路径 | 局部切线变化角 | 生成 `theta_max,k` |
| 凸区域 | `C_k` | 计算纵向/横向投影宽度 |
| 凸区域 | seed 或 center | 辅助生成 `t_k` |
| 凸区域 | 相邻区域 overlap | 限制 `rho_k`，可选 |
| 车辆参数 | `R_min` / `kappa_max` | 曲率约束右侧 |
| 配置参数 | `alpha, beta, gamma` | 几何安全系数 |
| 配置参数 | `theta_min, theta_abs_max, margin` | 角度上下限 |

## 7. `t_k` 的计算

`t_k` 是世界坐标下轨迹的几何移动方向，不等同于车辆车头方向。倒车时，车辆车头方向与 `t_k` 相反。

### 7.1 从粗路径切线计算

给定粗路径点：

```text
p_i = (x_i, y_i, theta_i)
```

使用窗口差分：

```text
d_path,k = p_{i+w,xy} - p_{i-w,xy}
t_path,k = normalize(d_path,k)
```

`w` 是平滑窗口。`w` 越大，方向越平滑，但会抹掉急弯。

### 7.2 从凸区域中心辅助计算

若每个凸区域有 seed 或 center：

```text
t_region,k = normalize(c_{k+1} - c_k)
```

### 7.3 融合公式

```text
t_k = normalize(lambda * t_path,k + (1 - lambda) * t_region,k)
```

推荐：

```text
lambda = 0.7 ~ 0.9
```

第一版可直接使用 `t_path,k`，区域中心方向作为 fallback。

### 7.4 gear 对 `t_k` 的影响

`t_k` 不随 gear 翻转。它始终表示轨迹在世界坐标中的移动方向。

设车辆航向单位向量：

```text
h_k = [cos(theta_k), sin(theta_k)]
```

前进段：

```text
t_k dot h_k > 0
```

倒车段：

```text
t_k dot h_k < 0
```

如果允许倒车，后续 heading 约束必须支持 `gear * h_k`。

## 8. `rho_k` 的计算

`rho_k` 是固定几何下界：

```text
t_k^T r'_k(s) >= rho_k
```

它不是物理速度，不依赖时间轨迹 `h(s)`，单位是米/参数单位。

### 8.1 粗路径投影长度

对分配到第 `k` 段的粗路径点：

```text
ell_path,k = sum max(0, t_k^T (p_{i+1,xy} - p_{i,xy}))
```

该量表示粗路径在 `t_k` 方向上的局部推进长度。

### 8.2 凸区域纵向宽度

对 HPolyhedron `C_k`：

```text
W_parallel,k = support(C_k, t_k) - support(C_k, -t_k)
```

其中：

```text
support(C, a) = max_{x in C} a^T x
```

可通过小规模线性规划计算。

### 8.3 overlap 宽度

若能计算相邻区域交叠：

```text
W_overlap,k = support(C_k ∩ C_{k+1}, t_k)
            - support(C_k ∩ C_{k+1}, -t_k)
```

该项可防止 `rho_k` 大于相邻区域之间实际可通行的推进尺度。

### 8.4 推荐公式

最小实现：

```text
rho_k = min(beta * ell_path,k,
            alpha * W_parallel,k)
```

更稳健实现：

```text
rho_k = min(beta  * ell_path,k,
            alpha * W_parallel,k,
            gamma * W_overlap,k)
```

推荐系数：

```text
alpha = 0.5 ~ 0.8
beta  = 0.6 ~ 0.85
gamma = 0.5 ~ 0.8
```

### 8.5 影响关系

| 输入量 | 对 `rho_k` 的影响 |
| --- | --- |
| `ell_path,k` 越大 | `rho_k` 可增大 |
| 纵向宽度越大 | `rho_k` 上限增大 |
| overlap 越小 | `rho_k` 必须减小 |
| `alpha / beta / gamma` 越大 | 前进约束更强，可能更易不可行 |
| `rho_k` 太小 | `kappa_max * rho_k^2` 过小，曲率约束过紧 |
| `rho_k` 太大 | `t_k^T D_i >= rho_k` 难满足 |

注意：`rho_k` 不是越小越容易。因为曲率约束右侧是 `kappa_max * rho_k^2`，`rho_k` 太小会迫使二阶导数极小，导致轨迹过直或不可行。

## 9. `theta_max,k` 的计算

`theta_max,k` 控制导数方向允许偏离 `t_k` 的角度。

方向锥为：

```text
|n_k^T D_i| <= tan(theta_max,k) * t_k^T D_i
```

### 9.1 局部路径转角需求

可由相邻平滑切线计算：

```text
theta_path,k = max angle(t_{k-r}, ..., t_{k+r})
```

或者用：

```text
theta_path,k = angle(t_{k-1}, t_{k+1}) / 2
```

### 9.2 凸区域横向裕度

横向宽度：

```text
W_lateral,k = support(C_k, n_k) - support(C_k, -n_k)
```

纵向宽度：

```text
W_parallel,k = support(C_k, t_k) - support(C_k, -t_k)
```

基于区域形状给出角度上限：

```text
theta_clear,k = atan(mu * W_lateral,k / max(W_parallel,k, eps))
```

`mu` 是保守系数，推荐：

```text
mu = 0.5 ~ 1.0
```

### 9.3 推荐公式

```text
theta_max,k = clamp(theta_path,k + margin,
                    theta_min,
                    min(theta_clear,k, theta_abs_max))
```

推荐默认：

```text
theta_min     = 25 deg
theta_abs_max = 45 deg
margin        = 10 deg
```

若不想第一版过复杂，可固定：

```text
theta_max = 35 deg
```

### 9.4 影响关系

| 输入量 | 对 `theta_max,k` 的影响 |
| --- | --- |
| 横向宽度越大 | 可允许更大偏角 |
| 纵向宽度越大 | 同横向宽度下偏角应更小 |
| 局部路径转角越大 | 需要更大偏角 |
| 粗路径锯齿越严重 | 应先平滑，不应直接增大偏角 |
| `theta_max,k` 太小 | 转弯段容易不可行 |
| `theta_max,k` 太大 | 曲率约束中 `tan(theta)` 项变大，充分条件更保守 |

## 10. 三个参数的耦合关系

三者不是独立参数。

```text
t_k      决定纵向/横向分解
rho_k    决定曲率约束右侧大小
theta    决定允许偏离 t_k 的程度
```

曲率约束：

```text
|n_k^T A_j| + tan(theta_max,k) * |t_k^T A_j|
    <= kappa_max * rho_k^2
```

因此：

- `t_k` 错误会导致所有分解错误，是最高风险参数；
- `rho_k` 太小会导致曲率约束过紧；
- `rho_k` 太大会导致参数速度下界不可满足；
- `theta_max,k` 太小会导致转弯不可行；
- `theta_max,k` 太大会增加曲率充分条件中的保守项。

## 11. 凸区域大小对参数的影响

| 凸区域特征 | 对 `t_k` 的影响 | 对 `rho_k` 的影响 | 对 `theta_max,k` 的影响 |
| --- | --- | --- | --- |
| 窄长走廊 | 走廊方向可信 | 受纵向宽度限制 | 应较小 |
| 宽大区域 | 方向不唯一，应更依赖粗路径 | 可较大但需 cap | 可较大 |
| overlap 小 | edge 方向不稳定 | 必须减小 | 不宜太大 |
| 区域中心偏离通道 | center 连线不可信 | 用路径长度限制 | 用保守值 |
| 区域过短 | 方向易退化 | `rho` 易过小 | 应合并段 |

## 12. 粗路径质量要求

该方案不要求粗路径本身严格满足曲率约束，但要求粗路径提供可靠拓扑和局部方向。

最低要求：

- 同伦类正确，穿过正确通道；
- 局部切线不过度抖动；
- 采样间距不能大量极短；
- gear 分段清楚；
- 前进/倒车方向与车辆航向基本一致；
- 一个 GCS 段内不要包含换挡。

建议预处理：

```text
去重 -> 重采样 -> 平滑切线 -> 按急弯/gear 切段
```

经验标准：

```text
相邻 t_k 夹角不要频繁大于 45 deg ~ 60 deg
ds_ref >= 0.5 m 或 0.1 * R_min
```

如果粗路径存在尖角，应先合并、重采样或重新生成走廊，而不是通过无限增大 `theta_max` 解决。

## 13. 允许倒车时的影响

当前 A* 基元中已经存在 `gear = +1 / -1`，但后续 GCS 轨迹规划主要接收 pose 序列，gear 信息没有作为一等数据传递。

### 13.1 不需要大改 GCS 基础框架

GCS 图、凸区域、边约束、成本绑定机制不需要重写。

### 13.2 需要修改的数据流和约束

若支持固定 gear 序列的倒车，需要：

1. A* 输出每段 gear；
2. 粗路径到 GCS 段映射时保留 gear；
3. heading 约束支持：

```text
gear = +1: r'(s) parallel  h(theta)
gear = -1: r'(s) parallel -h(theta)
```

4. 端点速度应支持 signed velocity 或 `(speed, gear)` 表示；
5. 成本中加入倒车惩罚和换挡惩罚。

### 13.3 不同倒车支持层级

| 层级 | 改动量 | 说明 |
| --- | --- | --- |
| 前进-only | 中等 | 最适合作为第一版 |
| 固定 gear 序列 | 中等偏上 | GCS 不选择换挡，只服从 A* gear |
| gear-layered GCS | 大 | 复制区域层，GCS 自己选择前进/倒车/换挡 |

建议先实现前进-only，再实现固定 gear 序列。不要第一版直接做 gear-layered GCS。

## 14. 推荐最小实现

第一版采用：

```text
t_k = smooth_tangent_from_astar_path(k)

W_parallel,k = support(C_k, t_k) - support(C_k, -t_k)
W_lateral,k  = support(C_k, n_k) - support(C_k, -n_k)

ell_path,k = sum max(0, t_k^T delta_p_i)

rho_k = min(0.75 * ell_path,k,
            0.60 * W_parallel,k)

theta_max,k = clamp(local_tangent_deviation_k + 10 deg,
                    25 deg,
                    45 deg)
```

对应约束：

```text
t_k^T D_i >= rho_k
 n_k^T D_i <= tan(theta_max,k) * t_k^T D_i
-n_k^T D_i <= tan(theta_max,k) * t_k^T D_i

sigma * n_k^T A_j + eta_k * tau * t_k^T A_j
    <= kappa_max * rho_k^2
sigma, tau in {-1, +1}
```

## 15. 实施建议

建议新增配置模式：

```text
curvature_constraint_mode = "direction_cone"
```

新增模块或函数：

```text
DirectionalCurvatureParameterBuilder
addDirectionalCurvatureConstraint(...)
```

推荐改动位置：

- `src/ackermann_gcs_pkg/ackermann_data_structures.py`
  - 增加新配置字段；
- `src/gcs_pkg/scripts/core/bezier.py`
  - 增加 edge-wise 方向锥曲率约束；
- `src/ackermann_gcs_pkg/ackermann_gcs_planner.py`
  - 在新模式下跳过 `h_bar_prime` 迭代；
- `src/path_planner` 或 `ackermann_gcs_pkg`
  - 增加参数生成器，将粗路径/区域映射到 GCS 段。

## 16. 验证指标

必须比较当前方案与新方案：

- GCS 总求解时间；
- GCS 求解次数；
- 规划成功率；
- 后验最大曲率；
- 曲率违反量；
- 速度、加速度、工作空间约束违反量；
- basic / narrow / s_curve / parking 场景成功率。

建议命令：

```text
pytest tests/unit/ -v
python scripts/batch_test_curvature_constraint.py
python scripts/hybrid_astar_gcs_planner.py
```

完整链路依赖 Drake，实际验证应在 Ubuntu 的 `iris-py3.12` 环境中执行。

## 17. 风险

- 该约束是充分条件，不是精确等价曲率约束；
- 参数质量依赖粗路径和凸区域几何；
- `t_k` 错误会造成不可行或过保守；
- `rho_k` 太小会导致曲率约束过紧；
- `rho_k` 太大会导致前进约束不可行；
- `theta_max,k` 太大时，曲率约束中的 `tan(theta)` 项会增加保守性；
- 倒车支持需要保留 gear 数据，否则 heading 约束会错误排斥倒车轨迹。

## 18. 结论

该方案可作为当前 `h_bar_prime` 迭代曲率约束的无迭代替代方案。它通过粗路径和凸区域几何在求解前确定 `t_k / rho_k / theta_max,k`，并将曲率约束转化为线性充分条件。

推荐实施顺序：

1. 前进-only 方向锥曲率约束；
2. 加入区域宽度对 `rho_k / theta_max,k` 的限制；
3. 批量验证求解时间与曲率违反量；
4. 保留当前 Lorentz 曲率约束作为 fallback；
5. 再考虑固定 gear 序列的倒车支持；
6. 最后才考虑 gear-layered GCS。

## 19. 稳健且高效的推荐落地方案

本节给出在不大幅降低系统运行效率的前提下，可行性和参数鲁棒性最好的推荐方案。核心原则是：

- 必算项只使用低成本几何量；
- 容易导致前处理变重的 overlap 尺度作为按需增强项；
- 参数先保守但不过度收紧；
- 后验验证只用于诊断和 fallback，不在单次规划中引入迭代修正。

### 19.1 推荐总体策略

采用“两级参数生成 + 按需增强”的方式。

第一级是默认路径，所有规划都执行：

```text
1. 从 Hybrid A* 粗路径生成平滑切线 t_path,k
2. 从 HPolyhedron 计算区域中心和纵向/横向 support 宽度
3. 用路径局部推进长度 ell_path,k 与区域纵向宽度 W_parallel,k 计算 rho_k
4. 用局部转角和横纵宽度比计算 theta_max,k
5. 添加线性方向锥曲率约束
```

第二级是增强路径，仅在以下情况触发：

```text
1. 区域 overlap 很小或 GCS 图边很密
2. 后验曲率验证频繁失败
3. rho_k 被区域宽度限制得过低
4. t_k 与区域连接方向冲突明显
```

增强路径才计算相邻区域交叠尺度 `W_overlap,k`，并只对候选路径附近的边或高风险边计算，避免对全图所有边做大量 LP。

### 19.2 参数获取的推荐数据源

| 参数 | 首选来源 | fallback | 不建议 |
| --- | --- | --- | --- |
| `t_k` | 平滑后的 A* 路径切线 | 相邻 IRIS seed/centroid 连线 | 从 GCS 解中反推 |
| `rho_k` | `min(beta * ell_path,k, alpha * W_parallel,k)` | 加入 `gamma * W_overlap,k` 限制 | 使用时间轨迹 `h_bar_prime` |
| `theta_max,k` | 局部路径转角 + 区域横纵宽度限制 | 固定 `35 deg` | 为追求可行性无限放大 |
| `R_min/kappa_max` | `VehicleParams` | 显式配置覆盖 | 从轨迹后验估计 |
| gear | A* 运动基元输出 | pose 差分与 heading 点积估计 | 忽略倒车方向 |

### 19.3 高效默认公式

默认公式应避免 overlap 全图计算。

```text
t_k = smooth_tangent_from_reference_path(k)
n_k = [-t_k,y, t_k,x]

ell_path,k = sum max(0, t_k^T (p_{i+1} - p_i))

W_parallel,k = support(C_k,  t_k) - support(C_k, -t_k)
W_lateral,k  = support(C_k,  n_k) - support(C_k, -n_k)

rho_k = min(0.75 * ell_path,k,
            0.60 * W_parallel,k)

theta_width,k = atan(0.8 * W_lateral,k / max(W_parallel,k, eps))

theta_max,k = clamp(local_tangent_deviation_k + 10 deg,
                    25 deg,
                    min(theta_width,k, 45 deg))
```

该版本的计算成本主要包括：

- 路径扫描：`O(N)`
- 每个区域 4 次 support LP：`O(4K)`
- 每段常数规模线性约束生成：`O(K * order)`

相比当前多次 GCS/SOCP 求解，该前处理成本通常较小。

### 19.4 support 宽度的性能控制

区域宽度计算需要解小型 LP。为避免重复计算，应缓存：

```text
cache_key = (region_id, rounded_direction)
```

其中 `rounded_direction` 可以按角度量化，例如每 `5 deg` 或 `10 deg` 一个桶。

如果区域已有顶点数组，可先用顶点近似：

```text
support(C, a) ~= max_v a^T v
```

顶点近似的优点是极快；缺点是对 IrisZo 的 HPolyhedron 可能没有 vertices。推荐策略：

```text
if region.vertices available:
    use vertex projection width
else:
    use LP support width
```

这样能在 IrisNp 区域上获得较低成本，在 IrisZo 区域上保持准确性。

### 19.5 overlap 尺度的按需计算

`W_overlap,k` 能提高鲁棒性，但如果对全图所有边计算，会产生：

```text
2 * num_edges 个额外 LP
```

在区域较多或图较密时会明显增加前处理时间。因此不建议第一版默认启用全图 overlap。

推荐触发条件：

```text
rho_k < rho_warning_threshold
or W_parallel,k / ell_path,k < width_ratio_threshold
or angle(t_path,k, t_region,k) > 45 deg
or previous validation reports curvature violation on similar scenario
```

只对触发条件对应的局部边计算：

```text
W_overlap,k = support(C_i intersection C_j, t_k)
            - support(C_i intersection C_j, -t_k)

rho_k = min(rho_k, 0.60 * W_overlap,k)
```

若 overlap LP 不可行，说明该边不应承载方向锥曲率约束，应标记为高风险边或交给 GCS 图连通性逻辑处理。

### 19.6 可行性优先的参数保护

为减少参数错误导致不可行，需要加入以下保护。

#### 19.6.1 `t_k` 保护

若粗路径切线与区域连接方向冲突：

```text
angle(t_path,k, t_region,k) > 60 deg
```

不要直接融合。优先使用粗路径切线，并降低 `rho_k`：

```text
t_k = t_path,k
rho_k *= 0.7
theta_max,k = min(theta_max,k + 10 deg, 45 deg)
```

原因是粗路径决定同伦类，区域中心可能因区域形状不规则而偏离通道中心。

#### 19.6.2 `rho_k` 保护

设置诊断阈值，但不要盲目向上 clamp：

```text
rho_warning_threshold = 0.2 * R_min
```

若：

```text
rho_k < rho_warning_threshold
```

优先处理方式是：

```text
1. 合并过短路径段
2. 扩大粗路径重采样间距
3. 对该段降低曲率约束强度或切换 fallback
```

不要简单写成：

```text
rho_k = max(rho_k, rho_min)
```

因为这可能制造不可行的一阶导前进约束。

#### 19.6.3 `theta_max,k` 保护

`theta_max,k` 应限制在：

```text
25 deg <= theta_max,k <= 45 deg
```

特殊宽阔区域可放宽到 `60 deg`，但不建议作为默认值。因为 `tan(theta_max,k)` 会直接放大曲率约束左侧的切向二阶导项，使充分条件更保守。

### 19.7 粗路径预处理要求

为保证参数鲁棒性，应在参数生成前对粗路径执行：

```text
1. 删除重复点和极短边
2. 按 ds_ref 重采样
3. 对位置切线做窗口平滑
4. 对 theta 做 unwrap
5. 如果有 gear，按 gear change 切段
6. 如果局部转角过大，插入过渡段或标记高风险段
```

推荐：

```text
ds_ref = max(0.5 m, 0.1 * R_min)
smooth_window_length = max(1.0 m, 0.2 * R_min)
max_adjacent_tangent_angle = 60 deg
```

如果粗路径无法满足这些条件，不应通过调整方向锥参数硬撑，而应回到 A*/走廊生成阶段改进路径质量。

### 19.8 对运行效率的影响

默认高效方案的额外开销为：

```text
O(N) 路径预处理
+ O(4K) 区域 support 宽度计算
+ O(K * order) 线性约束生成
```

其中：

- `N` 是粗路径点数；
- `K` 是凸区域或路径段数量；
- `order` 是 Bezier 阶数。

当前迭代方案的主要成本是多次 GCS/SOCP 求解。只要新方案避免全图 overlap 计算，并缓存 support 宽度，整体运行效率预计会提升。

建议默认关闭：

```text
compute_all_overlap_widths = False
```

建议默认开启：

```text
cache_region_support_widths = True
use_vertex_width_when_available = True
```

### 19.9 最推荐的实现等级

推荐按以下等级落地。

#### Level 1：高效前进-only 默认方案

```text
t_k: A* 平滑切线
rho_k: min(0.75 * ell_path,k, 0.60 * W_parallel,k)
theta_max,k: clamp(local_turn + 10 deg, 25 deg, 45 deg)
overlap: 不计算
gear: 不启用
```

用途：验证曲率后验、求解时间、成功率。

#### Level 2：鲁棒前进-only 方案

```text
t_k: A* 平滑切线 + 区域连接方向一致性检查
rho_k: 加入高风险段局部 overlap 限制
theta_max,k: 加入区域横纵宽度限制
support: 缓存
overlap: 只对高风险边计算
```

用途：作为推荐默认方案。

#### Level 3：固定 gear 倒车方案

```text
gear: 从 A* motion primitive 保留
t_k: 仍为世界坐标移动方向
heading: 使用 gear-aware 方向约束
cost: 加 reverse cost 和 gear switch cost
```

用途：支持倒车场景，但不让 GCS 自己选择换挡。

#### Level 4：gear-layered GCS

```text
复制 forward/reverse 区域层
边携带 gear
GCS 选择换挡点
加入换挡成本
```

用途：完整倒车/换挡规划。该级别改动大，不建议作为第一阶段。

### 19.10 推荐结论

在不大幅降低系统运行效率的前提下，最佳方案是 **Level 2：鲁棒前进-only 方案**：

```text
1. t_k 由平滑 A* 路径切线生成，并用区域连接方向做一致性检查；
2. rho_k 由路径推进长度和区域纵向宽度共同限制；
3. theta_max,k 由局部转角和区域横纵宽度共同限制；
4. support 宽度计算做缓存；
5. overlap 宽度只对高风险边按需计算；
6. 后验曲率验证只用于诊断和 fallback，不引入单次规划内迭代。
```

该方案在效率、可行性和参数鲁棒性之间的平衡最好。它避免了当前 `h_bar_prime` 多次 GCS 求解，也避免了全图 overlap 计算导致的前处理开销膨胀。

## 20. 逻辑闭环复核结论

截至本次复核，本文档的核心逻辑是闭环的，未发现阻断性的数学或工程逻辑错误。闭环关系如下：

1. 问题定义闭环：第 1-3 节明确当前 `h_bar_prime` 迭代方案的计算放大问题，并说明不追求精确曲率约束凸化，而采用可证明安全的凸充分条件。
2. 数学约束闭环：第 4-5 节从 `t_k / n_k / rho_k / theta_max,k` 出发，先约束一阶导控制点，再约束二阶导控制点；借助 Bezier 凸包性质，可把控制点约束推广到整段曲线。
3. 曲率证明闭环：方向锥给出 `|q| <= eta_k * alpha` 和 `alpha >= rho_k`，二阶导安全约束给出 `|c| + eta_k * |b| <= kappa_max * rho_k^2`，最终推出 `|kappa| <= kappa_max`。
4. 参数来源闭环：第 6-12 节说明 `t_k / rho_k / theta_max,k` 都来自求解前可获得的粗路径、凸区域和车辆参数，不依赖 GCS 求解结果。
5. 倒车边界闭环：第 13 节明确第一阶段采用前进-only，固定 gear 和 gear-layered GCS 作为后续等级，避免第一版把 heading 约束、gear 数据流和 GCS 结构改动混在一起。
6. 落地策略闭环：第 14-19 节给出最小实现、推荐 Level 2 方案、性能控制、风险保护、验证指标和 fallback 策略。

需要在实现中重点守住一个接口前提：方向锥参数必须按 GCS edge 或等价的 Bezier 段绑定，不能只按区域序号粗略保存。否则 GCS 选择非相邻边或跳边时，`t_k / rho_k / theta_max,k` 会和实际边不一致。

## 21. 完整任务清单

默认实施目标为 **Level 2：鲁棒前进-only 方案**。当前 Lorentz 曲率硬约束保留为 fallback；固定 gear 倒车和 gear-layered GCS 放入后续阶段。

### 21.1 阶段 0：基线确认

- [ ] **任务 0.1：确认当前曲率硬约束基线**
  - 文件：`src/gcs_pkg/scripts/core/bezier.py`
  - 文件：`src/ackermann_gcs_pkg/h_bar_prime_iteration.py`
  - 文件：`src/ackermann_gcs_pkg/ackermann_gcs_planner.py`
  - 动作：记录当前 `addCurvatureHardConstraint(...)`、`addCurvatureHardConstraintForEdges(...)` 和 `iterate_h_bar_prime(...)` 的调用路径。
  - 验收：能明确说明 `curvature_constraint_mode="hard"` 下是否进入 h_bar_prime 迭代，以及直接模式如何计算 `rho_min`。

- [ ] **任务 0.2：建立对比指标输出**
  - 文件：`scripts/batch_test_curvature_constraint.py`
  - 文件：`scripts/hybrid_astar_gcs_planner.py`
  - 动作：确认脚本能输出 GCS 求解次数、求解时间、成功率、后验最大曲率和曲率违反量。
  - 验收：后续新模式可以和 `"hard"` 模式在同一批场景上对比。

### 21.2 阶段 1：配置与数据结构

- [ ] **任务 1.1：扩展曲率约束模式枚举**
  - 修改：`src/ackermann_gcs_pkg/ackermann_data_structures.py`
  - 动作：在 `CurvatureConstraintMode` 中增加 `DIRECTION_CONE = "direction_cone"`。
  - 验收：`TrajectoryConstraints(curvature_constraint_mode="direction_cone", ...)` 能通过 `__post_init__` 校验。

- [ ] **任务 1.2：增加方向锥配置字段**
  - 修改：`src/ackermann_gcs_pkg/ackermann_data_structures.py`
  - 动作：在 `TrajectoryConstraints` 中增加以下字段：
    - `direction_cone_alpha: float = 0.60`
    - `direction_cone_beta: float = 0.75`
    - `direction_cone_gamma: float = 0.60`
    - `direction_cone_theta_min_deg: float = 25.0`
    - `direction_cone_theta_abs_max_deg: float = 45.0`
    - `direction_cone_theta_margin_deg: float = 10.0`
    - `direction_cone_width_mu: float = 0.8`
    - `direction_cone_compute_all_overlap_widths: bool = False`
    - `direction_cone_cache_support_widths: bool = True`
    - `direction_cone_use_vertex_width_when_available: bool = True`
    - `direction_cone_rho_warning_ratio: float = 0.2`
  - 验收：字段有正数和角度范围校验；`theta_min <= theta_abs_max`；`alpha/beta/gamma/width_mu` 均大于 0。

- [ ] **任务 1.3：补充配置单元测试**
  - 新增：`tests/unit/test_directional_curvature_config.py`
  - 动作：测试 `"direction_cone"` 模式合法、非法角度会抛 `ValueError`、非法系数会抛 `ValueError`。
  - 验证命令：`pytest tests/unit/test_directional_curvature_config.py -v`

### 21.3 阶段 2：参数生成器

- [ ] **任务 2.1：新增方向锥参数模块**
  - 新增：`src/ackermann_gcs_pkg/directional_curvature_parameters.py`
  - 动作：定义 `DirectionalCurvatureSegmentParams`，字段包括：
    - `edge_id: int | str`
    - `t: np.ndarray`
    - `n: np.ndarray`
    - `rho: float`
    - `theta_max: float`
    - `eta: float`
    - `kappa_max: float`
    - `risk_flags: tuple[str, ...]`
  - 验收：`t` 和 `n` 为二维单位向量且近似正交；`rho > 0`；`eta = tan(theta_max)`。

- [ ] **任务 2.2：实现粗路径预处理**
  - 修改：`src/ackermann_gcs_pkg/directional_curvature_parameters.py`
  - 动作：实现 `preprocess_reference_path(...)`，完成去重、极短边过滤、按 `ds_ref = max(0.5, 0.1 * R_min)` 重采样、位置切线平滑、`theta` unwrap、可选 gear 分段标记。
  - 验收：输入 `(x, y, theta)` 或 `(x, y, theta, gear)` 序列时，输出连续路径点、平滑切线和段索引映射。

- [ ] **任务 2.3：实现 support 宽度计算与缓存**
  - 修改：`src/ackermann_gcs_pkg/directional_curvature_parameters.py`
  - 动作：实现 `compute_support_width(region, direction)` 和缓存键 `(region_id, rounded_direction)`。
  - 验收：当区域有 `vertices` 时使用顶点投影；否则使用 Drake `HPolyhedron` support LP；相同区域和量化方向重复查询命中缓存。

- [ ] **任务 2.4：实现 edge-wise 参数构造**
  - 修改：`src/ackermann_gcs_pkg/directional_curvature_parameters.py`
  - 动作：实现 `DirectionalCurvatureParameterBuilder.build_for_edges(...)`，为每条 GCS edge 或等价 Bezier 段生成 `DirectionalCurvatureSegmentParams`。
  - 公式：
    - `t_k = smooth_tangent_from_reference_path(k)`，区域方向只做一致性检查和 fallback。
    - `rho_k = min(beta * ell_path,k, alpha * W_parallel,k)`。
    - `theta_width,k = atan(width_mu * W_lateral,k / max(W_parallel,k, eps))`。
    - `theta_max,k = clamp(local_turn_k + theta_margin, theta_min, min(theta_width,k, theta_abs_max))`。
  - 验收：每个可加约束的 edge 都有参数；缺少参数的 edge 不应静默使用错误参数，必须显式进入 fallback 或跳过并记录风险。

- [ ] **任务 2.5：实现高风险边增强**
  - 修改：`src/ackermann_gcs_pkg/directional_curvature_parameters.py`
  - 动作：当 `rho_k < rho_warning_ratio * R_min`、`W_parallel,k / ell_path,k` 过小、`angle(t_path,k, t_region,k) > 45 deg` 时，标记高风险并按需计算 `W_overlap,k`。
  - 验收：默认不全图计算 overlap；只有高风险边触发 overlap LP；overlap 不可行时标记 `overlap_infeasible`。

- [ ] **任务 2.6：补充参数生成器单元测试**
  - 新增：`tests/unit/test_directional_curvature_parameters.py`
  - 动作：覆盖单位向量生成、`rho_k` 公式、`theta_max,k` clamp、support 缓存、高风险边标记、overlap 触发条件。
  - 验证命令：`pytest tests/unit/test_directional_curvature_parameters.py -v`

### 21.4 阶段 3：Bezier/GCS 约束接口

- [ ] **任务 3.1：新增线性方向锥曲率约束 API**
  - 修改：`src/gcs_pkg/scripts/core/bezier.py`
  - 动作：新增 `addDirectionalCurvatureConstraint(edge_params, boundary_edge_ids=None, verbose=False)`。
  - 约束：对每个适用 edge 和每个一阶导控制点 `D_i` 添加：
    - `t_k^T D_i >= rho_k`
    - ` n_k^T D_i <= eta_k * t_k^T D_i`
    - `-n_k^T D_i <= eta_k * t_k^T D_i`
  - 约束：对每个适用 edge 和每个二阶导控制点 `A_j` 添加：
    - `sigma * n_k^T A_j + eta_k * tau * t_k^T A_j <= kappa_max * rho_k^2`
    - `sigma, tau in {-1, +1}`
  - 验收：新 API 只添加线性约束，不引入 `LorentzConeConstraint`；约束绑定记录在曲率约束列表中，能被现有移除逻辑或新增移除逻辑管理。

- [ ] **任务 3.2：补齐边界段策略**
  - 修改：`src/gcs_pkg/scripts/core/bezier.py`
  - 动作：支持 `boundary_edge_ids` 跳过起终点退化边，与 `addCurvatureHardConstraintForEdges(...)` 的边界语义保持一致。
  - 验收：起终点速度为 0 的场景不会因为 `rho_k > 0` 强制边界段不可行。

- [ ] **任务 3.3：补充约束构造测试**
  - 新增：`tests/unit/test_directional_curvature_bezier_constraints.py`
  - 动作：在可用 Drake 的环境中检查每条普通 edge 的线性约束数量：`3 * num_D + 4 * num_A`。
  - 验证命令：`pytest tests/unit/test_directional_curvature_bezier_constraints.py -v`
  - 环境说明：该测试依赖 Drake，应在 Ubuntu `iris-py3.12` 中运行；Windows 本机只运行不依赖 Drake 的参数生成器测试。

### 21.5 阶段 4：规划器集成

- [ ] **任务 4.1：扩展规划器输入**
  - 修改：`src/ackermann_gcs_pkg/ackermann_gcs_planner.py`
  - 动作：给 `plan_trajectory(...)` 增加可选参数 `reference_path=None` 和 `edge_reference_map=None`。
  - 验收：旧调用保持兼容；新模式下如果缺少 `reference_path`，必须回退到 `"hard"` 或返回清晰错误，不允许用起终点直线冒充完整参考路径。

- [ ] **任务 4.2：集成 direction_cone 模式**
  - 修改：`src/ackermann_gcs_pkg/ackermann_gcs_planner.py`
  - 动作：当 `curvature_constraint_mode == "direction_cone"` 时：
    - 不进入 `iterate_h_bar_prime(...)`。
    - 调用 `DirectionalCurvatureParameterBuilder` 生成 edge-wise 参数。
    - 调用 `bezier_gcs.addDirectionalCurvatureConstraint(...)`。
    - verbose 日志输出参数统计：edge 数量、rho 范围、theta 范围、高风险边数量、是否使用 overlap。
  - 验收：`direction_cone` 单次规划只执行一次 GCS 求解流程，不做 h_bar_prime 迭代修正。

- [ ] **任务 4.3：端到端编排传递粗路径**
  - 修改：`src/path_planner/support/gcs_optimizer.py`
  - 修改：`src/path_planner/planner.py`
  - 动作：将 Hybrid A* 输出的 `path` 作为 `reference_path` 传给 `AckermannGCSPlanner.plan_trajectory(...)`。
  - 验收：`path_planner` 编排层能够开启 `"direction_cone"` 模式，不丢失粗路径局部方向信息。

- [ ] **任务 4.4：脚本配置入口**
  - 修改：`scripts/batch_test_curvature_constraint.py`
  - 修改：`scripts/hybrid_astar_gcs_planner.py`
  - 动作：允许通过常量或命令行参数选择 `"none" / "hard" / "direction_cone"`。
  - 验收：同一脚本可以跑 hard baseline 和 direction_cone 对比。

### 21.6 阶段 5：验证与回归

- [ ] **任务 5.1：数学性质单元测试**
  - 新增：`tests/unit/test_directional_curvature_math.py`
  - 动作：构造满足方向锥约束的一阶/二阶导控制点，采样验证 `|kappa| <= kappa_max + tolerance`。
  - 验收：随机样本和边界样本均通过，覆盖 `theta=25 deg / 35 deg / 45 deg`。

- [ ] **任务 5.2：配置与参数生成测试**
  - 命令：`pytest tests/unit/test_directional_curvature_config.py tests/unit/test_directional_curvature_parameters.py tests/unit/test_directional_curvature_math.py -v`
  - 验收：Windows 本机可运行的纯 Python 测试通过。

- [ ] **任务 5.3：Ubuntu Drake 环境约束构造测试**
  - 命令：`pytest tests/unit/test_directional_curvature_bezier_constraints.py -v`
  - 环境：Ubuntu `iris-py3.12`
  - 验收：线性约束数量、边界 edge 跳过、缺参 fallback 均符合预期。

- [ ] **任务 5.4：批量场景对比**
  - 命令：`python scripts/batch_test_curvature_constraint.py`
  - 对比模式：`"hard"` 与 `"direction_cone"`
  - 验收指标：
    - direction_cone 的 GCS 求解次数不高于 hard 迭代模式。
    - 后验最大曲率不超过 `kappa_max + 1e-4`，或曲率违反量明确触发 fallback。
    - basic / narrow / s_curve / parking 场景有逐项成功率记录。

- [ ] **任务 5.5：端到端 smoke test**
  - 命令：`python scripts/hybrid_astar_gcs_planner.py`
  - 环境：Ubuntu `iris-py3.12`
  - 验收：A* -> corridor -> IRIS/IrisZo -> direction_cone GCS -> 轨迹评估链路跑通，并输出曲率、速度、加速度、工作空间约束报告。

### 21.7 阶段 6：fallback 与诊断

- [ ] **任务 6.1：实现高风险 fallback 策略**
  - 修改：`src/ackermann_gcs_pkg/ackermann_gcs_planner.py`
  - 动作：当参数生成失败、overlap 不可行、direction_cone 求解失败或后验曲率违反超阈值时，按配置切换到 `"hard"` 模式。
  - 验收：fallback 只触发一次，不引入新的无限重试或单次规划内迭代循环。

- [ ] **任务 6.2：输出诊断报告**
  - 修改：`src/ackermann_gcs_pkg/ackermann_data_structures.py`
  - 修改：`src/ackermann_gcs_pkg/ackermann_gcs_planner.py`
  - 动作：在 `PlanningResult` 或 `TrajectoryReport` 中记录 direction_cone 参数摘要和 fallback 原因。
  - 验收：批量脚本可以统计高风险边数量、fallback 次数、rho/theta 分布和曲率违反量。

### 21.8 阶段 7：固定 gear 倒车支持

- [ ] **任务 7.1：保留粗路径 gear 数据**
  - 修改：`src/path_planner/support/gcs_optimizer.py`
  - 修改：`src/path_planner/planner.py`
  - 动作：把 A* motion primitive 的 `gear` 保留到 `reference_path` 或单独的 `reference_gears`。
  - 验收：同一 GCS 段内不跨 gear change；跨 gear change 时切段或标记不可用。

- [ ] **任务 7.2：实现 gear-aware heading 约束**
  - 修改：`src/ackermann_gcs_pkg/ackermann_bezier_gcs.py`
  - 修改：`src/ackermann_gcs_pkg/rotation_matrix_heading_constraint.py`
  - 动作：支持 `gear = +1` 时 `r'(s)` 平行 `h(theta)`，`gear = -1` 时 `r'(s)` 平行 `-h(theta)`。
  - 验收：倒车参考路径不会被 heading 约束错误排斥。

- [ ] **任务 7.3：增加倒车成本**
  - 修改：`src/ackermann_gcs_pkg/ackermann_gcs_planner.py`
  - 动作：增加 reverse cost 和 gear switch cost，仅用于固定 gear 序列统计和轻微偏置，不让 GCS 自主选择 gear。
  - 验收：固定 gear 倒车场景能规划，且换挡次数与 A* 输出一致。

### 21.9 阶段 8：gear-layered GCS 后续方案

- [ ] **任务 8.1：设计 gear-layered 图结构**
  - 文档：`docs/curvature_constraint_direction_cone_design.md`
  - 动作：单独补充 gear-layered GCS 设计，不在 Level 2 实现中混入。
  - 验收：明确 forward/reverse 区域复制、跨层换挡边、换挡成本和 heading 约束的接口。

- [ ] **任务 8.2：实现 gear-layered 原型**
  - 修改：`src/gcs_pkg/scripts/core/bezier.py`
  - 修改：`src/ackermann_gcs_pkg/ackermann_gcs_planner.py`
  - 动作：在完成固定 gear 倒车并通过验证后，再实现 GCS 自主选择前进/倒车/换挡。
  - 验收：parking 场景中 GCS 能选择必要换挡，且曲率、速度、加速度和工作空间约束均通过后验检查。

### 21.10 最终验收清单

- [ ] `curvature_constraint_mode="direction_cone"` 可作为独立模式启用。
- [ ] 新模式不依赖 `h_bar_prime`，不调用 `iterate_h_bar_prime(...)`。
- [ ] 方向锥参数按 edge 或等价 Bezier 段绑定，缺参不会静默错配。
- [ ] 新增曲率约束为线性约束，满足凸性要求。
- [ ] 后验曲率、速度、加速度和工作空间约束报告完整。
- [ ] hard Lorentz 约束保留为 fallback。
- [ ] Windows 本机完成纯 Python 单元测试；Ubuntu `iris-py3.12` 完成 Drake 相关单元测试和端到端脚本验证。
