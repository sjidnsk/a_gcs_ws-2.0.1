# 无 h_bar_prime 迭代的 Ackermann-GCS 凸曲率约束设计报告

> 生成日期：2026-05-13  
> 目标：在不依赖 A* 路径、不依赖 `h_bar_prime` 迭代的前提下，设计一种数学上可证明、约束上保持凸性、工程上可嵌入当前 GCS 框架的 Ackermann 曲率约束方案。

---

## 1. 问题背景

当前项目中的曲率硬约束主要位于：

- `src/gcs_pkg/scripts/core/bezier.py`
  - `BezierGCS.addCurvatureHardConstraint(...)`
  - `BezierGCS.addCurvatureHardConstraintForEdges(...)`
  - `BezierGCS.removeCurvatureHardConstraints(...)`
- `src/ackermann_gcs_pkg/h_bar_prime_iteration.py`
  - `iterate_h_bar_prime(...)`
- `src/ackermann_gcs_pkg/ackermann_gcs_planner.py`
  - `AckermannGCSPlanner.plan_trajectory(...)`

当前曲率约束的核心形式是：

```text
||Q_j||_2 <= C
C = kappa_max * rho_min^2
rho_min = min_velocity * h_bar_prime * h_bar_prime_safety_factor
```

其中 `Q_j` 是 Bezier/B-spline 空间轨迹二阶导控制点：

```text
Q_j = n(n-1) * (P_{j+2} - 2P_{j+1} + P_j)
```

该形式的数学起点是正确的：

```text
|kappa(s)| <= ||r''(s)|| / ||r'(s)||^2
```

如果能保证：

```text
||r'(s)|| >= rho_min
||r''(s)|| <= kappa_max * rho_min^2
```

则一定有：

```text
|kappa(s)| <= kappa_max
```

真正的问题在于：当前实现中的 `rho_min` 依赖 `h_bar_prime`，而 `h_bar_prime` 又来自求解后的轨迹时间缩放 `h'(s)`。这造成了一个固定点式迭代问题：

```text
先求解轨迹 -> 估计 h_bar_prime -> 加曲率约束 -> 再求解 -> 再估计 h_bar_prime
```

这条链路有三个根本缺陷：

1. `h_bar_prime` 不是曲率公式的几何必要量，而是时间参数化的派生量。
2. `h_bar_prime` 是解的结果，却被用来定义下一次优化的可行域。
3. 即使均值 `h_bar_prime` 收敛，也不能严格推出全局或局部的 `||r'(s)||` 下界。

因此，本报告的目标不是继续修补 `h_bar_prime` 迭代，而是从第一性原理重新设计曲率约束。

---

## 2. PyDrake 官方能力与边界

本方案参考 PyDrake/Drake 官方文档中的相关能力：

- `GraphOfConvexSets` / `GraphOfConvexSets::Edge`
  - 支持在边上添加约束与成本。
  - 当前项目已经通过 `edge.AddConstraint(Binding[Constraint](..., edge.xu()))` 使用这一机制。
  - 官方文档：<https://drake.mit.edu/doxygen_cxx/classdrake_1_1geometry_1_1optimization_1_1_graph_of_convex_sets_1_1_edge.html>

- `LorentzConeConstraint`
  - 定义线性表达式 `A x + b` 落入 Lorentz cone：

    ```text
    z[0] >= sqrt(z[1]^2 + ... + z[n]^2)
    ```

  - 可表达：

    ```text
    ||linear_expression(x)||_2 <= constant
    ```

  - 官方文档：<https://drake.mit.edu/doxygen_cxx/classdrake_1_1solvers_1_1_lorentz_cone_constraint.html>

- `BsplineTrajectory`
  - 支持 `MakeDerivative(...)`、`control_points()` 等接口。
  - 当前项目已经用 `BsplineTrajectory_[Expression]` + `DecomposeLinearExpressions(...)` 构造导数控制点的线性表达式。
  - 官方文档：<https://drake.mit.edu/pydrake/pydrake.trajectories.html>

- `GcsTrajectoryOptimization`
  - Drake 官方轨迹优化接口中，对复杂导数约束常采用 convex surrogate 与 restriction 分层处理。
  - 这说明官方路线并不要求把所有非线性动力学精确塞入单个凸松弛。
  - 官方文档：<https://drake.mit.edu/pydrake/pydrake.planning.html>

- `PiecewiseConstantCurvatureTrajectory`
  - Drake 官方存在以弧长为自然参数的平面常曲率轨迹表示。
  - 它从概念上支持一个重要判断：车辆曲率首先是几何量，不应强依赖时间缩放变量。
  - 官方文档：<https://drake.mit.edu/doxygen_cxx/classdrake_1_1trajectories_1_1_piecewise_constant_curvature_trajectory.html>

- `Toppra`
  - Drake 官方用于路径已知后的时间参数化。
  - 支持给定 `s_dot_start`、`s_dot_end` 等边界速度条件。
  - 这说明“停车速度为 0”更适合在后处理时间参数化中处理，而不是让空间路径导数退化为 0。
  - 官方文档：<https://drake.mit.edu/doxygen_cxx/classdrake_1_1multibody_1_1_toppra.html>

官方文档中未发现“直接把精确 Ackermann 曲率硬约束完整凸化后放入 GCS”的现成案例。因此，本方案采用数学上严格的凸充分条件，而不是伪装成精确等价约束。

---

## 3. 第一性原理推导

### 3.1 曲率是空间几何量

设空间路径为：

```text
r(s) = [x(s), y(s)]^T
```

几何曲率为：

```text
kappa(s) = det(r'(s), r''(s)) / ||r'(s)||^3
```

其中：

```text
det(a, b) = a_x b_y - a_y b_x
```

由 Cauchy-Schwarz 不等式：

```text
|det(r', r'')| <= ||r'|| * ||r''||
```

所以：

```text
|kappa(s)| <= ||r''(s)|| / ||r'(s)||^2
```

因此，如果能保证：

```text
||r'(s)|| >= rho
||r''(s)|| <= kappa_max * rho^2
```

则必然有：

```text
|kappa(s)| <= kappa_max
```

这就是当前 `addCurvatureHardConstraint(...)` 的合理数学基础。

### 3.2 当前方案的问题

当前方案试图通过：

```text
rho = min_velocity * h_bar_prime
```

把物理速度下界转换为空间导数下界。

但真实关系是：

```text
v(t) = dr/dt = dr/ds * ds/dt
h(s) = t
h'(s) = dt/ds
ds/dt = 1 / h'(s)
v = r'(s) / h'(s)
```

所以：

```text
||r'(s)|| = ||v|| * h'(s)
```

若要从速度下界推出空间导数下界，需要同时拥有：

```text
||v(s)|| >= v_min
h'(s) >= h_min
```

当前项目中的 `h_bar_prime` 是均值估计，不是逐点下界。因此：

```text
h_bar_prime = average(h'(s))
```

不能推出：

```text
h'(s) >= h_bar_prime
```

也不能推出：

```text
||r'(s)|| >= min_velocity * h_bar_prime
```

这正是 `h_bar_prime` 迭代的根源问题。

### 3.3 需要的凸约束形态

我们真正需要的是一个不依赖求解结果的、可提前写入 GCS 的空间导数下界：

```text
||r'(s)|| >= rho
```

但是集合：

```text
{d | ||d|| >= rho}
```

不是凸集，不能直接作为 GCS convex set 约束。

因此必须使用凸充分条件。

---

## 4. 核心设计：航向锥 + 空间进度下界 + 二阶导锥约束

### 4.1 固定航向锥

对每个离散航向 `theta_m` 定义：

```text
e_m = [cos(theta_m), sin(theta_m)]^T
n_m = [-sin(theta_m), cos(theta_m)]^T
```

其中 `e_m` 是航向中心方向，`n_m` 是其法向方向。

给定锥半角 `alpha`，要求某段路径的一阶导数始终位于该航向锥内：

```text
e_m^T r'(s) >= rho
|n_m^T r'(s)| <= tan(alpha) * e_m^T r'(s)
```

等价写成线性不等式：

```text
e_m^T r'(s) >= rho
n_m^T r'(s) <= tan(alpha) * e_m^T r'(s)
-n_m^T r'(s) <= tan(alpha) * e_m^T r'(s)
```

由于 `theta_m`、`alpha`、`rho` 都是固定参数，上述全部是仿射约束。

### 4.2 施加到导数控制点

当前 `BezierGCS` 使用 Bezier/B-spline 控制点构造轨迹。导数曲线满足凸包性质：

```text
r'(s) lies in convex hull of first-derivative control points D_i
r''(s) lies in convex hull of second-derivative control points Q_j
```

因此，只要对所有一阶导控制点 `D_i` 施加：

```text
e_m^T D_i >= rho
n_m^T D_i <= tan(alpha) * e_m^T D_i
-n_m^T D_i <= tan(alpha) * e_m^T D_i
```

就能推出对任意 `s`：

```text
e_m^T r'(s) >= rho
```

进一步由于：

```text
||r'(s)|| >= e_m^T r'(s)
```

得到：

```text
||r'(s)|| >= rho
```

### 4.3 二阶导 Lorentz 锥

对所有二阶导控制点 `Q_j` 施加：

```text
||Q_j||_2 <= kappa_max * rho^2
```

由于 `r''(s)` 位于 `Q_j` 的凸包内，且欧氏范数是凸函数，有：

```text
||r''(s)|| <= max_j ||Q_j|| <= kappa_max * rho^2
```

### 4.4 曲率保证

综合：

```text
||r'(s)|| >= rho
||r''(s)|| <= kappa_max * rho^2
```

得到：

```text
|kappa(s)| <= ||r''(s)|| / ||r'(s)||^2
           <= (kappa_max * rho^2) / rho^2
           = kappa_max
```

这构成完整数学闭环。

---

## 5. 为什么不能直接使用 A* 路径方向

用户已明确限制：

> A* 路径只能用于生成局部走廊和提供 IrisZo 算法的种子点。

因此，本方案不从 A* 路径中读取：

- 节点顺序；
- 每条边方向；
- 局部切线方向；
- 曲率估计；
- 进度下界 `rho`。

约束方向来自全局离散航向集合：

```text
theta_m = 2*pi*m / heading_bin_count
```

而不是来自 A*。

这样做的好处是：

1. 曲率约束不继承 A* 的离散路径偏差。
2. GCS 仍可在凸区域图上重新选择路径。
3. 同一个物理区域可以重复出现。
4. 可以表达绕行、回环、调头等 A* 单一路径不愿表达的结构。

---

## 6. 图结构设计：layer-expanded heading GCS

### 6.1 为什么需要 layer

用户指出：

> 理论上来说同一个节点可以重复出现在选择的节点路径上。

当前普通 region 图中，如果每个物理区域只出现一次，那么一条简单路径通常不能重复访问同一节点。为了允许同一个物理区域重复出现，需要把图展开为“时间/段数层”：

```text
原始节点:
region_i

提升节点:
(layer_l, region_i, heading_bin_m, rho_bin_k)
```

同一个 `region_i` 可以出现在多个 `layer_l` 中，因此被选择路径可以多次经过同一物理区域。

### 6.2 提升节点定义

每个 GCS 顶点对应：

```text
V(l, i, m, k)
```

其中：

- `l`：段序号，`0 <= l < max_gcs_segments`
- `i`：原始凸区域编号
- `m`：航向 bin 编号
- `k`：空间进度下界 bin 编号

每个顶点的 convex set 仍然是：

```text
region_i^(order+1) x time_scaling_set
```

也就是说，几何可行域不因 heading/rho 标签变成非凸。heading/rho 只决定该顶点对应的边上附加哪些线性约束和 Lorentz 锥约束。

### 6.3 边连接规则

允许连接：

```text
V(l, i, m, k) -> V(l+1, j, n, q)
```

当且仅当：

1. `region_i` 与 `region_j` 相交，或 `i == j`；
2. `abs(wrap(theta_m - theta_n)) <= heading_transition_limit`；
3. `l + 1 < max_gcs_segments`；
4. 如果启用 full-dimensional overlap，则沿用现有 `findEdgesViaFullDimensionOverlaps()` 的判据。

推荐默认：

```text
heading_transition_limit = 2 * heading_cone_half_angle
```

这样相邻段的航向锥有交集，配合当前 `continuity >= 1` 时的 C1 连续性更容易可行。

### 6.4 source/target 连接

source 连接到：

```text
V(0, i, m, k)
```

条件：

- `source.position` 位于 `region_i`；
- `source.heading` 落入 heading bin `m` 的锥内。

target 可以从任意 layer 连接：

```text
V(l, i, m, k) -> target
```

条件：

- `target.position` 位于 `region_i`；
- `target.heading` 落入 heading bin `m` 的锥内。

这样目标可以在较短层数处提前结束，不强制走满 `max_gcs_segments`。

### 6.5 是否允许原地停留

可以允许：

```text
i == j
```

这代表同一物理区域内继续生成下一段轨迹。由于每段仍有：

```text
e_m^T r'(s) >= rho
```

所以它不是几何上的“原地不动”，而是在同一个凸区域内继续前进或转向。

---

## 7. 参数选择

### 7.1 heading_bin_count

推荐默认：

```python
heading_bin_count = 24
```

对应航向分辨率：

```text
delta_theta = 2*pi / 24 = 15 deg
```

更窄的 heading bin 能降低保守性，但会增大图规模。

### 7.2 heading_cone_half_angle

推荐默认：

```python
heading_cone_half_angle = 0.75 * (2*pi / heading_bin_count)
```

即对 24 个 bin：

```text
alpha = 11.25 deg
```

需要满足：

```text
alpha < pi/2
```

工程上建议：

```text
5 deg <= alpha <= 25 deg
```

### 7.3 rho 的物理尺度

`rho` 是每个归一化段参数 `s in [0,1]` 上的空间导数下界：

```text
||dr/ds|| >= rho
```

它不是速度，不是时间导数，也不应由 `h_bar_prime` 推导。

一个合理的尺度来自最小转弯半径：

```text
R_min = 1 / kappa_max
rho_base = R_min * delta_theta
```

其含义是：如果路径以最大曲率转过一个 heading bin，需要的弧长尺度约为 `R_min * delta_theta`。

推荐：

```python
rho_multipliers = (0.5, 1.0, 2.0)
rho_values = [rho_base * m for m in rho_multipliers]
```

多个 `rho` bin 的意义：

- 小 `rho`：更容易在狭窄区域中转向，但二阶导上界更紧；
- 大 `rho`：允许更大的二阶导，但要求路径在该段有更明显的空间推进。

### 7.4 max_gcs_segments

推荐默认：

```python
max_gcs_segments = 2 * len(workspace_regions)
```

如果环境复杂、需要调头或重复进入区域，可以增大到：

```python
max_gcs_segments = 3 * len(workspace_regions)
```

---

## 8. 接口设计

### 8.1 CurvatureConstraintMode

当前：

```python
class CurvatureConstraintMode(Enum):
    NONE = "none"
    HARD = "hard"
    TURNING_RADIUS = "turning_radius"
```

建议新增：

```python
class CurvatureConstraintMode(Enum):
    NONE = "none"
    HARD = "hard"
    DIRECTIONAL_CONVEX = "directional_convex"
    TURNING_RADIUS = "turning_radius"
```

其中：

- `hard`：保留旧 `h_bar_prime` 方案，作为兼容和实验对照；
- `directional_convex`：新推荐方案；
- `turning_radius`：旧模式如果已废弃，可继续保留但不推荐。

### 8.2 TrajectoryConstraints

建议在 `src/ackermann_gcs_pkg/ackermann_data_structures.py` 的 `TrajectoryConstraints` 中增加：

```python
heading_bin_count: int = 24
heading_cone_half_angle: Optional[float] = None
rho_multipliers: Tuple[float, ...] = (0.5, 1.0, 2.0)
max_gcs_segments: Optional[int] = None
curvature_validation_samples: int = 300
```

语义：

- `heading_bin_count`：离散航向数量；
- `heading_cone_half_angle`：航向锥半角；`None` 时自动使用 `0.75 * 2*pi / heading_bin_count`；
- `rho_multipliers`：基于 `rho_base = R_min * delta_theta` 的倍率；
- `max_gcs_segments`：layer 展开层数；`None` 时使用 `2 * len(workspace_regions)`；
- `curvature_validation_samples`：求解后真实曲率采样验证点数。

### 8.3 新增 GCS 类

建议新增：

```text
src/ackermann_gcs_pkg/directional_ackermann_bezier_gcs.py
```

核心类：

```python
class DirectionalAckermannBezierGCS(AckermannBezierGCS):
    def __init__(
        self,
        regions,
        vehicle_params,
        bezier_config=None,
        heading_bin_count=24,
        heading_cone_half_angle=None,
        rho_multipliers=(0.5, 1.0, 2.0),
        max_gcs_segments=None,
    ):
        ...

    def addDirectionalConvexCurvatureConstraint(
        self,
        max_curvature: float,
    ) -> None:
        ...
```

也可以不继承 `AckermannBezierGCS`，而是增加一个 factory 构造 `BezierGCS`。但为了最小化嵌入成本，推荐继承当前类。

### 8.4 与现有方法的对应关系

旧方法：

```python
bezier_gcs.addCurvatureHardConstraint(
    max_curvature=constraints.max_curvature,
    min_velocity=constraints.min_velocity,
    h_bar_prime=constraints.h_bar_prime,
    h_bar_prime_safety_factor=constraints.h_bar_prime_safety_factor,
)
```

新方法：

```python
bezier_gcs.addDirectionalConvexCurvatureConstraint(
    max_curvature=constraints.max_curvature,
)
```

新方法不接受：

- `min_velocity`
- `h_bar_prime`
- `h_bar_prime_safety_factor`

原因：

曲率是空间几何约束，不应依赖时间缩放均值。

### 8.5 Planner 分支

在 `AckermannGCSPlanner.plan_trajectory(...)` 中建议改为：

```python
if constraints.curvature_constraint_mode == "directional_convex":
    bezier_gcs = DirectionalAckermannBezierGCS(
        regions=workspace_regions,
        vehicle_params=self.vehicle_params,
        bezier_config=self.bezier_config,
        heading_bin_count=constraints.heading_bin_count,
        heading_cone_half_angle=constraints.heading_cone_half_angle,
        rho_multipliers=constraints.rho_multipliers,
        max_gcs_segments=constraints.max_gcs_segments,
    )
else:
    bezier_gcs = AckermannBezierGCS(
        regions=workspace_regions,
        vehicle_params=self.vehicle_params,
        bezier_config=self.bezier_config,
    )
```

然后：

```python
if constraints.curvature_constraint_mode == "directional_convex":
    bezier_gcs.addDirectionalConvexCurvatureConstraint(
        max_curvature=constraints.max_curvature,
    )
elif constraints.curvature_constraint_mode == "hard":
    # 保留旧 h_bar_prime 流程
    ...
```

### 8.6 返回结果

当前 `PlanningResult` 中已有：

- `curvature_violation`
- `curvature_stats`
- `h_bar_prime_iteration_result`

建议：

```text
directional_convex 模式下:
h_bar_prime_iteration_result = None
curvature_violation 必须由真实采样曲率验证得出
```

可新增调试信息：

```python
curvature_constraint_metadata = {
    "mode": "directional_convex",
    "heading_bin_count": ...,
    "heading_cone_half_angle": ...,
    "rho_values": ...,
    "max_gcs_segments": ...,
}
```

如果不希望改动数据结构，可先放入结果字典或 verbose 日志。

---

## 9. 约束构造细节

### 9.1 一阶导控制点

当前代码已有类似写法：

```python
u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
```

对每个 `D_i`：

```python
A_ctrl = DecomposeLinearExpressions(D_i, self.u_vars)
```

如果：

```text
D_i = A_ctrl @ x
```

则线性约束可以构造为：

```text
e^T A_ctrl x >= rho
n^T A_ctrl x <= tan(alpha) * e^T A_ctrl x
-n^T A_ctrl x <= tan(alpha) * e^T A_ctrl x
```

等价矩阵形式：

```text
-(e^T A_ctrl) x <= -rho
(n^T A_ctrl - tan(alpha) * e^T A_ctrl) x <= 0
(-n^T A_ctrl - tan(alpha) * e^T A_ctrl) x <= 0
```

使用 `LinearConstraint` 即可。

### 9.2 二阶导控制点

当前已有：

```python
u_path_ddot = self.u_r_trajectory.MakeDerivative(2).control_points()
```

对每个 `Q_j`：

```python
A_ctrl = DecomposeLinearExpressions(Q_j, self.u_vars)
C = max_curvature * rho ** 2
H = np.vstack([
    np.zeros((1, A_ctrl.shape[1])),
    A_ctrl,
])
b = np.zeros(A_ctrl.shape[0] + 1)
b[0] = C
curvature_con = LorentzConeConstraint(H, b)
```

这与当前 `addCurvatureHardConstraint(...)` 的 Lorentz cone 写法一致，只是 `C` 的来源从：

```text
kappa_max * (min_velocity * h_bar_prime)^2
```

改为：

```text
kappa_max * rho^2
```

### 9.3 约束绑定到对应 heading/rho 顶点

普通 `BezierGCS` 当前把同一组曲率约束加到所有非 source 边上。

新方案中，不同 heading/rho 顶点需要不同约束。因此不能简单地全局复用一组 `self.curvature_constraints`。

建议维护：

```python
self.directional_curvature_constraints_by_vertex_key = {
    (layer, region_id, heading_id, rho_id): [...],
}
```

或者更简单：

```python
self.directional_edge_constraints = []
```

在添加边时，根据 `edge.u()` 对应的 lifted vertex metadata 选择对应的 `theta_m` 与 `rho_k`，即时构造并绑定。

### 9.4 metadata

需要维护：

```python
@dataclass(frozen=True)
class LiftedVertexKey:
    layer: int
    region_id: int
    heading_id: int
    rho_id: int
```

以及：

```python
self.vertex_key_by_vertex_id: Dict[int, LiftedVertexKey]
self.vertex_by_key: Dict[LiftedVertexKey, GraphOfConvexSets.Vertex]
self.theta_by_heading_id: Dict[int, float]
self.rho_by_rho_id: Dict[int, float]
```

由于 Drake Vertex 对象未必适合作为长期稳定 key，可以使用：

```python
id(vertex)
```

或顶点名称字符串作为映射 key。

---

## 10. 起终点速度与停车问题

当前 `addSourceTargetWithHeading(...)` 支持传入 `source_velocity` / `target_velocity`。如果速度为 0，当前代码会识别 `source_is_v0`、`target_is_v0`，并对 heading 约束做退化处理。

需要特别注意：

```text
车辆停车速度 v = 0
不等于
空间路径导数 r'(s) = 0
```

如果让 `r'(s)=0`，几何曲率公式：

```text
kappa = det(r', r'') / ||r'||^3
```

在该点无定义。

因此新方案应采用以下语义：

1. GCS 几何路径阶段：
   - 固定起终点位置；
   - 固定起终点几何航向；
   - 不把停车速度转换为空间导数为 0。

2. 时间参数化阶段：
   - 允许 `ds/dt = 0`；
   - 使用类似 Drake `Toppra` 的思路处理起终点速度为 0。

当前项目未必已经接入 `Toppra`，但概念上应避免把停车条件错误塞入几何曲率约束。

---

## 11. 与现有 h_bar_prime 方案的关系

### 11.1 保留旧方案

建议暂时保留：

```python
curvature_constraint_mode = "hard"
```

作为旧方案对照。

但文档与默认配置应推荐：

```python
curvature_constraint_mode = "directional_convex"
```

### 11.2 弃用 h_bar_prime 迭代

在 `directional_convex` 模式下：

- 不调用 `iterate_h_bar_prime(...)`
- 不调用 `removeCurvatureHardConstraints(...)`
- 不读取 `h_bar_prime`
- 不读取 `h_bar_prime_safety_factor`
- 不读取 `max_h_bar_prime_iterations`

如果用户同时设置：

```python
curvature_constraint_mode = "directional_convex"
h_bar_prime is not None
```

建议输出 warning：

```text
h_bar_prime is ignored in directional_convex mode.
```

---

## 12. 可行性分析

### 12.1 凸性

新增约束类型只有：

- 线性不等式；
- Lorentz cone；
- 现有线性连续性约束；
- 现有 convex cost。

因此可嵌入当前 GCS convex relaxation。

### 12.2 数学保证

只要求解器返回满足约束的解，且 Bezier/B-spline 凸包性质成立，就有：

```text
max_s |kappa(s)| <= kappa_max
```

这是一个充分条件，不是近似采样条件。

采样验证仍然必要，但它是工程验收，不是数学保证来源。

### 12.3 与重复节点兼容

layer-expanded graph 允许：

```text
region_i at layer 2
region_i at layer 5
```

同时出现在选择路径上。

这解决了“同一个节点可以重复出现在选择路径上”的理论要求。

### 12.4 与 A* 角色兼容

A* 只影响 `workspace_regions` 的生成，不影响：

- heading bin；
- rho bin；
- 约束方向；
- GCS 选择路径；
- 重复节点机制。

---

## 13. 主要风险

### 13.1 保守性风险

该方案是充分条件，不是必要条件。可能存在真实曲率可行的路径被排除。

缓解：

- 增大 `heading_bin_count`；
- 增大 `max_gcs_segments`；
- 增加 `rho_multipliers`；
- 放宽 `heading_cone_half_angle`，但必须保持 `< pi/2`；
- 保留旧 `hard` 模式作为实验对照。

### 13.2 图规模风险

节点数量约为：

```text
max_gcs_segments * num_regions * heading_bin_count * len(rho_values)
```

边数量还会再乘以区域连接关系和航向转移数量。

缓解：

- 只对 IrisZo 局部走廊中的区域展开；
- 对 heading transition 做剪枝；
- 初期使用 `heading_bin_count=16` 或 `24`；
- 初期使用 `rho_multipliers=(0.5, 1.0)`；
- 对 target 允许提前结束，减少无效深层路径。

### 13.3 起终点速度语义风险

如果继续把 `source_velocity=0` 转换为 `r'(0)=0`，会破坏曲率定义。

缓解：

- 在 `directional_convex` 模式下，起终点速度为 0 时只用于后续时间参数化；
- GCS 阶段只约束位置和几何航向。

### 13.4 当前 SolvePathWithConstraints 重复添加约束

当前 `AckermannBezierGCS.SolvePathWithConstraints(...)` 每次调用时会把 custom heading constraints 添加到边上。多次求解可能造成重复约束累积。

缓解：

- 在实现新模式时让 heading/custom constraints 只绑定一次；
- 或增加 `_constraints_applied` 标记；
- 或在构图完成后立即绑定，不在每次 solve 前绑定。

---

## 14. 测试与验收标准

### 14.1 单元测试

新增测试建议：

1. 一阶导锥约束测试
   - 构造随机导数控制点；
   - 验证所有采样点 `r'(s)` 都满足 `e^T r'(s) >= rho`。

2. 二阶导 Lorentz 约束测试
   - 构造满足 `||Q_j|| <= kappa_max * rho^2` 的控制点；
   - 验证采样 `||r''(s)||` 不超过上界。

3. 曲率保证测试
   - 构造完整 Bezier 曲线；
   - 采样验证 `max(abs(kappa)) <= kappa_max + tolerance`。

4. 重复 region 测试
   - 构造一个必须或允许重复经过同一物理区域的简单场景；
   - 验证返回路径包含相同 `region_id` 的不同 layer。

5. 不调用 h_bar_prime 测试
   - `curvature_constraint_mode="directional_convex"`；
   - 验证 `iterate_h_bar_prime(...)` 不被调用；
   - 验证 `h_bar_prime_iteration_result is None`。

6. A* 解耦测试
   - 改变 A* 路径顺序但保持相同 `workspace_regions`；
   - 验证曲率约束构造不变。

### 14.2 集成测试

使用现有脚本扩展：

- `scripts/batch_test_curvature_constraint.py`
- `scripts/hybrid_astar_gcs_planner.py`

新增对比模式：

```text
none
hard
directional_convex
```

对比指标：

- 成功率；
- 最大曲率；
- 曲率违反量；
- 求解时间；
- 图节点数；
- 图边数；
- GCS rounding 成功率。

### 14.3 验收标准

`directional_convex` 模式满足：

```text
1. 不需要 h_bar_prime。
2. 不执行 h_bar_prime 迭代。
3. 所有新增硬约束均为 LinearConstraint 或 LorentzConeConstraint。
4. A* 不提供曲率约束方向。
5. 同一物理 region 可在不同 layer 中重复出现。
6. 真实曲率采样无明显违反，允许 solver tolerance 级误差。
```

---

## 15. 建议实施步骤

### Phase 1：数据结构和配置

修改：

```text
src/ackermann_gcs_pkg/ackermann_data_structures.py
```

新增：

- `CurvatureConstraintMode.DIRECTIONAL_CONVEX`
- `TrajectoryConstraints.heading_bin_count`
- `TrajectoryConstraints.heading_cone_half_angle`
- `TrajectoryConstraints.rho_multipliers`
- `TrajectoryConstraints.max_gcs_segments`
- `TrajectoryConstraints.curvature_validation_samples`

### Phase 2：新增 lifted graph GCS

新增：

```text
src/ackermann_gcs_pkg/directional_ackermann_bezier_gcs.py
```

实现：

- heading bins；
- rho bins；
- layer-expanded vertices；
- layer-to-layer edges；
- source/target gating；
- vertex metadata。

### Phase 3：新增凸曲率约束

实现：

```python
addDirectionalConvexCurvatureConstraint(max_curvature)
```

内部构造：

- 一阶导线性锥约束；
- 二阶导 Lorentz 锥约束；
- 按 lifted vertex 的 heading/rho metadata 绑定到边。

### Phase 4：Planner 集成

修改：

```text
src/ackermann_gcs_pkg/ackermann_gcs_planner.py
```

增加：

```python
if constraints.curvature_constraint_mode == "directional_convex":
    ...
```

确保：

- 不调用 `iterate_h_bar_prime(...)`；
- 不调用旧 `addCurvatureHardConstraint(...)`；
- 保留现有 cost 添加流程；
- 保留现有 `TrajectoryEvaluator` 曲率验证。

### Phase 5：幂等性修复

修复：

```text
src/ackermann_gcs_pkg/ackermann_bezier_gcs.py
```

目标：

- 避免 `SolvePathWithConstraints(...)` 每次调用重复绑定 heading/custom constraints。

### Phase 6：测试与对比

扩展：

```text
scripts/batch_test_curvature_constraint.py
```

加入：

```text
directional_convex
```

输出图规模、求解时间和曲率违反量。

---

## 16. 最终结论

`h_bar_prime` 迭代问题的根源不是参数调得不够好，而是把时间缩放的求解结果用于定义空间曲率可行域。它本质上不是一个稳健的凸建模方式。

推荐方案是：

```text
使用航向离散和 layer-expanded GCS 表达路径拓扑；
使用固定航向锥给出凸的一阶空间导数下界；
使用 Lorentz cone 限制二阶空间导数；
从而在 GCS 内得到严格凸、可证明的曲率充分条件。
```

该方案满足：

- 数学闭环；
- 可嵌入当前 PyDrake GCS 框架；
- 不依赖 A* 路径方向；
- 允许同一物理节点重复出现；
- 不需要 `h_bar_prime` 迭代；
- 保留求解后真实曲率验证作为工程安全网。

推荐把它作为新的默认 Ackermann 曲率硬约束模式：

```python
curvature_constraint_mode = "directional_convex"
```

