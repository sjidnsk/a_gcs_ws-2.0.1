# IrisZo 算法技术文档

## 1. 算法概述

**IrisZo**（Iterative Regional Inflation by Semidefinite Programming - Zero Order）是 Drake 几何优化库中的一种碰撞自由凸区域生成算法。该算法仅依赖碰撞检测器（zero-order information），在机器人配置空间中构造概率无碰撞的多面体区域。

### 1.1 核心思想

IrisZo 使用**并行零阶优化策略**，在配置空间中围绕给定的初始椭球体逐步膨胀，生成尽可能大的碰撞自由 H-多面体（HPolyhedron）。"零阶"意味着算法仅需碰撞检测器的布尔查询结果，无需梯度或高阶信息。

### 1.2 学术引用

> P. Werner, T. Cohn*, R. H. Jiang*, T. Seyde, M. Simchowitz, R. Tedrake, and D. Rus,  
> "Faster Algorithms for Growing Collision-Free Convex Polytopes in Robot Configuration Space,"  
> * Denotes equal contribution.  
> 论文链接: https://groups.csail.mit.edu/robotics-center/public_papers/Werner24.pdf

---

## 2. 数学基础

### 2.1 概率无碰撞保证

IrisZo 生成的多面体 $P$ 是**概率无碰撞**的，即用户可以控制碰撞体积占比超过 $\varepsilon$ 的概率不超过 $\delta$：

$$\Pr\left[\frac{\lambda(P \setminus \mathcal{C}_{\text{free}})}{\lambda(P)} > \varepsilon\right] \leq \delta$$

其中：
- $P$：生成的多面体区域
- $\mathcal{C}_{\text{free}}$：无碰撞配置空间
- $\lambda(\cdot)$：Lebesgue 测度（体积）
- $\varepsilon$：允许的碰撞体积占比上限
- $\delta$：概率置信水平

**解读**：该保证意味着，在多面体 $P$ 中，碰撞部分占总体积的比例超过 $\varepsilon$ 的概率不超过 $\delta$。通过调节 $\varepsilon$ 和 $\delta$，可以在区域大小与安全性之间取得平衡。

### 2.2 与其他 IRIS 变体的关系

Drake 中提供了三种 IRIS 变体，通过 `iris_options` 的类型选择：

| 变体 | 选项类型 | 优化方式 | 梯度需求 | 适用场景 |
|------|----------|----------|----------|----------|
| **Iris** | `IrisOptions` | 半定规划 (SDP) | 需要解析梯度 | 低维配置空间，有解析障碍物表示 |
| **IrisNp2** | `IrisNp2Options` | 非线性规划 | 需要梯度 | 中等维度，有梯度信息 |
| **IrisZo** | `IrisZoOptions` | 零阶优化 | 仅需碰撞检测 | 高维/复杂场景，仅有碰撞检测器 |

---

## 3. API 详解

### 3.1 主函数：`IrisZo()`

```cpp
geometry::optimization::HPolyhedron IrisZo(
    const CollisionChecker& checker,
    const geometry::optimization::Hyperellipsoid& starting_ellipsoid,
    const geometry::optimization::HPolyhedron& domain,
    const IrisZoOptions& options = IrisZoOptions()
);
```

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `checker` | `CollisionChecker` | 碰撞检测器，提供配置空间中的碰撞查询接口 |
| `starting_ellipsoid` | `Hyperellipsoid` | 初始椭球体，算法围绕其膨胀区域。通常为碰撞自由配置点附近的小球体 |
| `domain` | `HPolyhedron` | 搜索域，限制生成区域的最大范围。必须是有界多面体，通常为关节限位边界框 |
| `options` | `IrisZoOptions` | 算法参数，控制概率保证、迭代行为等 |

#### 返回值

- **`HPolyhedron`**：表示计算得到的碰撞自由区域，以 H-表示（半空间交集 $\{x \mid Ax \leq b\}$）存储。

#### 前置条件

1. `starting_ellipsoid` 的中心必须是碰撞自由的
2. `starting_ellipsoid` 和 `domain` 的环境维度必须与配置空间维度一致（除非指定了参数化，此时需匹配 `options.parameterization_dimension`）
3. `domain` 必须是有界的

#### 异常

- 若 `starting_ellipsoid` 的中心处于碰撞状态，抛出异常
- 若 `starting_ellipsoid` 的中心违反 `options.prog_with_additional_constraints` 中的用户约束，抛出异常

#### 注意事项

- 该函数可能需要求解大量 QP（二次规划）问题，运行时间较长
- 若使用需要许可证的求解器，建议在调用前获取许可证（参见 `AcquireLicense`）
- 该功能标记为**实验性**，可能随时变更或移除

### 3.2 辅助函数：`MakeIrisObstacles()`

```cpp
ConvexSets MakeIrisObstacles(
    const QueryObject<double>& query_object,
    std::optional<FrameId> reference_frame = std::nullopt
);
```

#### 功能

从 `SceneGraph` 的 `QueryObject` 中提取障碍物的凸集表示，用于 IRIS 算法在 3D 空间中的计算。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `query_object` | `QueryObject<double>` | 场景查询对象，包含当前上下文中的几何信息 |
| `reference_frame` | `std::optional<FrameId>` | 参考坐标系，默认为世界坐标系 |

#### 行为细节

- 场景中所有具有 proximity role 的几何体（包括锚定和动态几何体）均被视为**固定障碍物**，冻结在创建 `QueryObject` 时上下文所捕获的位姿中
- 当同一几何体有多种凸集表示时（如 Box 可表示为 `HPolyhedron` 或 `VPolytope`），优先选择当前 IRIS 实现中性能最优的表示

### 3.3 辅助函数：`SetEdgeContainmentTerminationCondition()`

```cpp
void SetEdgeContainmentTerminationCondition(
    IrisOptions* iris_options,
    const Eigen::Ref<const Eigen::VectorXd>& x_1,
    const Eigen::Ref<const Eigen::VectorXd>& x_2,
    const double epsilon = 1e-3,
    const double tol = 1e-6
);
```

#### 功能

修改 `iris_options`，使 IRIS 算法寻找包含连接 $x_1$ 和 $x_2$ 的边（edge）的区域。

#### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `iris_options` | `IrisOptions*` | 待修改的 IRIS 选项指针 |
| `x_1` | `VectorXd` | 边的起点 |
| `x_2` | `VectorXd` | 边的终点 |
| `epsilon` | `double` | 椭球体在非边方向上的扩展半径，默认 1e-3。必须 > 0 |
| `tol` | `double` | 边包含的容差，默认 1e-6 |

#### 行为细节

1. 设置 `iris_options.starting_ellipse` 为包含该边的超椭球体：
   - 中心位于边的中点 $(x_1 + x_2) / 2$
   - 沿边方向延伸以覆盖整个边
   - 在其他方向上延伸 $\epsilon$
2. 设置 `iris_options.termination_func`，当边不再被 IRIS 区域包含（容差 `tol`）时终止迭代

#### 异常

- 若 `x_1.size() != x_2.size()`，抛出 `std::exception`
- 若 `epsilon <= 0`，抛出 `std::exception`（因为超椭球体必须具有非零体积）

---

## 4. IrisZoOptions 参数配置

`IrisZoOptions` 收集 IRIS-ZO 算法的所有参数。头文件：`#include <drake/planning/iris/iris_zo.h>`

> **警告**：该功能标记为实验性，可能随时变更或移除，无弃用通知。

### 4.1 完整公共属性

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sampled_iris_options` | `CommonSampledIrisOptions` | `{}` | 采样与终止条件相关选项（详见 4.2） |
| `bisection_steps` | `int` | `10` | 最大二分搜索步数，控制零阶方向搜索的精度 |
| `parameterization` | `IrisParameterizationFunction` | `{}`（恒等参数化） | 区域生长子空间的参数化函数 |

### 4.2 CommonSampledIrisOptions 详解

`sampled_iris_options` 是 `CommonSampledIrisOptions` 类型，包含所有基于采样的 IRIS 变体（IrisZo、IrisNp2）共有的参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `epsilon` | `double` | — | 碰撞体积占比上限 $\varepsilon$，控制区域中允许的碰撞比例 |
| `delta` | `double` | — | 置信水平 $\delta$，控制概率保证的严格程度 |
| `iteration_limit` | `int` | — | 最大迭代次数 |
| `termination_threshold` | `double` | — | 终止阈值，当区域增长低于此值时停止 |
| `parallelism` | `Parallelism` | — | 并行度，控制零阶优化的并行线程数 |
| `prog_with_additional_constraints` | `MathematicalProgram` | — | 用户自定义的额外约束程序 |
| `configuration_space_margin` | `double` | — | 配置空间边界冗余量 |
| `num_collision_infeasible_samples` | `int` | — | 碰撞不可行样本数量 |
| `num_additional_constraints_infeasible_samples` | `int` | — | 额外约束不可行样本数量 |
| `relative_termination_threshold` | `double` | — | 相对终止阈值 |
| `require_relaxed` | `bool` | — | 是否要求松弛解 |

**概率保证参数调参建议**：
- 高安全场景：$\varepsilon = 10^{-4}$, $\delta = 10^{-6}$
- 平衡场景：$\varepsilon = 10^{-3}$, $\delta = 10^{-3}$
- 快速计算：$\varepsilon = 10^{-2}$, $\delta = 10^{-2}$

### 4.3 bisection_steps 详解

`bisection_steps` 控制零阶方向搜索中二分法的最大步数：

- **作用**：在每个搜索方向上，算法使用二分法寻找碰撞边界。`bisection_steps` 决定了搜索的精度——步数越多，边界定位越精确，但计算量也越大
- **默认值**：10
- **精度关系**：每步二分将搜索区间减半，10 步的精度约为 $2^{-10} \approx 0.1\%$ 的初始区间长度
- **调参建议**：
  - 粗糙快速：5-8（适合预计算/探索阶段）
  - 默认平衡：10
  - 高精度：15-20（适合最终安全验证）

### 4.4 parameterization 详解

`parameterization` 定义区域生长的子空间参数化：

- **类型**：`IrisParameterizationFunction`
- **默认**：恒等参数化（identity），对应在普通配置空间中生长区域
- **用途**：当需要在配置空间的低维子流形上生长区域时使用（如仅在某些关节方向上膨胀）
- **维度约束**：若指定参数化，`starting_ellipsoid` 和 `domain` 的维度需匹配 `parameterization` 的输出维度而非配置空间维度

### 4.5 序列化支持

`IrisZoOptions` 支持 YAML 序列化（通过 `Serialize` 方法），但仅序列化 YAML 内置类型的选项。该类支持拷贝构造、拷贝赋值、移动构造和移动赋值。

---

## 5. CollisionChecker 接口规范

`CollisionChecker` 是 IrisZo 的核心依赖接口，提供配置空间中的碰撞查询能力。头文件：`#include <drake/planning/collision_checker.h>`

### 5.1 接口定位

`CollisionChecker` 是一个**抽象基类**，为各种规划问题（如采样规划、IRIS 区域生成）提供距离查询基础。Drake 提供的具体实现包括 `SceneGraphCollisionChecker`（基于 SceneGraph）。

### 5.2 核心碰撞查询方法

#### 配置碰撞检查

| 方法 | 签名 | 说明 |
|------|------|------|
| `CheckConfigCollisionFree` | `(q, context_number?) → bool` | 检查单个配置是否碰撞自由。`true` = 无碰撞 |
| `CheckContextConfigCollisionFree` | `(model_context, q) → bool` | 显式上下文版本 |
| `CheckConfigsCollisionFree` | `(configs, parallelize?) → vector<uint8_t>` | 批量检查，支持 OpenMP 并行 |

#### 边碰撞检查

| 方法 | 签名 | 说明 |
|------|------|------|
| `CheckEdgeCollisionFree` | `(q1, q2, context_number?) → bool` | 检查 q1→q2 边是否碰撞自由 |
| `CheckContextEdgeCollisionFree` | `(model_context, q1, q2) → bool` | 显式上下文版本 |
| `CheckEdgeCollisionFreeParallel` | `(q1, q2, parallelize?) → bool` | OpenMP 并行版本 |
| `CheckEdgesCollisionFree` | `(edges, parallelize?) → vector<uint8_t>` | 批量边检查 |
| `MeasureEdgeCollisionFree` | `(q1, q2, context_number?) → EdgeMeasure` | 返回边的碰撞自由度量 |
| `MeasureEdgeCollisionFreeParallel` | `(q1, q2, parallelize?) → EdgeMeasure` | 并行版本 |
| `MeasureEdgesCollisionFree` | `(edges, parallelize?) → vector<EdgeMeasure>` | 批量度量 |

#### 机器人碰撞状态

| 方法 | 签名 | 说明 |
|------|------|------|
| `CalcRobotClearance` | `(q, influence_distance, context_number?) → RobotClearance` | 计算距离 $\phi$ 和雅可比 $J_{q_r}\phi$ |
| `ClassifyBodyCollisions` | `(q, context_number?) → vector<RobotCollisionType>` | 分类每个机器人体的碰撞类型 |

### 5.3 边碰撞检查的配置

边碰撞检查通过沿边采样配置来近似判断。关键配置参数：

| 参数/方法 | 说明 |
|-----------|------|
| `edge_step_size()` / `set_edge_step_size()` | 边步长，决定采样密度。步长越小，检查越精确但越慢 |
| `SetConfigurationDistanceFunction()` | 配置距离函数，如 $\|q_1 - q_2\|$ 或加权范数 $\|w^T(q_1 - q_2)\|$ |
| `SetConfigurationInterpolationFunction()` | 配置插值函数，默认对四元数用 Slerp，其余线性插值 |
| `SetDistanceAndInterpolationProvider()` | 统一设置距离和插值提供者 |

**默认插值**：对四元数值自由度使用 Slerp，其余使用线性插值。不适用于 BallRpyJoint 或非完整机器人。

**步长调优**：对于全旋转关节的机器人，推荐 `edge_step_size = 0.05` 弧度（基于 IIWA、Panda 等的经验值）。

### 5.4 碰撞类型枚举

```cpp
enum class RobotCollisionType : uint8_t {
    kNoCollision = 0x00,                    // 无碰撞
    kEnvironmentCollision = 0x01,           // 与环境碰撞
    kSelfCollision = 0x02,                  // 自碰撞
    kEnvironmentAndSelfCollision = 0x03     // 环境碰撞 + 自碰撞
};
```

### 5.5 RobotClearance 结构详解

`RobotClearance` 是 `CalcRobotClearance()` 的返回类型，表示机器人与世界之间距离测量的集合。头文件：`#include <drake/planning/robot_clearance.h>`

#### 概念模型

`RobotClearance` 表示一个表格，每行描述一个机器人身体（body R）与另一个身体（body O）之间的距离关系：

| body index R | body index O | type | $\phi^O(R)$ | $J_{q_r}\phi^O(R)$ |
|:------------:|:------------:|:----:|:-----------:|:------------------:|
|     ...      |     ...      | ...  |     ...     |        ...         |

#### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| body index R | `BodyIndex` | 机器人身体的索引 |
| body index O | `BodyIndex` | 另一个身体的索引（可能是机器人或环境） |
| type | `RobotCollisionType` | body O 的类型：`kSelfCollision`（机器人身体）或 `kEnvironmentCollision`（环境身体） |
| $\phi^O(R)$ | `double` | body O 在 body R 上的**有符号距离函数**。报告的距离已被填充值偏移。零表示在填充表面上，负值表示在边界内，正值表示在边界外 |
| $J_{q_r}\phi^O(R)$ | `RowVectorXd` | $\phi^O(R)$ 对机器人配置向量 $q_r$ 的雅可比矩阵（世界坐标系下的导数） |

#### 核心方法

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `size()` | `int` | 距离测量数量（表格行数） |
| `num_positions()` | `int` | 雅可比矩阵的列数（位置变量数量） |
| `robot_indices()` | `vector<BodyIndex>&` | 机器人身体索引向量 |
| `other_indices()` | `vector<BodyIndex>&` | 其他身体索引向量 |
| `collision_types()` | `vector<RobotCollisionType>&` | 碰撞类型向量 |
| `distances()` | `Map<VectorXd>` | 距离向量 $\phi^O(R)$ |
| `jacobians()` | `Map<MatrixXd>` | 雅可比矩阵（`size()` 行 × `num_positions()` 列） |

#### 重要说明

1. **重复索引**：单个机器人身体索引可能多次出现，因为该身体可能与模型中多个其他身体有距离测量
2. **零雅可比**：某些行可能包含零值雅可比（局部最优配置或被过滤的碰撞）
3. **配置子集**：$q_r$ 是完整配置 $q$ 的子集（当存在非机器人关节时），雅可比仅对机器人部分计算
4. **列顺序**：雅可比矩阵的列顺序与 `plant.GetPositions()` 一致，非机器人关节的列为零

#### 使用示例

```python
# 计算机器人在配置 q 处的间隙
clearance = checker.CalcRobotClearance(q, influence_distance=0.1)

# 访问距离测量
for i in range(clearance.size()):
    robot_body = clearance.robot_indices()[i]
    other_body = clearance.other_indices()[i]
    collision_type = clearance.collision_types()[i]
    distance = clearance.distances()[i]
    jacobian = clearance.jacobians()[i, :]
    
    print(f"Robot body {robot_body} to body {other_body}: "
          f"type={collision_type}, distance={distance}")
```

### 5.6 填充（Padding）机制

填充机制在不修改底层模型的情况下调整报告的体间距离，用于安全冗余：

| 方法 | 说明 |
|------|------|
| `SetPaddingBetween(bodyA, bodyB, padding)` | 设置特定体对的填充值 |
| `SetPaddingAllRobotEnvironmentPairs(padding)` | 设置所有(机器人,环境)对的填充 |
| `SetPaddingAllRobotRobotPairs(padding)` | 设置所有(机器人,机器人)对的填充 |
| `SetPaddingOneRobotBodyAllEnvironmentPairs(body, padding)` | 设置特定机器人体与所有环境体的填充 |
| `GetPaddingBetween(bodyA, bodyB)` | 查询特定体对的填充值 |
| `GetLargestPadding()` | 获取最大填充值 |

**填充语义**：正填充减小报告距离（等效扩大几何体），负填充增大报告距离（等效缩小几何体）。填充存储在 NxN 对称矩阵中，对角线和环境-环境对固定为零。

### 5.7 碰撞过滤（Filtering）机制

碰撞过滤允许忽略特定体对之间的碰撞检查：

| 方法 | 说明 |
|------|------|
| `SetCollisionFilteredBetween(bodyA, bodyB, filter)` | 设置/取消体对的碰撞过滤 |
| `IsCollisionFilteredBetween(bodyA, bodyB)` | 查询体对是否被过滤 |
| `SetCollisionFilteredWithAllBodies(body)` | 过滤某体与所有其他体的碰撞 |
| `SetCollisionFilterMatrix(matrix)` | 设置完整过滤矩阵 |
| `GetFilteredCollisionMatrix()` | 获取当前过滤矩阵 |

**过滤矩阵值**：`0` = 未过滤，`1` = 已过滤，`-1` = 不可更改的过滤（对角线、环境-环境对）。

### 5.8 动态几何管理

运行时可向机器人体添加/移除碰撞几何体（如抓取操作后添加被夹持物体的几何体）：

| 方法 | 说明 |
|------|------|
| `AddCollisionShape(group_name, description)` | 添加形状到体，返回是否成功 |
| `AddCollisionShapes(group_name, descriptions)` | 批量添加 |
| `RemoveAllAddedCollisionShapes(group_name)` | 按组名移除 |
| `GetAllAddedCollisionShapes()` | 获取所有已添加的几何体 |

几何体按组（group）管理，支持批量添加和移除。

### 5.9 多线程并行模型

`CollisionChecker` 提供两种并行模型：

#### 隐式上下文并行（Implicit Context Parallelism）

- 基于 OpenMP 线程池
- 通过 `AllocateContexts()` 在构造时分配每线程上下文池
- 使用 `model_context(context_number?)` 访问线程关联上下文
- 适用于 `omp parallel` 指令驱动的并行

#### 显式上下文并行（Explicit Context Parallelism）

- 基于任意线程模型（如 `std::async`）
- 通过 `MakeStandaloneModelContext()` 创建独立上下文
- 使用 `CheckContextEdgeCollisionFree(model_context, q1, q2)` 等显式上下文方法
- 每个线程使用独立上下文，无需 OpenMP

#### 混合线程模型

可同时使用 OpenMP 和任意线程。每个非 OpenMP 线程应通过 `Clone()` 创建独立的 `CollisionChecker` 实例。

#### 并行支持查询

- `SupportsParallelChecking()` → `bool`：是否支持真正并行检查
- `num_allocated_contexts()` → `int`：已分配的隐式上下文数量

### 5.10 上下文管理方法

| 方法 | 说明 |
|------|------|
| `model_context(context_number?)` | 访问隐式上下文池中的碰撞检查上下文 |
| `plant_context(context_number?)` | 访问 MultibodyPlant 子上下文 |
| `UpdatePositions(q, context_number?)` | 更新隐式上下文中的广义位置 q |
| `UpdateContextPositions(model_context, q)` | 显式上下文版本 |
| `MakeStandaloneModelContext()` | 创建并跟踪独立上下文 |
| `PerformOperationAgainstAllModelContexts(operation)` | 对所有上下文执行操作 |

### 5.11 构造参数（CollisionCheckerParams）

`CollisionCheckerParams` 是构造 `CollisionChecker` 的通用参数集，包含：

- 机器人模型（`RobotDiagram`）
- 模型实例索引（标识哪些体属于机器人）
- 配置距离函数
- 配置插值函数
- 隐式上下文并行度
- 边步长

---

## 6. 算法工作流程

### 6.1 整体框架（Algorithm 1）

```
Algorithm 1: IRIS 算法模板
Input: Domain D ⊆ C, collision-free seed s ∈ D, options O
Output: Polytope P ⊆ D

Algorithm:
  E ← Ball(s, r_start)    // 初始椭球体（种子点附近的小球）
  i ← 1                   // 外迭代计数器
  while not done do
    P ← SeparatingPlanes(D, E, i, O)  // 分离平面步骤（核心）
    E ← InscribedEllipsoid(P)          // 计算内接椭球体
    i ← i + 1
  end
  return P
```

**外迭代**：交替更新多面体 P 和内接椭球体 E，逐步扩大碰撞自由区域。

### 6.2 ZeroOrderSeparatingPlanes（Algorithm 2 - IRIS-ZO 核心）

```
Algorithm 2: ZeroOrderSeparatingPlanes
Input: Domain D ⊆ C, ellipsoid E = (E, c), current outer iteration i ∈ N
Output: Polytope P ⊆ D satisfying (1) for (ε, δ_i)

Algorithm:
  k ← 1, P ← D           // 初始化：P 为搜索域
  while True do
    // Step 1: 均匀采样
    S ∼ UniformSample(P, M)           // M 个均匀采样点
    
    // Step 2: 碰撞检测
    S_col ← PointsInCollision(S)      // 找出碰撞点
    
    // Step 3: 终止条件检查
    If UnadaptiveTest(δ_{i,k}, ε, τ) returns accept then break
    
    // Step 4: 二分搜索优化
    S_col^⋆ ← UpdatePointsViaBisection(S_col, c)
    
    // Step 5: 添加分离超平面
    P ← OrderAndPlaceNonRedundantHyperplanes(P, E, S_col^⋆, Δ)
    
    k ← k + 1
  end
  return P
```

### 6.3 详细步骤解析

#### Step 1: 均匀采样（Hit-and-Run Sampling）

- **方法**：使用 hit-and-run 采样算法生成 P 内的均匀分布样本
- **批量大小**：`max{M, N_p}`，其中：
  - $M = \lceil 2 \log(1/\delta) / (\varepsilon \tau^2) \rceil$（统计测试所需样本数）
  - $N_p$：每轮内迭代优化的粒子数量
- **混合步数**：需足够多以接近均匀分布

#### Step 2: 碰撞检测

- 对采样点集 $S$ 运行碰撞检测器
- 提取碰撞点子集 $S_{col} \subseteq S$
- 统计碰撞比例：$|S_{col}| / M$

#### Step 3: 统计终止测试（UnadaptiveTest）

**核心思想**：控制错误接受（false accept）的概率

```
UnadaptiveTest(δ, ε, τ):
  收集 M = ⌈2 log(1/δ)/(ετ²)⌉ 个样本
  计算 X̄_M = |S_col| / M（碰撞比例）
  if X̄_M > (1 - τ)ε then
    return reject   // 继续添加超平面
  else
    return accept   // 终止，返回 P
  end
```

**参数说明**：
- $\varepsilon$：允许的碰撞体积占比上限
- $\delta$：置信水平（错误接受概率上限）
- $\tau$：松弛参数，默认 0.5（平衡测试强度和计算成本）

**概率保证**：
$\Pr[\text{false accept} \mid \varepsilon_{tr} > \varepsilon] \leq \delta$

其中 $\varepsilon_{tr} = \lambda(P \setminus C_{free}) / \lambda(P)$ 是真实碰撞比例。

**多轮迭代的置信度分配**：
- 单外迭代：$\delta_k = 6\delta / (\pi^2 k^2)$（第 k 次内迭代）
- 多外迭代：$\delta_{i,k} = 36\delta / (\pi^4 i^2 k^2)$（第 i 次外迭代的第 k 次内迭代）

#### Step 4: 二分搜索优化（UpdatePointsViaBisection）

**目标**：将碰撞点向椭球体中心移动，找到更接近障碍物边界的点

```
对每个 q ∈ S_col 的前 N_p 个点:
  for b = 1 to N_b do  // N_b = bisection_steps
    q_mid = (q + c) / 2  // 向中心 c 移动
    if q_mid in collision then
      q = q_mid          // 继续向中心移动
    else
      // 保持 q 不变，向边界移动
    end
  end
  q^⋆ = 最接近 c 且仍在碰撞中的点
end
return S_col^⋆ = {q^⋆}
```

**关键特性**：
- **梯度无关**：仅依赖碰撞检测器的布尔查询
- **高度并行**：每个粒子的二分搜索独立进行
- **收敛保证**：保证收敛到某障碍物边界（但不一定是包含原 q 的障碍物或最接近 c 的障碍物）

#### Step 5: 添加分离超平面（OrderAndPlaceNonRedundantHyperplanes）

**排序**：按椭球度量距离升序排列候选点 $q^⋆ \in S_{col}^⋆$

$\text{distance} = \|q^⋆ - c\|_E = \sqrt{(q^⋆ - c)^T E (q^⋆ - c)}$

**超平面构造**：对每个非冗余候选点 $q^⋆$：

$a = E(q^⋆ - c), \quad b = a^T q^⋆$

**添加半空间**：$P \leftarrow P \cap \{q \mid a^T q \leq b - \Delta\}$

其中 $\Delta > 0$ 是配置空间边界冗余量（stepback），确保超平面与碰撞配置保持距离。

**冗余检测**：若新超平面不改变 P（被已有超平面包含），则跳过。

**终止条件**：添加 $N_f$ 个非冗余超平面或候选点耗尽。

### 6.4 算法参数对照表

| 参数 | 符号 | 说明 | IRIS-ZO | IRIS-NP2 |
|------|------|------|---------|----------|
| 碰撞占比上限 | $\varepsilon$ | 允许的碰撞体积占比 | ✓ | ✓ |
| 置信水平 | $\delta$ | 错误接受概率上限 | ✓ | ✓ |
| 松弛参数 | $\tau$ | 统计测试松弛度，默认 0.5 | ✓ | ✓ |
| 边界冗余 | $\Delta$ | 超平面后退距离 | ✓ | ✓ |
| 粒子数量 | $N_p$ | 每轮优化的粒子数 | ✓ | ✓ |
| 二分步数 | $N_b$ | 二分搜索最大步数 | ✓ | ✗ |
| 超平面数量 | $N_f$ | 每轮添加的最大超平面数 | ✓ | ✓ |

### 6.5 流程图

```
输入: checker, starting_ellipsoid, domain, options
  │
  ├─ 1. 验证前置条件
  │     ├─ 检查 starting_ellipsoid 中心是否碰撞自由
  │     ├─ 检查维度一致性（考虑 parameterization）
  │     └─ 检查 domain 有界性
  │
  ├─ 2. 初始化
  │     ├─ E ← starting_ellipsoid
  │     ├─ P ← domain
  │     └─ i ← 1 (外迭代计数)
  │
  ├─ 3. 外迭代循环
  │     │
  │     ├─ 3a. 内迭代 (ZeroOrderSeparatingPlanes)
  │     │     │
  │     │     ├─ 均匀采样: S ∼ UniformSample(P, M)
  │     │     ├─ 碰撞检测: S_col ← PointsInCollision(S)
  │     │     ├─ 统计测试: if UnadaptiveTest accepts → break
  │     │     ├─ 二分优化: S_col^⋆ ← Bisection(S_col, c, N_b)
  │     │     ├─ 排序候选点（按椭球度量）
  │     │     └─ 添加超平面（最多 N_f 个非冗余）
  │     │
  │     ├─ 3b. 更新内接椭球体
  │     │     └─ E ← InscribedEllipsoid(P)
  │     │
  │     └─ 3c. 终止判断
  │           ├─ 达到外迭代上限 → 停止
  │           └─ 区域增长不足 → 停止
  │
  └─ 4. 返回 P
```

---

## 7. 典型使用模式

### 7.1 基本用法

```python
from pydrake.geometry.optimization import IrisZo, IrisZoOptions, HPolyhedron, Hyperellipsoid
from pydrake.planning import CollisionChecker

# 1. 创建碰撞检测器
checker = CollisionChecker(...)  # 根据机器人模型配置

# 2. 定义初始椭球体（种子点附近的小球）
seed_point = np.array([0.5, 0.5, 0.0])  # 碰撞自由配置
radius = 0.01
starting_ellipsoid = Hyperellipsoid.MakeHyperSphere(radius, seed_point)

# 3. 定义搜索域（关节限位边界框）
joint_limits_lower = np.array([-3.14, -3.14, -3.14])
joint_limits_upper = np.array([3.14, 3.14, 3.14])
domain = HPolyhedron.MakeBox(joint_limits_lower, joint_limits_upper)

# 4. 配置算法参数
options = IrisZoOptions()
options.sampled_iris_options.epsilon = 1e-3       # 碰撞体积占比上限
options.sampled_iris_options.delta = 1e-3         # 置信水平
options.sampled_iris_options.iteration_limit = 100
options.bisection_steps = 10                       # 二分搜索步数

# 5. 运行 IrisZo
region = IrisZo(checker, starting_ellipsoid, domain, options)

# 6. 使用结果
print(f"区域维度: {region.ambient_dimension()}")
print(f"半空间数量: {region.A().shape[0]}")
```

### 7.2 沿路径生成多个区域

```python
regions = []
for seed_point in path_points:
    if not checker.CheckConfigCollisionFree(seed_point):
        continue

    starting_ellipsoid = Hyperellipsoid.MakeHyperSphere(0.01, seed_point)
    region = IrisZo(checker, starting_ellipsoid, domain, options)
    regions.append(region)
```

### 7.3 使用边包含终止条件

```python
from pydrake.geometry.optimization import IrisOptions, Iris

# 确保区域包含从 x1 到 x2 的边
iris_options = IrisOptions()
SetEdgeContainmentTerminationCondition(
    iris_options, x_1, x_2,
    epsilon=1e-3,  # 非边方向扩展半径
    tol=1e-6       # 包含容差
)

region = Iris(checker, iris_options.starting_ellipse, domain, iris_options)
```

### 7.4 从 SceneGraph 构建障碍物

```python
from pydrake.geometry.optimization import MakeIrisObstacles

# 从场景中提取障碍物凸集表示
query_object = plant.get_geometry_query_object(context)
obstacles = MakeIrisObstacles(query_object)

# obstacles 可用于 IRIS 算法的障碍物输入
```

---

## 8. 与项目中 IrisNp 实现的对比

本项目（`src/iris_pkg/`）当前使用的是 **IrisNp**（IRIS for Configuration Space with Nonlinear Programming），而非 IrisZo。以下对比两者差异：

| 特性 | IrisZo | IrisNp（项目当前实现） |
|------|--------|----------------------|
| **优化方式** | 零阶优化（仅需碰撞检测） | 非线性规划（需要梯度信息） |
| **Drake API** | `IrisZo()` + `IrisZoOptions` | `IrisNp` + `IrisOptions` |
| **概率保证** | 显式 $\Pr[\lambda(P\setminus C_{free})/\lambda(P) > \varepsilon] \leq \delta$ | 通过采样数隐式控制 |
| **并行性** | 原生支持并行零阶搜索 | 项目自行实现并行（`iris_np_parallel.py`） |
| **碰撞检测** | `CollisionChecker`（Drake 原生） | `SimpleCollisionCheckerForIrisNp`（自定义，带 LRU 缓存） |
| **区域表示** | `HPolyhedron`（Drake 原生） | `IrisNpRegion`（自定义，含顶点、面积、种子点） |
| **膨胀策略** | 并行方向搜索 + QP 优化 | 自适应/椭圆/方形膨胀（`IrisNpExpansion`） |
| **种子点策略** | 用户指定 | 自动提取 + Voronoi 优化 + 两批扩张 |
| **覆盖验证** | 概率采样验证 | 严格逐点验证（`IrisNpCoverageChecker`） |
| **区域修剪** | 无内置 | `RegionPruner`（基于 RTree 空间索引） |

### 8.1 迁移到 IrisZo 的考量

**优势**：
- 无需梯度信息，适用于更广泛的机器人模型
- 原生概率保证，数学上更严格
- Drake 原生并行支持，无需自行实现
- 适合高维配置空间

**挑战**：
- 零阶优化收敛速度可能慢于梯度方法
- 需要适配项目的碰撞检测接口（`SimpleCollisionCheckerForIrisNp` → `CollisionChecker`）
- 项目中的 Voronoi 优化、两批扩张、区域修剪等增强策略需要重新设计集成方式
- 实验性 API，稳定性风险

---

## 9. iris_options 变体选择机制

Drake 中 IRIS 变体的选择通过 `iris_options` 的类型自动确定：

```cpp
std::variant<geometry::optimization::IrisOptions,
             IrisNp2Options,
             IrisZoOptions> iris_options {
    geometry::optimization::IrisOptions{.iteration_limit = 1}
};
```

- 默认使用 `IrisOptions`（原始 IRIS），且 `iteration_limit = 1`
- 传入 `IrisZoOptions` 时自动调用 `IrisZo()`
- 传入 `IrisNp2Options` 时自动调用 `IrisNp2()`

**推荐**：在从 clique 构建时，建议仅运行 1 次迭代，以避免丢弃从 clique 获得的信息。

**调试可视化**：`IrisOptions` 可选包含 `meshcat` 实例以提供调试可视化。但若 `parallelism > 1`，内部 IRIS 调用的调试可视化将被禁用（meshcat 不支持非主线程绘制）。

---

## 10. 性能考量

### 10.1 计算复杂度

- **采样复杂度**：$M = O(\log(1/\delta) / (\varepsilon \tau^2))$，由概率保证决定
- **二分搜索**：每个粒子 $N_b$ 步，总共 $N_p \times N_b$ 次碰撞查询
- **超平面添加**：每轮最多 $N_f$ 个，需冗余检测
- **并行加速**：采样和二分搜索高度并行，性能随计算资源线性扩展

### 10.2 与 IRIS-NP 的性能对比

基于论文在 8 个机器人环境上的基准测试结果：

| 设置 | 指标 | IRIS-ZO vs IRIS-NP | IRIS-NP2 vs IRIS-NP |
|------|------|---------------------|---------------------|
| **Fast** (90% 无碰撞, 90% 置信) | 运行时间 | **15.5× 更快** | 6.55× 更快 (greedy) / 4.2× (ray) |
| | 超平面数量 | 1.4× 更少 | 2.1× 更少 (greedy) / 2.3× (ray) |
| **Precise** (99% 无碰撞, 95% 置信) | 运行时间 | **14× 更快** | 12.3× 更快 (greedy) / 8× (ray) |
| | 超平面数量 | 相当 | 2.4× 更少 (greedy) / 2.7× (ray) |

**关键发现**：
- IRIS-ZO 比 IRIS-NP **快一个数量级**，且通常使用更少的超平面
- IRIS-NP2 生成更大的区域，使用更少的超平面（利于下游计算）
- 随着环境复杂度增加（更多碰撞几何体），性能差距进一步扩大

### 10.3 测试环境

基准测试配置：
- **机器人**：Flipper, UR3, UR3Wrist, IIWAShelf, 4Shelves, IIWABins, 2IIWAs, Allegro
- **种子点**：每个环境 10 个，每个种子运行 10 次（共 100 次试验/环境）
- **硬件**：Intel Core i9-10850K (10 cores, 20 threads)
- **外迭代**：单次（隔离 SeparatingPlanes 性能差异）

### 10.4 求解器选择

- IrisZo 内部使用 QP 求解器
- 商业求解器（如 Gurobi、MOSEK）可能需要许可证，建议调用前通过 `AcquireLicense` 获取
- 开源求解器（如 OSQP、SCS）无需许可证但可能较慢

### 10.5 参数调优建议

| 场景 | $\varepsilon$ | $\delta$ | iteration_limit | parallelism | 说明 |
|------|---------------|----------|-----------------|-------------|------|
| 高安全（手术/人协作） | 1e-2 | 5e-2 | 200 | 最大可用 | 99% 无碰撞, 95% 置信 |
| 平衡（一般规划） | 1e-1 | 1e-1 | 100 | 4-8 | 90% 无碰撞, 90% 置信 |
| 快速探索（预计算） | 1e-1 | 1e-1 | 50 | 2-4 | 快速生成，后续精化 |

**参数影响**：
- $\varepsilon$ 越小 → 采样数 $M$ 增加 → 运行时间增加
- $\delta$ 越小 → 采样数 $M$ 增加 → 运行时间增加
- $N_b$ 越大 → 边界定位越精确 → 运行时间增加
- $N_p$ 越大 → 更多候选超平面 → 可能减少迭代次数
- $N_f$ 越大 → 每轮添加更多超平面 → 可能减少迭代次数

### 10.6 并行化潜力

IRIS-ZO 的并行化优势：
- **采样**：hit-and-run 采样可并行生成多条链
- **碰撞检测**：批量检测天然并行
- **二分搜索**：每个粒子独立进行，无数据依赖
- **未来方向**：SIMD 指令、GPU 加速可进一步提升性能

---

## 11. 实验性声明

IrisZo 在 Drake 中标记为**实验性功能**，可能在未来版本中：
- API 发生不兼容变更
- 被移除且无弃用通知
- 行为语义发生变化

建议在生产环境中使用时进行充分测试，并关注 Drake 版本更新日志。
