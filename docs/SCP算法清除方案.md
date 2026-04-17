# SCP算法清除方案

## 文档信息
- **项目名称**: a_gcs_ws (Ackermann GCS Workspace)
- **版本**: 2.0.1
- **创建日期**: 2026-04-17
- **变更类型**: 代码清除（SCP算法及曲率成本模块移除）

---

## 1. 清除背景与目标

### 1.1 背景

在系统演进过程中，曲率约束的实现经历了从SCP（Sequential Convex Programming）迭代逼近到曲率硬约束（Lorentz锥凸约束）的转变。曲率硬约束直接在GCS优化问题中添加凸约束，无需迭代求解，具有更好的可靠性和效率。

当前系统中同时存在两条求解路径：
- **路径1（曲率硬约束）**：当 `enable_curvature_hard_constraint=True` 时，直接在GCS中添加Lorentz锥凸约束，跳过SCP迭代
- **路径2（SCP迭代）**：当不启用硬约束时，使用SCP迭代逐步逼近曲率约束

路径2已被路径1完全取代，且曲率成本相关模块已被注释禁用。

### 1.2 目标

- 彻底清除SCP迭代求解路径和曲率成本模块代码
- 使曲率硬约束成为唯一的曲率约束实现方式
- 简化代码架构，消除冗余的双路径分支
- 保持其他功能（速度约束、航向角约束、成本函数、轨迹评估等）完全不受影响

---

## 2. 待删除文件清单

| 序号 | 文件路径 | 行数 | 删除原因 |
|------|---------|------|---------|
| 1 | `src/ackermann_gcs_pkg/ackermann_scp_solver.py` | 602 | SCP求解器主模块，已被曲率硬约束取代 |
| 2 | `src/ackermann_gcs_pkg/scp_optimization/__init__.py` | 24 | SCP优化子模块入口 |
| 3 | `src/ackermann_gcs_pkg/scp_optimization/trust_region_manager.py` | ~200 | 信任区域管理器 |
| 4 | `src/ackermann_gcs_pkg/scp_optimization/early_termination_checker.py` | ~400 | 提前终止检查器 |
| 5 | `src/ackermann_gcs_pkg/scp_optimization/parallel_curvature_linearizer.py` | ~300 | 并行曲率线性化器 |
| 6 | `src/ackermann_gcs_pkg/scp_optimization/performance_stats.py` | ~260 | 性能统计收集器 |
| 7 | `src/ackermann_gcs_pkg/scp_optimization/constraint_violation_calculator.py` | ~120 | 约束违反量计算器 |
| 8 | `src/ackermann_gcs_pkg/curvature_cost_module.py` | 582 | 曲率成本模块(已禁用) |
| 9 | `src/ackermann_gcs_pkg/curvature_cost_linearizer.py` | 533 | 曲率成本线性化器(已禁用) |
| 10 | `src/ackermann_gcs_pkg/curvature_derivative_cost.py` | 509 | 曲率导数成本(已禁用) |
| 11 | `src/ackermann_gcs_pkg/curvature_peak_cost.py` | 501 | 曲率峰值成本(已禁用) |
| 12 | `src/ackermann_gcs_pkg/curvature_squared_cost_calculator.py` | 196 | 曲率平方成本计算器(已禁用) |
| 13 | `src/ackermann_gcs_pkg/analytic_gradient_calculator.py` | 624 | 解析梯度计算器(已禁用) |
| 14 | `src/ackermann_gcs_pkg/cost_calculator_interface.py` | ~50 | 成本计算器接口(仅被上述模块使用) |

**总计**: 14个文件，约4,500+行代码

---

## 3. 待修改文件清单

| 序号 | 文件路径 | 修改内容 |
|------|---------|---------|
| 1 | `src/ackermann_gcs_pkg/ackermann_data_structures.py` | 删除15个SCP常量、11个SCP数据结构、2个注释禁用的曲率成本数据结构、LinearizedCostCoefficients；更新PlanningResult.num_iterations语义；移除`import time` |
| 2 | `src/ackermann_gcs_pkg/__init__.py` | 移除TYPE_CHECKING中SCP/曲率成本导入、__getattr__延迟导入、__all__导出 |
| 3 | `src/ackermann_gcs_pkg/ackermann_gcs_planner.py` | 移除SCP相关导入、简化__init__接口(移除scp_config参数)、移除曲率成本注释代码、简化plan_trajectory逻辑(移除SCP路径分支)、更新docstring |
| 4 | `scripts/visualize_3d_trajectory.py` | 移除SCPConfig导入和scp_config参数 |
| 5 | `scripts/batch_test_curvature_constraint.py` | 移除SCPConfig导入和scp_config参数 |
| 6 | `scripts/hybrid_astar_gcs_planner.py` | 移除SCPConfig导入和scp_config参数 |
| 7 | `src/path_planner/scripts/planner_support/gcs_optimizer.py` | 移除SCPConfig导入、删除SCP配置读取代码、移除scp_config参数 |

---

## 4. 保留文件清单

| 文件路径 | 保留原因 |
|---------|---------|
| `src/ackermann_gcs_pkg/curvature_utils.py` | 核心曲率计算工具，被ackermann_scp_solver(已删)、curvature_statistics、flat_output_mapper等广泛依赖 |
| `src/ackermann_gcs_pkg/curvature_statistics.py` | 曲率统计模块，被规划器使用 |
| `src/ackermann_gcs_pkg/constraint_utils.py` | 约束工具模块，被trajectory_evaluator使用 |
| `src/gcs_pkg/scripts/core/bezier.py` | 曲率硬约束实现(addCurvatureHardConstraint/addCurvatureHardConstraintForEdges) |
| `config/gcs/cost_configurator.py` | 曲率硬约束预设配置(curvature_constrained等)仍被使用 |
| `scripts/batch_test_curvature_constraint.py` | 批量测试曲率硬约束(修改后保留) |

---

## 5. 数据结构变更详情

### 5.1 删除的数据结构

| 数据结构 | 类型 | 原用途 |
|---------|------|--------|
| SCPConfig | dataclass | SCP迭代配置 |
| TrustRegionConfig | dataclass | 信任区域配置 |
| TerminationConfig | dataclass | 终止条件配置 |
| ParallelConfig | dataclass | 并行计算配置 |
| TerminationReason | enum | 终止原因 |
| ImprovementRecord | dataclass | 改进记录 |
| IterationStats | dataclass | 迭代统计 |
| ViolationReport | dataclass | 约束违反报告 |
| ConstraintThresholds | dataclass | 约束阈值 |
| PerformanceMetrics | dataclass | 性能指标 |
| LinearizedCostCoefficients | dataclass | 线性化成本系数 |
| CurvatureCostConfig | dataclass(注释) | 曲率成本配置 |
| CurvatureCostWeights | dataclass(注释) | 曲率成本权重 |

### 5.2 删除的常量

```
VELOCITY_VIOLATION_THRESHOLD, ACCELERATION_VIOLATION_THRESHOLD, CURVATURE_VIOLATION_THRESHOLD
DEFAULT_CONVERGENCE_TOLERANCE, DEFAULT_STAGNATION_THRESHOLD, DEFAULT_MIN_TRUST_REGION
DEFAULT_MAX_ITERATIONS, DEFAULT_STAGNATION_WINDOW, DEFAULT_MAX_SHRINK_COUNT
DEFAULT_TRUST_REGION_RADIUS, TRUST_REGION_EXPAND_FACTOR, TRUST_REGION_SHRINK_FACTOR
AGGRESSIVE_SHRINK_FACTOR, DEFAULT_BATCH_SIZE, SMALL_EPSILON
```

### 5.3 修改的数据结构

| 数据结构 | 修改内容 |
|---------|---------|
| PlanningResult | num_iterations语义从"SCP迭代次数"更新为"求解尝试次数（GCS多次舍入尝试）" |

### 5.4 保留的数据结构（无修改）

VehicleParams, EndpointState, CurvatureConstraintMode, TrajectoryConstraints, BezierConfig, ConstraintViolation, ContinuityReport, CurvatureDerivatives, CurvatureStats, ImprovementMetrics, TrajectoryReport

---

## 6. 规划器修改详情

### 6.1 接口变更

| 接口元素 | 变更前 | 变更后 |
|---------|--------|--------|
| `__init__` 参数 | `(vehicle_params, bezier_config, scp_config)` | `(vehicle_params, bezier_config)` |
| `plan_trajectory` 内部逻辑 | 双路径：曲率硬约束 or SCP迭代 | 单路径：曲率硬约束GCS求解 |

### 6.2 逻辑简化

**变更前**:
```python
use_curvature_hard_constraint = (
    constraints.enable_curvature_hard_constraint or
    constraints.curvature_constraint_mode == "hard"
)
if use_curvature_hard_constraint:
    # 曲率硬约束路径：多次GCS求解
    ...
else:
    # SCP迭代路径
    scp_solver = AckermannSCPSolver(...)
    trajectory, converged = scp_solver.solve()
```

**变更后**:
```python
# 直接执行GCS求解（曲率硬约束）
max_solve_attempts = self.bezier_config.max_rounding_attempts
max_rounded_paths = self.bezier_config.max_rounded_paths
for attempt in range(max_solve_attempts):
    result = bezier_gcs.SolvePathWithConstraints(...)
    ...
```

---

## 7. 包接口变更详情

### 7.1 移除的公共API符号

| 符号 | 原类型 |
|------|--------|
| AckermannSCPSolver | 类 |
| SCPConfig | 数据结构 |
| LinearizedCostCoefficients | 数据结构 |
| CostCalculatorInterface | 协议 |
| CurvatureSquaredCostCalculator | 类 |
| AnalyticGradientCalculator | 类 |

### 7.2 保留的公共API符号

VehicleParams, EndpointState, TrajectoryConstraints, BezierConfig, ConstraintViolation, ContinuityReport, TrajectoryReport, PlanningResult, CurvatureDerivatives, AckermannBezierGCS, AckermannGCSPlanner, TrajectoryEvaluator, FlatOutputMapper, compute_flat_output_mapping

---

## 8. 清除执行步骤

### 阶段1: 删除叶子模块（无内部依赖）
1. 删除 `src/ackermann_gcs_pkg/ackermann_scp_solver.py`
2. 删除 `src/ackermann_gcs_pkg/scp_optimization/` 整个目录
3. 删除7个曲率成本模块文件

### 阶段2: 修改数据结构
4. 修改 `ackermann_data_structures.py`：删除SCP常量、数据结构、注释块
5. 修改 `__init__.py`：移除SCP/曲率成本导出

### 阶段3: 修改核心模块
6. 修改 `ackermann_gcs_planner.py`：移除SCP路径，简化接口

### 阶段4: 修改调用方
7. 修改4个脚本/模块：移除SCPConfig导入和参数

### 阶段5: 验证
8. 全项目grep搜索确认无残留引用
9. 确认保留模块可正常使用

---

## 9. 风险评估与回退方案

### 9.1 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 遗漏某处SCPConfig引用 | 低 | 高 | 全项目grep搜索验证 |
| 外部代码依赖已移除API | 低 | 中 | 在文档中记录API变更 |
| __init__.py延迟导入修改引入bug | 低 | 中 | 逐个验证导出符号 |
| gcs_optimizer.py中SCP配置读取代码遗漏 | 低 | 低 | 仔细检查并移除 |

### 9.2 回退方案

- Git版本控制确保可回退到清除前的任意版本
- 清除方案文档完整记录所有变更，便于审计
- 如需恢复SCP功能，可从Git历史恢复相关文件

---

## 10. 验证检查清单

- [x] ackermann_scp_solver.py 已删除
- [x] scp_optimization/ 目录已删除
- [x] 7个曲率成本模块文件已删除
- [x] ackermann_data_structures.py 中无SCP数据结构和常量
- [x] ackermann_data_structures.py 中无注释禁用的曲率成本数据结构
- [x] PlanningResult.num_iterations 语义已更新
- [x] __init__.py 中无SCP/曲率成本导出
- [x] ackermann_gcs_planner.py 中无SCP路径分支
- [x] ackermann_gcs_planner.py __init__ 无scp_config参数
- [x] 4个脚本/模块中无SCPConfig导入
- [x] 全项目grep搜索确认无活跃代码残留引用
- [x] curvature_utils.py 完整保留
- [x] curvature_statistics.py 完整保留
- [x] constraint_utils.py 完整保留
- [x] bezier.py 中曲率硬约束方法完整保留
- [x] cost_configurator.py 中曲率硬约束预设保留
