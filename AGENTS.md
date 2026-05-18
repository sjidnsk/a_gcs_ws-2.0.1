# AGENTS.md

本文档记录当前仓库的项目认知与协作约定，供后续 Codex/代理或开发者快速接手。内容基于当前目录结构、源码入口、配置文件、脚本和文档梳理得到。

## 0. 指令优先级与安全底线

后续代理或开发者应按以下优先级合并使用本文件、任务说明和通用 LLM 行为准则：

1. 安全底线优先，尤其是文件删除限制；即使当前任务提出相反要求，也应停止并请求用户手动处理。
2. 用户当前任务中的明确目标优先，但不得覆盖本仓库的安全底线、运行环境目标和模块边界。
3. 本仓库项目约束优先于通用 `CLAUDE.md`、`GEMINI.md` 或其他代理行为准则。
4. 通用行为准则仅补充工作方式，不得覆盖本项目的环境、模块边界、验证和安全限制。

文件删除安全规则：

- 禁止批量删除文件或目录。
- 不要使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。
- 需要删除文件时，只能一次删除一个明确路径的文件，例如 `Remove-Item "C:\path\to\file.txt"`。
- 如果任务需要批量删除，应停止操作并请求用户手动处理；不要用脚本或 shell 命令代替用户批量删除。

## 1. 项目定位

- 项目名称：`a_gcs_ws`
- 当前版本：`2.0.1`，见 `setup.py`
- Python 包布局：`src/` 源码布局，`setup.py` 使用 `find_packages(where="src")`
- 主要目标：面向 Ackermann 转向车辆的分层路径规划与轨迹优化。
- 核心思想：先用 SE(2) 空间搜索生成粗路径，再生成局部走廊，随后通过 IRIS/IrisZo 生成凸可行区域，最后用 GCS 与 Bezier/B-spline 轨迹在约束下优化可行轨迹。

## 2. 推荐理解路径

建议按以下顺序阅读项目：

1. `docs/系统架构分析报告.md`
2. `docs/核心模块深度分析报告.md`
3. `docs/数据流与交互逻辑分析报告.md`
4. `docs/pydrake_project_reference.md`：PyDrake 官方文档项目速查，用于查询 `HPolyhedron`、IRIS/IrisZo、GCS、B-spline 和求解器接口。
5. `src/path_planner/planner.py`
6. `scripts/hybrid_astar_gcs_planner.py`
7. `src/ackermann_gcs_pkg/ackermann_gcs_planner.py`
8. `src/ackermann_gcs_pkg/ackermann_bezier_gcs.py`
9. `src/A_pkg/`、`src/C_space_pkg/`、`src/iris_pkg/`、`src/iriszo/`、`src/gcs_pkg/`

注意：部分中文文档或注释在当前 PowerShell 输出中显示为乱码，但文件结构和代码符号仍可识别。修改文档前应确认文件编码，避免把乱码继续写回源文档。

## 3. 总体架构

项目是分层路径规划系统，主要数据流如下：

```text
地图/障碍物 + 起终点
  -> A* / Hybrid A* 在 SE(2) 中搜索粗路径
  -> C-space 与局部走廊生成
  -> IRIS / IrisZo 生成凸可行区域
  -> GCS / Ackermann GCS 优化轨迹
  -> 可视化、性能统计与约束评估
```

主要层次：

- 全局搜索层：`A_pkg`
- 配置空间与走廊层：`C_space_pkg`
- 凸区域生成层：`iris_pkg` 和 `iriszo`
- GCS 优化层：`gcs_pkg`
- Ackermann 车辆约束层：`ackermann_gcs_pkg`
- 编排层：`path_planner`
- 可视化层：`visualization`
- 配置层：`config`

## 4. 目录与模块职责

### 4.1 `src/A_pkg`

SE(2) 路径搜索模块。

- `A_star_base.py`
  - `PlannerConfig`
  - `SearchNode`
  - `ReedsSheppSegment`
  - `BaseSE2Planner`
  - `normalize_angle`
  - `compute_distance`
- `A_star_fast_optimized.py`
  - `FastSE2AStarPlanner`
  - `BidirectionalSE2AStarPlanner`

该模块负责在带朝向的 SE(2) 空间中搜索粗路径，并为后续走廊生成提供路径骨架。

### 4.2 `src/C_space_pkg`

配置空间、机器人形状、障碍物处理和局部走廊生成模块。

- `se2.py`
  - `RobotShape`
  - `SE2ConfigurationSpace`
  - `SE2Visualizer`
  - `create_circle_robot`
  - `create_rectangle_robot`
  - `create_polygon_robot`
- `partial_corridor.py`
  - `CorridorConfig`
  - `CorridorResult`
  - `PathSmoother`
  - `CorridorGenerator`
  - `AStarCorridorPlanner`
  - `plan_and_create_corridor`
- `obstacles_optimized.py`
  - 二值地图到凸障碍物的转换
  - 非凸障碍物分解
  - 点在多边形内判断和凸性估计

该模块是 A* 输出与 IRIS/GCS 输入之间的桥接层。

### 4.3 `src/iris_pkg`

基于 Drake IrisNp 的凸区域生成模块。

公共入口在 `src/iris_pkg/__init__.py` 中导出：

- `IrisNpConfig`
- `IrisNpConfigOptimized`
- `IrisNpRegion`
- `IrisNpResult`
- `SimpleCollisionCheckerForIrisNp`
- `IrisNpRegionGenerator`
- `visualize_iris_np_result`
- `check_drake_availability`

核心目录：

- `core/iris_np_region.py`：主区域生成器
- `core/iris_np_expansion.py`：区域扩张
- `core/iris_np_collision.py`：碰撞检查
- `core/iris_np_region_pruner.py`：区域修剪
- `core/iris_np_coverage_checker.py`：覆盖验证
- `core/iris_np_voronoi_optimizer.py`：Voronoi 种子优化

此模块依赖 Drake，可通过 `check_drake_availability()` 检查可用性。

### 4.4 `src/iriszo`

自定义 IrisZo 零阶优化凸区域生成模块。

公共入口在 `src/iriszo/__init__.py` 中导出：

- `IrisZoConfig`
- `IrisZoRegion`
- `IrisZoResult`
- `IrisZoRegionGenerator`
- `CustomIrisZoAlgorithm`
- `CollisionCheckerAdapter`
- `HitAndRunSampler`
- `BisectionSearcher`
- `SeparatingHyperplaneGenerator`
- `IrisZoSeedExtractor`
- `CoverageValidator`
- `EnhancedCoverageValidator`
- `RegionPruner`
- `PerformanceReporter`
- `visualize_iriszo_result`

核心目录：

- `core/iriszo_algorithm.py`：主算法
- `core/iriszo_region.py`：区域生成器
- `core/iriszo_collision.py`：碰撞检查适配
- `core/iriszo_sampler.py` 与 `core/iriszo_sampler_jit.py`：采样
- `core/iriszo_bisection.py`：二分搜索
- `core/iriszo_hyperplane.py`：分离超平面生成
- `core/iriszo_coverage*.py`：覆盖率验证
- `core/iriszo_pruning.py`：区域修剪
- `core/iriszo_performance.py`：性能采集和报告

该模块在设计上可作为 IrisNp 的替代或补充，用于不完全依赖 Drake 原生 IrisNp 的场景。

### 4.5 `src/gcs_pkg`

通用 Graph of Convex Sets 轨迹优化实现。

核心文件：

- `scripts/core/base.py`
  - `BaseGCS`
  - 凸集维度、交集维度、冗余相关工具
- `scripts/core/bezier.py`
  - `BezierGCS`
  - `BezierTrajectory`
- `scripts/core/linear.py`
  - `LinearGCS`
- `scripts/core/phi_updater.py`
  - `IncrementalPhiUpdater`
- `scripts/rounding/rounding.py`
  - greedy/random forward/backward path extraction
  - MIP path extraction
  - average vertex position extraction
- `scripts/utils/preprocessing.py`
  - `removeRedundancies`

该模块为 Ackermann GCS 提供基础 GCS 能力。

### 4.6 `src/ackermann_gcs_pkg`

Ackermann 车辆轨迹规划核心模块，叠加车辆运动学、曲率、速度、加速度和连续性约束。

公共入口在 `src/ackermann_gcs_pkg/__init__.py` 中通过延迟导入导出：

- `VehicleParams`
- `EndpointState`
- `TrajectoryConstraints`
- `BezierConfig`
- `PlanningResult`
- `TrajectoryReport`
- `AckermannBezierGCS`
- `AckermannGCSPlanner`
- `TrajectoryEvaluator`
- `FlatOutputMapper`
- `iterate_h_bar_prime`

核心文件：

- `ackermann_data_structures.py`：车辆参数、端点状态、约束、规划结果、报告结构
- `ackermann_gcs_planner.py`：高层 Ackermann GCS 规划器
- `ackermann_bezier_gcs.py`：基于 BezierGCS 的 Ackermann 扩展
- `flat_output_mapper.py`：平坦输出映射
- `rotation_matrix_heading_constraint.py`：航向约束建模
- `curvature_utils.py`：曲率和曲率梯度计算
- `constraint_utils.py`：约束违反检测
- `trajectory_evaluator.py`：轨迹评估
- `trajectory_utils.py`：采样和导数计算
- `h_bar_prime_iteration.py`：时间导数相关迭代
- `numerical_safety_utils.py`：数值安全工具

该模块是最终轨迹可行性和车辆约束的主要实现位置。

### 4.7 `src/path_planner`

端到端编排层。

- `planner.py`
  - `HybridAStarGCSPlanner`
  - 检查 IRIS 模块可用性
  - 创建走廊生成器、IRIS 生成器、GCS 优化器和可视化器
  - 处理 A* 路径到 IRIS/GCS 输出的管线
- `support/`
  - `performance_monitor.py`：性能指标
  - `gcs_optimizer.py`：GCS 优化适配
  - `__init__.py`：导出 `PlannerConfig`、`PlannerResult`、`PerformanceMonitor`、`GCSOptimizer`
- `scenario_utils.py`
  - 测试场景地图、A* 路径规划、IRIS 区域转换和 Ackermann 端点状态等共享脚本工具

注意：`src/path_planner/planner.py` 是包内编排实现；根目录 `scripts/hybrid_astar_gcs_planner.py` 是命令行测试入口。脚本共享工具放在 `src/path_planner/scenario_utils.py`。

### 4.8 `src/visualization`

可视化与输出管理模块。

- `core/`
  - `base_visualizer.py`
  - `models.py`
  - `output_manager.py`
  - `path_builder.py`
- `trajectory/trajectory_visualizer.py`
- `environment/visualize_environment_with_bezier.py`
- `ackermann/`
  - `ackermann_visualizer.py`
  - `ackermann_visualizer_enhanced.py`
  - `plot_2d_trajectory.py`
  - `plot_3d_trajectory.py`
  - `plot_profiles.py`
  - `path_comparator.py`
  - `trajectory_sampler.py`
  - `region_renderer.py`
  - `control_point_extractor.py`

可视化配置位于 `config/visualization/`。

### 4.9 `config`

统一配置入口，推荐从子模块直接导入，避免循环导入。

- `config/iris/`
  - `IrisNpConfig`
  - `IrisNpConfigOptimized`
  - `get_high_safety_config`
  - `get_fast_processing_config`
  - `get_balanced_config`
- `config/planner/planner_config.py`
  - `PlannerConfig`
  - `PlannerResult`
  - 包含走廊、IRIS、GCS、Ackermann、可视化和性能监控配置
- `config/gcs/`
  - `LunarRoverGCSConfig`
  - `CostConfigurator`
  - `CostWeights`
  - 月面车/复杂地形相关 GCS 策略预设
- `config/solver/`
  - `AdaptiveSolverConfig`
  - `SolverPerformanceProfile`
  - `SolverType`
  - `ProblemSize`
  - MOSEK/Gurobi/CLP/SCS 等求解器配置
- `config/visualization/`
  - `VisualizationConfig`
  - `ControlPointConfig`
  - `PlotConfig`
- `config/environments/iris_env.yaml`
  - Conda 环境定义，环境名为 `iris-py3.12`

## 5. 脚本入口

根目录 `scripts/` 包含主要运维、验证和端到端运行脚本：

- `scripts/setup_iris_env.py`
  - 创建/验证 Conda 环境
  - 支持 `--env-name` 和 `--no-verify`
- `scripts/setup_iris_env.sh`
  - Shell 版本环境部署脚本
- `scripts/verify_environment.py`
  - 检查 NumPy、SciPy、Matplotlib、PyTest、Drake、CVXPY、Clarabel、SCS、OSQP、OpenCV、Numba 等依赖
- `scripts/hybrid_astar_gcs_planner.py`
  - 端到端 Hybrid A* + IRIS + Ackermann GCS 测试入口
  - 包含 `SCENARIO_CONFIGS`、`DEFAULT_VEHICLE_PARAMS`、`create_test_map`、`plan_path` 等辅助函数
- `scripts/batch_test_curvature_constraint.py`
  - 批量测试曲率硬约束规划成功率
- `scripts/visualize_3d_trajectory.py`
  - 3D 轨迹可视化入口

## 6. 环境与依赖

推荐环境来自 `config/environments/iris_env.yaml`：

- Conda 环境名：`iris-py3.12`
- Python：`3.12.12`
- Drake：`1.51.1`
- 项目运行需要安装 Drake，但 Drake 没有 Windows 版本。
- 当前电脑是 Windows 系统，因此本机 Conda 环境 `iris-py3.12` 中未安装 Drake，不能作为完整规划链路的目标运行环境。
- 此项目需要在 Ubuntu 系统的 Conda 环境 `iris-py3.12` 中运行；Ubuntu 环境中安装了 Drake。
- 设计、代码兼容性、路径处理和完整验证方案应以 Ubuntu 运行环境为目标；Windows 本机仅用于静态编辑、轻量检查和不依赖 Drake 的局部验证。
- 主要依赖：
  - NumPy
  - SciPy
  - Matplotlib
  - CVXPY
  - Clarabel
  - SCS
  - OSQP
  - Mosek
  - OpenCV
  - scikit-image
  - Numba
  - Rtree
  - PyYAML
  - pytest

常用环境命令：

```bash
conda env create -f config/environments/iris_env.yaml
conda activate iris-py3.12
python scripts/verify_environment.py
```

Windows 本机仅适合做不依赖 Drake 的静态编辑、轻量检查或纯 Python 局部验证；完整规划链路、IRIS/GCS 相关验证应在 Ubuntu 的 `iris-py3.12` 环境中执行。
运行项目前优先在 Ubuntu 中激活 `iris-py3.12`，不要在未安装 Drake 的 Windows Conda 环境或基础 Python 环境中直接运行规划链路。

## 7. 常用验证命令

在 Ubuntu 的 `iris-py3.12` 环境中修改运行逻辑前后，优先使用以下命令：

```bash
python scripts/verify_environment.py
pytest
pytest tests/unit/ -v
python scripts/batch_test_curvature_constraint.py
```

针对端到端链路，可从以下脚本入手：

```bash
python scripts/hybrid_astar_gcs_planner.py
python scripts/visualize_3d_trajectory.py
```

实际运行前应先确认当前环境中 Drake、Mosek 或替代求解器是否可用。某些脚本可能依赖图形后端或优化器许可证。文档变更或 Windows 本机静态编辑不要求运行完整规划链路，但最终说明应明确未运行的验证项及原因。

## 8. 测试现状

- `tests/conftest.py` 会把 `src/` 加入 `sys.path`，并尝试导入 `visualization` 包。
- `tests/obstacle_utils.py` 提供障碍物地图构造工具，包括 `ObstacleConfig`、`ObstacleMapBuilder`、`create_standard_test_map` 和 `safe_obstacle_set`。
- `tests/unit/` 当前仅见初始化文件，单元测试覆盖看起来还不完整。
- 更接近集成测试的脚本在 `scripts/batch_test_curvature_constraint.py` 和 `scripts/hybrid_astar_gcs_planner.py`。

新增功能或修复时，优先补充可独立运行的小测试；涉及规划链路时，再跑集成脚本。

## 9. 重要实现细节

### 9.1 导入路径

项目大量脚本会手动调整 `sys.path`，以便从仓库根目录或脚本目录运行。新增模块时应优先保持 `src/` 包布局，不要扩大路径 hack 的范围。

推荐导入方式：

```python
from config.iris import IrisNpConfig
from config.planner import PlannerConfig
from ackermann_gcs_pkg import AckermannGCSPlanner
```

`config/__init__.py` 和 `ackermann_gcs_pkg/__init__.py` 都使用延迟导入来降低循环依赖风险。

### 9.2 IRIS 模式

规划编排层会检查 `iris_pkg` 和 `iriszo` 可用性。源码中存在 `np` 与 `zo` 两套模式：

- `np`：Drake IrisNp 路线，依赖 Drake。
- `zo`：自定义 IrisZo 路线，包含采样、二分、分离超平面、覆盖验证、性能统计等。

修改 IRIS 相关逻辑时必须确认 `IrisNpRegion` 与 `IrisZoRegion` 输出是否能被后续 `convert_iris_to_hpolyhedron` 或 GCS 接收。

### 9.3 GCS 与 Ackermann 约束

GCS 基础能力在 `gcs_pkg`，Ackermann 车辆约束在 `ackermann_gcs_pkg`。两者边界不要混淆：

- 通用图凸集、Bezier 轨迹和 rounding 逻辑放在 `gcs_pkg`。
- 车辆参数、端点状态、曲率/速度/加速度约束、航向约束和轨迹报告放在 `ackermann_gcs_pkg`。

### 9.4 数值稳定性

`ackermann_gcs_pkg/numerical_safety_utils.py` 提供安全除法、开方、范数、数组边界和速度检查等工具。涉及曲率、导数、速度或除零风险时，优先复用这些工具。

### 9.5 性能监控

`path_planner/support/performance_monitor.py` 和 `iriszo/core/iriszo_performance.py` 均包含性能采集能力。修改规划链路时应保留或扩展现有性能字段，避免只返回裸结果。

## 10. 文档现状

`docs/` 包含较完整的架构和算法文档：

- `系统架构分析报告.md`
- `核心模块深度分析报告.md`
- `数据流与交互逻辑分析报告.md`
- `IrisZo算法技术文档.md`
- `iriszo_vs_iris_pkg_comparison.md`
- `environment_deployment.md`
- `AckermannGCS约束与成本设计详解.md`

注意事项：

- 文档标题和结构可读，但当前终端读取中文时存在编码乱码。
- 修改文档前先确认编码；必要时使用能正确处理 UTF-8 的编辑器或脚本。
- 不要直接复制终端中的乱码内容回文档。

## 11. 协作约定

后续代理在本仓库工作时遵守以下约定：

1. 先读相关模块和配置，再改代码；不要只根据文件名推断行为。
2. 不要随意移动 `src/` 包结构；当前项目依赖该布局。
3. 项目运行依赖 Drake；默认目标环境是 Ubuntu 系统中已安装 Drake 的 Conda 环境 `iris-py3.12`。当前 Windows 电脑的 `iris-py3.12` 未安装 Drake，不应作为完整运行环境。
4. 修改规划链路时，同时检查：
   - 输入地图/障碍物格式
   - 路径点格式
   - IRIS 区域格式
   - GCS 所需的 `HPolyhedron` 或等价凸集格式
   - Ackermann 端点和约束格式
5. 涉及 Drake、Mosek、Gurobi 等外部依赖时，先运行环境检查或提供明确降级路径。
6. 涉及数值优化时，避免只看 `success` 标志；还要检查轨迹报告中的曲率、速度、加速度、工作空间约束违反量。
7. 可视化脚本可能打开窗口或依赖后端；批量测试应使用非交互后端。
8. 处理中文文档和注释时注意编码，避免引入二次乱码。
9. 新增测试优先放在 `tests/unit/`；端到端或性能验证可以放在 `scripts/`，但要说明运行成本和依赖。

### 11.1 通用编程行为准则

以下准则来自 `CLAUDE.md` 风格的通用 LLM 编程约束，已经按本项目特点调整。它们用于减少误改、过度设计和未验证结论。

1. 修改前先说明关键假设、目标和验证方式。若需求、数据格式、运行环境或模块边界存在多种合理解释，应先指出分歧；无法从现有代码和文档确认时再向用户提问。
2. 优先选择能满足当前目标的最小实现。不要添加未要求的功能、抽象层、配置项或兼容路径；如果改动明显可以更短更清晰，应先简化。
3. 只改任务直接需要的代码和文档。不要顺手重构、移动目录、统一格式或清理无关死代码；如果发现无关问题，记录给用户即可。
4. 清理自己引入的未使用导入、变量、函数和临时文件，但不要删除任务开始前已经存在的可疑代码或文件，除非用户明确要求。
5. 匹配当前模块风格和边界：通用 GCS 逻辑放在 `gcs_pkg`，车辆约束放在 `ackermann_gcs_pkg`，IRIS/IrisZo 逻辑保持各自输出协议清晰。
6. 将任务转成可验证目标：修 bug 时优先补充能复现问题的测试；新增逻辑时优先补充小范围单元测试；重构时确认改动前后的行为一致。
7. 不要只凭 `success` 标志判断规划或优化成功。涉及数值优化时，还应检查曲率、速度、加速度、工作空间约束违反量和求解器状态。
8. 涉及曲率、导数、速度、除零或边界数组访问时，不要把异常情况视为“不可能”；优先复用 `ackermann_gcs_pkg/numerical_safety_utils.py` 中的安全工具。
9. 在当前 Windows 机器上只能声明完成静态编辑、轻量检查或不依赖 Drake 的局部验证。完整 IRIS/GCS/Ackermann 规划链路应在 Ubuntu 的 `iris-py3.12` 环境中验证。
10. 最终说明应区分“已验证通过”和“因环境限制未运行”。不要在没有证据时声称测试、规划链路或求解器行为已经通过。

## 12. 变更建议优先级

当前项目后续改进建议按优先级排序：

1. 修复或确认中文文档编码，保证架构文档可维护。
2. 增加 `tests/unit/` 下的核心模块单元测试。
3. 固化一个最小端到端 smoke test，覆盖 A* -> corridor -> IRIS/IrisZo -> Ackermann GCS。
4. 维护 `scripts/hybrid_astar_gcs_planner.py` 与 `src/path_planner/planner.py` 的职责边界。
5. 梳理求解器依赖和许可证要求，提供 MOSEK 不可用时的默认路径。
6. 为 IrisNp 与 IrisZo 输出建立统一适配层或协议文档。
