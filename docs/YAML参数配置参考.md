# YAML参数配置参考

本文说明 `config/experiments/*.yaml` 和 `config/scenarios/default.yaml` 中每个参数的用途、常见设置方式和调参建议。

配置优先级：

```text
ProjectConfig dataclass 默认值 < extends 引入的 YAML < 当前 YAML < CLI --set
```

常用命令：

```bash
python3 scripts/hybrid_astar_gcs_planner.py basic ackermann_gcs \
  --config config/experiments/default.yaml

python3 scripts/visualize_3d_trajectory.py basic \
  --set visualization.elev=35.0 \
  --set ackermann.vehicle.max_velocity=8.0

python3 scripts/batch_test_curvature_constraint.py \
  --config config/experiments/curvature_direction_cone.yaml \
  --set batch.num_runs=3
```

## 文件组织

推荐目录：

- `config/experiments/*.yaml`：实验入口，通常传给脚本的 `--config`。
- `config/scenarios/default.yaml`：共享场景库。
- `config/project/`：统一配置加载和 dataclass schema。
- `config/environments/iris_env.yaml`：Conda 环境定义，不参与规划参数加载。

`extends`

- 含义：继承其他 YAML。当前默认继承 `../scenarios/default.yaml`。
- 设置：相对当前 YAML 文件路径填写，可以是单个字符串或列表。
- 建议：新实验文件优先 `extends: default.yaml`，只覆盖关心的参数。

`planner_mode`

- 可选：`hybrid_astar_gcs`、`ackermann_gcs`。
- 设置：主脚本默认使用该模式；命令行位置参数可临时覆盖。
- 建议：调 Ackermann 车辆约束时用 `ackermann_gcs`；测试完整默认管线用 `hybrid_astar_gcs`。

## scenario / scenarios

`scenario.name`

- 含义：默认运行哪个场景。
- 设置：必须是 `scenarios` 中已有 key，例如 `basic`、`minimal`、`parking`。

`scenarios.<name>.map_size`

- 含义：栅格地图宽高，单位为 cell。
- 设置：正整数。
- 影响：越大场景越大，搜索和 IRIS 处理越慢。

`scenarios.<name>.start` / `goal`

- 字段：`x`、`y`、`heading_deg`。
- 单位：`x/y` 是米；`heading_deg` 是角度，范围 `[-180, 180]`。
- 注意：YAML 中一律写角度，不写弧度；加载器会转换为弧度。

`scenarios.<name>.corridor_width`

- 含义：该场景默认走廊宽度，单位米。
- 设置：正数。
- 建议：狭窄通道用较小值如 `2.0-5.0`；开阔或调试场景可放大。

## astar

`astar.min_radius`

- 含义：A* 车辆最小转弯半径近似值。
- 设置：正数。
- 建议：车辆越难转弯，值越大；太大可能找不到窄通道路径。

`astar.resolution`

- 含义：A* 搜索空间位置分辨率，单位米。
- 设置：正数。
- 建议：小值路径更细但更慢；大值更快但路径更粗。

`astar.theta_resolution`

- 含义：航向离散数量。
- 设置：正整数。
- 建议：常用 `16` 或 `32`；急转弯场景可增大。

`astar.max_iterations`

- 含义：搜索最大迭代次数。
- 设置：正整数。
- 建议：复杂地图失败时可增大；若运行过慢可降低。

`astar.goal_tolerance`

- 含义：到达目标的位置容差，单位米。
- 设置：正数。
- 建议：容差越小路径终点越准但搜索更难。

`astar.theta_tolerance_deg`

- 含义：到达目标的航向容差，单位度。
- 设置：正数。
- 建议：泊车、U 型转弯等终点姿态敏感场景应较小。

`astar.heuristic_weight`

- 含义：启发式权重。
- 设置：正数。
- 建议：大于 `1.0` 可加速但可能降低路径质量。

`astar.adaptive_jump`

- 含义：是否启用自适应跳步搜索。
- 设置：`true/false`。

`astar.collision_samples`

- 含义：A* 局部运动碰撞采样点数。
- 设置：正整数。
- 建议：安全性优先时增大；速度优先时减小。

`astar.high_precision_mode`

- 含义：是否启用高精度搜索策略。
- 设置：`true/false`。

`astar.path_interpolation`

- 含义：是否对粗路径插值。
- 设置：`true/false`。

`astar.verbose`

- 含义：是否输出 A* 日志。
- 设置：`true/false`。

## corridor

`corridor.width`

- 含义：全局覆盖走廊宽度；留空表示使用当前场景的 `corridor_width`。
- 设置：空值或正数。
- 建议：需要一次性改所有场景走廊宽度时设置该字段。

`corridor.smooth_path`

- 含义：生成走廊前是否平滑 A* 路径。
- 设置：`true/false`。

`corridor.smooth_window`

- 含义：路径平滑窗口。
- 设置：正整数。
- 建议：开启平滑时设置为奇数，例如 `3`、`5`。

`corridor.boundary_margin`

- 含义：走廊边界额外裕度，单位米。
- 设置：非负数。
- 建议：保守避障时增大；窄通道失败时减小。

## iris

`iris.use_iris`

- 含义：是否使用 IRIS/IrisZo 生成凸区域。
- 设置：`true/false`。

`iris.mode`

- 可选：`np`、`zo`。
- 含义：`np` 是 Drake IrisNp 路线；`zo` 是自定义 IrisZo 路线。
- 注意：当前包内规划器仍会按可用性选择实际模式，完整验证需要 Drake 环境。

`iris.config_preset`

- 含义：预留的 IRIS 配置预设名。
- 当前状态：配置中保留字段，实际适配仍主要通过下方显式参数。

`iris.iteration_limit`

- 含义：区域扩张最大迭代次数。
- 设置：正整数。
- 建议：区域过小可增大；性能差可降低。

`iris.termination_threshold`

- 含义：扩张终止阈值。
- 设置：正数。
- 建议：越小区域更精细但更慢。

`iris.configuration_space_margin`

- 含义：配置空间安全裕度，单位米。
- 设置：非负数。
- 建议：保守避障增大；窄通道失败减小。

`iris.min_seed_distance`

- 含义：种子点最小间距，单位米。
- 设置：正数。
- 建议：区域太多时增大；路径覆盖不够时减小。

`iris.max_seed_points`

- 含义：最多生成多少个种子点。
- 设置：正整数。
- 建议：长路径或复杂场景可增大。

`iris.merge_overlapping`

- 含义：是否合并重叠区域。
- 设置：`true/false`。

`iris.num_collision_infeasible_samples`

- 含义：碰撞不可行采样数量。
- 设置：正整数。
- 建议：安全性优先时增大；速度优先时减小。

`iris.requires_sample_as_member`

- 含义：采样点是否必须在区域内。
- 设置：`true/false`。

## gcs

`gcs.strategy_preset`

- 可选：`standard`、`high_risk`、`emergency`、`complex`。
- 含义：GCS 策略预设，会影响走廊、边界裕度、求解器配置等。

`gcs.cost_preset`

- 常用：`lunar_standard`、`lunar_high_risk`、`lunar_emergency`、`lunar_complex`、`time_optimal`、`path_optimal`、`energy_optimal`、`balanced`、`smooth`、`custom`。
- 含义：GCS 成本预设。
- 注意：若设为 `custom`，会使用 `gcs.cost_weights` 中的权重。

`gcs.order`

- 含义：GCS Bezier 曲线阶数。
- 设置：正整数。
- 建议：阶数越高曲线表达能力越强，求解更难；常用 `4-6`。

`gcs.continuity`

- 含义：段间连续性阶数。
- 设置：`0 <= continuity < order`。
- 建议：更高更平滑但约束更多。

`gcs.zero_velocity_at_boundaries`

- 含义：起终点是否强制零速度。
- 设置：`true/false`。
- 建议：停车/泊车设 `true`；连续行驶轨迹可设 `false`。

`gcs.min_time_derivative`

- 含义：时间轨迹导数下界，避免时间缩放过小。
- 设置：正数。
- 建议：速度/加速度数值不稳定时增大。

`gcs.curvature_constraint_mode`

- 可选：`none`、`hard`、`direction_cone`。
- 含义：包内 GCS 优化器使用的曲率模式。
- 注意：脚本直接调用 AckermannGCS 时主要看 `ackermann.constraints.curvature_constraint_mode`。

`gcs.enable_optimization`

- 含义：是否启用 GCS 优化。
- 设置：`true/false`。

### gcs.cost_weights

`time`

- 增大后更偏向短时间轨迹。

`path_length`

- 增大后更偏向短路径。

`energy`

- 增大后更抑制高速度/高能耗趋势。

`time_derivative_reg`

- 增大后更约束时间缩放导数变化，通常改善速度/加速度数值稳定性。

`regularization_r`

- 空间轨迹导数正则。增大可让路径更平滑，但可能更保守。

`regularization_h`

- 时间轨迹正则。增大可让时间分配更平滑。

`h_ref`

- 时间导数参考值。调低通常使时间尺度更大，速度更低。

`curvature_squared` / `curvature_derivative` / `curvature_peak`

- 曲率软惩罚项。默认 `0.0` 表示关闭；需要偏向低曲率轨迹时逐步增大。

### gcs.rounding

`flow_tol`

- 含义：rounding 路径提取的流量阈值。
- 建议：数值不稳时可放宽；需要更严格路径选择时减小。

`max_paths`

- 含义：每种 rounding 策略最多尝试路径数。
- 建议：成功率低时增大；速度优先时减小。

`max_trials`

- 含义：随机 rounding 最大尝试次数。

### gcs.solver

`max_time`

- 连续松弛求解最大时间，单位秒。

`mip_max_time`

- MIP/rounding 相关最大时间，单位秒。

`num_threads`

- 求解器线程数。不要超过机器核心数。

## ackermann

### ackermann.vehicle

`wheelbase`

- 车辆轴距，单位米。
- 增大后同样转角下最大曲率变小，车辆更难转弯。

`max_steering_angle_deg`

- 最大转向角，单位度，范围 `(0, 90)`。
- 增大后最大曲率变大，车辆更容易转弯；过接近 90 度会数值不稳定。

`max_velocity`

- 最大速度，单位 m/s。

`max_acceleration`

- 最大加速度，单位 m/s^2。

### ackermann.bezier

`order`

- Ackermann GCS Bezier 阶数。常用 `5`。

`continuity`

- Ackermann GCS 连续性阶数。必须小于 `order`。

`hdot_min`

- 时间导数最小值。过小可能导致速度/加速度数值不稳定。

`full_dim_overlap`

- 是否使用全维 overlap。默认 `false`。

`hyperellipsoid_num_samples_per_dim_factor`

- 超椭球采样因子。增大更保守/更慢。

`max_rounding_attempts`

- rounding 验证重试次数。

`max_rounded_paths`

- 每次 rounding 的路径数量上限。

### ackermann.constraints

`min_velocity`

- 曲率约束中使用的最小速度，单位 m/s。
- 对 `hard` 和 `direction_cone` 的保守性影响较大；增大通常更保守。

`curvature_constraint_mode`

- 可选：`none`、`hard`、`direction_cone`。
- 建议：先用 `hard` 做 baseline，再尝试 `direction_cone`。

`h_bar_prime`

- 手动指定平均时间导数。留空表示自动估计/迭代。

`h_bar_prime_safety_factor`

- h_bar_prime 保守系数，范围 `(0, 1]`。越小越保守。

`max_h_bar_prime_iterations`

- h_bar_prime 最大迭代次数。设为 `1` 基本等于不迭代。

`h_bar_prime_convergence_threshold`

- h_bar_prime 相对收敛阈值。

`h_bar_prime_relax_factor`

- 求解失败时 h_bar_prime 放宽因子。

`max_h_bar_prime_relax_attempts`

- h_bar_prime 放宽重试次数。

`h_bar_prime_safety_factor_decay`

- 迭代中自动收紧 safety factor 的衰减系数。

### direction_cone

`ackermann.direction_cone_profile`

- 可选：`default`、`loose`、`selective`，也可以新增自定义 profile。
- `default`：使用 `TrajectoryConstraints` 内部默认值。
- `loose`：更宽松的方向锥。
- `selective`：在 `loose` 基础上跳过几何风险边，当前推荐调试值。

`direction_cone_alpha` / `beta` / `gamma`

- direction cone 线性充分条件参数。
- 建议：不熟悉算法时先调 profile，不单独调这三个值。

`direction_cone_theta_min_deg`

- 最小方向锥角，单位度。
- 增大可放宽方向偏差容忍，但过大可能降低约束有效性。

`direction_cone_theta_abs_max_deg`

- 方向锥角绝对上限，单位度。

`direction_cone_theta_margin_deg`

- 方向锥角安全裕度，单位度。

`direction_cone_width_mu`

- 用于宽度条件的系数。增大通常更宽松。

`direction_cone_rho_warning_ratio`

- rho 风险提示比例。

`direction_cone_skip_risk_flags`

- 跳过哪些高风险边。常用：
  - `direction_mismatch`
  - `parallel_width_degenerate`
  - `parallel_width_small`
  - `path_projection_degenerate`
  - `overlap_infeasible`
  - `overlap_unavailable`
  - `theta_width_below_min`

`ackermann.verbose`

- Ackermann 规划过程是否输出详细日志。

## visualization

`enabled`

- 是否启用可视化。

`save`

- 是否保存图片。

`output_dir`

- 输出目录。

`num_samples`

- 轨迹采样点数。增大曲线更细，绘图/评估更慢。

`show_2d`

- 3D 可视化脚本中是否同时生成 2D 图。

`show_control_points`

- 是否显示控制点。

`show_iris_regions` / `show_obstacles` / `show_corridor` / `show_astar_path`

- 控制综合可视化中的图层显示。

`show_3d_trajectory`

- 是否显示 3D 轨迹。

`show_theta_profile`

- 是否显示 theta 曲线。

`elev` / `azim`

- 3D 图初始视角，单位度。

`figsize`

- Matplotlib 图尺寸，例如 `[20.0, 14.0]`。

`dpi`

- 输出图片分辨率。

`auto_save`

- 是否自动保存 2D/3D 输出。

`control_point_size`

- 控制点标记大小。

`control_point_color`

- 控制点颜色，使用 Matplotlib 颜色名或十六进制颜色。

`control_point_marker`

- 控制点 marker，例如 `D`、`o`、`x`。

## batch

`batch.scenario`

- 批量测试默认场景。

`batch.num_runs`

- 批量测试运行次数。

`batch.quiet_each_run`

- 是否隐藏每次规划内部输出，只打印统计结果。

## 调参速查

- A* 找不到路径：增大 `astar.max_iterations`、放宽 `astar.goal_tolerance`、减小 `astar.resolution`，或检查场景障碍物。
- IRIS 区域太少/覆盖不足：增大 `iris.max_seed_points`，减小 `iris.min_seed_distance`，适当减小 `iris.configuration_space_margin`。
- 窄通道失败：减小 `corridor.width` 或场景 `corridor_width`，减小 `iris.configuration_space_margin`。
- 轨迹曲率超限：使用 `ackermann.constraints.curvature_constraint_mode=hard`，增大 `regularization_r`，降低 `max_velocity`。
- 求解太慢：降低 `gcs.order`、`gcs.rounding.max_paths`、`gcs.rounding.max_trials`，或使用更快的 GCS/IRIS 预设。
- 速度/加速度抖动：增大 `time_derivative_reg`、`regularization_h`、`gcs.min_time_derivative`，或降低 `h_ref`。
