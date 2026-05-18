# Direction Cone Curvature Branch Test Guide

本文档用于在另一台设备上拉取 `codex/direction-cone-curvature` 分支，并验证方向锥曲率约束实现。

## 1. 克隆测试分支

新设备尚未克隆仓库时：

```bash
git clone -b codex/direction-cone-curvature https://github.com/sjidnsk/a_gcs_ws-2.0.1.git
cd a_gcs_ws-2.0.1
```

如果仓库已经存在：

```bash
cd a_gcs_ws-2.0.1
git fetch origin
git switch codex/direction-cone-curvature
git pull
```

确认当前分支：

```bash
git status -sb
```

预期看到：

```text
## codex/direction-cone-curvature...origin/codex/direction-cone-curvature
```

## 2. 推荐运行环境

完整规划链路建议在 Ubuntu 中运行。当前项目依赖 Drake，而 Drake 不支持 Windows。

推荐环境：

```bash
conda env create -f config/environments/iris_env.yaml
conda activate iris-py3.12
python scripts/verify_environment.py
```

如果环境已经创建过：

```bash
conda activate iris-py3.12
python scripts/verify_environment.py
```

## 3. 先跑单元测试

先运行全部单元测试：

```bash
pytest tests/unit/ -v
```

重点关注方向锥相关测试：

```bash
pytest tests/unit/test_directional_curvature_config.py -v
pytest tests/unit/test_directional_curvature_parameters.py -v
pytest tests/unit/test_directional_curvature_math.py -v
pytest tests/unit/test_directional_curvature_bezier_constraints.py -v
```

说明：

- `test_directional_curvature_config.py` 验证配置模式和参数校验。
- `test_directional_curvature_parameters.py` 验证路径预处理、support 宽度、参数生成和线性约束行。
- `test_directional_curvature_math.py` 验证方向锥充分条件的曲率上界性质。
- `test_directional_curvature_bezier_constraints.py` 依赖 Drake，应在 Ubuntu `iris-py3.12` 中运行。

## 4. 开启 direction_cone 模式

当前脚本通过 YAML 或命令行覆盖切换模式。

推荐使用配置文件：

```bash
python scripts/batch_test_curvature_constraint.py \
  --config config/experiments/curvature_direction_cone.yaml
```

也可以临时覆盖：

```bash
python scripts/batch_test_curvature_constraint.py \
  --set ackermann.constraints.curvature_constraint_mode=direction_cone
```

`ackermann.constraints.curvature_constraint_mode` 可选值：

```text
"none"
"hard"
"direction_cone"
```

建议先跑 `"hard"` 作为 baseline，再跑 `"direction_cone"` 做对比。

## 5. 批量场景测试

运行批量曲率约束测试：

```bash
python scripts/batch_test_curvature_constraint.py
```

重点记录：

- 成功率；
- GCS 求解时间；
- 后验最大曲率；
- 曲率违反量；
- 速度、加速度、工作空间约束违反量；
- 是否触发 fallback。

## 6. 端到端 smoke test

运行 Hybrid A* -> IRIS/IrisZo -> Ackermann GCS 链路：

```bash
python scripts/hybrid_astar_gcs_planner.py
```

该脚本可能依赖：

- Drake；
- MOSEK/Gurobi 或可用替代求解器；
- 图形后端；
- 优化器许可证。

若图形显示失败，优先确认 Matplotlib 后端；若求解器失败，先确认 `scripts/verify_environment.py` 输出。

## 7. 判断测试结果

`direction_cone` 模式的关键预期：

- 不依赖 `h_bar_prime` 迭代；
- 正常情况下只执行单次 GCS 求解流程；
- 新增曲率约束为线性约束；
- 后验曲率不超过 `kappa_max` 容差范围；
- 参数生成失败、求解失败或后验曲率违反时，应只触发一次 fallback 到 `"hard"`。

若 fallback 频繁触发，应重点检查：

- 粗路径是否锯齿严重；
- 局部路径段是否过短；
- IRIS 区域是否过窄或 overlap 过小；
- `rho` 是否被风险阈值标记为过低；
- `theta_max` 是否被区域横纵宽度限制得过小。

## 8. 测试后建议提交的信息

在测试设备上记录以下信息，便于回传问题：

```bash
git rev-parse --short HEAD
python scripts/verify_environment.py
pytest tests/unit/ -v
python scripts/batch_test_curvature_constraint.py --config config/experiments/curvature_direction_cone.yaml
```

如果端到端失败，请同时保留：

- 失败脚本名；
- 场景名；
- 当前 `ackermann.constraints.curvature_constraint_mode`；
- traceback；
- 后验曲率、速度、加速度、工作空间约束报告；
- 是否触发 fallback 及 fallback reason。
