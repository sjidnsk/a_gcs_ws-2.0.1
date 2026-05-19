# Ubuntu Level 3/4 Gear Support 测试建议

本文档用于把当前 Level 3/4 gear support 改动迁移到 Ubuntu 环境中验证。完整 IRIS/GCS/Ackermann 链路依赖 Drake，目标环境应使用 Ubuntu 的 `iris-py3.12` Conda 环境；Windows 只适合做静态检查和不依赖 Drake 的局部验证。

## 1. 迁移代码到 Ubuntu

推荐先把当前改动提交到一个测试分支，再在 Ubuntu 拉取该分支。这样比手动复制文件更容易复现问题。

```bash
git status -sb
git switch -c codex/level3-level4-gear-support
git add .
git commit -m "Implement level3 level4 gear support"
git push -u origin codex/level3-level4-gear-support
```

在 Ubuntu 上：

```bash
git clone -b codex/level3-level4-gear-support <repo-url>
cd a_gcs_ws-2.0.1
git status -sb
```

如果暂时不想提交，也可以在 Windows 导出 patch，再在 Ubuntu 应用：

```bash
git diff --binary > level3_level4_gear.patch
git apply level3_level4_gear.patch
```

## 2. 推荐 Ubuntu 环境

建议使用 Ubuntu 22.04 或 24.04 x86_64。Drake Python wheel 不支持 Windows，因此不要在 Windows 上判定完整 GCS 测试是否通过。

基础工具：

```bash
sudo apt update
sudo apt install -y git build-essential graphviz
```

创建或更新 Conda 环境：

```bash
conda env create -f config/environments/iris_env.yaml
conda activate iris-py3.12
python -m pip install -e .
```

如果环境已存在：

```bash
conda activate iris-py3.12
python -m pip install -e .
```

无显示器或远程 SSH 环境建议设置非交互 Matplotlib 后端：

```bash
export MPLBACKEND=Agg
```

## 3. 环境验证

先跑项目环境检查：

```bash
python scripts/verify_environment.py
```

再单独确认 Drake 和 GCS API 可用：

```bash
python - <<'PY'
import pydrake
from pydrake.geometry.optimization import GraphOfConvexSets, HPolyhedron
from pydrake.solvers import MosekSolver
print("pydrake ok")
print("Mosek available:", MosekSolver().available())
PY
```

如果 MOSEK 不可用或许可证缺失，很多建图单测仍可跑，但端到端求解可能失败。需要确认 `MOSEKLM_LICENSE_FILE` 或默认 license 路径是否配置正确。

## 4. 建议测试顺序

先跑不依赖求解器许可证的静态和单元测试，再跑 Drake/GCS 约束测试，最后跑集成脚本。

```bash
python -m compileall src/ackermann_gcs_pkg src/A_pkg src/path_planner config scripts tests/unit
```

Gear 基础和 A* 输出：

```bash
pytest tests/unit/test_gear_annotations.py -v
pytest tests/unit/test_astar_gear_output.py -v
```

Direction-cone 既有测试：

```bash
pytest tests/unit/test_directional_curvature_config.py \
       tests/unit/test_directional_curvature_parameters.py \
       tests/unit/test_directional_curvature_math.py -v
```

Drake 相关测试：

```bash
pytest tests/unit/test_gear_aware_heading_constraints.py -v
pytest tests/unit/test_gear_layered_gcs.py -v
pytest tests/unit/test_directional_curvature_bezier_constraints.py -v
```

如果这些通过，再跑完整单测：

```bash
pytest tests/unit/ -v
```

## 5. Level 3/4 集成验证

Level 3 固定参考 gear：

```bash
python scripts/batch_test_curvature_constraint.py \
  --config config/experiments/curvature_direction_cone_level3.yaml
```

Level 4 layered GCS 自主选 gear：

```bash
python scripts/batch_test_curvature_constraint.py \
  --config config/experiments/curvature_direction_cone_level4.yaml
```

建议每次记录以下字段：

- success / feasible 比例
- solve time 均值和最大值
- curvature / velocity / acceleration / workspace violation
- `gear_diagnostics`
- reverse 比例
- switch 次数
- fallback 次数和 `fallback_reason`

## 6. 端到端 smoke test

先确认默认配置不改变 Level 2 行为：

```bash
python scripts/hybrid_astar_gcs_planner.py \
  --config config/experiments/default.yaml
```

然后分别验证 Level 3 和 Level 4：

```bash
python scripts/hybrid_astar_gcs_planner.py \
  --config config/experiments/curvature_direction_cone_level3.yaml

python scripts/hybrid_astar_gcs_planner.py \
  --config config/experiments/curvature_direction_cone_level4.yaml
```

如果需要检查 gear 分段输出：

```bash
python scripts/visualize_3d_trajectory.py \
  --config config/experiments/curvature_direction_cone_level4.yaml
```

远程环境下如不需要弹窗，继续使用 `MPLBACKEND=Agg`。

## 7. 预期结果

默认配置：

- `gear_strategy: none`
- 不应出现强制使用 A* gear 的行为
- 轨迹速度字段 `velocity` 仍是非负 speed

Level 3：

- `gear_strategy: fixed_reference`
- 如果 reference path 带四元组 `(x, y, theta, gear)`，使用显式 gear
- 如果 reference path 只有三元组，则从路径运动方向推断 gear
- reverse/switch 只表现为成本和诊断，不应成为 A* gear 硬约束

Level 4：

- `gear_strategy: layered`
- 每个物理区域应有 forward/reverse 两层 vertex
- moving edge 保留 direction-cone / 连续性 / 速度等约束
- switch edge 应是 stationary segment
- direction-cone 不应加到 switch edge 上
- GCS 可以自主选择 forward/reverse 层，不依赖 A* gear

## 8. 常见问题排查

### Drake 导入失败

确认当前环境是 Ubuntu 的 `iris-py3.12`：

```bash
which python
python -c "import pydrake; print('ok')"
```

不要使用 Windows 或 base Python 环境跑完整链路。

### pytest 找不到项目包

优先安装 editable package：

```bash
python -m pip install -e .
```

临时方式：

```bash
export PYTHONPATH="$PWD/src"
```

### MOSEK 不可用

先确认是否只是许可证问题：

```bash
python - <<'PY'
from pydrake.solvers import MosekSolver
s = MosekSolver()
print("available:", s.available())
print("enabled:", s.enabled())
PY
```

如果 `available=True` 但求解失败，检查 MOSEK license。若没有 license，先跑不需要实际求解的单元测试，再决定是否改用项目支持的替代求解器路径。

### Level 4 构图失败后 fallback

查看 `PlanningResult.fallback_reason` 和 `gear_diagnostics.fallback_reason`。常见原因包括：

- IRIS 区域为空或维度不一致
- 区域 overlap 过小导致没有 moving edge
- source/target 点不在任何区域中
- Drake API 或求解器版本不匹配

### switch edge 没有速度为 0

重点检查：

- switch edge metadata 中 `is_switch=True`
- switch edge 的空间控制点等式是否添加
- C1 连续性是否启用
- `gear_switch_requires_stationary=True`

## 9. 建议的最小验收记录

建议把以下输出贴回问题或提交说明中：

```text
OS:
Python:
Drake:
MOSEK available/enabled:

compileall:
gear unit tests:
direction-cone unit tests:
gear layered Drake tests:

Level 3 batch:
  success:
  feasible:
  fallback:
  max curvature violation:
  gear diagnostics:

Level 4 batch:
  success:
  feasible:
  fallback:
  reverse ratio:
  switch count:
  max curvature violation:
  gear diagnostics:
```

这组信息足够判断问题是在环境、建图、求解器、direction-cone 参数，还是 gear layered 逻辑本身。
