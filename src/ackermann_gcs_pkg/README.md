# 阿克曼转向车辆GCS轨迹规划系统

基于微分平坦性理论和GCS（图凸集）算法的阿克曼转向车辆轨迹规划系统。

## 项目概述

本系统利用微分平坦性将非线性的车辆运动学模型转化为线性的平坦输出空间，在该空间内利用GCS进行轨迹参数化（贝塞尔曲线），并通过序列凸规划（SCP）处理非凸的曲率约束。

### 核心优势

- 将复杂的微分方程约束转化为代数约束
- 利用凸优化保证求解效率
- 精确满足起终点状态（位置、航向角、速度）
- 支持多种约束（速度、加速度、曲率、工作空间）

### 适用场景

- 自动驾驶泊车
- 狭窄空间机动
- 高速赛道规划

## 安装依赖

```bash
pip install numpy matplotlib scipy pydrake
```

## 快速开始

```python
import numpy as np
from pydrake.geometry.optimization import HPolyhedron

from ackermann_gcs_pkg import (
    VehicleParams,
    EndpointState,
    BezierConfig,
    SCPConfig,
)
from ackermann_gcs_pkg.ackermann_gcs_planner import AckermannGCSPlanner
from ackermann_gcs_pkg.ackermann_visualizer import visualize_trajectory

# 定义车辆参数
vehicle_params = VehicleParams(
    wheelbase=2.5,
    max_steering_angle=np.pi / 4,
    max_velocity=10.0,
    max_acceleration=5.0,
)

# 定义起终点状态
source = EndpointState(
    position=np.array([0.0, 5.0]),
    heading=0.0,
    velocity=2.0,
)

target = EndpointState(
    position=np.array([20.0, 5.0]),
    heading=0.0,
    velocity=2.0,
)

# 定义工作空间区域（简单矩形）
A = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
b = np.array([25.0, 0.0, 10.0, 0.0])
workspace_regions = [HPolyhedron(A, b)]

# 配置贝塞尔曲线和SCP参数
bezier_config = BezierConfig(order=5, continuity=1)
scp_config = SCPConfig(max_iterations=10, convergence_tolerance=1e-3)

# 创建规划器
planner = AckermannGCSPlanner(
    vehicle_params=vehicle_params,
    bezier_config=bezier_config,
    scp_config=scp_config,
)

# 规划轨迹
result = planner.plan_trajectory(
    source=source,
    target=target,
    workspace_regions=workspace_regions,
    cost_weights={"time": 1.0, "path_length": 0.1, "energy": 0.01},
    verbose=True,
)

# 可视化轨迹
if result.success:
    visualize_trajectory(
        trajectory=result.trajectory,
        vehicle_params=vehicle_params,
        source=source,
        target=target,
    )
```

## API 文档

### 数据结构

#### VehicleParams
车辆参数类。

```python
VehicleParams(
    wheelbase: float,           # 车辆轴距（米）
    max_steering_angle: float,  # 最大转向角（弧度）
    max_velocity: float,        # 最大速度（米/秒）
    max_acceleration: float,    # 最大加速度（米/秒²）
)
```

#### EndpointState
起终点状态类。

```python
EndpointState(
    position: np.ndarray,  # 位置坐标（米），形状为(2,)
    heading: float,        # 航向角（弧度），范围[-π, π]
    velocity: float = None,  # 速度（米/秒），可选
)
```

#### TrajectoryConstraints
轨迹约束类。

```python
TrajectoryConstraints(
    max_velocity: float,      # 最大速度（米/秒）
    max_acceleration: float,  # 最大加速度（米/秒²）
    max_curvature: float,     # 最大曲率（1/米）
    workspace_regions: List[HPolyhedron] = None,  # 工作空间区域
)
```

#### BezierConfig
贝塞尔曲线配置类。

```python
BezierConfig(
    order: int = 5,                                    # 贝塞尔曲线阶数
    continuity: int = 1,                               # 连续性阶数
    hdot_min: float = 1e-6,                            # 时间导数最小值
    full_dim_overlap: bool = False,                    # 是否使用全维重叠
    hyperellipsoid_num_samples_per_dim_factor: int = 32,  # 超椭球采样因子
)
```

#### SCPConfig
SCP配置类。

```python
SCPConfig(
    max_iterations: int = 10,              # 最大迭代次数
    convergence_tolerance: float = 1e-3,   # 收敛阈值
    initial_trust_region_radius: float = 1.0,  # 初始信任区域半径
    trust_region_shrink_factor: float = 0.5,   # 信任区域缩小因子
    trust_region_expand_factor: float = 2.0,   # 信任区域扩大因子
    min_trust_region_radius: float = 1e-6,     # 最小信任区域半径
)
```

### 核心类

#### AckermannGCSPlanner
阿克曼转向车辆GCS规划器。

```python
planner = AckermannGCSPlanner(
    vehicle_params: VehicleParams,
    bezier_config: BezierConfig = None,
    scp_config: SCPConfig = None,
)

result = planner.plan_trajectory(
    source: EndpointState,
    target: EndpointState,
    workspace_regions: List[HPolyhedron],
    constraints: TrajectoryConstraints = None,
    cost_weights: dict = None,
    verbose: bool = True,
) -> PlanningResult
```

#### PlanningResult
规划结果类。

```python
PlanningResult(
    success: bool,                    # 规划是否成功
    trajectory: BsplineTrajectory,    # 轨迹
    trajectory_report: TrajectoryReport,  # 轨迹评估报告
    solve_time: float,                # 求解时间（秒）
    num_iterations: int,              # SCP迭代次数
    convergence_reason: str,          # 收敛原因
    error_message: str,               # 错误消息
)
```

### 可视化

```python
from ackermann_gcs_pkg.ackermann_visualizer import visualize_trajectory

visualize_trajectory(
    trajectory: BsplineTrajectory,
    vehicle_params: VehicleParams,
    source: EndpointState = None,
    target: EndpointState = None,
    save_path: str = "/home/kai/WS/a_gcs_ws 2.0.1/ackermann_gcs_trajectory.png",
    dpi: int = 300,
)
```

## 测试说明

运行单元测试：

```bash
python -m pytest tests/ -v
```

## 许可证

MIT License
