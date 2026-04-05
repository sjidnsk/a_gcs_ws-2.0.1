# AckermannGCS可视化模块

## 概述

本模块提供了增强的AckermannGCS轨迹可视化功能，采用模块化设计，支持丰富的可视化选项。

## 主要特性

### 1. 增强的2D轨迹视图
- ✅ 显示IRIS区域（凸区域）
- ✅ 显示障碍物地图
- ✅ 显示走廊约束边界
- ✅ 显示A*路径与GCS轨迹对比
- ✅ 起终点标注（带航向角箭头）
- ✅ 速度热力图模式

### 2. 3D配置空间轨迹视图
- ✅ (x, y, θ)三维空间轨迹可视化
- ✅ 可调节视角（仰角、方位角）
- ✅ 起终点3D标注
- ✅ A*路径3D投影
- ✅ 坐标平面投影

### 3. 多维度曲线图
- ✅ 速度曲线（带最大速度限制）
- ✅ 航向角曲线
- ✅ 曲率曲线（带最大曲率限制）
- ✅ 转向角曲线（带最大转向角限制）
- ✅ 加速度曲线（带最大加速度限制）
- ✅ θ随路径长度变化曲线

### 4. 综合可视化布局
- ✅ 2x3子图布局
- ✅ 可配置的图表大小和分辨率
- ✅ 专业的字体和样式设置
- ✅ 自动保存功能

## 模块结构

```
visualization/
├── __init__.py                      # 模块入口
├── config.py                        # 可视化配置
├── trajectory_sampler.py            # 轨迹采样器
├── region_renderer.py               # IRIS区域渲染器
├── path_comparator.py               # 路径对比器
├── plot_2d_trajectory.py            # 2D轨迹绘制
├── plot_3d_trajectory.py            # 3D轨迹绘制
├── plot_profiles.py                 # 曲线图绘制
└── ackermann_visualizer_enhanced.py # 主可视化器
```

## 快速开始

### 基本使用

```python
from ackermann_gcs_pkg.visualization import visualize_ackermann_gcs_enhanced

# 执行可视化
visualize_ackermann_gcs_enhanced(
    trajectory=trajectory,
    vehicle_params=vehicle_params,
    workspace_regions=workspace_regions,
    source=source,
    target=target,
    obstacle_map=obstacle_map,
    astar_path=astar_path,
    corridor_width=corridor_width,
    save_path="./output/visualization.png"
)
```

### 自定义配置

```python
from ackermann_gcs_pkg.visualization import (
    visualize_ackermann_gcs_enhanced,
    VisualizationConfig
)

# 创建配置
config = VisualizationConfig(
    num_samples=200,              # 采样点数
    show_iris_regions=True,       # 显示IRIS区域
    show_obstacles=True,          # 显示障碍物
    show_corridor=True,           # 显示走廊
    show_astar_path=True,         # 显示A*路径
    show_3d_trajectory=True,      # 显示3D轨迹
    show_theta_profile=True,      # 显示θ曲线
    figsize=(20, 14),            # 图表大小
    dpi=150                       # 分辨率
)

# 执行可视化
visualize_ackermann_gcs_enhanced(
    trajectory=trajectory,
    vehicle_params=vehicle_params,
    workspace_regions=workspace_regions,
    source=source,
    target=target,
    config=config,
    save_path="./output/visualization.png"
)
```

### 使用主可视化器类

```python
from ackermann_gcs_pkg.visualization import (
    AckermannGCSVisualizer,
    VisualizationConfig
)

# 创建可视化器
config = VisualizationConfig(num_samples=200)
visualizer = AckermannGCSVisualizer(config)

# 执行综合可视化
visualizer.visualize(
    trajectory=trajectory,
    vehicle_params=vehicle_params,
    workspace_regions=workspace_regions,
    source=source,
    target=target,
    save_path="./output/full_visualization.png"
)

# 仅绘制2D视图
visualizer.visualize_2d_only(
    trajectory=trajectory,
    vehicle_params=vehicle_params,
    workspace_regions=workspace_regions,
    source=source,
    target=target,
    save_path="./output/2d_visualization.png"
)

# 仅绘制3D视图
visualizer.visualize_3d_only(
    trajectory=trajectory,
    vehicle_params=vehicle_params,
    source=source,
    target=target,
    save_path="./output/3d_visualization.png"
)
```

### 独立的3D可视化

```python
from ackermann_gcs_pkg.visualization import visualize_3d_trajectory
from ackermann_gcs_pkg.visualization import TrajectorySampler

# 采样轨迹
sampler = TrajectorySampler(num_samples=200)
trajectory_data = sampler.sample(trajectory, vehicle_params)

# 绘制3D轨迹
visualize_3d_trajectory(
    trajectory_data=trajectory_data,
    source=source,
    target=target,
    elev=25.0,  # 仰角
    azim=45.0,  # 方位角
    save_path="./output/3d_trajectory.png"
)
```

## 配置参数说明

### VisualizationConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| num_samples | int | 200 | 轨迹采样点数 |
| show_iris_regions | bool | True | 显示IRIS区域 |
| show_obstacles | bool | True | 显示障碍物 |
| show_corridor | bool | True | 显示走廊约束 |
| show_astar_path | bool | True | 显示A*路径 |
| show_3d_trajectory | bool | True | 显示3D轨迹 |
| show_velocity | bool | True | 显示速度曲线 |
| show_heading | bool | True | 显示航向角曲线 |
| show_curvature | bool | True | 显示曲率曲线 |
| show_steering | bool | True | 显示转向角曲线 |
| show_acceleration | bool | True | 显示加速度曲线 |
| show_theta_profile | bool | True | 显示θ随路径变化曲线 |
| iris_alpha | float | 0.2 | IRIS区域透明度 |
| obstacle_alpha | float | 0.5 | 障碍物透明度 |
| elev | float | 25.0 | 3D视角仰角 |
| azim | float | 45.0 | 3D视角方位角 |
| trajectory_color | str | 'red' | 轨迹颜色 |
| astar_color | str | 'green' | A*路径颜色 |
| figsize | tuple | (20, 14) | 图表大小 |
| dpi | int | 150 | 分辨率 |

## 可视化建议

### 1. 分析轨迹质量
- 查看2D轨迹是否平滑
- 检查IRIS区域覆盖是否合理
- 观察走廊约束是否满足

### 2. 验证约束满足
- 速度曲线是否在限制范围内
- 曲率是否满足车辆运动学约束
- 转向角是否在物理限制内

### 3. 对比A*与GCS
- 比较路径长度差异
- 分析轨迹平滑度改进
- 观察配置空间中的差异

### 4. 3D视角分析
- 从不同角度观察轨迹
- 分析θ变化的连续性
- 检查是否存在不必要的旋转

## 性能优化

- 使用缓存机制避免重复采样
- 可调节采样点数平衡精度和性能
- 支持部分可视化（仅2D或仅3D）

## 扩展功能

### 速度热力图

```python
from ackermann_gcs_pkg.visualization import Plot2DTrajectory, VisualizationConfig

config = VisualizationConfig()
plotter = Plot2DTrajectory(config)

# 绘制速度热力图
fig, ax = plt.subplots()
plotter.plot_velocity_heatmap(ax, trajectory_data)
plt.savefig("velocity_heatmap.png")
```

### 带关键点的可视化

```python
# 定义关键点
keypoints = [
    (x1, y1, "Start"),
    (x2, y2, "Waypoint 1"),
    (x3, y3, "End")
]

# 绘制带关键点的2D视图
fig, ax = plt.subplots()
plotter.plot_with_keypoints(ax, trajectory_data, keypoints=keypoints)
plt.savefig("trajectory_with_keypoints.png")
```

## 注意事项

1. 确保已安装所有依赖：matplotlib, numpy, pydrake
2. IRIS区域渲染可能需要较长时间，可调节透明度优化显示效果
3. 3D可视化在轨迹点数较多时可能较慢，建议适当减少采样点数
4. 保存路径的目录会自动创建

## 更新日志

### v1.0.0 (2024-04-02)
- ✅ 初始版本发布
- ✅ 实现模块化架构
- ✅ 支持2D/3D可视化
- ✅ 集成到测试脚本
