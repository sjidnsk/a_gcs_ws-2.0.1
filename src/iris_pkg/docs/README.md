# IrisNp 凸区域生成模块

基于 Drake IrisNp 的配置空间凸区域生成算法。

## 文件结构

```
iris_pkg/
├── __init__.py                 # 模块初始化，导出公共接口
├── iris_np_config.py           # 配置参数和常量定义
├── iris_np_region_data.py      # 数据结构（IrisNpRegion, IrisNpResult, RegionIndex）
├── iris_np_collision.py        # 碰撞检测器（SimpleCollisionCheckerForIrisNp, LRUCache）
├── iris_np_expansion.py        # 区域膨胀算法（IrisNpExpansion）
├── iris_np_parallel.py         # 并行处理辅助函数（init_worker, process_single_seed）
├── iris_np_utils.py            # 工具函数（碰撞检测、几何计算等）
└── iris_np_region.py           # 主模块（IrisNpRegionGenerator, 可视化函数）
```

## 模块说明

### 1. iris_np_config.py
- `IrisNpConfig`: 配置参数类
- 常量定义：DEFAULT_ITERATION_LIMIT, DEFAULT_TERMINATION_THRESHOLD 等

### 2. iris_np_region_data.py
- `IrisNpRegion`: 凸区域数据结构
- `IrisNpResult`: 处理结果数据结构
- `RegionIndex`: 区域空间索引（使用KDTree）

### 3. iris_np_collision.py
- `SimpleCollisionCheckerForIrisNp`: 优化的碰撞检测器
- `LRUCache`: LRU缓存实现

### 4. iris_np_expansion.py
- `IrisNpExpansion`: 区域膨胀算法基类
- 包含自适应膨胀、椭圆膨胀、方形膨胀等算法

### 5. iris_np_parallel.py
- `init_worker()`: 初始化工作进程
- `process_single_seed()`: 处理单个种子点（并行）

### 6. iris_np_utils.py
- `check_region_collision_optimized()`: 优化的区域碰撞检测
- `check_boundary_collision_fast()`: 快速边界碰撞检测
- `check_interior_collision_batch()`: 批量内部碰撞检测
- `check_interior_collision_sequential()`: 顺序内部碰撞检测
- `compute_polyhedron_vertices_optimized()`: 优化的多面体顶点计算
- `compute_polygon_area()`: 多边形面积计算
- `quick_boundary_collision_check()`: 快速边界碰撞检查
- `estimate_area_fast()`: 快速面积估算

### 7. iris_np_region.py
- `IrisNpRegionGenerator`: 凸区域生成器（主类）
- `visualize_iris_np_result()`: 可视化函数
- `check_drake_availability()`: Drake可用性检查

## 使用示例

```python
from iris_pkg import (
    IrisNpRegionGenerator,
    IrisNpConfig,
    visualize_iris_np_result
)

# 创建配置
config = IrisNpConfig(
    initial_region_size=0.1,
    max_region_size=10.0,
    enable_parallel_processing=True,
    num_parallel_workers=8
)

# 创建生成器
generator = IrisNpRegionGenerator(config)

# 从路径生成凸区域
result = generator.generate_from_path(
    path=path_points,
    obstacle_map=obstacle_map,
    resolution=0.05,
    origin=(0.0, 0.0)
)

# 可视化结果
visualize_iris_np_result(
    result=result,
    obstacle_map=obstacle_map,
    resolution=0.05,
    origin=(0.0, 0.0),
    path=path_points
)
```

## 依赖

- Python >= 3.8
- numpy
- scipy
- matplotlib
- pydrake (Drake)

## 安装 Drake

```bash
pip install drake
```

## 作者

Path Planning Team
