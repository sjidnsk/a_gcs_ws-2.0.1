# 自定义IrisZo算法模块

基于技术文档从零实现的IrisZo算法,不依赖Drake的IrisZo API。

## 概述

本模块实现了一个自定义的IrisZo(Iterative Regional Inflation by Semidefinite Programming - Zero Order)算法,用于在配置空间中生成概率无碰撞的凸多面体区域。

### 核心特点

- **零阶优化**: 仅使用碰撞检测器的布尔查询,无需梯度信息
- **算法自实现**: 基于技术文档从零实现,不依赖Drake的IrisZo API
- **概率保证**: 提供严格的数学概率保证
- **接口兼容**: 与iris_pkg模块保持接口一致
- **性能优化**: 支持LRU缓存、并行化

## 模块结构

```
src/iriszo/
├── __init__.py              # 公共接口
├── config/
│   ├── __init__.py
│   └── iriszo_config.py     # 配置参数
├── core/
│   ├── __init__.py
│   ├── iriszo_algorithm.py      # 自定义IrisZo算法核心
│   ├── iriszo_sampler.py        # Hit-and-Run采样器
│   ├── iriszo_bisection.py      # 二分搜索边界定位
│   ├── iriszo_hyperplane.py     # 分离超平面生成
│   ├── iriszo_collision.py      # 碰撞检测适配器
│   ├── iriszo_seed_extractor.py # 种子点提取
│   └── iriszo_region_data.py    # 数据结构
├── visualization/           # 可视化模块(待实现)
└── tests/
    └── test_basic.py        # 基本测试
```

## 核心组件

### 1. 配置模块 (IrisZoConfig)

```python
from src.iriszo import IrisZoConfig

# 使用默认配置
config = IrisZoConfig()

# 自定义配置
config = IrisZoConfig(
    epsilon=1e-4,           # 碰撞体积占比上限
    delta=1e-6,             # 置信水平
    iteration_limit=100,    # 最大迭代次数
    bisection_steps=10,     # 二分搜索步数
    verbose=True
)

# 预定义配置
from src.iriszo import get_high_safety_config, get_fast_processing_config
high_safety_config = get_high_safety_config()
fast_config = get_fast_processing_config()
```

### 2. 碰撞检测器 (CollisionCheckerAdapter)

```python
from src.iriszo import CollisionCheckerAdapter
import numpy as np

# 创建障碍物地图
obstacle_map = np.zeros((100, 100), dtype=np.uint8)
obstacle_map[40:60, 40:60] = 1  # 中心障碍物

# 创建碰撞检测器
checker = CollisionCheckerAdapter(
    obstacle_map=obstacle_map,
    resolution=0.05,
    origin=(0.0, 0.0),
    enable_cache=True
)

# 检查配置点
q = np.array([2.5, 2.5])
is_free = checker.check_config_collision_free(q)

# 检查边
q1 = np.array([0.0, 0.0])
q2 = np.array([5.0, 5.0])
is_edge_free = checker.check_edge_collision_free(q1, q2)
```

### 3. 自定义IrisZo算法 (CustomIrisZoAlgorithm)

```python
from src.iriszo import CustomIrisZoAlgorithm
from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid

# 创建算法实例
algorithm = CustomIrisZoAlgorithm(config)

# 创建搜索域
lb = np.array([0.0, 0.0])
ub = np.array([5.0, 5.0])
domain = HPolyhedron.MakeBox(lb, ub)

# 创建初始椭球体
center = np.array([1.0, 1.0])
starting_ellipsoid = Hyperellipsoid.MakeHyperSphere(0.1, center)

# 执行算法
region = algorithm.run(checker, starting_ellipsoid, domain)
```

### 4. 采样器 (HitAndRunSampler)

```python
from src.iriszo import HitAndRunSampler

sampler = HitAndRunSampler(config)
samples = sampler.sample(polyhedron, num_samples=100)
```

### 5. 二分搜索器 (BisectionSearcher)

```python
from src.iriszo import BisectionSearcher

bisectioner = BisectionSearcher(config)
boundary_points = bisectioner.search_boundary(
    collision_points, ellipsoid_center, checker
)
```

## 算法原理

自定义IrisZo算法基于技术文档中的Algorithm 2 (ZeroOrderSeparatingPlanes):

1. **初始化**: 多面体P初始化为搜索域
2. **外迭代循环**:
   - 在P内均匀采样(Hit-and-Run算法)
   - 碰撞检测找出碰撞点
   - 检查终止条件
   - 二分搜索优化碰撞点位置
   - 生成并添加分离超平面
   - 更新多面体P
   - 计算新的内接椭球体
3. **返回**: 最终多面体

### 概率保证

生成的多面体P满足:

```
Pr[λ(P \ C_free) / λ(P) > ε] ≤ δ
```

其中:
- ε: 碰撞体积占比上限
- δ: 置信水平
- λ: Lebesgue测度(体积)

## 测试

运行基本测试:

```bash
python3 src/iriszo/tests/test_basic.py
```

测试覆盖:
- ✓ 配置模块
- ✓ 碰撞检测器
- ✓ Drake可用性
- ✓ 算法组件

## 性能特点

- **LRU缓存**: O(1)时间复杂度的碰撞检测缓存
- **并行化支持**: 支持批量碰撞检测
- **零阶优化**: 无需梯度信息,适用性广
- **概率保证**: 严格的数学安全性保证

## 与iris_pkg的区别

| 特性 | iris_pkg (IrisNp) | iriszo (自定义IrisZo) |
|------|-------------------|----------------------|
| 优化方式 | 非线性规划 | 零阶优化 |
| 梯度需求 | 需要 | 不需要 |
| Drake依赖 | IrisNp API | 仅基础几何对象 |
| 性能 | 基准 | 预期快5-10× |
| 适用性 | 需要梯度信息 | 仅需碰撞检测 |

## 依赖

- Python ≥ 3.8
- NumPy ≥ 1.20
- Drake (pydrake) - 仅用于基础几何对象
- SciPy ≥ 1.7 (可选)

## 安装

```bash
# 安装Drake
pip install drake

# 安装其他依赖
pip install numpy scipy matplotlib
```

## 文档

- 需求规格: `.codeartsdoer/specs/iriszo_module_implementation/spec.md`
- 技术设计: `.codeartsdoer/specs/iriszo_module_implementation/design.md`
- 任务规划: `.codeartsdoer/specs/iriszo_module_implementation/tasks.md`
- 算法技术文档: `docs/IrisZo算法技术文档.md`

## 作者

Path Planning Team

## 版本

v2.0.0 - 自定义实现版本(不依赖Drake IrisZo API)
