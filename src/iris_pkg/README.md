# IrisNp 凸区域生成模块

基于 Drake IrisNp 的配置空间凸区域生成算法。

## 📁 文件结构

```
iris_pkg/
├── __init__.py                 # 模块初始化，导出所有公共接口
├── README.md                   # 本文档
│
├── config/                     # 配置模块
│   ├── __init__.py
│   ├── iris_np_config.py       # 基础配置类
│   ├── iris_np_config_optimized.py  # 优化配置类 + 预定义模板
│   └── iris_np_config_documentation.py  # 配置参数详细文档
│
├── core/                       # 核心功能模块
│   ├── __init__.py
│   ├── iris_np_region_data.py  # 数据结构 (IrisNpRegion, IrisNpResult, RegionIndex)
│   ├── iris_np_collision.py    # 碰撞检测器 (SimpleCollisionCheckerForIrisNp, LRUCache)
│   ├── iris_np_utils.py        # 工具函数 (碰撞检测、几何计算)
│   ├── iris_np_expansion.py    # 区域膨胀算法 (IrisNpExpansion)
│   ├── iris_np_parallel.py     # 并行处理辅助函数
│   └── iris_np_region.py       # 主生成器 (IrisNpRegionGenerator, 可视化)
│
└── docs/                       # 文档
    └── README.md               # 原始文档
```

## 🎯 模块说明

### 1. config/ - 配置模块

提供两种配置类：

- **`IrisNpConfig`**: 基础配置类，简单易用
- **`IrisNpConfigOptimized`**: 优化配置类，包含详细文档和预定义模板

预定义配置模板：
- `get_high_safety_config()`: 高安全要求配置
- `get_fast_processing_config()`: 快速处理配置
- `get_balanced_config()`: 平衡配置

### 2. core/ - 核心功能模块

包含 IrisNp 算法的核心实现：

| 文件 | 功能 | 主要类 |
|------|------|--------|
| `iris_np_region_data.py` | 数据结构 | `IrisNpRegion`, `IrisNpResult`, `RegionIndex` |
| `iris_np_collision.py` | 碰撞检测 | `SimpleCollisionCheckerForIrisNp`, `LRUCache` |
| `iris_np_utils.py` | 工具函数 | 碰撞检测、几何计算等辅助函数 |
| `iris_np_expansion.py` | 区域膨胀 | `IrisNpExpansion` 及各种膨胀算法 |
| `iris_np_parallel.py` | 并行处理 | `init_worker`, `process_single_seed` |
| `iris_np_region.py` | 主生成器 | `IrisNpRegionGenerator`, `visualize_iris_np_result` |

## 🚀 使用示例

### 基础使用（2D 区域生成）

```python
from iris_pkg import (
    IrisNpRegionGenerator,
    IrisNpConfigOptimized,
    get_high_safety_config,
    visualize_iris_np_result
)
import numpy as np

# 使用预定义配置
config = get_high_safety_config()

# 或自定义配置
config = IrisNpConfigOptimized(
    num_collision_infeasible_samples=100,
    configuration_space_margin=0.25,
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

print(f"生成 {result.num_regions} 个凸区域")
print(f"总面积: {result.total_area:.2f} 平方米")
print(f"覆盖率: {result.coverage_ratio*100:.1f}%")
```

## 📊 依赖关系

```
config/
└── (无依赖)

core/
├── config/ (依赖)
└── (内部依赖)
```

## 🔧 安装依赖

```bash
# 基础依赖
pip install numpy scipy matplotlib

# Drake (必需)
pip install drake
```

## 📖 配置参数说明

详细参数说明请参考：
- `config/iris_np_config_documentation.py`
- `config/iris_np_config_optimized.py` (包含详细注释)

## 🎓 核心概念

### 1. IrisNp 算法

IrisNp (IRIS for Configuration Space with Nonlinear Programming) 是 Drake 中用于在配置空间生成无碰撞凸区域的算法。

**核心优势**：
- 直接在配置空间工作
- 使用非线性优化处理碰撞
- 自动处理非凸障碍物
- 提供概率保证的无碰撞区域

### 2. 两批种子点扩张

- **第一批**: 正常扩张，覆盖主要路径
- **第二批**: 检查未覆盖路径点，优先沿切线方向膨胀

**优势**: 提高路径覆盖率，减少未覆盖区域

### 3. Theta 非凸性问题

运动学约束 `ẋ = v·cos(θ), ẏ = v·sin(θ)` 是非凸的，GCS 要求约束是凸的。

**解决方案**:
1. **3D 扩展**: 直接在 (x, y, θ) 空间处理
2. **4D 扩展**: 使用单位向量 (u, w) = (cos(θ), sin(θ))，配合 SOCP 松弛

**4D 扩展的优势**:
- 将非凸约束转化为凸约束
- GCS 优化问题保持凸性
- 更好的数值稳定性

## 📝 作者

Path Planning Team

## 📄 版本

v2.0.0 - 模块化重构版本
