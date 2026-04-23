# Conda环境部署指南

本文档说明如何在其他设备上快速部署`iris-py3.12` conda环境。

## 环境信息

- **环境名称**: `iris-py3.12`
- **Python版本**: 3.12.12
- **Drake版本**: 1.51.1
- **主要依赖**: NumPy, SciPy, Matplotlib, CVXPY, Drake

## 部署方法

### 方法1: 使用YAML配置文件(推荐)

最简单的方法，直接使用conda命令创建环境：

```bash
# 创建环境
conda env create -f config/iris_env.yaml

# 激活环境
conda activate iris-py3.12

# 验证安装
python -c "import drake; print(drake.__version__)"
```

### 方法2: 使用Python部署脚本

Python脚本提供更详细的验证和错误处理：

```bash
# 使用默认环境名
python scripts/setup_iris_env.py

# 指定自定义环境名
python scripts/setup_iris_env.py --env-name my_custom_env

# 跳过验证(快速部署)
python scripts/setup_iris_env.py --no-verify
```

**参数说明**:
- `--env-name NAME`: 指定环境名称(默认: iris-py3.12)
- `--no-verify`: 跳过环境验证步骤

### 方法3: 使用Shell部署脚本

Shell脚本适合在终端中快速执行：

```bash
# 添加执行权限(首次使用)
chmod +x scripts/setup_iris_env.sh

# 使用默认环境名
./scripts/setup_iris_env.sh

# 指定自定义环境名
./scripts/setup_iris_env.sh my_custom_env
```

## 部署后验证

激活环境后，建议执行以下验证步骤：

### 1. 检查Python版本
```bash
python --version
# 应输出: Python 3.12.12
```

### 2. 检查核心依赖
```bash
# 检查Drake
python -c "import drake; print(f'Drake: {drake.__version__}')"

# 检查NumPy和SciPy
python -c "import numpy, scipy; print(f'NumPy: {numpy.__version__}, SciPy: {scipy.__version__}')"

# 检查CVXPY和求解器
python -c "import cvxpy, clarabel, scs; print('CVXPY和求解器已安装')"
```

### 3. 运行测试
```bash
# 运行项目测试
pytest

# 或运行特定测试
pytest tests/unit/ -v
```

## 核心依赖列表

| 包名 | 版本 | 用途 |
|------|------|------|
| Python | 3.12.12 | Python解释器 |
| Drake | 1.51.1 | MIT机器人框架 |
| NumPy | 2.3.5 | 数值计算 |
| SciPy | 1.16.3 | 科学计算和优化 |
| Matplotlib | 3.10.8 | 可视化 |
| CVXPY | 1.7.5 | 凸优化建模 |
| Clarabel | 0.11.1 | 凸优化求解器 |
| Mosek | 11.1.2 | 商业优化求解器 |
| SCS | 3.2.11 | 分裂锥求解器 |
| PyTest | 9.0.2 | 测试框架 |

## 常见问题

### Q1: 环境创建失败，提示包冲突
**解决方案**: 尝试使用`--no-build-id`参数：
```bash
conda env create -f config/iris_env.yaml --no-build-id
```

### Q2: Drake安装失败
**解决方案**: Drake需要特定的系统依赖，请参考官方文档：
- https://drake.mit.edu/installation.html

### Q3: Mosek许可证问题
**解决方案**: Mosek需要许可证，可以：
1. 申请学术许可证: https://www.mosek.com/products/academic-licenses/
2. 使用免费求解器Clarabel或SCS

### Q4: 在不同操作系统上部署
**解决方案**: 
- **Linux**: 直接使用上述方法
- **macOS**: 可能需要调整某些系统依赖包
- **Windows**: 建议使用WSL2或考虑使用Drake的Windows版本

## 环境管理

### 删除环境
```bash
conda env remove -n iris-py3.12
```

### 导出环境(更新配置)
```bash
conda env export -n iris-py3.12 > config/iris_env.yaml
```

### 克隆环境
```bash
conda create --name iris-py3.12-backup --clone iris-py3.12
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `config/iris_env.yaml` | Conda环境配置文件 |
| `scripts/setup_iris_env.py` | Python部署脚本 |
| `scripts/setup_iris_env.sh` | Shell部署脚本 |
| `docs/environment_deployment.md` | 本文档 |

## 技术支持

如遇到问题，请检查：
1. Conda版本是否最新
2. 系统是否满足Drake的要求
3. 网络连接是否正常(需要下载包)

更多信息请参考项目主文档: `code.md`
