#!/bin/bash
# iris-py3.12环境快速部署脚本 (Shell版本)
# 
# 使用方法:
#   chmod +x setup_iris_env.sh
#   ./setup_iris_env.sh [环境名称]
#
# 示例:
#   ./setup_iris_env.sh              # 使用默认环境名 iris-py3.12
#   ./setup_iris_env.sh my_env       # 使用自定义环境名

set -e  # 遇到错误立即退出

# 配置
ENV_NAME="${1:-iris-py3.12}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_FILE="${SCRIPT_DIR}/../config/iris_env.yaml"

echo "============================================================"
echo "iris-py3.12环境部署脚本 (Shell版本)"
echo "============================================================"
echo "目标环境名称: ${ENV_NAME}"
echo ""

# 检查conda是否安装
echo "=== 检查Conda安装 ==="
if command -v conda &> /dev/null; then
    CONDA_VERSION=$(conda --version)
    echo "✓ Conda已安装: ${CONDA_VERSION}"
else
    echo "✗ Conda未安装"
    echo "请先安装Anaconda或Miniconda:"
    echo "  - Anaconda: https://www.anaconda.com/products/distribution"
    echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 检查YAML文件是否存在
if [ ! -f "${YAML_FILE}" ]; then
    echo "✗ 配置文件不存在: ${YAML_FILE}"
    echo "请确保 iris_env.yaml 文件存在于 config/ 目录"
    exit 1
fi

# 检查环境是否已存在
echo ""
echo "=== 检查环境是否存在 ==="
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "警告: 环境 '${ENV_NAME}' 已存在"
    read -p "是否删除并重新创建? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除环境: ${ENV_NAME}"
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "取消部署"
        exit 0
    fi
fi

# 创建环境
echo ""
echo "=== 创建Conda环境: ${ENV_NAME} ==="
echo "这可能需要几分钟时间，请耐心等待..."
conda env create -f "${YAML_FILE}" -n "${ENV_NAME}"

# 验证环境
echo ""
echo "=== 验证环境 ==="
echo "检查Python版本..."
PYTHON_VERSION=$(conda run -n "${ENV_NAME}" python --version)
echo "✓ ${PYTHON_VERSION}"

echo ""
echo "检查核心依赖包..."
CORE_PACKAGES=("numpy" "scipy" "matplotlib" "pytest" "drake" "cvxpy" "clarabel")
ALL_OK=true

for package in "${CORE_PACKAGES[@]}"; do
    if conda run -n "${ENV_NAME}" python -c "import ${package}" 2>/dev/null; then
        VERSION=$(conda run -n "${ENV_NAME}" python -c "import ${package}; print(${package}.__version__)" 2>/dev/null)
        echo "  ✓ ${package}: ${VERSION}"
    else
        echo "  ✗ ${package}: 未安装或导入失败"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" = true ]; then
    echo ""
    echo "============================================================"
    echo "环境部署完成!"
    echo "============================================================"
    echo ""
    echo "激活环境:"
    echo "  conda activate ${ENV_NAME}"
    echo ""
    echo "验证安装:"
    echo "  python -c 'import drake; print(drake.__version__)'"
    echo ""
    echo "运行测试:"
    echo "  pytest"
    echo ""
    echo "核心依赖:"
    echo "  - Python 3.12.12"
    echo "  - Drake 1.51.1 (MIT机器人框架)"
    echo "  - NumPy 2.3.5"
    echo "  - SciPy 1.16.3"
    echo "  - CVXPY 1.7.5"
    echo "  - 求解器: Clarabel, Mosek, SCS"
    echo "============================================================"
else
    echo ""
    echo "✗ 环境验证失败，请检查安装日志"
    exit 1
fi
