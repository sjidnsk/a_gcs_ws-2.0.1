#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iris-py3.12环境快速部署脚本

该脚本用于在其他设备上快速部署conda环境iris-py3.12。
支持Linux和macOS系统。

使用方法:
    python setup_iris_env.py [--env-name NAME] [--no-verify]

参数:
    --env-name NAME    指定环境名称(默认: iris-py3.12)
    --no-verify        跳过环境验证

作者: 自动生成
日期: 2026-04-23
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path


# Conda环境配置
CONDA_ENV_CONFIG = """name: iris-py3.12
channels:
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - _openmp_mutex=5.1=1_gnu
  - blas=1.0=mkl
  - bzip2=1.0.8=h5eee18b_6
  - ca-certificates=2026.3.19=h06a4308_0
  - expat=2.7.3=h7354ed3_4
  - icu=73.1=h6a678d5_0
  - iniconfig=2.3.0=py312h06a4308_0
  - intel-openmp=2025.0.0=h06a4308_1171
  - ld_impl_linux-64=2.44=h153f514_2
  - libexpat=2.7.3=h7354ed3_4
  - libffi=3.4.4=h6a678d5_1
  - libgcc=15.2.0=h69a1729_7
  - libgcc-ng=15.2.0=h166f726_7
  - libgomp=15.2.0=h4751f2c_7
  - libhwloc=2.12.1=default_hf1bbc79_1000
  - libstdcxx=15.2.0=hd5a066b_7
  - libstdcxx-ng=15.2.0=h166f726_7
  - libuuid=2.38.1=h0ce48e5_0
  - libxcb=1.17=h0ce48e5_0
  - mkl=2025.0.0=h06a4308_1171
  - mkl-service=2.2=py312h06a4308_1
  - mkl_fft=1.3.14=py312h5eee18b_0
  - mkl_random=1.2.3=py312h5eee18b_0
  - ncurses=6.4=h6a678d5_0
  - numpy=2.3.5=py312h5eee18b_0
  - numpy-base=2.3.5=py312h00548fb_0
  - openssl=3.5.6=h1b28b03_0
  - pip=25.3=pyhc872135_0
  - pluggy=1.5.0=py312h06a4308_0
  - pthread-stubs=0.3=h0ce48e5_1
  - pygments=2.20.0=py312h06a4308_0
  - pytest=9.0.2=py312h06a4308_0
  - python=3.12.12=hd17a9e1_1
  - readline=8.3=hc2a1206_0
  - setuptools=80.9.0=py312h06a4308_0
  - sqlite=3.51.1=he0a8d7e_0
  - tbb=2022.3.0=h698db13_0
  - tbb-devel=2022.3.0=h698db13_0
  - tk=8.6.15=h54e0aa7_0
  - tzdata=2025b=h04d1e81_0
  - wheel=0.45.1=py312h06a4308_0
  - xorg-libx11=1.8.12=h9b100fa_1
  - xorg-libxau=1.0.12=h9b100fa_0
  - xorg-libxdmcp=1.1.5=h9b100fa_0
  - xorg-xorgproto=2024.1=h5eee18b_1
  - xz=5.6.4=h5eee18b_1
  - zlib=1.3.1=hb25bd0a_0
  - pip:
      - anyio==4.12.1
      - certifi==2026.1.4
      - cffi==2.0.0
      - chardet==5.2.0
      - charset-normalizer==3.4.4
      - clarabel==0.11.1
      - contourpy==1.3.3
      - cvxpy==1.7.5
      - cycler==0.12.1
      - drake==1.51.1
      - fonttools==4.61.1
      - h11==0.16.0
      - httpcore==1.0.9
      - httpx==0.28.1
      - idna==3.11
      - imageio==2.37.2
      - jinja2==3.1.6
      - joblib==1.5.3
      - kiwisolver==1.4.9
      - lark==1.3.1
      - lark-oapi==1.5.3
      - lazy-loader==0.4
      - llvmlite==0.46.0
      - markupsafe==3.0.3
      - matplotlib==3.10.8
      - mosek==11.1.2
      - networkx==3.6.1
      - noise==1.2.2
      - nose==1.3.7
      - numba==0.64.0
      - opencv-python==4.13.0.90
      - osqp==1.0.5
      - packaging==25.0
      - pillow==12.1.0
      - psutil==7.2.2
      - pyclipper==1.4.0
      - pycparser==2.23
      - pycryptodome==3.23.0
      - pydot==4.0.1
      - pyjulia==0.0.6
      - pyparsing==3.3.1
      - python-dateutil==2.9.0.post0
      - pyyaml==6.0.3
      - requests==2.32.5
      - requests-toolbelt==1.0.0
      - rtree==1.4.1
      - scikit-image==0.26.0
      - scipy==1.16.3
      - scs==3.2.11
      - six==1.17.0
      - tifffile==2026.1.28
      - tqdm==4.67.1
      - typing-extensions==4.15.0
      - urllib3==2.6.3
      - websockets==16.0
"""

# 核心依赖包列表(用于验证)
CORE_PACKAGES = [
    'python',
    'numpy',
    'scipy',
    'matplotlib',
    'pytest',
    'drake',
    'cvxpy',
    'clarabel',
    'mosek',
    'scs',
]


def run_command(cmd, check=True, capture_output=False):
    """运行shell命令"""
    print(f"执行命令: {' '.join(cmd)}")
    try:
        if capture_output:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            return result.returncode, result.stdout, result.stderr
        else:
            result = subprocess.run(cmd, check=check)
            return result.returncode, None, None
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {e}")
        return e.returncode, None, None


def check_conda():
    """检查conda是否已安装"""
    print("\n=== 检查Conda安装 ===")
    returncode, stdout, stderr = run_command(['conda', '--version'], capture_output=True)
    
    if returncode == 0:
        print(f"✓ Conda已安装: {stdout.strip()}")
        return True
    else:
        print("✗ Conda未安装")
        print("请先安装Anaconda或Miniconda:")
        print("  - Anaconda: https://www.anaconda.com/products/distribution")
        print("  - Miniconda: https://docs.conda.io/en/latest/miniconda.html")
        return False


def check_env_exists(env_name):
    """检查conda环境是否已存在"""
    returncode, stdout, stderr = run_command(
        ['conda', 'env', 'list'], 
        capture_output=True
    )
    
    if returncode == 0:
        envs = stdout.strip().split('\n')
        for line in envs:
            if env_name in line and not line.startswith('#'):
                return True
    return False


def create_env_from_yaml(env_name, yaml_file):
    """从YAML文件创建conda环境"""
    print(f"\n=== 创建Conda环境: {env_name} ===")
    
    # 使用conda env create命令
    returncode, _, _ = run_command([
        'conda', 'env', 'create', 
        '-f', yaml_file,
        '-n', env_name
    ])
    
    return returncode == 0


def verify_environment(env_name):
    """验证环境是否正确安装"""
    print(f"\n=== 验证环境: {env_name} ===")
    
    # 检查Python版本
    print("\n检查Python版本...")
    returncode, stdout, stderr = run_command([
        'conda', 'run', '-n', env_name, 
        'python', '--version'
    ], capture_output=True)
    
    if returncode == 0:
        print(f"✓ Python版本: {stdout.strip()}")
    else:
        print("✗ Python版本检查失败")
        return False
    
    # 检查核心包
    print("\n检查核心依赖包...")
    all_ok = True
    for package in CORE_PACKAGES:
        returncode, stdout, stderr = run_command([
            'conda', 'run', '-n', env_name,
            'python', '-c', f'import {package}; print({package}.__version__)'
        ], capture_output=True)
        
        if returncode == 0:
            version = stdout.strip()
            print(f"  ✓ {package}: {version}")
        else:
            print(f"  ✗ {package}: 未安装或导入失败")
            all_ok = False
    
    return all_ok


def print_activation_instructions(env_name):
    """打印激活环境的说明"""
    print("\n" + "="*60)
    print("环境部署完成!")
    print("="*60)
    print(f"\n激活环境:")
    print(f"  conda activate {env_name}")
    print(f"\n验证安装:")
    print(f"  python -c 'import drake; print(drake.__version__)'")
    print(f"\n运行测试:")
    print(f"  pytest")
    print("\n核心依赖:")
    print("  - Python 3.12.12")
    print("  - Drake 1.51.1 (MIT机器人框架)")
    print("  - NumPy 2.3.5")
    print("  - SciPy 1.16.3")
    print("  - CVXPY 1.7.5")
    print("  - 求解器: Clarabel, Mosek, SCS")
    print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='iris-py3.12环境快速部署脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python setup_iris_env.py                    # 使用默认环境名
  python setup_iris_env.py --env-name my_env  # 指定环境名
  python setup_iris_env.py --no-verify        # 跳过验证
        """
    )
    parser.add_argument(
        '--env-name', 
        default='iris-py3.12',
        help='指定环境名称(默认: iris-py3.12)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='跳过环境验证'
    )
    
    args = parser.parse_args()
    env_name = args.env_name
    
    print("="*60)
    print("iris-py3.12环境部署脚本")
    print("="*60)
    print(f"目标环境名称: {env_name}")
    
    # 检查conda
    if not check_conda():
        sys.exit(1)
    
    # 检查环境是否已存在
    if check_env_exists(env_name):
        print(f"\n警告: 环境 '{env_name}' 已存在")
        response = input("是否删除并重新创建? (y/n): ")
        if response.lower() == 'y':
            print(f"删除环境: {env_name}")
            run_command(['conda', 'env', 'remove', '-n', env_name, '-y'])
        else:
            print("取消部署")
            sys.exit(0)
    
    # 创建临时YAML文件
    yaml_file = 'iris_env_temp.yaml'
    print(f"\n创建临时配置文件: {yaml_file}")
    with open(yaml_file, 'w') as f:
        f.write(CONDA_ENV_CONFIG)
    
    try:
        # 创建环境
        if not create_env_from_yaml(env_name, yaml_file):
            print("\n✗ 环境创建失败")
            sys.exit(1)
        
        # 验证环境
        if not args.no_verify:
            if not verify_environment(env_name):
                print("\n✗ 环境验证失败")
                sys.exit(1)
        
        # 打印使用说明
        print_activation_instructions(env_name)
        
    finally:
        # 清理临时文件
        if os.path.exists(yaml_file):
            os.remove(yaml_file)
            print(f"\n清理临时文件: {yaml_file}")


if __name__ == '__main__':
    main()
