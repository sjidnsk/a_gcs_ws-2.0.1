#!/usr/bin/env python3
"""
环境验证脚本
验证 iris-py3.12 环境是否正确安装所有依赖
"""

import sys
from typing import List, Tuple

def check_module(module_name: str, import_name: str = None) -> Tuple[bool, str]:
    """检查模块是否可以导入"""
    import_name = import_name or module_name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', '未知版本')
        return True, version
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"导入错误: {str(e)}"

def main():
    print("=" * 60)
    print("iris-py3.12 环境验证")
    print("=" * 60)
    print(f"Python 版本: {sys.version}")
    print("=" * 60)
    
    # 核心依赖列表
    dependencies = [
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("Matplotlib", "matplotlib"),
        ("PyTest", "pytest"),
        ("Drake", "drake"),
        ("CVXPY", "cvxpy"),
        ("Clarabel", "clarabel"),
        ("SCS", "scs"),
        ("OSQP", "osqp"),
        ("NetworkX", "networkx"),
        ("scikit-image", "skimage"),
        ("OpenCV", "cv2"),
        ("PyDot", "pydot"),
        ("PyYAML", "yaml"),
        ("Requests", "requests"),
        ("TQDM", "tqdm"),
        ("Numba", "numba"),
        ("Pillow", "PIL"),
        ("ImageIO", "imageio"),
    ]
    
    success_count = 0
    failed_modules = []
    
    print("\n依赖检查:")
    print("-" * 60)
    
    for name, import_name in dependencies:
        success, info = check_module(name, import_name)
        if success:
            print(f"✓ {name:20s} {info}")
            success_count += 1
        else:
            print(f"✗ {name:20s} 失败: {info}")
            failed_modules.append(name)
    
    print("-" * 60)
    print(f"\n总计: {success_count}/{len(dependencies)} 个依赖成功导入")
    
    if failed_modules:
        print(f"\n失败的模块: {', '.join(failed_modules)}")
        return 1
    else:
        print("\n✓ 所有依赖已正确安装!")
        
        # 检查 Drake 核心功能
        print("\n" + "=" * 60)
        print("Drake 核心功能验证:")
        print("-" * 60)
        
        try:
            from pydrake.geometry.optimization import HPolyhedron
            print("✓ HPolyhedron (凸多面体)")
        except Exception as e:
            print(f"✗ HPolyhedron: {e}")
            
        try:
            from pydrake.trajectories import BsplineTrajectory
            print("✓ BsplineTrajectory (B样条轨迹)")
        except Exception as e:
            print(f"✗ BsplineTrajectory: {e}")
            
        try:
            from pydrake.solvers import MathematicalProgram
            print("✓ MathematicalProgram (数学规划)")
        except Exception as e:
            print(f"✗ MathematicalProgram: {e}")
        
        print("=" * 60)
        print("\n🎉 环境验证完成！可以开始使用项目。")
        return 0

if __name__ == "__main__":
    sys.exit(main())
