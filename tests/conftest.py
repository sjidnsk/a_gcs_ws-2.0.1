"""
Pytest配置文件
"""

import os
import sys

# 确保src目录在Python路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')

# 移除空字符串(如果存在)
if '' in sys.path:
    sys.path.remove('')

if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# 确保visualization包可用
try:
    import visualization
    print(f"✓ visualization包已加载: {visualization.__file__}")
except ImportError:
    print(f"✗ visualization包导入失败")
    print(f"Python路径: {sys.path[:5]}")
