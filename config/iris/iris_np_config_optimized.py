"""
优化后的 IrisNp 配置类（已合并到 IrisNpConfig）

此类已合并到 iris_np_config.py 中的 IrisNpConfig，
保留此文件仅为向后兼容。

所有参数分组、docstring 和预定义配置模板已迁移到 IrisNpConfig。

作者: Path Planning Team
"""

from .iris_np_config import (
    IrisNpConfig,
    get_high_safety_config,
    get_fast_processing_config,
    get_balanced_config,
)

# 向后兼容别名
IrisNpConfigOptimized = IrisNpConfig
