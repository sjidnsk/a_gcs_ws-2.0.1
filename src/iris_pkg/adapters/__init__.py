"""
配置空间扩展模块

包含 2D 区域到 3D/4D 配置空间的扩展适配器。

作者: Path Planning Team
"""

from .iris_region_3d_adapter import (
    IrisNpRegion3D,
    ThetaRangeConfig,
    IrisRegion3DAdapter
)

from .iris_region_4d_adapter import (
    IrisNpRegion4D,
    ThetaRangeConfigEnhanced,
    IrisRegion4DAdapter
)

__all__ = [
    # 3D 扩展
    'IrisNpRegion3D',
    'ThetaRangeConfig',
    'IrisRegion3DAdapter',

    # 4D 扩展
    'IrisNpRegion4D',
    'ThetaRangeConfigEnhanced',
    'IrisRegion4DAdapter'
]
