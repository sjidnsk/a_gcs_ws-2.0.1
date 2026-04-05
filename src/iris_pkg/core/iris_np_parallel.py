"""
IrisNp 并行处理模块

包含并行处理种子点的辅助函数。

作者: Path Planning Team
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pydrake.geometry.optimization import HPolyhedron

from config.iris import IrisNpConfig
from .iris_np_region_data import IrisNpRegion
from .iris_np_collision import SimpleCollisionCheckerForIrisNp


# 全局变量用于进程池共享资源
_global_checker: Optional[SimpleCollisionCheckerForIrisNp] = None
_global_config: Optional[IrisNpConfig] = None
_global_domain: Optional[HPolyhedron] = None


def init_worker(
    obstacle_map: np.ndarray,
    resolution: float,
    origin: Tuple[float, float],
    config: IrisNpConfig,
    domain: HPolyhedron
):
    """
    初始化工作进程（共享资源）

    Args:
        obstacle_map: 障碍物地图
        resolution: 地图分辨率
        origin: 地图原点
        config: 配置参数
        domain: 域约束
    """
    global _global_checker, _global_config, _global_domain

    # 创建碰撞检测器（每个进程创建一次）
    _global_checker = SimpleCollisionCheckerForIrisNp(
        obstacle_map, resolution, origin,
        enable_cache=config.enable_collision_cache,
        cache_size=config.collision_cache_size
    )

    _global_config = config
    _global_domain = domain


def process_single_seed(task: Dict[str, Any]) -> Optional[IrisNpRegion]:
    """
    处理单个种子点（用于并行处理，使用共享资源）

    Args:
        task: 包含种子点和参数的字典

    Returns:
        生成的区域，如果失败则返回 None
    """
    from .iris_np_expansion import IrisNpExpansion

    global _global_checker, _global_config, _global_domain

    try:
        # 提取参数
        seed_point = task['seed_point']
        tangent_direction = task.get('tangent_direction', None)

        # 使用共享的checker和config（避免重复创建）
        if _global_checker is None or _global_config is None or _global_domain is None:
            # 如果全局变量未初始化，回退到创建新实例
            obstacle_map = task['obstacle_map']
            resolution = task['resolution']
            origin = task['origin']
            config = task['config']
            domain = task['domain']

            checker = SimpleCollisionCheckerForIrisNp(
                obstacle_map, resolution, origin,
                enable_cache=config.enable_collision_cache,
                cache_size=config.collision_cache_size
            )
            generator = IrisNpExpansion(config)
        else:
            checker = _global_checker
            config = _global_config
            domain = _global_domain
            generator = IrisNpExpansion(config)

        # 生成区域（传递切线方向）
        region = generator.simplified_iris_with_sampling(
            checker, seed_point, domain, task['obstacle_map'],
            task['resolution'], task['origin'], tangent_direction
        )

        return region

    except Exception as e:
        # 静默失败，返回 None
        return None
