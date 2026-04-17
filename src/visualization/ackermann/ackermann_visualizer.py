"""
阿克曼转向车辆轨迹可视化（已弃用）

本模块已弃用，请使用 ackermann_visualizer_enhanced.py 中的
AckermannGCSVisualizer 或 visualize_ackermann_gcs_enhanced()。

此文件保留仅为向后兼容，visualize_trajectory() 现在是增强版的适配器。
"""

import warnings
from typing import Optional

from pydrake.trajectories import BsplineTrajectory

from ackermann_gcs_pkg.ackermann_data_structures import VehicleParams, EndpointState


def visualize_trajectory(
    trajectory: BsplineTrajectory,
    vehicle_params: VehicleParams,
    source: Optional[EndpointState] = None,
    target: Optional[EndpointState] = None,
    save_path: str = "./ackermann_gcs_trajectory.png",
    dpi: int = 300,
    **kwargs,
) -> None:
    """
    可视化轨迹（已弃用，请使用 visualize_ackermann_gcs_enhanced）

    此函数现在是增强版可视化的适配器，所有功能由增强版提供。

    Args:
        trajectory: 轨迹
        vehicle_params: 车辆参数
        source: 起点状态，可选
        target: 终点状态，可选
        save_path: 图像保存路径
        dpi: 图像分辨率
        **kwargs: 传递给增强版的额外参数（如 workspace_regions）
    """
    warnings.warn(
        "visualize_trajectory() is deprecated, "
        "use visualize_ackermann_gcs_enhanced() instead",
        DeprecationWarning,
        stacklevel=2,
    )

    from .ackermann_visualizer_enhanced import visualize_ackermann_gcs_enhanced

    visualize_ackermann_gcs_enhanced(
        trajectory=trajectory,
        vehicle_params=vehicle_params,
        source=source,
        target=target,
        save_path=save_path,
        **kwargs,
    )
