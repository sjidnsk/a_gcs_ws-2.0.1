"""
阿克曼转向车辆轨迹可视化

本模块实现了轨迹可视化功能，生成包含轨迹、速度、航向角、曲率、转向角、加速度的6子图。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from pydrake.trajectories import BsplineTrajectory

from ackermann_gcs_pkg.ackermann_data_structures import VehicleParams, EndpointState
from ackermann_gcs_pkg.flat_output_mapper import compute_flat_output_mapping


def visualize_trajectory(
    trajectory: BsplineTrajectory,
    vehicle_params: VehicleParams,
    source: Optional[EndpointState] = None,
    target: Optional[EndpointState] = None,
    save_path: str = "/home/kai/WS/a_gcs_ws 2.0.1/ackermann_gcs_trajectory.png",
    dpi: int = 300,
) -> None:
    """
    可视化轨迹

    创建2x3子图：
    - 子图1：2D轨迹（位置、起终点、航向角）
    - 子图2：速度曲线
    - 子图3：航向角曲线
    - 子图4：曲率曲线
    - 子图5：转向角曲线
    - 子图6：加速度曲线

    Args:
        trajectory: 轨迹
        vehicle_params: 车辆参数
        source: 起点状态，可选
        target: 终点状态，可选
        save_path: 图像保存路径
        dpi: 图像分辨率
    """
    # 计算平坦输出映射
    mapping = compute_flat_output_mapping(trajectory, vehicle_params, num_samples=100)

    position = mapping["position"]
    velocity = mapping["velocity"]
    heading = mapping["heading"]
    curvature = mapping["curvature"]
    steering_angle = mapping["steering_angle"]
    acceleration = mapping["acceleration"]

    # 采样时间点
    t_samples = np.linspace(trajectory.start_time(), trajectory.end_time(), 100)

    # 创建2x3子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 子图1：2D轨迹
    ax1 = axes[0, 0]
    ax1.plot(position[0, :], position[1, :], "b-", linewidth=2, label="Trajectory")
    if source is not None:
        ax1.plot(source.position[0], source.position[1], "go", markersize=10, label="Source")
        # 绘制起点航向角箭头
        arrow_length = 2.0
        ax1.arrow(
            source.position[0],
            source.position[1],
            arrow_length * np.cos(source.heading),
            arrow_length * np.sin(source.heading),
            head_width=0.5,
            head_length=0.5,
            fc="g",
            ec="g",
        )
    if target is not None:
        ax1.plot(target.position[0], target.position[1], "ro", markersize=10, label="Target")
        # 绘制终点航向角箭头
        arrow_length = 2.0
        ax1.arrow(
            target.position[0],
            target.position[1],
            arrow_length * np.cos(target.heading),
            arrow_length * np.sin(target.heading),
            head_width=0.5,
            head_length=0.5,
            fc="r",
            ec="r",
        )
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("2D Trajectory")
    ax1.legend()
    ax1.grid(True)
    ax1.axis("equal")

    # 子图2：速度曲线
    ax2 = axes[0, 1]
    ax2.plot(t_samples, velocity, "b-", linewidth=2, label="Velocity")
    ax2.axhline(y=vehicle_params.max_velocity, color="r", linestyle="--", label="Max Velocity")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_title("Velocity")
    ax2.legend()
    ax2.grid(True)

    # 子图3：航向角曲线
    ax3 = axes[0, 2]
    ax3.plot(t_samples, heading, "b-", linewidth=2, label="Heading")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Heading (rad)")
    ax3.set_title("Heading")
    ax3.legend()
    ax3.grid(True)

    # 子图4：曲率曲线
    ax4 = axes[1, 0]
    ax4.plot(t_samples, curvature, "b-", linewidth=2, label="Curvature")
    ax4.axhline(
        y=vehicle_params.max_curvature,
        color="r",
        linestyle="--",
        label="Max Curvature",
    )
    ax4.axhline(
        y=-vehicle_params.max_curvature,
        color="r",
        linestyle="--",
        label="Min Curvature",
    )
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Curvature (1/m)")
    ax4.set_title("Curvature")
    ax4.legend()
    ax4.grid(True)

    # 子图5：转向角曲线
    ax5 = axes[1, 1]
    ax5.plot(t_samples, steering_angle, "b-", linewidth=2, label="Steering Angle")
    ax5.axhline(
        y=vehicle_params.max_steering_angle,
        color="r",
        linestyle="--",
        label="Max Steering Angle",
    )
    ax5.axhline(
        y=-vehicle_params.max_steering_angle,
        color="r",
        linestyle="--",
        label="Min Steering Angle",
    )
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Steering Angle (rad)")
    ax5.set_title("Steering Angle")
    ax5.legend()
    ax5.grid(True)

    # 子图6：加速度曲线
    ax6 = axes[1, 2]
    ax6.plot(t_samples, acceleration, "b-", linewidth=2, label="Acceleration")
    ax6.axhline(
        y=vehicle_params.max_acceleration,
        color="r",
        linestyle="--",
        label="Max Acceleration",
    )
    ax6.axhline(
        y=-vehicle_params.max_acceleration,
        color="r",
        linestyle="--",
        label="Min Acceleration",
    )
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Acceleration (m/s²)")
    ax6.set_title("Acceleration")
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    print(f"[Visualizer] Trajectory visualization saved to {save_path}")
    plt.close()
