"""
覆盖半径计算模块

实现覆盖半径的计算、动态调整和降级判定逻辑。

作者: Path Planning Team
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.iriszo_config import IrisZoConfig


class RadiusCalculator:
    """
    覆盖半径计算器

    负责计算基础覆盖半径、根据障碍物距离动态调整有效半径，
    以及判断是否应降级为点包含判定。

    Attributes:
        config: IrisZo配置对象
        resolution: 地图分辨率（米/像素）

    Example:
        >>> from iriszo.config import IrisZoConfig
        >>> config = IrisZoConfig()
        >>> calculator = RadiusCalculator(config, resolution=0.05)
        >>> base_radius = calculator.calculate_base_radius()
        >>> effective_radius = calculator.adjust_effective_radius(base_radius, 2.0)
    """

    def __init__(self, config: "IrisZoConfig", resolution: float):
        """
        初始化半径计算器

        Args:
            config: IrisZo配置对象
            resolution: 地图分辨率（米/像素）

        Raises:
            ValueError: 如果resolution <= 0
        """
        if resolution <= 0:
            raise ValueError(f"resolution必须>0，当前值: {resolution}")

        self.config = config
        self.resolution = resolution

    def calculate_base_radius(self) -> float:
        """
        计算基础覆盖半径

        根据配置参数计算基础覆盖半径，取栅格数和实际距离的较大值。

        Returns:
            基础覆盖半径（米）

        Example:
            >>> calculator = RadiusCalculator(config, resolution=0.05)
            >>> base_radius = calculator.calculate_base_radius()
            >>> # 如果栅格数=5，分辨率=0.05，实际距离=1.0
            >>> # 则 radius_from_grids = 5 * 0.05 = 0.25米
            >>> # 返回 max(0.25, 1.0) = 1.0米
        """
        radius_from_grids = self.config.coverage_radius_grids * self.resolution
        radius_from_meters = self.config.coverage_radius_meters
        return max(radius_from_grids, radius_from_meters)

    def adjust_effective_radius(
        self,
        base_radius: float,
        obstacle_distance: float
    ) -> float:
        """
        根据障碍物距离调整有效覆盖半径

        当障碍物距离小于基础半径时，使用安全余量因子调整；
        否则使用完整的基础半径。

        Args:
            base_radius: 基础覆盖半径（米）
            obstacle_distance: 到最近障碍物的距离（米）

        Returns:
            有效覆盖半径（米）

        Example:
            >>> # 无障碍物情况
            >>> effective_radius = calculator.adjust_effective_radius(1.0, 2.0)
            >>> # 返回 1.0（基础半径）
            >>>
            >>> # 有障碍物情况
            >>> effective_radius = calculator.adjust_effective_radius(1.0, 0.5)
            >>> # 返回 0.5 * 0.8 = 0.4（使用安全余量因子）
        """
        if obstacle_distance < base_radius:
            # 存在障碍物，使用安全余量因子调整
            return obstacle_distance * self.config.safety_factor
        else:
            # 无障碍物，使用完整基础半径
            return base_radius

    def should_downgrade_to_point_check(
        self,
        effective_radius: float
    ) -> bool:
        """
        判断是否应降级为点包含判定

        当有效半径过小时，降级为简单的点包含判定。

        Args:
            effective_radius: 有效覆盖半径（米）

        Returns:
            True如果应降级，False否则

        Example:
            >>> # 有效半径足够大
            >>> should_downgrade = calculator.should_downgrade_to_point_check(0.5)
            >>> # 返回 False
            >>>
            >>> # 有效半径过小
            >>> should_downgrade = calculator.should_downgrade_to_point_check(0.05)
            >>> # 返回 True（如果min_effective_radius=0.1）
        """
        return effective_radius < self.config.min_effective_radius
