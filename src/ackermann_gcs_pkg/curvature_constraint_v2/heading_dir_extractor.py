"""
M3: HeadingDirExtractor — 航向角方向提取器

从航向角约束配置或控制点方向，提取每条边的航向角方向向量d_θ，
使用slerp插值处理±π跨越。
"""

import logging
import numpy as np

from .constants import CURVATURE_V2_FLOAT_TOL, CURVATURE_V2_SINGULAR_TOL
from .exceptions import InvalidParameterError

logger = logging.getLogger(__name__)


class HeadingDirExtractor:
    """航向角方向提取器

    从航向角约束配置提取逐边航向角方向向量d_θ。

    策略优先级：
    1. 从航向角约束配置直接获取（最精确）
    2. 从一阶导数控制点方向估计（近似）
    3. 使用默认方向 [1, 0]（最保守）
    """

    def extract_per_edge(
        self,
        edges,
        edge_classifications: dict,
        heading_config,
        u_path_dot=None,
        u_vars=None,
    ) -> dict:
        """提取逐边航向角方向

        Args:
            edges: GCS图的边列表
            edge_classifications: 边分类结果，包含：
                - 'first_real_edge_ids': 第一条真实边的ID集合
                - 'target_edge_ids': 目标边的ID集合
                - 'boundary_edge_ids': 边界边ID集合
            heading_config: 航向角约束配置(HeadingConstraintConfig)
            u_path_dot: 一阶导数控制点（可选，用于策略2）
            u_vars: 决策变量列表（可选）

        Returns:
            {edge_id: np.array([cos_θ, sin_θ])} 逐边航向角方向映射
        """
        directions = {}

        # 获取起终航向角
        source_heading = getattr(heading_config, 'source_heading', None)
        target_heading = getattr(heading_config, 'target_heading', None)

        # 如果heading_config没有直接提供航向角，尝试从其他属性获取
        if source_heading is None:
            source_heading = getattr(heading_config, '_source_heading', 0.0)
        if target_heading is None:
            target_heading = getattr(heading_config, '_target_heading', 0.0)

        first_real_ids = edge_classifications.get('first_real_edge_ids', set())
        target_ids = edge_classifications.get('target_edge_ids', set())

        for edge in edges:
            edge_id = id(edge)

            # 策略1: 从航向角约束获取
            if edge_id in first_real_ids:
                # 第一条真实边使用起点航向角
                theta = source_heading
            elif edge_id in target_ids:
                # target边使用终点航向角
                theta = target_heading
            else:
                # 中间边: 使用slerp插值
                theta = self._interpolate_heading(
                    source_heading, target_heading, t=0.5
                )

            d_theta = self._normalize_direction(np.array([np.cos(theta), np.sin(theta)]))
            directions[edge_id] = d_theta

            logger.debug(f"[CURVATURE_V2][DEBUG][HeadingExtractor] "
                        f"Edge {edge_id}: d_θ = ({d_theta[0]:.4f}, {d_theta[1]:.4f}), "
                        f"θ = {np.degrees(np.arctan2(d_theta[1], d_theta[0])):.2f}°")

        logger.info(f"[CURVATURE_V2][INFO][HeadingExtractor] "
                    f"Extracted heading directions for {len(directions)} edges, "
                    f"source_heading={np.degrees(source_heading):.2f}°, "
                    f"target_heading={np.degrees(target_heading):.2f}°")

        return directions

    @staticmethod
    def _interpolate_heading(source_heading: float, target_heading: float,
                             t: float = 0.5) -> float:
        """航向角球面线性插值，正确处理±π跨越

        Args:
            source_heading: 起点航向角（弧度）
            target_heading: 终点航向角（弧度）
            t: 插值参数，0=source, 1=target

        Returns:
            插值后的航向角（弧度）
        """
        # 将航向角差归一化到 [-π, π]
        delta = (target_heading - source_heading + np.pi) % (2 * np.pi) - np.pi
        return source_heading + t * delta

    @staticmethod
    def _normalize_direction(d_theta: np.ndarray) -> np.ndarray:
        """归一化航向角方向向量，处理退化情况

        Args:
            d_theta: 方向向量

        Returns:
            归一化后的单位方向向量

        Raises:
            InvalidParameterError: 方向向量范数过小（退化）
        """
        norm = np.linalg.norm(d_theta)
        if norm < CURVATURE_V2_SINGULAR_TOL:
            raise InvalidParameterError(
                f"航向角方向向量范数过小 ({norm:.2e})，"
                f"可能航向角未定义（如全零速度）"
            )
        d_normalized = d_theta / norm

        # 验证归一化结果
        assert abs(np.linalg.norm(d_normalized) - 1.0) < CURVATURE_V2_FLOAT_TOL, \
            f"归一化验证失败: ||d||={np.linalg.norm(d_normalized)}"

        return d_normalized
