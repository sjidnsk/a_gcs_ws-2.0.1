"""
M2: AuxiliaryVarManager — 辅助变量管理器

管理逐边辅助变量σ_e和τ_e的创建、维度扩展、索引映射。

关键设计决策：
1. σ_e和τ_e是顶点变量（非边变量），因为Drake GCS约束通过edge.xu()绑定到上游顶点变量
2. 所有边共享同一对(σ_e, τ_e)索引位置，但不同边的u顶点不同，实际值由各自顶点优化决定
3. 顶点维度扩展必须在BezierGCS.__init__阶段完成（Drake GCS不提供vertex.set().Update()）
"""

import logging
import numpy as np

from pydrake.geometry.optimization import HPolyhedron
from pydrake.symbolic import MakeVectorContinuousVariable

from .exceptions import VertexExtensionError
from .constants import LOG_PREFIX

logger = logging.getLogger(__name__)


class AuxiliaryVarManager:
    """辅助变量管理器

    管理逐边辅助变量σ_e和τ_e的生命周期，包括维度扩展、索引映射、变量访问。

    Attributes:
        original_num_vars: 原始变量维度（扩展前）
        extended_num_vars: 扩展后变量维度 (= original + 2)
        edge_aux_var_map: 逐边辅助变量索引映射 {edge.id(): (sigma_idx, tau_idx)}
        sigma_offset: σ_e在u_vars中的全局偏移
        tau_offset: τ_e在u_vars中的全局偏移
    """

    def __init__(self, original_u_vars, gcs_edges, source_vertex):
        """初始化辅助变量管理器

        Args:
            original_u_vars: 决策变量列表（已包含σ_e和τ_e的扩展后u_vars）
            gcs_edges: GCS图的边列表
            source_vertex: source顶点（source边的约束跳过）
        """
        self.edge_aux_var_map = {}
        self._extended = False

        # v2模式下，u_vars已在BezierGCS.__init__中扩展了σ_e和τ_e
        # σ_e和τ_e位于u_vars末尾：sigma在-2位置，tau在-1位置
        self.extended_num_vars = len(original_u_vars)
        self.sigma_offset = self.extended_num_vars - 2
        self.tau_offset = self.extended_num_vars - 1
        self.original_num_vars = self.extended_num_vars - 2

        # 为每条非source边建立索引映射
        for edge in gcs_edges:
            if edge.u() != source_vertex:
                self.edge_aux_var_map[id(edge)] = (
                    self.sigma_offset,  # σ_e 在 u_vars 中的索引
                    self.tau_offset     # τ_e 在 u_vars 中的索引
                )

        logger.info(f"[{LOG_PREFIX}] AuxiliaryVarManager initialized: "
                    f"original_vars={self.original_num_vars}, "
                    f"extended_vars={self.extended_num_vars}, "
                    f"sigma_offset={self.sigma_offset}, tau_offset={self.tau_offset}, "
                    f"num_edges={len(self.edge_aux_var_map)}")

    def extend_vertex_set(self, vertex_set):
        """扩展顶点凸集: region × T × ℝ²

        ℝ² 用无约束的 HPolyhedron 表示（0行约束 = 无约束）

        Args:
            vertex_set: 当前顶点凸集

        Returns:
            扩展后的顶点凸集

        Raises:
            VertexExtensionError: 扩展失败或重复扩展
        """
        if self._extended:
            raise VertexExtensionError(
                "顶点已扩展，不可重复调用extend_vertex_set"
            )

        try:
            # ℝ² 用无约束的 HPolyhedron 表示
            # A矩阵0行×2列，b向量0维 → 无约束
            free_R2 = HPolyhedron(
                A=np.zeros((0, 2)),
                b=np.zeros(0)
            )
            extended_set = vertex_set.CartesianProduct(free_R2)
            self._extended = True

            logger.debug(f"[{LOG_PREFIX}] Vertex set extended: "
                        f"original_dim={vertex_set.ambient_dimension()}, "
                        f"extended_dim={extended_set.ambient_dimension()}")

            return extended_set
        except Exception as e:
            raise VertexExtensionError(
                f"顶点维度扩展失败: {e}"
            ) from e

    def get_sigma_index(self, edge):
        """获取边的σ_e变量索引

        Args:
            edge: GCS边对象

        Returns:
            int: σ_e在u_vars中的索引

        Raises:
            KeyError: 边不在映射中
        """
        edge_id = id(edge)
        if edge_id not in self.edge_aux_var_map:
            raise KeyError(f"Edge {edge.name()} (id={edge_id}) not in aux var map")
        return self.edge_aux_var_map[edge_id][0]

    def get_tau_index(self, edge):
        """获取边的τ_e变量索引

        Args:
            edge: GCS边对象

        Returns:
            int: τ_e在u_vars中的索引

        Raises:
            KeyError: 边不在映射中
        """
        edge_id = id(edge)
        if edge_id not in self.edge_aux_var_map:
            raise KeyError(f"Edge {edge.name()} (id={edge_id}) not in aux var map")
        return self.edge_aux_var_map[edge_id][1]

    def get_extended_u_vars(self, original_u_vars):
        """构建扩展后的变量列表

        在原始u_vars后追加σ_e和τ_e变量

        Args:
            original_u_vars: 原始决策变量列表

        Returns:
            np.ndarray: 扩展后的变量列表
        """
        sigma_var = MakeVectorContinuousVariable(1, "sigma")[0]
        tau_var = MakeVectorContinuousVariable(1, "tau")[0]
        return np.concatenate([original_u_vars, [sigma_var, tau_var]])

    @property
    def is_extended(self):
        """顶点是否已扩展"""
        return self._extended

    @property
    def aux_dim(self):
        """辅助变量维度（始终为2: σ_e + τ_e）"""
        return 2
