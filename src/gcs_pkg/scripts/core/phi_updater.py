"""
增量Phi约束更新模块

在GCS Rounding阶段，每次固定一条路径时需要设置所有边的Phi约束。
传统方式是全量清除再重建所有边的Phi约束，开销为O(|E|)。

本模块实现增量Phi约束更新：跟踪上一条Rounding路径的边集合，
仅对与当前路径差异的边执行Phi约束修改，开销为O(|E_prev △ E_curr|)。

典型场景中，连续两条Rounding路径的边重叠率约70-90%，
因此增量更新操作量约为全量的10-30%。
"""

from typing import Optional, Set, List, Any


class IncrementalPhiUpdater:
    """增量Phi约束更新器

    跟踪上一条Rounding路径的边集合，仅对与当前路径
    差异的边执行Phi约束修改，避免全量清除+重建。

    用法:
        updater = IncrementalPhiUpdater()
        for path_edges in active_edges:
            updater.update_phi_constraints(gcs, path_edges)
            result = gcs.SolveShortestPath(source, target, options)
        updater.reset()  # 新一轮Rounding开始时
    """

    def __init__(self):
        """初始化更新器，无历史路径状态"""
        self._prev_path_edges: Optional[Set[Any]] = None
        self._is_first_path: bool = True
        self._phi_constraint_count: int = 0

    def update_phi_constraints(
        self,
        gcs: Any,
        path_edges: List[Any]
    ) -> None:
        """增量更新Phi约束

        Args:
            gcs: Drake GCS图对象 (GraphOfConvexSets)
            path_edges: 当前Rounding路径的边列表

        算法:
            1. 若为首条路径(_is_first_path=True):
               - 对所有边执行全量Phi约束设置
               - 记录当前路径边集合为_prev_path_edges
               - _is_first_path = False
            2. 否则:
               - 计算差异边集合:
                 deactivate = _prev_path_edges - current_path_edges
                 activate = current_path_edges - _prev_path_edges
               - 对deactivate中的边: ClearPhiConstraints() -> AddPhiConstraint(False)
               - 对activate中的边: ClearPhiConstraints() -> AddPhiConstraint(True)
               - 更新_prev_path_edges = current_path_edges
        """
        current_set = set(path_edges)

        if self._is_first_path:
            # 首条路径：全量设置
            for edge in gcs.Edges():
                if edge in current_set:
                    edge.AddPhiConstraint(True)
                else:
                    edge.AddPhiConstraint(False)
                self._phi_constraint_count += 1
            self._prev_path_edges = current_set
            self._is_first_path = False
            return

        # 增量更新：仅处理差异边
        deactivate = self._prev_path_edges - current_set   # 上一轮在路径中，本轮不在
        activate = current_set - self._prev_path_edges     # 本轮在路径中，上一轮不在

        for edge in deactivate:
            edge.ClearPhiConstraints()
            edge.AddPhiConstraint(False)
            self._phi_constraint_count += 1

        for edge in activate:
            edge.ClearPhiConstraints()
            edge.AddPhiConstraint(True)
            self._phi_constraint_count += 1

        self._prev_path_edges = current_set

    def reset(self) -> None:
        """重置更新器状态（新一轮Rounding开始时调用）"""
        self._prev_path_edges = None
        self._is_first_path = True
        self._phi_constraint_count = 0

    def get_constraint_count(self) -> int:
        """返回累计执行的Phi约束操作次数

        Returns:
            累计操作次数
        """
        return self._phi_constraint_count

    def get_prev_path_edges(self) -> Optional[Set[Any]]:
        """返回上一条路径的边集合（用于调试）

        Returns:
            上一条路径的边集合，若无历史则返回None
        """
        return self._prev_path_edges
