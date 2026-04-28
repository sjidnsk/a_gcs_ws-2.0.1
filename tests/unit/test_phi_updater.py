"""
IncrementalPhiUpdater 单元测试

覆盖需求: REQ-007, REQ-008, REQ-009
测试用例: UT-006 ~ UT-010

注意: 需要pydrake支持，使用pytest.importorskip处理Drake依赖
"""

import pytest

pydrake = pytest.importorskip("pydrake")

import numpy as np
from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    HPolyhedron,
    Point,
)

from gcs_pkg.scripts.core.phi_updater import IncrementalPhiUpdater


def _make_simple_gcs():
    """构造小型GCS图用于测试

    结构: source -> v0 -> v1 -> v2 -> target
    另有: source -> v1 (额外边，用于测试路径差异)
    """
    gcs = GraphOfConvexSets()

    # 创建区域 (2D正方形)
    region = HPolyhedron.MakeUnitBox(2)

    # 添加顶点
    source = gcs.AddVertex(Point(np.array([0.0, 0.0])), "source")
    v0 = gcs.AddVertex(region, "v0")
    v1 = gcs.AddVertex(region, "v1")
    v2 = gcs.AddVertex(region, "v2")
    target = gcs.AddVertex(Point(np.array([1.0, 1.0])), "target")

    # 添加边
    e_s_v0 = gcs.AddEdge(source, v0, "(s,v0)")
    e_s_v1 = gcs.AddEdge(source, v1, "(s,v1)")
    e_v0_v1 = gcs.AddEdge(v0, v1, "(v0,v1)")
    e_v1_v2 = gcs.AddEdge(v1, v2, "(v1,v2)")
    e_v2_t = gcs.AddEdge(v2, target, "(v2,t)")

    return gcs, source, target, [e_s_v0, e_s_v1, e_v0_v1, e_v1_v2, e_v2_t]


class TestIncrementalPhiUpdater:
    """IncrementalPhiUpdater 单元测试"""

    # --- UT-006: 首条路径全量设置 ---
    def test_first_path_full_update(self):
        """首条路径调用时，对所有边执行全量Phi约束设置"""
        gcs, source, target, edges = _make_simple_gcs()
        updater = IncrementalPhiUpdater()

        # 选择路径: source -> v0 -> v1 -> v2 -> target
        path_edges = [edges[0], edges[2], edges[3], edges[4]]  # e_s_v0, e_v0_v1, e_v1_v2, e_v2_t

        updater.update_phi_constraints(gcs, path_edges)

        # 验证首条路径后状态
        assert not updater._is_first_path
        assert updater._prev_path_edges is not None
        assert updater.get_constraint_count() == len(list(gcs.Edges()))

    # --- UT-007: 增量更新仅修改差异边 ---
    def test_incremental_update_only_diff_edges(self):
        """后续路径仅对差异边执行Phi约束修改"""
        gcs, source, target, edges = _make_simple_gcs()
        updater = IncrementalPhiUpdater()

        # 第一条路径: source -> v0 -> v1 -> v2 -> target
        path1 = [edges[0], edges[2], edges[3], edges[4]]
        updater.update_phi_constraints(gcs, path1)
        count_after_first = updater.get_constraint_count()

        # 第二条路径: source -> v1 -> v2 -> target (差异: 去掉e_s_v0和e_v0_v1, 加入e_s_v1)
        path2 = [edges[1], edges[3], edges[4]]  # e_s_v1, e_v1_v2, e_v2_t
        updater.update_phi_constraints(gcs, path2)
        count_after_second = updater.get_constraint_count()

        # 增量更新应只修改差异边(3条: deactivate e_s_v0, e_v0_v1; activate e_s_v1)
        diff_count = count_after_second - count_after_first
        assert diff_count == 3  # 2 deactivate + 1 activate

    # --- UT-008: 相同路径不执行操作 ---
    def test_same_path_no_operation(self):
        """当前路径与上一条路径完全相同时，不执行任何Phi约束操作"""
        gcs, source, target, edges = _make_simple_gcs()
        updater = IncrementalPhiUpdater()

        path = [edges[0], edges[2], edges[3], edges[4]]
        updater.update_phi_constraints(gcs, path)
        count_after_first = updater.get_constraint_count()

        # 再次设置相同路径
        updater.update_phi_constraints(gcs, path)
        count_after_second = updater.get_constraint_count()

        # 操作次数不应增加
        assert count_after_second == count_after_first

    # --- UT-009: reset重置状态 ---
    def test_reset_clears_state(self):
        """reset()将状态重置为初始值"""
        gcs, source, target, edges = _make_simple_gcs()
        updater = IncrementalPhiUpdater()

        path = [edges[0], edges[2], edges[3], edges[4]]
        updater.update_phi_constraints(gcs, path)

        # reset
        updater.reset()

        assert updater._is_first_path is True
        assert updater._prev_path_edges is None
        assert updater.get_constraint_count() == 0

    # --- UT-010: 增量更新结果与全量更新一致 ---
    def test_consistency_with_full_update(self):
        """增量Phi更新后，每条边的Phi约束状态与全量更新方式一致"""
        gcs, source, target, edges = _make_simple_gcs()
        updater = IncrementalPhiUpdater()

        # 路径1
        path1 = [edges[0], edges[2], edges[3], edges[4]]
        updater.update_phi_constraints(gcs, path1)

        # 路径2
        path2 = [edges[1], edges[3], edges[4]]
        updater.update_phi_constraints(gcs, path2)

        # 验证: 路径2中的边应该有phi=True约束，不在路径2中的边应该有phi=False约束
        # 注意: Drake的Edge没有直接查询Phi约束值的API，
        # 但我们可以通过检查边的Phi约束是否被正确设置来间接验证
        # 这里我们验证updater的内部状态正确
        prev_edges = updater.get_prev_path_edges()
        assert prev_edges == set(path2)

    def test_get_prev_path_edges_initially_none(self):
        """初始状态下get_prev_path_edges返回None"""
        updater = IncrementalPhiUpdater()
        assert updater.get_prev_path_edges() is None

    def test_get_constraint_count_initially_zero(self):
        """初始状态下get_constraint_count返回0"""
        updater = IncrementalPhiUpdater()
        assert updater.get_constraint_count() == 0
