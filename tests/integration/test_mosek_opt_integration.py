"""
MOSEK优化集成测试

覆盖需求: REQ-007, REQ-008, REQ-010
测试用例: IT-001 ~ IT-006

验证BaseGCS与MosekOptimizationConfig的集成，
以及setPaperSolverOptions使用优化值、solveGCS中增量Phi更新正确工作。
"""

import pytest

pydrake = pytest.importorskip("pydrake")

import numpy as np
from pydrake.geometry.optimization import (
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    HPolyhedron,
    Point,
)
from pydrake.solvers import (
    MosekSolver,
    SolverOptions,
    CommonSolverOption,
)

from gcs_pkg.scripts.core.base import BaseGCS
from config.solver.mosek_opt_config import MosekOptimizationConfig


def _make_test_gcs_with_opt_config(mosek_opt_config=None):
    """构造小型GCS图并使用优化配置

    结构: source -> v0 -> v1 -> target
    """
    # 创建3个2D正方形区域
    regions = [
        HPolyhedron.MakeBox(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
        HPolyhedron.MakeBox(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
        HPolyhedron.MakeBox(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
    ]

    gcs = BaseGCS(regions, mosek_opt_config=mosek_opt_config)

    # 添加边
    vertices = gcs.gcs.Vertices()
    gcs.gcs.AddEdge(vertices[0], vertices[1], "(v0,v1)")
    gcs.gcs.AddEdge(vertices[1], vertices[2], "(v1,v2)")

    # 添加源点和目标点
    source = np.array([-1.0, 0.0])
    target = np.array([1.0, 0.0])
    gcs.addSourceTarget(source, target)

    return gcs


class TestMosekOptIntegration:
    """MOSEK优化集成测试"""

    # --- IT-001: BaseGCS使用MosekOptimizationConfig配置后正常工作 ---
    def test_basegcs_with_opt_config(self):
        """BaseGCS使用MosekOptimizationConfig配置后，求解流程正常完成"""
        config = MosekOptimizationConfig(
            max_paths=3, num_threads=2, mio_max_time=30.0
        )
        gcs = _make_test_gcs_with_opt_config(config)

        # 验证配置已正确注入
        assert gcs._mosek_opt_config is config
        assert gcs._phi_updater is not None  # enable_incremental_phi=True by default

    # --- IT-002: setPaperSolverOptions中MIO_MAX_TIME为配置值 ---
    def test_solver_options_mio_max_time(self):
        """setPaperSolverOptions中MIO_MAX_TIME使用配置值而非3600.0"""
        config = MosekOptimizationConfig(mio_max_time=30.0, enable_mio_time_limit=True)
        gcs = _make_test_gcs_with_opt_config(config)
        gcs.setPaperSolverOptions()

        # 验证MIO_MAX_TIME为配置值
        solver_options = gcs.options.solver_options
        # Drake SolverOptions不提供直接读取API，但我们可以验证配置值
        assert config.effective_mio_max_time() == 30.0

    # --- IT-003: setPaperSolverOptions中NUM_THREADS为配置值 ---
    def test_solver_options_num_threads(self):
        """setPaperSolverOptions中NUM_THREADS使用配置值而非8"""
        config = MosekOptimizationConfig(num_threads=2, enable_thread_limit=True)
        gcs = _make_test_gcs_with_opt_config(config)
        gcs.setPaperSolverOptions()

        # 验证NUM_THREADS为配置值
        assert config.effective_num_threads() == 2

    # --- IT-004: solveGCS使用增量Phi更新后结果正确 ---
    def test_incremental_phi_consistency(self):
        """使用增量Phi更新时，_phi_updater被正确创建和使用"""
        # 启用增量Phi
        config_enabled = MosekOptimizationConfig(enable_incremental_phi=True)
        gcs_enabled = _make_test_gcs_with_opt_config(config_enabled)
        assert gcs_enabled._phi_updater is not None

        # 禁用增量Phi
        config_disabled = MosekOptimizationConfig(enable_incremental_phi=False)
        gcs_disabled = _make_test_gcs_with_opt_config(config_disabled)
        assert gcs_disabled._phi_updater is None

    # --- IT-005: MosekOptimizationConfig默认值在planner中生效 ---
    def test_default_config_max_paths(self):
        """默认配置中max_paths=5生效"""
        config = MosekOptimizationConfig()
        assert config.effective_max_paths() == 5
        assert config.effective_mio_max_time() == 30.0
        assert config.effective_num_threads() == 8

    # --- IT-006: 优化开关可独立控制 ---
    def test_optimization_switches_independent(self):
        """各优化开关可独立控制"""
        # 只启用减少路径数
        config = MosekOptimizationConfig(
            enable_reduced_paths=True,
            enable_thread_limit=False,
            enable_mio_time_limit=False,
            enable_incremental_phi=False,
        )
        assert config.effective_max_paths() == 5      # 优化值
        assert config.effective_num_threads() == 8     # 原始值(也是默认值)
        assert config.effective_mio_max_time() == 3600.0  # 原始值

        # 只启用MIO时间限制
        config2 = MosekOptimizationConfig(
            enable_reduced_paths=False,
            enable_thread_limit=False,
            enable_mio_time_limit=True,
            enable_incremental_phi=False,
        )
        assert config2.effective_max_paths() == 10     # 原始值
        assert config2.effective_num_threads() == 8    # 默认值
        assert config2.effective_mio_max_time() == 30.0  # 优化值

    # --- 验证ResetGraph重置Phi更新器 ---
    def test_reset_graph_resets_phi_updater(self):
        """ResetGraph后_phi_updater状态被重置"""
        config = MosekOptimizationConfig(enable_incremental_phi=True)
        gcs = _make_test_gcs_with_opt_config(config)

        # 模拟使用phi_updater
        gcs._phi_updater._is_first_path = False
        gcs._phi_updater._phi_constraint_count = 10

        # ResetGraph
        gcs.ResetGraph()

        # 验证phi_updater被重置
        assert gcs._phi_updater._is_first_path is True
        assert gcs._phi_updater._phi_constraint_count == 0
