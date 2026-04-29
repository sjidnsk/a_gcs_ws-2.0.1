"""
M5: SolverAdapter — 求解器适配器

检测当前求解器是否支持旋转二阶锥约束，不支持时提供回退策略。
静态能力表 + 运行时旋转锥求解测试。
"""

import logging
import numpy as np

from .exceptions import SolverNotSupportedError

logger = logging.getLogger(__name__)


class SolverAdapter:
    """求解器适配器

    检测求解器旋转锥支持能力，不支持时回退到v1。

    已知支持旋转锥的求解器: MOSEK, SCS, Clarabel
    已知不支持: Gurobi（默认不信任其旋转锥支持）
    """

    # 已知支持旋转锥的求解器类名
    ROTATED_CONE_SOLVER_NAMES = {'MosekSolver', 'ScsSolver', 'ClarabelSolver'}

    def __init__(self, solver_type=None):
        """初始化求解器适配器

        Args:
            solver_type: 求解器类型（Drake Solver类实例或类名）
        """
        self.solver_type = solver_type
        self._supports_rotated_cone = None

    def check_rotated_cone_support(self, runtime_verify: bool = False) -> bool:
        """检测求解器是否支持旋转二阶锥

        Args:
            runtime_verify: 是否进行运行时测试（构造最小旋转锥问题实际求解验证）

        Returns:
            True如果支持，False如果不支持
        """
        if self._supports_rotated_cone is not None:
            return self._supports_rotated_cone

        # 获取求解器名称
        solver_name = self._get_solver_name()

        if solver_name is None:
            # 未指定求解器，假设支持（MOSEK默认）
            self._supports_rotated_cone = True
            logger.info("[CURVATURE_V2][INFO][SolverAdapter] "
                       "No solver specified, assuming rotated cone support (MOSEK default)")
        elif solver_name in self.ROTATED_CONE_SOLVER_NAMES:
            if runtime_verify:
                self._supports_rotated_cone = self._runtime_test()
            else:
                self._supports_rotated_cone = True
                logger.info(f"[CURVATURE_V2][INFO][SolverAdapter] "
                           f"Solver {solver_name} known to support rotated cone")
        elif solver_name == 'GurobiSolver':
            # 默认不信任Gurobi旋转锥
            self._supports_rotated_cone = False
            logger.warning(f"[CURVATURE_V2][WARNING][SolverAdapter] "
                          f"Solver Gurobi does not support "
                          f"RotatedLorentzConeConstraint. Falling back to v1.")
        else:
            # 未知求解器，保守假设不支持
            self._supports_rotated_cone = False
            logger.warning(f"[CURVATURE_V2][WARNING][SolverAdapter] "
                          f"Unknown solver {solver_name}, assuming no rotated cone support. "
                          f"Falling back to v1.")

        return self._supports_rotated_cone

    def _get_solver_name(self) -> str:
        """获取求解器类名

        Returns:
            求解器类名字符串，或None
        """
        if self.solver_type is None:
            return None

        # 如果是字符串，直接返回
        if isinstance(self.solver_type, str):
            return self.solver_type

        # 如果是类本身
        if hasattr(self.solver_type, '__name__'):
            return self.solver_type.__name__

        # 如果是类实例，获取类名
        if hasattr(self.solver_type, '__class__'):
            return self.solver_type.__class__.__name__

        return str(self.solver_type)

    def _runtime_test(self) -> bool:
        """运行时检测求解器是否真正支持旋转锥求解

        构造最小旋转锥测试问题并实际求解验证。

        Returns:
            True如果求解器正确支持旋转锥
        """
        try:
            from pydrake.solvers import (
                MathematicalProgram,
                RotatedLorentzConeConstraint,
                Binding,
                Constraint,
            )

            # 构造最小旋转锥测试问题
            prog = MathematicalProgram()
            x = prog.NewContinuousVariables(3, "x")

            # 约束: x[0]*x[1] >= x[2]^2, x[2] >= 1
            A = np.eye(3)
            b = np.zeros(3)
            prog.AddConstraint(
                Binding[Constraint](
                    RotatedLorentzConeConstraint(A, b), x
                )
            )
            prog.AddLinearConstraint(x[2] >= 1.0)
            prog.AddLinearCost(x[0] + x[1])

            # 尝试求解
            if self.solver_type is not None:
                result = self.solver_type().Solve(prog)
            else:
                # 使用默认求解器
                from pydrake.solvers import MosekSolver
                result = MosekSolver().Solve(prog)

            is_feasible = result.is_success()
            if is_feasible:
                # 验证解的正确性: x[0]*x[1] >= x[2]^2
                x_val = result.GetSolution(x)
                satisfied = x_val[0] * x_val[1] >= x_val[2]**2 - 1e-6
                if satisfied:
                    logger.info("[CURVATURE_V2][INFO][SolverAdapter] "
                               "Runtime test: solver correctly supports rotated cone")
                    return True
                else:
                    logger.warning("[CURVATURE_V2][WARNING][SolverAdapter] "
                                  "Runtime test: solver returned incorrect solution")
                    return False
            else:
                logger.warning("[CURVATURE_V2][WARNING][SolverAdapter] "
                              "Runtime test: solver failed to solve test problem")
                return False

        except ImportError as e:
            logger.warning(f"[CURVATURE_V2][WARNING][SolverAdapter] "
                          f"Runtime test failed (import error): {e}")
            return False
        except Exception as e:
            logger.warning(f"[CURVATURE_V2][WARNING][SolverAdapter] "
                          f"Runtime test failed: {e}")
            return False

    def get_fallback_strategy(self) -> str:
        """获取回退策略

        Returns:
            'v1' 如果需要回退，None 如果无需回退
        """
        if self.check_rotated_cone_support():
            return None
        else:
            return 'v1'
