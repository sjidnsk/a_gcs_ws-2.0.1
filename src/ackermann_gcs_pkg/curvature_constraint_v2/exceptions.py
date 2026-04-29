"""
曲率约束v2异常层次定义

异常层次：
CurvatureV2Error (基类)
  ├── InvalidParameterError          # 参数验证失败
  ├── SolverNotSupportedError        # 求解器不支持旋转锥
  ├── PrerequisiteViolationError     # P1~P6前提违反
  ├── ConstraintConstructionError    # 约束构建失败
  ├── VertexExtensionError           # 顶点维度扩展失败
  └── ConstraintValidationError      # 约束验证失败(非致命)
"""


class CurvatureV2Error(Exception):
    """曲率约束v2基类异常"""
    pass


class InvalidParameterError(CurvatureV2Error):
    """参数验证失败

    触发条件：
    - max_curvature <= 0
    - sigma_min <= 0
    - heading_directions格式错误或零向量
    """
    pass


class SolverNotSupportedError(CurvatureV2Error):
    """求解器不支持旋转二阶锥约束

    触发条件：
    - Gurobi等不支持RotatedLorentzConeConstraint的求解器
    """
    pass


class PrerequisiteViolationError(CurvatureV2Error):
    """前提条件P1~P6违反

    触发条件：
    - P1: 航向角方法非ROTATION_MATRIX
    - P2: 航向角点积约束覆盖不足
    - P3: 航向角点积约束被选择性禁用
    - P4: 求解器不支持旋转锥
    - P6: RotatedLorentzConeConstraint不可导入
    """

    def __init__(self, prerequisite_id: str, message: str):
        self.prerequisite_id = prerequisite_id
        super().__init__(f"前提条件{prerequisite_id}违反: {message}")


class ConstraintConstructionError(CurvatureV2Error):
    """约束构建失败

    触发条件：
    - DecomposeLinearExpressions失败
    - A矩阵维度不匹配
    - 旋转锥b向量不满足z[1]>0
    """
    pass


class VertexExtensionError(CurvatureV2Error):
    """顶点维度扩展失败

    触发条件：
    - HPolyhedron.CartesianProduct(ℝ²)失败
    """
    pass


class ConstraintValidationError(CurvatureV2Error):
    """约束验证失败（非致命）

    触发条件：
    - 约束数量不匹配
    - 维度不一致
    - 数值范围异常

    Note: 此异常为非致命异常，通常仅记录WARNING日志，不中断求解。
    """
    pass
