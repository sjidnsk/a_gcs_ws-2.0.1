import numpy as np
import pytest

pytest.importorskip("pydrake")

from pydrake.symbolic import MakeVectorContinuousVariable

from ackermann_gcs_pkg.rotation_matrix_heading_constraint import (
    DirectionConstraint,
    HeadingConstraintConfig,
    HeadingConstraintFactory,
)


def test_reverse_direction_constraint_uses_opposite_heading():
    variables = MakeVectorContinuousVariable(4, "x")
    control_points = [
        (variables[0], variables[1]),
        (variables[2], variables[3]),
    ]
    config = HeadingConstraintConfig(enable_multi_point=False)

    constraints = HeadingConstraintFactory.create_heading_constraints(
        0.0,
        control_points,
        variables,
        config,
        direction_gear=-1,
    )

    assert len(constraints) == 2


def test_direction_constraint_numeric_check_accepts_reverse_motion():
    is_satisfied, _, _ = DirectionConstraint.check_direction_satisfaction(
        np.array([0.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.pi,
    )
    assert is_satisfied
