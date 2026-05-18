import pytest

from ackermann_gcs_pkg.ackermann_data_structures import (
    CurvatureConstraintMode,
    TrajectoryConstraints,
)


def make_constraints(**overrides):
    kwargs = {
        "max_velocity": 8.0,
        "max_acceleration": 3.0,
        "max_curvature": 0.25,
        "curvature_constraint_mode": "direction_cone",
    }
    kwargs.update(overrides)
    return TrajectoryConstraints(**kwargs)


def test_direction_cone_mode_is_valid():
    constraints = make_constraints()

    assert CurvatureConstraintMode.DIRECTION_CONE.value == "direction_cone"
    assert constraints.curvature_constraint_mode == "direction_cone"
    assert constraints.direction_cone_alpha == pytest.approx(0.60)
    assert constraints.direction_cone_beta == pytest.approx(0.75)
    assert constraints.direction_cone_theta_min_deg == pytest.approx(25.0)
    assert constraints.direction_cone_skip_risk_flags == ()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("direction_cone_alpha", 0.0),
        ("direction_cone_beta", -0.1),
        ("direction_cone_gamma", 0.0),
        ("direction_cone_width_mu", -1.0),
        ("direction_cone_rho_warning_ratio", 0.0),
    ],
)
def test_direction_cone_rejects_non_positive_coefficients(field, value):
    with pytest.raises(ValueError, match=field):
        make_constraints(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"direction_cone_theta_min_deg": 0.0},
        {"direction_cone_theta_abs_max_deg": -1.0},
        {
            "direction_cone_theta_min_deg": 50.0,
            "direction_cone_theta_abs_max_deg": 45.0,
        },
        {"direction_cone_theta_margin_deg": -0.1},
    ],
)
def test_direction_cone_rejects_invalid_angle_configuration(overrides):
    with pytest.raises(ValueError):
        make_constraints(**overrides)


def test_direction_cone_accepts_skip_risk_flags():
    constraints = make_constraints(
        direction_cone_skip_risk_flags=["direction_mismatch"]
    )

    assert constraints.direction_cone_skip_risk_flags == ("direction_mismatch",)


def test_direction_cone_rejects_non_string_skip_risk_flags():
    with pytest.raises(ValueError, match="direction_cone_skip_risk_flags"):
        make_constraints(direction_cone_skip_risk_flags=("direction_mismatch", 1))
