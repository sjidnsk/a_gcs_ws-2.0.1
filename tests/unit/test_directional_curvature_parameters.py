import numpy as np
import pytest

from ackermann_gcs_pkg.ackermann_data_structures import TrajectoryConstraints
from ackermann_gcs_pkg.directional_curvature_parameters import (
    DirectionalCurvatureParameterBuilder,
    DirectionalCurvatureSegmentParams,
    compute_support_width,
    directional_curvature_linear_constraints,
    preprocess_reference_path,
)


class VertexRegion:
    def __init__(self, vertices):
        self.vertices = np.asarray(vertices, dtype=float)

    def ambient_dimension(self):
        return 2


class FakeVertex:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name


class FakeEdge:
    _next_id = 1

    def __init__(self, u_name, v_name, name):
        self._u = FakeVertex(u_name)
        self._v = FakeVertex(v_name)
        self._name = name
        self._id = FakeEdge._next_id
        FakeEdge._next_id += 1

    def u(self):
        return self._u

    def v(self):
        return self._v

    def name(self):
        return self._name

    def id(self):
        return self._id


def make_constraints(**overrides):
    kwargs = {
        "max_velocity": 8.0,
        "max_acceleration": 3.0,
        "max_curvature": 0.25,
        "curvature_constraint_mode": "direction_cone",
    }
    kwargs.update(overrides)
    return TrajectoryConstraints(**kwargs)


def rectangle(cx, cy, width, height):
    hx = width / 2.0
    hy = height / 2.0
    return VertexRegion(
        [
            [cx - hx, cy - hy],
            [cx + hx, cy - hy],
            [cx + hx, cy + hy],
            [cx - hx, cy + hy],
        ]
    )


def test_segment_params_normalize_tangent_and_compute_eta():
    params = DirectionalCurvatureSegmentParams(
        edge_id="edge",
        t=np.array([2.0, 0.0]),
        n=np.array([0.0, 3.0]),
        rho=1.25,
        theta_max=np.deg2rad(35.0),
        kappa_max=0.25,
    )

    assert np.allclose(params.t, [1.0, 0.0])
    assert np.allclose(params.n, [0.0, 1.0])
    assert np.dot(params.t, params.n) == pytest.approx(0.0)
    assert params.eta == pytest.approx(np.tan(np.deg2rad(35.0)))


def test_preprocess_reference_path_resamples_and_smooths_tangents():
    path = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ]

    processed = preprocess_reference_path(path, min_turning_radius=5.0)

    assert processed.points.shape[1] == 2
    assert processed.points[0, 0] == pytest.approx(0.0)
    assert processed.points[-1, 0] == pytest.approx(2.0)
    assert np.allclose(processed.tangents, np.array([[1.0, 0.0]] * len(processed.tangents)))
    assert np.all(np.diff(processed.cumulative_s) > 0)


def test_support_width_uses_vertices_and_cache():
    region = rectangle(0.0, 0.0, width=4.0, height=2.0)
    cache = {}

    width = compute_support_width(region, np.array([1.0, 0.0]), cache=cache)
    width_again = compute_support_width(region, np.array([1.0, 0.0]), cache=cache)

    assert width == pytest.approx(4.0)
    assert width_again == pytest.approx(width)
    assert len(cache) == 1


def test_builder_creates_region_params_with_rho_and_theta_formula():
    constraints = make_constraints()
    builder = DirectionalCurvatureParameterBuilder(
        constraints=constraints,
        min_turning_radius=4.0,
    )
    regions = [rectangle(1.0, 0.0, width=4.0, height=4.0)]
    reference_path = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]

    params_by_edge = builder.build_for_edges(regions, reference_path)
    params = params_by_edge[0]

    assert np.allclose(params.t, [1.0, 0.0])
    assert params.rho == pytest.approx(min(0.75 * 2.0, 0.60 * 4.0))
    assert np.rad2deg(params.theta_max) == pytest.approx(25.0)
    assert params.kappa_max == pytest.approx(0.25)
    assert params.risk_flags == ()


def test_builder_skips_source_and_boundary_edges():
    constraints = make_constraints()
    builder = DirectionalCurvatureParameterBuilder(
        constraints=constraints,
        min_turning_radius=4.0,
    )
    regions = [
        rectangle(1.0, 0.0, width=4.0, height=4.0),
        rectangle(3.0, 0.0, width=4.0, height=4.0),
    ]
    source_edge = FakeEdge("source", "v0", "source_to_v0")
    interior_edge = FakeEdge("v0", "v1", "v0_to_v1")
    target_edge = FakeEdge("v1", "target", "v1_to_target")

    params_by_edge = builder.build_for_edges(
        regions,
        [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)],
        edges=[source_edge, interior_edge, target_edge],
        boundary_edge_ids={id(target_edge)},
    )

    assert set(params_by_edge.keys()) == {id(interior_edge)}


def test_builder_flags_high_risk_when_rho_is_small():
    constraints = make_constraints(direction_cone_compute_all_overlap_widths=True)
    builder = DirectionalCurvatureParameterBuilder(
        constraints=constraints,
        min_turning_radius=10.0,
    )
    regions = [rectangle(0.25, 0.0, width=0.5, height=2.0)]

    params = builder.build_for_edges(
        regions,
        [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
    )[0]

    assert "rho_below_warning" in params.risk_flags
    assert "overlap_unavailable" in params.risk_flags


def test_builder_can_skip_configured_risk_flags():
    constraints = make_constraints(
        direction_cone_skip_risk_flags=("direction_mismatch",)
    )
    builder = DirectionalCurvatureParameterBuilder(
        constraints=constraints,
        min_turning_radius=4.0,
    )
    regions = [
        rectangle(0.0, 0.0, width=2.0, height=2.0),
        rectangle(0.0, 5.0, width=2.0, height=2.0),
    ]

    params_by_edge = builder.build_for_edges(
        regions,
        [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
    )

    assert set(params_by_edge.keys()) == {1}
    assert builder.last_summary["skipped_risk_edges"] == 1
    assert builder.last_summary["skip_risk_flags"] == ["direction_mismatch"]


def test_directional_curvature_linear_constraints_have_expected_rows():
    params = DirectionalCurvatureSegmentParams(
        edge_id="edge",
        t=np.array([1.0, 0.0]),
        n=np.array([0.0, 1.0]),
        rho=1.5,
        theta_max=np.deg2rad(35.0),
        kappa_max=0.25,
    )
    first_derivative_coefficients = [np.array([[1.0, 0.0], [0.0, 1.0]])]
    second_derivative_coefficients = [np.array([[2.0, 0.0], [0.0, 3.0]])]

    rows = directional_curvature_linear_constraints(
        params,
        first_derivative_coefficients,
        second_derivative_coefficients,
    )

    assert len(rows) == 7
    assert rows[0].kind == "forward_speed"
    assert rows[0].lower == pytest.approx(params.rho)
    assert np.isposinf(rows[0].upper)
    assert np.allclose(rows[0].coefficients, [1.0, 0.0])

    curvature_rows = [row for row in rows if row.kind == "curvature"]
    assert len(curvature_rows) == 4
    assert all(row.lower == -np.inf for row in curvature_rows)
    assert all(row.upper == pytest.approx(params.kappa_max * params.rho**2) for row in curvature_rows)
