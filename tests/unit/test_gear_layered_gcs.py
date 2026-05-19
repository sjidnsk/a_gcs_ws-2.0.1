import numpy as np
import pytest

pytest.importorskip("pydrake")

from pydrake.geometry.optimization import HPolyhedron

from ackermann_gcs_pkg.ackermann_bezier_gcs import GearLayeredAckermannBezierGCS
from ackermann_gcs_pkg.ackermann_data_structures import BezierConfig, VehicleParams


def vehicle_params():
    return VehicleParams(
        wheelbase=2.0,
        max_steering_angle=0.5,
        max_velocity=5.0,
        max_acceleration=2.0,
    )


def test_layered_gcs_creates_forward_reverse_and_switch_edges():
    regions = [
        HPolyhedron.MakeBox([0.0, -1.0], [2.0, 1.0]),
        HPolyhedron.MakeBox([1.0, -1.0], [3.0, 1.0]),
    ]

    gcs = GearLayeredAckermannBezierGCS(
        regions,
        vehicle_params(),
        bezier_config=BezierConfig(order=3, continuity=1),
        reverse_cost=0.5,
        gear_switch_cost=2.0,
    )

    vertex_names = sorted(vertex.name() for vertex in gcs.gcs.Vertices())
    assert vertex_names == ["f0", "f1", "r0", "r1"]
    assert gcs.layered_edge_summary["switch_edges"] == 4
    assert gcs.layered_edge_summary["moving_edges"] == 4
    assert gcs.layered_edge_summary["stationary_constraints"] > 0

    switch_edges = [
        edge for edge in gcs.gcs.Edges()
        if gcs.getEdgeMetadata(edge).get("is_switch")
    ]
    assert len(switch_edges) == 4
    assert {gcs.getEdgeMetadata(edge)["gear"] for edge in switch_edges} == {-1, 1}
