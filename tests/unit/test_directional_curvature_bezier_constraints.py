import numpy as np
import pytest

pytest.importorskip("pydrake")

from pydrake.geometry.optimization import HPolyhedron

from ackermann_gcs_pkg.directional_curvature_parameters import (
    DirectionalCurvatureSegmentParams,
)
from gcs_pkg.scripts.core.bezier import BezierGCS


def test_directional_curvature_bezier_api_adds_linear_rows_per_edge():
    regions = [
        HPolyhedron.MakeBox([0.0, -1.0], [2.0, 1.0]),
        HPolyhedron.MakeBox([1.0, -1.0], [3.0, 1.0]),
    ]
    gcs = BezierGCS(regions, order=3, continuity=1)
    params_by_edge = {
        id(edge): DirectionalCurvatureSegmentParams(
            edge_id=id(edge),
            t=np.array([1.0, 0.0]),
            n=np.array([0.0, 1.0]),
            rho=0.5,
            theta_max=np.deg2rad(35.0),
            kappa_max=0.25,
        )
        for edge in gcs.gcs.Edges()
    }

    summary = gcs.addDirectionalCurvatureConstraint(params_by_edge)

    assert summary["rows_per_edge"] == 3 * 3 + 4 * 2
    assert summary["constrained_edges"] == len(gcs.gcs.Edges())
    assert summary["constraints_added"] == len(gcs.gcs.Edges()) * summary["rows_per_edge"]
    assert len(gcs.curvature_constraints) == summary["constraints_added"]
