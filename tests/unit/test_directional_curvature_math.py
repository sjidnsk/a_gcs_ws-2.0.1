import numpy as np
import pytest

from ackermann_gcs_pkg.directional_curvature_parameters import (
    DirectionalCurvatureSegmentParams,
)


def curvature(first_deriv, second_deriv):
    det = first_deriv[0] * second_deriv[1] - first_deriv[1] * second_deriv[0]
    return abs(det) / np.linalg.norm(first_deriv) ** 3


@pytest.mark.parametrize("theta_deg", [25.0, 35.0, 45.0])
def test_directional_curvature_conditions_bound_sampled_curvature(theta_deg):
    kappa_max = 0.4
    rho = 1.2
    params = DirectionalCurvatureSegmentParams(
        edge_id="math",
        t=np.array([1.0, 0.0]),
        n=np.array([0.0, 1.0]),
        rho=rho,
        theta_max=np.deg2rad(theta_deg),
        kappa_max=kappa_max,
    )

    rng = np.random.default_rng(42)
    for _ in range(100):
        alpha = rho + rng.uniform(0.0, 2.0)
        q = rng.uniform(-params.eta * alpha, params.eta * alpha)
        b = rng.uniform(-0.2, 0.2)
        c_limit = kappa_max * rho**2 - params.eta * abs(b)
        if c_limit < 0.0:
            b = 0.0
            c_limit = kappa_max * rho**2
        c = rng.uniform(-c_limit, c_limit)

        first_deriv = alpha * params.t + q * params.n
        second_deriv = b * params.t + c * params.n

        assert curvature(first_deriv, second_deriv) <= kappa_max + 1e-12
