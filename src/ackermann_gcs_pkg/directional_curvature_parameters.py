"""Direction-cone curvature parameter generation.

This module is intentionally Drake-optional. It can compute the geometric
parameters from vertex-backed regions in pure Python, and falls back to small
linear programs for H-polyhedra that expose A/b constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


_EPS = 1e-9


@dataclass(frozen=True)
class PreprocessedReferencePath:
    points: np.ndarray
    headings: np.ndarray
    gears: Optional[np.ndarray]
    cumulative_s: np.ndarray
    tangents: np.ndarray
    ds_ref: float

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError("points must have shape (N, 2)")
        if self.tangents.shape != self.points.shape:
            raise ValueError("tangents must have the same shape as points")
        if len(self.cumulative_s) != len(self.points):
            raise ValueError("cumulative_s length must match points")


@dataclass
class DirectionalCurvatureSegmentParams:
    edge_id: int | str
    t: np.ndarray
    n: np.ndarray
    rho: float
    theta_max: float
    kappa_max: float
    risk_flags: tuple[str, ...] = ()
    eta: float = field(init=False)

    def __post_init__(self):
        t = _normalize_vector(self.t, "t")
        n_raw = np.asarray(self.n, dtype=float)
        if n_raw.shape != (2,):
            raise ValueError(f"n must be a 2D vector, got shape {n_raw.shape}")
        n = n_raw - np.dot(n_raw, t) * t
        if np.linalg.norm(n) <= _EPS:
            n = np.array([-t[1], t[0]])
        n = _normalize_vector(n, "n")

        if self.rho <= 0 or not np.isfinite(self.rho):
            raise ValueError(f"rho must be finite and positive, got {self.rho}")
        if self.theta_max <= 0 or self.theta_max >= np.pi / 2:
            raise ValueError(
                "theta_max must be in (0, pi/2), "
                f"got {self.theta_max}"
            )
        if self.kappa_max <= 0 or not np.isfinite(self.kappa_max):
            raise ValueError(
                f"kappa_max must be finite and positive, got {self.kappa_max}"
            )

        self.t = t
        self.n = n
        self.rho = float(self.rho)
        self.theta_max = float(self.theta_max)
        self.kappa_max = float(self.kappa_max)
        self.risk_flags = tuple(self.risk_flags)
        self.eta = float(np.tan(self.theta_max))

    def todict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "t": self.t.tolist(),
            "n": self.n.tolist(),
            "rho": self.rho,
            "theta_max": self.theta_max,
            "eta": self.eta,
            "kappa_max": self.kappa_max,
            "risk_flags": list(self.risk_flags),
        }


@dataclass(frozen=True)
class DirectionalLinearConstraintRow:
    coefficients: np.ndarray
    lower: float
    upper: float
    kind: str
    control_index: int


def directional_curvature_linear_constraints(
    params: DirectionalCurvatureSegmentParams,
    first_derivative_coefficients: Sequence[np.ndarray],
    second_derivative_coefficients: Sequence[np.ndarray],
) -> list[DirectionalLinearConstraintRow]:
    """Build linear rows for one edge's direction-cone curvature constraints."""
    rows: list[DirectionalLinearConstraintRow] = []

    for idx, coefficients in enumerate(first_derivative_coefficients):
        matrix = _as_derivative_matrix(coefficients)
        t_row = params.t @ matrix
        n_row = params.n @ matrix
        rows.append(
            DirectionalLinearConstraintRow(
                coefficients=t_row,
                lower=params.rho,
                upper=np.inf,
                kind="forward_speed",
                control_index=idx,
            )
        )
        rows.append(
            DirectionalLinearConstraintRow(
                coefficients=n_row - params.eta * t_row,
                lower=-np.inf,
                upper=0.0,
                kind="direction_cone_upper",
                control_index=idx,
            )
        )
        rows.append(
            DirectionalLinearConstraintRow(
                coefficients=-n_row - params.eta * t_row,
                lower=-np.inf,
                upper=0.0,
                kind="direction_cone_lower",
                control_index=idx,
            )
        )

    curvature_bound = params.kappa_max * params.rho ** 2
    for idx, coefficients in enumerate(second_derivative_coefficients):
        matrix = _as_derivative_matrix(coefficients)
        t_row = params.t @ matrix
        n_row = params.n @ matrix
        for sigma in (-1.0, 1.0):
            for tau in (-1.0, 1.0):
                rows.append(
                    DirectionalLinearConstraintRow(
                        coefficients=sigma * n_row + params.eta * tau * t_row,
                        lower=-np.inf,
                        upper=curvature_bound,
                        kind="curvature",
                        control_index=idx,
                    )
                )

    return rows


def preprocess_reference_path(
    reference_path: Sequence[Sequence[float]],
    min_turning_radius: float,
    ds_ref: Optional[float] = None,
    tangent_window_length: Optional[float] = None,
    min_segment_length: float = 1e-6,
) -> PreprocessedReferencePath:
    """Deduplicate, resample, unwrap headings, and smooth path tangents."""
    if min_turning_radius <= 0:
        raise ValueError(
            f"min_turning_radius must be positive, got {min_turning_radius}"
        )

    raw = np.asarray(reference_path, dtype=float)
    if raw.ndim != 2 or raw.shape[0] < 2 or raw.shape[1] < 2:
        raise ValueError("reference_path must have shape (N, >=2), N >= 2")

    points = raw[:, :2]
    headings = raw[:, 2] if raw.shape[1] >= 3 else _headings_from_points(points)
    headings = np.unwrap(headings)
    gears = raw[:, 3] if raw.shape[1] >= 4 else None

    keep = [0]
    for idx in range(1, len(points)):
        if np.linalg.norm(points[idx] - points[keep[-1]]) >= min_segment_length:
            keep.append(idx)
    if len(keep) < 2:
        raise ValueError("reference_path must contain at least two distinct points")

    points = points[keep]
    headings = headings[keep]
    gears = gears[keep] if gears is not None else None

    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total_length = float(cumulative[-1])
    if total_length <= min_segment_length:
        raise ValueError("reference_path length is too small")

    if ds_ref is None:
        ds_ref = max(0.5, 0.1 * min_turning_radius)
    if ds_ref <= 0:
        raise ValueError(f"ds_ref must be positive, got {ds_ref}")

    num_intervals = max(1, int(np.ceil(total_length / ds_ref)))
    sample_s = np.linspace(0.0, total_length, num_intervals + 1)
    resampled_points = np.column_stack(
        [
            np.interp(sample_s, cumulative, points[:, 0]),
            np.interp(sample_s, cumulative, points[:, 1]),
        ]
    )
    resampled_headings = np.interp(sample_s, cumulative, headings)
    resampled_gears = None
    if gears is not None:
        source_indices = np.searchsorted(cumulative, sample_s, side="right") - 1
        source_indices = np.clip(source_indices, 0, len(gears) - 1)
        resampled_gears = gears[source_indices]

    if tangent_window_length is None:
        tangent_window_length = max(1.0, 0.2 * min_turning_radius)
    half_window = max(1, int(round(0.5 * tangent_window_length / ds_ref)))
    tangents = np.zeros_like(resampled_points)
    for idx in range(len(resampled_points)):
        lo = max(0, idx - half_window)
        hi = min(len(resampled_points) - 1, idx + half_window)
        delta = resampled_points[hi] - resampled_points[lo]
        if np.linalg.norm(delta) <= _EPS:
            if idx + 1 < len(resampled_points):
                delta = resampled_points[idx + 1] - resampled_points[idx]
            else:
                delta = resampled_points[idx] - resampled_points[idx - 1]
        tangents[idx] = _normalize_vector(delta, "reference tangent")

    return PreprocessedReferencePath(
        points=resampled_points,
        headings=resampled_headings,
        gears=resampled_gears,
        cumulative_s=sample_s,
        tangents=tangents,
        ds_ref=float(ds_ref),
    )


def compute_support_width(
    region: Any,
    direction: np.ndarray,
    cache: Optional[Dict[Any, float]] = None,
    use_vertex_width_when_available: bool = True,
    cache_enabled: bool = True,
    direction_angle_resolution_deg: float = 5.0,
) -> float:
    """Compute support(C, d) - support(C, -d)."""
    direction = _normalize_vector(direction, "direction")
    key = None
    if cache_enabled:
        key = _support_cache_key(region, direction, direction_angle_resolution_deg)
        if cache is not None and key in cache:
            return cache[key]

    vertices = _extract_vertices(region) if use_vertex_width_when_available else None
    max_support = _support_value(region, direction, vertices=vertices)
    min_support = _support_value(region, -direction, vertices=vertices)
    width = float(max_support + min_support)
    if not np.isfinite(width) or width < -1e-8:
        raise ValueError(f"support width is invalid: {width}")
    width = max(0.0, width)

    if cache_enabled and cache is not None:
        cache[key] = width
    return width


class DirectionalCurvatureParameterBuilder:
    """Build edge-wise direction-cone curvature parameters."""

    def __init__(
        self,
        constraints: Any,
        min_turning_radius: Optional[float] = None,
        support_cache: Optional[Dict[Any, float]] = None,
    ):
        self.constraints = constraints
        self.kappa_max = float(constraints.max_curvature)
        self.min_turning_radius = (
            float(min_turning_radius)
            if min_turning_radius is not None
            else 1.0 / self.kappa_max
        )
        if self.min_turning_radius <= 0:
            raise ValueError("min_turning_radius must be positive")

        self.alpha = float(constraints.direction_cone_alpha)
        self.beta = float(constraints.direction_cone_beta)
        self.gamma = float(constraints.direction_cone_gamma)
        self.theta_min = np.deg2rad(constraints.direction_cone_theta_min_deg)
        self.theta_abs_max = np.deg2rad(
            constraints.direction_cone_theta_abs_max_deg
        )
        self.theta_margin = np.deg2rad(
            constraints.direction_cone_theta_margin_deg
        )
        self.width_mu = float(constraints.direction_cone_width_mu)
        self.compute_all_overlap_widths = bool(
            constraints.direction_cone_compute_all_overlap_widths
        )
        self.cache_support_widths = bool(
            constraints.direction_cone_cache_support_widths
        )
        self.use_vertex_width_when_available = bool(
            constraints.direction_cone_use_vertex_width_when_available
        )
        self.rho_warning_ratio = float(
            constraints.direction_cone_rho_warning_ratio
        )
        self.skip_risk_flags = set(
            getattr(constraints, "direction_cone_skip_risk_flags", ())
        )
        self.support_cache = support_cache if support_cache is not None else {}
        self.last_summary: Dict[str, Any] = {}

    def build_for_edges(
        self,
        regions: Sequence[Any],
        reference_path: Sequence[Sequence[float]],
        edges: Optional[Iterable[Any]] = None,
        edge_reference_map: Optional[Mapping[Any, Any]] = None,
        boundary_edge_ids: Optional[set[Any]] = None,
    ) -> Dict[Any, DirectionalCurvatureSegmentParams]:
        if not regions:
            raise ValueError("regions must not be empty")

        processed = preprocess_reference_path(
            reference_path,
            min_turning_radius=self.min_turning_radius,
        )
        boundary_edge_ids = boundary_edge_ids or set()
        params_by_edge: Dict[Any, DirectionalCurvatureSegmentParams] = {}
        skipped_risk_edges = 0

        if edges is None:
            for region_idx in range(len(regions)):
                params = self._build_params(
                    edge_id=region_idx,
                    region_idx=region_idx,
                    next_region_idx=region_idx + 1
                    if region_idx + 1 < len(regions)
                    else None,
                    regions=regions,
                    processed=processed,
                    edge_reference=None,
                )
                if self._should_skip_params(params):
                    skipped_risk_edges += 1
                    continue
                params_by_edge[region_idx] = params
        else:
            for edge in edges:
                if _edge_touches_source(edge):
                    continue
                if _edge_matches_any(edge, boundary_edge_ids):
                    continue
                region_idx = _edge_u_region_index(edge)
                if region_idx is None or not (0 <= region_idx < len(regions)):
                    continue
                edge_id = id(edge)
                params = self._build_params(
                    edge_id=edge_id,
                    region_idx=region_idx,
                    next_region_idx=_edge_v_region_index(edge),
                    regions=regions,
                    processed=processed,
                    edge_reference=_lookup_edge_reference(edge, edge_reference_map),
                )
                if self._should_skip_params(params):
                    skipped_risk_edges += 1
                    continue
                params_by_edge[edge_id] = params

        if not params_by_edge:
            raise ValueError("no direction-cone parameters could be built")
        self.last_summary = self._summarize(params_by_edge)
        self.last_summary["skipped_risk_edges"] = skipped_risk_edges
        self.last_summary["skip_risk_flags"] = sorted(self.skip_risk_flags)
        return params_by_edge

    def _should_skip_params(self, params: DirectionalCurvatureSegmentParams) -> bool:
        return bool(self.skip_risk_flags.intersection(params.risk_flags))

    def _build_params(
        self,
        edge_id: Any,
        region_idx: int,
        next_region_idx: Optional[int],
        regions: Sequence[Any],
        processed: PreprocessedReferencePath,
        edge_reference: Optional[Any],
    ) -> DirectionalCurvatureSegmentParams:
        start_s, end_s = _edge_s_interval(
            edge_reference=edge_reference,
            region_idx=region_idx,
            num_regions=len(regions),
            processed=processed,
        )
        mid_s = 0.5 * (start_s + end_s)
        sample_idx = int(np.searchsorted(processed.cumulative_s, mid_s))
        sample_idx = int(np.clip(sample_idx, 0, len(processed.tangents) - 1))

        t_path = processed.tangents[sample_idx]
        n_path = np.array([-t_path[1], t_path[0]])
        risk_flags: list[str] = []

        ell_path = _projected_path_length(processed, t_path, start_s, end_s)
        if ell_path <= _EPS:
            risk_flags.append("path_projection_degenerate")
            ell_path = _EPS

        region = regions[region_idx]
        w_parallel = compute_support_width(
            region,
            t_path,
            cache=self.support_cache,
            use_vertex_width_when_available=self.use_vertex_width_when_available,
            cache_enabled=self.cache_support_widths,
        )
        w_lateral = compute_support_width(
            region,
            n_path,
            cache=self.support_cache,
            use_vertex_width_when_available=self.use_vertex_width_when_available,
            cache_enabled=self.cache_support_widths,
        )
        if w_parallel <= _EPS:
            risk_flags.append("parallel_width_degenerate")
            w_parallel = _EPS

        rho_scale = 1.0
        if next_region_idx is not None and 0 <= next_region_idx < len(regions):
            t_region = _region_connection_direction(
                regions[region_idx],
                regions[next_region_idx],
            )
            if t_region is not None:
                mismatch = _angle_between(t_path, t_region)
                if mismatch > np.deg2rad(45.0):
                    risk_flags.append("direction_mismatch")
                if mismatch > np.deg2rad(60.0):
                    rho_scale = 0.7

        rho = rho_scale * min(self.beta * ell_path, self.alpha * w_parallel)
        if rho < self.rho_warning_ratio * self.min_turning_radius:
            risk_flags.append("rho_below_warning")

        width_ratio = w_parallel / max(ell_path, _EPS)
        if width_ratio < 0.25:
            risk_flags.append("parallel_width_small")

        should_compute_overlap = (
            self.compute_all_overlap_widths
            or "rho_below_warning" in risk_flags
            or "parallel_width_small" in risk_flags
            or "direction_mismatch" in risk_flags
        )
        if should_compute_overlap:
            if next_region_idx is not None and 0 <= next_region_idx < len(regions):
                try:
                    overlap_width = _compute_overlap_width(
                        regions[region_idx],
                        regions[next_region_idx],
                        t_path,
                    )
                    rho = min(rho, self.gamma * overlap_width)
                    risk_flags.append("overlap_used")
                except ValueError as exc:
                    flag = (
                        "overlap_infeasible"
                        if "infeasible" in str(exc)
                        else "overlap_unavailable"
                    )
                    risk_flags.append(flag)
            else:
                risk_flags.append("overlap_unavailable")

        theta_width = float(np.arctan(self.width_mu * w_lateral / max(w_parallel, _EPS)))
        local_turn = _local_tangent_deviation(processed, sample_idx)
        desired_theta = local_turn + self.theta_margin
        theta_upper = min(theta_width, self.theta_abs_max)
        if theta_upper < self.theta_min:
            theta_max = max(theta_upper, np.deg2rad(1.0))
            risk_flags.append("theta_width_below_min")
        else:
            theta_max = min(max(desired_theta, self.theta_min), theta_upper)

        if rho <= _EPS:
            raise ValueError(f"rho for edge {edge_id!r} is non-positive")

        return DirectionalCurvatureSegmentParams(
            edge_id=edge_id,
            t=t_path,
            n=n_path,
            rho=rho,
            theta_max=theta_max,
            kappa_max=self.kappa_max,
            risk_flags=tuple(dict.fromkeys(risk_flags)),
        )

    @staticmethod
    def _summarize(
        params_by_edge: Mapping[Any, DirectionalCurvatureSegmentParams]
    ) -> Dict[str, Any]:
        values = list(params_by_edge.values())
        rhos = np.array([p.rho for p in values], dtype=float)
        thetas = np.array([p.theta_max for p in values], dtype=float)
        return {
            "num_edges": len(values),
            "rho_min": float(np.min(rhos)),
            "rho_max": float(np.max(rhos)),
            "theta_min_deg": float(np.rad2deg(np.min(thetas))),
            "theta_max_deg": float(np.rad2deg(np.max(thetas))),
            "risk_edge_count": int(sum(bool(p.risk_flags) for p in values)),
            "overlap_edge_count": int(
                sum("overlap_used" in p.risk_flags for p in values)
            ),
        }


def _normalize_vector(vector: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    if arr.shape != (2,):
        raise ValueError(f"{name} must be a 2D vector, got shape {arr.shape}")
    norm = float(np.linalg.norm(arr))
    if norm <= _EPS or not np.isfinite(norm):
        raise ValueError(f"{name} must have finite non-zero norm")
    return arr / norm


def _as_derivative_matrix(coefficients: np.ndarray) -> np.ndarray:
    matrix = np.asarray(coefficients, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 2:
        raise ValueError(
            "derivative coefficients must have shape (2, num_vars), "
            f"got {matrix.shape}"
        )
    return matrix


def _headings_from_points(points: np.ndarray) -> np.ndarray:
    headings = np.zeros(len(points), dtype=float)
    for idx in range(len(points)):
        if idx + 1 < len(points):
            delta = points[idx + 1] - points[idx]
        else:
            delta = points[idx] - points[idx - 1]
        headings[idx] = np.arctan2(delta[1], delta[0])
    return headings


def _extract_vertices(region: Any) -> Optional[np.ndarray]:
    vertices = None
    if hasattr(region, "get_vertices_ordered"):
        vertices = region.get_vertices_ordered()
    elif hasattr(region, "vertices"):
        vertices_attr = getattr(region, "vertices")
        vertices = vertices_attr() if callable(vertices_attr) else vertices_attr

    if vertices is None:
        return None
    arr = np.asarray(vertices, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim != 2:
        return None
    if arr.shape[1] == 2:
        return arr
    if arr.shape[0] == 2:
        return arr.T
    return None


def _get_region_ab(region: Any) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(region, "A"):
        A_attr = getattr(region, "A")
        A = A_attr() if callable(A_attr) else A_attr
    else:
        A = None
    if hasattr(region, "b"):
        b_attr = getattr(region, "b")
        b = b_attr() if callable(b_attr) else b_attr
    else:
        b = None
    if A is None or b is None:
        raise ValueError("region has no vertices or A/b representation")
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0]:
        raise ValueError("invalid A/b region representation")
    return A, b


def _support_value(
    region: Any,
    direction: np.ndarray,
    vertices: Optional[np.ndarray] = None,
) -> float:
    if vertices is not None:
        return float(np.max(vertices @ direction))
    A, b = _get_region_ab(region)
    return _support_value_from_ab(A, b, direction)


def _support_value_from_ab(A: np.ndarray, b: np.ndarray, direction: np.ndarray) -> float:
    try:
        from scipy.optimize import linprog
    except ImportError as exc:
        raise ValueError("scipy is required for A/b support queries") from exc

    result = linprog(
        c=-direction,
        A_ub=A,
        b_ub=b,
        bounds=[(None, None)] * A.shape[1],
        method="highs",
    )
    if not result.success:
        raise ValueError("support LP infeasible")
    return float(direction @ result.x)


def _support_cache_key(
    region: Any,
    direction: np.ndarray,
    angle_resolution_deg: float,
) -> Any:
    if len(direction) == 2:
        angle = float(np.arctan2(direction[1], direction[0]))
        bucket = int(round(np.rad2deg(angle) / angle_resolution_deg))
        return (id(region), bucket)
    rounded = tuple(np.round(direction, 3))
    return (id(region), rounded)


def _compute_overlap_width(region_a: Any, region_b: Any, direction: np.ndarray) -> float:
    A1, b1 = _get_region_ab(region_a)
    A2, b2 = _get_region_ab(region_b)
    A = np.vstack((A1, A2))
    b = np.concatenate((b1, b2))
    max_support = _support_value_from_ab(A, b, direction)
    min_support = _support_value_from_ab(A, b, -direction)
    width = max_support + min_support
    if width <= _EPS:
        raise ValueError("overlap infeasible")
    return float(width)


def _region_center(region: Any) -> Optional[np.ndarray]:
    vertices = _extract_vertices(region)
    if vertices is not None:
        return np.mean(vertices, axis=0)
    for attr_name in ("ChebyshevCenter", "center"):
        if hasattr(region, attr_name):
            attr = getattr(region, attr_name)
            value = attr() if callable(attr) else attr
            arr = np.asarray(value, dtype=float)
            if arr.shape == (2,):
                return arr
    return None


def _region_connection_direction(region_a: Any, region_b: Any) -> Optional[np.ndarray]:
    center_a = _region_center(region_a)
    center_b = _region_center(region_b)
    if center_a is None or center_b is None:
        return None
    delta = center_b - center_a
    if np.linalg.norm(delta) <= _EPS:
        return None
    return _normalize_vector(delta, "region connection")


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize_vector(a, "a")
    b = _normalize_vector(b, "b")
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.arccos(dot))


def _local_tangent_deviation(
    processed: PreprocessedReferencePath,
    sample_idx: int,
    radius: int = 1,
) -> float:
    lo = max(0, sample_idx - radius)
    hi = min(len(processed.tangents) - 1, sample_idx + radius)
    if lo == hi:
        return 0.0
    return 0.5 * _angle_between(processed.tangents[lo], processed.tangents[hi])


def _projected_path_length(
    processed: PreprocessedReferencePath,
    direction: np.ndarray,
    start_s: float,
    end_s: float,
) -> float:
    total = 0.0
    for idx, delta in enumerate(np.diff(processed.points, axis=0)):
        seg_start = processed.cumulative_s[idx]
        seg_end = processed.cumulative_s[idx + 1]
        if seg_end < start_s - _EPS or seg_start > end_s + _EPS:
            continue
        total += max(0.0, float(np.dot(direction, delta)))
    return total


def _edge_s_interval(
    edge_reference: Optional[Any],
    region_idx: int,
    num_regions: int,
    processed: PreprocessedReferencePath,
) -> tuple[float, float]:
    if edge_reference is not None:
        if isinstance(edge_reference, (int, np.integer)):
            idx = int(np.clip(edge_reference, 0, len(processed.cumulative_s) - 1))
            return processed.cumulative_s[idx], processed.cumulative_s[idx]
        if isinstance(edge_reference, Sequence) and len(edge_reference) == 2:
            lo = int(np.clip(edge_reference[0], 0, len(processed.cumulative_s) - 1))
            hi = int(np.clip(edge_reference[1], 0, len(processed.cumulative_s) - 1))
            if hi < lo:
                lo, hi = hi, lo
            return processed.cumulative_s[lo], processed.cumulative_s[hi]

    total = processed.cumulative_s[-1]
    start_s = total * region_idx / max(1, num_regions)
    end_s = total * (region_idx + 1) / max(1, num_regions)
    return float(start_s), float(end_s)


def _edge_vertex_name(vertex: Any) -> str:
    if vertex is None:
        return ""
    name_attr = getattr(vertex, "name", None)
    if callable(name_attr):
        return str(name_attr())
    return str(name_attr or "")


def _parse_region_index(name: str) -> Optional[int]:
    if name.startswith("v"):
        suffix = name[1:]
        if suffix.isdigit():
            return int(suffix)
    return None


def _edge_u_region_index(edge: Any) -> Optional[int]:
    return _parse_region_index(_edge_vertex_name(edge.u()))


def _edge_v_region_index(edge: Any) -> Optional[int]:
    return _parse_region_index(_edge_vertex_name(edge.v()))


def _edge_touches_source(edge: Any) -> bool:
    return _edge_vertex_name(edge.u()) == "source"


def _edge_keys(edge: Any) -> set[Any]:
    keys: set[Any] = {id(edge)}
    for attr_name in ("id", "name"):
        attr = getattr(edge, attr_name, None)
        if callable(attr):
            try:
                keys.add(attr())
            except Exception:
                pass
    return keys


def _edge_matches_any(edge: Any, candidates: set[Any]) -> bool:
    return bool(_edge_keys(edge).intersection(candidates))


def _lookup_edge_reference(edge: Any, edge_reference_map: Optional[Mapping[Any, Any]]):
    if edge_reference_map is None:
        return None
    for key in _edge_keys(edge):
        if key in edge_reference_map:
            return edge_reference_map[key]
    return None
