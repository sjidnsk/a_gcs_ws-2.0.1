"""Gear annotations for Ackermann GCS planning.

The utilities here are Drake-optional.  Gear is represented as +1 for forward
motion and -1 for reverse motion.  It is metadata for GCS edge selection and
diagnostics, not a replacement for geometric feasibility checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


_EPS = 1e-9


@dataclass(frozen=True)
class GearSegment:
    start_s: float
    end_s: float
    gear: int
    source: str = "infer"

    def __post_init__(self):
        if self.end_s < self.start_s:
            raise ValueError("GearSegment end_s must be >= start_s")
        object.__setattr__(self, "gear", normalize_gear(self.gear))


@dataclass(frozen=True)
class GearAnnotatedPath:
    points: np.ndarray
    gears: np.ndarray
    cumulative_s: np.ndarray
    segments: tuple[GearSegment, ...]

    def __post_init__(self):
        if self.points.ndim != 2 or self.points.shape[1] < 3:
            raise ValueError("points must have shape (N, >=3)")
        if len(self.gears) != len(self.points):
            raise ValueError("gears length must match points")
        if len(self.cumulative_s) != len(self.points):
            raise ValueError("cumulative_s length must match points")

        normalized_gears = np.array(
            [normalize_gear(value) for value in self.gears], dtype=int
        )
        object.__setattr__(self, "gears", normalized_gears)
        object.__setattr__(self, "segments", tuple(self.segments))


def normalize_gear(value) -> int:
    """Normalize a gear value to +1 or -1."""
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"gear must be +1 or -1, got {value!r}") from exc
    if numeric not in (-1, 1):
        raise ValueError(f"gear must be +1 or -1, got {value!r}")
    return numeric


def strip_gears(path: Sequence[Sequence[float]]) -> list[tuple[float, float, float]]:
    """Return the pose columns from a path that may include gear metadata."""
    poses = []
    for point in path:
        if len(point) < 3:
            raise ValueError("path points must contain at least x, y, theta")
        poses.append((float(point[0]), float(point[1]), float(point[2])))
    return poses


def annotate_reference_path(
    path: Sequence[Sequence[float]],
    source: str = "infer",
) -> GearAnnotatedPath:
    """Annotate a reference path using explicit gears or inferred gears."""
    raw = _as_path_array(path)
    if raw.shape[1] >= 4:
        gears = np.array([normalize_gear(value) for value in raw[:, 3]], dtype=int)
        return _build_annotated_path(raw[:, :3], gears, source)
    return infer_gears_from_path(path)


def infer_gears_from_path(
    path: Sequence[Sequence[float]],
    dot_tolerance: float = 1e-8,
) -> GearAnnotatedPath:
    """Infer point-wise gears from motion direction relative to heading."""
    raw = _as_path_array(path)
    poses = raw[:, :3]
    num_points = poses.shape[0]
    gears = np.ones(num_points, dtype=int)
    previous_gear = 1

    for idx in range(max(0, num_points - 1)):
        delta = poses[idx + 1, :2] - poses[idx, :2]
        if np.linalg.norm(delta) <= _EPS:
            gear = previous_gear
        else:
            heading_vec = np.array([np.cos(poses[idx, 2]), np.sin(poses[idx, 2])])
            projection = float(np.dot(delta, heading_vec))
            if projection < -dot_tolerance:
                gear = -1
            elif projection > dot_tolerance:
                gear = 1
            else:
                gear = previous_gear
        gears[idx] = gear
        previous_gear = gear

    if num_points > 1:
        gears[-1] = previous_gear
    return _build_annotated_path(poses, gears, "infer")


def gear_at_s(annotated_path: GearAnnotatedPath, s: float) -> int:
    """Return the gear active at arc-length coordinate ``s``."""
    if not annotated_path.segments:
        return normalize_gear(annotated_path.gears[0])

    s_value = float(s)
    if s_value <= annotated_path.segments[0].start_s:
        return annotated_path.segments[0].gear
    if s_value >= annotated_path.segments[-1].end_s:
        return annotated_path.segments[-1].gear

    for segment in annotated_path.segments:
        if segment.start_s <= s_value <= segment.end_s:
            return segment.gear
    return annotated_path.segments[-1].gear


def gear_summary(gears: Iterable[int]) -> dict:
    """Build compact gear diagnostics from a gear sequence."""
    normalized = [normalize_gear(value) for value in gears]
    switches = sum(
        1 for prev, curr in zip(normalized, normalized[1:]) if prev != curr
    )
    return {
        "gear_sequence": normalized,
        "reverse_count": int(sum(gear == -1 for gear in normalized)),
        "switch_count": int(switches),
    }


def _as_path_array(path: Sequence[Sequence[float]]) -> np.ndarray:
    raw = np.asarray(path, dtype=float)
    if raw.ndim != 2 or raw.shape[0] < 1 or raw.shape[1] < 3:
        raise ValueError("path must have shape (N, >=3)")
    return raw


def _build_annotated_path(
    poses: np.ndarray,
    gears: np.ndarray,
    source: str,
) -> GearAnnotatedPath:
    cumulative_s = _cumulative_path_length(poses[:, :2])
    segments: list[GearSegment] = []
    if len(poses) == 1:
        segments.append(GearSegment(0.0, 0.0, normalize_gear(gears[0]), source))
    else:
        for idx in range(len(poses) - 1):
            segments.append(
                GearSegment(
                    start_s=float(cumulative_s[idx]),
                    end_s=float(cumulative_s[idx + 1]),
                    gear=normalize_gear(gears[idx]),
                    source=source,
                )
            )
    return GearAnnotatedPath(
        points=np.asarray(poses, dtype=float),
        gears=np.asarray(gears, dtype=int),
        cumulative_s=cumulative_s,
        segments=tuple(segments),
    )


def _cumulative_path_length(points: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.array([0.0], dtype=float)
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    return np.concatenate(([0.0], np.cumsum(lengths)))
