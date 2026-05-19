import numpy as np
import pytest

from ackermann_gcs_pkg.gear_annotations import (
    GearSegment,
    annotate_reference_path,
    gear_at_s,
    infer_gears_from_path,
    normalize_gear,
    strip_gears,
)


def test_normalize_gear_accepts_only_forward_or_reverse():
    assert normalize_gear(1) == 1
    assert normalize_gear(-1) == -1

    with pytest.raises(ValueError, match="gear"):
        normalize_gear(0)


def test_strip_gears_keeps_pose_columns_only():
    path = [(0.0, 0.0, 0.0, 1), (1.0, 0.0, 0.0, -1)]

    assert strip_gears(path) == [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]


def test_infer_gears_from_path_detects_reverse_motion():
    path = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]

    annotated = infer_gears_from_path(path)

    assert annotated.gears.tolist() == [-1, -1, -1]
    assert all(segment.gear == -1 for segment in annotated.segments)


def test_annotate_reference_path_uses_explicit_gear_columns():
    path = [
        (0.0, 0.0, 0.0, 1),
        (1.0, 0.0, 0.0, -1),
        (0.0, 0.0, 0.0, -1),
    ]

    annotated = annotate_reference_path(path, source="astar")

    assert annotated.gears.tolist() == [1, -1, -1]
    assert annotated.segments[0] == GearSegment(0.0, 1.0, 1, "astar")
    assert annotated.segments[1].gear == -1
    assert gear_at_s(annotated, 1.5) == -1


def test_zero_length_segments_inherit_previous_gear():
    path = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
    ]

    annotated = infer_gears_from_path(path)

    assert np.allclose(annotated.cumulative_s, [0.0, 0.0, 1.0])
    assert annotated.gears.tolist() == [1, -1, -1]
    assert gear_at_s(annotated, 0.0) == 1
