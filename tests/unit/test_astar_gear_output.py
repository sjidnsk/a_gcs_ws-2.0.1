from types import SimpleNamespace

from A_pkg.A_star_base import SearchNode
from A_pkg.A_star_fast_optimized import FastSE2AStarPlanner


def test_reconstruct_path_keeps_legacy_pose_output_and_records_gears():
    planner = object.__new__(FastSE2AStarPlanner)
    planner.config = SimpleNamespace(path_interpolation=False)

    start = SearchNode(0.0, (0.0, 0.0, 0.0), 0.0, 0.0, motion_gear=1)
    reverse = SearchNode(
        1.0,
        (-1.0, 0.0, 0.0),
        1.0,
        0.0,
        parent=start,
        motion_gear=-1,
    )

    path = planner._reconstruct_path(reverse)

    assert path == [(0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]
    assert planner.last_path_with_gears == [
        (0.0, 0.0, 0.0, -1),
        (-1.0, 0.0, 0.0, -1),
    ]


def test_interpolated_points_inherit_segment_gear():
    planner = object.__new__(FastSE2AStarPlanner)
    planner.resolution = 1.0

    path_with_gears = [
        (0.0, 0.0, 0.0, -1),
        (-3.0, 0.0, 0.0, -1),
    ]

    interpolated = planner._interpolate_path_with_gears(path_with_gears)

    assert interpolated[0][3] == -1
    assert interpolated[-1][3] == -1
    assert all(point[3] == -1 for point in interpolated)
