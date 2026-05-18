"""Shared scenario helpers for planner scripts.

This module keeps reusable setup code out of the command-line entrypoints.
The functions here are intentionally lightweight at import time; Drake and
planner classes are imported lazily inside the functions that need them.
"""

from functools import lru_cache
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from config.project import ProjectConfig, load_project_config


@lru_cache(maxsize=1)
def default_project_config() -> ProjectConfig:
    """Load the default YAML config lazily for programmatic callers."""
    return load_project_config()


def create_endpoint_state(
    position: Tuple[float, float],
    heading: float,
    velocity: Optional[float] = None,
):
    """Create an Ackermann endpoint state."""
    from ackermann_gcs_pkg.ackermann_data_structures import EndpointState

    return EndpointState(
        position=np.array(position),
        heading=heading,
        velocity=velocity,
    )


def convert_iris_to_hpolyhedron(iris_regions: Sequence[Any]) -> List[Any]:
    """Convert IRIS/IrisZo regions to Drake HPolyhedron objects."""
    try:
        from pydrake.geometry.optimization import HPolyhedron
    except ImportError as exc:
        HPolyhedron = None
        drake_import_error = exc
    else:
        drake_import_error = None

    workspace_regions = []
    for region in iris_regions:
        if hasattr(region, "polyhedron"):
            workspace_regions.append(region.polyhedron)
            continue

        if hasattr(region, "A") and hasattr(region, "b"):
            if HPolyhedron is None:
                raise ImportError(
                    "pydrake is required to convert A/b IRIS regions to "
                    "HPolyhedron"
                ) from drake_import_error
            workspace_regions.append(HPolyhedron(region.A, region.b))
            continue

        raise ValueError(f"Unknown IRIS region type: {type(region)}")

    return workspace_regions


def create_test_map(map_size: int = 200, scenario: str = "basic") -> np.ndarray:
    """Create an obstacle map for a named test scenario."""
    obstacle_map = np.zeros((map_size, map_size), dtype=np.uint8)

    if scenario == "basic":
        obstacle_map[40:80, 60:100] = 1
        obstacle_map[120:160, 40:80] = 1
        obstacle_map[100:140, 120:160] = 1
        obstacle_map[60:100, 140:160] = 1
        obstacle_map[80:100, 160:200] = 1
        for i in range(map_size):
            for j in range(map_size):
                if (i - 150) ** 2 + (j - 150) ** 2 < 25**2:
                    obstacle_map[i, j] = 1

    elif scenario == "minimal":
        obstacle_map[0:10, :] = 1
        obstacle_map[90:100, :] = 1
        obstacle_map[:, 0:10] = 1
        obstacle_map[:, 90:100] = 1

    elif scenario == "simple_straight":
        obstacle_map[30:50, 80:120] = 1
        obstacle_map[150:170, 80:120] = 1
        obstacle_map[0:20, :] = 1
        obstacle_map[180:200, :] = 1
        obstacle_map[:, 0:20] = 1
        obstacle_map[:, 180:200] = 1

    elif scenario == "sharp_turn":
        obstacle_map[0:80, 0:80] = 1
        obstacle_map[120:200, 120:200] = 1
        obstacle_map[0:20, :] = 1
        obstacle_map[180:200, :] = 1
        obstacle_map[:, 0:20] = 1
        obstacle_map[:, 180:200] = 1

    elif scenario == "corridor_passage":
        obstacle_map[40:80, 70:90] = 1
        obstacle_map[120:160, 70:90] = 1
        obstacle_map[0:20, :] = 1
        obstacle_map[180:200, :] = 1
        obstacle_map[:, 0:20] = 1
        obstacle_map[:, 180:200] = 1

    elif scenario == "slalom":
        obstacle_map[60:90, 80:110] = 1
        obstacle_map[120:150, 130:160] = 1
        obstacle_map[180:210, 80:110] = 1
        obstacle_map[0:20, :] = 1
        obstacle_map[230:250, :] = 1
        obstacle_map[:, 0:20] = 1
        obstacle_map[:, 230:250] = 1

    elif scenario == "maze_navigation":
        obstacle_map[0:20, :] = 1
        obstacle_map[280:300, :] = 1
        obstacle_map[:, 0:20] = 1
        obstacle_map[:, 280:300] = 1
        obstacle_map[80:120, 80:120] = 1
        obstacle_map[160:200, 160:200] = 1
        obstacle_map[80:120, 180:220] = 1

    elif scenario == "gentle_turn":
        obstacle_map[30:60, 130:180] = 1
        obstacle_map[150:180, 30:80] = 1
        obstacle_map[0:20, :] = 1
        obstacle_map[180:200, :] = 1
        obstacle_map[:, 0:20] = 1
        obstacle_map[:, 180:200] = 1

    elif scenario == "narrow":
        obstacle_map[0:100, :] = 1
        obstacle_map[150:250, :] = 1

    elif scenario == "complex":
        obstacle_map[50:150, 30:80] = 1
        obstacle_map[180:280, 120:170] = 1
        obstacle_map[300:400, 50:100] = 1
        obstacle_map[100:200, 350:400] = 1
        obstacle_map[350:450, 300:350] = 1

        for i in range(50, 150):
            for j in range(200, 300):
                if abs(i - j + 150) < 10:
                    obstacle_map[i, j] = 1

        circles = [
            (100, 250, 30),
            (200, 200, 25),
            (300, 350, 35),
            (400, 150, 28),
            (150, 400, 32),
            (380, 280, 22),
            (250, 100, 27),
            (80, 320, 20),
            (420, 380, 25),
            (320, 80, 23),
        ]
        for cx, cy, radius in circles:
            for i in range(max(0, cx - radius), min(map_size, cx + radius)):
                for j in range(max(0, cy - radius), min(map_size, cy + radius)):
                    if (i - cx) ** 2 + (j - cy) ** 2 < radius**2:
                        obstacle_map[i, j] = 1

        obstacle_map[200:250, 50:80] = 1
        obstacle_map[200:250, 100:130] = 1
        obstacle_map[200:250, 150:180] = 1
        obstacle_map[200:250, 200:230] = 1
        obstacle_map[200:250, 250:280] = 1
        obstacle_map[200:250, 300:330] = 1
        obstacle_map[200:250, 350:380] = 1
        obstacle_map[200:250, 400:430] = 1

        obstacle_map[50:100, 200:250] = 1
        obstacle_map[50:150, 200:220] = 1
        obstacle_map[300:350, 400:450] = 1
        obstacle_map[300:400, 400:420] = 1

        small_obstacles = [
            (120, 120, 15),
            (180, 320, 12),
            (280, 220, 14),
            (350, 180, 13),
            (420, 320, 16),
            (90, 380, 11),
            (220, 380, 14),
            (380, 120, 12),
            (150, 80, 15),
            (320, 320, 13),
        ]
        for cx, cy, radius in small_obstacles:
            for i in range(max(0, cx - radius), min(map_size, cx + radius)):
                for j in range(max(0, cy - radius), min(map_size, cy + radius)):
                    if (i - cx) ** 2 + (j - cy) ** 2 < radius**2:
                        obstacle_map[i, j] = 1

    elif scenario == "u_turn":
        obstacle_map[0:120, 0:150] = 1
        obstacle_map[120:200, 0:45] = 1
        obstacle_map[120:200, 105:150] = 1

    elif scenario == "s_curve":
        obstacle_map[0:80, 0:150] = 1
        obstacle_map[120:200, 50:200] = 1
        obstacle_map[0:80, 150:200] = 1
        obstacle_map[120:200, 0:50] = 1

    elif scenario == "dynamic":
        obstacle_map[50:100, 50:100] = 1
        obstacle_map[150:200, 100:150] = 1
        obstacle_map[100:150, 150:200] = 1

    elif scenario == "multi_goal":
        obstacle_map[50:80, 80:120] = 1
        obstacle_map[120:150, 50:80] = 1
        obstacle_map[120:150, 120:150] = 1
        obstacle_map[80:120, 170:200] = 1

    elif scenario == "parking":
        obstacle_map[0:40, 0:200] = 1
        obstacle_map[160:200, 0:200] = 1
        obstacle_map[0:200, 0:40] = 1
        obstacle_map[0:200, 160:200] = 1
        obstacle_map[70:130, 70:90] = 1

    return obstacle_map


def plan_path(
    c_space: Any,
    start: Tuple[float, float, float],
    goal: Tuple[float, float, float],
    project_config: Optional[ProjectConfig] = None,
) -> Optional[List[Any]]:
    """Run the SE(2) A* planner for a scenario."""
    from A_pkg.A_star_fast_optimized import FastSE2AStarPlanner
    from C_space_pkg.se2 import create_rectangle_robot

    project_config = project_config or default_project_config()
    robot = create_rectangle_robot(length=1.5, width=1.0)
    planner = FastSE2AStarPlanner(
        c_space=c_space,
        robot=robot,
        min_radius=project_config.astar.min_radius,
        resolution=project_config.astar.resolution,
        theta_resolution=project_config.astar.theta_resolution,
        config=project_config.astar_planner_config(),
    )
    return planner.plan(start, goal)

