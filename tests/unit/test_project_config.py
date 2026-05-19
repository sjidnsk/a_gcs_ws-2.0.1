import math

import pytest

from config.project import load_project_config, parse_override


def test_load_default_config_resolves_scenarios_and_angles():
    config = load_project_config()

    basic = config.scenario_dict("basic")

    assert len(config.scenarios) >= 10
    assert basic["map_size"] == 200
    assert basic["start"] == pytest.approx((18.0, 2.25, 2.5))
    assert basic["goal"] == pytest.approx((18.0, 19.5, 0.1))


def test_overrides_use_dot_path_and_yaml_scalar_types():
    config = load_project_config(
        overrides=[
            "scenario.name=minimal",
            "ackermann.vehicle.max_velocity=8.5",
            "visualization.enabled=false",
        ]
    )

    assert config.scenario.name == "minimal"
    assert config.vehicle_params().max_velocity == pytest.approx(8.5)
    assert config.visualization.enabled is False
    assert parse_override("batch.num_runs=3") == (["batch", "num_runs"], 3)


def test_invalid_values_are_rejected():
    with pytest.raises(ValueError, match="max_velocity"):
        load_project_config(overrides=["ackermann.vehicle.max_velocity=-1.0"])

    with pytest.raises(ValueError, match="curvature_constraint_mode"):
        load_project_config(overrides=["ackermann.constraints.curvature_constraint_mode=banana"])

    with pytest.raises(ValueError, match="gear_strategy"):
        load_project_config(overrides=["ackermann.constraints.gear_strategy=banana"])

    with pytest.raises(ValueError, match="heading_deg"):
        load_project_config(overrides=["scenarios.basic.start.heading_deg=181.0"])


def test_unknown_scenario_error_is_clear():
    with pytest.raises(ValueError, match="Available scenarios"):
        load_project_config(overrides=["scenario.name=missing"])


def test_runtime_adapters_create_existing_dataclasses():
    config = load_project_config(overrides=["scenario.name=minimal"])

    vehicle = config.vehicle_params()
    bezier = config.bezier_config()
    constraints = config.trajectory_constraints(workspace_regions=[])
    planner_config = config.planner_config("minimal", enable_gcs_optimization=False)

    assert vehicle.wheelbase == pytest.approx(2.5)
    assert vehicle.max_steering_angle == pytest.approx(math.radians(85.0))
    assert bezier.order == 5
    assert constraints.curvature_constraint_mode == "direction_cone"
    assert constraints.gear_strategy == "none"
    assert constraints.min_velocity == pytest.approx(3.0)
    assert config.cost_weights()["reverse"] == pytest.approx(0.0)
    assert planner_config.corridor_width == pytest.approx(5.0)
    assert planner_config.enable_gcs_optimization is False
