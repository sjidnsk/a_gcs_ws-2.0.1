"""
Project-level configuration loading and adapters.

This module keeps experiment parameters editable in YAML while preserving the
existing dataclass-based runtime objects used by the planners.
"""

import copy
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, get_args, get_origin, get_type_hints

import numpy as np
import yaml


CONFIG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(CONFIG_ROOT, "experiments", "default.yaml")

VALID_PLANNER_MODES = ("hybrid_astar_gcs", "ackermann_gcs")
VALID_CURVATURE_MODES = ("none", "hard", "direction_cone")


T = TypeVar("T")


@dataclass
class PoseConfig:
    """2D pose in YAML-friendly units."""

    x: float = 0.0
    y: float = 0.0
    heading_deg: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, float(np.deg2rad(self.heading_deg)))

    def validate(self, path: str) -> None:
        if not -180.0 <= self.heading_deg <= 180.0:
            raise ValueError(f"{path}.heading_deg must be in [-180, 180], got {self.heading_deg}")


@dataclass
class ScenarioSelectionConfig:
    name: str = "basic"


@dataclass
class ScenarioConfig:
    map_size: int = 200
    start: PoseConfig = field(default_factory=PoseConfig)
    goal: PoseConfig = field(default_factory=PoseConfig)
    corridor_width: float = 5.0

    def to_runtime_dict(self) -> Dict[str, Any]:
        return {
            "map_size": self.map_size,
            "start": self.start.as_tuple(),
            "goal": self.goal.as_tuple(),
            "corridor_width": self.corridor_width,
        }

    def validate(self, path: str) -> None:
        if self.map_size <= 0:
            raise ValueError(f"{path}.map_size must be positive, got {self.map_size}")
        if self.corridor_width <= 0:
            raise ValueError(f"{path}.corridor_width must be positive, got {self.corridor_width}")
        self.start.validate(f"{path}.start")
        self.goal.validate(f"{path}.goal")


@dataclass
class AStarConfig:
    min_radius: float = 1.5
    resolution: float = 0.5
    theta_resolution: int = 16
    max_iterations: int = 100000
    goal_tolerance: float = 0.5
    theta_tolerance_deg: float = 22.5
    heuristic_weight: float = 1.2
    adaptive_jump: bool = True
    collision_samples: int = 3
    high_precision_mode: bool = True
    path_interpolation: bool = True
    verbose: bool = True

    def validate(self) -> None:
        if self.min_radius <= 0:
            raise ValueError("astar.min_radius must be positive")
        if self.resolution <= 0:
            raise ValueError("astar.resolution must be positive")
        if self.theta_resolution <= 0:
            raise ValueError("astar.theta_resolution must be positive")
        if self.max_iterations <= 0:
            raise ValueError("astar.max_iterations must be positive")
        if self.goal_tolerance <= 0:
            raise ValueError("astar.goal_tolerance must be positive")
        if self.collision_samples <= 0:
            raise ValueError("astar.collision_samples must be positive")


@dataclass
class CorridorConfig:
    width: Optional[float] = None
    smooth_path: bool = False
    smooth_window: int = 3
    boundary_margin: float = 0.5

    def width_for(self, scenario: ScenarioConfig) -> float:
        return self.width if self.width is not None else scenario.corridor_width

    def validate(self) -> None:
        if self.width is not None and self.width <= 0:
            raise ValueError("corridor.width must be positive when provided")
        if self.smooth_window <= 0:
            raise ValueError("corridor.smooth_window must be positive")
        if self.boundary_margin < 0:
            raise ValueError("corridor.boundary_margin must be non-negative")


@dataclass
class IrisConfig:
    use_iris: bool = True
    mode: str = "np"
    config_preset: str = "high_safety"
    iteration_limit: int = 100
    termination_threshold: float = 0.01
    configuration_space_margin: float = 0.2
    min_seed_distance: float = 1.0
    max_seed_points: int = 50
    merge_overlapping: bool = False
    num_collision_infeasible_samples: int = 50
    requires_sample_as_member: bool = True

    def validate(self) -> None:
        if self.mode not in ("np", "zo"):
            raise ValueError(f"iris.mode must be 'np' or 'zo', got {self.mode!r}")
        if self.iteration_limit <= 0:
            raise ValueError("iris.iteration_limit must be positive")
        if self.termination_threshold <= 0:
            raise ValueError("iris.termination_threshold must be positive")
        if self.configuration_space_margin < 0:
            raise ValueError("iris.configuration_space_margin must be non-negative")
        if self.min_seed_distance <= 0:
            raise ValueError("iris.min_seed_distance must be positive")
        if self.max_seed_points <= 0:
            raise ValueError("iris.max_seed_points must be positive")


@dataclass
class CostWeightsConfig:
    time: float = 3.0
    path_length: float = 1.5
    energy: float = 3.0
    time_derivative_reg: float = 3.0
    regularization_r: float = 5.0
    regularization_h: float = 2.0
    h_ref: float = 0.08
    curvature_squared: float = 0.0
    curvature_derivative: float = 0.0
    curvature_peak: float = 0.0

    def to_runtime_dict(self) -> Dict[str, float]:
        data = asdict(self)
        return {key: value for key, value in data.items() if value is not None}

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if value is not None and value < 0:
                raise ValueError(f"gcs.cost_weights.{name} must be non-negative")


@dataclass
class RoundingConfig:
    flow_tol: float = 1e-5
    max_paths: int = 5
    max_trials: int = 100


@dataclass
class SolverConfig:
    max_time: float = 60.0
    mip_max_time: float = 30.0
    num_threads: int = 8


@dataclass
class GCSConfig:
    strategy_preset: str = "standard"
    cost_preset: str = "lunar_standard"
    order: int = 4
    continuity: int = 2
    zero_velocity_at_boundaries: bool = True
    min_time_derivative: float = 1.0
    curvature_constraint_mode: str = "none"
    enable_optimization: bool = True
    cost_weights: CostWeightsConfig = field(default_factory=CostWeightsConfig)
    rounding: RoundingConfig = field(default_factory=RoundingConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    def validate(self) -> None:
        if self.order < 1:
            raise ValueError("gcs.order must be >= 1")
        if not 0 <= self.continuity < self.order:
            raise ValueError("gcs.continuity must be in [0, order)")
        if self.min_time_derivative <= 0:
            raise ValueError("gcs.min_time_derivative must be positive")
        if self.curvature_constraint_mode not in VALID_CURVATURE_MODES:
            raise ValueError(
                f"gcs.curvature_constraint_mode must be one of {VALID_CURVATURE_MODES}, "
                f"got {self.curvature_constraint_mode!r}"
            )
        self.cost_weights.validate()


@dataclass
class VehicleConfig:
    wheelbase: float = 2.5
    max_steering_angle_deg: float = 85.0
    max_velocity: float = 10.0
    max_acceleration: float = 8.0

    def validate(self) -> None:
        if self.wheelbase <= 0:
            raise ValueError("ackermann.vehicle.wheelbase must be positive")
        if not 0 < self.max_steering_angle_deg < 90:
            raise ValueError("ackermann.vehicle.max_steering_angle_deg must be in (0, 90)")
        if self.max_velocity <= 0:
            raise ValueError("ackermann.vehicle.max_velocity must be positive")
        if self.max_acceleration <= 0:
            raise ValueError("ackermann.vehicle.max_acceleration must be positive")


@dataclass
class BezierConfigData:
    order: int = 5
    continuity: int = 1
    hdot_min: float = 0.01
    full_dim_overlap: bool = False
    hyperellipsoid_num_samples_per_dim_factor: int = 32
    max_rounding_attempts: int = 3
    max_rounded_paths: int = 5


@dataclass
class TrajectoryConstraintConfig:
    min_velocity: float = 3.0
    curvature_constraint_mode: str = "direction_cone"
    h_bar_prime: Optional[float] = None
    h_bar_prime_safety_factor: float = 0.7
    max_h_bar_prime_iterations: int = 3
    h_bar_prime_convergence_threshold: float = 0.15
    h_bar_prime_relax_factor: float = 1.3
    max_h_bar_prime_relax_attempts: int = 3
    h_bar_prime_safety_factor_decay: float = 0.8

    def validate(self) -> None:
        if self.min_velocity < 0:
            raise ValueError("ackermann.constraints.min_velocity must be non-negative")
        if self.curvature_constraint_mode not in VALID_CURVATURE_MODES:
            raise ValueError(
                "ackermann.constraints.curvature_constraint_mode must be one of "
                f"{VALID_CURVATURE_MODES}, got {self.curvature_constraint_mode!r}"
            )


@dataclass
class AckermannConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    bezier: BezierConfigData = field(default_factory=BezierConfigData)
    constraints: TrajectoryConstraintConfig = field(default_factory=TrajectoryConstraintConfig)
    direction_cone_profile: str = "selective"
    direction_cone_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    verbose: bool = True

    def validate(self) -> None:
        self.vehicle.validate()
        self.constraints.validate()
        if self.constraints.curvature_constraint_mode == "direction_cone":
            if self.direction_cone_profile not in self.direction_cone_profiles:
                raise ValueError(
                    "ackermann.direction_cone_profile must refer to an entry in "
                    f"ackermann.direction_cone_profiles, got {self.direction_cone_profile!r}"
                )


@dataclass
class VisualizationConfigData:
    enabled: bool = True
    save: bool = True
    output_dir: str = "./output"
    num_samples: int = 200
    show_2d: bool = True
    show_control_points: bool = True
    show_iris_regions: bool = True
    show_obstacles: bool = True
    show_corridor: bool = True
    show_astar_path: bool = True
    show_3d_trajectory: bool = True
    show_theta_profile: bool = True
    elev: float = 25.0
    azim: float = 45.0
    figsize: Tuple[float, float] = (20.0, 14.0)
    dpi: int = 150
    auto_save: bool = True
    control_point_size: int = 60
    control_point_color: str = "orange"
    control_point_marker: str = "D"

    def validate(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("visualization.num_samples must be positive")
        if self.dpi <= 0:
            raise ValueError("visualization.dpi must be positive")
        if self.control_point_size <= 0:
            raise ValueError("visualization.control_point_size must be positive")


@dataclass
class BatchConfig:
    scenario: str = "basic"
    num_runs: int = 10
    quiet_each_run: bool = True

    def validate(self) -> None:
        if self.num_runs <= 0:
            raise ValueError("batch.num_runs must be positive")


@dataclass
class ProjectConfig:
    planner_mode: str = "hybrid_astar_gcs"
    scenario: ScenarioSelectionConfig = field(default_factory=ScenarioSelectionConfig)
    scenarios: Dict[str, ScenarioConfig] = field(default_factory=dict)
    astar: AStarConfig = field(default_factory=AStarConfig)
    corridor: CorridorConfig = field(default_factory=CorridorConfig)
    iris: IrisConfig = field(default_factory=IrisConfig)
    gcs: GCSConfig = field(default_factory=GCSConfig)
    ackermann: AckermannConfig = field(default_factory=AckermannConfig)
    visualization: VisualizationConfigData = field(default_factory=VisualizationConfigData)
    batch: BatchConfig = field(default_factory=BatchConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectConfig":
        config = _coerce_dataclass(cls, data, "config")
        config.validate()
        return config

    def validate(self) -> None:
        if self.planner_mode not in VALID_PLANNER_MODES:
            raise ValueError(
                f"planner_mode must be one of {VALID_PLANNER_MODES}, got {self.planner_mode!r}"
            )
        if not self.scenarios:
            raise ValueError("scenarios must contain at least one scenario")
        for name, scenario in self.scenarios.items():
            scenario.validate(f"scenarios.{name}")
        if self.scenario.name not in self.scenarios:
            raise ValueError(
                f"scenario.name {self.scenario.name!r} is not defined. "
                f"Available scenarios: {', '.join(sorted(self.scenarios))}"
            )
        if self.batch.scenario not in self.scenarios:
            raise ValueError(
                f"batch.scenario {self.batch.scenario!r} is not defined. "
                f"Available scenarios: {', '.join(sorted(self.scenarios))}"
            )
        self.astar.validate()
        self.corridor.validate()
        self.iris.validate()
        self.gcs.validate()
        self.ackermann.validate()
        self.visualization.validate()
        self.batch.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def selected_scenario(self, name: Optional[str] = None) -> ScenarioConfig:
        scenario_name = name or self.scenario.name
        try:
            return self.scenarios[scenario_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown scenario {scenario_name!r}. "
                f"Available scenarios: {', '.join(sorted(self.scenarios))}"
            ) from exc

    def scenario_dict(self, name: Optional[str] = None) -> Dict[str, Any]:
        return self.selected_scenario(name).to_runtime_dict()

    def direction_cone_overrides(self) -> Dict[str, Any]:
        if self.ackermann.constraints.curvature_constraint_mode != "direction_cone":
            return {}
        profile = self.ackermann.direction_cone_profiles.get(self.ackermann.direction_cone_profile)
        if profile is None:
            raise ValueError(f"Unknown direction cone profile {self.ackermann.direction_cone_profile!r}")
        return dict(profile)

    def vehicle_params(self):
        from ackermann_gcs_pkg.ackermann_data_structures import VehicleParams

        return VehicleParams(
            wheelbase=self.ackermann.vehicle.wheelbase,
            max_steering_angle=float(np.deg2rad(self.ackermann.vehicle.max_steering_angle_deg)),
            max_velocity=self.ackermann.vehicle.max_velocity,
            max_acceleration=self.ackermann.vehicle.max_acceleration,
        )

    def bezier_config(self):
        from ackermann_gcs_pkg.ackermann_data_structures import BezierConfig

        return BezierConfig(**asdict(self.ackermann.bezier))

    def trajectory_constraints(self, workspace_regions: Optional[List[Any]] = None):
        from ackermann_gcs_pkg.ackermann_data_structures import TrajectoryConstraints

        vehicle_params = self.vehicle_params()
        constraint_data = asdict(self.ackermann.constraints)
        curvature_mode = constraint_data.pop("curvature_constraint_mode")
        constraint_data.update(self.direction_cone_overrides())
        return TrajectoryConstraints(
            max_velocity=vehicle_params.max_velocity,
            max_acceleration=vehicle_params.max_acceleration,
            max_curvature=vehicle_params.max_curvature,
            workspace_regions=workspace_regions,
            enable_curvature_hard_constraint=(curvature_mode == "hard"),
            curvature_constraint_mode=curvature_mode,
            **constraint_data,
        )

    def cost_weights(self) -> Dict[str, float]:
        return self.gcs.cost_weights.to_runtime_dict()

    def planner_config(
        self,
        scenario_name: Optional[str] = None,
        *,
        enable_visualization: Optional[bool] = None,
        save_visualization: Optional[bool] = None,
        enable_gcs_optimization: Optional[bool] = None,
    ):
        from config.planner import PlannerConfig

        scenario = self.selected_scenario(scenario_name)
        corridor_width = self.corridor.width_for(scenario)
        planner_config = PlannerConfig(
            corridor_width=corridor_width,
            smooth_path=self.corridor.smooth_path,
            smooth_window=self.corridor.smooth_window,
            boundary_margin=self.corridor.boundary_margin,
            use_iris=self.iris.use_iris,
            iris_mode=self.iris.mode,
            iris_iteration_limit=self.iris.iteration_limit,
            iris_termination_threshold=self.iris.termination_threshold,
            iris_configuration_space_margin=self.iris.configuration_space_margin,
            iris_min_seed_distance=self.iris.min_seed_distance,
            iris_max_seed_points=self.iris.max_seed_points,
            iris_merge_overlapping=self.iris.merge_overlapping,
            iris_num_collision_infeasible_samples=self.iris.num_collision_infeasible_samples,
            iris_requires_sample_as_member=self.iris.requires_sample_as_member,
            enable_gcs_optimization=(
                self.gcs.enable_optimization
                if enable_gcs_optimization is None
                else enable_gcs_optimization
            ),
            gcs_order=self.gcs.order,
            gcs_continuity=self.gcs.continuity,
            gcs_time_weight=self.gcs.cost_weights.time,
            gcs_path_length_weight=self.gcs.cost_weights.path_length,
            gcs_energy_weight=self.gcs.cost_weights.energy,
            gcs_strategy_preset=self.gcs.strategy_preset,
            gcs_cost_preset=self.gcs.cost_preset,
            gcs_custom_time_weight=self.gcs.cost_weights.time,
            gcs_custom_path_weight=self.gcs.cost_weights.path_length,
            gcs_custom_energy_weight=self.gcs.cost_weights.energy,
            gcs_custom_regularization_weight=self.gcs.cost_weights.regularization_r,
            gcs_zero_velocity_at_boundaries=self.gcs.zero_velocity_at_boundaries,
            gcs_min_time_derivative=self.gcs.min_time_derivative,
            gcs_curvature_constraint_mode=self.gcs.curvature_constraint_mode,
            ackermann_wheelbase=self.ackermann.vehicle.wheelbase,
            ackermann_v_min=self.ackermann.constraints.min_velocity,
            ackermann_v_max=self.ackermann.vehicle.max_velocity,
            ackermann_delta_max=float(np.deg2rad(self.ackermann.vehicle.max_steering_angle_deg)),
            ackermann_max_acceleration=self.ackermann.vehicle.max_acceleration,
            curvature_squared_weight=self.gcs.cost_weights.curvature_squared,
            curvature_derivative_weight=self.gcs.cost_weights.curvature_derivative,
            curvature_peak_weight=self.gcs.cost_weights.curvature_peak,
            enable_visualization=(
                self.visualization.enabled
                if enable_visualization is None
                else enable_visualization
            ),
            save_visualization=(
                self.visualization.save
                if save_visualization is None
                else save_visualization
            ),
            output_dir=self.visualization.output_dir,
        )
        # PlannerConfig expands GCS strategy presets in __post_init__, and those
        # presets include a generic corridor width. Project YAML keeps corridor
        # width scenario-specific unless corridor.width explicitly overrides it.
        planner_config.corridor_width = corridor_width
        return planner_config

    def astar_planner_config(self):
        from A_pkg.A_star_fast_optimized import PlannerConfig as AStarPlannerConfig

        return AStarPlannerConfig(
            max_iterations=self.astar.max_iterations,
            goal_tolerance=self.astar.goal_tolerance,
            theta_tolerance=float(np.deg2rad(self.astar.theta_tolerance_deg)),
            heuristic_weight=self.astar.heuristic_weight,
            adaptive_jump=self.astar.adaptive_jump,
            collision_samples=self.astar.collision_samples,
            high_precision_mode=self.astar.high_precision_mode,
            path_interpolation=self.astar.path_interpolation,
            verbose=self.astar.verbose,
        )

    def visualization_config(self):
        from config.visualization import VisualizationConfig

        return VisualizationConfig(
            num_samples=self.visualization.num_samples,
            show_iris_regions=self.visualization.show_iris_regions,
            show_obstacles=self.visualization.show_obstacles,
            show_corridor=self.visualization.show_corridor,
            show_astar_path=self.visualization.show_astar_path,
            show_3d_trajectory=self.visualization.show_3d_trajectory,
            show_theta_profile=self.visualization.show_theta_profile,
            elev=self.visualization.elev,
            azim=self.visualization.azim,
            figsize=tuple(self.visualization.figsize),
            dpi=self.visualization.dpi,
        )


def load_project_config(
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    *,
    export_resolved_path: Optional[str] = None,
) -> ProjectConfig:
    """Load project configuration from YAML files and CLI-style overrides."""

    resolved_path = _resolve_config_path(config_path or DEFAULT_CONFIG_PATH)
    data = _load_yaml_with_extends(resolved_path)
    if overrides:
        override_data = _overrides_to_dict(overrides)
        data = _deep_merge(data, override_data)
    config = ProjectConfig.from_dict(data)
    if export_resolved_path:
        dump_resolved_config(config, export_resolved_path)
    return config


def dump_resolved_config(config: ProjectConfig, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(config.to_dict(), stream, allow_unicode=True, sort_keys=False)


def parse_override(override: str) -> Tuple[List[str], Any]:
    if "=" not in override:
        raise ValueError(f"Override must use key=value syntax, got {override!r}")
    key, raw_value = override.split("=", 1)
    if not key:
        raise ValueError("Override key cannot be empty")
    return key.split("."), yaml.safe_load(raw_value)


def _resolve_config_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(CONFIG_ROOT, path))


def _load_yaml_with_extends(path: str, seen: Optional[set] = None) -> Dict[str, Any]:
    seen = seen or set()
    path = os.path.abspath(path)
    if path in seen:
        raise ValueError(f"Recursive config extends detected at {path}")
    seen.add(path)

    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")

    parent_refs = data.pop("extends", [])
    if isinstance(parent_refs, str):
        parent_refs = [parent_refs]

    merged: Dict[str, Any] = {}
    for parent_ref in parent_refs:
        parent_path = parent_ref
        if not os.path.isabs(parent_path):
            parent_path = os.path.join(os.path.dirname(path), parent_ref)
        merged = _deep_merge(merged, _load_yaml_with_extends(parent_path, seen))
    return _deep_merge(merged, data)


def _overrides_to_dict(overrides: List[str]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for override in overrides:
        path, value = parse_override(override)
        cursor = data
        for part in path[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Override path conflicts with scalar value: {override!r}")
        cursor[path[-1]] = value
    return data


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _coerce_dataclass(cls: Type[T], data: Mapping[str, Any], path: str) -> T:
    if not isinstance(data, Mapping):
        raise ValueError(f"{path} must be a mapping")
    field_names = {item.name for item in fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"Unknown config keys at {path}: {', '.join(sorted(unknown))}")

    hints = get_type_hints(cls)
    values = {}
    for item in fields(cls):
        if item.name not in data:
            continue
        values[item.name] = _coerce_value(
            hints.get(item.name, item.type),
            data[item.name],
            f"{path}.{item.name}",
        )
    return cls(**values)


def _coerce_value(expected_type: Any, value: Any, path: str) -> Any:
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if origin is Optional:
        expected_type = args[0]
        origin = get_origin(expected_type)

    if origin is list or origin is List:
        item_type = args[0] if args else Any
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a list")
        return [_coerce_value(item_type, item, f"{path}[]") for item in value]

    if origin is tuple or origin is Tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path} must be a tuple/list")
        return tuple(value)

    if origin is dict or origin is Dict:
        key_type = args[0] if args else Any
        value_type = args[1] if len(args) > 1 else Any
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be a mapping")
        converted = {}
        for item_key, item_value in value.items():
            key = item_key if key_type is Any else key_type(item_key)
            converted[key] = _coerce_value(value_type, item_value, f"{path}.{item_key}")
        return converted

    if origin is not None and type(None) in args:
        non_none = [arg for arg in args if arg is not type(None)][0]
        if value is None:
            return None
        return _coerce_value(non_none, value, path)

    if expected_type is Any:
        return value

    if isinstance(expected_type, type) and is_dataclass(expected_type):
        return _coerce_dataclass(expected_type, value, path)

    if expected_type is float and isinstance(value, int):
        return float(value)

    if expected_type in (str, int, float, bool):
        if not isinstance(value, expected_type):
            raise ValueError(
                f"{path} must be {expected_type.__name__}, got {type(value).__name__}"
            )
    return value
