# PyDrake Official Documentation Notes for This Project

Last updated: 2026-05-16

Purpose: this file is a local, searchable index of the PyDrake APIs that matter to this repository's A* -> corridor -> IRIS/IrisZo -> GCS -> Ackermann trajectory pipeline. Use `rg "pydrake|HPolyhedron|GraphOfConvexSets|IrisZo|GcsTrajectoryOptimization" docs src config scripts` to find both this note and live code usage.

Important environment assumption: the runnable target is Ubuntu with Conda env `iris-py3.12` and Drake installed. The current Windows machine is useful for static edits only; full IRIS/GCS validation should happen in Ubuntu.

## Official Sources Queried

- PyDrake geometry optimization: https://drake.mit.edu/pydrake/pydrake.geometry.optimization.html
- PyDrake planning: https://drake.mit.edu/pydrake/pydrake.planning.html
- Drake planning IRIS Doxygen group: https://drake.mit.edu/doxygen_cxx/group__planning__iris.html
- PyDrake solvers: https://drake.mit.edu/pydrake/pydrake.solvers.html
- PyDrake trajectories: https://drake.mit.edu/pydrake/pydrake.trajectories.html
- PyDrake math / `BsplineBasis`: https://drake.mit.edu/pydrake/pydrake.math.html
- Using Drake from Python: https://drake.mit.edu/python_bindings.html

The online docs track the current Drake documentation; the project currently pins Drake `1.51.1` in `config/environments/iris_env.yaml`. If an API is questionable, verify in the Ubuntu env with:

```bash
python - <<'PY'
import pydrake
print(pydrake.__file__)
PY
```

## Project API Map

| Project area | Main files | PyDrake APIs to check |
| --- | --- | --- |
| IRIS / convex region conversion | `src/iris_pkg/core/*`, `scripts/hybrid_astar_gcs_planner.py` | `HPolyhedron`, `Iris`, `IrisOptions`, `IrisNp`, `IrisNp2`, `IrisZo`, `Hyperellipsoid` |
| GCS graph construction | `src/gcs_pkg/scripts/core/base.py` | `GraphOfConvexSets`, `GraphOfConvexSetsOptions`, `Point`, `HPolyhedron`, `Hyperellipsoid` |
| Bezier/B-spline GCS | `src/gcs_pkg/scripts/core/bezier.py` | `BsplineBasis`, `KnotVectorType`, `BsplineTrajectory_[Expression]`, `Binding`, `Constraint`, `Cost` |
| Ackermann constraints | `src/ackermann_gcs_pkg/*` | `LinearConstraint`, `LinearEqualityConstraint`, `Binding[Constraint]`, symbolic expressions |
| Solver configuration | `config/solver/solver_config.py` | `SolverOptions`, `CommonSolverOption`, `MosekSolver`, `GurobiSolver`, `ClpSolver`, `ScsSolver` |
| Visualization / region rendering | `src/visualization/*` | `HPolyhedron.A()`, `HPolyhedron.b()`, `VPolytope`, `BsplineTrajectory` |

## `HPolyhedron` and Convex Sets

`HPolyhedron` represents `{x | A x <= b}`. It may be unbounded, so code that assumes finite vertices or area must either build a bounded domain or handle failure. Current project usage constructs boxes with `HPolyhedron.MakeBox(lb, ub)` and converts custom IRIS regions with `HPolyhedron(A, b)`.

Useful official behaviors for this codebase:

- `HPolyhedron(A, b)` requires `A.rows() == b.size()`.
- `HPolyhedron.MakeBox(lb, ub)` is the right way to create the map/domain bounding box.
- `A()` and `b()` return copies of the half-space representation.
- `PointInSet(x, tol=1e-8)` is available from `ConvexSet`; use it instead of reimplementing `A @ x <= b` unless performance requires a vectorized path.
- `ChebyshevCenter()` solves an LP and can throw if the solver fails. It is a good interior-point heuristic, not necessarily a visual/geometric center.
- `Intersection(other, check_for_redundancy=False, tol=1e-9)` stacks inequalities and can optionally skip redundant rows.
- `CartesianPower(n)` and `CartesianProduct(other)` return `HPolyhedron`s and match this project's Bezier vertex set construction: region repeated for control points, then product with the time-scaling set.
- `HPolyhedron(VPolytope, tol)` uses qhull; high-dimensional or poorly sampled conversions can be expensive or fragile. This matters in `BezierGCS` when approximating `Hyperellipsoid`.
- `UniformSample()` can sample an `HPolyhedron`, but it needs a feasible previous sample or uses `ChebyshevCenter()`. This can be useful for future region QA.

Project rule of thumb: use `HPolyhedron` as the interchange format between IRIS/IrisZo and GCS. If an internal region type carries metadata (`seed_point`, `area`, coverage score), preserve it until the GCS boundary, then convert to Drake convex sets.

## IRIS, IrisNp, IrisNp2, and IrisZo

There are two different IRIS families relevant to this repo:

- `pydrake.geometry.optimization.Iris`, `IrisOptions`, and `IrisNp`: older geometry-optimization bindings.
- `pydrake.planning.IrisNp2`, `IrisZo`, `CommonSampledIrisOptions`, and `CollisionChecker`: planning bindings for sampled configuration-space region generation.

Official notes that affect this project:

- `Iris(obstacles, sample, domain, options)` returns an `HPolyhedron`.
- `IrisNp(plant, context, options)` returns one collision-free C-space region and requires a `MultibodyPlant` connected to `SceneGraph`; the plant context must contain the seed configuration. This does not match the project's current simple 2D grid collision checker interface.
- `IrisNp2(checker, starting_ellipsoid, domain, options)` returns an `HPolyhedron` and requires a `SceneGraphCollisionChecker`.
- `IrisZo(checker, starting_ellipsoid, domain, options)` returns an `HPolyhedron`, uses a `CollisionChecker`, and is explicitly experimental in Drake documentation.
- `IrisZoOptions` contains `sampled_iris_options`, `bisection_steps`, and `parameterization`.
- `CommonSampledIrisOptions` is shared by `IrisNp2` and `IrisZo`; relevant fields include `num_particles`, `tau`, `delta`, `epsilon`, iteration limits, `parallelism`, `configuration_space_margin`, `random_seed`, `mixing_steps`, and termination thresholds.
- `starting_ellipsoid` should usually be a small `Hyperellipsoid` around a collision-free seed. The center must be collision-free.
- `domain` must be bounded and dimension-compatible with the configuration space or parameterization.
- Official `IrisZo` can be long-running and solve many QPs; if using MOSEK/Gurobi, acquire/check licenses before batch calls.

Project implication: the current `src/iriszo` implementation is custom and not the same thing as Drake's official `IrisZo`. Do not silently swap it for `pydrake.planning.IrisZo`; doing so requires adapting `SimpleCollisionCheckerForIrisNp` / grid-map collision logic to Drake's `CollisionChecker` or `SceneGraphCollisionChecker`.

## `GraphOfConvexSets`

`GraphOfConvexSets` is marked experimental in official docs. It models a graph whose vertices are convex sets and whose edges carry convex costs/constraints on the source/target vertex variables.

Current project usage aligns with official patterns:

- `GraphOfConvexSets()` creates an empty graph.
- `AddVertex(convex_set, name)` associates a convex set with a vertex.
- `AddEdge(u, v, name)` connects vertices already in the same graph.
- `SolveShortestPath(source, target, options)` returns a `MathematicalProgramResult`.
- To start or end at a precise state, add a `Point` vertex and use that as `source` / `target`; do not rely on a general region vertex choosing the desired point.
- Edge activation is available through `edge.phi()`; current custom rounding extracts paths by checking `result.GetSolution(edge.phi())`.

Important option defaults:

- Raw `GraphOfConvexSets.SolveShortestPath` defaults to `convex_relaxation = false`, `max_rounded_paths = 0`, `preprocessing = false` when options are not provided.
- `pydrake.planning.GcsTrajectoryOptimization.SolvePath` uses different defaults: `convex_relaxation = true`, `max_rounded_paths = 5`, `preprocessing = true`.

Useful `GraphOfConvexSetsOptions` fields:

- `convex_relaxation`: solve relaxed edge activations; often tight but not guaranteed.
- `max_rounded_paths`, `max_rounding_trials`, `rounding_seed`, `flow_tolerance`: control Drake's random rounding when solving a relaxation.
- `preprocessing`: removes edges that cannot lie on the source-target path, but the removal is not exact.
- `solver`, `restriction_solver`, `preprocessing_solver`: can use different solvers for main, convex restriction, and preprocessing.
- `solver_options`, `restriction_solver_options`, `preprocessing_solver_options`: can use different tolerances/logging for each phase.

Project improvement candidate: after custom rounding finds `path_edges`, prefer evaluating `gcs.SolveConvexRestriction(path_edges, options, initial_guess=relaxation_result)` instead of fixing every edge with `AddPhiConstraint(True/False)` and re-solving `SolveShortestPath`. The official method is intended for the fixed-path convex problem and can be smaller.

## `GcsTrajectoryOptimization`

Drake also provides `pydrake.planning.GcsTrajectoryOptimization`, a higher-level trajectory API built on GCS.

Relevant official capabilities:

- Constructor: `GcsTrajectoryOptimization(num_positions, continuous_revolute_joints=[])`.
- `AddRegions(regions, edges_between_regions, order, h_min=1e-6, h_max=20, name='', edge_offsets=None)` creates a subgraph over convex regions.
- It handles intersection checks and wrap-around logic for continuous revolute joints if configured.
- `AddTimeCost`, `AddPathLengthCost`, `AddPathEnergyCost` are built in.
- `AddVelocityBounds` adds linear velocity bounds.
- `AddNonlinearDerivativeBounds` supports higher-order derivative bounds by adding true nonlinear constraints to MIP/restriction and convex surrogates to relaxation.
- `AddPathContinuityConstraints(k)` must be called for each derivative order `1..k` if true C^k path continuity is required.
- Official docs warn that path continuity constraints are on control points of `r(s)`, not directly on time-scaled `q(t)`; returned trajectories may need `NormalizeSegmentTimes()` for valid continuity interpretation.

Project implication: the custom `BezierGCS` currently duplicates much of this, but also contains Ackermann-specific constraints and project-specific rounding/reporting. Treat `GcsTrajectoryOptimization` as a reference or migration target, not a drop-in replacement.

## B-spline / Bezier Trajectories

Relevant APIs:

- `pydrake.math.BsplineBasis(order, knots)` and `BsplineBasis(order, num_basis_functions, KnotVectorType.kClampedUniform, start, end)`.
- `pydrake.trajectories.BsplineTrajectory(basis, control_points)`.
- Template forms such as `BsplineTrajectory_[Expression]` are valid and used for symbolic constraint construction.
- `control_points()`, `basis()`, `value(t)`, `vector_values(times)`, `EvalDerivative(t, derivative_order)`, and `MakeDerivative(derivative_order)` are central methods.

Official details that matter:

- `order` is degree + 1.
- For B-splines, a knot breakpoint with multiplicity `p` has continuity `C^(k-p-1)` where `k` is order. Repeated knots reduce continuity.
- `vector_values()` only works directly when rows or columns equal 1.
- `MakeDerivative()` requires non-negative derivative order.
- `EvaluateLinearInControlPoints()` can produce linear maps from control points to evaluated derivatives; this is useful for cleaner future constraints.

Project implication: keep the custom `BezierTrajectory` time inversion checks conservative. It divides by `time_traj.EvalDerivative(s, 1)`, so `hdot_min` and derivative constraints are not cosmetic; they prevent singular time scaling.

## MathematicalProgram, Constraints, and Symbolic Expressions

Current code frequently builds constraints with:

- `MakeMatrixContinuousVariable`
- `MakeVectorContinuousVariable`
- `DecomposeLinearExpressions`
- `Binding[Constraint]`, `Binding[Cost]`
- `LinearConstraint`, `LinearEqualityConstraint`, `L2NormCost`, `QuadraticCost`, `LorentzConeConstraint`

Development guidance:

- Prefer `DecomposeLinearExpressions` when turning `Expression` arrays into `LinearConstraint` / `LinearCost` coefficients.
- Bind constraints to `edge.xu()`, `edge.xv()`, or `np.append(u.x(), v.x())` exactly matching how the symbolic expression variables were ordered.
- If adding nonlinear Ackermann constraints, be explicit about which GCS transcription they should participate in: MIP, relaxation, restriction. Drake supports this concept; custom code should document any surrogate used in relaxation.

## Solver Configuration

Official solver behaviors relevant to this repo:

- `Solve(prog, initial_guess=None, solver_options=None)` chooses a solver based on availability and the program.
- `SolverOptions.SetOption(...)` does not validate solver-specific option names immediately. Unsupported or invalid options fail during solve.
- `CommonSolverOption.kPrintToConsole` applies only to solvers that support it.
- `SolverInterface.available()` means compiled into Drake; `enabled()` means runtime configuration is valid, such as license environment variables.
- MOSEK and Gurobi licenses can be acquired with `MosekSolver.AcquireLicense()` / `GurobiSolver.AcquireLicense()`. This is useful for repeated or long-running solves.
- For parallel solves with commercial solvers, do not exceed available license seats.

Project implications:

- Before full chain validation, run `python scripts/verify_environment.py` in Ubuntu.
- When configuring `GraphOfConvexSetsOptions`, set `options.solver` only after confirming the solver is both available and enabled.
- If MOSEK is unavailable, `ClpSolver` and `ScsSolver` may not support every cone/MIP/nonlinear formulation used by GCS + Ackermann constraints. Treat fallback as capability-dependent, not guaranteed.

Suggested runtime probe:

```bash
python - <<'PY'
from pydrake.solvers import MosekSolver, GurobiSolver, ClpSolver, ScsSolver
for Solver in [MosekSolver, GurobiSolver, ClpSolver, ScsSolver]:
    s = Solver()
    print(Solver.__name__, "available=", s.available(), "enabled=", s.enabled())
PY
```

## Development Checklist for PyDrake Changes

1. Confirm target Drake version in Ubuntu, not Windows.
2. Search existing code before editing: `rg "GraphOfConvexSets|HPolyhedron|Iris|MosekSolver|BsplineTrajectory"`.
3. Keep `HPolyhedron` as the GCS boundary format.
4. Verify all convex sets have matching `ambient_dimension()`.
5. For GCS source/target, use `Point` vertices for exact endpoints.
6. Decide explicitly between raw `GraphOfConvexSets` and `GcsTrajectoryOptimization`; do not mix assumptions about their defaults.
7. If using relaxation + rounding, record relaxation result, candidate paths, fixed-path/restriction result, and final constraint violations.
8. For Ackermann trajectories, validate beyond solver success: curvature, velocity, acceleration, workspace containment, and endpoint heading.
9. For solver changes, check `available()` and `enabled()` and document expected fallback limitations.
10. Run full chain only in Ubuntu `iris-py3.12`; Windows checks should be limited to import-free static analysis or pure-Python tests.
