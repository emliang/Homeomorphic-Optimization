# Solver Notes

This note records the active solver-facing configuration surface. It is meant
to answer "which solver names are current" before reading older experiment
artifacts or implementation history.

## Runtime / Test Environment

Use the `ML` conda environment for this project. Do not validate solver or
experiment changes with the system/default Python, because it can load older
Torch/CVXPY dependencies and produce irrelevant compatibility failures.

Canonical commands:

```bash
conda run --no-capture-output -n ML python -m pytest -q tests
conda run --no-capture-output -n ML python scripts/hom_alm/run_convex_eq_compare.py
```

Targeted checks should use the same prefix, for example:

```bash
conda run --no-capture-output -n ML python -m pytest -q tests/test_optimizer_loops.py
```

## Timing Records

Experiment records use `iter_time` as the backward-compatible per top-level
iteration time. ALM-type solvers additionally expose:

- `outer_iter_time`: per outer iteration wall time.
- `inner_iter_time`: iterative inner-solver time for `ALM` and `Hom-ALM`.
- `solver_iter_time`: CVXPY subproblem time for `Penalty-EQ`,
  `Prox-Penalty-EQ`, `ALM-EQ`, and `Prox-ALM-EQ`.

Convex Hom comparisons also attach `initial_ip_time` to Hom solvers and
`last_trans_time` to Hom solver records. `total_wall_time` is measured around
each complete `optimizer.optimize(...)` call and is the preferred total runtime
for comparison tables.

The same timing surface is propagated to other experiment tracks when the data
is available:

- JCC iterative variants expose `total_wall_time`, `outer_iter_time`, and
  `inner_iter_time` through their comparison rows.
- JCC / parametric exact solver baselines attach solver `runtime_total` as
  `total_wall_time`.
- INN-PGD / QCQP parametric runs expose `total_wall_time`, `total_iter_time`,
  and `avg_outer_iter_time` from their optimizer iteration timings.

## Solution Metric Records

The shared comparison surface is normalized across experiment tracks:

- `objective`: single-row objective used for best-method selection.
- `feasible`: single-row feasibility flag.
- `violation`: single-row violation metric.
- `objective_mean`: aggregate objective, equal to `objective` for a single
  instance.
- `feasibility_rate`: aggregate feasibility rate, `1.0` or `0.0` for a single
  instance.
- `violation_mean`: aggregate violation, equal to `violation` for a single
  instance.

Convex fixed-problem summaries additionally report the full residual split:

- `final_full_violation`
- `final_equality_violation`
- `final_inequality_violation`

For these convex fixed-problem summaries, `final_objective` and
`final_violation` are recomputed at `x_solved` before reporting, so the
summary reflects the actual returned solution rather than only the optimizer's
internal trajectory scalar. The internal values are preserved as
`optimizer_reported_final_objective` and `optimizer_reported_final_violation`.

Domain-specific raw payloads can still expose names such as `x_opt`,
`x_optimal`, `chance_feasibility_rate`, `raw_*`, or `refined_*`. Those are not
the cross-experiment comparison contract; they are retained for domain-level
diagnostics and plotting.

## 2026-04-24 Coefficient Convention

Quadratic penalty and proximal terms now use the same coefficient convention
across active solvers:

```text
0.5 * penalty_coef  * ||residual||^2
0.5 * proximal_coef * ||state - center||^2
```

As a result:

- explicit gradients do not multiply these terms by `2`
- outer cached Hessian/linear terms also use the undoubled coefficients
- `gradient_penalty_x(...)` is interpreted as the gradient of
  `0.5 * ||violation||^2`

## 2026-04-24 Convex Instance Controls

Active convex single-instance benchmarks now expose two small generator knobs:

- `margin_scale`
- `anchor_interior_ratio`

These are available on the benchmark/config surface, not only through raw
`problem_config`, so instance families can be swept more systematically.
The convex instance objective selector is explicit: use `obj="linear"`,
`obj="quad"`, or `obj="low_rank_quad"`.

## 2026-04-27 GaugeMap Explicit Gradient Rules

`GaugeMap.forward(..., method="explicit")` supports three active-boundary
gradient reconstruction rules:

- `explicit_gradient_rule="polynomial"`: differentiates the selected
  candidate root equation directly. This is the default because it is fastest
  in the current Hom-ALM batch-1 CPU path.
- `explicit_gradient_rule="implicit"`: reconstructs the boundary-distance
  derivative from the active constraint normal using the implicit function
  theorem. It is kept as a validation and research option.
- `explicit_gradient_rule="hybrid"`: uses the cheap polynomial closed form for
  linear and box constraints, and the implicit active-normal formula for SOC
  and quadratic constraints. This is the simplest shared route for extending
  new nonlinear convex constraint families.

For a ray `x(t) = x0 + t u` and active boundary equation `f(x(t)) = 0`, the
implicit rule uses

```text
dt/du = -t * grad f(x_boundary) / <grad f(x_boundary), u>
```

The implementation stores `beta = 1 / t`, so the explicit backward receives
`d beta / d u = beta * grad f(x_boundary) / <grad f(x_boundary), u>`.

The benchmark/config surface exposes this as
`common_config.hom_map_explicit_gradient_rule`. Smooth explicit GaugeMap only
softens the tied active boundary candidates, i.e., candidates within
`hom_map_smooth_tie_tol` of the hard maximum gauge value:

```json
{
  "smooth": true,
  "hom_map_smooth_tie_tol": 1e-7,
  "hom_map_smooth_temperature": 1e-4
}
```

This keeps the nonsmooth winner-only path for `smooth=false`, while `smooth=true`
uses only the tied maximum boundary-distance candidates and aggregates their
candidate gradients with softmax weights. It is intended for corner/intersection
regions where several constraints are nearly active, without paying the cost of
full-candidate smooth gradients.

Use
`tests/dev/profile_hom_gauge_gradient_rules.py` to compare autograd,
implicit reconstruction, and the polynomial rule on synthetic convex sets.

## 2026-04-22 Linearized Proximal Cleanup

`linearized_prox_gd` is not an active solver name.

Audit result:

- Active optimizer code accepts only `inner_solver="gd"` or
  `inner_solver="prox_gd"` for `ALM` and `Hom-ALM`.
- Test-only experiment configs under `tests/fixtures/configs/` no longer use
  `linearized_prox_gd`.
- The only intentional live mentions are documentation explaining the removal
  and one regression test that verifies the old name is rejected.
- Historical run artifacts under `results/` may still contain old serialized
  configs. They are not active inputs and should not be edited in place.

Reason:

When the nonsmooth term is the indicator of the convex inequality feasible
set, the linearized proximal step is projected-gradient descent:

```text
prox_{alpha I_C}(x - alpha grad f(x)) = P_C(x - alpha grad f(x))
```

Keeping `linearized_prox_gd` as an ALM inner-solver name duplicated the PGD
concept. Current experiments should use `PGD` / `Hom-PGD` for first-order
projected-gradient comparisons, or the CVXPY `*-EQ` baselines when a
fixed-outer ALM subproblem should keep convex inequalities hard and be solved
directly.

## Active Algorithm Solvers

| Algorithm id | Optimizer / route | Main variable | Inequality handling |
|---|---|---|---|
| `PGD` | `PGDOptimizer` | x | Euclidean projection onto convex feasible set |
| `Hom-PGD` | `HomPGDOptimizer` | z | latent ball projection, then map z to x |
| `FW` | `FrankWolfeOptimizer` | x | convex linear oracle |
| `RD` | `RadialDualOptimizer` | radial-dual variable | generalized gauge radial-dual update through `hom_map.gauge` |
| `ALM` | `LagrangianOptimizer` | x | iterative penalty/dual gradient inner loop |
| `Hom-ALM` | `HomALMOptimizer` | z | homeomorphism handles inequalities; ALM handles equalities |
| `Penalty-EQ` | `EqualityConstrainedALMOptimizer` | x | CVXPY subproblem keeps convex inequalities hard; no dual |
| `Prox-Penalty-EQ` | `EqualityConstrainedALMOptimizer` | x | CVXPY hard inequalities plus outer proximal term; no dual |
| `ALM-EQ` | `EqualityConstrainedALMOptimizer` | x | CVXPY hard inequalities plus equality dual/penalty |
| `Prox-ALM-EQ` | `EqualityConstrainedALMOptimizer` | x | CVXPY hard inequalities plus equality dual/penalty/prox |

## ALM Inner Solvers

Shared ALM-family policy now lives in common optimizer helpers in
`src/homopt/optim/_core_impl.py`:

- `_init_penalty_family_controls(...)`
- `_maybe_update_penalty(...)`
- `_clamp_dual_state(...)`

These helpers are used by:

- `LagrangianOptimizer`
- `HomALMOptimizer`
- `EqualityConstrainedALMOptimizer`

The shared policy covers:

- `use_lagrangian / use_penalty / use_proximal`
- `dual_learning_rate`
- `penalty_coef / penalty_growth / max_penalty`
- `proximal_coef`
- `max_dual`
- `return_best_violation`

Algorithm-specific code still owns:

- x-space vs z-space state and inner updates
- homeomorphism / latent-ball logic
- CVXPY exact subproblem solves for `*-EQ`

`ALM` and `Hom-ALM` support:

```text
inner_solver = "gd"
inner_solver = "prox_gd"
```

Learning-rate controls are split by loop:

- `common_config.learning_rate` is the shared single-loop primal step size and
  also seeds the ALM outer-loop primal step size.
- `inner_learning_rate` controls the inner gradient step size. If it is
  omitted, the inner loop uses the current outer-carried `learning_rate`.
- `common_config.lr_decay` seeds the ALM outer-loop adaptive decay.
- `inner_lr_decay` controls only adaptive inner-loop decay.
- `common_config.stepsize_rule` seeds the ALM outer-loop step-size rule.
- `outer_common` must not set primal step controls; it owns outer-loop
  thresholds, dual, penalty, and proximal controls.
- Direct `ALM` / `Hom-ALM` entries in `algorithm_config` may use
  `learning_rate`, `lr_decay`, and `stepsize_rule` for per-method overrides;
  the builder maps those names to optimizer-internal outer-loop fields.
- `inner_solver_common` must use `inner_learning_rate`, `inner_lr_decay`, and
  `inner_stepsize_rule`.
- Generated iterative `ALM` and `Hom-ALM` runtime configs carry
  `outer_stepsize_rule` and `outer_lr_decay` because those are optimizer
  internals, not script-level controls.

The normalizers reject script-level `outer_*` primal step controls so scripts do
not grow a second naming scheme for single-loop defaults.

For `Penalty-EQ`, `Prox-Penalty-EQ`, `ALM-EQ`, and `Prox-ALM-EQ`, iterative
inner-loop fields are filtered out of ALM-family common overrides because
those methods solve each convex subproblem through CVXPY.

Implementation ownership:

- `merge_alm_outer_common(...)` applies shared outer-loop controls to all
  ALM-family algorithms.
- `merge_alm_inner_solver_common(...)` applies inner-solver controls only to
  iterative `ALM` and `Hom-ALM`.
- `normalize_alm_algorithm_config(...)` handles direct
  `algorithm_config` overrides by algorithm type.

New benchmark code should use these helpers instead of hand-merging
`outer_common`, `inner_solver_common`, or ALM-family `algorithm_config`.

`gd`:

- one fixed-outer gradient loop per ALM outer iteration
- dual variable, penalty coefficient, and optional outer proximal center stay
  fixed inside the inner loop

`prox_gd`:

- staged legacy proximal-refresh mode
- split `inner_iterations` into `inner_proximal_update_iterations` stages
- freeze an inner proximal center during each stage
- add the inner proximal term through its gradient
- refresh only the inner proximal center between stages

Compatibility:

- `prox_gd` requires
  `inner_proximal_update_iterations <= inner_iterations`.
- `acceleration_method="nag"` requires `opt="gd"`.
- `normalized_gd` is an update backend, not an acceleration method.
- `inner_restart=false` is the default: inner solver state such as NAG
  velocity and optimizer momentum is preserved across ALM outer iterations and
  across `prox_gd` proximal-refresh stages. Set `inner_restart=true` to reset
  NAG velocity at those boundaries.
- `Hom-ALM` defaults proximal fields to z-space; plain `ALM` defaults to
  x-space.

Iterative `ALM` and `Hom-ALM` can also stop their inner loop adaptively:

```json
{
  "inner_stopping_rule": "adaptive",
  "inner_iterations_min": 10,
  "inner_iterations_max": 100,
  "inner_tol_factor": 0.1,
  "inner_tol_min": 1e-6
}
```

The default is `"fixed"`, which preserves the historical fixed
`inner_iterations` budget. In adaptive mode the inner loop stops after at least
`inner_iterations_min` steps when the inner error is below
`max(inner_tol_min, inner_tol_factor * current_violation)`, and never runs past
`inner_iterations_max`. `Hom-ALM` uses equality violation for this tolerance;
plain `ALM` uses the same violation family as its ALM penalty update.

## CVXPY Equality Baselines

The CVXPY `*-EQ` baselines solve one convex subproblem per outer iteration.
They intentionally do not use `inner_solver`, `inner_iterations`,
`inner_proximal_*`, `acceleration_method`, or `acceleration_space`.

`ConvexSolver.solve_type="eq_alm"`:

- keeps convex inequalities and box bounds as hard CVXPY constraints
- relaxes only linear equalities into the objective
- optionally adds equality dual, equality penalty, and outer proximal terms

This is the preferred route when the comparison needs an ALM-style outer loop
but an exact or high-accuracy convex subproblem solve.

## Hom-PGD Proximal Mode

`Hom-PGD` has a separate first-order proximal mode:

```text
use_proximal = True
proximal_update_iterations = K
proximal_coef = gamma
proximal_space = "z"  # or "x"
```

This is not an ALM inner solver. For `proximal_space="z"`, the z-space
quadratic proximal center has a closed-form update followed by latent ball
projection. For `proximal_space="x"`, the proximal term is differentiated
through the homeomorphic map.
