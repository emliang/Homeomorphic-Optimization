# Algorithm Notes

## Runtime / Test Environment

HomOPT development and validation should use the `ML` conda environment.
Avoid system/default Python for tests and experiments; it may load older
dependencies and report compatibility failures that are not relevant to the
active project environment.

Use:

```bash
conda run --no-capture-output -n ML python -m pytest -q tests
conda run --no-capture-output -n ML python scripts/hom_alm/run_convex_eq_compare.py
```

## 2026-05-25: Experiment Script and Config Surface

Active experiment scripts now use one public folder scheme under `scripts/`.
Script filenames encode the problem and purpose within each experiment family,
and active configs should not depend on old `legacy` names or compatibility
aliases.

Experiment families:

- `scripts/hom_pgd/run_convex_ineq_compare.py`: Hom-PGD family, convex inequality-only SOCP.
- `scripts/hom_alm/run_convex_eq_compare.py`: Hom-ALM family, convex inequalities plus
  equalities.
- `scripts/hom_alm/run_stiefel_eq_compare.py`: Hom-ALM and Stiefel-manifold baselines,
  nonlinear equality plus convex side inequalities.
- `scripts/inn_pgd/run_qcqp_inn_compare.py`: INN-PGD family for nonconvex objectives with
  convex inequalities.
- `run_*_hom_pgd.py`: route-specific Hom-PGD experiments such as adversarial
  attack, MaxCut SDP, poly-star, and ablation tests.
- NN predictor plus post-processing scripts are the fourth experiment family
  and should be named by their predictor/post-processing role.

Current INN-PGD scripts:

- `run_qcqp_inn_compare.py`: high-dimensional QCQP comparison. It trains
  or loads an INN map, runs `INNPGDOptimizer`, and compares against IPOPT on
  the same sampled instances. It does not generate 2D diagnostic plots.
- `run_qcqp_inn_ablation.py`: sweeps INN network and optimizer
  parameters over the same QCQP INN-PGD route. It is for parameter sensitivity,
  not solver/baseline comparison.
- `run_qcqp_inn_toy.py`: 2D QCQP diagnostic comparison and
  visualization. It owns 2D trajectories, mapping plots, IPOPT comparison, and
  fixed-instance `ALM`/`Penalty`/`Prox-Penalty` baselines.
- `run_jcc_opf_compare.py`: JCC/OPF INN-PGD comparison. The previous JCC
  experiment design includes INN-PGD plus solver/ALM-style baselines
  (`mixed_integer`, `CVaR`, `scenario`, `ALM`, `Penalty`, `Prox-Penalty`). Keep this script
  in the INN-PGD family; the solver/penalty rows are baselines for that
  experiment, not a separate experiment family. `run_inn_pgd=True` enables the
  INN-PGD method; `model_config`, `train_config`, and `optimizer_config`
  control the INN map and latent PGD loop. `run_inn_pgd=False` is only for fast
  solver/baseline smoke checks. The current INN flow requires at least two
  active OPF decision variables, so 30-bus smoke runs should keep
  `run_inn_pgd=False`; use 57-bus or larger for actual JCC INN-PGD runs.

Entrypoint rule:

- Single-instance scripts use
  `main = make_simple_benchmark_entrypoint(EXPERIMENT_NAME, BENCHMARK, PARAMS)`.
- Multi-instance scripts expose `run_instance(...)`, `run_all_instances(...)`,
  and `main = run_all_instances`.
- Custom visualization scripts may call `run_script_experiment(...)` directly
  when the benchmark payload is not a standard comparison run.
- `SCRIPT_EXPERIMENTS` is the catalog of editable script entrypoints. Package
  config presets are not part of the script-only experiment workflow.

Config group rule:

- Convex/Hom-PGD/Hom-ALM scripts use `problem_config`, `common_config`,
  `outer_common`, `inner_solver_common`, and `algorithm_config`.
- Stiefel scripts use `problem_config`, `mapping`, `initialization`,
  `outer_common`, `inner_solver_common`, and `algorithm_config`.
- QCQP/INN scripts use `problem_config`, `model_config`,
  `optimizer_config`, and `train_config`. In these scripts, the top-level
  `n_samples` is the number of sampled comparison instances, while
  `train_config["n_samples"]` is the INN training sample count.
- JCC/INN scripts use `problem_config`, `solver_configs`, `algorithm_config`,
  `model_config`, `optimizer_config`, and `train_config`. In this script,
  `algorithm_config` controls `ALM`/`Penalty`/`Prox-Penalty` baselines, while
  `optimizer_config` controls INN-PGD.
- Internal training payloads use `model_config`, `ipnn_model_config`, and
  `predictor_model_config`.

Removed public config names for active experiment scripts include
`problem_para`, `inn_para`, `inn_pgd_args`, `lagrangian_args`,
`common_overrides`, `problem_overrides`, `algorithm_params`,
`problem_config_overrides`, `algorithm_config_overrides`,
`solver_config_overrides`, `cvar_solver_config`, and
`scenario_solver_config`. QCQP learning post-processing still uses singular
`solver_config` for exact projection and warm-start solver settings.
`problem_type="eq"` is not an active spelling; equality-constrained convex
examples should use `problem_type="socp_eq"`.

## 2026-05-05: Experiment Algorithm IDs and Paper Labels

Algorithm ids should encode whether the outer proximal term is part of the
method. Do not run a proximal method as `ALM` or `Hom-ALM` and rely on
`method_labels` to rename it in plots. Use the explicit proximal ids instead:


| Algorithm id               | Proximal term             | Paper/display label  |
| -------------------------- | ------------------------- | -------------------- |
| `PGD`                      | no                        | `PGD`                |
| `Hom-PGD`                  | no                        | `Hom-PGD`            |
| `FW`                       | no                        | `FW`                 |
| `RD`                       | no                        | `RD`                 |
| `Penalty`                  | no                        | `Penalty`            |
| `Prox-Penalty`             | yes                       | `PPP`                |
| `ALM`                      | no                        | `ALM`                |
| `Prox-ALM`                 | yes                       | `PALM`               |
| `Hom-ALM`                  | no                        | `Hom-ALM`            |
| `Prox-Hom-ALM`             | yes                       | `Hom-PALM`           |
| `Penalty-EQ`               | no                        | `Penalty-Eq`         |
| `Prox-Penalty-EQ`          | yes                       | `PPP-Eq`             |
| `ALM-EQ`                   | no                        | `ALM-Eq`             |
| `Prox-ALM-EQ`              | yes                       | `PALM-Eq`            |
| `StiefelRetractionPenalty` | penalty schedule, no dual | `Retraction-Penalty` |
| `StiefelRetractionALM`     | ALM side constraints      | `Retraction-ALM`     |


The canonical label dictionaries live in `homopt.experiments._labels`.
Experiment scripts should use these shared dictionaries instead of maintaining
local label maps. If a comparison needs both prox and non-prox variants in the
same figure, list both algorithm ids, for example `ALM` and `Prox-ALM`, or
`Hom-ALM` and `Prox-Hom-ALM`. This keeps records, summaries, colors, and
legends unambiguous.

Config rule:

- `Penalty`, `ALM`, and `Hom-ALM` should not set `use_proximal=True`.
- `Prox-Penalty`, `Prox-ALM`, `Prox-Hom-ALM`, `Prox-Penalty-EQ`, and `Prox-ALM-EQ` should set
`use_proximal=True` and use the relevant `proximal_coef`.
- `proximal_coef > 0` alone does not define a proximal algorithm; the effective
switch is the explicit proximal algorithm id plus `use_proximal=True`.
- Algorithms without the `-EQ` suffix put all constraints in the penalty/dual
  residual. Algorithms with the `-EQ` suffix only penalize/dualize equalities;
  inequalities remain explicit in the CVXPY subproblem.

## 2026-05-05: Stiefel Retraction Baselines With Convex Side Constraints

The active Stiefel experiment route is
`scripts/hom_alm/run_stiefel_eq_compare.py`. It compares methods for problems
of the form:

```text
minimize    -tr(X.T A X)
subject to  X.T X = I
            convex side constraints on vec(X)
```

The Stiefel equality and the convex side constraints are handled differently
by the baseline families:


| Algorithm id               | Solver class                                                 | Stiefel equality                            | Convex side inequalities                                                |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------- | ----------------------------------------------------------------------- |
| `StiefelRetractionPenalty` | `StiefelRetractionSolver` / `StiefelRetractionPenaltySolver` | QR retraction after every tangent step      | quadratic penalty on clipped violation, with outer-loop penalty growth  |
| `StiefelRetraction`        | `StiefelRetractionSolver`                                    | QR retraction                               | same penalty baseline, not a default comparison id                       |
| `StiefelRetractionALM`     | `StiefelRetractionALMSolver`                                 | QR retraction after every tangent step      | inequality augmented Lagrangian with nonnegative duals                  |
| `ALM-EQ-PGD`               | `StiefelALMEQPGDSolver`                                      | equality ALM                                | CVXPY projection keeps convex side constraints hard                     |
| `StiefelIPOPT`             | `StiefelPyomoIPOPTSolver`                                    | Pyomo nonlinear equality constraints        | Pyomo nonlinear/linear constraints; treated as the reference solver row |
| `Hom-ALM`                  | `HomALMOptimizer`                                            | equality ALM in latent coordinates          | gauge/homeomorphic map encodes convex side constraints                  |
| `Prox-Hom-ALM`             | `HomALMOptimizer`                                            | equality proximal ALM in latent coordinates | gauge/homeomorphic map encodes convex side constraints                  |


`StiefelRetractionPenalty` is intentionally a simple Riemannian quadratic-penalty
baseline. For a fixed outer penalty coefficient, its inner loop minimizes:

```text
f(X) + 0.5 * penalty_coef * ||max(g(X), 0)||^2
```

on the Stiefel manifold. The gradient of this penalized objective is projected
to the Stiefel tangent space, then the candidate point is QR-retracted back to
`X.T X = I`. After each outer loop, the penalty coefficient is updated as:

```text
penalty_coef <- min(max_penalty, penalty_coef * penalty_growth)
```

The method still does not maintain inequality dual variables, so a small
Riemannian gradient norm is only stationarity for the current penalized
objective, not a full KKT certificate for the original inequality-constrained
problem.

`StiefelRetractionALM` keeps the same manifold update but replaces the fixed
side-constraint penalty with an inequality ALM merit:

```text
f(X) + sum_i (max(lambda_i + rho * g_i(X), 0)^2 - lambda_i^2) / (2 * rho)
```

After each outer loop it updates:

```text
lambda <- clamp(lambda + dual_learning_rate * rho * g(X), min=0, max=max_dual)
rho    <- min(max_penalty, rho * penalty_growth)
```

The inner loop is still a Riemannian gradient loop: compute the Euclidean
gradient of the current ALM merit, project it to the Stiefel tangent space, take
a step, and QR-retract. The method is therefore closer to the standard
Riemannian ALM treatment of side inequalities than the pure quadratic-penalty
baseline, while remaining much lighter than the CVXPY-projected `ALM-EQ-PGD`
route.

Recommended interpretation in experiments:

- Use `StiefelRetractionPenalty` as a cheap classical Riemannian penalty
baseline.
- Use `StiefelRetractionALM` when the comparison needs side-constraint
multiplier dynamics but should still solve inner steps by first-order
retraction rather than by CVXPY.
- Prefer `StiefelRetractionPenalty`; `StiefelRetraction` should only appear
when a comparison explicitly needs the shorter solver id.
- Use `StiefelIPOPT` as the Stiefel reference solve, analogous to the
`ConvexSolver` CVXPY ground-truth row in convex single-instance comparisons.
In `run_stiefel_eq_compare.py`, listing `StiefelIPOPT` in `algorithms` requests
the reference solve and prepends a solver row to the comparison table; it is
not run as an iterative algorithm record.
- Stiefel IPOPT reference results are cacheable with `reference_cache=True`
using `artifacts/stiefel_reference_context_cache.npy`. The cache key includes
the Stiefel problem config, initialization config, seed, and `StiefelIPOPT`
solver config. The benchmark payload reports `reference_cache_hit`,
`reference_cache_path`, `reference_cache_load_time`, and
`reference_cached_solver_time`, matching the convex reference-cache reporting
style.
- Iterative Stiefel baselines share the script-level budget where possible:
`Prox-ALM`, `Prox-Hom-ALM`, `StiefelRetractionPenalty`,
`StiefelRetractionALM`, and `ALM-EQ-PGD` use the same
`outer_common.outer_iterations` and effective inner iteration count from
`inner_solver_common`. `StiefelRetractionPenalty` uses
`outer_common.penalty_coef`, `outer_common.penalty_growth`, and
`outer_common.max_penalty` for its outer penalty schedule.
- Paper-scale comparison runs should keep `lagrangian_gradient="explicit"`
  and `hom_map_gradient="explicit"`. Hidden safety fallbacks are intentionally
  avoided: x-space ALM/PALM problems must provide `gradient_lagrangian_x`,
  Hom-PGD/Hom-ALM x-space proximal terms require `hom_map.vjp`, and nested
  runtime plots require an explicit `outer_iter_time` trace.
- `GaugeMap.forward(..., method="explicit")` and `GaugeMap.vjp(...,
  method="explicit")` fail fast when the geometry does not support the explicit
  rule; they must not fall back to autograd. If a benchmark requires autograd,
  the algorithm config must set that path explicitly.
- New experiment payloads should place non-core outputs under `metrics`.
  Arbitrary dict payload fields are rejected by the runner.
- With `verbose=True`, HAL/Stiefel iterative logs use the compact columns
`outer`, `obj`, `eqvio`, `ineqvio`, `lr`, `dual_norm`, `penalty`, `lag_gap`,
and `iter_time`, with scalar values shown in scientific notation with four
significant digits.

For PCA/subspace-learning data instances, `objective_normalization="top_sum"`
scales the covariance matrix by the sum of its top `n_cols` eigenvalues. This
does not change the unconstrained PCA subspace, but it keeps the reference
Stiefel objective near `-1` across datasets such as digits, Olivetti, LFW, and
MNIST. That makes equality-ALM penalty parameters comparable across problem
families instead of tying them to raw covariance spectral scale.

The scale presets can set `restricted_rows="top_loading"` with a
`restricted_row_count`. The script then selects the rows with largest squared
loading in the unconstrained top-`n_cols` PCA subspace before building the group
norm constraint. This makes the side constraint geometrically tight: reducing
`group_budget` below that top-loading norm excludes the unconstrained PCA
solution, so the reference objective should move above `-1`.

The scale presets do not include `norm_radius=sqrt(n_cols)`. That radius is
implied by the Stiefel equality because `||X||_F^2 = trace(X.T X) = n_cols`, so
it would only add a redundant SOC constraint. The active side-constraint
pressure in these presets should come from the group-norm budget and box bounds,
not from a duplicate global norm radius.

Each scale preset carries a `scale_label`. Script entrypoints should route
run-name construction through `homopt.experiments._naming` via
`scripts/_common.py`, so output directories are consistently
instance-labeled across Stiefel, SOCP-EQ, and SOCP runs. For example,
`results/hom_alm/stiefel_eq_compare_intermediate_lfw` and
`stiefel_intermediate_lfw_objective_convergence_by_iter.pdf`. This prevents
small/intermediate/medium/large runs from overwriting each other's result files.

The Stiefel plotting layer fixes algorithm order and display labels separately
from solver identifiers. The current order is `ALM`,
`Prox-ALM`, `StiefelRetractionPenalty`, `StiefelRetractionALM`, `Hom-ALM`,
`Prox-Hom-ALM`, displayed as `ALM`, `PALM`, `Retraction-Penalty`,
`Retraction-ALM`, `Hom-ALM`, and `Hom-PALM`; colors remain keyed by the
internal solver identifier so the algorithm-color pair is stable across
figures. The Stiefel reference line is labeled `IPOPT`.

For scalar convergence metrics, the shared plotting primitive writes four
figures per metric: linear-x iteration, log-x iteration, linear-x time, and
log-x time. This applies uniformly to Stiefel, SOCP-EQ, and SOCP comparison
scripts because they all call `save_comparison_visualizations`.

Convex single-instance comparisons expose the same plotting controls:
`plot_algorithm_order`, `method_labels`, and `reference_label`. The default
reference label is `MOSEK`; paper-facing labels are centralized in
`homopt.experiments._labels`. Proximal variants are algorithm ids, not only
display labels: `Prox-ALM` displays as `PALM`, and `Prox-Hom-ALM` displays as
`Hom-PALM`; non-proximal `ALM` and `Hom-ALM` keep their own labels. Convex runs
also accept `scale_label`; `run_convex_eq_compare.py`,
`run_convex_ineq_compare.py`, and `run_stiefel_eq_compare.py` use the shared
naming helpers in
`homopt.experiments._naming`, and `benchmarks_single.py` uses the same helper
for visualization prefixes. Example SOCP-EQ artifacts are
`results/hom_alm/convex_eq_compare_socp_eq_n1000_q100_eq900` and
`socp_eq_n1000_q100_eq900_objective_convergence_by_iter.pdf`.

## 2026-04-24: Quadratic Penalty / Proximal Coefficient Convention

The active convention is:

- penalty term: `0.5 * penalty_coef * ||residual||^2`
- proximal term: `0.5 * proximal_coef * ||state - center||^2`

Consequences:

- explicit gradients use `penalty_coef * residual` and
`proximal_coef * (state - center)`
- outer-cache constructions use `+ penalty_coef * A^T A` and
`+ proximal_coef * I`, not doubled versions
- `gradient_penalty_x(...)` returns the gradient of
`0.5 * ||violation||^2`, so optimizer-side multiplication stays
`penalty_coef * gradient_penalty_x(x)`

This convention is now aligned across:

- `ConvexOptEq`
- `StiefelProblem`
- shared penalty-gradient helpers used by convex / QCQP / JCC routes

## 2026-04-24: Convex Instance Generation Knobs

The deterministic convex single-instance generator now exposes two light
instance-shape controls:

- `margin_scale`
- `anchor_interior_ratio`
- `quad_diag_lower`
- `quad_diag_upper`
- `low_rank_quad_ridge`

They are threaded through:

- `create_test_problem(...)`
- `build_convex_problem_config(...)`
- `socp_hompgd_benchmark(...)`
- `convex_algorithm_comparison(...)`

Semantics:

- `obj` must be one of `linear`, `quad`, or `low_rank_quad`; unknown objective
names now raise instead of silently falling back to a linear objective
- `margin_scale` controls the local-distance slack added when constructing
linear, quadratic, and SOC right-hand sides from the reference feasible
anchor; the residual slack is scaled by the local constraint-gradient norm so
SOC and quadratic margins are more comparable
- `anchor_interior_ratio` controls how deep the sampled feasible anchor lies
inside the box bounds before the remaining constraints are generated around it
- for `obj="quad"`, `Q` is now diagonal with entries sampled uniformly from
`[quad_diag_lower, quad_diag_upper]`, giving a controlled objective spectrum
- the previous low-rank quadratic generator is still available explicitly as
`obj="low_rank_quad"` and uses `B B^T / n_var + low_rank_quad_ridge * I`
- quadratic inequality constraints now use diagonal PSD matrices with entries
sampled from `[quad_diag_lower, quad_diag_upper]`, matching the controlled
spectrum convention used for `obj="quad"` instead of the earlier low-rank
cylinder-style random matrices

Defaults:

- `margin_scale = 0.1`
- `anchor_interior_ratio = 0.8`
- `quad_diag_lower = 1e-2`
- `quad_diag_upper = 1.0`
- `low_rank_quad_ridge = 1e-2`

This keeps the generator low-complexity while making the benchmark family more
systematic than the previous fully implicit hard-coded margins.

## 2026-04-22: ALM inner-solver policy after PGD equivalence cleanup

`linearized_prox_gd` is no longer an active `ALM` / `Hom-ALM` inner solver.
When the nonsmooth term is the indicator of the convex inequality feasible
set, the linearized proximal step is exactly a projected-gradient step. Keeping
it as a separate inner-solver name created a duplicate concept.

Current intended routes:

- Use `PGD` or `Hom-PGD` for first-order projected-gradient baselines over
convex inequalities.
- Use `Penalty-EQ`, `Prox-Penalty-EQ`, `ALM-EQ`, or `Prox-ALM-EQ` when the
fixed-outer ALM subproblem should be solved directly by CVXPY with convex
inequalities kept as hard constraints.
- Use iterative `ALM` / `Hom-ALM` with `inner_solver="gd"` for smooth
fixed-outer penalty/dual-gradient inner loops.
- Keep `inner_solver="prox_gd"` only as the staged mode that refreshes
an inner proximal center and handles that proximal term through its gradient.

For Hom methods, the PGD equivalence is in the active optimization space:
`Hom-PGD` projects in latent z-space onto the ball and maps back to x. This is
not generally the same as Euclidean projection onto the original x-space
inequality set, because the homeomorphic map is not generally an isometry.

The closed-form helper for a quadratic proximal center remains in the codebase
for `Hom-PGD` proximal mode, but it is not exposed as an `ALM` inner-solver
choice.

## 2026-04-14: ALM / Hom-ALM outer-loop parameter unification

### Scope

This note records the package-level unification of the outer-loop configuration
used by:

- `ALM` (`LagrangianOptimizer`)
- `Hom-ALM` (`HomALMOptimizer`)
- JCC lagrangian-style iterative routes

The goal is to keep one canonical parameter family for outer penalty / dual /
proximal control, instead of letting each route drift into its own defaults.

This note now also records the current naming and update-rule conventions for
the main convex optimization baselines:

- `PGD`
- `Hom-PGD`
- `FW`
- `RD`
- `ALM`
- `ALM-EQ`
- `Penalty-EQ`
- `Prox-Penalty-EQ`
- `Prox-ALM-EQ`
- `Hom-ALM`

### Canonical algorithm names and config blocks

At the experiment config level, the canonical algorithm ids are:


| Algorithm                                                      | Config block      |
| -------------------------------------------------------------- | ----------------- |
| Projected Gradient Descent                                     | `PGD`             |
| Homomorphic Projected Gradient Descent                         | `Hom-PGD`         |
| Frank-Wolfe                                                    | `FW`              |
| Radial Dual                                                    | `RD`              |
| Augmented Lagrangian Method                                    | `ALM`             |
| Equality penalty method with CVXPY convex subproblems          | `Penalty-EQ`      |
| Equality proximal-penalty method with CVXPY convex subproblems | `Prox-Penalty-EQ` |
| Equality-only ALM with CVXPY convex subproblems                | `ALM-EQ`          |
| Equality proximal-ALM with CVXPY convex subproblems            | `Prox-ALM-EQ`     |
| Homomorphic Augmented Lagrangian Method                        | `Hom-ALM`         |


These names are also the canonical keys under the runtime `params` payload
passed into `run_algorithm(...)`.

### Parameter naming rules

The current naming convention is:

1. Use full words for canonical parameter names.
2. Use nested `*_subproblem` blocks for embedded inner solver configs.
3. Use `outer_iterations` / `inner_iterations` inside each algorithm or
  subproblem block.
4. Do not emit historical short aliases from active experiment configs.

Examples:

- `projection_subproblem`
- `linearization_subproblem`
- `projection_outer_iterations`
- `projection_inner_iterations`
- `linearization_outer_iterations`
- `linearization_inner_iterations`
- `max_penalty`

### Baseline-specific parameter surfaces

`PGD`:

- outer optimization uses the first-order common fields
- embedded projection route uses `projection_subproblem`
- canonical flat counters are
`projection_outer_iterations` and `projection_inner_iterations`

`Hom-PGD`:

- uses the first-order common fields
- main extra mapping field is `hom_p_norm`
- no embedded penalty subproblem is used in the current route

`FW`:

- outer optimization uses the first-order common fields
- embedded linearization route uses `linearization_subproblem`
- canonical flat counters are
`linearization_outer_iterations` and `linearization_inner_iterations`

`RD`:

- uses the first-order common fields
- no extra nested penalty config block is currently required

`ALM` and `Hom-ALM`:

- use the full canonical penalty-method config family
- all outer penalty / dual / proximal fields should stay in the top-level
algorithm block rather than being renamed per benchmark
- the gradient inner-solver controls are shared: `inner_solver`, inner
proximal refresh fields, and acceleration fields have the same names and
defaults, with `ALM` defaulting to x-space and `Hom-ALM` defaulting to z-space

`Penalty-EQ`, `Prox-Penalty-EQ`, `ALM-EQ`, and `Prox-ALM-EQ`:

- use the equality-CVXPY outer-loop subset generated by
`build_eq_alm_method_params(...)`
- inequalities remain hard convex CVXPY constraints, while only equalities
enter the dual/penalty and optional outer proximal terms
- they do not use gradient inner-solver or acceleration controls because each
outer subproblem is solved by CVXPY

### Canonical outer-loop config

The shared owner now lives in
`src/homopt/experiments/_benchmark_common.py`:

- `default_penalty_method_config()`
- `normalize_penalty_method_config(...)`
- `build_penalty_method_params(...)`
- `build_eq_alm_method_params(...)`
- `build_hom_penalty_method_params(...)`
- `normalize_jcc_lagrangian_config(...)`

Canonical default fields:


| Field                              | Default                                            |
| ---------------------------------- | -------------------------------------------------- |
| `learning_rate`                    | `1e-3`                                             |
| `inner_learning_rate`              | optional override for ALM inner gradient step size |
| `outer_iterations`                 | `50`                                               |
| `inner_iterations`                 | `10`                                               |
| `max_running_time`                 | `300`                                              |
| `convergence_threshold`            | `1e-6`                                             |
| `stepsize_rule`                    | `"adaptive"`                                       |
| `inner_stepsize_rule`              | `"constant"`                                       |
| `lr_decay`                         | `0.999`                                            |
| `inner_lr_decay`                   | `0.999`                                            |
| `min_lr`                           | `1e-6`                                             |
| `inner_min_lr`                     | `1e-6`                                             |
| `momentum`                         | `0.0`                                              |
| `dual_learning_rate`               | `1e-1`                                             |
| `penalty_coef`                     | `10.0`                                             |
| `penalty_growth`                   | `1.1`                                              |
| `proximal_coef`                    | `0.1`                                              |
| `max_penalty`                      | `1e2`                                              |
| `max_dual`                         | `1e2`                                              |
| `use_lagrangian`                   | `True`                                             |
| `use_penalty`                      | `True`                                             |
| `use_proximal`                     | `False`                                            |
| `inner_solver`                     | `"gd"`                                             |
| `inner_proximal_update_iterations` | `1`                                                |
| `inner_proximal_coef`              | `0.0`                                              |
| `inner_proximal_space`             | `"x"`                                              |
| `acceleration_method`              | `"none"`                                           |
| `acceleration_space`               | `"x"`                                              |
| `opt`                              | `"gd"`                                             |


JCC keeps one route-specific override:

- `convergence_threshold = 1e-5`

`build_hom_penalty_method_params(...)` intentionally overrides the generic
penalty-method coordinate defaults to `z` for `proximal_space`,
`inner_proximal_space`, and `acceleration_space`. Plain `ALM` and JCC remain
direct x-space penalty methods by default.

Everything else should inherit from the same canonical family unless there is a
clear algorithmic reason to split it out.

ALM-family experiment overrides are normalized before they are merged:

- `common_config` owns shared primal step controls:
`learning_rate`, `stepsize_rule`, `lr_decay`, and `min_lr`.
- `outer_common` owns outer-loop thresholds, dual, penalty, and proximal
controls, not primal step controls.
- Direct `ALM` / `Hom-ALM` entries in `algorithm_config` should use
`learning_rate`, `stepsize_rule`, `lr_decay`, and `min_lr` for per-method
primal step overrides.
- `inner_solver_common` should use `inner_learning_rate`,
`inner_stepsize_rule`, `inner_lr_decay`, and `inner_min_lr`.
- Generated iterative `ALM` and `Hom-ALM` runtime parameter dictionaries use
optimizer-internal `outer_stepsize_rule` and `outer_lr_decay`; scripts should
not set those names directly.
- CVXPY `*-EQ` methods receive only active outer penalty/proximal/dual and
solver fields; gradient-inner-loop fields are intentionally filtered out.

Benchmark-side ALM-family defaults in
`src/homopt/experiments/_benchmark_common.py` now also reuse a shared helper,
`penalty_family_shared_defaults(...)`, for:

- `dual_learning_rate`
- `penalty_coef / penalty_growth`
- `proximal_coef`
- `max_penalty / max_dual`
- `use_lagrangian / use_penalty / use_proximal`

### Optimizer-side semantics

`src/homopt/optim/_core_impl.py` now treats these fields consistently across
`LagrangianOptimizer`, `HomALMOptimizer`, and `EqualityConstrainedALMOptimizer`.

Shared semantics:

- `dual_learning_rate` controls the dual-update multiplier together with the
current `penalty_coef`.
- `penalty_coef` is the current outer penalty coefficient.
- `penalty_growth` is the multiplicative factor applied when the active penalty
violation metric does not decrease enough across outer iterations. The
current trigger uses a relative tolerance and treats
`current_violation >= previous_violation * 0.999` as "not decreased".
- `max_penalty` is the hard cap for `penalty_coef`.
- `max_dual` is the clamp bound for dual variables.
- `proximal_coef` is only active when `use_proximal=True`.
- `proximal_space` chooses where the proximal term is applied.
- `outer_stepsize_rule` controls primal lr across outer iterations.
- `inner_stepsize_rule` controls primal lr updates inside each outer subproblem.
- `min_lr` is the explicit lower bound for the outer primal learning rate.
- `inner_min_lr` is the explicit lower bound for inner primal learning rates.
These bounds are not inferred from `convergence_threshold`.

Update logic:

1. Run the inner loop for the current outer state.
2. Evaluate objective and constraint violation.
3. Update dual variables using `penalty_coef * dual_learning_rate`.
4. If the active penalty violation metric does not decrease, multiply
  `penalty_coef` by `penalty_growth`, clipped by `max_penalty`.
5. Clamp dual variables by `max_dual`.

### Unified primal lr policy for ALM / Hom-ALM

`ALM` and `Hom-ALM` now use the same primal learning-rate strategy.

The policy is split into two explicit controls:

- `outer_stepsize_rule`
- `inner_stepsize_rule`
- `min_lr`
- `inner_min_lr`

Recommended interpretation:

- outer rule controls how the base primal lr changes from one outer iteration to
the next
- inner rule controls how the current primal lr is adjusted while solving the
current outer subproblem

Current defaults:

- `outer_stepsize_rule` defaults to `"adaptive"`
- `inner_stepsize_rule` defaults to `"constant"`

This means the default penalty-style behavior is now:

- outer primal lr adapts/diminishes according to the configured outer rule
- inner primal lr stays fixed unless the user explicitly asks for inner
adaptation
- adaptive/diminishing lr updates are clipped by the matching explicit minimum
lr field

This removes the previous inconsistency where `ALM` and `Hom-ALM` interpreted
`stepsize_rule` differently.

For `ALM`, this also means the constant-inner-stepsize route now skips the
adaptive inner-lr bookkeeping, matching `Hom-ALM`.

### Unified acceleration controls

Acceleration is now represented by explicit method and coordinate-space fields:

```
acceleration_method = "none" | "nag"
acceleration_space = "z" | "x"
```

Current behavior:

- `acceleration_method="nag"` selects standard Nesterov accelerated gradient
- `acceleration_method="none"` leaves acceleration disabled
- `momentum` is the NAG lookahead coefficient `beta`
- `opt` must be `"gd"` when `acceleration_method="nag"`
- old `accelerate`, `acceleration_method="fista"`, and `opt="nesterov"`
parameter paths are no longer part of the active config surface

For homeomorphic methods, `acceleration_space="z"` extrapolates directly in
latent coordinates. `acceleration_space="x"` maps the current and previous
latent states through the homeomorphism, extrapolates in decision space, maps
the extrapolated point back through `hom_map.inverse(...)`, and projects to the
latent ball before taking the gradient/update.

The implemented NAG form is:

```
y_k = x_k + beta * v_k
g_k = grad f(y_k)
x_{k+1} = project_or_map(y_k - alpha * g_k)
v_{k+1} = x_{k+1} - x_k
```

For Hom methods, the same rule is applied in the selected acceleration space.

### Update-rule summary by algorithm family

First-order family (`PGD`, `Hom-PGD`, `FW`, `RD`):

- primal lr uses the first-order scheduler semantics already implemented in
`run_first_order_loop(...)`
- `stepsize_rule` is the active lr rule for these methods
- there is no outer dual update in the main loop

Penalty family (`ALM`, `Hom-ALM`, JCC lagrangian variants):

- primal lr is split into `outer_stepsize_rule` and `inner_stepsize_rule`
- dual update is additive with `penalty_coef * dual_learning_rate`
- penalty update is multiplicative with `penalty_growth`, triggered only when
the active penalty violation metric does not decrease, and capped by
`max_penalty`
- dual variables are clipped by `max_dual`

Embedded penalty subproblems used by first-order baselines:

- `PGD -> projection_subproblem`
- `FW -> linearization_subproblem`

These embedded blocks use the same canonical penalty-method field names as
`ALM` / `Hom-ALM`, so nested penalty-style logic does not introduce a second
naming dialect.

### Current algorithm flow

The active optimizer chain is:

```text
experiment config
  -> canonical builder / normalization helper
  -> params[algorithm_name]
  -> run_algorithm(...)
  -> optimizer.optimize(...)
  -> unified payload with x_solved, x_traj, obj_traj, cons_traj, iter_time
```

Timing payload conventions:

- `iter_time`: per outer / top-level iteration wall time.
- `total_wall_time`: full `optimizer.optimize(...)` wall time measured by
`run_algorithm`.
- `outer_iter_time`: explicit copy of per outer-iteration wall time for
ALM-type solvers.
- `inner_iter_time`: iterative `ALM` / `Hom-ALM` inner-solver time per outer
iteration.
- `solver_iter_time`: CVXPY-backed `*-EQ` subproblem solve time per outer
iteration.
- `initial_ip_time`: gauge origin / interior-point finder time attached to
Hom solvers in convex comparison experiments.
- `last_trans_time`: final latent-to-primal transform time for Hom solvers.

First-order methods:

```text
PGD:
  initialize x
  for each iteration:
    y = NAG lookahead(x) if acceleration_method == "nag" else x
    compute gradient at y
    apply update backend
    project with projection_subproblem / exact projection if needed
    record objective and violation

Hom-PGD:
  initialize x, map z = hom_map.inverse(x)
  for each iteration:
    choose lookahead in z-space or x-space
    compute gradient of f(hom_map.forward(z)) at the lookahead point
    update z and project to the latent ball
    record the mapped decision x = hom_map.forward(z)

Hom-Prox-GD is implemented as an optional staged mode of `Hom-PGD`, not as an
ALM inner solver:

```text
use_proximal = True
proximal_update_iterations = K
proximal_coef = gamma
proximal_space = "z"  # or "x"

steps_per_stage = max_iterations // K

for prox_stage in 0 .. K - 1:
  freeze proximal center at the current z / H(z)
  reset NAG velocity for this stage
  for local_iter in 0 .. steps_per_stage - 1:
    choose lookahead in z-space or x-space
    add the fixed-center proximal term
    update z and project to the latent ball
  refresh the proximal center from the new iterate
```

For `proximal_space="z"`, the z-proximal term is kept in the local quadratic
model and the closed-form proximal point is projected to the latent ball. For
`proximal_space="x"`, the term `gamma/2 ||H(z)-x_c||^2` is differentiated
through the Hom map. The proximal mode requires `opt="gd"`; Adam-style updates
are rejected because they are not standard proximal-gradient steps.

FW:
  initialize x
  for each iteration:
    solve the linear oracle using linearization_subproblem / exact oracle
    move x toward the oracle point
    record objective and violation

RD:
  initialize radial variables
  for each iteration:
    optionally apply NAG lookahead to the radial dual variable
    take a radial-dual gradient step
    map back to the primal decision for reporting

```

Penalty methods:

```text
ALM:
  initialize primal x, dual lambda, penalty rho
  for each outer iteration:
    freeze lambda, rho, and optional outer proximal center
    run the selected inner solver in x-space on the fixed augmented Lagrangian subproblem
    evaluate objective and constraint violation
    update lambda by penalty_coef x dual_learning_rate
    update rho by multiplying penalty_growth only when the active penalty
    violation metric does not decrease, capped by max_penalty
    update outer primal learning rate according to outer_stepsize_rule

Penalty-EQ / Prox-Penalty-EQ / ALM-EQ / Prox-ALM-EQ:
  initialize primal x, equality dual lambda when enabled, penalty rho
  for each outer iteration:
    solve a convex CVXPY subproblem over the original convex inequalities
    add equality quadratic penalty to the objective
    add equality dual term only for ALM-EQ / Prox-ALM-EQ
    add outer proximal term only for the Prox-* variants
    evaluate objective and full/equality/inequality violation at the solution
    update equality lambda when enabled and rho with the ALM outer rules

Hom-ALM:
  initialize x, map z = hom_map.inverse(x)
  for each outer iteration:
    freeze lambda, rho, and optional outer proximal center
    run the selected inner solver in z-space
    map z to x for objective and equality-violation reporting
    update lambda and rho with the same ALM outer rules
```

Within one `ALM` or `Hom-ALM` outer iteration, `lambda`, `rho`, and the outer
proximal term do not change. The default inner solver is plain gradient
descent:

```text
inner_solver = "gd"

for inner_iter in 0 .. inner_iterations - 1:
  step = NAG lookahead(current) if enabled else current
  grad = grad fixed_outer_subproblem(step)
  next = inner_update(step, grad)
  update NAG velocity from current to next
  optionally update inner_lr according to inner_stepsize_rule
```

The staged proximal-refresh inner solver adds a fixed-center proximal term
inside each stage and handles that term through its gradient:

```text
inner_solver = "prox_gd"
inner_proximal_update_iterations = K
inner_proximal_coef = gamma
inner_proximal_space = "x"  # ALM default; Hom-ALM default is "z"

steps_per_stage = inner_iterations // K

for prox_stage in 0 .. K - 1:
  freeze inner proximal center at the current solution
  reset NAG velocity for this stage
  for local_iter in 0 .. steps_per_stage - 1:
    step = NAG lookahead(current) if enabled else current
    grad = grad fixed_outer_subproblem(step)
           + grad frozen_inner_proximal_term(step, center)
    next = inner_update(step, grad)
    update NAG velocity from current to next
  refresh only the inner proximal center from the new solution
```

The implementation intentionally uses integer floor division for
`steps_per_stage`. If the exact full inner budget matters, choose `K` such that
`inner_iterations % K == 0`.

For plain `ALM`, the inner variable is x. For `Hom-ALM`, the inner variable is
z. If `inner_proximal_space="x"` in `Hom-ALM`, the x-space proximal term is
differentiated through the homeomorphic map back to z.

There is no separate `linearized_prox_gd` inner-solver option. If the
nonsmooth part is the indicator of the convex inequality feasible set, the
linearized proximal step is projected-gradient descent. Use the `PGD` /
`Hom-PGD` baselines for first-order projected-gradient comparisons, or use the
`*-EQ` CVXPY baselines to solve each fixed-outer ALM subproblem directly while
keeping convex inequalities hard.

### Solver Configuration Reference

There are two solver layers in the current package:

1. Optimizer solvers selected by `run_algorithm(...)` through algorithm ids
  such as `PGD`, `Hom-PGD`, `ALM`, and `Hom-ALM`.
2. Exact or modeling solvers exposed through `homopt.solvers`, mainly used as
  projection, linear-oracle, reference, or CVXPY/Pyomo subproblem solvers.

Optimizer dispatch:


| Algorithm id      | Optimizer class                   | Requires `hom_map` | Main variable | Subproblem solver                                                                        |
| ----------------- | --------------------------------- | ------------------ | ------------- | ---------------------------------------------------------------------------------------- |
| `PGD`             | `PGDOptimizer`                    | no                 | x             | exact projection for `ConvexOpt` / `MaxCutSDP`; `projection_subproblem` for `ToyStarOpt` |
| `Hom-PGD`         | `HomPGDOptimizer`                 | yes                | z             | latent ball projection                                                                   |
| `FW`              | `FrankWolfeOptimizer`             | no                 | x             | exact linear oracle for `ConvexOpt`; `linearization_subproblem` for `ToyStarOpt`         |
| `RD`              | `RadialDualOptimizer`             | yes                | radial dual y | generalized gauge radial-dual update; constraints enter through `hom_map.gauge`           |
| `ALM`             | `LagrangianOptimizer`             | no                 | x             | gradient inner loop                                                                      |
| `Prox-ALM`        | `LagrangianOptimizer`             | no                 | x             | `ALM` with outer proximal term enabled; paper label `PALM`                               |
| `Penalty-EQ`      | `EqualityConstrainedALMOptimizer` | no                 | x             | `ConvexSolver.solve_type="eq_alm"` without dual                                          |
| `Prox-Penalty-EQ` | `EqualityConstrainedALMOptimizer` | no                 | x             | `eq_alm` with outer proximal term, without dual                                          |
| `ALM-EQ`          | `EqualityConstrainedALMOptimizer` | no                 | x             | `eq_alm` with equality dual                                                              |
| `Prox-ALM-EQ`     | `EqualityConstrainedALMOptimizer` | no                 | x             | `eq_alm` with equality dual and outer proximal term                                      |
| `Hom-ALM`         | `HomALMOptimizer`                 | yes                | z             | gradient inner loop over latent ball                                                     |
| `Prox-Hom-ALM`    | `HomALMOptimizer`                 | yes                | z             | `Hom-ALM` with outer proximal term enabled; paper label `Hom-PALM`                       |


Common optimizer config fields:


| Field                   | Applies to                                                   | Allowed / meaning                                                    |
| ----------------------- | ------------------------------------------------------------ | -------------------------------------------------------------------- |
| `learning_rate`         | all iterative optimizers                                     | initial primal step size                                             |
| `max_iterations`        | first-order methods                                          | outer/main iteration count                                           |
| `outer_iterations`      | penalty methods                                              | ALM outer iteration count                                            |
| `inner_iterations`      | `ALM`, `Hom-ALM`, embedded subproblems                       | fixed-outer inner budget                                             |
| `inner_stopping_rule`   | `ALM`, `Hom-ALM`                                             | `"fixed"` or `"adaptive"` inner-loop stopping                        |
| `inner_iterations_min`  | adaptive `ALM`, `Hom-ALM`                                    | minimum inner steps before adaptive stop                             |
| `inner_iterations_max`  | adaptive `ALM`, `Hom-ALM`                                    | maximum inner steps in adaptive mode                                 |
| `inner_tol_factor`      | adaptive `ALM`, `Hom-ALM`                                    | multiplier for current violation in inner tolerance                  |
| `inner_tol_min`         | adaptive `ALM`, `Hom-ALM`                                    | lower bound for adaptive inner tolerance                             |
| `max_running_time`      | all iterative optimizers                                     | wall-clock cap in seconds                                            |
| `convergence_threshold` | all iterative optimizers                                     | stopping threshold                                                   |
| `stepsize_rule`         | first-order methods                                          | `"constant"`, `"adaptive"`, or `"diminish"`                          |
| `outer_stepsize_rule`   | penalty runtime params                                       | optimizer-internal outer primal lr policy                            |
| `inner_stepsize_rule`   | penalty methods                                              | inner primal lr policy                                               |
| `lr_decay`              | first-order adaptive lr                                      | multiplicative decay                                                 |
| `min_lr`                | first-order and penalty methods                              | lower bound for single-loop and outer primal lr                      |
| `inner_min_lr`          | penalty methods                                              | lower bound for inner primal lr                                      |
| `opt`                   | gradient-update optimizers                                   | `"gd"`, `"normalized_gd"`, or `"adam"`; `opt="nesterov"` is rejected |
| `acceleration_method`   | `PGD`, `Hom-PGD`, `ALM`, `Hom-ALM`, `RD`                     | `"none"` or `"nag"`                                                  |
| `acceleration_space`    | Hom methods and `RD`                                         | `"z"` or `"x"` where implemented                                     |
| `momentum`              | NAG / GD momentum paths                                      | NAG beta when `acceleration_method="nag"`; GD momentum otherwise     |


Penalty-method config fields:


| Field                              | Applies to               | Meaning                                                                                                         |
| ---------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `dual_learning_rate`               | `ALM`, `Hom-ALM`, `*-EQ` | multiplicative factor in the outer dual update `penalty_coef * dual_learning_rate * residual`                   |
| `penalty_coef`                     | penalty-enabled methods  | current quadratic penalty coefficient                                                                           |
| `penalty_growth`                   | penalty-enabled methods  | multiplicative penalty update factor, triggered only when the active penalty violation metric does not decrease |
| `max_penalty`                      | penalty-enabled methods  | penalty coefficient cap                                                                                         |
| `max_dual`                         | dual-enabled methods     | dual variable clamp bound                                                                                       |
| `use_lagrangian`                   | `ALM`, `Hom-ALM`, `*-EQ` | enable dual term and dual update                                                                                |
| `use_penalty`                      | `ALM`, `Hom-ALM`, `*-EQ` | enable quadratic penalty term                                                                                   |
| `use_proximal`                     | `ALM`, `Hom-ALM`, `*-EQ` | enable outer fixed-center proximal term                                                                         |
| `proximal_coef`                    | proximal-enabled methods | outer proximal coefficient                                                                                      |
| `proximal_space`                   | `ALM`, `Hom-ALM`         | `"x"` for plain ALM default, `"z"` for Hom-ALM default                                                          |
| `inner_solver`                     | `ALM`, `Hom-ALM`         | `"gd"` or `"prox_gd"`                                                                                           |
| `inner_proximal_update_iterations` | staged inner solvers     | number of fixed-center proximal refresh stages                                                                  |
| `inner_proximal_coef`              | staged inner solvers     | inner proximal coefficient                                                                                      |
| `inner_proximal_space`             | staged inner solvers     | `"x"` or `"z"`; z is the Hom-ALM default                                                                        |
| `lagrangian_gradient`              | `ALM`                    | default `"explicit"` for comparable experiments; `"autograd"` is an explicit debug/diagnostic choice            |
| `hom_map_gradient`                 | `Hom-PGD`, `Hom-ALM`     | default `"explicit"` for comparable experiments; `"autograd"` is an explicit debug/diagnostic choice            |
| `return_best_violation`            | `ALM`, `Hom-ALM`, `*-EQ` | return the best recorded violation point instead of the final point                                             |


Consistency constraints:

- `acceleration_method="nag"` requires `opt="gd"`.
- `normalized_gd` uses the L2-normalized gradient direction for each sample.
It is a first-order update backend, not an acceleration method.
- `prox_gd` requires `inner_proximal_update_iterations <= inner_iterations`.
- `Hom-PGD` proximal mode requires `use_proximal=True`,
`proximal_update_iterations <= max_iterations`, and `opt="gd"`.
- Non-Hom first-order baselines expose `acceleration_space="x"` in their
generated configs for clarity. `RD` still applies its NAG lookahead to the
radial-dual variable internally.
- `FW` does not currently consume the NAG fields; its main update is the
standard convex combination toward the linear-oracle point.
- `*-EQ` CVXPY baselines intentionally ignore iterative inner-solver,
acceleration, inner-step-size, `inner_iterations`, and `proximal_space`
controls; each outer iteration solves one convex CVXPY subproblem.
- `Hom-ALM` uses equality-only dual residuals when `eq_cons` is present; convex
inequalities are intended to be enforced through the homeomorphism.

Exact/modeling solver reference:


| Solver class                                                 | Supported `solve_type`                                                               | Main options                                                                                                                               | Typical caller                                                                           |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| `ConvexSolver`                                               | `opt`, `initialized_opt`, `proj`, `linear`, `central_ip`, `analytical_ip`, `ip`, `eq_alm` | `equality`, `dual_var`, `penalty_coef`, `proximal_coef`, `x_outer`, `solver_options`, `time_limit_sec`, `verbose`                          | convex reference solve, PGD projection, FW linear oracle, `*-EQ` baselines, gauge origin |
| `MaxCutSolver`                                               | `opt`, `ip`, `proj`                                                                  | `x_init` for projection                                                                                                                    | MaxCut reference / projection                                                            |
| `ChanceConstraintSolver`                                     | `opt`, `initialized_opt`                                                             | `approach`, `x_init`; valid approaches are `"robust-scenario"` and `"mixed-integer"`                                                        | chance-constraint exact baseline                                                         |
| `QCQPSolver`                                                 | `opt`, `initialized_opt`, `proj`                                                     | Pyomo `solver`, `options`, `x_init`                                                                                                        | QCQP exact initialized/projection baseline                                               |
| `JCCDCOPFSolver` and variants                                | `opt`, `initialized_opt`                                                             | CVXPY solver override and `options` through solver config                                                                                  | DC-OPF chance-constraint baselines                                                       |
| `JCCLinearSolver` and variants                               | delegated to problem solver                                                          | `solve_type`, `x_init`, `solver`, `options`                                                                                                | linear JCC baselines                                                                     |
| `StiefelRetractionSolver` / `StiefelRetractionPenaltySolver` | `solve_result(...)` iterative API                                                    | `learning_rate`, `max_iterations`, `stepsize_rule`, `lr_decay`, `min_lr`, `inequality_penalty_coef`                                                  | Stiefel retraction baseline with fixed side-constraint penalty                           |
| `StiefelRetractionALMSolver`                                 | `solve_result(...)` iterative API                                                    | `learning_rate`, `outer_iterations`, `inner_iterations`, `lr_decay`, `min_lr`, `dual_learning_rate`, `penalty_coef`, `penalty_growth`, `max_penalty`, `max_dual` | Stiefel retraction baseline with ALM side-constraint multipliers                         |
| `StiefelALMEQPGDSolver`                                      | `solve_result(...)` iterative API                                                    | `learning_rate`, `outer_iterations`, `inner_iterations`, `lr_decay`, `min_lr`, `projection_solver_options`, `projection_solver_priority`                         | equality-ALM baseline with CVXPY projection onto convex side constraints                 |
| `StiefelPyomoIPOPTSolver`                                    | `solve_result(...)` modeling API                                                     | `solver`, `solver_options`, `multi_start`, `tee`                                                                                           | small Stiefel NLP reference solve                                                        |


`ConvexSolver` modeling structure by `solve_type`:

```text
opt / initialized_opt:
  minimize quadratic objective
  keep convex inequalities, box bounds, and equalities hard

proj:
  minimize ||x - x_init||^2
  keep convex inequalities, box bounds, and equalities hard

linear:
  minimize <grad, x>
  keep convex inequalities, box bounds, and equalities hard

central_ip / analytical_ip:
  introduce epsilon
  shrink convex inequalities and box bounds by epsilon
  optionally keep equalities hard according to equality=True/False

ip:
  find any feasible point, using numeric eps as a strict feasibility margin
  optionally keep equalities hard according to equality=True/False

eq_alm:
  keep convex inequalities and box bounds hard
  relax only linear equalities into the objective through
  optional dual, quadratic penalty, and proximal terms
```

Exact-solver compatibility cleanup on 2026-04-22:

- `solve_exact_result(...)` now preserves `solver_options` when the target
solver accepts that name, and only maps it to `options` for solvers whose
public signature uses `options`.
- `ConvexSolver.solve_type="eq_alm"` reports `hard_equalities=False` in result
extras, matching the model where equalities are relaxed into the objective.
- `ChanceConstraintSolver` defaults to `approach="robust-scenario"`.
- `QCQPSolver.solve(...)` rejects unknown `solve_type` values instead of
silently treating them as `opt`.
- `QCQPSolver.check_feasibility(...)` now computes the radius-ball violation as
`max(sum(x_i^2) - R^2, 0)`, matching the Pyomo model.

### Acceleration structure

There is one active acceleration rule: standard NAG. The update backend remains
ordinary gradient descent when NAG is enabled:

```text
opt = "gd"
acceleration_method = "nag"
momentum = beta
```

The lookahead point is where the gradient is evaluated:

```text
y_k = x_k + beta * v_k
g_k = grad f(y_k)
x_{k+1} = project_or_map(y_k - alpha * g_k)
v_{k+1} = x_{k+1} - x_k
```

For homeomorphic methods, `acceleration_space` selects the coordinate system
for the lookahead:

- `acceleration_space="z"`: extrapolate directly in latent coordinates.
- `acceleration_space="x"`: map `z -> x`, extrapolate in decision space, map
the lookahead point back through `hom_map.inverse(...)`, then project to the
latent ball.

`acceleration_method="fista"` and `opt="nesterov"` are not active interfaces.
FISTA-style acceleration is intentionally not mixed into the current Hom-ALM
path because its proximal-gradient sequence and restart conventions are a
different algorithmic choice from standard NAG.

### 2026-04-22 normalized gradient smoke

`NormalizedGDOptimizer` was added as a small first-order update backend exposed
by:

```text
opt = "normalized_gd"
```

The update is:

```text
d_k = grad f(x_k) / max(||grad f(x_k)||_2, eps)
x_{k+1} = project_or_map(x_k - alpha d_k)
```

It can be used by `Hom-PGD` through `algorithm_config={"Hom-PGD": {"opt": "normalized_gd"}}`. It is intentionally not combined with NAG or staged
proximal Hom-PGD mode; those paths still require `opt="gd"`.

Smoke comparison on one 10D convex SOCP/QCQP-style instance, same seed,
warm-start, `learning_rate=0.05`, `max_iterations=80`:


| Hom-PGD opt     | final objective | final violation |
| --------------- | --------------- | --------------- |
| `gd`            | `-0.357844`     | `0.0`           |
| `normalized_gd` | `-0.378409`     | `0.0`           |


High-dimensional box-only Hom-PGD smoke, same seed, warm-start,
`learning_rate=0.05`, `max_iterations=300`:


| n_var  | Hom-PGD opt     | final objective | final violation |
| ------ | --------------- | --------------- | --------------- |
| `100`  | `gd`            | `44.809719`     | `0.0`           |
| `100`  | `normalized_gd` | `39.646709`     | `0.0`           |
| `1000` | `gd`            | `580.994263`    | `0.0`           |
| `1000` | `normalized_gd` | `579.602112`    | `0.0`           |


For high-dimensional SOC/quadratic constrained instances, the interactive
smoke bottleneck is currently the CVXPY feasibility solve used to construct
the gauge origin, not the Hom-PGD update backend itself.

### Equality-specific Hom-ALM behavior

`Hom-ALM` should be interpreted as:

- equality constraints handled by outer dual / penalty updates
- inequality constraints handled by the homeomorphism

Because of that, its outer stopping and violation tracking should use equality
residuals only. This was aligned in the implementation.

`Hom-ALM` now also exposes:

- `proximal_space = "z"` as the default
- `proximal_space = "x"` as an explicit alternative

Recommended interpretation:

- use `z`-prox when you want regularization in the actual optimization variable
used by `Hom-ALM`
- use `x`-prox when you want outer-loop stabilization in the original decision
space after the homeomorphic map

`ALM` and `Hom-ALM` also support the same optional staged inner-solver
proximal-refresh mode:

```
inner_solver = "prox_gd"
inner_proximal_update_iterations = K
inner_proximal_coef = ...
inner_proximal_space = "x"  # or "z" for Hom-ALM
```

Within one fixed outer iteration, the dual variable, penalty coefficient, and
optional outer proximal center stay fixed. The `prox_gd` inner solver splits
the total `inner_iterations` budget into `K` fixed-center stages, runs
`inner_iterations // K` local gradient steps on the same outer subproblem plus
the gradient of an inner proximal term, then refreshes only the inner proximal
center from the new inner solution. This inner proximal center is separate
from the outer-loop `use_proximal` / `proximal_coef` / `proximal_space` term.

The old `linearized_prox_gd` name was removed from active configuration. Its
mathematical role overlaps projected-gradient descent when the nonsmooth term
is the convex-inequality indicator, so experiments should use `PGD` /
`Hom-PGD` or the CVXPY `*-EQ` subproblem solvers instead.

For the package-native convex test problems, `convex_algorithm_comparison(...)`
now uses `inner_iterations = 50` for the default `ALM` and `Hom-ALM` iterative
penalty baselines. The dedicated `hal_prox_gd` and `hal_prox_gd_x` configs keep
the same 50-step inner budget.

### Stability note

`run_penalty_outer_loop(...)` already applies the outer-loop diminish rule. The
`Hom-ALM` optimizer should therefore not apply an extra diminish transform in
its own `update_learning_rate(...)` callback. That duplication was removed.

### 2026-04-19 Hom-ALM test follow-up

The latest 10D equality-constrained convex benchmark exposed two correctness
issues in the ALM-side evaluation path:

- `ConvexOptEq.lagrangian_x(...)` now applies the quadratic penalty to equality
residuals as well as clipped inequality residuals.
- `_constraint_violation(..., equality_only=False)` now computes
`max(max(g_ineq(x), 0), abs(h_eq(x)))` instead of taking `abs()` of all raw
residuals, so satisfied inequalities with large negative slack are no longer
counted as violations.

The `Hom-ALM` equality-only metric was already using equality residuals, so its
reported equality violation did not change from this fix. In the current 10D
benchmark, default `Hom-ALM` reaches about `1e-3` equality violation at its best
iterate and about `1e-2` at the final iterate. Exact nonsmooth gauge is faster
than smooth gauge with similar numerical behavior. The best narrow-sweep
variant so far is:

```
opt = "gd"
acceleration_method = "nag"
acceleration_space = "z"
momentum = 0.9
dual_learning_rate = 5e-2
penalty_coef = 100.0
penalty_growth = 0.0
smooth = False
```

This reaches about `1.1e-3` final equality violation on the tested seed, but it
still does not meet the `1e-5` feasibility threshold. The current bottleneck is
therefore algorithmic convergence/stabilization, not just config exposure.

### 2026-04-19 equality benchmark reporting cleanup

The equality-constrained single-problem example now separates optimizer-internal
violation from benchmark-reporting violation:

- `ALM` optimizes and reports full violation internally because it owns both
convex inequalities and linear equalities.
- `Hom-ALM` still uses equality-only violation internally because convex
inequalities are handled by the gauge/homeomorphic map.
- The benchmark comparison rows now recompute a common reporting split from
the final decision:
`final_full_violation`, `final_inequality_violation`, and
`final_equality_violation`.
- The row field `violation` is the common full violation for all methods.
- `final_objective` is also recomputed at `x_solved`, so objective and
violation refer to the same reported decision.
- `optimizer_reported_final_objective` and
`optimizer_reported_final_violation` preserve the raw optimizer trajectory
endpoint for debugging.

`problem_type="socp_eq"` defaults to one linear equality if the caller
does not provide `n_lin_eq` or an explicit `A_eq, b_eq` override. The old
`problem_type="eq"` spelling is rejected. This prevents equality examples from
silently depending on an unclear shorthand.

### 2026-04-19 solver efficiency cleanup

The first solver-internal cleanup pass targeted low-risk overhead that appears
inside iterative experiment loops:

- CVXPY installed-solver discovery is now cached by module and priority tuple.
This avoids repeated `installed_solvers()` calls for every exact-solver call.
- Exact-solver signature inspection is cached for `solve_exact_result(...)`.
PGD/FW repeatedly call exact projection or linear-oracle solvers, so repeated
`inspect.signature(...)` was unnecessary overhead.
- `PGDOptimizer` and `FrankWolfeOptimizer` now cache package exact-solver
instances instead of rebuilding `ConvexSolver` / `MaxCutSolver` inside every
projection or linear-oracle call.
- `FrankWolfeOptimizer` no longer constructs a `LinearProblem` wrapper for
`ConvexOpt`; it directly uses `problem.gradient_objective_x(x)` and reserves
the wrapper for `ToyStarOpt`, where it is still needed.
- `MaxCutSolver` now builds an upper-triangle edge-index map and preserves the
original edge/weight order, including reversed edges. This removes
list-membership overhead and fixes the previous `(j, i) in self.node` check.

This cleanup does not change algorithm semantics. The remaining larger
efficiency opportunity is CVXPY problem reuse with `Parameter` objects for
projection and linear-oracle calls; that is a bigger refactor because objective
data changes between iterations.

### 2026-04-22 convex equality comparison cleanup

The fixed-problem convex comparison route now keeps solver-family defaults in
`_benchmark_common.py` instead of repeating them inside
`convex_algorithm_comparison(...)`:

- `build_convex_penalty_method_kwargs(...)` owns the shared iterative
`ALM` / `Hom-ALM` defaults for convex comparison runs:
`inner_iterations = 50`, `inner_solver = "gd"`, `penalty_coef = 10.0`,
`penalty_growth = 1.1`, `proximal_coef = 0.1`, `max_penalty = 1e2`, and
`max_dual = 1e2`.
- `ALM` and `Hom-ALM` use the same inner-solver settings by default. The
intended experiment difference is the optimization space: x-space for
`ALM`, z-space for `Hom-ALM`. For equality problems both use
`dual_learning_rate = 1e-1`; inequality-only ALM keeps the smaller
`5e-3` default.
- `EQ_CVXPY_BASELINE_SPECS` and `build_eq_cvxpy_baseline_params(...)` own the
four exact-subproblem equality baselines:
`Penalty-EQ`, `Prox-Penalty-EQ`, `ALM-EQ`, and `Prox-ALM-EQ`.
- The `*-EQ` baselines intentionally omit iterative inner-solver fields such
as `inner_solver`, `inner_proximal_*`, `acceleration_method`, and
`inner_stepsize_rule`, and they also omit `inner_iterations` and
`proximal_space`; each outer step is a CVXPY convex subproblem.
- `ConvexSolver.solve_type="eq_alm"` now delegates target construction to
`_eq_alm_objective_expr(...)`. The model keeps convex inequalities and box
constraints hard, relaxes only linear equalities into the objective through
optional dual, penalty, and proximal terms, and preserves the existing
`0.5 * coef * ||.||^2` coefficient convention.

Validation:

- `python -m py_compile src/homopt/experiments/_benchmark_common.py src/homopt/experiments/benchmarks_single.py src/homopt/solvers/core.py tests/test_experiment_utils.py tests/test_benchmark_workflows.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_experiment_utils.py tests/test_benchmark_workflows.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_optimizer_loops.py tests/test_solver_exports.py`

### 2026-04-22 convex first-order solver cleanup

The same fixed-problem convex route now also shares the first-order baseline
subproblem config:

- `build_first_order_subproblem_params(...)` owns the common Lagrangian
subproblem defaults used by PGD projection and FW linear-oracle fallbacks:
`inner_solver = "gd"`, `inner_proximal_update_iterations = 1`,
`inner_proximal_coef = 0.0`, `inner_proximal_space = "x"`,
`acceleration_method = "none"`, `acceleration_space = "x"`,
`dual_learning_rate = 1e-2`, and the canonical penalty/proximal caps.
- `build_first_order_method_params(...)` now calls this helper for both
`projection_subproblem` and `linearization_subproblem`, so the only intended
difference is the requested outer/inner budget.
- `convex_algorithm_comparison(...)` now builds its common solver payload from
`base_common_params(...)` and then applies route-specific fields such as
`initial_point_mode`, `acceleration_method`, `acceleration_space`, `smooth`,
and `momentum`. Iterative algorithms use `initial_point_mode` for start-point
semantics; exact solver refinement uses `solve_type="initialized_opt"`.
- Optimizer-side fallbacks for `PGDOptimizer` and `FrankWolfeOptimizer` use one
private helper with the same structure, so tests that instantiate optimizers
directly do not silently get a different subproblem config.
- `ConvexSolver.solve(...)` now uses small objective helpers for quadratic,
projection, linear-oracle, and equality-ALM modes. Local CVXPY variables are
named as decision variables in the solve branches, while the mathematical
model remains unchanged.

Validation:

- `python -m py_compile src/homopt/experiments/_benchmark_common.py src/homopt/experiments/benchmarks_single.py src/homopt/optim/_core_impl.py src/homopt/solvers/core.py tests/test_experiment_utils.py tests/test_optimizer_loops.py tests/test_solver_exports.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_experiment_utils.py tests/test_optimizer_loops.py tests/test_solver_exports.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_benchmark_workflows.py`

### Practical guidance

When adding a new penalty-style solver or benchmark route:

1. Start from `normalize_penalty_method_config(...)`.
2. Override only route-specific items.
3. Keep `penalty / dual / proximal` names identical to the canonical fields.
4. Do not introduce a second config family unless the algorithm really has
  different outer-loop semantics.

When adding a new first-order baseline:

1. Reuse `build_first_order_method_params(...)`.
2. Prefer a descriptive nested subproblem name like `*_subproblem`.
3. Prefer `*_outer_iterations` / `*_inner_iterations` for canonical flat names.
4. Do not emit old short aliases such as `proj_outer_iter` or `loo_subproblem`
  from active experiment builders.

### Validation

Regression coverage now checks:

- helper-level default consistency across `ALM`, `Hom-ALM`, and JCC config
assembly
- `penalty_growth` and `max_penalty` behavior across
`LagrangianOptimizer`, `HomALMOptimizer`, and
`EqualityConstrainedALMOptimizer`

Validation run:

- `conda run --no-capture-output -n ML python -m pytest tests`
- result: `239 passed`
