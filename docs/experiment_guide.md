# Experiment Guide

This is the active guide for HomOPT experiment routes, script run flow, and
extension patterns.

## Owner Map

- `scripts/hom_pgd/`
  - editable script presets for inequality-only Hom-PGD comparisons and toys
- `scripts/hom_alm/`
  - editable script presets for equality + inequality Hom-ALM comparisons
- `scripts/inn_pgd/`
  - editable script presets for nonconvex QCQP/JCC INN-PGD experiments
- `scripts/learning_postprocess/`
  - editable script presets for predictor + post-processing experiments

Practical rule:

- keep scripts thin: script-local defaults, overrides, and entrypoint only
- put benchmark execution under `src/homopt/experiments/benchmarks/`
- expose script workloads through the family facade modules such as
  `homopt.experiments.hom_pgd`, `homopt.experiments.hom_alm`,
  `homopt.experiments.inn_pgd`, and `homopt.experiments.learning_postprocess`

## Two Main Routes

### Route A: Iterative Optimization

This route covers methods that start from an initial iterate and improve it by
optimization steps.

Examples:

- `PGD`
- `Hom-PGD`
- `INN-PGD`
- `FW`
- `ALM`
- `RD`

The owning layers are:

- problem definitions in `src/homopt/problems`
- optimizer logic in `src/homopt/optim`
- INN-specific logic in `src/homopt/models`
- experiment orchestration in `src/homopt/experiments`

### Route B: Learning-Based Prediction And Refinement

This route covers methods that first predict a candidate solution or feasible
interior point, then optionally refine it.

Current learning + post-processing examples:

- raw `constant_decision` predictor
- raw `nn_decision` predictor
- `predict_only`
- `predict_diff_projection` via penalty-gradient post-processing

The active learning + post-processing scripts are for convex-parametric and
AC-OPF predictor/refiner studies. Nonconvex QCQP is intentionally handled by
the INN-PGD route, not by a `run_qcqp_predictor_postprocess.py` script.

The learning + post-processing flow has three explicit stages:

1. Data collection: build/load the parametric problem and sample test
   instances once.
2. Predictor training/loading: train or load the NN predictor; constant
   predictors have zero train time.
3. Post-process evaluation: run each selected refiner on the same predictor
   outputs and test instances.

If a future learning + post-processing baseline uses INN or IPNN as a refiner,
that baseline must own a separate post-process model training/loading stage
before refinement. The current INN/IPNN training path is the INN-PGD family.

The raw predictor stage is package-native:

- `predictor_type="constant"` keeps the old smoke-style constant decision route
- `predictor_type="nn"` trains or loads a residual MLP predictor before the
  shared refinement/evaluation loop
- predictor training artifacts and train time are recorded separately from
  post-process evaluation time
- benchmark metrics include `data_collection_time_sec`,
  `predictor_train_time_sec`, `postprocess_train_time_sec`, and
  `refine_time_sec`

Learning-route logic should use package-native adapters in `src/homopt/problems`
and `src/homopt/learning`, not archived script utilities.

## Shared Contract

Both routes should share:

- objective evaluation
- constraint residual evaluation
- feasibility checks
- projection or refinement hooks when available
- parametric instance sampling when relevant
- result persistence through `src/homopt/experiments/results.py`

Top-level result fields should stay aligned:

- `objective`
- `feasible`
- `metrics`
- `artifacts`
- `context`

Comparable route metrics should include:

- mean / best objective
- feasibility rate
- average constraint violation
- runtime per instance
- total runtime
- solved / success rate

Learning routes may add route-specific metrics such as prediction time,
refinement time, post-refinement gain, and train time inside `metrics`.

## Run Flow

Supported experiment entrypoints:

1. `scripts/<family>/run_*.py` for editable script-style runs.

Editable scripts use:

1. script-local params
2. direct package workload callable
3. `record_script_run(...)`
4. `run_and_record(...)`

Each run writes:

- `config.json`
- `result.json`
- optional `artifacts/`

Editable script entrypoints use `results/<family>/...`, where `<family>`
matches the script folder such as `hom_pgd`, `hom_alm`, `inn_pgd`, or
`learning_postprocess`.

## Test Fixture Configs

There are no active root-level experiment config directories. Test-only JSON
fixtures live under `tests/fixtures/configs/`.

Each config has three top-level fields:

```json
{
  "name": "convex_smoke",
  "run": {
    "callable": "homopt.experiments.examples:convex_problem_summary",
    "kwargs": {
      "seed": 7
    }
  },
  "output": {
    "root": "results",
    "group": "smoke"
  }
}
```

Those JSON files are parser smoke fixtures, not a supported experiment-running
interface.

## Canonical Config Groups

### Single / Fixed-Problem Benchmarks

Owned by modules under `src/homopt/experiments/benchmarks/`.

Canonical groups:

- `common_config`
- `problem_config`
- `algorithm_config`

Fixed convex optimization flow:

- Main comparison entrypoint: `convex_algorithm_comparison(...)`.
- Lightweight Hom-PGD smoke entrypoint: `socp_hompgd_benchmark(...)`.
- Problem builder: `build_convex_problem_config(...)` creates a deterministic
  `ConvexOpt` instance for `problem_type="socp"` and `ConvexOptEq` for
  `problem_type="socp_eq"`.
- Reference context: `prepare_convex_reference_context(...)` builds the
  interior point and gauge map; exact CVXPY objective reference is computed for
  comparison runs unless a smoke path explicitly disables it.
- Default convex-inequality algorithms: `PGD`, `FW`, `RD`, `Hom-PGD`.
- Default equality algorithms: `ALM`, `Prox-ALM`, `Penalty-EQ`,
  `Prox-Penalty-EQ`, `ALM-EQ`, `Prox-ALM-EQ`, `Hom-ALM`,
  `Prox-Hom-ALM`.
- Default iterative `ALM` and `Hom-ALM` settings come from
  `build_convex_penalty_method_kwargs(...)`; the intended default difference
  is x-space versus z-space, not a different inner solver.
- `PGD` projection and `FW` linear-oracle fallback subproblems come from
  `build_first_order_subproblem_params(...)`; both use the same x-space
  Lagrangian subproblem defaults unless overridden by `projection_subproblem`
  or `linearization_subproblem`.
- The `*-EQ` CVXPY baselines preserve convex inequalities as hard constraints
  and apply penalty, optional proximal regularization, and optional ALM dual
  updates only to the equality residual.
- The four `*-EQ` baselines share one experiment-side specification table;
  they differ only by `use_lagrangian`, `use_penalty`, and `use_proximal`.
- Optional exact solver row: pass `include_reference_solver=True`.
- High-dimensional equality presets use `reference_need_opt=False` to skip the
  exact optimum solve while still building the interior point needed by the
  gauge map.

Current high-dimensional equality-ratio presets:

| Preset | `n_var` | `n_qua_cons` | `n_soc_cons` | `n_lin_eq` | outer | inner | time cap |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hal_ratio_n100_ineq8_eq2` | 100 | 40 | 40 | 20 | 100 | 100 | 600 |
| `hal_ratio_n100_ineq5_eq5` | 100 | 25 | 25 | 50 | 100 | 100 | 600 |
| `hal_ratio_n100_ineq2_eq8` | 100 | 10 | 10 | 80 | 100 | 100 | 600 |
| `hal_ratio_n1000_ineq8_eq2` | 1000 | 400 | 400 | 200 | 1000 | 100 | 600 |
| `hal_ratio_n1000_ineq5_eq5` | 1000 | 250 | 250 | 500 | 1000 | 100 | 600 |
| `hal_ratio_n1000_ineq2_eq8` | 1000 | 100 | 100 | 800 | 1000 | 100 | 600 |

Metric rule:

- Optimizer-internal violation trajectories may follow algorithm-specific
  update logic.
- Benchmark comparison rows recompute a common final residual split from
  `x_solved`: `full_violation`, `inequality_violation`, and, when applicable,
  `equality_violation`.
- The comparable row field `violation` is always the full violation.

### JCC Benchmarks

Owned by `src/homopt/experiments/benchmarks/jcc/`.

Canonical groups:

- `problem_config`
- `solver_configs`
- `algorithm_config`

### Parametric QCQP / INN / Learning Benchmarks

Owned by `src/homopt/experiments/benchmarks/qcqp/`,
`src/homopt/experiments/benchmarks/convex_parametric.py`, and
`src/homopt/experiments/benchmarks/acopf_parametric.py`.

Canonical groups:

- `problem_config`
- `predictor_model_config`
- `predictor_train_config`
- `model_config`
- `optimizer_config`
- `train_config`
- `solver_config`

## Adding A New Experiment

Put new logic in the deepest layer that actually owns the behavior.

Recommended path:

1. Add a workflow callable under the owning `src/homopt/experiments/benchmarks/`
   module.
2. Add a thin editable script under the appropriate `scripts/<family>/` folder.
3. Add at least one regression test for the public entry surface.
4. Update this guide or `docs/reproduce_experiments.md` if the user-facing flow
   changes.

Minimal workflow shape:

```python
def my_new_experiment(seed=2025, output_dir=None, device=None, dtype=None, **kwargs):
    return {
        "objective": 0.0,
        "feasible": True,
        "artifacts": {},
        "metrics": {},
    }
```

Do not:

- put real experiment logic in `scripts/run_*.py`
- add compatibility shim layers for new experiments
- add archived-script loading to normal workflows
- duplicate parser logic across entrypoints
- duplicate CSV/JSON summary writers in each script

## Archived References

The old route/config/template docs are archived under:

- `docs/archive/experiment_docs/`

Historical predecessor code is archived under:

- `archive/Homeomorphic_Projection/`
- `archive/backup/`
