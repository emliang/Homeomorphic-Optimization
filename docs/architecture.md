# HomOPT architecture

## Runtime shape

The runtime is package-first under `src/homopt`. New experiments should run through `homopt.experiments`, not archived root scripts.

Core layers (lower layers never import from `homopt.experiments`):

1. `homopt.problems`
   Problem definitions: convex/SOCP (`convex/`), QCQP and parametric QCQP
   (`qcqp/`), parametric convex (`convex_parametric/`), MaxCut, JCC (`jcc/`),
   Stiefel (`stiefel/`), and AC-OPF (`acopf/`).
2. `homopt.mappings`
   Gauge mapping (`gauge/`) and star mapping used by Hom-PGD-style methods.
3. `homopt.optim`
   Shared optimizers and algorithm dispatch: `core/` (oracles, penalty,
   constraints, recording), `first_order/` (PGD, Hom-PGD, Frank-Wolfe, radial
   dual), `alm/` (ALM, Hom-ALM, equality), `inn/`, and `manifolds/` (Stiefel).
4. `homopt.models`
   INN models, flow blocks (`flows/`), CvxINN (`cvxinn/`), adversarial and
   IPNN components, latent-domain sampling.
5. `homopt.learning`
   Predictors, refiners (`refiners/`), bisection (`bisection/`), adversarial
   attack/data/config helpers (`adversarial/`), and training loops
   (`training/`).
6. `homopt.solvers`
   Package-owned exact/reference solver interfaces (CVXPY, Pyomo, QCQP, JCC,
   MaxCut, Stiefel, chance constraints, reference caching).
7. `homopt.viz`
   Plotting surface: shared `style`/`primitives`/`traces`/`artifacts` helpers
   plus per-figure modules.
8. `homopt.records`
   Standardized run-artifact persistence (`artifacts.py`: JSON/CSV/NumPy
   payloads, comparison artifact load/save).
9. `homopt.experiments`
   Reproducible run surface consumed by `scripts/` (catalog, run framework,
   family facades, and package-owned workloads).
10. `homopt.utils`
   Cross-cutting import/runtime helpers.

At the experiment level, the active script families are:

- `hom_pgd`: Hom-PGD baselines for inequality-only constraints
- `hom_alm`: Hom-ALM/PALM baselines for equality plus inequality constraints
- `inn_pgd`: INN-PGD training and parametric-test workflows
- `learning_postprocess`: predictor plus post-processing baselines

These routes should share the same problem definitions, result schema, and
visualization surface. See `docs/experiment_guide.md`.

## Experiment architecture

`homopt.experiments` mixes four kinds of code; keeping the distinction in mind
makes the package easier to navigate:

### A. Run framework (orchestration)

1. `catalog.py`
   Single source of truth for the script experiment catalog
   (`SCRIPT_EXPERIMENTS`, `EXPERIMENT_FAMILIES`).
2. `context.py`
   `ExperimentContext` dataclass propagated by script entrypoints
   (entrypoint, seed, device, dtype, output dir).
3. `results.py`
   `ExperimentResult` dataclass plus JSON-safe `save_result` / `load_result`.
4. `runner.py`
   `run_and_record`: normalizes a workload payload into `ExperimentResult` and
   persists it.
5. `entrypoints.py`
   `record_script_run`, `default_script_output_dir`, and comparison-summary
   helpers shared by script runs.
6. `script_runtime.py`
   Helpers for directly runnable scripts (device/CUDA fallback, run naming).

Standardized persistence layout per run: `config.json`, `result.json`, and an
`artifacts/` directory.

### B. Family facades (public entrypoints for `scripts/`)

`hom_pgd.py`, `hom_alm.py`, `inn_pgd.py`, and `learning_postprocess.py` re-export
the benchmark callables each script family needs. Scripts import only from these
facades, never from `benchmarks/` internals. The four facades mirror the
`scripts/{hom_pgd,hom_alm,inn_pgd,learning_postprocess}/` folders and the
`EXPERIMENT_FAMILIES` entries in `catalog.py`.

### C. Package-owned workloads

- `benchmarks/`
  Workload implementations grouped by problem family: `convex.py`,
  `convex_eq_2d.py`, `convex_parametric.py`, `acopf_parametric.py`, `maxcut.py`,
  `star.py`, `parametric_registry.py`, plus the sub-packages `qcqp/`, `jcc/`,
  and `stiefel/`.
- `adversarial/`
  The adversarial-attack experiment workflow/evaluation surface. Attack,
  dataset/config/selection helpers, model definition, training, and
  visualization live in owner layers and are exposed here through naming seams
  (see below).

### D. Shared experiment helpers

- `common/`
  Cross-workload helpers: comparison assembly, IO, labels/naming, reports,
  parametric learning/runtime, run records, single-problem setup, and INN
  baselines.
- `method_specs/`
  Method-spec factories (`build_first_order_method_params`,
  `build_penalty_method_params`, `normalize_jcc_lagrangian_config`, ...) in
  `common.py`.

For exact call flow see `docs/experiment_guide.md`.
For old-script-to-new-entrypoint mapping see `docs/reproduce_experiments.md`.
For the recommended extension pattern and iterative-vs-learning route split, see
`docs/experiment_guide.md`.

## Naming seams (intentional re-export modules)

Some modules under `homopt.experiments` are deliberately thin `import *`
re-exports of an implementation that lives in a lower owner layer. The intent is
that the **implementation has a single home in the natural owner layer**, while
the experiments layer keeps a stable, domain-local name so that everything
related to one experiment is reachable under one namespace.

These seams are **not dead code**; their exact contents are asserted by
`tests/test_structural_boundaries.py`
(`test_common_has_no_benchmark_facade`,
`test_reference_solver_helpers_live_under_solvers`,
`test_adversarial_attack_core_lives_in_dedicated_module`,
`test_common_uses_clear_module_names`). Removing or editing a seam requires
updating the corresponding assertion.

Current seams (re-export module -> owner of the implementation):

- `experiments/common/artifacts.py` -> `homopt.records.artifacts`
- `experiments/common/comparison.py` -> `homopt.records.comparison`
- `experiments/common/io.py` -> `homopt.utils.io`
- `experiments/common/methods.py` -> `homopt.experiments.method_specs.common`
- `experiments/common/references.py` -> `homopt.solvers.references`
- `experiments/method_specs/__init__.py` -> `experiments/method_specs/common.py`
- `experiments/adversarial/attack.py` -> `homopt.learning.adversarial.attack`
- `experiments/adversarial/common.py` -> `homopt.learning.adversarial.common`
- `experiments/adversarial/config.py` -> `homopt.learning.adversarial.config`
- `experiments/adversarial/data.py` -> `homopt.learning.adversarial.data`
- `experiments/adversarial/selection.py` -> `homopt.learning.adversarial.selection`
- `experiments/adversarial/model.py` -> `homopt.models.adversarial`
- `experiments/adversarial/training.py` -> `homopt.learning.training.adversarial`
- `experiments/adversarial/visualization.py` -> `homopt.viz.adversarial`

Note on convention: new code that does not need the experiment-local name should
import from the owner layer directly. For example `benchmarks/convex.py` imports
reference helpers from `homopt.solvers.references`, not from the
`experiments/common/references.py` seam (this direction is enforced by
`test_reference_solver_helpers_live_under_solvers`).

## Optimizer import convention

Optimizer families consume shared helpers from `homopt.optim.core` in two
different styles today:

- `optim/alm/` uses explicit submodule imports
  (`from homopt.optim.core.constraints import ...`, etc.). This is enforced by
  `test_alm_optimizers_use_explicit_core_helper_imports`.
- `optim/first_order/` (`pgd.py`, `frank_wolfe.py`, `hom_pgd.py`,
  `radial_dual.py`) still uses `from homopt.optim.core import *`, and
  `optim/core/__init__.py` builds `__all__` dynamically from `globals()`.

The explicit-import style is the preferred direction; the first-order family has
not been migrated yet.

## Current Boundary

The normal reproduction path stays in `src/homopt` and the family folders under `scripts/`.
Archived-script loading and root `utils/` wrappers have been removed from the
active API. New experiments should use package-owned workloads and the current
track-based script names.

## Historical docs

Reconstruction-era plan/checklists/logs were archived to:

- `docs/archive/reconstruction/`
