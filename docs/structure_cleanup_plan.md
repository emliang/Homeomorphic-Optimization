# Structure Cleanup Plan

Status: **Historical log (Rounds 1-7 completed)** | Created: 2026-04-11

> This file is a historical record of the 2026-04 cleanup rounds. The package
> has since been reorganized (the `gauge_eq/`, `gauge_ineq/`, and
> `inn_parametric/` packages and the `benchmarks_single.py` /
> `benchmarks_parametric.py` modules referenced in the inventories below no
> longer exist; experiment workloads now live under `experiments/benchmarks/`
> with `common/`, `method_specs/`, and `adversarial/`). For the **current**
> package structure and module responsibilities, see
> [`architecture.md`](architecture.md), which is the single source of truth.
> The round-by-round notes below are kept for "why" context only.

## Round 1 Outcome

- Archived `Homeomorphic_Projection/`, `backup/`, `tmp/`, legacy results.
- Renamed `vis/` -> `notebooks/`, `scripts/legacy_runs/` -> `scripts/experiments/`.
- Flattened `configs/` into `smoke/` + `experiments/`.
- Merged experiment docs into `experiment_guide.md`.

## Round 2 Outcome

- Deleted `__pycache__/`, added `.gitignore`.
- Flattened `public.py` + `_impl.py` in `mappings/`, `models/`, `problems/`.
- Reduced `common/` to a single compat `__init__.py`.
- Removed `legacy.py`, root `utils/`, `_helpers.py`/`_domain_helpers.py` forwarding layers.
- Removed solver re-exports from `problems/__init__.py`.
- Simplified `benchmarks.py` facade to `**kwargs` forwarding.
- Unified config keys to canonical names.
- Validation: `177 passed`.

## Round 3: Remaining Internal Issues (2026-04-11)

### Round 3 Outcome

- Removed `_legacy_benchmarks_module()` and related `_resolve_*` monkeypatch indirection from `benchmarks_single.py` and `benchmarks_parametric.py`.
- Updated benchmark workflow tests to monkeypatch the owning modules (`benchmarks_single`, `benchmarks_parametric`) instead of the facade for solver/callable dependencies.
- Deleted `viz/mdh.py`; `viz/__init__.py` imports directly from `_mdh_impl.py`.
- Reduced `common/__init__.py` to the compatibility symbols still tested: base classes, `run_and_record`, and lazy `GDOptimizer`.
- Removed duplicate `__all__` entries in `common/__init__.py`.
- Deleted `optim/core.py`; `homopt.optim.__init__` now lazy-loads optimizer symbols directly from `_core_impl.py`.
- Kept `_parsers.py` because it provides the shared CLI parser builder used by both `cli.py` and `main.py`.
- Renamed reconstruction-era `_migrated` tests to domain regression names: `test_inn.py`, `test_jcc.py`, `test_mappings.py`, `test_optimizers.py`, `test_problems.py`, and `test_qcqp.py`.
- Updated `architecture.md` for the current experiment helper modules and removed `legacy.py` references.
- Validation: full test suite passed: `177 passed`.

### Current inventory

```
src/homopt/       86 .py   (core package)
  common/          1 .py   (compat __init__.py only)
  experiments/    23 .py
  gauge_eq/        5 .py
  gauge_ineq/      5 .py
  inn_parametric/  5 .py
  learning/        4 .py
  mappings/        4 .py
  models/          7 .py   (incl. flows/)
  optim/          10 .py
  problems/        5 .py
  solvers/         3 .py
  utils/           7 .py
  viz/             4 .py
  (root)           3 .py   (__init__, _version, sampling)
tests/            33 .py
scripts/          15 .py
configs/          21 .json
```

### Issue K: `_legacy_benchmarks_module()` monkeypatch indirection

Both `benchmarks_single.py` and `benchmarks_parametric.py` define a
`_legacy_benchmarks_module()` function that does `sys.modules.get("homopt.experiments.benchmarks")`.
This is used by `_resolve_jcc_problem_builder`, `_resolve_jcc_solver_classes`,
`_resolve_qcqp_solver_factory`, `_resolve_qcqp_iterative_callable`, and
`_resolve_qcqp_learning_callable` to allow the `benchmarks.py` facade to
monkeypatch the solver/callable resolution at runtime.

This pattern adds 6 indirection functions across two files. It exists solely so
tests can monkeypatch `benchmarks.run_algorithm` and `benchmarks.QCQPSolver` on
the facade module. Now that the facade uses `**kwargs` forwarding and the
`_forward_single_benchmark` helper already patches `_single.run_algorithm`, the
remaining `_resolve_*` indirections in the owning modules may no longer be
necessary.

**Recommendation**: Audit which tests actually monkeypatch through
`benchmarks.*`. If the only monkeypatch is `benchmarks.run_algorithm` (already
handled by `_forward_single_benchmark`), remove all `_legacy_benchmarks_module`
and `_resolve_*` indirections. Have the owning modules call their dependencies
directly.

**Resolution**: Removed these indirections. Tests that need solver/callable
fakes now patch the owning modules directly; facade monkeypatching remains only
for `benchmarks.run_algorithm`.

### Issue L: `viz/mdh.py` is still a forwarding module

`viz/mdh.py` only re-exports 4 functions from `viz/_mdh_impl.py`.
`viz/__init__.py` imports from `viz/mdh.py`, which imports from `viz/_mdh_impl.py`.

**Recommendation**: Have `viz/__init__.py` import directly from `_mdh_impl.py`
and delete `mdh.py`. Same pattern as how `models/__init__.py` now imports
directly from `_inn_impl.py`.

**Resolution**: Completed.

### Issue M: `common/__init__.py` still re-exports everything

`common/__init__.py` still eagerly imports `BaseProblem`, `BaseMap`,
`BaseOptimizer`, `BasePredictor`, `BaseRefiner`, `ExperimentResult`,
`run_and_record`, etc. plus `from homopt.utils import *`, plus 2 lazy symbols
(`GDOptimizer`, `run_algorithm`). Latent-domain sampling now lives under
`homopt.models.sampling`, not the package root.

Only 2 test files (`test_subproject_layout.py`, `test_dependency_boundaries.py`)
import from `homopt.common`. No `src/` or `scripts/` code does.

**Recommendation**: Reduce `common/__init__.py` to only what those 2 test files
actually need (base classes + `run_and_record` + `GDOptimizer`). Remove the
star-imports from `utils` which leak internals. Or update the 2 test files and
delete `common/` entirely.

**Resolution**: Reduced to the tested compatibility surface: base classes,
`run_and_record`, and lazy `GDOptimizer`.

### Issue N: `__all__` duplicates in `common/__init__.py`

`ParametricProblemBase` and `StateBackedParametricProblemBase` each appear twice
in `common/__init__.py`'s `__all__` list.

**Recommendation**: Deduplicate.

**Resolution**: Completed by reducing `common/__init__.py`.

### Issue O: Test naming inconsistency

Tests use mixed naming:
- `_migrated` suffix (6 files): `test_inn_migrated.py`, `test_jcc_migrated.py`,
  `test_mappings_migrated.py`, `test_optimizers_migrated.py`,
  `test_problems_migrated.py`, `test_qcqp_migrated.py`
- `_contract` suffix: `test_optimizer_contract.py`, `test_problem_contract.py`
- `_split` suffix: `test_optimizer_split.py`, `test_benchmark_module_split.py`
- `_boundaries` suffix: `test_dependency_boundaries.py`,
  `test_reconstruction_boundaries.py`

The `_migrated` tests were from the reconstruction phase. Now that
reconstruction is complete, they are just regular regression tests.

**Recommendation**: Rename `_migrated` tests to drop that suffix or group by
domain (e.g., `test_inn.py`, `test_jcc.py`, `test_mappings.py`).

**Resolution**: Renamed the six reconstruction-era files to domain regression
names.

### Issue P: `_parsers.py` in experiments

`_parsers.py` exists under `experiments/` alongside `cli.py` and `main.py`.
Check if the parser logic should live inside `cli.py` and `main.py` directly
or if `_parsers.py` provides shared parser building used by both.

**Recommendation**: If `_parsers.py` only defines `build_parser` functions used
by exactly one caller each, inline them. If shared across `cli.py` and
`main.py`, keep it.

**Resolution**: Kept `_parsers.py` because the single helper is shared by both
`cli.py` and `main.py`.

### Issue Q: `optim/core.py` vs `optim/_core_impl.py`

`optim/` has both `core.py` and `_core_impl.py`. The `__init__.py` lazy-exports
point to `optim.core`. Check if `core.py` re-exports from `_core_impl.py`
(same pattern as old `models/inn.py` -> `_inn_impl.py`).

**Recommendation**: If `core.py` just forwards, have `__init__.py` import from
`_core_impl.py` directly and delete `core.py`. If `core.py` has its own logic,
keep both but document the split.

**Resolution**: Deleted `optim/core.py`; `homopt.optim` lazy exports now target
`_core_impl.py` directly.

### Issue R: Documentation staleness

`architecture.md` previously listed the retired legacy preset shim in the active
flow and has since been updated to the current package-owned experiment layout.

**Recommendation**: Update `architecture.md` to reflect the current experiment
file list and the removal of `legacy.py`.

**Resolution**: Completed.

## Priority for Round 3

| Priority | Action | Effort |
|---|---|---|
| **P0** | Fix `__all__` duplicates in `common/__init__.py` | 1 min |
| **P1** | Remove `_legacy_benchmarks_module()` indirection if unused | Medium |
| **P1** | Delete `viz/mdh.py` forwarding module | Small |
| **P1** | Reduce or delete `common/` entirely | Small |
| **P2** | Rename `_migrated` test files | Small |
| **P2** | Audit `_parsers.py` and `optim/core.py` forwarding | Small |
| **P2** | Update `architecture.md` to match current state | Small |

## Round 4: Remaining Issues (2026-04-11)

### Round 4 Outcome

- Rewrote `homopt/__init__.py` docstring to remove reconstruction-era `utils/` language.
- Updated `README.md` intro, current status, project/package structure, and reading order to match the active package layout.
- Pointed experiment registry lazy exports to `_benchmark_registry.py`.
- Routed `poly_star_benchmark` and `socp_hompgd_benchmark` top-level lazy exports through the `benchmarks.py` facade so their monkeypatch behavior is consistent across import paths.
- Added `qcqp_inn_experiment` and `qcqp_inn_sensitivity_sweep` to `benchmarks.py.__all__`.
- Deleted `common/` entirely after updating tests to import from canonical owner modules.
- Updated dependency-boundary tests to validate `homopt.optim` lazy imports; `GDOptimizer` and `AdamOptimizer` now lazy-load from `homopt.optim.shared`.
- Updated the script boundary test to inspect `scripts/experiments/` instead of deleted `scripts/legacy_runs/`.
- Renamed `test_reconstruction_boundaries.py` to `test_structural_boundaries.py`.
- Validation: full test suite passed: `177 passed`.

### Current inventory

```
src/homopt/       85 .py   (core package)
  experiments/    23 .py
  gauge_eq/        5 .py
  gauge_ineq/      5 .py
  inn_parametric/  5 .py
  learning/        4 .py
  mappings/        4 .py   (gauge.py, star.py are impl files now)
  models/          7 .py   (incl. flows/)
  optim/          10 .py   (core.py deleted)
  problems/        5 .py
  solvers/         3 .py
  utils/           7 .py
  viz/             4 .py   (mdh.py deleted)
  (root)           3 .py   (__init__, _version, sampling)
tests/            33 .py
scripts/          15 .py
configs/          21 .json
```

### Issue S: `homopt/__init__.py` docstring is stale

The docstring still says "first reconstruction phase keeps behavior compatible
with the legacy `utils/` modules." Root `utils/` was removed in Round 2 and
reconstruction is complete.

**Recommendation**: Rewrite to a clean one-liner describing the package.

**Resolution**: Completed.

### Issue T: README.md is significantly outdated

1. Line 14: "Shared abstractions and cross-track exports live under
   `homopt.common`." — `common/` is now a minimal compat layer with 0 src
   consumers, not the main export surface.
2. Line 18: "legacy `utils/` modules are narrow compatibility wrappers" — root
   `utils/` was deleted in Round 2.
3. Lines 24–52: Project Structure tree still lists `utils/` at repo root; it no
   longer exists. Latent-domain sampling belongs under `models/sampling.py`.
4. Line 402: Duplicate `architecture.md` link in "What To Read First".
5. Several paragraphs still reference "reconstruction phase" language.

**Recommendation**: Rewrite the intro and Project/Package Structure sections
to reflect the current state. Remove reconstruction-era language.

**Resolution**: Completed.

### Issue U: `__init__.py` lazy-export source inconsistency

`experiments/__init__.py` lazy exports point to three different sources for the
three registry constants, all of which are actually defined in
`_benchmark_registry.py`:

- `ALL_MAINLINE_BENCHMARKS` → `homopt.experiments.benchmarks` (facade)
- `PARAMETRIC_BENCHMARKS` → `homopt.experiments.benchmarks_parametric`
- `SINGLE_PROBLEM_BENCHMARKS` → `homopt.experiments.benchmarks_single`

**Recommendation**: Point all three to `homopt.experiments._benchmark_registry`
for consistency and to reduce unnecessary module loading.

**Resolution**: Completed.

### Issue V: Lazy exports for `poly_star_benchmark` / `socp_hompgd_benchmark` bypass facade wrapper

`experiments/__init__.py` lazy-exports `poly_star_benchmark` and
`socp_hompgd_benchmark` from `benchmarks_single` directly. But in the facade
(`benchmarks.py`), these two have a `_forward_single_benchmark` wrapper that
enables test monkeypatching of `benchmarks.run_algorithm`.

`homopt.experiments.poly_star_benchmark` → raw unwrapped function
`homopt.experiments.benchmarks.poly_star_benchmark` → wrapped version

**Recommendation**: Point the lazy exports for these two to
`homopt.experiments.benchmarks` (the facade) so behavior is consistent
regardless of import path. All other benchmark callables already use
direct-import semantics where wrapping is not needed.

**Resolution**: Completed.

### Issue W: Facade `__all__` is incomplete

`benchmarks.py` imports `qcqp_inn_experiment` and `qcqp_inn_sensitivity_sweep`
from `benchmarks_parametric` but does not list them in `__all__`. Scripts under
`scripts/experiments/` import these from the facade.

**Recommendation**: Add the missing names to `benchmarks.py.__all__`.

**Resolution**: Completed.

### Issue X: `common/` can be deleted

`common/` has exactly 1 file and 0 `src/` or `scripts/` consumers. Only 2 test
files use it:
- `test_subproject_layout.py` (6 imports from `homopt.common`)
- `test_dependency_boundaries.py` (subprocess tests of `import homopt.common`
  and `from homopt.common import GDOptimizer`)

These can be trivially rewritten to use canonical import paths. The
`test_dependency_boundaries.py` tests verify lazy-import discipline; they can
test `homopt.optim` / `homopt.problems` directly.

**Recommendation**: Rewrite the 2 test files, then delete `common/` entirely.
Update `README.md` to remove `common/` references.

**Resolution**: Completed.

### Issue Y: `test_reconstruction_boundaries.py` references deleted directory

Line 27: `scripts_root = REPO_ROOT / "scripts" / "legacy_runs"` — this
directory was renamed to `scripts/experiments/` in Round 1. The test silently
passes (no files = no offenders) but doesn't validate anything.

**Recommendation**: Update the path to `scripts/experiments/`, or remove the
test if the boundary check is no longer needed.

**Resolution**: Completed.

### Issue Z: `test_reconstruction_boundaries.py` naming

File is still named with "reconstruction" prefix. It's now a general
structural boundary test, not reconstruction-specific.

**Recommendation**: Rename to `test_structural_boundaries.py`.

**Resolution**: Completed.

## Priority for Round 4

| Priority | Issue | Action | Effort |
|---|---|---|---|
| **P0** | S | Fix `homopt/__init__.py` docstring | 1 min |
| **P0** | W | Add missing names to `benchmarks.py.__all__` | 1 min |
| **P1** | T | Rewrite README.md intro/structure sections | Medium |
| **P1** | U | Consistent `_benchmark_registry` lazy-export source | Small |
| **P1** | V | Route `poly_star`/`socp_hompgd` lazy exports through facade | Small |
| **P1** | X | Delete `common/` entirely after updating 2 tests | Small |
| **P2** | Y | Fix or remove stale directory reference in boundary test | Small |
| **P2** | Z | Rename `test_reconstruction_boundaries.py` | Small |

## Round 5: Final Audit (2026-04-11)

After four rounds of cleanup the package was already in clean,
production-quality shape. Round 5 completed the remaining optional polish items.

### Round 5 Outcome

- Collapsed the redundant compatibility-boundary removal note in `architecture.md`.
- Sorted `problems/__init__.py` lazy-export mapping and `__all__`.
- Renamed `test_common_sharing.py` to `test_cross_package_exports.py`.
- Updated README Config Controls to show canonical config keys before legacy shim keys.
- Kept `optim/shared.py` as a semantic lightweight re-export module.
- Validation: full test suite passed: `177 passed`.

### Audit summary

| Area | Verdict |
|---|---|
| `common/` | Deleted. No stale imports. |
| Root `utils/` | Deleted. No stale imports. |
| `__pycache__` | None on disk. `.gitignore` covers it. |
| Forwarding modules | None remain (`core.py`, `mdh.py`, `inn.py`, `_helpers.py` all removed). |
| `__init__.py` lazy exports | Consistent: registry → `_benchmark_registry`; facade-wrapped → `benchmarks`; others → owning module. |
| Facade `benchmarks.py` | Thin; `__all__` is complete; only two `_forward_single_benchmark` wrappers. |
| Test naming | Domain-based (`test_inn.py`, etc.); structural boundary test renamed. |
| Documentation | `README.md`, `architecture.md`, `experiment_guide.md`, `INDEX.md` all aligned with current layout. |
| Subproject isolation | `gauge_eq`, `gauge_ineq`, `inn_parametric` have zero cross-imports and zero imports from `experiments/`. |
| Config schema | Canonical keys documented in `experiment_guide.md`. |
| Test coverage | 177 passed across 33 test files. |

### Issue AA (P2): `architecture.md` redundancy

Lines 67–68 and 74–75 both state that root `utils/` and `experiments.legacy`
were removed. One mention is sufficient.

**Recommendation**: Collapse the "Compatibility boundary" section into a single
paragraph.

**Resolution**: Completed.

### Issue AB (P2): `problems/__init__.py` unsorted `__all__`

`_SYMBOL_TO_TARGET` and `__all__` entries are not alphabetically sorted,
unlike `optim/__init__.py` which is sorted. Consistency improves auditability.

**Recommendation**: Sort `_SYMBOL_TO_TARGET` keys and `__all__` entries.

**Resolution**: Completed.

### Issue AC (P2): `test_common_sharing.py` name is misleading

Now that `common/` is deleted, the test file name suggests it tests a
`common` package. It actually validates that cross-package exports are
accessible from their canonical owner modules.

**Recommendation**: Rename to `test_cross_package_exports.py`.

**Resolution**: Completed.

### Issue AD (P2): README config controls are now canonical

The README and experiment guide now use `problem_config`, `model_config`,
`optimizer_config`, `train_config`, `solver_configs`, and `algorithm_config`
as the active config surface.

**Resolution**: Completed.

### Issue AE (P2): `optim/shared.py` is a 2-line re-export

`optim/shared.py` only re-exports `AdamOptimizer` and `GDOptimizer` from
`optim/_updates.py`. The `optim/__init__.py` could point directly to
`_updates` for these two symbols. However, `shared.py` provides a clean
semantic grouping name, so this is a judgment call.

**Recommendation**: Keep as-is; the indirection cost is negligible and the name
adds clarity.

**Resolution**: Kept as-is.

## Priority for Round 5

| Priority | Issue | Action | Effort |
|---|---|---|---|
| **P2** | AA | Collapse redundant removal note in `architecture.md` | 1 min |
| **P2** | AB | Sort `problems/__init__.py` `__all__` | 1 min |
| **P2** | AC | Rename `test_common_sharing.py` | 1 min |
| **P2** | AD | Add canonical keys note to README Config Controls | Small |
| **—** | AE | Keep `optim/shared.py` as-is | None |

**Overall verdict (Round 5)**: The package structure is clean, well-documented,
and free of structural redundancy. The cleanup plan can be considered
**complete**.

## Round 6: Consistency & Language Audit (2026-04-11)

Round 6 focused on internal consistency across all namespace `__init__.py`
files, stale reconstruction-era language in source docstrings, and a remaining
1-line forwarding function.

### Round 6 Outcome

- Removed `solve_exact_solver_result`; benchmark modules now call `homopt.solvers.solve_exact_result` directly.
- Replaced stale reconstruction-era language in active source docstrings/comments with package-owned/current wording.
- Sorted `models/__init__.py`, `solvers/__init__.py`, and `utils/__init__.py` exports.
- Sorted `experiments/__init__.py` `_LAZY_EXPORTS`.
- Reduced extra blank lines in `experiments/benchmarks.py`.
- Kept the tiny duplicate resolver in `config.py` / `schema.py` to avoid adding another shared module.
- Validation: targeted benchmark/facade/export tests passed: `38 passed`; full test suite passed: `177 passed`.

### Current inventory

```
src/homopt/       85 .py   (core package)
  experiments/    23 .py
  gauge_eq/        5 .py
  gauge_ineq/      5 .py
  inn_parametric/  5 .py
  learning/        4 .py
  mappings/        4 .py
  models/          7 .py   (incl. flows/)
  optim/          10 .py
  problems/        5 .py
  solvers/         3 .py
  utils/           7 .py
  viz/             4 .py
  (root)           3 .py   (__init__, _version, sampling)
tests/            33 .py
scripts/          15 .py
configs/          21 .json
```

### Issue AF (P1): `solve_exact_solver_result` is a trivial forwarding function

`_benchmark_common.py` line 51 defines a 1-line wrapper that forwards to
`solve_exact_result` from `homopt.solvers`. Used in 6 call sites across
`benchmarks_single.py`, `benchmarks_parametric.py`, and `_benchmark_common.py`.

**Recommendation**: Replace all call sites with direct `solve_exact_result`
imports, then delete the wrapper.

**Resolution**: Completed.

### Issue AG (P2): Stale "reconstructed" / "reconstruction" language

7 source files still use reconstruction-era language in docstrings or comments:

- `utils/__init__.py:1` — "reconstructed package"
- `catalog.py:4` — "reconstruction surface"
- `main.py:1` — "reconstructed experiment families"
- `adversarial.py:1,254,1306` — "reconstructed from legacy 1.3" etc.
- `benchmarks_single.py:962,965` — "reconstructed problem and optimizer stack"
- `models/_inn_impl.py:51` — "reconstructed layers"

**Recommendation**: Replace "reconstructed" with neutral descriptions
("package-owned", "current", or simply remove the qualifier).

**Resolution**: Completed for active source files.

### Issue AH (P2): `models/__init__.py` `__all__` not alphabetically sorted

The `__all__` list follows `_inn_impl.py` import order, not alphabetical.
Inconsistent with `problems/`, `optim/`, `learning/`, `mappings/`, `viz/`
which are all sorted.

**Recommendation**: Sort `__all__` alphabetically.

**Resolution**: Completed.

### Issue AI (P2): `solvers/__init__.py` exports not alphabetically sorted

`_SYMBOL_TO_TARGET` and `__all__` are in insertion order, not sorted.

**Recommendation**: Sort both.

**Resolution**: Completed.

### Issue AJ (P2): `utils/__init__.py` `__all__` not alphabetically sorted

**Recommendation**: Sort.

**Resolution**: Completed.

### Issue AK (P2): `experiments/__init__.py` `_LAZY_EXPORTS` keys not sorted

Dict keys are partially sorted with `maxcut_algorithm_comparison`,
`learning_route_summary`, and `toy_star_summary` out of order. The `__all__`
output is correct (uses `sorted()`), but the source dict should match.

**Recommendation**: Sort `_LAZY_EXPORTS` keys alphabetically.

**Resolution**: Completed.

### Issue AL (P3): `benchmarks.py` extra blank lines

4 blank lines (48–51) before `__all__`; PEP 8 allows at most 2.

**Recommendation**: Reduce to 2.

**Resolution**: Completed.

### Issue AM (P3): `config.py` / `schema.py` duplicate resolver

Both define identical `resolve_experiment_callable` (4 lines). Exists to avoid
circular import (`config.py` imports from `schema.py`).

**Recommendation**: Keep as-is; the cost of a shared module exceeds the 4-line
saving.

**Resolution**: Kept as-is.

## Priority for Round 6

| Priority | Issue | Action | Effort |
|---|---|---|---|
| **P1** | AF | Replace `solve_exact_solver_result` with `solve_exact_result` | Small |
| **P2** | AG | Remove stale "reconstructed" language from 7 files | Small |
| **P2** | AH | Sort `models/__init__.py` `__all__` | 1 min |
| **P2** | AI | Sort `solvers/__init__.py` exports | 1 min |
| **P2** | AJ | Sort `utils/__init__.py` `__all__` | 1 min |
| **P2** | AK | Sort `experiments/__init__.py` `_LAZY_EXPORTS` | 1 min |
| **P3** | AL | Reduce blank lines in `benchmarks.py` | 1 min |
| **—** | AM | Keep `config.py`/`schema.py` duplication as-is | None |

## Round 7 Outcome

- Renamed the equality-constrained convex optimization class from
  `ConvexQCQP` to `ConvexOptEq`.
- Removed the old `ConvexQCQP` package export instead of keeping a compatibility
  alias.
- Updated the equality-gauge namespace to export `ConvexOptEq`.
- Updated the single-problem benchmark builder so `problem_type="socp_eq"`
  constructs `ConvexOptEq`; `problem_type="eq"` is rejected and the
  parametric/nonconvex `qcqp_*` route remains unchanged.
- Updated active tests to use the new name. Historical archive notes may still
  mention `ConvexQCQP`, but active `src/`, `tests/`, README, and non-archive
  docs do not.

### Issue AN (P1): Misleading equality problem name

`ConvexQCQP` suggested a QCQP-specific problem class, but the implementation is
really `ConvexOpt` plus optional linear equality constraints:

```
ConvexOpt   = convex objective + box/linear/quadratic/SOC inequalities
ConvexOptEq = ConvexOpt + A_eq x = b_eq
```

**Resolution**: Completed without compatibility alias.

## Constraints

- All test suites must pass after each step
- No changes to the external experiment reproduction surface
- Structural moves should be small, tested commits
