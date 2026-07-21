# Documentation Index

This file is the fastest entry point for project notes, experiment docs, and reconstruction logs.

## Active Docs

- [`architecture.md`](architecture.md)
  - Current package structure and module responsibilities.
- [`algorithm_notes.md`](algorithm_notes.md)
  - Short algorithm-side implementation notes for current optimizer and mapping decisions.
- [`gauge_mapping_notes.md`](gauge_mapping_notes.md)
  - Concise input contract, forward map, explicit VJP, and current limits for `GaugeMap`.
- [`solver_notes.md`](solver_notes.md)
  - Active solver ids, inner-solver compatibility, CVXPY subproblem routes, and cleanup status for removed solver names.
- [`experiment_guide.md`](experiment_guide.md)
  - Main experiment routes, run flow, config groups, and extension pattern.
- [`main_experiment_notes.md`](main_experiment_notes.md)
  - Paper-facing formulation and current settings for the two main experiments:
    SOCP-EQ and constrained Stiefel PCA.
- [`experiment_research_notes.md`](experiment_research_notes.md)
  - Literature-backed experiment candidates and formulations for new benchmark families.
- [`reproduce_experiments.md`](reproduce_experiments.md)
  - Reproduction-oriented experiment entry notes.
- [`structure_cleanup_plan.md`](structure_cleanup_plan.md)
  - Historical log of the 2026-04 cleanup rounds (1-7). Kept for "why" context;
    its module inventories predate the current `experiments/benchmarks/` layout.
    See `architecture.md` for the current structure.
- [`visualization_refactor_notes.md`](visualization_refactor_notes.md)
  - Concrete plotting refactor plan, paper-style contract, target visualization module structure, and phase-by-phase validation checklist.

## Recent Notes

- `2026-04-22` Added a normalized gradient descent update backend.
  - Optimizer result: `NormalizedGDOptimizer` is exposed through
    `homopt.optim` and selectable as `opt="normalized_gd"` for first-order
    paths such as `Hom-PGD`.
  - Compatibility result: NAG and staged proximal Hom-PGD still require
    `opt="gd"`; normalized GD is a direction-normalizing update backend, not
    an acceleration method.
  - Smoke result: on one 10D convex Hom-PGD comparison, `normalized_gd` reached
    final objective `-0.378409` with zero violation versus `gd` at `-0.357844`
    with zero violation under the same 80-step budget.
- `2026-04-22` Added dedicated solver notes and rechecked removed solver names.
  - Audit result: active source/config paths no longer use
    `linearized_prox_gd`; remaining mentions are removal documentation, a
    rejection regression test, and historical result artifacts.
  - Documentation result: `solver_notes.md` now records active algorithm ids,
    ALM inner solver options, CVXPY `*-EQ` subproblem routes, and Hom-PGD
    proximal mode.
- `2026-04-22` Documented solver configuration and compatibility matrix.
  - Optimizer result: `algorithm_notes.md` now lists the active
    `run_algorithm(...)` ids, optimizer classes, required `hom_map` relations,
    main optimization variables, subproblem solvers, and compatibility
    constraints for NAG, proximal modes, and ALM inner solvers.
  - Exact-solver result: notes now cover `ConvexSolver`, `MaxCutSolver`,
    `QCQPSolver`, JCC solver facades, `StiefelRetractionSolver`, and the
    supported `solve_type` / option surfaces.
  - Modeling result: `ConvexSolver` modes now have an explicit flow note for
    `opt`, `proj`, `linear`, `ip`, interior-point origin search, and `eq_alm`.
  - Cleanup result: exact-solver option aliasing is target-aware, `eq_alm`
    metadata reports relaxed equalities explicitly, QCQP solve-type and
    radius-ball checks were tightened, and `ChanceConstraintSolver` now accepts
    the old `robust` alias while defaulting to `robust-scenario`.
- `2026-04-22` Cleaned convex equality/non-equality solver comparison settings.
  - Main result: `convex_algorithm_comparison(...)` now reuses shared helpers
    for iterative `ALM` / `Hom-ALM` defaults and exact-subproblem `*-EQ`
    CVXPY baselines.
  - First-order result: `PGD` projection and `FW` linear-oracle subproblems now
    reuse one shared optimizer-default helper, and optimizer
    direct-construction fallbacks mirror that same default structure.
  - Consistency result: default `ALM` and `Hom-ALM` share the same
    `inner_solver="gd"` and inner-budget settings; only the intended x-space
    versus z-space fields differ. First-order subproblems keep their own
    x-space `gd` Lagrangian defaults.
  - CVXPY result: `ConvexSolver.solve_type="eq_alm"` now has a dedicated
    objective-construction helper, and the other convex CVXPY modes use shared
    quadratic/projection/linear objective helpers.
  - Validation: targeted experiment/optimizer/solver tests passed with
    `102 passed`.
- `2026-04-22` Removed `linearized_prox_gd` from active ALM inner solvers.
  - Main result: the linearized proximal step is treated as projected-gradient
    descent when the nonsmooth part is the convex-inequality indicator, so it
    is no longer exposed as a separate `ALM` / `Hom-ALM` inner-solver name.
  - Hom-PGD result: `Hom-PGD` also has an optional staged proximal mode via
    `use_proximal=True`, `proximal_update_iterations`, `proximal_coef`, and
    `proximal_space`; this is a first-order Hom-Prox-GD mode, not an ALM inner
    solver.
  - Experiment result: package-native convex test problems now use
    `inner_iterations=50` for default `ALM` / `Hom-ALM`; `hal_prox_gd` and
    `hal_prox_gd_x` use the same inner budget.
  - Solver result: use `PGD` / `Hom-PGD` for first-order projected-gradient
    comparisons, or the `*-EQ` CVXPY baselines when fixed-outer ALM
    subproblems should keep convex inequalities hard and be solved directly.
    Legacy `prox_gd` remains supported for staged gradient-of-proximal-term
    inner updates.
  - Documentation result: `algorithm_notes.md` records the active solver
    policy and the PGD equivalence.
  - Validation: targeted optimizer/config/solver tests passed with `68 passed`;
    full tests passed with `261 passed`.
- `2026-04-22` Started visualization refactor notes and shared plotting infrastructure.
  - Main result: `visualization_refactor_notes.md` now records the paper figure style contract, target `homopt.viz` module layout, and staged migration plan.
  - Structure target: plotting style, artifact writing, trace normalization, landscape construction, and high-level figure orchestration should be separate modules.
  - Current execution phase: shared style/artifact/trace helpers are being introduced while preserving public visualization APIs.
- `2026-04-21` Updated algorithm structure flow notes.
  - Main result: `algorithm_notes.md` now documents the active execution flow
    from experiment config to `run_algorithm(...)` to optimizer payloads.
  - Optimizer result: PGD, Hom-PGD, FW, RD, ALM, the `*-EQ` CVXPY equality
    baselines, and Hom-ALM now have explicit flow sketches, including the
    fixed-outer-state interpretation of Hom-ALM.
  - Inner-solver result: `ALM` and `Hom-ALM` share the same `gd` / `prox_gd`
    inner-solver controls.
  - Acceleration result: standard NAG is the only active acceleration path;
    FISTA-style and `opt="nesterov"` paths are documented as inactive.
  - Validation: `conda run --no-capture-output -n ML python -m pytest tests`
    passed with `239 passed`.
- `2026-04-19` Added research notes for nonconvex-equality + convex-inequality ML benchmarks.
  - Main result: candidate families are prioritized as Stiefel/orthogonality, Euclidean distance geometry, DE-constrained optimization, AC-OPF-lite, moment-constrained molecules, equivariant training, and constrained diffusion.
  - Formulation result: the target experiment shape is `min f(x; c)` subject to `h(x; c)=0` for nonconvex equalities and `g(x; c)<=0` for convex inequalities, with Hom-ALM handling equalities and gauge/homeomorphic maps handling convex inequalities.
  - Next implementation target: start with `StiefelProblem`, then `DistanceGeometryProblem`, before moving to applied DE/AC-OPF examples.
- `2026-04-14` Unified `ALM / Hom-ALM / JCC` outer-loop penalty config.
  - Main result: one canonical penalty-method config family now owns `outer_iterations`, `inner_iterations`, `dual_learning_rate`, `penalty_coef`, `penalty_growth`, `proximal_coef`, `max_penalty`, and `max_dual`.
  - Optimizer result: `LagrangianOptimizer` and `HomALMOptimizer` now interpret and update these fields consistently, including unified `outer_stepsize_rule` / `inner_stepsize_rule` handling.
  - Naming result: first-order baseline subproblems now use canonical names such as `projection_subproblem` and `linearization_subproblem`; active builders no longer emit short aliases.
  - Stability result: `Hom-ALM` now tracks equality-only violation at the outer layer and no longer applies an extra outer diminish decay on top of `run_penalty_outer_loop`.
  - Validation: superseded by the 2026-04-21 full-suite run.
- `2026-04-13` Integrated raw QCQP NN predictor into the package-native learning route.
  - Main result: legacy-style `train_nn_solver/test_nn_solver` functionality is now represented by `DecisionPredictorNet`, `NeuralDecisionPredictor`, and predictor resource loading in `benchmarks_parametric.py`.
  - Route result: QCQP learning now has one explicit mainline, `predictor -> optional refiner -> shared metrics/artifacts`, instead of keeping raw predictor logic in archived scripts.
  - Config result: `qcqp_learning_benchmark(...)` accepts `predictor_type`, `predictor_model_config`, and `predictor_train_config` in addition to the existing post-process model config groups.
- `2026-04-11` Round 2 structure cleanup completed.
  - Main result: package-owned routes now have clearer ownership boundaries across `problems`, `solvers`, `experiments`, `models`, `mappings`, and `utils`.
  - Experiment result: active scripts now import family facades directly; the old `benchmarks.py` compatibility facade was removed later in the script-only cleanup.
  - Config result: direct parametric/QCQP benchmark calls use canonical `problem_config`, `model_config`, `optimizer_config`, and `train_config`.
  - Validation: `conda run --no-capture-output -n ML python -m pytest tests` passed with `177 passed`.
- `2026-04-11` Round 3 structure cleanup completed.
  - Main result: removed remaining benchmark facade monkeypatch indirection, `viz/mdh.py`, and `optim/core.py` forwarding modules.
  - Compatibility result: `common/__init__.py` is now a narrow compatibility surface rather than a star-export namespace.
  - Test result: reconstruction-era `_migrated` tests were renamed to domain regression names.
  - Parser audit: `_parsers.py` was later removed with the config/preset CLI path during script-only cleanup.
  - Validation: `conda run --no-capture-output -n ML python -m pytest tests` passed with `177 passed`.
- `2026-04-11` Round 4 structure cleanup completed.
  - Main result: deleted `common/` entirely, fixed experiment lazy-export ownership, and completed README/docstring cleanup.
  - Test result: structural boundary checks now target `scripts/experiments/`, and the old reconstruction-boundary test was renamed.
  - Import result: `homopt.optim` exposes `GDOptimizer` and `AdamOptimizer` through the lightweight shared optimizer module.
  - Validation: `conda run --no-capture-output -n ML python -m pytest tests` passed with `177 passed`.
- `2026-04-11` Round 5 final polish completed.
  - Main result: collapsed the redundant `architecture.md` compatibility note, sorted `problems/__init__.py` exports, and kept `optim/shared.py` as the semantic optimizer update owner.
  - Test result: renamed `test_common_sharing.py` to `test_cross_package_exports.py`.
  - Config result: README Config Controls now lists canonical keys before legacy QCQP shim keys.
  - Validation: `conda run --no-capture-output -n ML python -m pytest tests` passed with `177 passed`.
- `2026-04-11` Round 6 consistency cleanup completed.
  - Main result: removed the `solve_exact_solver_result` benchmark forwarding helper and direct-routed benchmark exact-solver calls to `homopt.solvers.solve_exact_result`.
  - Language result: active source docstrings/comments no longer use stale reconstruction-era wording for current package code.
  - Export result: sorted namespace exports in `models`, `solvers`, `utils`, and sorted lazy experiment exports.
  - Validation: targeted benchmark/facade/export tests passed with `38 passed`; full test suite passed with `177 passed`.

## Reconstruction Notes

- [`archive/reconstruction/README.md`](archive/reconstruction/README.md)
  - Reconstruction archive entry point.
- [`archive/reconstruction/2026-03-26_reconstruction_log.md`](archive/reconstruction/2026-03-26_reconstruction_log.md)
  - Main rolling reconstruction log and branch summaries.
- [`archive/reconstruction/reconstruction_plan.md`](archive/reconstruction/reconstruction_plan.md)
  - Reconstruction plan snapshot.
- [`archive/reconstruction/reconstruction_checklist.md`](archive/reconstruction/reconstruction_checklist.md)
  - Reconstruction checklist snapshot.
- [`archive/experiment_docs/README.md`](archive/experiment_docs/README.md)
  - Archived docs merged into `experiment_guide.md`.

## Experiment Artifacts

- [`results/paper_figures`](results/paper_figures)
  - Figures, summaries, and branch experiment outputs.
- [`results/paper_figures/cvxinn_highdim`](results/paper_figures/cvxinn_highdim)
  - High-dimensional `CvxINN` branch results and comparisons.
- [`results/paper_figures/cvxinn_family`](results/paper_figures/cvxinn_family)
  - Convex-family distortion demos and GaugeMap comparisons.

## Notes Retrieval

For quick lookup:

- Use active docs for current structure and supported experiment paths.
- Use the reconstruction log for branch history and “why” context.
- Use `results/paper_figures/*` for concrete experiment outputs.
- When searching from the terminal, prefer `rg "<keyword>" docs results`.
