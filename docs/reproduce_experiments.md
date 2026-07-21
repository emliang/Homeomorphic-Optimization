# Reproduce experiments

Current experiments are run from editable Python scripts. JSON configs are
test fixtures only.

## Which Path To Use

1. Open the appropriate `scripts/<family>/run_*.py`.
2. Edit script-local parameters such as `BASE_PARAMS`, `QUICK_OVERRIDES`, or
   instance overrides.
3. Run the script directly.

Do not use package presets for current experiments. Small JSON configs live
under `tests/fixtures/configs/` only to exercise the package config parser in
tests.

## Experiment Map

| Experiment family | Editable script | Package callable |
| --- | --- | --- |
| Hom-PGD, inequality-only convex constraints | `scripts/hom_pgd/run_convex_ineq_compare.py` | `homopt.experiments.hom_pgd:convex_algorithm_comparison` |
| Hom-ALM, equality + inequality convex constraints | `scripts/hom_alm/run_convex_eq_compare.py` | `homopt.experiments.hom_alm:convex_algorithm_comparison` |
| INN-PGD, nonconvex inequality constraints | `scripts/inn_pgd/run_qcqp_inn_compare.py` | `homopt.experiments.inn_pgd:qcqp_inn_experiment` |
| INN-PGD ablation sweep | `scripts/inn_pgd/run_qcqp_inn_ablation.py` | `homopt.experiments.inn_pgd:qcqp_inn_sensitivity_sweep` |
| INN-PGD 2D toy/visualization | `scripts/inn_pgd/run_qcqp_inn_toy.py` | `homopt.experiments.inn_pgd:qcqp_inn_experiment` |
| JCC/OPF INN-PGD | `scripts/inn_pgd/run_jcc_opf_compare.py` | `homopt.experiments.inn_pgd:jcc_problem_benchmark` |
| Convex-parametric predictor + post-processing | `scripts/learning_postprocess/run_convex_parametric_predictor_postprocess.py` | `homopt.experiments.learning_postprocess:convex_parametric_learning_benchmark` |
| AC-OPF predictor + post-processing | `scripts/learning_postprocess/run_acopf_predictor_postprocess.py` | `homopt.experiments.learning_postprocess:acopf_parametric_learning_benchmark` |
| Stiefel constrained subspace experiments | `scripts/hom_alm/run_stiefel_eq_compare.py` | script-local `stiefel_algorithm_comparison` |

## Calling Flow

1. Edit the relevant `scripts/<family>/run_*.py`.
2. Call the package workload directly.
3. Persist through `record_script_run(...)`.
4. Write `config.json`, `result.json`, and optional `artifacts/`.

Editable script runs write to `results/<family>/<experiment_name>/` or the
instance-labeled variant selected by the script.
