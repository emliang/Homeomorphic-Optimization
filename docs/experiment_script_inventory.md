# Experiment script inventory

The main editable experiment scripts are organized by formulation under
`scripts/<family>/`. Use the names below directly.

| Category | Active scripts | Notes |
| --- | --- | --- |
| Hom-PGD, inequality-only convex constraints | `scripts/hom_pgd/run_convex_ineq_compare.py`, `scripts/hom_pgd/run_poly_star_compare.py`, `scripts/hom_pgd/run_poly_star_ablation.py`, `scripts/hom_pgd/run_maxcut_sdp_compare.py`, `scripts/hom_pgd/run_adversarial_attack.py` | The poly-star, MaxCut-SDP, and adversarial scripts are Hom-PGD variants on different problem families. |
| Hom-ALM, equality + inequality convex constraints | `scripts/hom_alm/run_convex_eq_compare.py`, `scripts/hom_alm/run_convex_eq_2d.py`, `scripts/hom_alm/run_stiefel_eq_compare.py` | `run_stiefel_eq_compare.py` is the nonconvex Stiefel equality-constrained variant with convex inequalities. |
| INN-PGD, nonconvex inequality constraints | `scripts/inn_pgd/run_qcqp_inn_compare.py`, `scripts/inn_pgd/run_qcqp_inn_ablation.py`, `scripts/inn_pgd/run_qcqp_inn_toy.py`, `scripts/inn_pgd/run_jcc_opf_compare.py` | `run_qcqp_inn_compare.py` is the high-dimensional INN-PGD vs IPOPT comparison. `run_qcqp_inn_toy.py` owns 2D trajectory/mapping plots and 2D ALM/Penalty/Prox-Penalty baselines. `run_qcqp_inn_ablation.py` sweeps network/optimizer parameters. The JCC script owns its OPF solver and penalty baselines. |
| NN predictor + post-processing | `scripts/learning_postprocess/run_qcqp_predictor_postprocess.py` | Predictor/refiner comparison for Homeomorphic Projection style post-processing. |

No `run_legacy_*.py` entrypoints should remain under `scripts/`.
