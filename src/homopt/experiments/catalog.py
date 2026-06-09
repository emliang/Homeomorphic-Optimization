"""Central catalog for script-only experiment runs."""

from __future__ import annotations

SCRIPT_EXPERIMENTS = {
    "adversarial_attack": "scripts/hom_pgd/run_adversarial_attack.py",
    "convex_eq_2d": "scripts/hom_alm/run_convex_eq_2d.py",
    "convex_eq_compare": "scripts/hom_alm/run_convex_eq_compare.py",
    "convex_ineq_compare": "scripts/hom_pgd/run_convex_ineq_compare.py",
    "jcc_opf_compare": "scripts/inn_pgd/run_jcc_opf_compare.py",
    "maxcut_sdp_compare": "scripts/hom_pgd/run_maxcut_sdp_compare.py",
    "poly_star_ablation": "scripts/hom_pgd/run_poly_star_ablation.py",
    "poly_star_compare": "scripts/hom_pgd/run_poly_star_compare.py",
    "qcqp_inn_ablation": "scripts/inn_pgd/run_qcqp_inn_ablation.py",
    "qcqp_inn_compare": "scripts/inn_pgd/run_qcqp_inn_compare.py",
    "qcqp_inn_toy": "scripts/inn_pgd/run_qcqp_inn_toy.py",
    "qcqp_predictor_postprocess": "scripts/learning_postprocess/run_qcqp_predictor_postprocess.py",
    "stiefel_eq_compare": "scripts/hom_alm/run_stiefel_eq_compare.py",
}

EXPERIMENT_FAMILIES = {
    "hom_pgd": {
        "description": "Hom-PGD family for inequality-only convex constraints.",
        "scripts": [
            "convex_ineq_compare",
            "poly_star_compare",
            "poly_star_ablation",
            "maxcut_sdp_compare",
            "adversarial_attack",
        ],
    },
    "hom_alm": {
        "description": "Hom-ALM family for equality plus inequality constraints.",
        "scripts": [
            "convex_eq_compare",
            "convex_eq_2d",
            "stiefel_eq_compare",
        ],
    },
    "inn_pgd": {
        "description": "INN-PGD family for nonconvex objectives with learned inequality mappings.",
        "scripts": [
            "qcqp_inn_compare",
            "qcqp_inn_ablation",
            "qcqp_inn_toy",
            "jcc_opf_compare",
        ],
    },
    "learning_postprocess": {
        "description": "Neural predictor plus post-processing baselines.",
        "scripts": [
            "qcqp_predictor_postprocess",
        ],
    },
}


__all__ = [
    "EXPERIMENT_FAMILIES",
    "SCRIPT_EXPERIMENTS",
]
