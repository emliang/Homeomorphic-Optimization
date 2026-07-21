# Experiment Scripts

Editable experiment entrypoints are grouped by method family:

- `hom_pgd/`: Hom-PGD experiments for inequality-only constraints.
- `hom_alm/`: Hom-ALM experiments for equality plus inequality constraints.
- `inn_pgd/`: INN-PGD experiments and shared INN-PGD configs.
- `learning_postprocess/`: neural predictor plus post-processing experiments.

Shared script helpers live in `scripts/_common.py`.
Family-level editable configs live in each family's `_configs.py`.
