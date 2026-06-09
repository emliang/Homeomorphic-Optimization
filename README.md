# Homeomorphic Optimization

[![Package](https://img.shields.io/badge/package-homopt-blue)](#installation)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](#requirements)
[![Focus](https://img.shields.io/badge/focus-Hom--PGD-green)](#hom-pgd)
[![Status](https://img.shields.io/badge/status-research%20package-orange)](#status)

<p align="center">
  <img src="Webpage/Hom.png" alt="Homeomorphic transformation intuition" width="900">
</p>

Homeomorphic Optimization is a research package and user guide for constrained
optimization with homeomorphic maps. The current public-facing focus is
**Hom-PGD**: a projected-gradient-style method that optimizes in a latent space
while a homeomorphic or gauge map keeps decision variables inside the target
inequality-constrained feasible set.

The package is designed for users who want to understand the method, run the
included experiments, and adapt the implementation to new inequality-constrained
optimization problems.

## Latest Documentation

Start with these files:

- `README.md`: high-level package overview and Hom-PGD user guide.
- `docs/architecture.md`: package structure and module responsibilities.
- `docs/algorithm_notes.md`: active algorithm IDs and implementation notes.
- `docs/reproduce_experiments.md`: current experiment entrypoints.
- `docs/experiment_guide.md`: how experiment scripts are organized.

## Supported Methods

The current README documents **Hom-PGD** in detail.

| Method | Status in This README | Purpose |
| --- | --- | --- |
| `Hom-PGD` | Detailed | Inequality-constrained optimization through a latent-to-feasible map. |
| `PGD` | Baseline | Project in decision space after gradient steps. |
| `FW` | Baseline | Use a linearized subproblem for feasible descent. |
| `RD` | Baseline | Radial-dual style comparison route. |
| `ALM` | Baseline in selected comparisons | Penalty/dual baseline for constrained optimization. |
| `Hom-ALM`, `INN-PGD`, learning post-processing | Not covered here yet | Present in the package, but intentionally omitted from this user page for now. |

## Hom-PGD

Hom-PGD targets problems of the form:

$$
\begin{aligned}
\min_{x} \quad & f(x) \\
\text{s.t.} \quad & x \in \mathcal{C}.
\end{aligned}
$$

where $\mathcal{C}$ is an inequality-defined feasible set that can be represented by a
homeomorphic, gauge, star-shaped, or related latent-to-decision map.

Instead of iterating directly on $x$, Hom-PGD introduces a latent variable $z$
and a map:

$$
x = \psi(z).
$$

The optimizer updates $z$, evaluates the objective through $x = \psi(z)$, and
records the resulting feasible decision trajectory.

<p align="center">
  <img src="Webpage/Hom-PGD_framework.png" alt="PGD and Hom-PGD comparison framework" width="900">
</p>

The figure contrasts the two viewpoints. PGD works in the original decision
space and must handle the feasible set $\mathcal{K}$ directly. Hom-PGD uses a
homeomorphic mapping $\psi$ to move between the original decision variable $x$
and a latent variable $z$, turning the objective into

$$
h(z) = f(\psi(z))
$$

and the feasible region into a simpler latent set

$$
\mathcal{B} = \psi^{-1}(\mathcal{K}).
$$

### Why Use Hom-PGD?

Classical projected gradient methods follow this pattern:

$$
x_{k+1} =
\Pi_{\mathcal{C}}\!\left(x_k - \eta \nabla f(x_k)\right).
$$

Hom-PGD changes the pattern:

$$
\begin{aligned}
z_{k+1} &= z_k - \eta \nabla_z f(\psi(z_k)), \\
x_{k+1} &= \psi(z_{k+1}).
\end{aligned}
$$

This is useful when repeated projection is expensive, unstable, or less natural
than parameterizing the feasible geometry directly.

### What the Map Does

The map $\psi$ is the main modeling object. In the current package it can encode
several inequality geometries used by the Hom-PGD experiments:

- convex SOCP-style feasible regions through gauge maps;
- 2D polytope and star-shaped toy feasible sets;
- MaxCut SDP-style comparison routes;
- adversarial-attack feasible balls and related norm constraints.

For a user, the important mental model is:

```text
problem:      defines f(x) and constraints
map:          converts latent z into feasible or geometry-aware x
optimizer:    updates z and records objective/violation traces
experiment:   compares Hom-PGD against baselines under the same budget
```

### Hom-PGD Configuration

The main editable Hom-PGD SOCP script is:

```text
scripts/hom_pgd/run_convex_ineq_compare.py
```

It compares:

```python
["PGD", "FW", "ALM", "RD", "Hom-PGD"]
```

The Hom-PGD-specific default block is:

```python
"Hom-PGD": {
    "learning_rate": 1e-3,
    "hom_p_norm": 2,
    "momentum": 0.99,
}
```

Common controls include:

| Parameter | Meaning |
| --- | --- |
| `learning_rate` | Latent-space step size. |
| `stepsize_rule` | Step-size policy, such as `adaptive`, `constant`, or `diminish`. |
| `hom_p_norm` | Norm used by the latent ball projection route. |
| `momentum` | Momentum for the update backend when enabled. |
| `max_iterations` | Maximum optimization iterations. |
| `max_running_time` | Wall-clock budget for an experiment run. |
| `visualize` | Whether to write plots and comparison artifacts. |
| `visualize_only` | Reuse stored artifacts without rerunning the experiment. |

## Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/<user-or-org>/Homeomorphic-Optimization.git
cd Homeomorphic-Optimization
pip install -e .
```

For development and experiment workflows:

```bash
pip install -e .[dev,research,solvers,viz,models]
```

## Requirements

Base package:

- Python >= 3.7
- NumPy
- PyTorch

Optional experiment groups:

- `dev`: pytest
- `research`: SciPy, NetworkX, power-system helpers
- `solvers`: CVXPY and Pyomo routes
- `viz`: Matplotlib, pandas, seaborn
- `models`: torchvision and model-adjacent dependencies

The local research environment used by this project is named `ML`:

```bash
conda run --no-capture-output -n ML python -m pytest -q tests
```

## Quick Start

Run the main Hom-PGD inequality comparison:

```bash
python scripts/hom_pgd/run_convex_ineq_compare.py
```

Run the 2D poly/star-shaped feasible-set comparison:

```bash
python scripts/hom_pgd/run_poly_star_compare.py
```

Run the MaxCut SDP Hom-PGD comparison:

```bash
python scripts/hom_pgd/run_maxcut_sdp_compare.py
```

Run tests:

```bash
python -m pytest -q tests
```

## Hom-PGD Experiment Scripts

| Script | Purpose |
| --- | --- |
| `scripts/hom_pgd/run_convex_ineq_compare.py` | Main convex inequality/SOCP comparison. |
| `scripts/hom_pgd/run_poly_star_compare.py` | 2D polytope, star-shaped, or intersection toy set. |
| `scripts/hom_pgd/run_poly_star_ablation.py` | Sensitivity checks for the poly/star route. |
| `scripts/hom_pgd/run_maxcut_sdp_compare.py` | MaxCut SDP-style Hom-PGD comparison. |
| `scripts/hom_pgd/run_adversarial_attack.py` | Norm-constrained adversarial-attack workflow. |

Each script is intentionally editable. The intended workflow is:

1. Open the relevant `scripts/hom_pgd/run_*.py` file.
2. Modify `BASE_PARAMS`, `QUICK_OVERRIDES`, or instance overrides.
3. Run the script directly.
4. Inspect the generated result folder under `results/hom_pgd/`.

## Result Artifacts

Experiment runs write compact summaries and method records under `results/`.
A typical comparison contains:

```text
config.json
result.json
artifacts/
  manifest.json
  summary.json
  records/
    Hom-PGD.npy
    PGD.npy
    ...
```

Use:

- `summary.json` for method-level metrics;
- `records/<method>.npy` for full iteration traces;
- generated figures for convergence, feasibility, and method comparisons.

## Package Layout

```text
src/homopt/
  problems/       Problem definitions
  mappings/       Gauge, star, and homeomorphic maps
  optim/          Optimizer loops and algorithm dispatch
  experiments/    Reproducible workloads and result persistence
  solvers/        Baseline solver wrappers
  viz/            Plotting and artifact helpers

scripts/hom_pgd/  Hom-PGD experiment entrypoints
docs/             Architecture and experiment documentation
tests/            Regression and package-boundary tests
```

## Build from Source and Test

For local development:

```bash
pip install -e .[dev,research,solvers,viz,models]
python -m pytest -q tests
```

For the project conda environment:

```bash
conda run --no-capture-output -n ML python -m pytest -q tests
```

## Roadmap

This README currently focuses on Hom-PGD. Future user-facing documentation can
add separate pages for:

- Hom-ALM and proximal Hom-ALM;
- INN-PGD and learned feasible maps;
- neural prediction plus optimization-based post-processing;
- solver-specific benchmark setup and reproducibility notes.

## Status

This is a research package. The Hom-PGD workflow is organized around editable
experiment scripts and shared package modules, but the public API should still
be treated as evolving.

## Citation

If this repository supports your research, please cite:

```bibtex
@inproceedings{
liu2026fast,
title={Fast Projection-Free Approach (without Optimization Oracle) for Optimization over Compact Convex Set},
author={Chenghao Liu and Enming Liang and Minghua Chen},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=bP5cU0OYSn}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
