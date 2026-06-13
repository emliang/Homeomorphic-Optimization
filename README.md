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

## Where to Start

Start with these files and modules:

- `README.md`: high-level package overview and Hom-PGD user guide.
- `scripts/hom_pgd/`: editable Hom-PGD experiment scripts.
- `scripts/hom_pgd/_configs.py`: shared defaults for the Hom-PGD scripts.
- `src/homopt/experiments/hom_pgd.py`: public Hom-PGD experiment entrypoints.
- `src/homopt/experiments/benchmarks/`: benchmark implementations used by the scripts.
- `src/homopt/experiments/common/`: shared configuration, recording, labeling, and comparison helpers.

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

Shared script defaults live in:

```text
scripts/hom_pgd/_configs.py
```

It compares:

```python
["PGD", "FW", "ALM", "RD", "Hom-PGD"]
```


Common controls include:

| Parameter | Meaning |
| --- | --- |
| `learning_rate` | Latent-space step size. |
| `stepsize_rule` | Step-size policy, such as `adaptive`, `constant`, or `diminish`. |
| `hom_p_norm` | Norm used by the latent ball projection route. |
| `momentum` | Momentum for the update backend when enabled. |
| `max_iterations` | Shared iteration budget. For ALM-family methods this is aligned to the outer-loop budget. |
| `max_running_time` | Wall-clock budget for an experiment run. |
| `initial_point_mode` | Initial point policy; current Hom-PGD configs default to `gauge_center`. |
| `visualize` | Whether to write plots and comparison artifacts. |
| `visualize_only` | Reuse stored artifacts without rerunning the experiment. |

Per-method overrides go under `algorithm_config`, while shared controls go under
`common_config`. For routine experiments, edit `QUICK_OVERRIDES` in the target
script first.

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

## Quick Start

Recommended first step: run the 2D poly/star-shaped feasible-set comparison.
It is the easiest script for checking the end-to-end workflow because the
problem is small, visual, and quick to edit.

```bash
python scripts/hom_pgd/run_poly_star_compare.py
```

Then run the main Hom-PGD inequality comparison:

```bash
python scripts/hom_pgd/run_convex_ineq_compare.py
```

Run the MaxCut SDP Hom-PGD comparison:

```bash
python scripts/hom_pgd/run_maxcut_sdp_compare.py
```

## Hom-PGD Experiment Scripts

| Script | Purpose |
| --- | --- |
| `scripts/hom_pgd/_configs.py` | Shared defaults and run-label helpers for the Hom-PGD scripts. |
| `scripts/hom_pgd/run_convex_ineq_compare.py` | Main convex inequality/SOCP comparison. |
| `scripts/hom_pgd/run_poly_star_compare.py` | 2D polytope, star-shaped, or intersection toy set. |
| `scripts/hom_pgd/run_maxcut_sdp_compare.py` | MaxCut SDP-style Hom-PGD comparison. |
| `scripts/hom_pgd/run_adversarial_attack.py` | Norm-constrained adversarial-attack workflow. |

Each script is intentionally editable. The intended workflow is:

1. Open the relevant `scripts/hom_pgd/run_*.py` file.
2. Modify `BASE_PARAMS`, `QUICK_OVERRIDES`, or instance overrides.
3. Run the script directly.
4. Inspect the generated result folder under `results/hom_pgd/`.

## Development Workflow

For new Hom-PGD experiments, use the 2D poly/star script as the first
development target. It gives fast feedback on instance construction, algorithm
dispatch, result recording, and visualization before moving to larger SOCP,
MaxCut, or adversarial workflows.

### Add a New Instance

An instance is one concrete problem geometry and objective. The recommended path
is:

1. Define or extend the problem class under `src/homopt/problems/`.
   Deterministic problems should expose `objective_x(x)` and
   `constraint_x(x, clip=True)`. Add `project(x)` only when projection-based
   baselines such as `PGD` need a custom projection.
2. Add the feasible-set map under `src/homopt/mappings/` when Hom-PGD needs a
   new latent-to-decision transformation. For gauge-style sets, follow
   `src/homopt/mappings/gauge/`; for toy star-shaped sets, follow
   `src/homopt/mappings/star.py`.
3. Add a benchmark builder under `src/homopt/experiments/benchmarks/`. The
   benchmark should build the problem, build the map, prepare algorithm params,
   call `run_algorithm(...)`, summarize records, and save visualizations.
4. Export the benchmark from `src/homopt/experiments/hom_pgd.py` if it should
   become a public Hom-PGD entrypoint.
5. Add or extend an editable script under `scripts/hom_pgd/`. Keep the script
   thin: define `BASE_PARAMS`, `QUICK_OVERRIDES`, optional
   `INSTANCE_OVERRIDES`, then call `make_benchmark_entrypoint(...)`.
6. Start with one small instance. After it runs, add multiple instances through
   `INSTANCE_OVERRIDES` as `(scale_label, overrides)` pairs. If `scale_label` is
   `None`, the script can build a label from the final problem dimensions.

For existing poly/star or SOCP instances, most changes only require editing the
script-level config: `problem_type`, `alpha`, `num_star`, `poly_config`,
`n_var`, `n_linear_cons`, `n_soc_cons`, `n_qua_cons`, bounds, seeds, and
objective/constraint type fields.

### Add a New Algorithm

An algorithm is a method that can be selected in the script-level `algorithms`
list. The recommended path is:

1. Implement the optimizer under `src/homopt/optim/`. First-order methods
   usually belong in `src/homopt/optim/first_order.py`; ALM-style methods belong
   in the ALM or Lagrangian modules. Follow the existing `optimize(...)`
   contract: return final decision, decision trajectory, objective trajectory,
   violation trajectory, and per-iteration timing. If the algorithm transforms
   through a map, also return or expose the transform timing and latent
   trajectory consistently with `HomPGDOptimizer`.
2. Register the algorithm name in `src/homopt/optim/registry.py` inside
   `_algorithm_specs()`. The registry decides which optimizer class to build and
   whether the run result includes map transform time.
3. Add default parameter construction in
   `src/homopt/experiments/common/method_specs.py` when the algorithm should
   participate in shared benchmark builders.
4. Add the algorithm name to the target benchmark's `algorithms` list and add
   per-method defaults or overrides under `algorithm_config`.
5. Run the smallest compatible script first, usually
   `scripts/hom_pgd/run_poly_star_compare.py`, then move to larger benchmarks.

Use `common_config` for shared controls such as `max_iterations`,
`max_running_time`, `learning_rate`, `stepsize_rule`, `momentum`, and
`initial_point_mode`. Use `algorithm_config` for method-specific controls such
as Hom-PGD's `hom_p_norm`, PGD projection subproblem settings, or an algorithm's
own learning rate.

For ALM-family baselines, `common_config.max_iterations` is treated as the
shared outer-loop budget. Do not set a conflicting `outer_iterations` inside a
per-method override.

### Config Flow

Script parameters are layered in this order:

```text
shared defaults in scripts/hom_pgd/_configs.py
  -> BASE_PARAMS in the selected run script
  -> QUICK_OVERRIDES
  -> INSTANCE_OVERRIDES, when present
```

### Experiment Flow

The script flow is intentionally simple:

```text
run_*.py
  -> merge BASE_PARAMS and QUICK_OVERRIDES
  -> optionally expand INSTANCE_OVERRIDES
  -> call src/homopt/experiments/hom_pgd.py
  -> run the benchmark under src/homopt/experiments/benchmarks/
  -> record config, result, traces, and figures under results/hom_pgd/
```

When adding a new experiment, keep the script thin. Put reusable benchmark logic
under `src/homopt/experiments/benchmarks/`, shared config or labeling helpers in
`scripts/hom_pgd/_configs.py` or `src/homopt/experiments/common/`, and leave
the top-level script as the editable user entrypoint.

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
  problems/                Problem definitions
    convex/                Convex inequality, MaxCut, and toy-set problems
    convex_parametric/     Parametric convex problem families
    jcc/                   Joint chance-constrained problem families
    qcqp/                  QCQP problem families
    stiefel/               Stiefel-manifold equality problems
  mappings/                Gauge, star, and homeomorphic maps
    gauge/                 Gauge-map implementation and explicit derivatives
  optim/                   Optimizer loops, ALM routes, and dispatch
  experiments/             Reproducible workloads and result persistence
    hom_pgd.py             Public Hom-PGD experiment entrypoints
    benchmarks/            Benchmark implementations
    common/                Shared experiment helpers
    adversarial/           Adversarial-attack workflow
  solvers/                 Baseline solver wrappers
  viz/                     Plotting and artifact helpers

scripts/hom_pgd/  Hom-PGD experiment entrypoints
```

The repository includes the full `homopt` package source so shared internals
remain importable. The public-facing scripts and this README currently focus on
Hom-PGD.

## Build from Source

For local development:

```bash
pip install -e .[dev,research,solvers,viz,models]
```

After installation, a quick import smoke check is:

```bash
python -c "import homopt; from homopt.experiments.hom_pgd import convex_algorithm_comparison; print('homopt import OK')"
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

## Implementation Note

The original paper implementation uses automatic differentiation for gradient
calculation. This repository implements explicit gradients for all included
methods where supported. As a result, per-iteration cost comparisons from this
codebase may not exactly match the per-iteration cost comparisons reported in
the paper.

## Citation

If this repository supports your research, please cite the relevant work:

```bibtex
@inproceedings{
liu2026fast,
title={Fast Projection-Free Approach (without Optimization Oracle) for Optimization over Compact Convex Set},
author={Chenghao Liu and Enming Liang and Minghua Chen},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2026},
url={https://openreview.net/forum?id=bP5cU0OYSn}
}

@inproceedings{
liu2026hompgd,
title={Hom-{PGD}\${\textasciicircum}+\$: Fast Reparameterized Optimization over Non-convex Ball-Homeomorphic Set},
author={Chenghao Liu and Enming Liang and Minghua Chen},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=cOKyRhR8uZ}
}

@inproceedings{
li2026gauge,
title={Gauge Flow Matching: Efficient Constrained Generative Modeling over General Convex Set and Beyond},
author={Xinpeng Li and Enming Liang and Minghua Chen},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=vxq1OnaAMq}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
