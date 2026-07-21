# Main Experiment Notes: SOCP-EQ and Stiefel

These notes summarize the two current main numerical experiment families used by
the editable scripts:

- `scripts/hom_alm/run_convex_eq_compare.py`
- `scripts/hom_alm/run_stiefel_eq_compare.py`

The goal of this document is to keep the paper-facing formulation, current
script settings, baseline roles, and reported metrics aligned.

## Shared Reporting Protocol

For both experiment families, the feasibility metrics are split by constraint
type. The code evaluates the full residual vector with equality residuals
included and no clipping, then reports

```math
\mathrm{vio}_{\mathrm{ineq}}(x)
= \max_i [g_i(x)]_+,
\qquad
\mathrm{vio}_{\mathrm{eq}}(x)
= \max_j |h_j(x)|,
```

where `[a]_+ = max(a, 0)`. The full violation is

```math
\mathrm{vio}_{\mathrm{full}}(x)
= \max\{\mathrm{vio}_{\mathrm{ineq}}(x),
        \mathrm{vio}_{\mathrm{eq}}(x)\}.
```

Thus the reported violation is a max residual certificate, not an average or an
`L2` norm.

The convergence figures use the same plotting protocol across experiments:

- objective value versus outer iteration;
- objective value versus running time;
- equality and inequality violation versus iteration/time;
- extra log-x versions for both iteration and time axes;
- solver reference objective shown as a black dotted horizontal line, annotated
  directly by `MOSEK` or `IPOPT` rather than included in the legend.

## Experiment 1: SOCP-EQ With Convex Inequalities

### Formulation

The SOCP-EQ benchmark is a convex quadratic program with convex inequality
constraints and linear equality constraints. In the current script, only
convex quadratic inequalities and box constraints are active; linear and SOC
inequality counts are set to zero.

The mathematical form is

```math
\begin{aligned}
\min_x \quad
& \frac{1}{2} x^\top Q x + p^\top x \\
\mathrm{s.t.}\quad
& g_i(x) \le 0, \qquad i=1,\dots,m_{\mathrm{ineq}},\\
& A_{\mathrm{eq}}x - b_{\mathrm{eq}} = 0,\\
& L \le x \le U .
\end{aligned}
```

The active convex quadratic inequality residuals are

```math
g_i(x)
= \frac{1}{2} x^\top Q_i x + p_i^\top x - b_i,
\qquad Q_i \succeq 0.
```

The box constraints are represented as residuals

```math
L - x \le 0,\qquad x - U \le 0.
```

The equality residual is

```math
h(x) = A_{\mathrm{eq}}x - b_{\mathrm{eq}}.
```

### Current Script Settings

Current active script: `scripts/hom_alm/run_convex_eq_compare.py`.

The active single-instance configuration is:

| Field | Value |
| --- | --- |
| `problem_type` | `socp_eq` |
| `seed` | `2026` |
| `n_var` | `1000` |
| `n_linear_cons` | `0` |
| `n_soc_cons` | `0` |
| `n_qua_cons` | `100` |
| `n_lin_eq` | `500` |
| box bounds | `[-5, 5]` |
| objective type | diagonal quadratic |
| constraint quadratic type | diagonal PSD quadratic |
| dtype/device | `float32`, `cuda:0` |
| reference solver label | `MOSEK` |

Constraint counts for the active instance:

```text
variables:   1000
inequality:  2100 = 100 quadratic + 2000 box
equality:     500
```

The output scale label is generated from the active dimensions:

```text
socp_eq_n1000_q100_eq500
```

If `INSTANCE_OVERRIDES` is empty, the script runs the current
`BASE_PARAMS + QUICK_OVERRIDES` instance directly. If `INSTANCE_OVERRIDES`
is nonempty, each entry can use `label=None` to keep the saved run name tied to
the actual problem dimensions.

### Algorithms

The active algorithm list is

```text
PGD
Prox-Penalty
Prox-ALM
Prox-ALM-EQ
Prox-Hom-ALM
```

The suffix convention is important:

- no `-EQ` suffix: equality and inequality residuals both enter the
  penalty/Lagrangian terms directly;
- `-EQ` suffix: only equality constraints enter the penalty/Lagrangian terms,
  while convex inequalities are handled explicitly by the CVXPY subproblem.

Display labels used in plots are configured through `CONVEX_METHOD_LABELS`.
In particular, `Prox-Hom-ALM` is displayed as `Hom-PALM`.

### Optimization Settings

The ALM-style outer-loop budget is:

```text
outer_iterations: 1000
outer_learning_rate: 1e-3
outer_stepsize_rule: adaptive
outer_lr_decay: 0.995
dual_learning_rate: 1e-1
penalty_coef: 1
penalty_growth: 1.01
proximal_coef: 1e-1
max_penalty: 1e1
max_dual: 1e2
convergence_threshold: 1e-6
first_order_lagrangian_gap_threshold: 1e-6
```

The inner subproblem solver uses momentum-GD:

```text
inner_iterations: 100
inner_learning_rate: 1e-3
inner_stepsize_rule: adaptive
momentum: 0.9
inner_proximal_update_iterations: 10
inner_proximal_coef: 1e-3
```

For Hom-ALM/Hom-PALM, the homeomorphism uses the inequality side for the
origin constraint set:

```text
hom_origin_constraints: inequality
hom_origin_ip_mode: central_ip
smooth: True
initial_point_mode: gauge_center
hom_map_explicit_gradient_rule: polynomial
hom_map_smooth_tie_tol: 1e-7
hom_map_smooth_temperature: 1e-4
```

### Reference Role

`MOSEK` is used as the reference convex solver. Its objective value is treated
as the ground-truth reference objective in plots and comparison tables.

## Experiment 2: Constrained PCA on the Stiefel Manifold

### Formulation

The Stiefel experiment is a constrained PCA/subspace-learning problem. The
decision variable is a matrix

```math
X \in \mathbb{R}^{n \times k},
```

flattened internally as a vector of length `n*k`.

The objective is

```math
\min_X \quad -\mathrm{tr}(X^\top A X),
```

where `A` is a data covariance matrix. For data-driven PCA instances, the
covariance is standardized and normalized. The current presets use
`objective_normalization="top_sum"`, i.e. the covariance is divided by the sum
of the top `k` eigenvalues. Under the unconstrained PCA solution, this makes the
best reference objective close to `-1`; active side constraints move the
constrained optimum above that value.

The nonlinear equality is the Stiefel constraint

```math
X^\top B X = I_k.
```

In the current data-driven settings, `B` defaults to identity, so this is the
standard Stiefel constraint

```math
X^\top X = I_k.
```

The convex side inequalities include box constraints

```math
L \le X_{ij} \le U
```

and one group/SOC budget over selected feature rows

```math
\|X_{\mathcal{R}, :}\|_F \le \rho.
```

The row set `R` is selected by `restricted_rows="top_loading"` in the active
presets, meaning the rows are chosen from the largest covariance loading
coordinates.

### Current Script Settings

Current active script: `scripts/hom_alm/run_stiefel_eq_compare.py`.

The active single-instance preset is `SMALL_DIGITS_OVERRIDES`:

| Field | Value |
| --- | --- |
| data source | `sklearn_digits` |
| feature dimension `n` | `64` |
| columns `k` | `3` |
| decision variables | `64 * 3 = 192` |
| restricted rows | top-loading rows |
| restricted row count | `16` |
| group budget | `0.75` |
| box bounds | `[-0.5, 0.5]` |
| objective normalization | `top_sum` |
| dtype/device | `float32`, `cuda:1` |
| reference solver label | `IPOPT` |

Constraint counts for the active small instance:

```text
variables:   192
inequality:  385 = 384 box + 1 group/SOC
equality:      9 = k^2
```

The current sequential instance list is:

| Scale label | Data | Variables | Inequality count | Equality count |
| --- | --- | ---: | ---: | ---: |
| `small_digits` | digits, 64 features, `k=3` | `192` | `385` | `9` |
| `intermediate_lfw` | LFW, 713 features, `k=3` | `2139` | `4279` | `9` |
| `medium_olivetti` | Olivetti, 4096 features, `k=2` | `8192` | `16385` | `4` |

Additional commented candidate presets:

| Scale label | Data | Variables | Inequality count | Equality count |
| --- | --- | ---: | ---: | ---: |
| `medium_lfw` | LFW, 713 features, `k=5` | `3565` | `7131` | `25` |
| `large_mnist` | MNIST, 784 features, `k=13` | `10192` | `20385` | `169` |

### Algorithms

The active algorithm list is

```text
Prox-ALM
StiefelRetractionPenalty
StiefelRetractionALM
Prox-Hom-ALM
```

The reference solve is requested by `include_reference_solver=True` and uses
`StiefelIPOPT` even when `StiefelIPOPT` is not explicitly included in the
algorithm list.

Plot order is fixed as

```text
ALM
Prox-ALM
StiefelRetractionPenalty
StiefelRetractionALM
Hom-ALM
Prox-Hom-ALM
```

with display labels such as

```text
Prox-ALM                  -> PALM
Prox-Hom-ALM              -> Hom-PALM
StiefelRetractionPenalty  -> Retraction-Penalty
StiefelRetractionALM      -> Retraction-ALM
```

### Baseline Semantics

`Prox-ALM` operates directly in the original `x` variables and applies proximal
ALM terms to the full residual.

`Prox-Hom-ALM` operates through the homeomorphism map with proximal terms in
`z` space. The mapping settings are

```text
smooth: True
x_origin: zero
hom_map_explicit_gradient_rule: polynomial
hom_map_smooth_tie_tol: 1e-7
hom_map_smooth_temperature: 1e-4
```

`StiefelRetractionPenalty` keeps the nonlinear Stiefel equality hard by
projecting gradients to the Stiefel tangent space and applying QR retraction
after each tangent step. Convex side inequalities are handled by a quadratic
penalty with an outer-loop penalty schedule.

`StiefelRetractionALM` uses the same Stiefel retraction update for the equality
constraint but handles convex side inequalities through an augmented Lagrangian
with nonnegative inequality duals.

`StiefelIPOPT` is the ground-truth/reference solver, analogous to `MOSEK` in
the SOCP-EQ experiment. The script caches reference solves with
`reference_cache=True` in `artifacts/stiefel_reference_context_cache.npy`.

### Optimization Settings

The shared ALM-style outer budget is:

```text
outer_iterations: 1000
outer_learning_rate: 1e-2
outer_stepsize_rule: adaptive
outer_lr_decay: 0.995
dual_learning_rate: 1e-1
penalty_coef: 1.0
penalty_growth: 1.01
proximal_coef: 1e-1
max_penalty: 1e3
max_dual: 1e3
convergence_threshold: 1e-6
```

The inner solver is momentum-GD:

```text
inner_iterations: 200
inner_iterations_min: 100
inner_iterations_max: 100
inner_learning_rate: 1e-3
inner_stepsize_rule: adaptive
momentum: 0.9
inner_proximal_update_iterations: 10
inner_proximal_coef: 1e-3
```

The initial point is a random vector projected to a small ball:

```text
initialization.mode: random_ball
initialization.scale: 0.1
```

### Reference Role

`IPOPT` is used as the nonlinear-programming reference solver. Its objective
value is plotted as the optimal objective reference line and printed as the
reference solver row in comparison tables.

## Paper-Facing Summary

The two main experiments are designed to separate two sources of difficulty:

1. **SOCP-EQ** tests algorithms on a large convex problem with many convex
   inequalities and many linear equalities. It isolates the effect of equality
   handling, proximal stabilization, and the homeomorphism mapping while using
   MOSEK as a reliable convex reference.

2. **Stiefel constrained PCA** tests algorithms on a nonconvex equality
   manifold with convex side constraints. It compares direct proximal ALM,
   homeomorphic proximal ALM, and classical retraction-based baselines, using
   IPOPT as the nonlinear reference solve.

Together, these experiments cover both a convex equality-constrained benchmark
with scalable conic/quadratic inequalities and a nonconvex equality-constrained
benchmark with data-driven objective structure and active convex side
constraints.
