# Experiment Research Notes

## 2026-04-19: Nonconvex equality + convex inequality ML benchmark candidates

### Scope

This note records candidate machine-learning experiment families beyond the
current convex example.

The target structure is:

```text
minimize    f(x; c)
subject to  h(x; c) = 0      nonconvex equality constraints
            g(x; c) <= 0     convex inequality constraints
```

where `c` is optional instance/context data. This matches the current two main
HomOPT routes:

- Single/fixed-problem route: solve one instance and compare solvers.
- Parametric learning route: predict a candidate solution for each instance,
  then compare solver/refinement/post-processing methods under the same metrics.

The intended Hom-ALM split is:

- convex inequalities are handled by a homeomorphic/gauge map;
- nonconvex equalities are handled by the ALM outer loop;
- all routes report objective, equality violation, convex inequality violation,
  feasibility rate, runtime, and post-processing improvement.

### Primary candidates

| Priority | Family | Core equality | Convex inequality side | Why it fits HomOPT |
|---|---|---|---|---|
| P0 | Stiefel / orthogonality learning | `X.T @ X = I` or `X.T @ B(c) @ X = I` | norm, box, or linear resource budgets | clean nonconvex equality, explicit gradients, strong ML relevance |
| P0 | Euclidean distance geometry | `||p_i - p_j||^2 = d_ij^2` on observed edges | coordinate box, center/radius bounds, upper-distance SOC bounds | direct NeurIPS 2024 nonconvex reconstruction task |
| P1 | Differential-equation constrained optimization | discretized ODE/PDE residual `F(y, u; c) = 0` | control bounds, energy budgets, state upper bounds | recent ICLR 2025 learning-to-solve benchmark direction |
| P1 | AC-OPF-lite / physics-informed GNN | AC power-flow balance equations | generator boxes, voltage upper balls, current/flow upper SOC-style constraints | strongest applied parametric optimization story; use a convex-inequality subset first |
| P1 | Moment-constrained molecular structure | fixed moments of inertia, embedded in a Stiefel manifold | coordinate or radius bounds | ICLR 2025 exact-constraint molecular generation route |
| P2 | Equivariant model training | equivariance residual `F(T_g x) = R_g F(x)` | norm/spectral/box budgets | broad NeurIPS 2025 constrained-learning direction but more model-specific |
| P2 | Constrained diffusion/sampling | task-specific nonlinear constraints or ODE constraints | image/trajectory/material bounds | relevant trend, but less aligned with the current solver benchmark core |

### Detailed formulations

#### `StiefelProblem`: orthogonality-constrained learning

Single-instance formulation:

```text
data/context:  A in R^{n x n}, optional B positive definite in R^{n x n}
decision:      X in R^{n x p}, p <= n

minimize       f(X; A) = -tr(X.T @ A @ X)
subject to     h(X) = vec(X.T @ X - I_p) = 0
               g_box(X) = [L - vec(X), vec(X) - U] <= 0        optional
               g_norm(X) = ||vec(X)||_2 - R <= 0               optional
               g_group(X) = ||X[S, :]||_F - beta <= 0          optional
```

Generalized Stiefel variant:

```text
minimize       f(X; A) = -tr(X.T @ A @ X)
subject to     h(X; B) = vec(X.T @ B @ X - I_p) = 0
               g(X) <= 0
```

Learning/parametric variant:

```text
context:       c = (A, B, task data)
decision:      X(c)
predictor:     NN(c) -> X_hat
post-process:  X_hat -> X_feasible by ALM / Hom-ALM / projection baseline
```

Properties:

- The equality is nonconvex and smooth.
- The common objective has explicit gradient
  `grad_X f = -(A + A.T) @ X`.
- The standard equality residual has explicit Jacobian-vector product
  `D h[X](Delta) = vec(Delta.T @ X + X.T @ Delta)`.
- The optional inequalities above are convex and can be used to define the
  gauge target set. The Frobenius norm bound is redundant when
  `X.T @ X = I_p`, but it is useful as a controlled convex inequality target
  for Hom-ALM tests.
- `g_group` is the practical constrained-PCA side constraint: it limits how
  much the learned subspace can use selected feature groups.

This is the smallest clean benchmark for nonconvex equality handling.

Constrained PCA application form:

```text
data:          D in R^{N x n}
context:       covariance A = centered(D).T @ centered(D) / (N - 1)
decision:      X in R^{n x p}

minimize       -tr(X.T @ A @ X)
subject to     X.T X = I_p
               ||X[S, :]||_F <= beta
               L <= vec(X) <= U
```

Interpretation:

- `X` is an orthonormal low-dimensional representation.
- `S` is a restricted feature group, such as protected attributes, expensive
  sensors, private features, or unstable covariates.
- `beta` is a hard exposure budget for that group.
- This is a better application benchmark than a single-coordinate toy bound
  because the convex side constraint has a clear ML meaning.

#### `DistanceGeometryProblem`: Euclidean distance reconstruction

Let `G = (V, E_obs)` be an observed distance graph, and let `d_ij` be observed
edge distances.

```text
context:       observed graph E_obs, distances d_ij, embedding dimension r
decision:      P = [p_1, ..., p_n] in R^{r x n}

minimize       f(P) = 0.5 * sum_{(i,j) in E_fit} w_ij
                         (||p_i - p_j||_2^2 - d_ij^2)^2
               or f(P) = 0.5 * ||P - P_prior||_F^2

subject to     h_ij(P) = ||p_i - p_j||_2^2 - d_ij^2 = 0,
                         for (i,j) in E_eq

               L <= vec(P) <= U
               ||p_i - p_center||_2 <= R_i
               ||p_i - p_j||_2 <= u_ij, for selected upper-bound edges
```

Equivalent inequality residual form:

```text
g_box(P)       = [L - vec(P), vec(P) - U] <= 0
g_radius_i(P)  = ||p_i - p_center||_2 - R_i <= 0
g_upper_ij(P)  = ||p_i - p_j||_2 - u_ij <= 0
```

Properties:

- The equality is nonconvex and smooth.
- The coordinate box is linear and convex.
- The radius and upper-distance constraints are second-order-cone constraints,
  hence convex.
- Translation and rotation symmetries should be fixed for stable metrics. Use
  either a centering equality `sum_i p_i = 0`, anchor a few points, or report
  Procrustes-aligned reconstruction error.

Parametric learning route:

```text
context:       c = sampled graph + observed distances
predictor:     NN(c) -> P_hat
post-process:  compare raw prediction, ALM, Hom-ALM, projection/refinement
metrics:       objective, equality residual, convex inequality residual,
               Procrustes error, runtime
```

This is the strongest direct bridge to the NeurIPS 2024 distance-geometry
reconstruction setting.

#### `DEConstrainedProblemLite`: differential-equation constrained optimization

Use a time-discretized ODE first. PDE variants should wait until the ODE route
is stable.

```text
context:       initial state y_0, target trajectory y_ref, dynamics parameter c
decision:      y_0, ..., y_T in R^m
               u_0, ..., u_{T-1} in R^q

minimize       f(y, u; c) =
                   sum_{t=0}^T 0.5 * ||y_t - y_ref_t||_Q^2
                 + sum_{t=0}^{T-1} 0.5 * ||u_t||_R^2

subject to     h_t(y, u; c) =
                   y_{t+1} - y_t - dt * F(y_t, u_t; c) = 0,
                   t = 0, ..., T-1

               y_0 = y_init
               u_min <= u_t <= u_max
               y_min <= y_t <= y_max
               sum_t ||u_t||_2^2 <= E
```

Residual form:

```text
g_u_lower_t(u) = u_min - u_t <= 0
g_u_upper_t(u) = u_t - u_max <= 0
g_y_lower_t(y) = y_min - y_t <= 0
g_y_upper_t(y) = y_t - y_max <= 0
g_energy(u)    = sum_t ||u_t||_2^2 - E <= 0
```

Properties:

- The equality is nonconvex when `F` is nonlinear.
- Box constraints are linear and convex.
- The energy budget is convex quadratic.
- This problem naturally supports both single-instance solver comparison and
  parametric learning over different initial states, targets, or dynamics
  parameters.

Minimal first dynamics:

```text
F(y, u; c) = A y + B u + alpha * tanh(C y)
```

This is nonlinear enough to test nonconvex equality behavior while keeping all
gradients simple.

#### `ACOPFLiteProblem`: AC power-flow equality with convex safety bounds

Use rectangular voltage coordinates and keep only convex safety inequalities in
the first version.

```text
context:       network admittance Y = G + iB, loads p_d/q_d,
               generator costs and limits
decision:      v_re, v_im in R^{n_bus}
               p_g, q_g in R^{n_gen}

minimize       f(p_g) = sum_{k in generators}
                   a_k * p_g_k^2 + b_k * p_g_k + c_k

subject to     active power balance at each bus i:
               p_g_i - p_d_i =
                   sum_j G_ij * (v_re_i v_re_j + v_im_i v_im_j)
                 + sum_j B_ij * (v_im_i v_re_j - v_re_i v_im_j)

               reactive power balance at each bus i:
               q_g_i - q_d_i =
                   sum_j G_ij * (v_im_i v_re_j - v_re_i v_im_j)
                 - sum_j B_ij * (v_re_i v_re_j + v_im_i v_im_j)

               p_g_min <= p_g <= p_g_max
               q_g_min <= q_g <= q_g_max
               ||[v_re_i, v_im_i]||_2 <= V_i_max
               ||Y_ij * ([v_re_i, v_im_i] - [v_re_j, v_im_j])||_2 <= I_ij_max
```

Residual form:

```text
h_p_i(x; c)       = p_balance_residual_i = 0
h_q_i(x; c)       = q_balance_residual_i = 0
g_pg_box(x)       = [p_g_min - p_g, p_g - p_g_max] <= 0
g_qg_box(x)       = [q_g_min - q_g, q_g - q_g_max] <= 0
g_v_upper_i(x)    = ||[v_re_i, v_im_i]||_2 - V_i_max <= 0
g_i_upper_ij(x)   = ||line_current_ij(v)||_2 - I_ij_max <= 0
```

Properties:

- Power-balance equations are nonconvex quadratic equalities.
- Generator boxes are linear and convex.
- Voltage upper bounds and line-current upper bounds are SOC constraints if
  line current is represented as an affine function of voltage.
- Do not include voltage lower bounds `V_i_min <= ||v_i||_2` in the first
  version. That constraint is nonconvex.
- Do not include apparent-power branch limits in the first lite version unless
  a convex relaxation is used; the exact rectangular AC expression is generally
  nonconvex.

Parametric learning route:

```text
context:       load profile and generator/network parameters
predictor:     GNN/MLP(c) -> raw dispatch and voltages
post-process:  compare raw, feasibility-seeking, ALM, Hom-ALM, and NLP solver
metrics:       cost, power-balance violation, convex safety violation,
               runtime, feasibility rate
```

This is the highest-value applied benchmark, but it should come after the
cleaner Stiefel and distance-geometry tests.

#### `MomentConstrainedMoleculeProblem`: exact moments of inertia

Use a simplified fixed-principal-axis formulation first.

```text
context:       atom masses m_i, target principal moments lambda_1, lambda_2,
               lambda_3, optional molecular formula/features
decision:      P = [p_1, ..., p_n] in R^{3 x n}

minimize       f(P; c) = molecular score / learned energy / distance to prior

subject to     center of mass:
               sum_i m_i * p_i = 0

               inertia tensor:
               I(P) = sum_i m_i * (||p_i||_2^2 * I_3 - p_i p_i.T)

               h_moment(P) = vec(I(P) - diag(lambda_1, lambda_2, lambda_3)) = 0

               L <= vec(P) <= U
               ||p_i||_2 <= R_i
               ||p_i - p_j||_2 <= u_ij, optional steric upper bounds
```

Properties:

- The center-of-mass equality is linear.
- The inertia equality is nonconvex quadratic.
- Coordinate/radius/upper-distance inequalities are convex.
- The formulation fixes the principal-axis orientation. A rotation-invariant
  version would constrain only the eigenvalues of `I(P)`, but that is more
  complex and should not be the first implementation.

This is a chemically meaningful Stiefel-adjacent example, but it is less
minimal than `StiefelProblem`.

#### `EquivariantTrainingProblem`: approximate equivariance as constraints

Use a finite sampled group/action set so the constraints are finite-dimensional.

```text
context:       training data (x_i, y_i), sampled group elements g in G_sample
decision:      neural-network parameters theta

minimize       f(theta) = empirical task loss

subject to     h_{i,g}(theta) =
                   F_theta(T_g x_i) - R_g F_theta(x_i) = 0

               ||theta||_2 <= R
               ||W_l||_op <= rho_l, for selected layers
               L <= theta <= U
```

Properties:

- The equality is nonlinear and nonconvex in network parameters.
- Norm, spectral-norm, and box bounds are convex in the selected parameter
  blocks.
- This is relevant to constrained learning, but it is less solver-native than
  Stiefel or distance geometry because the objective and equality residuals are
  full neural-network training losses.

Use this only after the solver-side benchmark interface is stable.

#### `ConstrainedSamplingProblem`: diffusion or generative sampling constraints

This is a later-stage benchmark because it is closer to constrained generation
than deterministic solver comparison.

```text
context:       prompt / conditioning / observed measurements c
decision:      generated sample x, or full sampling trajectory x_0, ..., x_T

minimize       f(x; c) = diffusion prior energy / denoising objective /
                         measurement-fitting loss

subject to     h_eq(x; c) = 0, for physics, ODE, or exact measurement constraints
               l <= x <= u
               ||A x - b||_2 <= epsilon
               C x <= d
```

Properties:

- Equality constraints are task-specific and may be nonlinear.
- Box, affine, and measurement-ball inequalities are convex.
- This is a good future comparison point for projection-based constrained
  sampling, but not the first Hom-ALM benchmark to implement.

### Recommended first implementations

Start with two narrow, reusable problems rather than a generic over-abstract
framework:

1. `StiefelProblem`
   - Variable: matrix `X in R^{n x p}`.
   - Objective examples:
     - PCA/eigenvalue: `min -tr(X.T @ A @ X)`.
     - LoRA/subspace toy: `min loss(U, S, V; data)` with `U.T U = I`,
       `V.T V = I`.
   - Equality:
     - `h(X) = vec(X.T @ X - I)`.
   - Convex inequalities:
     - optional `||X||_F <= radius`, elementwise bounds, or linear budgets.
   - Baselines:
     - ALM, Hom-ALM, Riemannian/projected baselines if available.

2. `DistanceGeometryProblem`
   - Variable: point cloud `P in R^{r x n}`.
   - Objective:
     - weighted reconstruction loss or downstream embedding quality.
   - Equality:
     - `h_ij(P) = ||p_i - p_j||_2^2 - d_ij^2` for observed edges.
   - Convex inequalities:
     - coordinate box, centered norm bound, optional upper-distance constraints
       `||p_i - p_j||_2 <= u_ij`.
   - Parametric route:
     - instance parameter is the partial distance graph and observed distances;
     - predictor outputs raw coordinates; post-processing compares ALM,
       Hom-ALM, feasibility-seeking, and projection-style methods.

AC-OPF-lite should be the first high-value applied case after those two are
stable. Keep only convex safety inequalities initially; lower voltage bounds and
apparent-power constraints can introduce additional nonconvex inequalities and
should not be mixed into the first Hom-ALM/gauge test.

### Related papers and formulations

#### Hard-constrained learning / optimization proxies

- `PiNet: Optimizing hard-constrained neural networks with orthogonal projection layers`
  - Status: ICLR 2026 Oral.
  - Formulation fit: parametric constrained optimization proxy with a neural
    output layer that projects onto convex constraints.
  - Use in HomOPT: baseline family for convex-inequality hard feasibility;
    not enough by itself for nonconvex equality, but a strong comparison for
    learning-based prediction/refinement.
  - Source: https://openreview.net/forum?id=EJ680UQeZG

- `FSNet: Feasibility-Seeking Neural Network for Constrained Optimization with Guarantees`
  - Status: NeurIPS 2025 Poster.
  - Formulation fit: parametric constrained optimization with a differentiable
    feasibility-seeking step minimizing constraint violation.
  - Use in HomOPT: direct baseline against `predict_ray_bisection`,
    `predict_exact_projection`, and `Hom-ALM` post-processing.
  - Source: https://openreview.net/forum?id=oum1txoy1D

- `HardNet: Hard-Constrained Neural Networks with Universal Approximation Guarantees`
  - Status: NeurIPS 2025 COML workshop contribution.
  - Formulation fit: hard neural output constraints, especially multiple
    input-dependent inequalities.
  - Use in HomOPT: learning baseline for convex inequality feasibility.
  - Source: https://neurips.cc/virtual/2025/123317

- `HoP: Homeomorphic Polar Learning for Hard Constrained Optimization`
  - Status: ICLR 2026 submission, not treated as accepted evidence here.
  - Formulation fit: learn-to-optimize with a homeomorphic mapping that outputs
    strictly feasible points without post-correction.
  - Use in HomOPT: conceptually close to our gauge/homeomorphism route; useful
    as a literature pointer, but should be cited with submission status.
  - Source: https://openreview.net/forum?id=ZEiZUa9y3M

#### Stiefel / orthogonality / manifold constraints

- `Stiefel Flow Matching for Moment-Constrained Structure Elucidation`
  - Status: ICLR 2025 Conference.
  - Formulation fit: molecular point clouds with fixed moments of inertia are
    embedded in `St(n, 4)`, so exact moment constraints become manifold
    constraints.
  - Use in HomOPT: exact nonconvex equality benchmark with a molecular
    generation/reconstruction interpretation.
  - Source: https://proceedings.iclr.cc/paper_files/paper/2025/hash/35fdecdf8861bc15110d48fbec3193cf-Abstract-Conference.html

- `Distributed Retraction-Free and Communication-Efficient Optimization on the Stiefel Manifold`
  - Status: ICML 2025 Poster.
  - Formulation fit: stochastic optimization on `X.T X = I`, motivated by PCA
    and orthogonal neural network robustness.
  - Use in HomOPT: baseline inspiration for large-scale orthogonality tests.
  - Source: https://icml.cc/virtual/2025/poster/45108

- `Efficient Optimization with Orthogonality Constraint: a Randomized Riemannian Submanifold Method`
  - Status: ICML 2025 Poster.
  - Formulation fit: nonconvex objective under orthogonality constraints.
  - Use in HomOPT: comparison point for scaling Stiefel equality problems.
  - Source: https://icml.cc/virtual/2025/poster/46238

- `Optimization without Retraction on the Random Generalized Stiefel Manifold`
  - Status: ICML 2024, PMLR.
  - Formulation fit: `X.T @ B @ X = I_p`, with applications to CCA, ICA, and
    generalized eigenvalue problems.
  - Use in HomOPT: parametric equality case where `B(c)` changes by instance.
  - Source: https://proceedings.mlr.press/v235/vary24a.html

- `StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold`
  - Status: NeurIPS 2025 Spotlight Poster.
  - Formulation fit: LoRA variant with `U` and `V` constrained to Stiefel
    manifolds.
  - Use in HomOPT: ML-facing orthogonality example tied to foundation-model
    fine-tuning.
  - Source: https://neurips.cc/virtual/2025/poster/119898

#### Geometry reconstruction / distance constraints

- `Sample-Efficient Geometry Reconstruction from Euclidean Distances using Non-Convex Optimization`
  - Status: NeurIPS 2024 Main Conference.
  - Formulation fit: recover point coordinates from partial Euclidean distance
    measurements using nonconvex formulations.
  - Use in HomOPT: `DistanceGeometryProblem` with equality distance constraints
    and convex coordinate/radius bounds.
  - Source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/8d57f138d14fdfdc520eb29804116d9e-Abstract-Conference.html

#### Differential equations / physical constraints

- `Learning to Solve Differential Equation Constrained Optimization Problems`
  - Status: ICLR 2025 Conference.
  - Formulation fit: optimize control/configuration variables subject to
    ordinary or stochastic differential equation constraints.
  - Use in HomOPT: `DEConstrainedProblemLite` with control bounds as convex
    inequalities and discretized dynamic residual as equality.
  - Source: https://proceedings.iclr.cc/paper_files/paper/2025/hash/99e0f8a68823e12cafda5106c74b04bf-Abstract-Conference.html

- `Learning vector fields of differential equations on manifolds with geometrically constrained operator-valued kernels`
  - Status: ICLR 2025 Conference.
  - Formulation fit: learn ODE vector fields on manifolds and preserve manifold
    membership during integration.
  - Use in HomOPT: dynamic equality/manifold constraint examples, but less
    directly solver-comparison oriented than DE-constrained optimization.
  - Source: https://proceedings.iclr.cc/paper_files/paper/2025/hash/318f3ae8be3c97cb7555e1c932f472a1-Abstract-Conference.html

- `Physics-informed GNN for non-linear constrained optimization: PINCO, a solver for the AC optimal power flow`
  - Status: ICLR 2025 submission on OpenReview, not treated as accepted
    evidence here.
  - Formulation fit: AC-OPF has nonconvex power-flow equality constraints and
    operational inequalities.
  - Use in HomOPT: applied parametric benchmark target after simplified
    `ACOPFLiteProblem` is stable.
  - Source: https://openreview.net/forum?id=BfI0D1ci9r

#### Constrained optimization algorithms and constrained learning context

- `Stochastic Smoothed Primal-Dual Algorithms for Nonconvex Optimization with Linear Inequality Constraints`
  - Status: ICML 2025 Spotlight Poster.
  - Formulation fit: stochastic nonconvex objective with linear inequalities.
  - Use in HomOPT: algorithmic baseline context for primal-dual feasibility
    handling.
  - Source: https://icml.cc/virtual/2025/poster/44818

- `Barrier Algorithms for Constrained Non-Convex Optimization`
  - Status: ICML 2024, PMLR.
  - Formulation fit: nonconvex objective with general convex set constraints
    and linear constraints.
  - Use in HomOPT: theory context for convex inequality handling without
    projection-heavy methods.
  - Source: https://proceedings.mlr.press/v235/dvurechensky24a.html

- `Global Solutions to Non-Convex Functional Constrained Problems via Hidden Convexity`
  - Status: NeurIPS 2025 COML workshop contribution.
  - Formulation fit: nonconvex constrained problems that become convex under an
    invertible nonlinear transformation.
  - Use in HomOPT: closest theory-side pointer to the homeomorphism/gauge
    philosophy.
  - Source: https://neurips.cc/virtual/2025/123298

- NeurIPS 2025 Workshop on Constrained Optimization for Machine Learning
  - Status: NeurIPS 2025 workshop.
  - Relevance: confirms that constrained optimization for large-scale,
    nonconvex, stochastic deep learning is an active ML venue topic.
  - Source: https://constrained-opt-ml.github.io/

#### Constrained generation / diffusion

- `Constrained Synthesis with Projected Diffusion Models`
  - Status: NeurIPS 2024 Main Conference.
  - Formulation fit: diffusion sampling recast as constrained optimization with
    convex and challenging nonconvex constraints and ODE-style constraints.
  - Use in HomOPT: later-stage generative benchmark, not first implementation.
  - Source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/a28af221f2f70be183afc16797a56b91-Abstract-Conference.html

- `Fast constrained sampling in pre-trained diffusion models`
  - Status: NeurIPS 2025 Poster.
  - Formulation fit: fast sampling under linear and nonlinear constraints.
  - Use in HomOPT: later comparison against projection/optimization-based
    constrained sampling.
  - Source: https://nips.cc/virtual/2025/poster/120010

### Broad search outcome

Recent venue search suggests three active clusters:

1. Hard-feasible neural prediction for constrained optimization.
   - Examples: `PiNet`, `FSNet`, `HardNet`, `HoP`.
   - HomOPT relevance: direct baselines for the parametric learning route.

2. Nonconvex equality/manifold constraints in ML models.
   - Examples: Stiefel/orthogonality, moment-constrained molecules, LoRA
     subspace learning, ODEs on manifolds.
   - HomOPT relevance: clean `h(x)=0` testbeds for ALM vs Hom-ALM.

3. Physics/generative constrained optimization.
   - Examples: DE-constrained optimization, AC-OPF, constrained diffusion.
   - HomOPT relevance: strongest applied story, but higher implementation cost.

### Next plan

1. Implement `StiefelProblem` first.
   - It is the smallest clean nonconvex-equality benchmark.
   - Explicit objective, equality residual, and gradients are straightforward.

2. Implement `DistanceGeometryProblem` second.
   - It provides a NeurIPS 2024 reconstruction task and a natural parametric
     learning route.

3. Add one shared experiment config group for `nonconvex_eq_convex_ineq`.
   - Keep the output metrics aligned with the existing convex and parametric
     routes.

4. Only after those are stable, implement `ACOPFLiteProblem` or
   `DEConstrainedProblemLite`.
   - Keep the first applied version narrow: convex safety inequalities only,
     nonconvex equalities in ALM.

### 2026-04-19 implementation checkpoint: `StiefelProblem`

Implemented `StiefelProblem` in `src/homopt/problems/_stiefel_impl.py` and
exported it through `homopt.problems`.

Current supported deterministic form:

```text
decision:      X in R^{n x p}, flattened as x in R^{n p}

minimize       -tr(X.T @ A_obj @ X)

subject to     vec(X.T @ B_eq @ X - I_p) = 0
               A_ineq x <= b_ineq                 optional
               L <= x <= U                         optional
               ||x||_2 <= norm_radius              optional
```

Implementation details:

- `B_eq` defaults to identity, giving the standard Stiefel constraint
  `X.T @ X = I_p`.
- `constraint_x(x, eq_cons=True)` follows the existing HomOPT layout:
  inequality residuals first, equality residuals last.
- `ineq_cons`, `eq_cons`, `ncon`, and `n_eq` are exposed for the shared
  `ALM` / `Hom-ALM` outer-loop logic.
- The convex side uses only constraint families already supported by
  `GaugeMap`: linear inequalities, box bounds, and SOC norm ball.
- `gradient_lagrangian_x(..., method="explicit")` is implemented and tested
  against autograd.
- `gradient_lagrangian_z(...)` uses autograd through the existing `GaugeMap`.

Initial smoke results:

| Case | Solver | Config sketch | Best equality violation | Final equality violation | Convex inequality violation |
|---|---|---|---:|---:|---:|
| `n=3, p=1` | `Hom-ALM` | `outer=150`, `inner=40`, `penalty=100`, `NAG beta=0.9` | `1.06e-2` | `1.39e-2` | `0` |
| `n=3, p=1` | `ALM` | same primal/dual/penalty config | `1.03e-2` | `1.39e-2` | `0` |
| `n=4, p=2` | `Hom-ALM` | `outer=120`, `inner=30`, `penalty=100`, `NAG beta=0.9` | `8.53e-3` | `2.29e-2` | `0` |

Current interpretation:

- The problem contract and gauge integration are working.
- The convex side is preserved exactly by Hom-ALM because the gauge handles it.
- Equality convergence is still modest, similar to the earlier
  equality-constrained convex tests. The next algorithmic step is best-iterate
  selection and/or a Stiefel-specific feasibility refinement baseline, not more
  problem-layer abstraction.

### 2026-04-19 Stiefel baseline and best-iterate update

Added two implementation pieces:

- `return_best_violation` option for `ALM` and `Hom-ALM`.
  - The penalty-loop recorder now tracks the best violation iterate even when
    full decision trajectories are not stored.
  - The default behavior is unchanged.
  - When `return_best_violation=True`, the optimizer returns the recorded
    best-violation decision as `x_solved`.
- `StiefelRetractionSolver` in `src/homopt/solvers/stiefel.py`.
  - Uses Euclidean objective gradient, tangent-space projection, and QR
    retraction.
  - Currently supports standard Stiefel constraints `X.T @ X = I`; generalized
    `X.T @ B @ X = I` is intentionally not implemented yet.

Comparison on the first smoke cases:

| Case | Method | Objective | Equality violation | Convex inequality violation |
|---|---|---:|---:|---:|
| `n=3, p=1` | `ALM` with best iterate | `-2.0778` | `1.03e-2` | `0` |
| `n=3, p=1` | `Hom-ALM` with best iterate | `-2.1300` | `1.06e-2` | `0` |
| `n=3, p=1` | `StiefelRetractionSolver` | `-3.0000` | `1.19e-7` | `0` |
| `n=4, p=2` | `ALM` with best iterate | `-7.1161` | `1.85e-2` | `0` |
| `n=4, p=2` | `Hom-ALM` with best iterate | `-7.1100` | `2.35e-2` | `0` |
| `n=4, p=2` | `StiefelRetractionSolver` | `-7.0000` | `2.38e-7` | `0` |

Interpretation:

- For pure Stiefel orthogonality, a native retraction baseline is much stronger
  than generic ALM/Hom-ALM.
- Hom-ALM is still useful for the mixed target we care about: nonconvex
  equality plus convex inequality handled by the gauge map.
- The next meaningful Stiefel experiment should add active convex inequalities
  that are not automatically preserved by QR retraction, otherwise the
  retraction baseline dominates for the wrong reason.

### 2026-04-19 active Stiefel inequality smoke

Added tight box-side Stiefel cases to test the mixed-constraint setting:

```text
p=1 case:
  X in R^{3 x 1}
  X.T X = 1
  |X_1| <= 0.2
  ||X||_2 <= 1

p=2 case:
  X in R^{4 x 2}
  X.T X = I_2
  |X_{1,1}| <= 0.2, |X_{1,2}| <= 0.2
  ||vec(X)||_2 <= sqrt(2)
```

These constraints are intentionally active because the objective matrix favors
the first row/coordinate.

| Case | Method | Objective | Equality violation | Convex inequality violation |
|---|---|---:|---:|---:|
| `p=1`, `|x_1|<=0.2` | `ALM` with best iterate | `-2.0554` | `7.32e-3` | `3.65e-3` |
| `p=1`, `|x_1|<=0.2` | `Hom-ALM` with best iterate | `-1.9437` | `0` | `0` |
| `p=1`, `|x_1|<=0.2` | `StiefelRetractionSolver` | `-3.0000` | `1.19e-7` | `8.00e-1` |
| `p=2`, first row bounded | `ALM` with best iterate | `-5.2141` | `9.16e-3` | `6.46e-3` |
| `p=2`, first row bounded | `Hom-ALM` with best iterate | `-5.1186` | `7.62e-4` | `0` |
| `p=2`, first row bounded | `StiefelRetractionSolver` | `-7.0000` | `1.19e-7` | `5.76e-1` |

Interpretation:

- Tight convex inequalities expose the intended tradeoff clearly.
- `StiefelRetractionSolver` is a strong manifold-only optimizer but ignores the
  extra convex side constraints and violates them substantially.
- `ALM` reduces both residual types but still leaves small inequality
  violations under the tested config.
- `Hom-ALM` preserves convex inequalities exactly through the gauge map, while
  equality residual depends on the ALM convergence quality.
- For this mixed Stiefel benchmark, the right baseline pair is:
  `Hom-ALM` versus `StiefelRetraction + inequality penalty/barrier`, not plain
  QR retraction alone.

### 2026-04-19 constrained PCA group-norm implementation

Extended `StiefelProblem` with practical feature-group norm constraints:

```text
group_norm_constraints = [
  {
    "name": "restricted_features",
    "rows": [0, 1, ...],
    "budget": beta
  }
]
```

This adds the convex SOC residual:

```text
g_group(X) = ||X[rows, :]||_F - beta <= 0
```

Implementation details:

- Group constraints are represented through the existing `GaugeMap` SOC
  tensors `G, h, C, d`, so no new mapping class is needed.
- Multiple group constraints are supported.
- `flat_indices` is also accepted for constraints over arbitrary flattened
  entries.
- `create_constrained_pca_stiefel_config(...)` builds the application config
  from either raw data or a covariance matrix.
- The explicit `gradient_lagrangian_x(..., method="explicit")` path now handles
  all Stiefel SOC constraints, including full-radius and group-norm constraints.

Initial constrained PCA smoke:

```text
A = diag(5, 4, 3, 2, 1)
p = 2
restricted rows = [0, 1]
budget = 0.45
```

| Method | Objective | Equality violation | Convex inequality violation | Restricted group norm |
|---|---:|---:|---:|---:|
| `ALM` with best iterate | `-4.1186` | `1.21e-2` | `0` | `9.37e-2` |
| `Hom-ALM` with best iterate | `-2.6883` | `1.61e-1` | `0` | `3.38e-2` |
| `StiefelRetractionSolver` | `-9.0000` | `2.38e-7` | `9.64e-1` | `1.4142` |

Interpretation:

- The group-norm constraint behaves as intended: native Stiefel retraction
  finds the unconstrained top subspace and violates the restricted-feature
  budget heavily.
- The gauge side keeps group-norm inequality violation at zero.
- Current `Hom-ALM` settings are not yet robust for the tighter `p=2`
  constrained-PCA equality; this should be treated as an algorithm/config tuning
  target, not a problem-modeling issue.
- The next fair baseline is `StiefelRetraction + group-norm penalty`:

```text
minimize    -tr(X.T @ A @ X) + rho * max(||X[S, :]||_F - beta, 0)^2
subject to  X.T X = I_p
```
