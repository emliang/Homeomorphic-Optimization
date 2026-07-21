# Gauge Mapping Notes

`GaugeMap` maps an unconstrained latent point `z` into a convex inequality set by
radially scaling from an interior center `x0`. It is used for inequality
constraints only; equality constraints are handled by ALM/PALM layers.

## Constraint Input Contract

The map reads standard tensor attributes from the problem object. A missing
attribute or `None` disables that family.

| Constraint family | Problem attributes | Constraint form | Candidate computation |
| --- | --- | --- | --- |
| Linear | `A`, `b` | `A x <= b` | Linear boundary candidate from slack at `x0`. |
| Box lower | `L` | `L <= x` | Per-coordinate lower-bound candidate. |
| Box upper | `U` | `x <= U` | Per-coordinate upper-bound candidate. |
| SOC | `G`, `h`, `C`, `d` | `||G_i x + h_i|| <= C_i x + d_i` | Quadratic root along each ray. |
| Quadratic | `Qq`, `pq`, `bq` | `0.5 x^T Q_i x + pq_i^T x <= bq_i` | Quadratic root along each ray. |
| General convex | `general_convex_constraint_x(x)`, `general_convex_constraint_gradient_x(x)` | `g_i(x) <= 0` | Per-constraint bisection along each ray, then implicit boundary gradient. |

Static tensors derived from these attributes are cached by `(device, dtype)` in
`GaugeMap._get_static_terms(...)`.

## Forward Map

For each batch point:

| Step | Quantity | Description |
| --- | --- | --- |
| Direction | `r = ||z||`, `u = z / r` | Work on the ray from center in direction `u`. |
| Candidates | `beta_j(u)` | Every active constraint family returns positive boundary candidates. |
| Active boundary | `beta = max_j beta_j(u)` | The largest `beta` corresponds to the nearest limiting boundary in the map's scaling convention. |
| Output | `x = x0 + z / beta` | Implemented as `scaling = 1 / beta`, `x = scaling * z + center`. |

With `smooth=False`, the map uses the hard maximum candidate. With
`smooth=True`, it applies log-sum-exp only over candidates tied within
`smooth_tie_tol` of the hard maximum.

## Backward / VJP

| Mode | Computation | Notes |
| --- | --- | --- |
| `method="autograd"` | PyTorch differentiates the forward computation. | Simpler but may rebuild graph work. |
| `method="explicit"` | Forward stores `u`, `r`, `beta`, `scaling`, and `best_grad_u`; backward calls `explicit_vjp_from_state`. | Main fast path for Hom-PGD. |

The explicit VJP differentiates:

```text
x = x0 + scaling(z) z
scaling = 1 / beta(u)
u = z / ||z||
```

So the returned gradient has a direct term from `scaling * grad_x` and a radial
correction through `d beta / d u`.

## Explicit Gradient Rules

| Rule | Meaning | Typical use |
| --- | --- | --- |
| `polynomial` | Uses closed-form derivative of each candidate formula. | Default explicit path. |
| `implicit` | Uses the active boundary constraint's implicit derivative. | Robust reference for nonlinear families. |
| `hybrid` | Uses simple closed forms where possible and implicit-style formulas for nonlinear families. | Diagnostic alternative. |

For general convex constraints without closed-form radial roots, the map uses
explicit bisection to find the boundary point `x0 + rho u`. Its gradient is not
a numerical finite-difference gradient. It uses the implicit formula
`d beta / d u = beta * grad g(x_boundary) / <grad g(x_boundary), u>`.

## Runtime State Reuse

`GaugeMap.forward(..., method="explicit", return_state=True)` returns both
`x` and the explicit state. Hom-PGD reuses this state when computing
problem-level `gradient_objective_z(...)` and proximal terms, avoiding duplicate
gauge candidate construction within the same iteration.

## Current Limits

| Limit | Reason |
| --- | --- |
| Explicit gauge requires `p_norm=2`. | Candidate formulas and VJP are implemented for Euclidean rays. |
| Equality constraints are not part of `GaugeMap`. | They are handled by ALM/PALM penalty and dual updates. |
| Unsupported constraint families should fail explicitly. | The active path avoids silent numerical fallback. |
| General convex constraints require a strictly feasible center. | Bisection assumes `g_i(x0) < 0` before searching outward along each ray. |
