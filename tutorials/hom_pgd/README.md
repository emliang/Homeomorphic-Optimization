# Hom-PGD Tutorials

These notebooks introduce the gauge-map workflow behind Hom-PGD, from geometry
to trajectories and convergence records. Run them in order from the repository
root with `jupyter lab tutorials/hom_pgd`.

| Notebook | Focus | Reference solver |
| --- | --- | --- |
| `01_poly_gauge_geometry.ipynb` | Polytope, gauge center, and original/latent geometry | ConvexSolver (displayed as MOSEK) |
| `02_star_hom_pgd_trajectory.ipynb` | Star domain and Hom-PGD trajectory | IPOPT |
| `03_poly_star_intersection.ipynb` | Polytope-star intersection | IPOPT |
| `04_hom_pgd_convergence.ipynb` | Shared-instance PGD/RD/Hom-PGD comparison | IPOPT |

Every notebook uses the public Hom-PGD experiment facade. It constructs one
instance, obtains one explicit reference solution, and derives visualizations
from the exact in-memory records saved by that run. The notebooks do not
reconstruct a second problem or implement private solver, metric, or plotting
paths.

Each notebook writes PDFs to its matching directory:

```text
results/figs/tutorials/hom_pgd/<notebook-name>/
```
