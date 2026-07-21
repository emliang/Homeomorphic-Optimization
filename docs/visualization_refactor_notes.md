# Visualization Refactor Notes

## Goal

Build a unified plotting codebase for HomOPT experiments so paper figures share one visual language and future visualization work does not duplicate style, trace parsing, landscape evaluation, or artifact saving logic.

The target is not only "working plots"; figures should be publication-ready by default:

- Large readable labels and ticks matching the adjusted notebook style.
- Consistent algorithm colors and line styles across Hom-PGD, Hom-ALM, ALM baselines, INN/MDH plots, and CvxINN diagnostics.
- Clear feasible/infeasible semantics: feasible regions use a soft green fill, constraint boundaries use strong boundary colors, equality constraints are visibly distinct.
- Stable figure dimensions, DPI, frame width, grid weight, marker sizes, and legend styling.
- Minimal plotting logic in experiment runners.

## Current Pain Points

- `src/homopt/viz/_convex2d_impl.py` mixes style constants, trace parsing, metric computation, x-space landscapes, z-space landscapes, metric plots, trajectory plots, and artifact orchestration.
- `src/homopt/viz/plots.py` keeps legacy Hom-PGD plotting APIs and overlaps with the newer convex 2D plotting logic.
- `src/homopt/viz/_mdh_impl.py` and `src/homopt/viz/_coninn_impl.py` each define their own matplotlib import, savefig behavior, axis styling, and feasible-region drawing.
- `src/homopt/experiments/_benchmark_common.py` still contains plotting code for INN training; experiment modules should call visualization utilities, not own plotting details.
- Trace records use several naming conventions (`x_traj`, `z_traj`, `x_trajectory`, `latent_trajectory`, `obj_traj`, `obj_trajectory`), so visualizers repeatedly normalize the same concepts.

## Paper Style Contract

These defaults should live in one style module and be reused everywhere:

- DPI: `300`
- Axis label fontsize: `18`
- Tick label fontsize: `16`
- Legend fontsize: `13-14` for dense multi-method plots, `14-16` for single/legacy panels.
- Axis spine linewidth: about `1.35`
- Major tick width/length: about `1.25 / 5`
- Grid: gray, alpha `0.25`, linewidth `0.8`
- Main curve linewidth: about `2.6`
- Trajectory linewidth: about `2.8`
- Objective contours: black, linewidth about `0.95`
- Constraint/equality boundaries: linewidth about `2.4`
- Start marker: circle, black edge, size about `7`
- Final marker: star, black edge, size about `12`

Color semantics:

- Hom methods: red `#E83947`
- Equality boundary: blue `#0072B2`
- Inequality boundary / ball boundary: red `#E83947`
- Feasible region fill: soft green `#54A24B` with low alpha
- Baselines: fixed colors in `ALGORITHM_COLORS`

## Target Module Structure

```text
src/homopt/viz/
  style.py          # paper style, palettes, require_matplotlib, axis/legend helpers
  artifacts.py      # save_figure, relative_artifacts, artifact path helpers
  traces.py         # canonical trace parsing and metric extraction
  landscapes.py     # x/z 2D grid and constraint/objective evaluation
  primitives.py     # low-level drawing primitives: metrics, landscapes, trajectories
  convex2d.py       # high-level convex 2D artifact orchestration
  inn_training.py   # INN training curves and trajectory artifacts
  mdh.py            # MDH high-level functions using shared primitives
  coninn.py         # CvxINN high-level functions using shared primitives
  legacy.py         # backward-compatible old Hom-PGD plotting entrypoints
```

The current public API exposed by `homopt.viz` should remain stable during migration.

## Execution Plan

### Phase 1: No-Behavior Shared Infrastructure

- Extract style constants and matplotlib helpers to `viz/style.py`.
- Extract artifact helpers to `viz/artifacts.py`.
- Extract trace normalization and convex metric computation to `viz/traces.py`.
- Update `_convex2d_impl.py`, `plots.py`, `_mdh_impl.py`, `_coninn_impl.py`, and `_benchmark_common.py` to import shared helpers where low risk.
- Keep existing public names in `homopt.viz.__init__`.
- Verify with unit tests and `scripts/experiments/run_socp_eq_hom_alm_2d_viz.py`.

### Phase 2: Landscapes and Primitives

- Move x-space and z-space grid construction into `viz/landscapes.py`.
- Introduce `draw_landscape`, `draw_trajectory`, `draw_metric_convergence`, and `draw_unit_ball` in `viz/primitives.py`.
- Refactor convex 2D x/z trajectory plots to use primitives.
- Add smoke tests for nonempty PDF figure generation.

Status: completed on 2026-04-22.

Implemented module boundaries:

- `src/homopt/viz/landscapes.py` owns x-space plot bounds, x-space objective/constraint grid evaluation, and z-space grid evaluation through the Hom map.
- `src/homopt/viz/primitives.py` owns x-space/z-space landscape drawing, Lp unit-ball drawing, trajectory lines/start/final markers, and metric convergence panels.
- `src/homopt/viz/_convex2d_impl.py` stays at the orchestration layer: it builds traces, calls landscape/primitives helpers, and saves per-method trajectory and metric artifacts.
- `tests/test_viz_convex2d.py` adds a lightweight smoke test that writes nonempty x-space, z-space, and metric convergence PDF figures.

### Phase 3: Move Experiment-Layer Plotting Out

- Move `save_inn_training_visualizations` from `_benchmark_common.py` to `viz/inn_training.py`.
- Keep a thin compatibility wrapper in `_benchmark_common.py`.
- Apply the paper style to INN training curves and decision trajectory plots.

Status: completed on 2026-04-22.

Implemented module boundaries:

- `src/homopt/viz/inn_training.py` owns INN training metric curves, INN-PGD convergence plots, and 2D decision trajectory artifacts.
- `src/homopt/experiments/_benchmark_common.py` keeps only a thin compatibility wrapper, so existing parametric benchmark imports remain valid.
- `homopt.viz.save_inn_training_visualizations` is exposed through the visualization namespace.
- `tests/test_viz_convex2d.py` covers both the new visualization entrypoint and the `_benchmark_common.py` wrapper.

### Phase 4: MDH and CvxINN Style Unification

- Replace local `_require_matplotlib`, `_save_figure`, axis/grid/legend defaults in `_mdh_impl.py` and `_coninn_impl.py`.
- Use shared feasible-region and trajectory primitives where applicable.
- Keep their high-level APIs unchanged.

Status: completed on 2026-04-22 for shared matplotlib import, savefig, axis frame/grid, and legend styling.

Implemented cleanup:

- `src/homopt/viz/artifacts.py` now provides the single figure-saving path, including `save_figure_with_suffix` for MDH-style derived filenames.
- `src/homopt/viz/_mdh_impl.py` uses shared `require_matplotlib`, `save_figure`, `save_figure_with_suffix`, `apply_paper_axis_style`, `set_axis_labels`, and `style_legend`.
- `src/homopt/viz/_coninn_impl.py` uses shared `require_matplotlib`, `save_figure`, axis labels, frame/grid, and legend styling.
- Public MDH and CvxINN function names and return shapes are unchanged.

### Phase 5: Legacy Cleanup

- Move `plots.py` compatibility functions to a legacy module or keep `plots.py` as a wrapper.
- Make old Hom-PGD plots call the same trace/style primitives used by Hom-ALM.
- Remove duplicated style constants from any remaining modules.

Status: completed on 2026-04-22.

Implemented cleanup:

- `src/homopt/viz/legacy.py` owns backward-compatible Hom-PGD plotting implementation.
- `src/homopt/viz/plots.py` is now only a compatibility import wrapper for the old module path.
- `src/homopt/viz.__init__` imports legacy public helpers from `legacy.py`.
- Legacy Hom-PGD/Hom-ALM trajectory plotting now uses `draw_trajectory`.
- Legacy convergence plots now save through `save_figure`, apply shared legend/frame/grid style, and share one internal curve-plot helper instead of repeating six near-identical plotting blocks.
- A redundancy scan now leaves `fig.savefig` only in `src/homopt/viz/artifacts.py`, the intended savefig boundary.

## Verification Checklist

After each phase:

- `python -m py_compile` on touched visualization and experiment modules.
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_package_layout.py tests/test_script_entrypoints.py`
- For convex 2D changes: `conda run --no-capture-output -n ML python scripts/experiments/run_socp_eq_hom_alm_2d_viz.py`
- Inspect representative generated figures:
  - `hal_2d_objective_convergence.pdf`
  - `hal_2d_trajectories_x_space.pdf`
  - `hal_2d_trajectories_z_space.pdf`
- Run full tests before finalizing a phase.

## Verification Log

### 2026-04-22

Phase 2 completed and verified with:

- `python -m py_compile src/homopt/viz/_convex2d_impl.py src/homopt/viz/landscapes.py src/homopt/viz/primitives.py tests/test_viz_convex2d.py`
- `conda run --no-capture-output -n ML python scripts/experiments/run_socp_eq_hom_alm_2d_viz.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_viz_convex2d.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_package_layout.py tests/test_script_entrypoints.py tests/test_viz_convex2d.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests`

Results:

- `run_socp_eq_hom_alm_2d_viz.py` completed with feasible output and regenerated nonempty objective, equality/inequality/full violation, x-space trajectory, z-space trajectory, and individual trajectory artifacts under `results/socp_eq_hom_alm_2d_viz/artifacts/`.
- Targeted package/script/viz tests: `9 passed`.
- Full tests: `240 passed, 845 warnings`.

Phase 3, Phase 4, and legacy cleanup pass completed and verified with:

- `python -m py_compile src/homopt/viz/artifacts.py src/homopt/viz/primitives.py src/homopt/viz/_convex2d_impl.py src/homopt/viz/inn_training.py src/homopt/viz/plots.py src/homopt/viz/_mdh_impl.py src/homopt/viz/_coninn_impl.py src/homopt/viz/__init__.py src/homopt/experiments/_benchmark_common.py tests/test_viz_convex2d.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_viz_convex2d.py tests/test_package_layout.py tests/test_script_entrypoints.py`
- `conda run --no-capture-output -n ML python scripts/experiments/run_socp_eq_hom_alm_2d_viz.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests`

Results:

- Redundancy scan: `_require_matplotlib`, `_save_figure`, `plt.savefig`, hard-coded `dpi=150`, and hard-coded `dpi=180` are removed from visualization modules; direct `fig.savefig` remains only inside `artifacts.py`.
- `run_socp_eq_hom_alm_2d_viz.py` completed with feasible output and regenerated the convex 2D artifacts.
- Targeted package/script/viz tests: `10 passed, 1 warning`.
- Full tests: `241 passed, 845 warnings`.

Phase 5 legacy module split completed and verified with:

- `python -m py_compile src/homopt/viz/legacy.py src/homopt/viz/plots.py src/homopt/viz/__init__.py tests/test_viz_convex2d.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_viz_convex2d.py tests/test_package_layout.py tests/test_script_entrypoints.py`
- `conda run --no-capture-output -n ML python scripts/experiments/run_socp_eq_hom_alm_2d_viz.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests`

Results:

- `plots.py` is now a compatibility wrapper; implementation moved to `legacy.py`.
- `vis()` legacy convergence plotting now uses a shared helper and has smoke coverage for all six output PDF figures.
- Redundancy scan still leaves direct `fig.savefig` only in `artifacts.py`.
- `run_socp_eq_hom_alm_2d_viz.py` completed with feasible output and regenerated the convex 2D artifacts.
- Targeted package/script/viz tests: `11 passed, 1 warning`.
- Full tests: `242 passed, 845 warnings`.

Boundary-instance follow-up completed and verified with:

- `python -m py_compile scripts/experiments/run_socp_eq_hom_alm_2d_viz.py tests/test_viz_convex2d.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_viz_convex2d.py`
- `conda run --no-capture-output -n ML python scripts/experiments/run_socp_eq_hom_alm_2d_viz.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_package_layout.py tests/test_script_entrypoints.py tests/test_viz_convex2d.py`

Results:

- The 2D Hom-ALM visualization instance now uses `x1 + x2 = 1` plus an active
  quadratic ball centered at `(0.5, 0.5)`. The objective minimizer is placed
  outside that ball along the equality line, so the reference optimum is the
  equality/quadratic-boundary point near `(1.1010, -0.1010)`.
- The script regenerated objective, equality/inequality/full violation,
  x-space trajectory, z-space trajectory, and individual trajectory artifacts.
- Targeted viz test: `4 passed, 1 warning`.
- Targeted package/script/viz tests: `12 passed`.

Gauge-space contour follow-up completed and verified with:

- `python -m py_compile scripts/experiments/run_socp_eq_hom_alm_2d_viz.py src/homopt/viz/primitives.py tests/test_viz_convex2d.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_viz_convex2d.py`
- `conda run --no-capture-output -n ML python scripts/experiments/run_socp_eq_hom_alm_2d_viz.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_package_layout.py tests/test_script_entrypoints.py tests/test_viz_convex2d.py`

Results:

- The active 2D quadratic constraint is now an anisotropic ellipsoid rather
  than a ball. This avoids the degenerate case where the gauge map is just a
  constant radial scaling around the hom center and `f(H(z))` looks like the
  x-space quadratic objective.
- The z-space contour code masks objective contours outside the latent unit
  ball and has regression coverage that checks the plotted scalar field equals
  `f(H(z))`, not `f(z)`.
- The gauge map diagnostic for unit directions gives mapped radii roughly
  `0.606`, `0.876`, and `0.919`, confirming direction-dependent deformation.
- Targeted viz test: `5 passed`.
- Targeted package/script/viz tests: `13 passed`.

Grid-style follow-up completed and verified with:

- `python -m py_compile src/homopt/viz/style.py tests/test_viz_convex2d.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_viz_convex2d.py`
- `conda run --no-capture-output -n ML python scripts/experiments/run_socp_eq_hom_alm_2d_viz.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_package_layout.py tests/test_script_entrypoints.py tests/test_viz_convex2d.py`

Results:

- `PAPER_STYLE["show_grid"]` is now the shared background-grid switch and is
  set to `False` by default.
- `apply_paper_axis_style(...)` reads the shared style config when no explicit
  per-call override is passed and disables gridlines by default.
- The 2D Hom-ALM visualization artifacts were regenerated with the no-grid
  default.
- Targeted viz test: `6 passed, 1 warning`.
- Targeted package/script/viz tests: `14 passed`.

Inequality-only hom-origin follow-up completed and verified with:

- `python -m py_compile scripts/experiments/run_socp_eq_hom_alm_2d_viz.py tests/test_viz_convex2d.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_viz_convex2d.py`
- `conda run --no-capture-output -n ML python scripts/experiments/run_socp_eq_hom_alm_2d_viz.py`
- `conda run --no-capture-output -n ML python -m pytest -q tests/test_package_layout.py tests/test_script_entrypoints.py tests/test_viz_convex2d.py`

Results:

- `run_socp_eq_hom_alm_2d_viz.py` now sets `hom_origin_constraints = "ineq"`.
  Hom-ALM therefore builds the gauge map from an inequality-interior point and
  leaves equality handling to the ALM terms.
- The regenerated run used hom origin approximately `(0.4524, 0.4202)`, with
  equality violation about `1.27e-1`; all raw inequality residuals were
  negative.
- The 2D visualization artifacts were regenerated with this off-equality hom
  center.
- Targeted viz test: `7 passed, 50 warnings`.
- Targeted package/script/viz tests: `15 passed, 50 warnings`.
