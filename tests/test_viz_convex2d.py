import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


def _load_socp_eq_hom_alm_2d_viz_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_dir = repo_root / "scripts" / "hom_alm"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    spec = importlib.util.spec_from_file_location("run_convex_eq_2d", script_dir / "run_convex_eq_2d.py")
    script_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script_module)

    from homopt.experiments.benchmarks import convex_eq_2d as benchmark_module

    benchmark_module.PARAMS = script_module.PARAMS
    return benchmark_module


def test_socp_eq_hom_alm_2d_viz_instance_places_reference_optimum_on_constraint_boundary():
    module = _load_socp_eq_hom_alm_2d_viz_module()
    problem = module._build_problem(module.PARAMS)

    x_boundary = torch.as_tensor(
        module.EXPECTED_BOUNDARY_OPTIMUM,
        dtype=problem.Q.dtype,
        device=problem.Q.device,
    ).view(1, -1)
    x_interior = torch.as_tensor([0.5, 0.5], dtype=problem.Q.dtype, device=problem.Q.device).view(1, -1)
    x_unconstrained = torch.as_tensor(
        module.TOY_OBJECTIVE_CENTER,
        dtype=problem.Q.dtype,
        device=problem.Q.device,
    ).view(1, -1)

    raw_ineq = problem.constraint_x(x_boundary, clip=False, eq_cons=False)
    eq_resid = problem.eq_constraint_x(x_boundary)
    quad_resid = (
        0.5 * torch.sum((x_boundary @ problem.Qq[0]) * x_boundary, dim=-1, keepdim=True)
        + x_boundary @ problem.pq.T
        - problem.bq
    )

    assert module.PARAMS["problem_type"] == "socp_eq"
    assert problem.n_eq == 1
    assert eq_resid.abs().max().item() <= 1e-6
    assert raw_ineq[0, 6].item() == pytest.approx(0.0, abs=1e-6)
    assert raw_ineq[0, 10].item() < 0.0
    assert torch.allclose(problem.A.cpu(), torch.as_tensor(module.TOY_LINEAR_A, dtype=problem.A.dtype))
    assert torch.allclose(problem.b.cpu(), torch.as_tensor(module.TOY_LINEAR_B, dtype=problem.b.dtype))
    assert torch.allclose(problem.Qq[0].cpu(), torch.as_tensor(module.TOY_QUAD_Q, dtype=problem.Qq.dtype))
    assert torch.allclose(problem.pq[0].cpu(), torch.as_tensor(module.TOY_QUAD_P, dtype=problem.pq.dtype))
    assert torch.allclose(problem.bq.cpu(), torch.as_tensor(module.TOY_QUAD_B, dtype=problem.bq.dtype))
    assert torch.allclose(raw_ineq[:, 10:11], quad_resid)
    assert torch.clamp(raw_ineq, min=0).max().item() <= 1e-6
    assert torch.clamp(problem.constraint_x(x_unconstrained, clip=False, eq_cons=False), min=0).max().item() > 1e-3
    assert problem.objective_x(x_boundary).item() < problem.objective_x(x_interior).item()


def test_socp_eq_hom_alm_2d_viz_uses_inequality_only_hom_origin():
    from homopt.experiments.common.convex_reference import prepare_convex_reference_context

    module = _load_socp_eq_hom_alm_2d_viz_module()
    problem = module._build_problem(module.PARAMS)
    reference = prepare_convex_reference_context(
        problem,
        runtime_device="cpu",
        runtime_dtype=problem.Q.dtype,
        need_opt=False,
        smooth=module.PARAMS["common_config"]["smooth"],
        hom_origin_constraints=module.PARAMS["hom_origin_constraints"],
    )
    origin = torch.as_tensor(reference["x_origin"], dtype=problem.Q.dtype, device=problem.Q.device).view(1, -1)
    raw_ineq = problem.constraint_x(origin, clip=False, eq_cons=False)

    assert module.PARAMS["hom_origin_constraints"] == "ineq"
    assert reference["hom_origin_constraints"] == "ineq"
    assert reference["x_origin_eq_violation"] > 1e-3
    assert torch.clamp(raw_ineq, min=0).max().item() <= 1e-6
    assert raw_ineq.max().item() < -1e-3


def test_hal_2d_z_landscape_uses_gauge_mapped_objective():
    from homopt.experiments.common.convex_reference import prepare_convex_reference_context
    from homopt.viz.landscapes import convex_z_landscape_grid

    module = _load_socp_eq_hom_alm_2d_viz_module()
    problem = module._build_problem(module.PARAMS)
    reference = prepare_convex_reference_context(
        problem,
        runtime_device="cpu",
        runtime_dtype=problem.Q.dtype,
        need_opt=False,
        smooth=module.PARAMS["common_config"]["smooth"],
        hom_origin_constraints=module.PARAMS["hom_origin_constraints"],
    )
    hom_map = module._build_hom_map(
        module.PARAMS,
        problem,
        {"reference_origin": reference["x_origin"]},
    )

    Z1, Z2, objective, eq_resid, inside, _ = convex_z_landscape_grid(
        problem,
        hom_map,
        grid_size=13,
        bounds=(-1.0, 1.0),
    )
    points = np.column_stack([Z1.reshape(-1), Z2.reshape(-1)])
    z_tensor = torch.as_tensor(points, dtype=problem.Q.dtype, device=problem.Q.device)
    with torch.no_grad():
        raw_z_objective = problem.objective_x(z_tensor).detach().cpu().numpy().reshape(Z1.shape)
        mapped_objective = problem.objective_x(hom_map.forward(z_tensor)).detach().cpu().numpy().reshape(Z1.shape)
        mapped_eq_resid = problem.eq_constraint_x(hom_map.forward(z_tensor)).detach().cpu().numpy().reshape(Z1.shape)

    assert np.nanmax(np.abs(objective - mapped_objective)) <= 1e-6
    assert np.nanmax(np.abs(eq_resid - mapped_eq_resid)) <= 1e-6
    assert np.max(np.abs(objective[inside] - raw_z_objective[inside])) > 0.25

    unit_dirs_np = np.vstack(([1.0, 0.0], [0.0, 1.0], [1.0, -1.0]))
    unit_dirs_np[-1] = unit_dirs_np[-1] / np.linalg.norm(unit_dirs_np[-1])
    unit_dirs = torch.as_tensor(
        unit_dirs_np,
        dtype=problem.Q.dtype,
        device=problem.Q.device,
    )
    mapped = hom_map.forward(unit_dirs).detach().cpu().numpy()
    center = hom_map.center.detach().cpu().numpy().reshape(-1)
    radii = np.linalg.norm(mapped - center, axis=1)
    assert np.ptp(radii) > 0.15


def test_hal_2d_hom_map_prefers_common_config_smooth():
    module = _load_socp_eq_hom_alm_2d_viz_module()
    problem = module._build_problem(module.PARAMS)
    params = dict(module.PARAMS)
    params["smooth"] = False
    params["common_config"] = {**params.get("common_config", {}), "smooth": True}
    hom_map = module._build_hom_map(
        params,
        problem,
        {"reference_origin": np.zeros(problem.nvar)},
    )

    assert hom_map.smooth is True


def test_paper_style_disables_background_grid_by_default():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from homopt.viz.style import PAPER_STYLE, apply_paper_axis_style

    fig, ax = plt.subplots(1, 1)
    ax.plot([0.0, 1.0], [0.0, 1.0])
    apply_paper_axis_style(ax)

    grid_lines = [*ax.get_xgridlines(), *ax.get_ygridlines()]
    assert PAPER_STYLE["show_grid"] is False
    assert not any(line.get_visible() for line in grid_lines)
    plt.close(fig)


def test_paper_style_sets_pdf_font_and_save_defaults(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    from homopt.viz.artifacts import save_figure
    from homopt.viz.style import PAPER_STYLE, require_matplotlib

    require_matplotlib()
    assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["ps.fonttype"] == 42
    assert mpl.rcParams["font.family"][0] == PAPER_STYLE["font_family"]

    captured = {}
    original_savefig = plt.Figure.savefig

    def capture_savefig(self, *args, **kwargs):
        captured.update(kwargs)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_savefig)
    fig, ax = plt.subplots(1, 1)
    ax.plot([0, 1], [0, 1])
    save_figure(fig, tmp_path / "styled.pdf")
    plt.close(fig)

    assert captured["dpi"] == PAPER_STYLE["dpi"]
    assert captured["bbox_inches"] == "tight"


def test_convex2d_primitives_generate_nonempty_figures(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from homopt.viz.primitives import (
        draw_convex_landscape,
        draw_convex_z_landscape,
        draw_metric_convergence,
        draw_trajectory,
    )

    grid = np.linspace(-1.4, 1.4, 48)
    X, Y = np.meshgrid(grid, grid)
    objective = (X - 0.15) ** 2 + 0.8 * (Y + 0.25) ** 2
    inequality = X**2 + Y**2 - 1.0
    eq_resid = X + Y - 0.2
    equality = np.abs(eq_resid)
    trajectory = np.array([[-1.0, 0.8], [-0.55, 0.48], [-0.12, 0.31], [0.12, 0.12]])

    x_path = tmp_path / "x_space.pdf"
    fig, ax = plt.subplots(1, 1)
    draw_convex_landscape(ax, X, Y, objective, inequality, equality, eq_resid)
    draw_trajectory(ax, trajectory, color="#E83947", marker=".", label="Hom-ALM")
    fig.savefig(x_path)
    plt.close(fig)

    Z1, Z2 = np.meshgrid(grid, grid)
    z_objective = (Z1 - 0.2) ** 2 + (Z2 + 0.1) ** 2
    z_eq_resid = Z1 - 0.5 * Z2
    inside = np.linalg.norm(np.column_stack([Z1.reshape(-1), Z2.reshape(-1)]), axis=1).reshape(Z1.shape) <= 1.0

    z_path = tmp_path / "z_space.pdf"
    fig, ax = plt.subplots(1, 1)
    draw_convex_z_landscape(ax, Z1, Z2, z_objective, z_eq_resid, inside, 2)
    draw_trajectory(ax, trajectory * 0.55, color="#E83947", marker=".", label="Hom-ALM")
    fig.savefig(z_path)
    plt.close(fig)

    metric_path = tmp_path / "objective.pdf"
    traces = {
        "ALM": {
            "objective": np.array([1.2, 0.8, 0.55]),
            "time": np.array([0.0, 0.1, 0.2]),
        },
        "Hom-ALM": {
            "objective": np.array([1.0, 0.58, 0.32]),
            "time": np.array([0.0, 0.08, 0.17]),
        },
    }
    metric_paths = draw_metric_convergence(traces, "objective", "Objective value", metric_path)

    assert set(metric_paths) == {"iteration", "iteration_logx", "time", "time_logx"}
    assert metric_paths["iteration_logx"].name == "objective_by_iter_logx.pdf"
    assert metric_paths["time_logx"].name == "objective_by_time_logx.pdf"
    for path in (x_path, z_path, *metric_paths.values()):
        assert path.exists()
        assert path.stat().st_size > 1000


def test_metric_time_logx_keeps_initial_zero_time_point(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib.axes import Axes

    from homopt.viz.primitives import draw_metric_convergence

    captured_x = []
    original_plot = Axes.plot

    def capture_plot(self, *args, **kwargs):
        if len(args) >= 2:
            captured_x.append(np.asarray(args[0], dtype=float).copy())
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture_plot)
    traces = {
        "PGD": {
            "objective": np.array([2.0, 1.0, 0.5]),
            "time": np.array([0.0, 0.1, 0.2]),
        }
    }
    draw_metric_convergence(traces, "objective", "Objective value", tmp_path / "objective.pdf")

    assert any(
        x_data.size == 3
        and np.all(x_data > 0.0)
        and x_data[0] == pytest.approx(0.01)
        and x_data[1] == pytest.approx(0.1)
        for x_data in captured_x
    )


def test_metric_reference_objective_label_is_not_in_legend(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib.axes import Axes

    from homopt.viz.primitives import draw_metric_convergence

    legend_labels = []
    reference_annotations = []
    original_legend = Axes.legend
    original_annotate = Axes.annotate

    def capture_legend(self, *args, **kwargs):
        _, labels = self.get_legend_handles_labels()
        legend_labels.append(list(labels))
        return original_legend(self, *args, **kwargs)

    def capture_annotate(self, text, *args, **kwargs):
        reference_annotations.append(text)
        return original_annotate(self, text, *args, **kwargs)

    monkeypatch.setattr(Axes, "legend", capture_legend)
    monkeypatch.setattr(Axes, "annotate", capture_annotate)
    traces = {
        "ALM": {
            "objective": np.array([2.0, 1.5, 1.1]),
            "time": np.array([0.0, 0.1, 0.2]),
        }
    }
    draw_metric_convergence(
        traces,
        "objective",
        "Objective value",
        tmp_path / "objective.pdf",
        reference_value=1.0,
        reference_label="MOSEK",
    )

    assert reference_annotations == ["MOSEK"] * 4
    assert legend_labels
    assert all("ALM" in labels for labels in legend_labels)
    assert all("MOSEK" not in labels for labels in legend_labels)


def test_reference_objective_from_payload_accepts_top_level_value():
    from homopt.viz.traces import reference_objective_from_payload

    assert reference_objective_from_payload({"reference_objective": 0.125}) == pytest.approx(0.125)


def test_metric_convergence_can_hide_algorithm_legend(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib.axes import Axes

    from homopt.viz.primitives import draw_metric_convergence

    legend_calls = []

    def fail_on_legend(self, *args, **kwargs):
        del self, args, kwargs
        legend_calls.append(True)
        raise AssertionError("legend should not be drawn when show_legend=False")

    monkeypatch.setattr(Axes, "legend", fail_on_legend)
    traces = {
        "ALM": {
            "objective": np.array([2.0, 1.5, 1.1]),
            "time": np.array([0.0, 0.1, 0.2]),
        }
    }
    paths = draw_metric_convergence(
        traces,
        "objective",
        "Objective value",
        tmp_path / "objective.pdf",
        show_legend=False,
    )

    assert not legend_calls
    assert all(path.exists() for path in paths.values())


def test_record_time_axis_rejects_mismatched_lengths():
    from homopt.viz.traces import record_time_axis

    with pytest.raises(ValueError, match="Cannot build time axis"):
        record_time_axis({"iter_time": [0.1]}, 3)


def test_record_time_axis_uses_nested_solver_timing_trace():
    from homopt.viz.traces import record_time_axis

    time = record_time_axis({"extras": {"iter_time": [0.1, 0.2]}}, 3)

    assert np.allclose(time, [0.0, 0.1, 0.3])


def test_record_time_axis_rejects_mismatched_nested_solver_timing_trace():
    from homopt.viz.traces import record_time_axis

    with pytest.raises(ValueError, match="Cannot build time axis"):
        record_time_axis({"extras": {"iter_time": [0.1]}}, 3)


def test_inn_training_visualizations_generate_artifacts(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    from homopt.viz.inn_training import save_inn_training_visualizations

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    training_record = {
        "penalty_list": [1.0, 0.6, 0.2],
        "volume_list": [0.2, 0.35, 0.5],
        "dist_list": [0.8, 0.7, 0.55],
    }
    obj_traj = torch.tensor([1.2, 0.8, 0.5])
    cons_traj = torch.tensor([0.2, 0.05, 0.01])
    decision_traj = torch.tensor([[-0.8, 0.6], [-0.3, 0.25], [0.1, 0.0]])

    artifacts = save_inn_training_visualizations(
        tmp_path,
        artifact_dir,
        training_record,
        obj_traj,
        cons_traj,
        decision_traj,
    )
    expected = {
        "training_metrics",
        "inn_pgd_convergence",
        "decision_trajectory_2d",
    }
    assert expected <= set(artifacts)
    for artifact in artifacts.values():
        path = tmp_path / artifact
        assert path.exists()
        assert path.stat().st_size > 1000


def test_inn_training_visualizations_can_skip_optimizer_diagnostics(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    from homopt.viz.inn_training import save_inn_training_visualizations

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    artifacts = save_inn_training_visualizations(
        tmp_path,
        artifact_dir,
        {
            "penalty_list": [1.0, 0.5],
            "volume_list": [0.1, 0.2],
            "dist_list": [0.7, 0.6],
        },
        torch.tensor([1.0, 0.5]),
        torch.tensor([0.1, 0.0]),
        torch.tensor([[0.0, 0.0], [0.1, 0.1]]),
        plot_optimizer_diagnostics=False,
    )

    assert set(artifacts) == {"training_metrics"}
    assert (tmp_path / artifacts["training_metrics"]).exists()


def test_qcqp_2d_comparison_visualizations_include_outer_per_iter_runtime(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    from matplotlib.axes import Axes

    import homopt.viz.inn_training as inn_viz
    from homopt.viz.inn_training import save_qcqp_2d_comparison_visualizations

    bar_kwargs = []
    original_bar = Axes.bar

    def capture_bar(self, *args, **kwargs):
        bar_kwargs.append(dict(kwargs))
        return original_bar(self, *args, **kwargs)

    qcqp_grid_sizes = []
    latent_grid_sizes = []

    def _fake_landscape(grid_size):
        grid = np.linspace(-1.0, 1.0, int(grid_size))
        X, Y = np.meshgrid(grid, grid)
        return {
            "X": X,
            "Y": Y,
            "objective": X + Y,
            "residual": X * X + Y * Y - 0.5,
        }

    def _skip_qcqp_landscape(*args, **kwargs):
        del args
        qcqp_grid_sizes.append(kwargs.get("grid_size"))
        assert "eval_batch_size" not in kwargs
        return _fake_landscape(kwargs.get("grid_size"))

    def _skip_latent_landscape(*args, **kwargs):
        del args
        latent_grid_sizes.append(kwargs.get("grid_size"))
        assert "eval_batch_size" not in kwargs
        landscape = _fake_landscape(kwargs.get("grid_size"))
        landscape["lim"] = 1.5
        return landscape

    monkeypatch.setattr(Axes, "bar", capture_bar)
    monkeypatch.setattr(inn_viz, "_compute_qcqp_landscape", _skip_qcqp_landscape)
    monkeypatch.setattr(inn_viz, "_compute_latent_landscape", _skip_latent_landscape)

    artifacts = save_qcqp_2d_comparison_visualizations(
        output_dir=tmp_path,
        artifact_root_path=tmp_path / "artifacts",
        problem=object(),
        model=None,
        input_params=torch.zeros(1, 1),
        objective_params=None,
        inn_record={
            "x_trajectory": np.asarray([[0.0, 0.0], [0.1, 0.0]]),
            "z_trajectory": np.asarray([[0.0, 0.0], [0.1, 0.0]]),
            "obj_trajectory": np.asarray([1.0, 0.5]),
            "cons_trajectory": np.asarray([0.1, 0.0]),
            "per_iter_time": [0.2],
        },
        iterative_records={
            "ALM": [
                {
                    "record": {
                        "x_traj": np.asarray([[0.0, 0.0], [0.0, 0.1]]),
                        "obj_traj": np.asarray([1.2, 0.7]),
                        "cons_traj": np.asarray([0.2, 0.01]),
                        "outer_iter_time": [0.05, 0.07],
                    }
                }
            ]
        },
        n_samples=1,
        instance_idx=0,
        plot_objective_gap=False,
        plot_gap_plus_violation=False,
    )

    runtime_artifact = artifacts["qcqp_2d_outer_per_iter_time"]
    assert runtime_artifact == "artifacts/instances/inst0/qcqp_2d_outer_per_iter_time.pdf"
    assert (tmp_path / runtime_artifact).exists()
    assert (tmp_path / runtime_artifact).stat().st_size > 1000
    assert any(np.any(np.asarray(kwargs.get("yerr", []), dtype=float) > 0.0) for kwargs in bar_kwargs)
    assert qcqp_grid_sizes == [inn_viz.QCQP_TRAJECTORY_GRID_SIZE]
    assert latent_grid_sizes == [inn_viz.QCQP_LATENT_TRAJECTORY_GRID_SIZE]


def test_latent_landscape_clips_to_sampling_region_for_outlier_start():
    import homopt.viz.inn_training as inn_viz

    outlier_traj = np.asarray([[-7.0, 1.8], [0.0, 1.0], [0.1, 0.9]], dtype=float)
    landscape = inn_viz._compute_latent_landscape(
        problem=object(),
        model=None,
        input_params=None,
        traj=outlier_traj,
        grid_size=17,
    )

    lim = inn_viz.QCQP_LATENT_TRAJECTORY_VIEW_LIM
    assert landscape["lim"] == lim
    assert np.isclose(float(np.min(landscape["X"])), -lim)
    assert np.isclose(float(np.max(landscape["X"])), lim)
    assert np.isclose(float(np.min(landscape["Y"])), -lim)
    assert np.isclose(float(np.max(landscape["Y"])), lim)


def test_qcqp_2d_comparison_visualizations_reject_missing_instance_records(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)

    from homopt.viz.inn_training import save_qcqp_2d_comparison_visualizations

    with pytest.raises(ValueError, match="cannot plot instance 1"):
        save_qcqp_2d_comparison_visualizations(
            output_dir=tmp_path,
            artifact_root_path=tmp_path / "artifacts",
            problem=object(),
            model=None,
            input_params=torch.zeros(1, 1),
            objective_params=None,
            inn_record={
                "x_trajectory": np.zeros((4, 2)),
                "z_trajectory": np.zeros((4, 2)),
                "obj_trajectory": np.zeros(4),
                "cons_trajectory": np.zeros(4),
                "per_iter_time": [0.1],
            },
            iterative_records={"ALM": [{"record": {"obj_traj": np.zeros(2), "cons_traj": np.zeros(2)}}]},
            n_samples=2,
            instance_idx=1,
        )


def test_inn_decision_trajectory_can_draw_qcqp_landscape(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from matplotlib.axes import Axes

    import homopt.viz.inn_training as inn_viz
    from homopt.viz.inn_training import save_inn_training_visualizations
    from homopt.viz.style import PAPER_STYLE

    class TinyQCQP:
        nvar = 2
        fixed_L = np.array([-2.0, -2.0], dtype=np.float32)
        fixed_U = np.array([2.0, 2.0], dtype=np.float32)
        device = torch.device("cpu")

        def objective(self, x):
            return (x[:, 0] - 0.2) ** 2 + 0.7 * (x[:, 1] + 0.1) ** 2

        def check_feasibility(self, input_batch, x):
            del input_batch
            return ((x[:, :1] - 0.1) ** 2 + (x[:, 1:2] + 0.1) ** 2) - 1.2

    contour_kwargs = []
    contourf_kwargs = []
    plot_kwargs = []
    patch_alphas = []
    original_contour = Axes.contour
    original_contourf = Axes.contourf
    original_plot = Axes.plot
    original_add_patch = Axes.add_patch

    def capture_contour(self, *args, **kwargs):
        contour_kwargs.append(dict(kwargs))
        return original_contour(self, *args, **kwargs)

    def capture_contourf(self, *args, **kwargs):
        contourf_kwargs.append(dict(kwargs))
        return original_contourf(self, *args, **kwargs)

    def capture_plot(self, *args, **kwargs):
        plot_kwargs.append(dict(kwargs))
        return original_plot(self, *args, **kwargs)

    def capture_add_patch(self, patch):
        patch_alphas.append(patch.get_alpha())
        return original_add_patch(self, patch)

    monkeypatch.setattr(Axes, "contour", capture_contour)
    monkeypatch.setattr(Axes, "contourf", capture_contourf)
    monkeypatch.setattr(Axes, "plot", capture_plot)
    monkeypatch.setattr(Axes, "add_patch", capture_add_patch)
    monkeypatch.setattr(inn_viz, "QCQP_TRAJECTORY_GRID_SIZE", 80)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    decision_traj = torch.tensor([[-1.0, 0.7], [-0.4, 0.3], [0.05, -0.05]])

    artifacts = save_inn_training_visualizations(
        tmp_path,
        artifact_dir,
        {"penalty_list": [], "volume_list": [], "dist_list": []},
        torch.tensor([1.0, 0.7, 0.4]),
        torch.tensor([0.3, 0.1, 0.02]),
        decision_traj,
        problem=TinyQCQP(),
        input_params=torch.zeros(1, 1),
    )

    assert "decision_trajectory_2d" in artifacts
    assert PAPER_STYLE["inn_feasible_fill_alpha"] in patch_alphas or any(
        kwargs.get("colors") == [PAPER_STYLE["inn_feasible_fill_color"]]
        and kwargs.get("alpha") == PAPER_STYLE["inn_feasible_fill_alpha"]
        for kwargs in contourf_kwargs
    )
    assert any(kwargs.get("colors") == PAPER_STYLE["objective_contour_color"] for kwargs in contour_kwargs)
    assert any(kwargs.get("color") == PAPER_STYLE["inn_constraint_boundary_color"] for kwargs in plot_kwargs) or any(
        kwargs.get("colors") == PAPER_STYLE["inn_constraint_boundary_color"]
        or kwargs.get("colors") == [PAPER_STYLE["inn_constraint_boundary_color"]]
        for kwargs in contour_kwargs
    )


def test_mdh_mapping_visualization_uses_paper_constraint_style(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    from homopt.viz.mdh import visualize_mdh_mapping_transformation
    from homopt.viz.style import PAPER_STYLE

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, z, condition):
            shift = torch.cat([0.45 * condition[:, :1], -0.2 * condition[:, :1]], dim=1)
            return 1.25 * z + shift

    class TinyData:
        nvar = 2
        feasibility_batch_sizes = []

        def generate_problem_samples(self, n_samples, seed=2030):
            del seed
            values = torch.linspace(-1.0, 1.0, steps=n_samples).view(n_samples, 1)
            return values, None

        def scale(self, input_params, points):
            del input_params
            return points

        def complete_partial(self, input_params, points):
            del input_params
            return points

        def check_feasibility(self, input_params, points):
            self.feasibility_batch_sizes.append(int(points.shape[0]))
            center = torch.cat([0.45 * input_params[:, :1], -0.2 * input_params[:, :1]], dim=1)
            return ((points - center) ** 2).sum(dim=1, keepdim=True) - 1.9

    data = TinyData()
    scatter_kwargs = []
    contour_kwargs = []
    contourf_kwargs = []
    plot_kwargs = []
    patch_alphas = []
    original_scatter = Axes.scatter
    original_contour = Axes.contour
    original_contourf = Axes.contourf
    original_plot = Axes.plot
    original_add_patch = Axes.add_patch

    def capture_scatter(self, *args, **kwargs):
        captured = dict(kwargs)
        if len(args) >= 2:
            captured.update({"x": np.asarray(args[0]), "y": np.asarray(args[1])})
        scatter_kwargs.append(captured)
        return original_scatter(self, *args, **kwargs)

    def capture_contour(self, *args, **kwargs):
        contour_kwargs.append(dict(kwargs))
        return original_contour(self, *args, **kwargs)

    def capture_contourf(self, *args, **kwargs):
        contourf_kwargs.append(dict(kwargs))
        return original_contourf(self, *args, **kwargs)

    def capture_plot(self, *args, **kwargs):
        plot_kwargs.append(dict(kwargs))
        return original_plot(self, *args, **kwargs)

    def capture_add_patch(self, patch):
        patch_alphas.append(patch.get_alpha())
        return original_add_patch(self, patch)

    monkeypatch.setattr(Axes, "scatter", capture_scatter)
    monkeypatch.setattr(Axes, "contour", capture_contour)
    monkeypatch.setattr(Axes, "contourf", capture_contourf)
    monkeypatch.setattr(Axes, "plot", capture_plot)
    monkeypatch.setattr(Axes, "add_patch", capture_add_patch)

    fig = visualize_mdh_mapping_transformation(
        TinyModel(),
        data,
        save_path=tmp_path / "mdh_mapping.pdf",
        n_samples=24,
        grid_resolution=32,
    )
    plt.close(fig)

    assert (tmp_path / "mdh_mapping_mdh_mapping_visualization.pdf").exists()
    assert scatter_kwargs == []
    assert data.feasibility_batch_sizes == [32 * 32, 32 * 32, 32 * 32]
    assert any(kwargs.get("color") == PAPER_STYLE["inn_unit_boundary_color"] for kwargs in plot_kwargs)
    assert any(kwargs.get("color") in PAPER_STYLE["inn_instance_colors"] for kwargs in plot_kwargs)
    assert PAPER_STYLE["inn_feasible_fill_alpha"] in patch_alphas or any(
        kwargs.get("colors") == ["white", PAPER_STYLE["inn_feasible_fill_color"]]
        and kwargs.get("alpha") == PAPER_STYLE["inn_feasible_fill_alpha"]
        for kwargs in contourf_kwargs
    )
    assert any(
        kwargs.get("color") == PAPER_STYLE["inn_constraint_boundary_color"]
        and kwargs.get("linewidth") == PAPER_STYLE["constraint_linewidth"]
        for kwargs in plot_kwargs
    ) or any(
        kwargs.get("colors") == [PAPER_STYLE["inn_constraint_boundary_color"]]
        and kwargs.get("linewidths") == PAPER_STYLE["constraint_linewidth"]
        for kwargs in contour_kwargs
    )
