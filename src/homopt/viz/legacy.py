"""Backward-compatible plotting helpers for legacy Hom-PGD experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .artifacts import save_figure
from .primitives import draw_trajectory
from .style import (
    ALGORITHM_COLORS,
    ALGORITHM_LINE_STYLES,
    PAPER_STYLE,
    apply_paper_axis_style,
    require_matplotlib,
    set_axis_labels,
    style_legend,
)

_DEFAULT_COLORS = ALGORITHM_COLORS
_DEFAULT_LINE_STYLES = ALGORITHM_LINE_STYLES


def _setup_base_plot(problem, hom_map):
    plt = require_matplotlib()
    fig1, ax1 = plt.subplots(1, 1, figsize=(5.3, 5))
    fig2, ax2 = plt.subplots(1, 1, figsize=(5.3, 5))

    x = np.linspace(-2.5, 2.5, 300)
    y = np.linspace(-2.5, 2.5, 300)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X, Y], axis=-1).reshape(-1, 2)

    theta = np.linspace(0, 2 * np.pi, 300)
    boundary = np.column_stack((np.cos(theta), np.sin(theta)))
    boundary_tensor = torch.as_tensor(boundary, dtype=torch.float32)
    points_tensor = torch.as_tensor(points, dtype=torch.float32)
    with torch.no_grad():
        K = hom_map.forward(boundary_tensor).detach().cpu().numpy()
        Z_trans = problem.objective_x(hom_map.forward(points_tensor)).detach().cpu().numpy().reshape(X.shape)
        Z_orig = problem.objective_x(points_tensor).detach().cpu().numpy().reshape(X.shape)
    Z_trans = Z_trans - Z_trans.min()
    Z_orig = Z_orig - Z_orig.min()

    max_val = max(Z_trans.max(), Z_orig.max(), 1e-3)
    levels = np.logspace(-2, np.log10(max_val), 15)

    return (fig1, fig2), (ax1, ax2), boundary, K, X, Y, Z_trans, Z_orig, levels


def _plot_contours_and_boundary(ax1, ax2, boundary, K, X, Y, Z_trans, Z_orig, levels):
    contour1 = ax1.contour(
        X,
        Y,
        Z_trans,
        colors="black",
        levels=levels,
        alpha=0.78,
        linewidths=PAPER_STYLE["objective_contour_linewidth"],
    )
    contour2 = ax2.contour(
        X,
        Y,
        Z_orig,
        colors="black",
        levels=levels,
        alpha=0.78,
        linewidths=PAPER_STYLE["objective_contour_linewidth"],
    )
    ax1.plot(
        boundary[:, 0],
        boundary[:, 1],
        "b-",
        label="Unit Ball B",
        linewidth=PAPER_STYLE["constraint_linewidth"],
        alpha=0.72,
    )
    ax1.fill(boundary[:, 0], boundary[:, 1], "b", alpha=0.1)
    ax2.plot(
        K[:, 0],
        K[:, 1],
        "r-",
        label="Transformed Set K",
        linewidth=PAPER_STYLE["constraint_linewidth"],
        alpha=0.72,
    )
    ax2.fill(K[:, 0], K[:, 1], "r", alpha=0.1)
    ax1.clabel(contour1, inline=True, fontsize=10, fmt="%.2f")
    ax2.clabel(contour2, inline=True, fontsize=10, fmt="%.2f")
    return contour1, contour2


def _setup_axes(ax1, ax2):
    set_axis_labels(ax1, r"$z_1$", r"$z_2$", fontweight="bold")
    set_axis_labels(ax2, r"$x_1$", r"$x_2$", fontweight="bold")
    for ax in (ax1, ax2):
        ax.set_aspect("equal", adjustable="box")
        apply_paper_axis_style(ax)


def plot_both_spaces(problem, hom_map):
    plt = require_matplotlib()
    (fig1, fig2), (ax1, ax2), boundary, K, X, Y, Z_trans, Z_orig, levels = _setup_base_plot(problem, hom_map)
    _plot_contours_and_boundary(ax1, ax2, boundary, K, X, Y, Z_trans, Z_orig, levels)
    _setup_axes(ax1, ax2)
    plt.tight_layout()
    return fig1, fig2, ax1, ax2


def _trajectory_array(trajectory):
    if hasattr(trajectory, "detach"):
        trajectory = trajectory.detach().cpu()
    return np.asarray(trajectory, dtype=float)


def plot_optimization_trajectory(problem, hom_map, trajectory, space="z", label=None, save_dir=None):
    plt = require_matplotlib()
    if space == "z":
        z_trajectory = trajectory.detach()
        x_trajectory = hom_map.forward(z_trajectory)
    else:
        x_trajectory = trajectory.detach()
        z_trajectory = hom_map.inverse(x_trajectory)

    save_dir_path = Path("pics/ToyExample/" if save_dir is None else save_dir)
    fig1, fig2, ax1, ax2 = plot_both_spaces(problem, hom_map)
    color = _DEFAULT_COLORS.get(label, "#54A24B")

    if label in {"Hom-PGD", "Hom-ALM"} or space == "z":
        draw_trajectory(
            ax1,
            _trajectory_array(z_trajectory),
            color=color,
            marker=".",
            label="Trajectory",
            start_label="Start",
            final_label="End",
        )
        style_legend(ax1.legend(fontsize=PAPER_STYLE["legend_fontsize"]))
        save_figure(fig1, save_dir_path / "{0}_tran_{1}.pdf".format(problem, label))
        plt.close(fig1)
    else:
        plt.close(fig1)

    draw_trajectory(
        ax2,
        _trajectory_array(x_trajectory),
        color=color,
        marker=".",
        label="Trajectory",
        start_label="Start",
        final_label="End",
    )
    style_legend(ax2.legend(fontsize=PAPER_STYLE["legend_fontsize"]))
    save_figure(fig2, save_dir_path / "{0}_orig_{1}.pdf".format(problem, label))
    plt.close(fig2)


def _plot_method(ax, x_values, y_values, method, *, linewidth=None, label=None):
    ax.plot(
        x_values,
        y_values,
        linestyle=_DEFAULT_LINE_STYLES.get(method, "-"),
        color=_DEFAULT_COLORS.get(method),
        alpha=0.9,
        linewidth=PAPER_STYLE["curve_linewidth"] if linewidth is None else linewidth,
        label=label,
    )


def _legacy_curve_plot(
    plt,
    method_list,
    results,
    output_path,
    *,
    value_fn,
    x_fn,
    xlabel,
    ylabel,
    xscale=None,
    yscale=None,
    ylim=None,
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for method in method_list:
        data = np.asarray(value_fn(method), dtype=float)
        _plot_method(ax, x_fn(method, data), data, method)
    set_axis_labels(ax, xlabel, ylabel)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    if ylim is not None:
        ax.set_ylim(*ylim)
    for method in method_list:
        _plot_method(ax, [], [], method, linewidth=PAPER_STYLE["legend_linewidth"], label=method)
    style_legend(ax.legend(fontsize=16))
    apply_paper_axis_style(ax)
    save_figure(fig, output_path)
    plt.close(fig)


def _iteration_axis(_method, data):
    return np.arange(1, data.shape[0] + 1)


def _time_axis(method, _data, results):
    return np.cumsum([0] + list(results[method]["iter_time"]))


def vis(method_list, results, path, solver=False):
    plt = require_matplotlib()
    del solver

    path = str(path)
    _legacy_curve_plot(
        plt,
        method_list,
        results,
        path + "_convergence_rate.pdf",
        value_fn=lambda method: results[method]["obj_gap"],
        x_fn=_iteration_axis,
        xlabel="num of iteration",
        ylabel=r"optimality gap $|f_k-f^*|/|f^*|$",
        xscale="log",
        yscale="log",
    )
    _legacy_curve_plot(
        plt,
        method_list,
        results,
        path + "_convergence_total_time.pdf",
        value_fn=lambda method: results[method]["obj_gap"],
        x_fn=lambda method, data: _time_axis(method, data, results),
        xlabel="running time (s)",
        ylabel=r"optimality gap $|f_k-f^*|/|f^*|$",
        yscale="log",
    )
    _legacy_curve_plot(
        plt,
        method_list,
        results,
        path + "_convergence_cons.pdf",
        value_fn=lambda method: results[method]["cons_traj"],
        x_fn=lambda method, data: _time_axis(method, data, results),
        xlabel="running time (s)",
        ylabel="constraint violation",
        yscale="log",
        ylim=(1e-5, 1e1),
    )
    _legacy_curve_plot(
        plt,
        method_list,
        results,
        path + "_solution_convergence_rate.pdf",
        value_fn=lambda method: results[method]["obj_gap"] + results[method]["cons_traj"],
        x_fn=_iteration_axis,
        xlabel="num of iteration",
        ylabel=r"$|f_k-f^*|/|f^*|\;+\;\|[g(x_k)]_{+}\|_{\infty}$",
        xscale="log",
        yscale="log",
    )
    _legacy_curve_plot(
        plt,
        method_list,
        results,
        path + "_solution_convergence_total_time.pdf",
        value_fn=lambda method: results[method]["obj_gap"] + results[method]["cons_traj"],
        x_fn=lambda method, data: _time_axis(method, data, results),
        xlabel="running time (s)",
        ylabel=r"$|f_k-f^*|/|f^*|\;+\;\|[g(x_k)]_{+}\|_{\infty}$",
        yscale="log",
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    mean_list = []
    std_list = []
    for method in method_list:
        iter_time = np.array(results[method]["iter_time"])
        mean_list.append(np.mean(iter_time))
        std_list.append(np.std(iter_time))
    ax.bar(
        method_list,
        mean_list,
        color=[_DEFAULT_COLORS.get(method, "C{0}".format(method_list.index(method))) for method in method_list],
        alpha=0.9,
        yerr=std_list,
        capsize=5,
    )
    set_axis_labels(ax, "", "per-iteration time (s)")
    ax.set_yscale("log")
    apply_paper_axis_style(ax)
    plt.tight_layout()
    save_figure(fig, path + "_per_iter_time.pdf")
    plt.close(fig)


__all__ = [
    "plot_both_spaces",
    "plot_optimization_trajectory",
    "vis",
]
