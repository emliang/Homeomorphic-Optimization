"""Small artifact helpers shared by visualization modules."""

from __future__ import annotations

from pathlib import Path


def figure_output_path(path):
    """Normalize figure output paths to PDF."""

    path = Path(path)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg"}:
        return path.with_suffix(".pdf")
    return path


def relative_artifacts(output_dir, paths):
    return {name: str(figure_output_path(path).relative_to(output_dir)) for name, path in paths.items()}


def save_figure(fig, path, *, dpi=None, **savefig_kwargs):
    path = figure_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if dpi is None:
        from .style import PAPER_STYLE

        dpi = PAPER_STYLE["dpi"]
    savefig_kwargs.setdefault("dpi", dpi)
    savefig_kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path, **savefig_kwargs)
    return path


def save_figure_with_suffix(fig, save_path, suffix, *, dpi=None):
    if not save_path:
        return None
    base_path = figure_output_path(save_path)
    suffix = suffix.replace(".png", ".pdf")
    output_path = base_path.with_name(base_path.stem + suffix)
    return save_figure(fig, output_path, dpi=dpi)


__all__ = ["figure_output_path", "relative_artifacts", "save_figure", "save_figure_with_suffix"]
