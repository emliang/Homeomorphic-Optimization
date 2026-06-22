"""Shared compact verbose table formatting for iterative optimizers."""

from __future__ import annotations


VERBOSE_COLUMNS = (
    ("outer", "outer", 6),
    ("objective", "obj", 10),
    ("eq_violation", "eqvio", 10),
    ("ineq_violation", "ineqvio", 10),
    ("learning_rate", "lr", 10),
    ("dual_norm", "dual_norm", 10),
    ("penalty", "penalty", 10),
    ("lag_gap", "lag_gap", 10),
    ("iter_time", "iter_time", 10),
)


def as_float(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
        if hasattr(value, "numel"):
            if value.numel() == 0:
                return None
        elif getattr(value, "size", 1) == 0:
            return None
        value = value[0]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def format_value(value):
    scalar = as_float(value)
    if scalar is None:
        return "NA"
    return f"{scalar:.3e}"


def format_table_cell(value, width, *, integer=False):
    if integer:
        text = str(int(value))
    else:
        text = format_value(value)
    return f"{text:>{width}}"


def format_header():
    header = "  ".join(f"{label:>{width}}" for _, label, width in VERBOSE_COLUMNS)
    divider = "  ".join("-" * width for _, _, width in VERBOSE_COLUMNS)
    return header, divider


def format_row(values):
    cells = [
        format_table_cell(values.get(key), width, integer=(key == "outer"))
        for key, _, width in VERBOSE_COLUMNS
    ]
    return "  ".join(cells)


def write_header(write):
    header, divider = format_header()
    write(header)
    write(divider)


def write_row(write, **values):
    write(format_row(values))


def _first_present(mapping, keys, default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def build_row_values(*, eval_payload=None, learning_rate=None, dual_state=None, iter_time=None, extra_metrics=None, **values):
    eval_payload = eval_payload or {}
    extra_metrics = extra_metrics or {}
    dual_norm = values.get("dual_norm")
    if dual_norm is None and dual_state is not None:
        dual_norm = dual_state.detach().norm() if hasattr(dual_state, "detach") else None
    return {
        "objective": values.get("objective", eval_payload.get("objective")),
        "eq_violation": values.get(
            "eq_violation",
            _first_present(eval_payload, ("eq_violation", "equality_violation", "violation")),
        ),
        "ineq_violation": values.get(
            "ineq_violation",
            _first_present(eval_payload, ("ineq_violation", "inequality_violation"), 0.0),
        ),
        "learning_rate": values.get("learning_rate", learning_rate),
        "dual_norm": dual_norm,
        "penalty": values.get("penalty", extra_metrics.get("penalty")),
        "lag_gap": values.get("lag_gap", extra_metrics.get("lag_gap")),
        "iter_time": values.get("iter_time", iter_time),
    }


def maybe_write_row(write, *, enabled, printed_header, iteration, interval, **values):
    if not enabled:
        return printed_header
    interval = max(1, int(interval))
    if iteration != 0 and (iteration + 1) % interval != 0:
        return printed_header
    if not printed_header:
        write_header(write)
        printed_header = True
    write_row(write, outer=iteration + 1, **values)
    return printed_header


def write_iteration_row(
    write,
    *,
    enabled,
    printed_header,
    iteration,
    interval,
    eval_payload=None,
    learning_rate=None,
    dual_state=None,
    iter_time=None,
    extra_metrics=None,
    **values,
):
    row = build_row_values(
        eval_payload=eval_payload,
        learning_rate=learning_rate,
        dual_state=dual_state,
        iter_time=iter_time,
        extra_metrics=extra_metrics,
        **values,
    )
    return maybe_write_row(
        write,
        enabled=enabled,
        printed_header=printed_header,
        iteration=iteration,
        interval=interval,
        **row,
    )


__all__ = [
    "VERBOSE_COLUMNS",
    "as_float",
    "build_row_values",
    "format_header",
    "format_row",
    "format_table_cell",
    "format_value",
    "maybe_write_row",
    "write_header",
    "write_iteration_row",
    "write_row",
]
