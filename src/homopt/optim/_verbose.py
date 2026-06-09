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


__all__ = [
    "VERBOSE_COLUMNS",
    "as_float",
    "format_header",
    "format_row",
    "format_table_cell",
    "format_value",
    "write_header",
    "write_row",
]
