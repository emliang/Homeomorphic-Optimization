"""MATPOWER/PGLib-OPF case loading utilities."""

from __future__ import annotations

from pathlib import Path
import re
import urllib.request

import numpy as np


PGLIB_OPF_REPO_RAW_URL = "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master"
PGLIB_OPF_CASE_FILES = {
    30: ("pglib_opf_case30_ieee.m",),
    57: ("pglib_opf_case57_ieee.m",),
    118: ("pglib_opf_case118_ieee.m",),
    200: ("pglib_opf_case200_activ.m",),
    300: ("pglib_opf_case300_ieee.m",),
    500: ("pglib_opf_case500_goc.m",),
    793: ("pglib_opf_case793_goc.m",),
    1354: ("pglib_opf_case1354_pegase.m",),
    2000: ("pglib_opf_case2000_goc.m",),
}


def repo_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data"


def default_pglib_opf_data_dir() -> Path:
    return repo_data_dir() / "pglib-opf"


def pglib_case_filename(num_bus: int, case_name: str | None = None) -> str:
    if case_name:
        name = str(case_name).strip()
        return name if name.endswith(".m") else f"{name}.m"
    candidates = PGLIB_OPF_CASE_FILES.get(int(num_bus))
    if not candidates:
        return f"pglib_opf_case{int(num_bus)}.m"
    return candidates[0]


def resolve_pglib_opf_case_path(
    num_bus: int,
    *,
    case_name: str | None = None,
    data_dir=None,
    download_if_missing: bool = True,
) -> Path:
    """Resolve a PGLib-OPF MATPOWER case path, downloading it when requested."""

    root = Path(data_dir).expanduser() if data_dir is not None else default_pglib_opf_data_dir()
    root.mkdir(parents=True, exist_ok=True)
    filename = pglib_case_filename(num_bus, case_name=case_name)
    direct = root / filename
    if direct.exists():
        return direct

    local_matches = sorted(root.glob(f"pglib_opf_case{int(num_bus)}_*.m"))
    if local_matches:
        return local_matches[0]

    if not download_if_missing:
        raise FileNotFoundError(
            f"PGLib-OPF case for {int(num_bus)} buses is missing under {root}. "
            "Set download_if_missing=True or place the MATPOWER .m case there."
        )

    url = f"{PGLIB_OPF_REPO_RAW_URL}/{filename}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            direct.write_bytes(response.read())
    except Exception as exc:  # pragma: no cover - network availability varies
        raise FileNotFoundError(
            f"Could not download PGLib-OPF case {filename} from {url}. "
            f"Place the file under {root} or check network access."
        ) from exc
    return direct


def load_pglib_opf_case(
    num_bus: int,
    *,
    case_name: str | None = None,
    data_dir=None,
    download_if_missing: bool = True,
) -> dict:
    """Load a PGLib-OPF MATPOWER case into a PYPOWER-style dict."""

    path = resolve_pglib_opf_case_path(
        num_bus,
        case_name=case_name,
        data_dir=data_dir,
        download_if_missing=download_if_missing,
    )
    return load_matpower_case(path)


def load_matpower_case(path) -> dict:
    """Parse a MATPOWER `.m` or `.mat` case into a normalized dict."""

    path = Path(path).expanduser()
    if path.suffix == ".mat":
        return _load_matpower_mat(path)
    if path.suffix == ".m":
        return _parse_matpower_m_file(path)
    raise ValueError(f"Unsupported OPF case file extension: {path.suffix}")


def _load_matpower_mat(path: Path) -> dict:
    try:
        import scipy.io
    except ImportError as exc:  # pragma: no cover - dependency availability varies
        raise ImportError("Loading MATPOWER .mat cases requires scipy.") from exc

    raw = scipy.io.loadmat(path)
    struct = raw.get("mpc", raw.get("ppc"))
    if struct is None:
        raise ValueError(f"MATPOWER .mat case {path} must contain 'mpc' or 'ppc'.")
    val = struct[0, 0]
    return {
        "version": str(np.asarray(val["version"]).reshape(-1)[0]),
        "baseMVA": float(np.asarray(val["baseMVA"]).reshape(-1)[0]),
        "bus": np.asarray(val["bus"], dtype=float),
        "gen": np.asarray(val["gen"], dtype=float),
        "branch": np.asarray(val["branch"], dtype=float),
        "gencost": np.asarray(val["gencost"], dtype=float),
    }


def _parse_matpower_m_file(path: Path) -> dict:
    text = path.read_text()
    text = _strip_matlab_comments(text)
    version_match = re.search(r"mpc\.version\s*=\s*['\"]([^'\"]+)['\"]\s*;", text)
    base_match = re.search(r"mpc\.baseMVA\s*=\s*([0-9eE.+-]+)\s*;", text)
    if base_match is None:
        raise ValueError(f"MATPOWER case {path} does not define mpc.baseMVA.")
    return {
        "version": version_match.group(1) if version_match else "2",
        "baseMVA": float(base_match.group(1)),
        "bus": _parse_matpower_matrix(text, "bus", path),
        "gen": _parse_matpower_matrix(text, "gen", path),
        "branch": _parse_matpower_matrix(text, "branch", path),
        "gencost": _parse_matpower_matrix(text, "gencost", path),
    }


def _strip_matlab_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lines.append(line.split("%", 1)[0])
    return "\n".join(lines)


def _parse_matpower_matrix(text: str, name: str, path: Path) -> np.ndarray:
    match = re.search(rf"mpc\.{re.escape(name)}\s*=\s*\[(.*?)\]\s*;", text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"MATPOWER case {path} does not define mpc.{name}.")
    rows = []
    for row_text in match.group(1).split(";"):
        row_text = row_text.strip()
        if not row_text:
            continue
        values = [float(item) for item in row_text.split()]
        rows.append(values)
    if not rows:
        raise ValueError(f"MATPOWER case {path} has an empty mpc.{name} matrix.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"MATPOWER case {path} has ragged mpc.{name} rows.")
    return np.asarray(rows, dtype=float)


__all__ = [
    "PGLIB_OPF_CASE_FILES",
    "PGLIB_OPF_REPO_RAW_URL",
    "default_pglib_opf_data_dir",
    "load_matpower_case",
    "load_pglib_opf_case",
    "pglib_case_filename",
    "repo_data_dir",
    "resolve_pglib_opf_case_path",
]
