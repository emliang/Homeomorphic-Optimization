"""Dataset adapters for parametric AC-OPF instances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import torch

from homopt.problems.opf_cases import load_matpower_case, resolve_pglib_opf_case_path


@dataclass(frozen=True)
class ACOPFDataset:
    """Normalized AC-OPF dataset payload.

    Loads are stored in MW/MVAr and decisions are stored in the archive
    convention ``[Pg, Qg, Vm, Va]`` before problem-level per-unit scaling.
    """

    ppc: dict
    pd: np.ndarray | None = None
    qd: np.ndarray | None = None
    pg: np.ndarray | None = None
    qg: np.ndarray | None = None
    vm: np.ndarray | None = None
    va: np.ndarray | None = None
    max_change_load: float = 0.0
    base_load: float = 1.0
    source_path: str | None = None

    @property
    def has_reference_decisions(self):
        return all(item is not None for item in (self.pg, self.qg, self.vm, self.va))


def load_acopf_dataset(path) -> ACOPFDataset:
    """Load an archived AC-OPF pickle and normalize field names."""

    path_obj = Path(path).expanduser()
    if path_obj.suffix in {".m", ".mat"}:
        return ppc_to_acopf_dataset(load_matpower_case(path_obj), source_path=str(path_obj))
    with path_obj.open("rb") as fh:
        raw = pickle.load(fh)
    if not isinstance(raw, dict):
        raise TypeError(f"AC-OPF dataset must be a pickle dict, got {type(raw).__name__}.")
    return normalize_acopf_dataset(raw, source_path=str(path_obj))


def normalize_acopf_dataset(raw: dict, *, source_path: str | None = None) -> ACOPFDataset:
    ppc = _extract_ppc(raw)
    max_change = _scalar(raw.get("MaxChangeLoad", 0.0))
    base_load = _scalar(raw.get("BaseLoad", 1.0))

    if "Pd" in raw and "Qd" in raw:
        pd = _real_array(raw["Pd"])
        qd = _real_array(raw["Qd"])
    elif "Dem" in raw:
        demand = np.asarray(raw["Dem"])
        pd = np.asarray(np.real(demand), dtype=float)
        qd = np.asarray(np.imag(demand), dtype=float)
    else:
        pd = qd = None

    if all(key in raw for key in ("Pg", "Qg", "Vm", "Va")):
        pg = _real_array(raw["Pg"])
        qg = _real_array(raw["Qg"])
        vm = _real_array(raw["Vm"])
        va = _real_array(raw["Va"])
    elif all(key in raw for key in ("Gen", "Vol")):
        gen = np.asarray(raw["Gen"])
        vol = np.asarray(raw["Vol"])
        pg = np.asarray(np.real(gen), dtype=float)
        qg = np.asarray(np.imag(gen), dtype=float)
        vm = np.asarray(np.abs(vol), dtype=float)
        va = np.asarray(np.angle(vol), dtype=float)
    else:
        pg = qg = vm = va = None

    return ACOPFDataset(
        ppc=ppc,
        pd=_nan_filtered(pd),
        qd=_nan_filtered(qd),
        pg=_nan_filtered(pg),
        qg=_nan_filtered(qg),
        vm=_nan_filtered(vm),
        va=_nan_filtered(va),
        max_change_load=max_change,
        base_load=base_load,
        source_path=source_path,
    )


def load_pglib_acopf_dataset(
    case: int,
    *,
    samples: int = 10000,
    seed: int = 2025,
    load_std: float = 0.05,
    case_name: str | None = None,
    data_dir=None,
    download_if_missing: bool = True,
) -> ACOPFDataset:
    """Load a PGLib-OPF case and synthesize parametric load samples."""

    case_path = resolve_pglib_opf_case_path(
        int(case),
        case_name=case_name,
        data_dir=data_dir,
        download_if_missing=download_if_missing,
    )
    ppc = load_matpower_case(case_path)
    return ppc_to_acopf_dataset(
        ppc,
        samples=samples,
        seed=seed,
        load_std=load_std,
        source_path=str(case_path),
    )


def ppc_to_acopf_dataset(
    ppc: dict,
    *,
    samples: int | None = None,
    seed: int = 2025,
    load_std: float = 0.05,
    source_path: str | None = None,
) -> ACOPFDataset:
    """Build an ACOPF dataset from one MATPOWER/PGLib case.

    PGLib-OPF provides static OPF cases rather than solved parametric load
    archives.  For parametric experiments we perturb each bus load around the
    case nominal load; reference decisions are intentionally absent.
    """

    ppc = _normalize_ppc_dict(ppc)
    bus = np.asarray(ppc["bus"], dtype=float)
    pd_nominal = bus[:, 2]
    qd_nominal = bus[:, 3]
    if samples is None:
        pd = pd_nominal.reshape(1, -1)
        qd = qd_nominal.reshape(1, -1)
    else:
        rng = np.random.default_rng(int(seed))
        n_samples = max(int(samples), 1)
        scale = rng.normal(loc=1.0, scale=float(load_std), size=(n_samples, bus.shape[0]))
        scale = np.clip(scale, 0.0, None)
        pd = pd_nominal.reshape(1, -1) * scale
        qd = qd_nominal.reshape(1, -1) * scale
    return ACOPFDataset(
        ppc=ppc,
        pd=pd,
        qd=qd,
        max_change_load=float(load_std),
        base_load=1.0,
        source_path=source_path,
    )


def _normalize_ppc_dict(ppc: dict) -> dict:
    return {
        "version": ppc.get("version", "2"),
        "baseMVA": float(np.asarray(ppc["baseMVA"]).reshape(-1)[0]),
        "bus": np.asarray(ppc["bus"], dtype=float),
        "gen": np.asarray(ppc["gen"], dtype=float),
        "branch": np.asarray(ppc["branch"], dtype=float),
        "gencost": np.asarray(ppc["gencost"], dtype=float),
    }


def build_acopf_input_tensor(dataset: ACOPFDataset, *, base_mva, device=None, dtype=None):
    if dataset.pd is None or dataset.qd is None:
        raise ValueError("AC-OPF dataset does not contain load samples.")
    inputs = np.concatenate([dataset.pd / float(base_mva), dataset.qd / float(base_mva)], axis=1)
    return torch.as_tensor(inputs, device=device, dtype=dtype)


def build_acopf_reference_tensor(dataset: ACOPFDataset, *, base_mva, device=None, dtype=None):
    if not dataset.has_reference_decisions:
        return None
    decisions = np.concatenate(
        [
            dataset.pg / float(base_mva),
            dataset.qg / float(base_mva),
            dataset.vm,
            dataset.va,
        ],
        axis=1,
    )
    return torch.as_tensor(decisions, device=device, dtype=dtype)


def _extract_ppc(raw):
    if isinstance(raw.get("ppc"), dict):
        return raw["ppc"]
    if isinstance(raw.get("mpc"), dict):
        return raw["mpc"]
    raise ValueError("AC-OPF dataset must contain a MATPOWER-style 'ppc' dict.")


def _real_array(value):
    return np.asarray(np.real(value), dtype=float)


def _scalar(value):
    array = np.asarray(value)
    if array.size == 0:
        return 0.0
    return float(array.reshape(-1)[0])


def _nan_filtered(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"AC-OPF dataset arrays must be two-dimensional, got shape {array.shape}.")
    return array


__all__ = [
    "ACOPFDataset",
    "build_acopf_input_tensor",
    "build_acopf_reference_tensor",
    "load_pglib_acopf_dataset",
    "load_acopf_dataset",
    "normalize_acopf_dataset",
    "ppc_to_acopf_dataset",
]
