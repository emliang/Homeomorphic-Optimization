"""Manual profiling utility for Hom gauge gradient reconstruction rules.

The explicit GaugeMap route has three reconstruction rules:

- polynomial: direct derivative of the selected candidate equation.
- implicit: implicit-function reconstruction from the active boundary normal.
- hybrid: closed-form linear/box terms plus implicit SOC/quad terms.

This script compares these routes against autograd for validity and reports the VJP
runtime of each route on synthetic feasible convex sets.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from homopt.mappings import GaugeMap
from homopt.problems import ConvexOptEq


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _synthetic_config(case: str, n_var: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    if case == "socp":
        n_linear, n_soc, n_quad = 0, 90, 0
    elif case == "mixed":
        n_linear, n_soc, n_quad = 20, 40, 40
    else:
        raise ValueError("case must be 'socp' or 'mixed'.")

    Q = np.eye(n_var, dtype=np.float32)
    p = rng.normal(scale=0.05, size=n_var).astype(np.float32)
    A = rng.normal(scale=0.25, size=(n_linear, n_var)).astype(np.float32) if n_linear else None
    b = rng.uniform(0.5, 1.5, size=n_linear).astype(np.float32) if n_linear else None

    k_soc = min(8, n_var)
    G = rng.normal(scale=0.12, size=(n_soc, k_soc, n_var)).astype(np.float32) if n_soc else None
    h = np.zeros((n_soc, k_soc), dtype=np.float32) if n_soc else None
    C = rng.normal(scale=0.02, size=(n_soc, n_var)).astype(np.float32) if n_soc else None
    d = rng.uniform(1.0, 2.0, size=n_soc).astype(np.float32) if n_soc else None

    if n_quad:
        raw = rng.normal(scale=0.08, size=(n_quad, n_var, n_var)).astype(np.float32)
        Qq = np.einsum("mki,mkj->mij", raw, raw) + 1e-3 * np.eye(n_var, dtype=np.float32)[None, :, :]
        pq = rng.normal(scale=0.02, size=(n_quad, n_var)).astype(np.float32)
        bq = rng.uniform(0.5, 1.5, size=n_quad).astype(np.float32)
    else:
        Qq = pq = bq = None

    return {
        "Q": Q,
        "p": p,
        "A": A,
        "b": b,
        "A_eq": None,
        "b_eq": None,
        "Qq": Qq,
        "pq": pq,
        "bq": bq,
        "G": G,
        "h": h,
        "C": C,
        "d": d,
        "L": -5.0 * np.ones(n_var, dtype=np.float32),
        "U": 5.0 * np.ones(n_var, dtype=np.float32),
    }


def _time_vjp(mapping: GaugeMap, z: torch.Tensor, grad_x: torch.Tensor, method: str, warmup: int, repeats: int) -> float:
    device = z.device
    for _ in range(warmup):
        mapping.vjp(z, grad_x, method=method)
    _sync(device)
    start = time.perf_counter()
    for _ in range(repeats):
        mapping.vjp(z, grad_x, method=method)
    _sync(device)
    return (time.perf_counter() - start) / repeats


def _run_case(case: str, n_var: int, batch_size: int, seed: int, device_name: str, warmup: int, repeats: int) -> None:
    device = torch.device(device_name)
    problem = ConvexOptEq(_synthetic_config(case, n_var, seed)).to_device(device)
    x_origin = torch.zeros(n_var, device=device)
    z = torch.randn(batch_size, n_var, device=device) * 0.35
    grad_x = torch.randn(batch_size, n_var, device=device)

    autograd_map = GaugeMap(problem, p_norm=2, x_origin=x_origin)
    implicit_map = GaugeMap(problem, p_norm=2, x_origin=x_origin, explicit_gradient_rule="implicit")
    hybrid_map = GaugeMap(problem, p_norm=2, x_origin=x_origin, explicit_gradient_rule="hybrid")
    polynomial_map = GaugeMap(problem, p_norm=2, x_origin=x_origin, explicit_gradient_rule="polynomial")

    grad_auto = autograd_map.vjp(z, grad_x, method="autograd")
    grad_implicit = implicit_map.vjp(z, grad_x, method="explicit")
    grad_hybrid = hybrid_map.vjp(z, grad_x, method="explicit")
    grad_polynomial = polynomial_map.vjp(z, grad_x, method="explicit")

    rows = [
        ("autograd", 0.0, _time_vjp(autograd_map, z, grad_x, "autograd", warmup, repeats)),
        (
            "implicit",
            float((grad_implicit - grad_auto).abs().max().detach().cpu().item()),
            _time_vjp(implicit_map, z, grad_x, "explicit", warmup, repeats),
        ),
        (
            "polynomial",
            float((grad_polynomial - grad_auto).abs().max().detach().cpu().item()),
            _time_vjp(polynomial_map, z, grad_x, "explicit", warmup, repeats),
        ),
        (
            "hybrid",
            float((grad_hybrid - grad_auto).abs().max().detach().cpu().item()),
            _time_vjp(hybrid_map, z, grad_x, "explicit", warmup, repeats),
        ),
    ]

    print(f"\ncase={case} n_var={n_var} batch={batch_size} device={device}")
    print(f"{'rule':<12} {'max_abs_vs_autograd':>22} {'vjp_time_ms':>14}")
    print("-" * 52)
    for name, err, elapsed in rows:
        print(f"{name:<12} {err:>22.4e} {elapsed * 1e3:>14.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("socp", "mixed", "all"), default="all")
    parser.add_argument("--n-var", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()

    cases = ("socp", "mixed") if args.case == "all" else (args.case,)
    for case in cases:
        _run_case(case, args.n_var, args.batch_size, args.seed, args.device, args.warmup, args.repeats)


if __name__ == "__main__":
    main()
