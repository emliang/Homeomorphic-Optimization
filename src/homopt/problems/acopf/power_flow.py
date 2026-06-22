"""AC power-flow matrix construction from MATPOWER case data."""

from __future__ import annotations

import numpy as np


BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
GS = 4
BS = 5
VM = 7
VA = 8
VMAX = 11
VMIN = 12

GEN_BUS = 0
PG = 1
QG = 2
QMAX = 3
QMIN = 4
PMAX = 8
PMIN = 9

F_BUS = 0
T_BUS = 1
BR_R = 2
BR_X = 3
BR_B = 4
RATE_A = 5
TAP = 8
SHIFT = 9
BR_STATUS = 10
ANGMIN = 11
ANGMAX = 12

COST = 4


def build_bus_index(bus):
    bus_numbers = np.asarray(bus[:, BUS_I], dtype=int)
    return {int(bus_id): idx for idx, bus_id in enumerate(bus_numbers)}


def generator_bus_indices(ppc):
    bus_lookup = build_bus_index(ppc["bus"])
    return np.asarray([bus_lookup[int(bus_id)] for bus_id in ppc["gen"][:, GEN_BUS]], dtype=int)


def branch_bus_indices(ppc):
    bus_lookup = build_bus_index(ppc["bus"])
    branch = ppc["branch"]
    return np.asarray(
        [[bus_lookup[int(row[F_BUS])], bus_lookup[int(row[T_BUS])]] for row in branch],
        dtype=int,
    )


def make_admittance_matrices(ppc):
    """Construct dense Ybus, Yf, and Yt matrices without a PyPower dependency."""

    base_mva = float(ppc["baseMVA"])
    bus = np.asarray(ppc["bus"], dtype=float)
    branch = np.asarray(ppc["branch"], dtype=float)
    nb = int(bus.shape[0])
    nl = int(branch.shape[0])
    bus_lookup = build_bus_index(bus)

    ybus = np.zeros((nb, nb), dtype=np.complex128)
    yf = np.zeros((nl, nb), dtype=np.complex128)
    yt = np.zeros((nl, nb), dtype=np.complex128)
    for idx, row in enumerate(branch):
        if row.shape[0] > BR_STATUS and row[BR_STATUS] == 0:
            continue
        f = bus_lookup[int(row[F_BUS])]
        t = bus_lookup[int(row[T_BUS])]
        series = complex(row[BR_R], row[BR_X])
        if abs(series) == 0:
            ys = 0.0 + 0.0j
        else:
            ys = 1.0 / series
        bc = row[BR_B] if row.shape[0] > BR_B else 0.0
        tap = row[TAP] if row.shape[0] > TAP and row[TAP] != 0 else 1.0
        shift = row[SHIFT] if row.shape[0] > SHIFT else 0.0
        tap = tap * np.exp(1j * np.pi / 180.0 * shift)

        ytt = ys + 1j * bc / 2.0
        yff = ytt / (tap * np.conj(tap))
        yft = -ys / np.conj(tap)
        ytf = -ys / tap

        ybus[f, f] += yff
        ybus[f, t] += yft
        ybus[t, f] += ytf
        ybus[t, t] += ytt
        yf[idx, f] = yff
        yf[idx, t] = yft
        yt[idx, f] = ytf
        yt[idx, t] = ytt

    ybus += np.diag((bus[:, GS] + 1j * bus[:, BS]) / base_mva)
    return ybus, yf, yt


__all__ = [
    "ANGMAX",
    "ANGMIN",
    "BR_STATUS",
    "BUS_TYPE",
    "COST",
    "PD",
    "PG",
    "PMAX",
    "PMIN",
    "QD",
    "QG",
    "QMAX",
    "QMIN",
    "RATE_A",
    "VA",
    "VM",
    "VMAX",
    "VMIN",
    "branch_bus_indices",
    "generator_bus_indices",
    "make_admittance_matrices",
]
