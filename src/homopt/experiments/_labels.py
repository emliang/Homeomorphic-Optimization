"""Shared display labels for experiment plots."""

from __future__ import annotations


CONVEX_METHOD_LABELS = {
    "PGD": "PGD",
    "Hom-PGD": "Hom-PGD",
    "FW": "FW",
    "RD": "RD",
    "Penalty": "Penalty",
    "Prox-Penalty": "PPP",
    "ALM": "ALM",
    "Prox-ALM": "PALM",
    "Hom-ALM": "Hom-ALM",
    "Prox-Hom-ALM": "Hom-PALM",
    "Penalty-EQ": "Penalty-Eq",
    "Prox-Penalty-EQ": "PPP-Eq",
    "ALM-EQ": "ALM-Eq",
    "Prox-ALM-EQ": "PALM-Eq",
}

STIEFEL_METHOD_LABELS = {
    **CONVEX_METHOD_LABELS,
    "StiefelRetraction": "Retraction-Penalty",
    "StiefelRetractionPenalty": "Retraction-Penalty",
    "StiefelRetractionALM": "Retraction-ALM",
    "ALM-EQ-PGD": "ALM-Eq-PGD",
}


__all__ = [
    "CONVEX_METHOD_LABELS",
    "STIEFEL_METHOD_LABELS",
]
