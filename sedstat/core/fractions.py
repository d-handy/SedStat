"""Aggregate a grain size distribution into Wentworth fractions (clay/silt/sand/gravel).

The silt/sand boundary is 63 µm for GRADISTAT/Folk & Ward/ISO 14688, and
50 µm for USDA — see ``sedstat.core.classification`` for scheme details.
"""

from __future__ import annotations

import numpy as np

from sedstat.core.classification import ClassificationScheme
from sedstat.core.results import Fractions

# Stored as (lower_inclusive_µm, upper_exclusive_µm, Fractions_field_name)
_BOUNDS_WENTWORTH: list[tuple[float, float, str]] = [
    (0.0, 2.0, "clay"),
    (2.0, 6.3, "fine_silt"),
    (6.3, 20.0, "medium_silt"),
    (20.0, 63.0, "coarse_silt"),  # silt/sand boundary at 63 µm
    (63.0, 200.0, "fine_sand"),
    (200.0, 630.0, "medium_sand"),
    (630.0, 2000.0, "coarse_sand"),
    (2000.0, np.inf, "gravel"),
]

_BOUNDS_USDA: list[tuple[float, float, str]] = [
    (0.0, 2.0, "clay"),
    (2.0, 6.3, "fine_silt"),
    (6.3, 20.0, "medium_silt"),
    (20.0, 50.0, "coarse_silt"),  # silt/sand boundary at 50 µm (USDA)
    (50.0, 200.0, "fine_sand"),
    (200.0, 630.0, "medium_sand"),
    (630.0, 2000.0, "coarse_sand"),
    (2000.0, np.inf, "gravel"),
]

_FRACTION_BOUNDS: dict[ClassificationScheme, list[tuple[float, float, str]]] = {
    ClassificationScheme.GRADISTAT: _BOUNDS_WENTWORTH,
    ClassificationScheme.FOLK_WARD_1957: _BOUNDS_WENTWORTH,
    ClassificationScheme.USDA: _BOUNDS_USDA,
    ClassificationScheme.ISO14688: _BOUNDS_WENTWORTH,
}


def compute_fractions(
    upper_bounds_um: np.ndarray,
    values: np.ndarray,
    scheme: ClassificationScheme = ClassificationScheme.GRADISTAT,
) -> Fractions:
    """Aggregate values into Wentworth fractions.

    Args:
        upper_bounds_um: Upper class boundary in µm for each class, length N.
        values: Weight / volume percentage per class, length N.
        scheme: Controls the silt/sand boundary (63 µm for
            GRADISTAT/Wentworth, 50 µm for USDA).

    Returns:
        Fractions: Percentages summing to 100 %.
    """
    upper_bounds_um = np.asarray(upper_bounds_um, dtype=float)
    values = np.asarray(values, dtype=float)

    bounds = _FRACTION_BOUNDS[scheme]
    totals: dict[str, float] = {name: 0.0 for _, _, name in bounds}

    for ub, val in zip(upper_bounds_um, values, strict=True):
        for lo, hi, name in bounds:
            if lo <= ub < hi:
                totals[name] += float(val)
                break

    total = sum(totals.values())
    if total == 0:
        raise ValueError("Total is zero — all values are zero or no classes match known fractions.")

    scale = 100.0 / total
    return Fractions(**{name: totals[name] * scale for _, _, name in bounds})
