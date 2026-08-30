"""Core grain-size computation — the pure NumPy statistical engine.

No GUI or plotting dependency: everything here works with a bare
``pip install sedstat`` (no extras). This module is the package's primary
public surface for programmatic use — ``sedstat.gui`` is one *consumer*
of it, not a prerequisite.

Individual method functions (``moments_arithmetic``, ``folk_ward_logarithmic``,
...) and the many ``classify_*`` backward-compatibility functions are kept in
their own submodules (``sedstat.core.statistics``,
``sedstat.core.classification``) for advanced use; this module re-exports
only the entry points most callers need for a normal analysis workflow.
"""

from __future__ import annotations

from sedstat.core.bimodality import bimodality_coefficient, detect_modes
from sedstat.core.classification import ClassificationScheme, Classifier
from sedstat.core.fractions import compute_fractions
from sedstat.core.percentiles import compute_percentiles
from sedstat.core.results import (
    FolkWardStats,
    Fractions,
    GrainSizeResult,
    MomentStats,
    Percentiles,
)
from sedstat.core.statistics import compute_all, compute_percentile
from sedstat.core.stokes import stokes_settling_diameter_um

__all__ = [
    "ClassificationScheme",
    "Classifier",
    "FolkWardStats",
    "Fractions",
    "GrainSizeResult",
    "MomentStats",
    "Percentiles",
    "bimodality_coefficient",
    "compute_all",
    "compute_fractions",
    "compute_percentile",
    "compute_percentiles",
    "detect_modes",
    "stokes_settling_diameter_um",
]
