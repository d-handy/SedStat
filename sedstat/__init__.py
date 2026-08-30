"""Grain size distribution analysis for laser granulometer and sieve/pipette lab data.

This top-level module re-exports only the two symbols needed for the most
common one-liner usage (``from sedstat import compute_all``). Everything
else lives in ``sedstat.core`` — see ``sedstat/core/__init__.py`` for the
full public surface (``Classifier``, ``ClassificationScheme``,
``compute_fractions``, ...).
"""

from sedstat.core import GrainSizeResult, compute_all

__all__ = ["GrainSizeResult", "compute_all"]
