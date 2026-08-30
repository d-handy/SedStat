"""Bimodality detection for grain size distributions: peak detection and the
Pearson/Rock bimodality coefficient.

Reference: Rock, N. M. S. (1988). Numerical geology. Springer.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def detect_modes(
    classes_um: list[float] | np.ndarray,
    values: list[float] | np.ndarray,
    min_pct: float = 2.0,
    smooth_window: int = 3,
    min_prominence: float | None = None,
    min_separation: int | None = None,
) -> list[tuple[float, float]]:
    """Return (size_µm, volume_%) for each significant local maximum.

    Peak search itself is delegated to ``scipy.signal.find_peaks`` (a
    strict-local-maximum scan, same definition this function used to
    hand-roll) rather than reimplemented, so ``min_prominence``/
    ``min_separation`` come for free instead of needing their own bespoke
    logic. One edge case differs from the old hand-rolled scan: a
    perfectly flat plateau of equal neighbouring values is reported by
    ``find_peaks`` as one peak at the plateau's midpoint, whereas the old
    strict ``>`` comparison never flagged any point inside a plateau at
    all. Real volume-% data is essentially never exactly flat across
    several classes, so this is a documented, accepted difference, not a
    behavior change worth guarding against.

    Args:
        classes_um: Upper class boundaries in µm, length N.
        values: Volume % per class, length N.
        min_pct: Minimum volume % (of the *raw*, unsmoothed value at a
            candidate peak's own class) a peak must have to qualify as a
            mode.
        smooth_window: Moving-average window size applied before peak
            search. Set to 1 to skip smoothing.
        min_prominence: Minimum topographic prominence (in the smoothed
            curve's own volume-% units) a peak must have, passed straight
            to ``find_peaks``. ``None`` (the default) applies no
            prominence filter.
        min_separation: Minimum index distance between two reported
            peaks, passed straight to ``find_peaks`` as ``distance``.
            ``None`` (the default) applies no separation constraint.

    Returns:
        List of (size_µm, volume_%) tuples, sorted by grain size ascending.
    """
    v = np.asarray(values, dtype=float)
    c = np.asarray(classes_um, dtype=float)

    if smooth_window > 1 and len(v) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(v, kernel, mode="same")
    else:
        smoothed = v.copy()

    peak_indices, _ = find_peaks(smoothed, prominence=min_prominence, distance=min_separation)

    # min_pct is checked against the *raw* value at each candidate peak's
    # index, not the smoothed one — same convention the original
    # hand-rolled scan used (peak *location* comes from the smoothed
    # curve, but a peak's reported "how big is it" stays true to the
    # actual measured data).
    modes = [(float(c[i]), float(v[i])) for i in peak_indices if v[i] >= min_pct]
    return sorted(modes, key=lambda t: t[0])


def bimodality_coefficient(skewness: float, kurtosis: float) -> float:
    """Pearson bimodality coefficient (BC).

    BC = (skewness² + 1) / kurtosis

    Values above 0.555 suggest bimodality (the BC of a uniform distribution).
    Pass logarithmic-moment statistics for standard sediment applications.

    Returns NaN if kurtosis is zero.
    """
    if kurtosis == 0:
        return float("nan")
    return (skewness**2 + 1.0) / kurtosis
