"""Percentile interpolation from a cumulative grain size curve."""

from __future__ import annotations

import numpy as np

from sedstat.core.results import Percentiles


def compute_percentile(cumul: np.ndarray, boundaries_um: np.ndarray, pct: float) -> float:
    """Interpolate one arbitrary percentile grain diameter (µm), e.g. P1 or P99.

    Interpolates in phi (log2) space, not linearly in µm — this is the
    standard sedimentological convention (equivalent to reading a percentile
    off a cumulative curve plotted on a logarithmic size axis, the universal
    way such curves are drawn) and keeps this function consistent with
    ``statistics.folk_ward_logarithmic``, which already interpolates in phi.
    Interpolating directly in µm instead gives a measurably different (and
    internally inconsistent, since it disagrees with the Folk & Ward phi16/
    50/84-derived percentiles for the same sample) result — confirmed both
    analytically and against a live GRADISTAT v9.1 run. ``compute_percentiles``
    below (the fixed P5-P95 set GRADISTAT Table II uses) calls this same
    function for each of its 9 values, so there's exactly one phi-space
    interpolation implementation for both the fixed set and an arbitrary
    one-off percentile to ever drift out of sync.

    Args:
        cumul: Cumulative percentage values (0–100), length N. Callers
            should include a leading ``cumul[0] == 0`` anchor point paired
            with the true lower bound of the first class in
            ``boundaries_um`` — without it, percentiles that fall inside
            the first class clamp to that class's own upper bound instead
            of interpolating within it (``np.interp`` clamps rather than
            extrapolates below ``cumul[0]``).
        boundaries_um: Upper class boundaries in µm, same length N (or, with
            the leading anchor point above, the first class's *lower* bound
            followed by every class's upper bound).
        pct: The target percentile, 0–100 (e.g. ``1.0`` for P1, ``99.0`` for
            P99). Not restricted to GRADISTAT's fixed P5-P95 set.

    Returns:
        The interpolated grain diameter in µm.

    Example:
        >>> import numpy as np
        >>> cumul = np.array([0.0, 20.0, 60.0, 100.0])
        >>> boundaries = np.array([1.0, 10.0, 100.0, 1000.0])
        >>> round(compute_percentile(cumul, boundaries, 50.0), 1)
        56.2
    """
    phi = -np.log2(np.asarray(boundaries_um, dtype=float) / 1000.0)
    phi_val = float(np.interp(pct, cumul, phi))
    return 1000.0 * 2.0**-phi_val


def compute_percentiles(cumul: np.ndarray, boundaries_um: np.ndarray) -> Percentiles:
    """Interpolate GRADISTAT Table II's fixed P5-P95 percentile set (µm).

    See :func:`compute_percentile` for the interpolation convention (phi
    space) and for computing a percentile outside this fixed set.

    Args:
        cumul: Cumulative percentage values (0–100) — see
            :func:`compute_percentile`'s ``cumul`` for the required leading
            anchor point.
        boundaries_um: Upper class boundaries in µm — see
            :func:`compute_percentile`'s ``boundaries_um``.

    Returns:
        Percentiles: Named percentile values plus derived ratios.
    """
    return Percentiles(
        p5=compute_percentile(cumul, boundaries_um, 5),
        p10=compute_percentile(cumul, boundaries_um, 10),
        p16=compute_percentile(cumul, boundaries_um, 16),
        p25=compute_percentile(cumul, boundaries_um, 25),
        p50=compute_percentile(cumul, boundaries_um, 50),
        p75=compute_percentile(cumul, boundaries_um, 75),
        p84=compute_percentile(cumul, boundaries_um, 84),
        p90=compute_percentile(cumul, boundaries_um, 90),
        p95=compute_percentile(cumul, boundaries_um, 95),
    )
