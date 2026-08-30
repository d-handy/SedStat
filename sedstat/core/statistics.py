"""Grain size statistics — all five methods from Blott & Pye (2001) Table II.

Reference: Blott, S. J. & Pye, K. (2001). GRADISTAT: a grain size
distribution and statistics package for the analysis of unconsolidated
sediments. Earth Surface Processes and Landforms, 26, 1237–1248.
DOI: 10.1002/esp.261
"""

from __future__ import annotations

import numpy as np

from sedstat.core.classification import ClassificationScheme, Classifier
from sedstat.core.fractions import compute_fractions
from sedstat.core.percentiles import compute_percentile as _interp_percentile
from sedstat.core.percentiles import compute_percentiles
from sedstat.core.results import (
    FolkWardStats,
    GrainSizeResult,
    MomentStats,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Real instrument exports rarely sum to exactly 100% (vendor-software
# rounding, a partially-obscured run, etc.). A total outside this band
# usually indicates a unit mismatch or a corrupted export rather than
# ordinary rounding noise, so compute_all() rejects it rather than silently
# rescaling something that isn't actually a percentage distribution.
_TOTAL_TOLERANCE_PCT = 10.0


def _validate(classes_um: np.ndarray, values: np.ndarray) -> None:
    if len(values) == 0:
        raise ValueError("values must not be empty.")
    if np.any(np.array(values) < 0):
        raise ValueError("values must not contain negative numbers.")
    total = float(np.sum(values))
    if total == 0:
        raise ValueError("values sum to zero.")
    lo, hi = 100.0 - _TOTAL_TOLERANCE_PCT, 100.0 + _TOTAL_TOLERANCE_PCT
    if not (lo <= total <= hi):
        raise ValueError(
            f"values sum to {total:.2f}; expected 100 ± {_TOTAL_TOLERANCE_PCT:.0f}. "
            "Normalize your data before passing it to compute_all()."
        )
    if len(classes_um) not in (len(values), len(values) + 1):
        raise ValueError(
            f"classes_um length ({len(classes_um)}) must equal len(values) "
            f"({len(values)}) or len(values)+1 ({len(values) + 1})."
        )
    if np.any(np.diff(np.asarray(classes_um, dtype=float)) <= 0):
        raise ValueError(
            "classes_um must be strictly increasing (ascending class boundaries); "
            "found a non-increasing step."
        )
    if np.any(np.asarray(classes_um, dtype=float) <= 0):
        # Geometric/logarithmic moments take log()/log2() of these boundaries
        # (directly, or via sqrt(lower*upper) midpoints). A zero or negative
        # boundary — e.g. a "0-2 µm" first class written with a literal 0 —
        # silently yields -inf/nan there instead of raising, so reject it here.
        raise ValueError(
            "classes_um must be strictly positive; grain size in µm cannot be "
            "zero or negative. Use a small positive lower bound (e.g. 0.01) "
            "instead of 0 for an open-ended finest class."
        )


def _class_bounds_um(classes_um: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) boundary arrays in µm for each class.

    If classes_um has N+1 entries (boundaries), lower/upper are read off
    directly. If classes_um has N entries (upper bounds only), the lower
    bound of the first class is estimated as classes_um[0]² / classes_um[1]
    — i.e. the first class is assumed to have the same width ratio as the
    second. With only a single class (N=1) there is no second class to
    borrow a ratio from, so the lower bound is instead estimated as half
    the upper bound (a one-octave/one-phi-unit-wide class, matching this
    package's power-of-two Wentworth convention) — a cruder approximation,
    but this single-class case only arises from deliberately coarse
    degenerate inputs (e.g. one sieve mesh plus a pan fraction).
    """
    n = len(values)
    if len(classes_um) == n + 1:
        lower = np.array(classes_um[:-1], dtype=float)
        upper = np.array(classes_um[1:], dtype=float)
    else:
        # Only upper bounds provided — estimate lower bound of first class
        upper = np.array(classes_um, dtype=float)
        first_lower = upper[0] / 2.0 if len(upper) < 2 else upper[0] ** 2 / upper[1]
        lower = np.concatenate([[first_lower], upper[:-1]])
    return lower, upper


def _midpoints_um(classes_um: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Geometric midpoints between consecutive class boundaries (µm).

    sqrt(lower * upper) for each class — used by moments_geometric() and
    (via _phi()) moments_logarithmic(). See _class_bounds_um() for how the
    lower/upper boundaries themselves are determined.
    """
    lower, upper = _class_bounds_um(classes_um, values)
    return np.sqrt(lower * upper)


def _arithmetic_midpoints_um(classes_um: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Arithmetic (linear) midpoints between consecutive class boundaries (µm).

    (lower + upper) / 2 — used by moments_arithmetic() (Table IIa) only.
    Not the same as _midpoints_um()'s geometric mean: GRADISTAT's
    reference implementation keeps these two conventions deliberately
    separate (classical Folk & Ward Table II distinction), and swapping
    the geometric mean in here is a confirmed ~1.5% numerical error
    against a GRADISTAT v9.1 cross-check.
    """
    lower, upper = _class_bounds_um(classes_um, values)
    return (lower + upper) / 2.0


def _phi(um: np.ndarray) -> np.ndarray:
    """Convert µm to phi: φ = -log2(d_mm) = -log2(d_µm / 1000)."""
    return -np.log2(np.asarray(um, dtype=float) / 1000.0)


def _cumulative(values: np.ndarray) -> np.ndarray:
    """Cumulative sum; values must already be normalised to sum ~100."""
    return np.cumsum(values)


def _interp_phi(percentile: float, cumul: np.ndarray, phi: np.ndarray) -> float:
    """Interpolate grain size in phi at a given cumulative percentile.

    cumul is monotonically increasing (0–100), so np.interp works directly.
    phi is decreasing (high phi = fine, low phi = coarse) — the returned value
    therefore decreases as percentile increases, which is the correct convention.
    """
    return float(np.interp(percentile, cumul, phi))


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, or NaN for a degenerate zero-width spread.

    Folk & Ward skewness/kurtosis divide by inter-percentile spreads (e.g.
    phi16 - phi84), and the moment methods divide by std**3/std**4 — both
    zero for a near-single-class distribution, a real case for e.g.
    calibration standards or heavily truncated exports. All operands here
    are plain Python floats (not numpy scalars), so an unguarded division
    by zero would raise ZeroDivisionError and abort the whole compute_all()
    call instead of yielding the mathematically undefined (NaN) result.
    """
    if denominator == 0.0:
        return float("nan")
    return numerator / denominator


# ---------------------------------------------------------------------------
# Method (a): Arithmetic method of moments  [Table IIa]
# ---------------------------------------------------------------------------


def moments_arithmetic(midpoints_um: np.ndarray, values: np.ndarray) -> MomentStats:
    """Arithmetic method of moments (Table IIa).

    All calculations in metric (µm) units assuming a normal distribution.
    Seldom used in sedimentology but provided for completeness.
    """
    f = np.asarray(values, dtype=float)
    m = np.asarray(midpoints_um, dtype=float)

    mean = float(np.sum(f * m) / 100.0)
    variance = float(np.sum(f * (m - mean) ** 2) / 100.0)
    std = float(np.sqrt(variance))
    skewness = _safe_ratio(float(np.sum(f * (m - mean) ** 3)), 100.0 * std**3)
    kurtosis = _safe_ratio(float(np.sum(f * (m - mean) ** 4)), 100.0 * std**4)

    return MomentStats(mean=mean, std=std, skewness=skewness, kurtosis=kurtosis)


# ---------------------------------------------------------------------------
# Method (b): Geometric method of moments  [Table IIb]
# ---------------------------------------------------------------------------


def moments_geometric(midpoints_um: np.ndarray, values: np.ndarray) -> MomentStats:
    """Geometric method of moments (Table IIb).

    Based on a log-normal distribution with metric (µm) size values.
    """
    f = np.asarray(values, dtype=float)
    m = np.asarray(midpoints_um, dtype=float)
    ln_m = np.log(m)

    mean = float(np.exp(np.sum(f * ln_m) / 100.0))
    ln_mean = np.log(mean)
    ln_std = float(np.sqrt(np.sum(f * (ln_m - ln_mean) ** 2) / 100.0))
    std = float(np.exp(ln_std))
    skewness = _safe_ratio(float(np.sum(f * (ln_m - ln_mean) ** 3)), 100.0 * ln_std**3)
    kurtosis = _safe_ratio(float(np.sum(f * (ln_m - ln_mean) ** 4)), 100.0 * ln_std**4)

    return MomentStats(mean=mean, std=std, skewness=skewness, kurtosis=kurtosis)


# ---------------------------------------------------------------------------
# Method (c): Logarithmic method of moments  [Table IIc]
# ---------------------------------------------------------------------------


def moments_logarithmic(midpoints_phi: np.ndarray, values: np.ndarray) -> MomentStats:
    """Logarithmic method of moments (Table IIc).

    Based on a log-normal distribution with phi size values.
    """
    f = np.asarray(values, dtype=float)
    m = np.asarray(midpoints_phi, dtype=float)

    mean = float(np.sum(f * m) / 100.0)
    variance = float(np.sum(f * (m - mean) ** 2) / 100.0)
    std = float(np.sqrt(variance))
    skewness = _safe_ratio(float(np.sum(f * (m - mean) ** 3)), 100.0 * std**3)
    kurtosis = _safe_ratio(float(np.sum(f * (m - mean) ** 4)), 100.0 * std**4)

    return MomentStats(mean=mean, std=std, skewness=skewness, kurtosis=kurtosis)


# ---------------------------------------------------------------------------
# Method (d): Folk & Ward logarithmic (phi)  [Table IId]
# ---------------------------------------------------------------------------


def folk_ward_logarithmic(cumul: np.ndarray, phi: np.ndarray) -> FolkWardStats:
    """Folk & Ward (1957) graphical measures, logarithmic/phi (Table IId).

    phi must be the upper-class-boundary phi values sorted fine-to-coarse
    (i.e. phi decreasing as index increases).  cumul is % finer from the
    fine end, so phi5 > phi95 numerically.
    """

    def p(pct):
        return _interp_phi(pct, cumul, phi)

    phi5 = p(5)
    phi16 = p(16)
    phi25 = p(25)
    phi50 = p(50)
    phi75 = p(75)
    phi84 = p(84)
    phi95 = p(95)

    # phi16 > phi84 (fine end has higher phi than coarse end)
    mean = (phi16 + phi50 + phi84) / 3.0
    std = (phi16 - phi84) / 4.0 + (phi5 - phi95) / 6.6
    ski = _safe_ratio(phi16 + phi84 - 2.0 * phi50, 2.0 * (phi16 - phi84)) + _safe_ratio(
        phi5 + phi95 - 2.0 * phi50, 2.0 * (phi5 - phi95)
    )
    kg = _safe_ratio(phi5 - phi95, 2.44 * (phi25 - phi75))

    return FolkWardStats(mean=mean, std=std, skewness=ski, kurtosis=kg)


# ---------------------------------------------------------------------------
# Method (e): Folk & Ward geometric (metric)  [Table IIe]
# ---------------------------------------------------------------------------


def folk_ward_geometric(cumul: np.ndarray, um: np.ndarray) -> FolkWardStats:
    """Folk & Ward graphical measures, geometric/metric (Table IIe).

    Modified formulation using natural logarithms of metric (µm) percentiles.
    Note: skewness sign convention is reversed relative to the phi method —
    positive skewness in µm corresponds to a coarse tail (Folk & Ward Table IIe).

    Percentiles are obtained via phi-space interpolation (matching
    folk_ward_logarithmic and the standard convention of reading a
    percentile off a cumulative curve plotted on a log size axis), then
    converted back to µm before taking natural logs — not interpolated
    directly in µm space, which gives a measurably different and internally
    inconsistent result (would disagree with folk_ward_logarithmic's own
    phi-derived percentiles for the same sample). Confirmed both
    analytically and against a live GRADISTAT v9.1 run, which also
    interpolates in phi/log space throughout and converts to a linear size
    unit only at the very end.
    """
    phi = -np.log2(np.asarray(um, dtype=float) / 1000.0)

    def p(pct):
        phi_val = _interp_phi(pct, cumul, phi)
        return 1000.0 * 2.0**-phi_val

    p5 = p(5)
    p16 = p(16)
    p25 = p(25)
    p50 = p(50)
    p75 = p(75)
    p84 = p(84)
    p95 = p(95)

    ln5, ln16, ln25, ln50, ln75, ln84, ln95 = (
        np.log(p5),
        np.log(p16),
        np.log(p25),
        np.log(p50),
        np.log(p75),
        np.log(p84),
        np.log(p95),
    )

    # ln84 > ln16 and ln95 > ln5 (D84 > D16 in µm, D95 > D5)
    mean = float(np.exp((ln16 + ln50 + ln84) / 3.0))
    std = float(np.exp((ln84 - ln16) / 4.0 + (ln95 - ln5) / 6.6))
    ski = float(
        _safe_ratio(ln16 + ln84 - 2.0 * ln50, 2.0 * (ln84 - ln16))
        + _safe_ratio(ln5 + ln95 - 2.0 * ln50, 2.0 * (ln95 - ln5))
    )
    kg = float(_safe_ratio(ln95 - ln5, 2.44 * (ln75 - ln25)))

    return FolkWardStats(mean=mean, std=std, skewness=ski, kurtosis=kg)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _anchored_cumul_um(
    classes_arr: np.ndarray, values_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cumul_anchored, um_anchored), the (0%, first-class-lower-bound) anchored arrays.

    folk_ward_geometric/compute_percentiles/compute_percentile interpolate
    against these. Shared by compute_all and compute_percentile so this
    anchoring convention can't drift between them.

    Anchoring with an explicit (0%, first class's own lower bound) leading
    point matters because without it, np.interp *clamps* rather than
    interpolates for any percentile at or below the first class's own
    cumulative weight (e.g. a coarse first class holding 15% of the sample
    makes every P-value ≤15 silently equal that class's upper bound instead
    of the correct interior value) — a real, independently verifiable bug
    (confirmed analytically and cross-checked against a live GRADISTAT v9.1
    run), not merely a convention choice.

    Args:
        classes_arr: Already-validated class boundaries/upper bounds in µm.
        values_arr: Already-validated, already-normalized-to-100 values.
    """
    cumul = _cumulative(values_arr)
    upper_um = classes_arr[1:] if len(classes_arr) == len(values_arr) + 1 else classes_arr
    lower_bounds, _ = _class_bounds_um(classes_arr, values_arr)
    cumul_anchored = np.concatenate([[0.0], cumul])
    um_anchored = np.concatenate([[lower_bounds[0]], upper_um])
    return cumul_anchored, um_anchored


def compute_percentile(
    classes_um: list[float] | np.ndarray, values: list[float] | np.ndarray, pct: float
) -> float:
    """Interpolate one arbitrary percentile grain diameter (µm), e.g. P1 or P99.

    ``compute_all()``'s own ``result.percentiles`` only ever has the fixed
    GRADISTAT Table II set (P5-P95); this is the entry point for anything
    outside it. Does its own validation/normalization/anchoring — safe to
    call standalone, not just from inside ``compute_all``.

    Args:
        classes_um: Class boundaries or upper bounds in µm — same shape
            ``compute_all`` takes.
        values: Weight or volume percentage per class — same shape
            ``compute_all`` takes.
        pct: The target percentile, 0-100 (e.g. ``1.0`` for P1, ``99.0``
            for P99). Not restricted to GRADISTAT's fixed P5-P95 set.

    Returns:
        The interpolated grain diameter in µm.

    Example:
        >>> classes_um = [63.0, 125.0, 250.0, 500.0, 1000.0]
        >>> values = [10.0, 25.0, 35.0, 20.0, 10.0]
        >>> round(compute_percentile(classes_um, values, 99.0), 1)
        933.0
    """
    classes_arr: np.ndarray = np.asarray(classes_um, dtype=float)
    values_arr: np.ndarray = np.asarray(values, dtype=float)

    _validate(classes_arr, values_arr)
    values_arr = values_arr / values_arr.sum() * 100.0

    cumul_anchored, um_anchored = _anchored_cumul_um(classes_arr, values_arr)
    return _interp_percentile(cumul_anchored, um_anchored, pct)


def compute_all(
    classes_um: list[float] | np.ndarray,
    values: list[float] | np.ndarray,
    scheme: ClassificationScheme = ClassificationScheme.GRADISTAT,
) -> GrainSizeResult:
    """Compute grain size statistics using all five methods.

    Args:
        classes_um: Class boundaries or upper bounds in µm. Length N+1
            (boundaries) or N (upper bounds). Must be monotonically
            increasing.
        values: Weight or volume percentage per class. Length N. Must sum
            to ~100.
        scheme: Classification scheme used for the grain-size fraction
            boundaries and the textural descriptions (sorting, skewness,
            kurtosis, grade class).

    Returns:
        GrainSizeResult: Dataclass with results for all five methods,
        percentiles, fractions, and textural descriptions.

    Example:
        >>> classes_um = [63.0, 125.0, 250.0, 500.0, 1000.0]
        >>> values = [10.0, 25.0, 35.0, 20.0, 10.0]
        >>> result = compute_all(classes_um, values)
        >>> round(result.percentiles.d50, 1)
        168.2
    """
    classes_arr: np.ndarray = np.asarray(classes_um, dtype=float)
    values_arr: np.ndarray = np.asarray(values, dtype=float)

    _validate(classes_arr, values_arr)

    # Normalize to exactly 100 so all interpolations are well-defined
    values_arr = values_arr / values_arr.sum() * 100.0

    mid_um = _midpoints_um(classes_arr, values_arr)
    mid_um_arith = _arithmetic_midpoints_um(classes_arr, values_arr)
    mid_phi = _phi(mid_um)

    # See _anchored_cumul_um's own docstring for why an explicit (0%,
    # first-class-lower-bound) anchor point is needed.
    cumul_anchored, um_anchored = _anchored_cumul_um(classes_arr, values_arr)
    phi_anchored = _phi(um_anchored)

    # --- Five methods ---
    # NOTE: moments_arithmetic() takes the *arithmetic* midpoint (mid_um_arith),
    # not the geometric one (mid_um) that moments_geometric()/moments_logarithmic()
    # use — see _arithmetic_midpoints_um()'s docstring for why these must differ.
    arith = moments_arithmetic(mid_um_arith, values_arr)
    geo = moments_geometric(mid_um, values_arr)
    log_ = moments_logarithmic(mid_phi, values_arr)
    fw_log = folk_ward_logarithmic(cumul_anchored, phi_anchored)
    fw_geo = folk_ward_geometric(cumul_anchored, um_anchored)

    # --- Percentiles (µm, from cumulative curve on upper class boundaries) ---
    pct = compute_percentiles(cumul_anchored, um_anchored)

    # --- Fractions ---
    # Use class upper bounds for fraction assignment (matches GRADISTAT)
    upper_bounds = classes_arr if len(classes_arr) == len(values_arr) else classes_arr[1:]
    frac = compute_fractions(upper_bounds, values_arr, scheme)

    # --- Textural descriptions ---
    clf = Classifier(scheme)
    desc = {
        "mean_wentworth": clf.mean_wentworth_phi(fw_log.mean),
        "fw_log_sorting": clf.sorting_fw_log(fw_log.std),
        "fw_log_skewness": clf.skewness_fw_log(fw_log.skewness),
        "fw_log_kurtosis": clf.kurtosis_fw_log(fw_log.kurtosis),
        "fw_geo_mean_wentworth": clf.mean_wentworth_um(fw_geo.mean),
        "fw_geo_sorting": clf.sorting_fw_geo(fw_geo.std),
        "fw_geo_skewness": clf.skewness_fw_geo(fw_geo.skewness),
        "fw_geo_kurtosis": clf.kurtosis_fw_geo(fw_geo.kurtosis),
        "geo_sorting": clf.sorting_moments_geo(geo.std),
        "geo_skewness": clf.skewness_moments_geo(geo.skewness),
        "geo_kurtosis": clf.kurtosis_moments_geo(geo.kurtosis),
        "log_sorting": clf.sorting_moments_log(log_.std),
        "log_skewness": clf.skewness_moments_log(log_.skewness),
        "log_kurtosis": clf.kurtosis_moments_log(log_.kurtosis),
    }

    return GrainSizeResult(
        classes_um=classes_arr.tolist(),
        values=values_arr.tolist(),
        arithmetic=arith,
        geometric=geo,
        logarithmic=log_,
        folk_ward_log=fw_log,
        folk_ward_geo=fw_geo,
        percentiles=pct,
        fractions=frac,
        descriptions=desc,
        scheme=scheme,
    )
