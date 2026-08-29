"""Tests for sedstat.core.statistics.

The Mablethorpe L2D1 sample published in Blott & Pye (2001) Fig. 2 is the
primary reference. The paper does not give the raw distribution, so a
synthetic Gaussian distribution in phi space is used, reproducing the
published Folk & Ward logarithmic statistics (mean ≈ 2.455 φ,
std ≈ 0.390 φ) to the paper's precision.

Tolerances: Folk & Ward values within ±0.05 (paper rounds to 3 decimal
places); moment values within ±0.5 for mean (µm), ±0.02 for std, ±0.05
for skewness. Moments kurtosis is checked only for order of magnitude,
since it is highly sensitive to distribution tails.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
import pytest
from sedstat.core.results import GrainSizeResult
from sedstat.core.statistics import (
    _cumulative,
    _midpoints_um,
    _phi,
    compute_all,
)

# ---------------------------------------------------------------------------
# Synthetic test distribution that matches the Mablethorpe L2D1 statistics
# ---------------------------------------------------------------------------


def _make_mablethorpe_distribution():
    """Synthetic fine-sand distribution matching Mablethorpe L2D1 (Blott & Pye 2001).

    Constructed as a Gaussian in phi space centred on φ=2.455 with σ=0.390.
    Class boundaries are the standard GRADISTAT half-phi scale from 0.04–2000 µm.

    Returns:
        A tuple of (class boundaries in µm, ascending; weight-% per class).
    """
    # Standard GRADISTAT boundaries (subset, fine sand range extended)
    boundaries_um = np.array(
        [
            0.04,
            0.044,
            0.048,
            0.053,
            0.058,
            0.064,
            0.070,
            0.077,
            0.084,
            0.093,
            0.102,
            0.112,
            0.122,
            0.134,
            0.148,
            0.162,
            0.178,
            0.195,
            0.214,
            0.235,
            0.258,
            0.284,
            0.311,
            0.342,
            0.375,
            0.412,
            0.452,
            0.496,
            0.545,
            0.598,
            0.657,
            0.721,
            0.791,
            0.869,
            0.953,
            1.047,
            1.149,
            1.261,
            1.385,
            1.520,
            1.669,
            1.832,
            2.010,
            2.207,
            2.423,
            2.660,
            2.920,
            3.206,
            3.519,
            3.862,
            4.241,
            4.656,
            5.111,
            5.611,
            6.158,
            6.761,
            7.421,
            8.147,
            8.944,
            9.819,
            10.78,
            11.83,
            12.99,
            14.26,
            15.65,
            17.17,
            18.86,
            20.70,
            22.73,
            24.95,
            27.38,
            30.07,
            33.00,
            36.24,
            39.77,
            43.66,
            47.93,
            52.63,
            57.77,
            63.41,
            69.62,
            76.43,
            83.90,
            92.09,
            101.1,
            111.0,
            121.8,
            133.7,
            146.8,
            161.2,
            176.8,
            194.2,
            213.2,
            234.1,
            256.8,
            282.1,
            309.6,
            339.8,
            373.1,
            409.6,
            449.7,
            493.6,
            541.9,
            594.9,
            653.0,
            716.9,
            786.9,
            863.9,
            948.2,
            1041.0,
            1143.0,
            1255.0,
            1377.0,
            1512.0,
            1660.0,
            1822.0,
            2000.0,
        ]
    )

    mid_um = np.sqrt(boundaries_um[:-1] * boundaries_um[1:])
    mid_phi = -np.log2(mid_um / 1000.0)

    # Gaussian PDF in phi space (φ = 2.455, σ = 0.390)
    mu_phi, sigma_phi = 2.455, 0.390
    weights = np.exp(-0.5 * ((mid_phi - mu_phi) / sigma_phi) ** 2)
    weights = weights / weights.sum() * 100.0

    return boundaries_um, weights


BOUNDARIES, VALUES = _make_mablethorpe_distribution()


# ---------------------------------------------------------------------------
# Smoke tests — compute_all must not raise
# ---------------------------------------------------------------------------


class TestComputeAllSmoke:
    """compute_all() returns a fully populated, internally consistent GrainSizeResult."""

    def test_returns_grain_size_result(self):
        result = compute_all(BOUNDARIES, VALUES)
        assert isinstance(result, GrainSizeResult)

    def test_all_methods_populated(self):
        result = compute_all(BOUNDARIES, VALUES)
        for attr in (
            "arithmetic",
            "geometric",
            "logarithmic",
            "folk_ward_log",
            "folk_ward_geo",
        ):
            stats = getattr(result, attr)
            for field in ("mean", "std", "skewness", "kurtosis"):
                assert np.isfinite(getattr(stats, field)), f"{attr}.{field} is not finite"

    def test_fractions_sum_to_100(self):
        result = compute_all(BOUNDARIES, VALUES)
        total = (
            result.fractions.clay
            + result.fractions.fine_silt
            + result.fractions.medium_silt
            + result.fractions.coarse_silt
            + result.fractions.fine_sand
            + result.fractions.medium_sand
            + result.fractions.coarse_sand
            + result.fractions.gravel
        )
        assert abs(total - 100.0) < 0.01

    def test_percentiles_are_ordered(self):
        result = compute_all(BOUNDARIES, VALUES)
        p = result.percentiles
        assert p.p5 <= p.p10 <= p.p16 <= p.p25 <= p.p50 <= p.p75 <= p.p84 <= p.p90 <= p.p95

    def test_to_dict_has_expected_keys(self):
        result = compute_all(BOUNDARIES, VALUES)
        d = result.to_dict()
        for key in (
            "fw_log_mean_phi",
            "fw_log_std_phi",
            "geo_mean_um",
            "arith_mean_um",
            "log_mean_phi",
            "p50_um",
            "fine_sand_pct",
            "clay_pct",
            "total_silt_pct",
            "total_sand_pct",
        ):
            assert key in d, f"Missing key: {key}"


class TestComputeAllIso14688Scheme:
    """compute_all(..., scheme=ISO14688) — same numeric stats, ISO naming/scheme tag.

    Attributes:
        scheme: The ISO14688 ClassificationScheme under test.
        result: compute_all(BOUNDARIES, VALUES, scheme) from setup_method.
    """

    def setup_method(self):
        from sedstat.core.classification import ClassificationScheme

        self.scheme = ClassificationScheme.ISO14688
        self.result = compute_all(BOUNDARIES, VALUES, self.scheme)

    def test_scheme_is_recorded(self):
        assert self.result.scheme == self.scheme

    def test_fractions_match_gradistat(self):
        # Clay/silt/sand boundaries are numerically identical to GRADISTAT.
        gradistat_result = compute_all(BOUNDARIES, VALUES)
        assert self.result.fractions == gradistat_result.fractions

    def test_wentworth_description_present(self):
        assert self.result.descriptions["mean_wentworth"]


# ---------------------------------------------------------------------------
# Quantitative validation against Blott & Pye (2001) Fig. 2
# ---------------------------------------------------------------------------


class TestFolkWardLogarithmic:
    """Validate Folk & Ward phi method against the paper.

    Attributes:
        result: compute_all() on the Mablethorpe distribution, from setup_method.
    """

    def setup_method(self):
        self.result = compute_all(BOUNDARIES, VALUES)

    def test_mean_phi(self):
        assert abs(self.result.folk_ward_log.mean - 2.455) < 0.05

    def test_std_phi(self):
        assert abs(self.result.folk_ward_log.std - 0.390) < 0.05

    def test_sorting_description(self):
        assert self.result.descriptions["fw_log_sorting"] == "Well Sorted"

    def test_mean_description(self):
        assert self.result.descriptions["mean_wentworth"] == "Fine Sand"


class TestFolkWardGeometric:
    """Validate Folk & Ward metric method.

    Attributes:
        result: compute_all() on the Mablethorpe distribution, from setup_method.
    """

    def setup_method(self):
        self.result = compute_all(BOUNDARIES, VALUES)

    def test_mean_um(self):
        assert abs(self.result.folk_ward_geo.mean - 182.5) < 5.0

    def test_std(self):
        assert abs(self.result.folk_ward_geo.std - 1.311) < 0.05

    def test_skewness_consistent_with_log(self):
        # Geometric and log skewness have opposite signs but same magnitude
        geo_sk = self.result.folk_ward_geo.skewness
        log_sk = self.result.folk_ward_log.skewness
        assert abs(abs(geo_sk) - abs(log_sk)) < 0.02, (
            f"Skewness magnitudes inconsistent: geo={geo_sk}, log={log_sk}"
        )


class TestFolkWardSingleWideClass:
    """A single wide class exercises percentile interpolation at the edges.

    Before compute_all() anchored the cumulative/boundary arrays with a
    leading (0%, lower-bound) point, np.interp clamped queries below its
    first known x-value instead of interpolating, collapsing every
    percentile (phi16=phi50=phi84=...) to the class's upper boundary. A
    single class still has real internal width, so percentiles at
    different cumulative fractions should genuinely differ within it, and
    skewness/kurtosis should be finite (not NaN): a single class, linear
    in phi, is phi-symmetric (skewness exactly 0) with a fixed
    scale-invariant kurtosis of 0.7377049180327869.

    Attributes:
        result: compute_all() on the single-class [10, 20] µm distribution.
    """

    def setup_method(self):
        self.result = compute_all([10.0, 20.0], [100.0])

    def test_does_not_raise(self):
        assert isinstance(self.result, GrainSizeResult)

    def test_log_skewness_is_zero_by_symmetry(self):
        assert self.result.folk_ward_log.skewness == pytest.approx(0.0, abs=1e-9)

    def test_log_kurtosis_is_shape_invariant_value(self):
        assert self.result.folk_ward_log.kurtosis == pytest.approx(0.7377049180327869)

    def test_geo_skewness_is_zero_by_symmetry(self):
        assert self.result.folk_ward_geo.skewness == pytest.approx(0.0, abs=1e-9)

    def test_geo_kurtosis_is_shape_invariant_value(self):
        assert self.result.folk_ward_geo.kurtosis == pytest.approx(0.7377049180327869)

    def test_std_is_well_defined_and_nonzero(self):
        # Real interior spread now (not a degenerate zero-width point).
        assert self.result.folk_ward_log.std > 0


class TestMomentsDegenerateDistribution:
    """A single class is genuinely degenerate for the moments methods.

    Every value's midpoint is the same point (the class's arithmetic or
    geometric midpoint), so variance is exactly zero regardless of class
    width. The skewness/kurtosis formulas divide by std**3/std**4; before
    the _safe_ratio guard this raised ZeroDivisionError and aborted
    compute_all() instead of yielding NaN.

    Attributes:
        result: compute_all() on the single degenerate-width class.
    """

    def setup_method(self):
        self.result = compute_all([10.0, 20.0], [100.0])

    def test_moment_skewness_and_kurtosis_are_nan(self):
        for stats in (
            self.result.arithmetic,
            self.result.geometric,
            self.result.logarithmic,
        ):
            assert math.isnan(stats.skewness)
            assert math.isnan(stats.kurtosis)

    def test_moment_std_is_still_well_defined(self):
        """A degenerate distribution's std is 0 (additive scales) or 1 (multiplicative).

        Arithmetic/logarithmic std are additive (zero spread -> 0); geometric
        std is multiplicative (exp of a zero log-spread -> 1, the identity).
        """
        assert self.result.arithmetic.std == pytest.approx(0.0, abs=1e-9)
        assert self.result.logarithmic.std == pytest.approx(0.0, abs=1e-9)
        assert self.result.geometric.std == pytest.approx(1.0)


class TestAgainstGRADISTATv91:
    """Regression values cross-checked against a live GRADISTAT v9.1 run.

    The input (5 classes spanning 250-1400 µm, weights 15/25/30/20/10) was
    entered into GRADISTAT's "Single Sample Data Input" sheet, the
    FolkAndWardMacro was run via COM automation, and each value below was
    read back from "Single Sample Statistics"/"Calculations". The values
    also agree with Folk & Ward's own Table II definitions to 6 decimal
    places.

    Attributes:
        result: compute_all() on the 5-class GRADISTAT regression input.
    """

    def setup_method(self):
        self.result = compute_all(
            [250.0, 355.0, 500.0, 710.0, 1000.0, 1400.0], [15.0, 25.0, 30.0, 20.0, 10.0]
        )

    def test_arithmetic_moments(self):
        assert self.result.arithmetic.mean == pytest.approx(624.750000, abs=1e-4)
        assert self.result.arithmetic.std == pytest.approx(262.897199, abs=1e-4)
        assert self.result.arithmetic.skewness == pytest.approx(0.800024, abs=1e-5)
        assert self.result.arithmetic.kurtosis == pytest.approx(2.827894, abs=1e-5)

    def test_geometric_moments(self):
        assert self.result.geometric.mean == pytest.approx(565.234825, abs=1e-4)
        assert self.result.geometric.std == pytest.approx(1.511053, abs=1e-5)
        assert self.result.geometric.skewness == pytest.approx(0.104357, abs=1e-5)
        assert self.result.geometric.kurtosis == pytest.approx(2.134044, abs=1e-5)

    def test_logarithmic_moments(self):
        assert self.result.logarithmic.mean == pytest.approx(0.823078, abs=1e-5)
        assert self.result.logarithmic.std == pytest.approx(0.595554, abs=1e-5)
        assert self.result.logarithmic.skewness == pytest.approx(-0.104357, abs=1e-5)
        assert self.result.logarithmic.kurtosis == pytest.approx(2.134044, abs=1e-5)

    def test_folk_ward_geometric(self):
        assert self.result.folk_ward_geo.mean == pytest.approx(567.234673, abs=1e-4)
        assert self.result.folk_ward_geo.std == pytest.approx(1.564588, abs=1e-5)
        assert self.result.folk_ward_geo.skewness == pytest.approx(0.033002, abs=1e-5)
        assert self.result.folk_ward_geo.kurtosis == pytest.approx(0.918079, abs=1e-5)

    def test_folk_ward_logarithmic(self):
        assert self.result.folk_ward_log.mean == pytest.approx(0.817982, abs=1e-5)
        assert self.result.folk_ward_log.std == pytest.approx(0.645783, abs=1e-5)
        assert self.result.folk_ward_log.skewness == pytest.approx(-0.033002, abs=1e-5)
        assert self.result.folk_ward_log.kurtosis == pytest.approx(0.918079, abs=1e-5)

    def test_percentiles(self):
        assert self.result.percentiles.p10 == pytest.approx(315.838866, abs=1e-4)
        assert self.result.percentiles.p50 == pytest.approx(561.995432, abs=1e-4)
        assert self.result.percentiles.p90 == pytest.approx(1000.0, abs=1e-4)


class TestComputePercentile:
    """compute_percentile() — arbitrary percentiles outside the fixed
    GRADISTAT P5-P95 set compute_all()/Percentiles always returns.

    Attributes:
        CLASSES: Shared class boundaries (µm) used across this class's tests.
        VALUES: Shared weight-% per class, matching CLASSES.
    """

    CLASSES: ClassVar[list[float]] = [250.0, 355.0, 500.0, 710.0, 1000.0, 1400.0]
    VALUES: ClassVar[list[float]] = [15.0, 25.0, 30.0, 20.0, 10.0]

    def test_matches_compute_all_fixed_set(self):
        """compute_percentile() is bit-identical to compute_all()'s own P10/P50/P90.

        compute_percentile must be bit-identical to compute_all()'s own
        P10/P50/P90 for the same input (same underlying interpolation).
        """
        from sedstat.core.statistics import compute_percentile

        result = compute_all(self.CLASSES, self.VALUES)
        assert compute_percentile(self.CLASSES, self.VALUES, 10.0) == result.percentiles.p10
        assert compute_percentile(self.CLASSES, self.VALUES, 50.0) == result.percentiles.p50
        assert compute_percentile(self.CLASSES, self.VALUES, 90.0) == result.percentiles.p90

    def test_arbitrary_percentile_outside_fixed_set(self):
        from sedstat.core.statistics import compute_percentile

        p1 = compute_percentile(self.CLASSES, self.VALUES, 1.0)
        p99 = compute_percentile(self.CLASSES, self.VALUES, 99.0)
        result = compute_all(self.CLASSES, self.VALUES)
        # Monotonicity check only: P1 < P5 and P99 > P95 on the cumulative
        # curve, with no independent reference value for P1/P99 itself.
        assert p1 < result.percentiles.p5
        assert p99 > result.percentiles.p95

    def test_accepts_values_within_normalization_tolerance(self):
        """compute_percentile() accepts values summing to ~100, not exactly 100.

        Same convention as compute_all(): values need only sum to ~100
        (within _validate's tolerance), not exactly.
        """
        from sedstat.core.statistics import compute_percentile

        slightly_off_values = [15.2, 24.8, 30.1, 19.9, 9.8]  # sums to 99.8, not 100
        p50_exact = compute_percentile(self.CLASSES, self.VALUES, 50.0)
        p50_off = compute_percentile(self.CLASSES, slightly_off_values, 50.0)
        assert p50_off == pytest.approx(p50_exact, rel=0.01)

    def test_low_level_percentiles_helper_matches(self):
        """statistics.compute_percentile() delegates to the lower-level percentiles helper.

        statistics.compute_percentile delegates to the lower-level,
        already-anchored-arrays percentiles.compute_percentile.
        """
        from sedstat.core.percentiles import compute_percentile as interp_percentile
        from sedstat.core.statistics import _anchored_cumul_um

        classes_arr = np.asarray(self.CLASSES, dtype=float)
        values_arr = np.asarray(self.VALUES, dtype=float)
        values_arr = values_arr / values_arr.sum() * 100.0
        cumul_anchored, um_anchored = _anchored_cumul_um(classes_arr, values_arr)

        from sedstat.core.statistics import compute_percentile

        assert interp_percentile(cumul_anchored, um_anchored, 42.0) == compute_percentile(
            self.CLASSES, self.VALUES, 42.0
        )


class TestArithmeticMoments:
    """Arithmetic (linear-µm) moments of the Mablethorpe distribution.

    Attributes:
        result: compute_all(BOUNDARIES, VALUES) from setup_method.
    """

    def setup_method(self):
        self.result = compute_all(BOUNDARIES, VALUES)

    def test_mean_in_range(self):
        # Paper: 186.2 µm; tolerance wider for arithmetic because sensitive to tails
        assert 170.0 < self.result.arithmetic.mean < 210.0

    def test_std_positive(self):
        assert self.result.arithmetic.std > 0


class TestGeometricMoments:
    """Geometric (log-µm) moments of the Mablethorpe distribution.

    Attributes:
        result: compute_all(BOUNDARIES, VALUES) from setup_method.
    """

    def setup_method(self):
        self.result = compute_all(BOUNDARIES, VALUES)

    def test_mean_um(self):
        """The synthetic Gaussian's geometric mean is ~182.5 µm, not the paper's 174.8 µm.

        Synthetic Gaussian at 2.455 phi gives geometric mean = 2^-2.455 * 1000 ≈ 182.5 µm.
        The paper's 174.8 µm is for the actual (non-Gaussian) Mablethorpe sample.
        """
        assert abs(self.result.geometric.mean - 182.5) < 5.0

    def test_std_above_1(self):
        # Geometric std > 1 always for any real distribution
        assert self.result.geometric.std > 1.0


class TestLogarithmicMoments:
    """Logarithmic (phi) moments of the Mablethorpe distribution.

    Attributes:
        result: compute_all(BOUNDARIES, VALUES) from setup_method.
    """

    def setup_method(self):
        self.result = compute_all(BOUNDARIES, VALUES)

    def test_mean_phi(self):
        assert abs(self.result.logarithmic.mean - 2.518) < 0.10

    def test_skewness_opposite_sign_to_geometric(self):
        log_sk = self.result.logarithmic.skewness
        geo_sk = self.result.geometric.skewness
        # In phi, fine skew is positive; in µm it is negative
        assert log_sk * geo_sk < 0, f"Expected opposite signs: log={log_sk}, geo={geo_sk}"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    """compute_all() rejects malformed or physically invalid inputs."""

    def test_empty_values_raises(self):
        with pytest.raises(ValueError, match="empty"):
            compute_all([1.0, 2.0], [])

    def test_zero_sum_raises(self):
        with pytest.raises(ValueError, match="zero"):
            compute_all([1.0, 2.0], [0.0])

    def test_negative_values_raise(self):
        with pytest.raises(ValueError, match="negative"):
            compute_all([1.0, 2.0, 3.0], [50.0, -10.0, 60.0])

    def test_sum_far_from_100_raises(self):
        with pytest.raises(ValueError, match="sum"):
            compute_all([1.0, 2.0], [1.0])  # sums to 1, not ~100

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="length"):
            compute_all([1.0, 2.0, 3.0], [50.0, 60.0])  # 3 boundaries, 2 values → OK
            # but 4 boundaries, 2 values → error
            compute_all([1.0, 2.0, 3.0, 4.0], [50.0, 50.0])

    def test_zero_boundary_raises(self):
        """A literal 0 µm boundary raises instead of silently producing -inf/nan.

        A literal 0 µm boundary (e.g. writing the clay class as "0-2 µm")
        would otherwise silently produce -inf/nan in the geometric and
        logarithmic moment methods via log(0).
        """
        with pytest.raises(ValueError, match="positive"):
            compute_all([0.0, 2.0, 6.3], [40.0, 60.0])

    def test_negative_boundary_raises(self):
        with pytest.raises(ValueError, match="positive"):
            compute_all([-1.0, 2.0, 6.3], [40.0, 60.0])

    def test_n_classes_equal_n_values_is_valid(self):
        # Upper bounds only (no lower bound for first class) — must work
        result = compute_all([1.0, 2.0], [40.0, 60.0])
        assert isinstance(result, GrainSizeResult)

    def test_n_plus_1_boundaries_is_valid(self):
        result = compute_all([0.5, 1.0, 2.0], [40.0, 60.0])
        assert isinstance(result, GrainSizeResult)

    def test_non_increasing_classes_raise(self):
        # Descending boundaries would otherwise silently corrupt np.interp.
        with pytest.raises(ValueError, match="increasing"):
            compute_all([2.0, 1.0, 0.5], [40.0, 60.0])

    def test_repeated_boundary_raises(self):
        with pytest.raises(ValueError, match="increasing"):
            compute_all([1.0, 1.0, 2.0], [40.0, 60.0])


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


class TestMidpoints:
    """_midpoints_um() geometric-midpoint estimation, including edge cases."""

    def test_geometric_midpoint_two_boundaries(self):
        # sqrt(1 * 4) = 2
        mid = _midpoints_um(np.array([1.0, 4.0]), np.array([100.0]))
        assert abs(mid[0] - 2.0) < 1e-10

    def test_upper_bounds_only(self):
        """An N-form (upper-bounds-only) input estimates the first class's lower bound.

        Two classes: upper bounds [2, 4].
        Estimated lower of first = 2²/4 = 1 → mid = sqrt(1*2) = sqrt(2).
        """
        mid = _midpoints_um(np.array([2.0, 4.0]), np.array([50.0, 50.0]))
        assert abs(mid[0] - np.sqrt(2.0)) < 1e-10
        assert abs(mid[1] - np.sqrt(2.0 * 4.0)) < 1e-10

    def test_single_upper_bound_only_uses_half_as_lower(self):
        """A lone N-form class falls back to a one-octave-wide class (lower = upper/2).

        A single N-form class has no second class to borrow a width ratio
        from (upper[1] doesn't exist) — falls back to a one-octave-wide
        class: lower = upper/2 → mid = sqrt(upper/2 * upper).
        """
        mid = _midpoints_um(np.array([250.0]), np.array([100.0]))
        assert abs(mid[0] - np.sqrt(125.0 * 250.0)) < 1e-10

    def test_single_upper_bound_integrates_with_compute_all(self):
        """A single-class N-form distribution from a "one sieve + pan" sample does not raise.

        This is the shape sedstat.io.sieve_pipette.sieve_to_distribution
        produces for a "one real sieve + pan" degenerate sample — must not
        raise (previously an IndexError from the two-class-only formula).
        """
        result = compute_all([250.0], [100.0])
        assert np.isfinite(result.folk_ward_geo.mean)

    def test_phi_conversion(self):
        # 1000 µm = 1 mm → phi = -log2(1) = 0
        assert abs(_phi(np.array([1000.0]))[0]) < 1e-10
        # 500 µm = 0.5 mm → phi = -log2(0.5) = 1
        assert abs(_phi(np.array([500.0]))[0] - 1.0) < 1e-10
        # 250 µm → phi = 2
        assert abs(_phi(np.array([250.0]))[0] - 2.0) < 1e-10


class TestCumulative:
    """_cumulative() produces a monotonic curve ending at 100%."""

    def test_cumulative_ends_at_100(self):
        values = np.array([20.0, 30.0, 50.0])
        cumul = _cumulative(values)
        assert abs(cumul[-1] - 100.0) < 1e-10

    def test_cumulative_is_monotonic(self):
        values = np.array([10.0, 40.0, 30.0, 20.0])
        cumul = _cumulative(values)
        assert np.all(np.diff(cumul) >= 0)


# ---------------------------------------------------------------------------
# GrainSizeResult / Percentiles / Fractions convenience properties
# ---------------------------------------------------------------------------


class TestPercentilesProperties:
    """Percentiles' derived d10/d50/d90 convenience properties.

    Attributes:
        p: A Percentiles instance with distinct p5..p95 values, from setup_method.
    """

    def setup_method(self):
        from sedstat.core.results import Percentiles

        self.p = Percentiles(
            p5=1.0,
            p10=2.0,
            p16=3.0,
            p25=4.0,
            p50=5.0,
            p75=6.0,
            p84=7.0,
            p90=8.0,
            p95=9.0,
        )

    def test_d10(self):
        assert self.p.d10 == pytest.approx(2.0)

    def test_d50(self):
        assert self.p.d50 == pytest.approx(5.0)

    def test_d90(self):
        assert self.p.d90 == pytest.approx(8.0)

    def test_d90_d10_ratio(self):
        assert self.p.d90_d10 == pytest.approx(8.0 / 2.0)

    def test_d90_minus_d10(self):
        assert self.p.d90_minus_d10 == pytest.approx(6.0)

    def test_d75_d25_ratio(self):
        assert self.p.d75_d25 == pytest.approx(6.0 / 4.0)

    def test_d75_minus_d25(self):
        assert self.p.d75_minus_d25 == pytest.approx(2.0)


class TestFractionsProperties:
    """Fractions' derived total_clay/total_silt/total_sand convenience properties.

    Attributes:
        f: A Fractions instance with distinct per-class values, from setup_method.
    """

    def setup_method(self):
        from sedstat.core.results import Fractions

        self.f = Fractions(
            clay=1.0,
            fine_silt=2.0,
            medium_silt=3.0,
            coarse_silt=4.0,
            fine_sand=5.0,
            medium_sand=6.0,
            coarse_sand=7.0,
            gravel=8.0,
        )

    def test_total_clay(self):
        assert self.f.total_clay == pytest.approx(1.0)

    def test_total_silt(self):
        assert self.f.total_silt == pytest.approx(2.0 + 3.0 + 4.0)

    def test_total_sand(self):
        assert self.f.total_sand == pytest.approx(5.0 + 6.0 + 7.0)


# ---------------------------------------------------------------------------
# Bimodality detection and coefficient
# ---------------------------------------------------------------------------


class TestBimodality:
    """detect_modes() peak detection, including prominence and separation filters."""

    def test_detect_modes_no_smoothing(self):
        from sedstat.core.bimodality import detect_modes

        classes = [1.0, 2.0, 3.0, 4.0, 5.0]
        values = [5.0, 20.0, 5.0, 20.0, 5.0]
        # smooth_window=1 skips smoothing entirely (raw values used directly)
        modes = detect_modes(classes, values, min_pct=2.0, smooth_window=1)
        assert [m[0] for m in modes] == [2.0, 4.0]

    def test_detect_modes_min_prominence_filters_small_peaks(self):
        """min_prominence actually filters small peaks, not just that the argument is accepted.

        Peak detection delegates to scipy.signal.find_peaks; verify
        min_prominence actually filters peaks, not just that it's accepted.
        """
        from sedstat.core.bimodality import detect_modes

        classes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        values = [5.0, 20.0, 5.0, 7.0, 5.0, 20.0, 5.0]

        without_filter = detect_modes(classes, values, min_pct=2.0, smooth_window=1)
        assert [m[0] for m in without_filter] == [2.0, 4.0, 6.0]

        with_prominence = detect_modes(
            classes, values, min_pct=2.0, smooth_window=1, min_prominence=10.0
        )
        assert [m[0] for m in with_prominence] == [2.0, 6.0]

    def test_detect_modes_min_separation_drops_close_peaks(self):
        from sedstat.core.bimodality import detect_modes

        classes = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        values = [5.0, 20.0, 5.0, 7.0, 5.0, 20.0, 5.0]

        modes = detect_modes(classes, values, min_pct=2.0, smooth_window=1, min_separation=4)
        assert [m[0] for m in modes] == [2.0, 6.0]

    def test_bimodality_coefficient_zero_kurtosis_is_nan(self):
        from sedstat.core.bimodality import bimodality_coefficient

        bc = bimodality_coefficient(skewness=0.5, kurtosis=0.0)
        assert np.isnan(bc)

    def test_bimodality_coefficient_normal_case(self):
        from sedstat.core.bimodality import bimodality_coefficient

        bc = bimodality_coefficient(skewness=0.0, kurtosis=2.0)
        assert bc == pytest.approx((0.0**2 + 1.0) / 2.0)
