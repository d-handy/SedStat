"""Analytically-provable reference-distribution tests for compute_all()."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest
from sedstat.core.statistics import compute_all

# ---------------------------------------------------------------------------
# Distribution builders
# ---------------------------------------------------------------------------


def _gaussian_phi_distribution(
    mu_phi: float, sigma_phi: float, n_bins: int = 4000, range_sigma: float = 6.0
) -> tuple[np.ndarray, np.ndarray]:
    """A Gaussian(mu_phi, sigma_phi) distribution, finely binned in phi space.

    Args:
        mu_phi: Mean of the distribution, in phi units.
        sigma_phi: Standard deviation of the distribution, in phi units.
        n_bins: Number of size classes to bin the distribution into.
        range_sigma: Half-width of the binned range, in multiples of
            sigma_phi on either side of mu_phi.

    Returns:
        A tuple of (class boundaries in µm, N+1, ascending; weight-% per
        class, N).
    """
    phi_hi = mu_phi + range_sigma * sigma_phi
    phi_lo = mu_phi - range_sigma * sigma_phi
    phi_boundaries = np.linspace(phi_hi, phi_lo, n_bins + 1)  # descending phi -> ascending um
    um_boundaries = 1000.0 * 2.0 ** (-phi_boundaries)
    mid_phi = (phi_boundaries[:-1] + phi_boundaries[1:]) / 2.0
    weights = np.exp(-0.5 * ((mid_phi - mu_phi) / sigma_phi) ** 2)
    weights = weights / weights.sum() * 100.0
    return um_boundaries, weights


def _gaussian_um_distribution(
    mu_um: float, sigma_um: float, n_bins: int = 4000, range_sigma: float = 6.0
) -> tuple[np.ndarray, np.ndarray]:
    """A Gaussian(mu_um, sigma_um) distribution, finely binned in linear µm space.

    Args:
        mu_um: Mean of the distribution, in µm.
        sigma_um: Standard deviation of the distribution, in µm.
        n_bins: Number of size classes to bin the distribution into.
        range_sigma: Half-width of the binned range, in multiples of
            sigma_um on either side of mu_um.

    Returns:
        A tuple of (class boundaries in µm, N+1, ascending; weight-% per
        class, N).
    """
    lo = mu_um - range_sigma * sigma_um
    hi = mu_um + range_sigma * sigma_um
    boundaries = np.linspace(lo, hi, n_bins + 1)
    mid = (boundaries[:-1] + boundaries[1:]) / 2.0
    weights = np.exp(-0.5 * ((mid - mu_um) / sigma_um) ** 2)
    weights = weights / weights.sum() * 100.0
    return boundaries, weights


def _exact_normal_percentiles(mu: float, sigma: float) -> dict[str, float]:
    """Exact P5..P95 of a continuous N(mu, sigma), via the stdlib normal quantile.

    Args:
        mu: Mean of the normal distribution.
        sigma: Standard deviation of the normal distribution.

    Returns:
        A mapping of percentile label (e.g. "p50") to its exact value.
    """
    nd = NormalDist(mu, sigma)
    return {f"p{int(p * 100)}": nd.inv_cdf(p) for p in (0.05, 0.16, 0.25, 0.50, 0.75, 0.84, 0.95)}


def _exact_folk_ward(percentiles: dict[str, float], *, log_space: bool) -> tuple[float, ...]:
    """Apply the published Folk & Ward (1957) formulas to exact percentiles.

    Independent of sedstat.core.statistics's own Folk & Ward implementation;
    re-derives the textbook formula from scratch as the oracle.

    Args:
        percentiles: Mapping of percentile label (e.g. "p50") to value, as
            returned by `_exact_normal_percentiles`.
        log_space: Whether `percentiles` are already in log space (True) or
            need to be log-transformed first (False).

    Returns:
        A tuple (mean, std, skewness, kurtosis) as magnitudes, since only
        scale/formula correctness is under test here, not sign convention.
    """
    if log_space:
        p5, p16, p25, p50, p75, p84, p95 = (
            percentiles[k] for k in ("p5", "p16", "p25", "p50", "p75", "p84", "p95")
        )
    else:
        p5, p16, p25, p50, p75, p84, p95 = (
            math.log(percentiles[k]) for k in ("p5", "p16", "p25", "p50", "p75", "p84", "p95")
        )

    mean = (p16 + p50 + p84) / 3.0
    std = abs((p84 - p16) / 4.0 + (p95 - p5) / 6.6)
    skew = abs(
        (p16 + p84 - 2.0 * p50) / (2.0 * (p84 - p16)) + (p5 + p95 - 2.0 * p50) / (2.0 * (p95 - p5))
    )
    kurt = abs((p95 - p5) / (2.44 * (p75 - p25)))

    if log_space:
        return mean, std, skew, kurt
    return math.exp(mean), math.exp(std), skew, kurt


# ---------------------------------------------------------------------------
# Logarithmic moments + Folk & Ward logarithmic — Gaussian in phi space
# ---------------------------------------------------------------------------

_MU_PHI, _SIGMA_PHI = 3.0, 0.5


class TestLogarithmicMomentsAgainstAnalyticNormal:
    """A Gaussian in phi space has exactly known moments.

    mean=mu, std=sigma, skewness=0, kurtosis=3 (classical/Pearson
    convention, matching classification.py's Mesokurtic band being
    centred near 3, not 0).

    Attributes:
        result: compute_all() on the finely-binned Gaussian, from setup_method.
    """

    def setup_method(self):
        um_boundaries, weights = _gaussian_phi_distribution(_MU_PHI, _SIGMA_PHI)
        self.result = compute_all(um_boundaries, weights)

    def test_mean(self):
        assert self.result.logarithmic.mean == pytest.approx(_MU_PHI, abs=1e-3)

    def test_std(self):
        assert self.result.logarithmic.std == pytest.approx(_SIGMA_PHI, abs=1e-3)

    def test_skewness_is_zero(self):
        assert self.result.logarithmic.skewness == pytest.approx(0.0, abs=1e-6)

    def test_kurtosis_is_three(self):
        assert self.result.logarithmic.kurtosis == pytest.approx(3.0, abs=1e-3)


class TestFolkWardLogAgainstAnalyticNormal:
    """Folk & Ward's formulas applied to exact normal quantiles as the oracle.

    Uses statistics.NormalDist, independent of sedstat's own percentile
    code. compute_all() on a finely discretized version of the same
    Gaussian must match it.

    Attributes:
        result: compute_all() on the finely-binned Gaussian, from setup_method.
        exact_mean: Analytic Folk & Ward log-space mean (oracle), from setup_method.
        exact_std: Analytic Folk & Ward log-space std (oracle), from setup_method.
        exact_skew: Analytic Folk & Ward log-space skewness magnitude (oracle).
        exact_kurt: Analytic Folk & Ward log-space kurtosis magnitude (oracle).
    """

    def setup_method(self):
        um_boundaries, weights = _gaussian_phi_distribution(_MU_PHI, _SIGMA_PHI)
        self.result = compute_all(um_boundaries, weights)
        exact_pct = _exact_normal_percentiles(_MU_PHI, _SIGMA_PHI)
        self.exact_mean, self.exact_std, self.exact_skew, self.exact_kurt = _exact_folk_ward(
            exact_pct, log_space=True
        )

    def test_mean(self):
        assert self.result.folk_ward_log.mean == pytest.approx(self.exact_mean, abs=1e-3)

    def test_std_matches_graphical_estimator_not_true_sigma(self):
        """Folk & Ward's log std is the graphical estimator (~0.4978), not the true sigma (0.5).

        Folk & Ward's std is a graphical estimator (quartile/decile-based),
        not the true population std -- for a normal it's ~0.4978, not
        exactly 0.5.
        """
        assert self.exact_std == pytest.approx(0.4978, abs=1e-3)  # sanity-check the oracle
        assert abs(self.result.folk_ward_log.std) == pytest.approx(self.exact_std, abs=1e-3)

    def test_skewness_is_zero(self):
        assert self.result.folk_ward_log.skewness == pytest.approx(0.0, abs=1e-6)

    def test_kurtosis_matches_graphical_estimator_not_exactly_one(self):
        """Folk & Ward's log kurtosis for a normal is ~0.9995, not exactly 1.0.

        Folk & Ward's kurtosis constant (2.44) makes a normal distribution
        come out near, but not exactly, 1.0 (~0.9995).
        """
        assert self.exact_kurt == pytest.approx(0.9995, abs=1e-3)
        assert abs(self.result.folk_ward_log.kurtosis) == pytest.approx(self.exact_kurt, abs=1e-3)


# ---------------------------------------------------------------------------
# Arithmetic moments + Folk & Ward geometric — Gaussian in linear µm space
# ---------------------------------------------------------------------------

_MU_UM, _SIGMA_UM = 500.0, 50.0


class TestArithmeticMomentsAgainstAnalyticNormal:
    """A Gaussian in linear µm space has known arithmetic moments (skewness=0, kurtosis=3).

    Attributes:
        result: compute_all() on the finely-binned Gaussian, from setup_method.
    """

    def setup_method(self):
        um_boundaries, weights = _gaussian_um_distribution(_MU_UM, _SIGMA_UM)
        self.result = compute_all(um_boundaries, weights)

    def test_mean(self):
        assert self.result.arithmetic.mean == pytest.approx(_MU_UM, abs=0.1)

    def test_std(self):
        assert self.result.arithmetic.std == pytest.approx(_SIGMA_UM, abs=0.1)

    def test_skewness_is_zero(self):
        assert self.result.arithmetic.skewness == pytest.approx(0.0, abs=1e-4)

    def test_kurtosis_is_three(self):
        assert self.result.arithmetic.kurtosis == pytest.approx(3.0, abs=1e-3)


class TestFolkWardGeoAgainstAnalyticNormal:
    """Same as TestFolkWardLogAgainstAnalyticNormal, for the geometric form.

    Folk & Ward's formulas are applied to ln(exact percentile), since
    folk_ward_geometric works in natural logs of µm percentiles. A
    linear-space Gaussian is not symmetric in ln-space, so unlike the
    phi-space case above, the exact target has nonzero skewness and a
    mean that differs from mu_um.

    Attributes:
        result: compute_all() on the finely-binned Gaussian, from setup_method.
        exact_mean: Analytic Folk & Ward geometric-space mean (oracle).
        exact_std: Analytic Folk & Ward geometric-space std (oracle).
        exact_skew: Analytic Folk & Ward geometric-space skewness magnitude (oracle).
        exact_kurt: Analytic Folk & Ward geometric-space kurtosis magnitude (oracle).
    """

    def setup_method(self):
        um_boundaries, weights = _gaussian_um_distribution(_MU_UM, _SIGMA_UM)
        self.result = compute_all(um_boundaries, weights)
        exact_pct = _exact_normal_percentiles(_MU_UM, _SIGMA_UM)
        self.exact_mean, self.exact_std, self.exact_skew, self.exact_kurt = _exact_folk_ward(
            exact_pct, log_space=False
        )

    def test_oracle_is_not_the_naive_arithmetic_mean(self):
        """The ln-space Folk & Ward oracle mean sits measurably below the naive arithmetic mean.

        Sanity-check on the oracle: ln-space Folk & Ward mean for a
        linear-space Gaussian is measurably below mu_um (498.3, not 500).
        """
        assert self.exact_mean == pytest.approx(498.3, abs=0.1)
        assert self.exact_mean != pytest.approx(_MU_UM, abs=0.5)

    def test_mean(self):
        assert self.result.folk_ward_geo.mean == pytest.approx(self.exact_mean, abs=0.5)

    def test_std(self):
        assert abs(self.result.folk_ward_geo.std) == pytest.approx(self.exact_std, abs=0.01)

    def test_skewness(self):
        assert abs(self.result.folk_ward_geo.skewness) == pytest.approx(self.exact_skew, abs=0.01)

    def test_kurtosis(self):
        assert abs(self.result.folk_ward_geo.kurtosis) == pytest.approx(self.exact_kurt, abs=0.01)


# ---------------------------------------------------------------------------
# Percentiles — exact, no Gaussian/erf needed: a uniform distribution's
# percentile p is trivially p-th-fraction-of-the-way between its bounds.
# ---------------------------------------------------------------------------


class TestPercentilesOfUniformDistribution:
    """A uniform distribution's percentiles are exact linear interpolations of its bounds.

    Attributes:
        lo: Lower bound (µm) of the uniform distribution, from setup_method.
        hi: Upper bound (µm) of the uniform distribution, from setup_method.
        result: compute_all() on the 1000-class uniform distribution, from setup_method.
    """

    def setup_method(self):
        # 1000 equal-width, equal-weight classes spanning [100, 1100] µm.
        self.lo, self.hi = 100.0, 1100.0
        boundaries = np.linspace(self.lo, self.hi, 1001)
        values = np.full(1000, 100.0 / 1000)
        self.result = compute_all(boundaries, values)

    @pytest.mark.parametrize("pct", [5, 10, 16, 25, 50, 75, 84, 90, 95])
    def test_percentile_matches_exact_linear_interpolation(self, pct):
        """compute_all()'s percentile matches the exact linear-interpolation formula.

        Args:
            pct: The percentile (5-95) to query and compare against the linear formula.
        """
        expected = self.lo + (pct / 100.0) * (self.hi - self.lo)
        actual = getattr(self.result.percentiles, f"p{pct}")
        assert actual == pytest.approx(expected, abs=0.5)
