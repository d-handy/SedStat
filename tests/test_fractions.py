"""Tests for sedstat.core.fractions."""

import pytest

from sedstat.core.classification import ClassificationScheme
from sedstat.core.fractions import compute_fractions


class TestFractionBoundaries:
    """Each class is routed to the correct fraction."""

    def test_all_clay(self):
        f = compute_fractions([0.5, 1.0, 1.9], [30.0, 40.0, 30.0])
        assert abs(f.clay - 100.0) < 0.01

    def test_boundary_2_is_fine_silt_not_clay(self):
        f = compute_fractions([2.0], [100.0])
        assert abs(f.clay - 0.0) < 0.01
        assert abs(f.fine_silt - 100.0) < 0.01

    def test_boundary_63_is_fine_sand(self):
        f = compute_fractions([63.0], [100.0])
        assert abs(f.coarse_silt - 0.0) < 0.01
        assert abs(f.fine_sand - 100.0) < 0.01

    def test_all_gravel(self):
        f = compute_fractions([2500.0, 5000.0], [60.0, 40.0])
        assert abs(f.gravel - 100.0) < 0.01

    def test_mixed_fractions_sum_to_100(self):
        classes = [1.0, 4.0, 10.0, 40.0, 100.0, 300.0, 1000.0]
        values = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        f = compute_fractions(classes, values)
        total = (
            f.clay
            + f.fine_silt
            + f.medium_silt
            + f.coarse_silt
            + f.fine_sand
            + f.medium_sand
            + f.coarse_sand
            + f.gravel
        )
        assert abs(total - 100.0) < 0.01

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="zero"):
            compute_fractions([1.0, 4.0], [0.0, 0.0])

    def test_unmatched_class_is_silently_excluded_from_totals(self):
        """A class matching no bucket is dropped, and the rest renormalize to 100 alone.

        Every bucket has lo >= 0.0, so a negative upper bound matches none
        of them — the inner loop exhausts without a `break`. That class's
        value is dropped rather than raising (only an *all*-unmatched
        distribution raises, via test_zero_total_raises above), and the
        remaining matched classes are renormalized to 100 on their own.
        """
        f = compute_fractions([-5.0, 2.0], [50.0, 50.0])
        assert f.clay == pytest.approx(0.0)
        assert f.fine_silt == pytest.approx(100.0)


class TestISO14688FractionBoundaries:
    """ISO 14688-1:2017 shares Wentworth's clay/silt/sand boundaries exactly."""

    def test_matches_gradistat(self):
        classes = [1.0, 4.0, 10.0, 40.0, 100.0, 300.0, 1000.0]
        values = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        gradistat = compute_fractions(classes, values, ClassificationScheme.GRADISTAT)
        iso = compute_fractions(classes, values, ClassificationScheme.ISO14688)
        assert iso == gradistat

    def test_boundary_63_is_fine_sand(self):
        f = compute_fractions([63.0], [100.0], ClassificationScheme.ISO14688)
        assert abs(f.fine_sand - 100.0) < 0.01


class TestFractionHelpers:
    """Fractions' total_silt/total_sand convenience properties."""

    def test_total_silt(self):
        f = compute_fractions([4.0, 10.0, 40.0], [34.0, 33.0, 33.0])
        assert abs(f.total_silt - 100.0) < 0.01

    def test_total_sand(self):
        f = compute_fractions([100.0, 300.0, 1000.0], [34.0, 33.0, 33.0])
        assert abs(f.total_sand - 100.0) < 0.01
