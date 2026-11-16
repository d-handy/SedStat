"""Tests for plot_distribution() in sedstat.plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.figure import Figure

from sedstat.plotting import plot_distribution

BOUNDARIES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
VALUES = [1.0, 2.0, 5.0, 15.0, 30.0, 25.0, 12.0, 7.0, 3.0]  # sums to 100

assert abs(sum(VALUES) - 100.0) < 0.1


class TestPlotDistribution:
    """plot_distribution() figure factory behavior."""

    def test_returns_figure(self):
        fig = plot_distribution(BOUNDARIES, VALUES)
        assert isinstance(fig, Figure)

    def test_has_two_axes(self):
        fig = plot_distribution(BOUNDARIES, VALUES)
        assert len(fig.get_axes()) == 2

    def test_label_in_title(self):
        fig = plot_distribution(BOUNDARIES, VALUES, label="TestSample")
        assert "TestSample" in fig.get_axes()[0].get_title()

    def test_accepts_existing_figure(self):
        import matplotlib.pyplot as plt

        existing = plt.figure()
        returned = plot_distribution(BOUNDARIES, VALUES, fig=existing)
        assert returned is existing

    def test_upper_bounds_only(self):
        # N values, N upper bounds (no separate lower bound)
        upper_only = BOUNDARIES[1:]
        fig = plot_distribution(upper_only, VALUES)
        assert len(fig.get_axes()) == 2

    def test_phi_scale_xlabel(self):
        fig = plot_distribution(BOUNDARIES, VALUES, phi_scale=True)
        ax_cum = fig.get_axes()[1]
        assert "φ" in ax_cum.get_xlabel()

    def test_phi_scale_linear_xscale(self):
        fig = plot_distribution(BOUNDARIES, VALUES, phi_scale=True)
        ax_bar = fig.get_axes()[0]
        assert ax_bar.get_xscale() == "linear"

    def test_um_scale_log_xscale(self):
        fig = plot_distribution(BOUNDARIES, VALUES, phi_scale=False)
        ax_bar = fig.get_axes()[0]
        assert ax_bar.get_xscale() == "log"

    def test_single_class_does_not_raise(self):
        """A single-class N-form distribution no longer crashes with IndexError.

        A single-class N-form distribution (e.g. one lone pipette draw)
        used to crash with IndexError: upper[0]**2/upper[1] needs a
        second class that doesn't exist here. Found live by clicking
        through the actual GUI, not by static reading.
        """
        fig = plot_distribution([250.0], [100.0])
        assert len(fig.get_axes()) == 2

    def test_single_class_phi_scale_does_not_raise(self):
        fig = plot_distribution([250.0], [100.0], phi_scale=True)
        assert len(fig.get_axes()) == 2

    def _d50_annotation_value(self, fig):
        ax_cum = fig.get_axes()[1]
        for child in ax_cum.get_children():
            if hasattr(child, "get_text") and "D50" in str(child.get_text()):
                return float(child.get_text().split("=")[1].split()[0])
        raise AssertionError("No D50 annotation found")

    def test_um_scale_d50_matches_compute_all(self):
        """The plotted D50 (µm scale) matches compute_all()'s own P50, interpolated in phi space.

        Cross-checked against compute_all()'s own percentiles.p50, which
        is itself verified against a live GRADISTAT v9.1 run (see
        tests/test_statistics.py::TestAgainstGRADISTATv91). This function
        used to interpolate D50 linearly in µm space instead of phi space
        (570.0 µm here vs. the correct 561.9954 µm) — a ~1.4% error, not
        rounding noise, and inconsistent with how every percentile
        elsewhere in this codebase is computed.
        """
        from sedstat.core.statistics import compute_all

        classes = [250.0, 355.0, 500.0, 710.0, 1000.0, 1400.0]
        values = [15.0, 25.0, 30.0, 20.0, 10.0]
        expected = compute_all(classes, values).percentiles.p50

        fig = plot_distribution(classes, values, phi_scale=False)
        assert self._d50_annotation_value(fig) == pytest.approx(expected, abs=0.01)

    def test_phi_scale_d50_agrees_with_um_scale(self):
        """The phi-scale D50 (converted back to µm) agrees with the µm-scale D50.

        phi_scale=True used to plot the cumulative curve (and derive D50)
        against the wrong class edge (phi_lo, the fine/lower boundary,
        instead of phi_hi, the coarse/upper boundary) — every point was
        one class-width too far toward the fine side. Found live: it put
        D50 at 397.9 µm (1.33 φ) against the true 562.0 µm (0.83 φ), a
        ~30% error. Both display modes must now agree once converted.
        """
        import math

        classes = [250.0, 355.0, 500.0, 710.0, 1000.0, 1400.0]
        values = [15.0, 25.0, 30.0, 20.0, 10.0]

        fig_um = plot_distribution(classes, values, phi_scale=False)
        fig_phi = plot_distribution(classes, values, phi_scale=True)

        d50_um = self._d50_annotation_value(fig_um)
        d50_phi = self._d50_annotation_value(fig_phi)
        d50_um_from_phi = 1000.0 * 2.0**-d50_phi

        # Tolerance covers each annotation's own ".2f" display rounding
        # (phi's last displayed digit alone is worth a few µm once
        # exponentiated) — comfortably tighter than the ~168 µm (30%) the
        # original phi_lo/phi_hi bug produced.
        assert d50_um_from_phi == pytest.approx(d50_um, abs=3.0)
        assert math.isclose(d50_um, 561.9954323071133, abs_tol=0.5)

    def test_d50_annotation_has_legibility_halo(self):
        """The D50 label carries a white stroke halo so it stays legible over the curve.

        The D50 label sits at (d50_value, 50) -- on the cumulative curve
        by definition, since that's where the curve crosses 50%. Found
        live: for a steep curve (common right around the median, worse
        in phi-scale mode) a plain small offset wasn't enough to clear
        the line, and the text rendered partly on top of it. Fixed with
        a white stroke halo (same technique sedstat.plotting.ternary
        already uses for its point labels) so it stays legible
        regardless of exactly where the curve falls relative to it.
        """
        classes = [250.0, 355.0, 500.0, 710.0, 1000.0, 1400.0]
        values = [15.0, 25.0, 30.0, 20.0, 10.0]
        fig = plot_distribution(classes, values)
        ax_cum = fig.get_axes()[1]
        d50_text = next(
            child
            for child in ax_cum.get_children()
            if hasattr(child, "get_text") and "D50" in str(child.get_text())
        )
        assert len(d50_text.get_path_effects()) > 0
