"""Tests for sedstat.plotting — Qt-free figure factory functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure
from sedstat.plotting import (
    PlotStyle,
    plot_cumulative_overlay,
    plot_distribution,
    plot_stratigraphic,
    plot_ternary,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BOUNDARIES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
VALUES = [1.0, 2.0, 5.0, 15.0, 30.0, 25.0, 12.0, 7.0, 3.0]  # sums to 100

assert abs(sum(VALUES) - 100.0) < 0.1


# ---------------------------------------------------------------------------
# plot_distribution
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# plot_stratigraphic
# ---------------------------------------------------------------------------


class TestPlotStratigraphic:
    """plot_stratigraphic() figure factory behavior."""

    def test_returns_figure(self):
        fig = plot_stratigraphic(
            [BOUNDARIES, BOUNDARIES],
            [VALUES, VALUES],
            ["S1", "S2"],
        )
        assert isinstance(fig, Figure)

    def test_has_two_axes(self):
        # main plot + colorbar = 2 axes
        fig = plot_stratigraphic([BOUNDARIES], [VALUES], ["S1"])
        assert len(fig.get_axes()) == 2

    def test_ytick_labels_match(self):
        labels = ["Alpha", "Beta", "Gamma"]
        fig = plot_stratigraphic(
            [BOUNDARIES] * 3,
            [VALUES] * 3,
            labels,
        )
        ax = fig.get_axes()[0]
        tick_texts = [t.get_text() for t in ax.get_yticklabels()]
        for lbl in labels:
            assert lbl in tick_texts

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="least one"):
            plot_stratigraphic([], [], [])

    def test_depths_ylabel(self):
        fig = plot_stratigraphic(
            [BOUNDARIES, BOUNDARIES],
            [VALUES, VALUES],
            ["S1", "S2"],
            depths=[1.0, 2.5],
        )
        ax = fig.get_axes()[0]
        assert ax.get_ylabel() == "Depth [m]"

    def test_partial_depths_fallback(self):
        # If any depth is None the plot must still render (falls back to integer order)
        fig = plot_stratigraphic(
            [BOUNDARIES, BOUNDARIES],
            [VALUES, VALUES],
            ["S1", "S2"],
            depths=[1.0, None],
        )
        ax = fig.get_axes()[0]
        assert ax.get_ylabel() == "Sample"

    def test_single_depth_sample(self):
        # n_samples == 1 branch of _compute_y_axis (line 33)
        fig = plot_stratigraphic(
            [BOUNDARIES],
            [VALUES],
            ["S1"],
            depths=[5.0],
        )
        ax = fig.get_axes()[0]
        assert ax.get_ylabel() == "Depth [m]"

    def test_classes_with_n_plus_1_edges_used_directly(self):
        """_compute_x_edges() uses N+1 boundaries as-is, without inserting a synthetic edge.

        When len(classes) == n_classes + 1 the boundaries
        are used as-is (no synthetic lower edge inserted).
        """
        from sedstat.plotting.stratigraphic import _compute_x_edges

        classes = np.array(BOUNDARIES)  # 10 boundaries
        n_classes = len(VALUES)  # 9 values
        edges, upper = _compute_x_edges(classes, n_classes)
        assert edges is classes
        assert list(upper) == list(classes[1:])

    def test_classes_with_n_edges_synthesizes_lower_edge(self):
        """_compute_x_edges() synthesizes a lower edge for N-form (upper-bounds-only) input.

        When len(classes) == n_classes (upper bounds
        only), a synthetic first edge (classes[0] / 2) is prepended.
        """
        from sedstat.plotting.stratigraphic import _compute_x_edges

        classes = np.array(BOUNDARIES[1:])  # 9 upper bounds
        n_classes = len(VALUES)  # 9 values
        edges, upper = _compute_x_edges(classes, n_classes)
        assert edges[0] == pytest.approx(classes[0] / 2.0)
        assert list(edges[1:]) == list(classes)
        assert list(upper) == list(classes)


# ---------------------------------------------------------------------------
# plot_cumulative_overlay
# ---------------------------------------------------------------------------


class TestPlotCumulativeOverlay:
    """plot_cumulative_overlay() figure factory behavior."""

    def test_returns_figure(self):
        fig = plot_cumulative_overlay(
            [BOUNDARIES, BOUNDARIES],
            [VALUES, VALUES],
            ["A", "B"],
        )
        assert isinstance(fig, Figure)

    def test_has_one_axes(self):
        fig = plot_cumulative_overlay([BOUNDARIES], [VALUES], ["A"])
        assert len(fig.get_axes()) == 1

    def test_legend_contains_labels(self):
        labels = ["Sample1", "Sample2"]
        fig = plot_cumulative_overlay(
            [BOUNDARIES, BOUNDARIES],
            [VALUES, VALUES],
            labels,
        )
        legend_texts = [t.get_text() for t in fig.get_axes()[0].get_legend().get_texts()]
        assert set(labels) == set(legend_texts)

    def test_too_many_samples_raises(self):
        with pytest.raises(ValueError, match="20"):
            plot_cumulative_overlay(
                [BOUNDARIES] * 21,
                [VALUES] * 21,
                [str(i) for i in range(21)],
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="least one"):
            plot_cumulative_overlay([], [], [])

    def test_legend_is_outside_the_axes_by_default(self):
        fig = plot_cumulative_overlay(
            [BOUNDARIES, BOUNDARIES],
            [VALUES, VALUES],
            ["Sample1", "Sample2"],
        )
        fig.canvas.draw()
        ax = fig.get_axes()[0]
        legend_bbox = ax.get_legend().get_window_extent()
        axes_bbox = ax.get_window_extent()
        assert legend_bbox.x0 >= axes_bbox.x1

    def test_legend_outside_false_keeps_legend_inside_axes(self):
        fig = plot_cumulative_overlay(
            [BOUNDARIES, BOUNDARIES],
            [VALUES, VALUES],
            ["Sample1", "Sample2"],
            style=PlotStyle(legend_outside=False),
        )
        fig.canvas.draw()
        ax = fig.get_axes()[0]
        legend_bbox = ax.get_legend().get_window_extent()
        axes_bbox = ax.get_window_extent()
        assert legend_bbox.x0 < axes_bbox.x1

    def test_show_percentile_lines_false_draws_none(self):
        fig = plot_cumulative_overlay(
            [BOUNDARIES, BOUNDARIES],
            [VALUES, VALUES],
            ["A", "B"],
            style=PlotStyle(show_percentile_lines=False),
        )
        ax = fig.get_axes()[0]
        # 2 sample curves + 0 percentile lines = 2 lines total
        assert len(ax.get_lines()) == 2

    def test_explicit_percentile_lines_overrides_style(self):
        fig = plot_cumulative_overlay(
            [BOUNDARIES, BOUNDARIES],
            [VALUES, VALUES],
            ["A", "B"],
            percentile_lines=(10.0,),
            style=PlotStyle(show_percentile_lines=False),
        )
        ax = fig.get_axes()[0]
        # 2 sample curves + 1 percentile line = 3 lines total
        assert len(ax.get_lines()) == 3


# ---------------------------------------------------------------------------
# plot_ternary
# ---------------------------------------------------------------------------

CLAY = [10.0, 50.0, 5.0]
SILT = [30.0, 30.0, 10.0]
SAND = [60.0, 20.0, 85.0]
TLABELS = ["A", "B", "C"]


class TestPlotTernary:
    """plot_ternary() figure factory behavior."""

    def test_returns_figure(self):
        fig = plot_ternary(CLAY, SILT, SAND, TLABELS)
        assert isinstance(fig, Figure)

    def test_has_one_axes(self):
        fig = plot_ternary(CLAY, SILT, SAND, TLABELS)
        assert len(fig.get_axes()) == 1

    def test_single_sample(self):
        fig = plot_ternary([20.0], [30.0], [50.0], ["S1"])
        assert isinstance(fig, Figure)

    def test_normalisation_accepts_non_100_sum(self):
        # Fractions that don't sum to 100 should still render without error
        fig = plot_ternary([5.0], [10.0], [20.0], ["S1"])
        assert isinstance(fig, Figure)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="least one"):
            plot_ternary([], [], [], [])

    def test_close_points_get_labels_on_opposite_sides(self):
        """A label for a point close to another flips to the opposite side instead of overlapping.

        Two samples with near-identical composition — the routine case a
        ternary plot exists to reveal, not an edge case — used to both get
        labelled at a fixed offset above their point, merging into
        illegible overlapping text (found visually: "Core0"+"Core1"
        merged into "CcCore1"). The second, close label must now flip to
        the opposite side of its point instead of stacking on the first.
        """
        fig = plot_ternary(
            clay=[20.0, 20.5],
            silt=[30.0, 29.5],
            sand=[50.0, 50.0],
            labels=["Core0", "Core1"],
        )
        ax = fig.get_axes()[0]
        texts = {t.get_text(): t for t in ax.texts if t.get_text() in ("Core0", "Core1")}
        assert texts["Core0"].get_va() != texts["Core1"].get_va()

    def test_well_separated_points_both_labelled_above(self):
        # Sanity check the fix doesn't flip labels that don't need it.
        fig = plot_ternary(
            clay=[80.0, 5.0],
            silt=[10.0, 5.0],
            sand=[10.0, 90.0],
            labels=["Clayey", "Sandy"],
        )
        ax = fig.get_axes()[0]
        texts = {t.get_text(): t for t in ax.texts if t.get_text() in ("Clayey", "Sandy")}
        assert texts["Clayey"].get_va() == "bottom"
        assert texts["Sandy"].get_va() == "bottom"

    def test_more_than_twelve_samples_uses_legend(self):
        # n > 12 switches from inline text labels to a legend (lines 136-141)
        n = 13
        clay = [5.0] * n
        silt = [10.0] * n
        sand = [85.0] * n
        labels = [f"S{i}" for i in range(n)]
        fig = plot_ternary(clay, silt, sand, labels)
        ax = fig.get_axes()[0]
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert set(legend_texts) == set(labels)
