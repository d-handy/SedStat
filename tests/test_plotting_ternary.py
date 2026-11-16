"""Tests for plot_ternary() in sedstat.plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.figure import Figure

from sedstat.plotting import plot_ternary

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
