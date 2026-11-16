"""Tests for plot_cumulative_overlay() in sedstat.plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest
from matplotlib.figure import Figure

from sedstat.plotting import PlotStyle, plot_cumulative_overlay

BOUNDARIES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
VALUES = [1.0, 2.0, 5.0, 15.0, 30.0, 25.0, 12.0, 7.0, 3.0]  # sums to 100

assert abs(sum(VALUES) - 100.0) < 0.1


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
