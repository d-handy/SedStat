"""Tests for plot_stratigraphic() in sedstat.plotting."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from sedstat.plotting import plot_stratigraphic

BOUNDARIES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
VALUES = [1.0, 2.0, 5.0, 15.0, 30.0, 25.0, 12.0, 7.0, 3.0]  # sums to 100

assert abs(sum(VALUES) - 100.0) < 0.1


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
