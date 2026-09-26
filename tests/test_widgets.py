"""GUI widget tests — requires pytest-qt and Qt offscreen platform.

Run with:
    pytest tests/test_widgets.py -q
The QT_QPA_PLATFORM=offscreen env var is set in conftest.py so no display
is needed.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog
from sedstat.gui.widgets.cumulative_plot import CumulativeOverlayWidget
from sedstat.gui.widgets.distribution_plot import DistributionPlotWidget
from sedstat.gui.widgets.results_table import (
    _COLUMN_GROUPS,
    _DEFAULT_HIDDEN_GROUPS,
    _DEPTH_COL,
    _HEADERS,
    _PINNED_HEADERS,
    ResultsTableWidget,
)
from sedstat.gui.widgets.sample_detail import SampleDetailWidget
from sedstat.gui.widgets.stratigraphic_plot import StratigraphicPlotWidget
from sedstat.gui.widgets.ternary_plot import TernaryPlotWidget

from sedstat.core.classification import ClassificationScheme
from sedstat.core.statistics import compute_all
from sedstat.io.beckman_coulter import LSRecord

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BOUNDARIES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
VALUES = [1.0, 2.0, 5.0, 15.0, 30.0, 25.0, 12.0, 7.0, 3.0]


def _make_result():
    return compute_all(BOUNDARIES, VALUES)


def _make_rec(name: str = "sample.av") -> LSRecord:
    return LSRecord(
        path=Path(name),
        classes_um=BOUNDARIES,
        values=VALUES,
        concentration=12.5,
        group_id=name.split(".")[0],
    )


# ---------------------------------------------------------------------------
# ResultsTableWidget
# ---------------------------------------------------------------------------


class TestResultsTableWidget:
    """ResultsTableWidget adds/reads sample rows and reflects column config."""

    def test_add_result_increases_row_count(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        assert widget.rowCount() == 0
        widget.add_result(_make_result(), _make_rec("a.av"))
        assert widget.rowCount() == 1
        widget.add_result(_make_result(), _make_rec("b.av"))
        assert widget.rowCount() == 2

    def test_row_data_returns_correct_pair(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        rec = _make_rec("sample.av")
        result = _make_result()
        widget.add_result(result, rec)
        data = widget.row_data(0)
        assert data is not None
        _r, rc = data
        assert rc.path == rec.path

    def test_same_basename_different_folders_gets_disambiguated(self, qtbot):
        """Two samples sharing a basename get distinct rows instead of overwriting each other.

        Two samples that share a basename (e.g. two "samples.csv" sieve/
        pipette imports from different folders) must not silently
        overwrite each other in _rows — previously they did, so row 0
        kept showing sample A's cells but row_data(0) resolved to
        sample B's data.
        """
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        rec_a = _make_rec("folderA/samples.csv")
        rec_b = _make_rec("folderB/samples.csv")
        assert rec_a.path.name == rec_b.path.name  # same basename, different files

        widget.add_result(_make_result(), rec_a)
        widget.add_result(_make_result(), rec_b)

        assert widget.rowCount() == 2
        name_0 = widget.item(0, 0).text()
        name_1 = widget.item(1, 0).text()
        assert name_0 != name_1  # disambiguated, not a silent duplicate key

        data_0 = widget.row_data(0)
        data_1 = widget.row_data(1)
        assert data_0 is not None and data_1 is not None
        assert data_0[1].path == rec_a.path
        assert data_1[1].path == rec_b.path
        assert widget.loaded_paths() == {str(rec_a.path), str(rec_b.path)}

    def test_readding_same_path_reuses_same_row_name(self, qtbot):
        """Re-adding the same file path replaces its own row instead of piling up "(2)", "(3)", ...

        Re-importing/refreshing the *same* file should keep replacing its
        own row (same display name), not pile up "(2)", "(3)", ... entries.
        """
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        rec = _make_rec("folderA/samples.csv")
        widget.add_result(_make_result(), rec)
        widget.add_result(_make_result(), rec)
        assert widget.item(0, 0).text() == widget.item(1, 0).text()

    def test_visible_rows_includes_all_when_no_filter(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        for name in ("a.av", "b.av", "c.av"):
            widget.add_result(_make_result(), _make_rec(name))
        assert len(widget.visible_rows()) == 3

    def test_visible_rows_respects_hidden(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        for name in ("a.av", "b.av", "c.av"):
            widget.add_result(_make_result(), _make_rec(name))
        widget.setRowHidden(1, True)
        assert len(widget.visible_rows()) == 2

    def test_depth_edit_updates_depths_dict(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        rec = _make_rec("sample.av")
        widget.add_result(_make_result(), rec)
        # Simulate user editing the depth cell (column 1)
        item = widget.item(0, 1)
        item.setText("3.75")
        # itemChanged signal fires synchronously in Qt
        assert widget._depths.get("sample.av") == pytest.approx(3.75)

    def test_to_dataframe_shape(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("s1.av"))
        widget.add_result(_make_result(), _make_rec("s2.av"))
        df = widget.to_dataframe()
        assert df.shape[0] == 2
        assert "filename" in df.columns
        assert "depth_m" in df.columns

    def test_add_result_with_none_concentration_formats_empty(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        rec = LSRecord(
            path=Path("noconc.av"),
            classes_um=BOUNDARIES,
            values=VALUES,
            concentration=None,
        )
        widget.add_result(_make_result(), rec)
        # Conc. [%] is table column 3 (Filename=0, Depth=1, GroupID=2, Conc.=3)
        assert widget.item(0, 3).text() == ""

    def test_row_data_negative_index_returns_none(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        assert widget.row_data(-1) is None

    def test_row_data_out_of_range_returns_none(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        assert widget.row_data(5) is None

    def test_row_data_missing_filename_item_returns_none(self, qtbot):

        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.insertRow(0)  # no item set in column 0
        assert widget.row_data(0) is None

    def test_to_dataframe_skips_hidden_rows(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        widget.add_result(_make_result(), _make_rec("b.av"))
        widget.setRowHidden(0, True)
        df = widget.to_dataframe()
        assert df.shape[0] == 1
        assert df.iloc[0]["filename"] == "b.av"

    def test_to_dataframe_skips_row_without_filename_item(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        widget.insertRow(1)  # no item in column 0 for this row
        df = widget.to_dataframe()
        assert df.shape[0] == 1

    def test_to_dataframe_skips_entry_not_in_rows_dict(self, qtbot):
        from PySide6.QtWidgets import QTableWidgetItem

        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        widget.insertRow(1)
        ghost = QTableWidgetItem("ghost.av")  # not present in widget._rows
        widget.setItem(1, 0, ghost)
        df = widget.to_dataframe()
        assert df.shape[0] == 1
        assert df.iloc[0]["filename"] == "a.av"

    def test_loaded_paths(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        assert widget.loaded_paths() == {str(Path("a.av"))}

    def test_loaded_paths_in_order_matches_insertion_order(self, qtbot):
        """loaded_paths_in_order() stays insertion-order, unlike a hash-seed-dependent set.

        Regression: session-save used to do list(loaded_paths()) -- a
        set's iteration order depends on Python's per-process string hash
        seed, so the exact same loaded files could serialize to a
        session file in a different order between two runs of the app
        (confirmed live across PYTHONHASHSEED=1/2/42, no data changed).
        loaded_paths_in_order() must stay insertion-order regardless.
        """
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        names = ["c.av", "a.av", "z.av", "m.av", "b.av"]
        for name in names:
            widget.add_result(_make_result(), _make_rec(name))
        assert widget.loaded_paths_in_order() == [str(Path(n)) for n in names]

    def test_depth_for(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        assert widget.depth_for("a.av") is None
        widget.set_depth("a.av", 4.2)
        assert widget.depth_for("a.av") == pytest.approx(4.2)

    def test_all_depths_excludes_none(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        widget.add_result(_make_result(), _make_rec("b.av"))
        widget.set_depth("a.av", 2.0)
        assert widget.all_depths() == {"a.av": 2.0}

    def test_set_depth_updates_cell_text(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        widget.set_depth("a.av", 7.25)
        assert widget.item(0, 1).text() == "7.25"
        widget.set_depth("a.av", None)
        assert widget.item(0, 1).text() == ""

    def test_update_depths_bulk(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        widget.add_result(_make_result(), _make_rec("b.av"))
        widget.update_depths({"a.av": 1.0, "b.av": 2.0})
        assert widget.depth_for("a.av") == pytest.approx(1.0)
        assert widget.depth_for("b.av") == pytest.approx(2.0)

    def test_raw_data_returns_result_rec_pairs(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        widget.add_result(_make_result(), _make_rec("b.av"))
        rows = widget.raw_data()
        assert len(rows) == 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in rows)

    def test_replace_all_same_scheme_rerenders_correctly(self, qtbot):
        """replace_all() under the same scheme re-renders the table identically.

        replace_all() never computes anything itself — the widget must not
        import sedstat.core.statistics — so re-rendering unchanged data
        under the same scheme should leave the table visibly identical.
        """
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        before = widget.item(0, 0).text()
        widget.replace_all(widget.raw_data(), ClassificationScheme.GRADISTAT)
        assert widget.item(0, 0).text() == before

    def test_replace_all_changes_scheme_and_preserves_depths(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        widget.add_result(_make_result(), _make_rec("b.av"))
        widget.set_depth("a.av", 3.0)

        # Recomputation under the new scheme is the caller's job (see
        # MainWindow._show_preferences), mirrored here for the test.
        new_scheme = ClassificationScheme.FOLK_WARD_1957
        new_rows = [
            (compute_all(result.classes_um, result.values, new_scheme), rec)
            for result, rec in widget.raw_data()
        ]
        widget.replace_all(new_rows, new_scheme)

        assert widget._scheme == new_scheme
        assert widget.rowCount() == 2
        assert widget.depth_for("a.av") == pytest.approx(3.0)

    def test_clear_data(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        widget.set_depth("a.av", 1.0)
        widget.clear_data()
        assert widget.rowCount() == 0
        assert widget.loaded_paths() == set()
        assert widget.all_depths() == {}

    def test_header_click_sets_sort_column_ascending(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        assert widget._sort_column is None
        widget._on_header_clicked(3)
        assert widget._sort_column == 3
        assert widget._sort_order == Qt.SortOrder.AscendingOrder

    def test_header_click_same_column_toggles_descending(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget._on_header_clicked(3)
        widget._on_header_clicked(3)  # same column again -> toggles order
        assert widget._sort_column == 3
        assert widget._sort_order == Qt.SortOrder.DescendingOrder
        widget._on_header_clicked(3)  # toggles back
        assert widget._sort_order == Qt.SortOrder.AscendingOrder

    def test_header_click_different_column_resets_to_ascending(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget._on_header_clicked(3)
        widget._on_header_clicked(3)  # now descending
        widget._on_header_clicked(5)  # different column -> back to ascending
        assert widget._sort_column == 5
        assert widget._sort_order == Qt.SortOrder.AscendingOrder

    def test_add_result_resorts_when_sort_column_already_set(self, qtbot):
        """A newly added row is re-sorted into place when a sort column is already active.

        Once the user has clicked a header to sort, newly added rows must
        be re-sorted into place rather than just appended.
        """
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("b.av"))
        widget._on_header_clicked(0)  # sort by Filename, ascending
        widget.add_result(_make_result(), _make_rec("a.av"))
        assert widget.item(0, 0).text() == "a.av"
        assert widget.item(1, 0).text() == "b.av"

    def test_replace_all_skips_reapplying_none_depths(self, qtbot):
        """replace_all() skips re-applying an explicit None depth instead of resetting it again.

        saved_depths can contain an explicit None (set via set_depth(...,
        None)); replace_all's re-apply loop must skip those rather than
        calling set_depth(filename, None) again.
        """
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        widget.set_depth("a.av", None)
        assert "a.av" in widget._depths
        assert widget._depths["a.av"] is None

        widget.replace_all(widget.raw_data(), ClassificationScheme.GRADISTAT)

        assert widget._depths.get("a.av") is None
        assert widget.item(0, _DEPTH_COL).text() == ""

    def test_item_changed_invalid_text_sets_depth_none(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        item = widget.item(0, 1)
        item.setText("not-a-number")
        assert widget._depths.get("a.av") is None

    def test_item_changed_no_filename_item_is_noop(self, qtbot):
        from PySide6.QtWidgets import QTableWidgetItem

        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.insertRow(0)  # column 0 left empty
        depth_item = QTableWidgetItem("1.0")
        # Directly invoke the slot to exercise the fn_item-is-None branch
        # without relying on itemChanged's exact emission semantics here.
        widget.setItem(0, 1, depth_item)
        widget._on_item_changed(depth_item)
        assert widget._depths == {}

    # -----------------------------------------------------------------
    # Column visibility groups
    # -----------------------------------------------------------------

    def test_column_groups_returns_all_group_names(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        assert widget.column_groups() == list(_COLUMN_GROUPS)
        assert len(widget.column_groups()) == 8

    def test_fresh_widget_matches_default_hidden_groups(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        for group in widget.column_groups():
            expected_hidden = group in _DEFAULT_HIDDEN_GROUPS
            assert widget.is_group_hidden(group) is expected_hidden, group

    def test_fresh_widget_sample_columns_always_visible(self, qtbot):
        """The pinned Sample columns belong to no group, so a group toggle never hides them.

        The pinned "Sample" columns (Filename/Depth/GroupID/Conc.) belong
        to no group at all — they must never be hidden by a group toggle.
        """
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        grouped_headers = {h for headers in _COLUMN_GROUPS.values() for h in headers}
        pinned = [h for h in _HEADERS if h not in grouped_headers]
        assert pinned == ["Filename", "Depth [m]", "GroupID", "Conc. [%]"]
        for header_text in pinned:
            assert not widget.isColumnHidden(_HEADERS.index(header_text))

    def test_set_group_hidden_round_trips(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.set_group_hidden("Fractions", False)
        assert widget.is_group_hidden("Fractions") is False
        for header_text in _COLUMN_GROUPS["Fractions"]:
            assert not widget.isColumnHidden(_HEADERS.index(header_text))

        widget.set_group_hidden("Fractions", True)
        assert widget.is_group_hidden("Fractions") is True
        for header_text in _COLUMN_GROUPS["Fractions"]:
            assert widget.isColumnHidden(_HEADERS.index(header_text))

    def test_hidden_column_data_still_readable(self, qtbot):
        """Column hiding is purely visual — cell data, row_data(), and export stay unaffected.

        Hiding is purely visual (setColumnHidden) — cell data, row_data(),
        and CSV export must all keep working regardless of visibility.
        """
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        assert widget.is_group_hidden("Fractions")  # hidden by default
        clay_col = _HEADERS.index("Clay [%]")
        assert widget.item(0, clay_col).text() != ""
        data = widget.row_data(0)
        assert data is not None

    # -----------------------------------------------------------------
    # row_detail() (single-sample detail view)
    # -----------------------------------------------------------------

    def test_row_detail_out_of_range_returns_none(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        assert widget.row_detail(-1) is None
        assert widget.row_detail(1) is None

    def test_row_detail_groups_and_headers(self, qtbot):
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        groups = widget.row_detail(0)
        assert groups is not None

        assert groups[0][0] == "Sample"
        assert [h for h, _ in groups[0][1]] == list(_PINNED_HEADERS)

        assert [name for name, _ in groups[1:]] == widget.column_groups()
        for name, pairs in groups[1:]:
            assert [h for h, _ in pairs] == list(_COLUMN_GROUPS[name])

    def test_row_detail_includes_hidden_group_values(self, qtbot):
        """row_detail() still reports a group's values even when that group is hidden in the table.

        A group being toggled off in the table must not affect what
        row_detail() reports — the Details panel shows everything.
        """
        widget = ResultsTableWidget()
        qtbot.addWidget(widget)
        widget.add_result(_make_result(), _make_rec("a.av"))
        assert widget.is_group_hidden("Fractions")  # hidden by default

        groups = dict(widget.row_detail(0))
        fractions = groups["Fractions"]
        assert len(fractions) == len(_COLUMN_GROUPS["Fractions"])
        assert all(value != "" for _, value in fractions)


# ---------------------------------------------------------------------------
# SampleDetailWidget
# ---------------------------------------------------------------------------


class TestSampleDetailWidget:
    """SampleDetailWidget toggles between its empty state and a populated sample view."""

    def test_starts_in_empty_state(self, qtbot):
        widget = SampleDetailWidget()
        qtbot.addWidget(widget)
        widget.show()
        assert widget._empty_label.isVisible()
        assert not widget._scroll.isVisible()

    def test_update_sample_shows_title_and_groups(self, qtbot):
        widget = SampleDetailWidget()
        qtbot.addWidget(widget)
        widget.show()
        groups = [
            ("Sample", [("Filename", "a.av"), ("GroupID", "C0")]),
            ("Percentiles", [("d10 [µm]", "1.234"), ("d50 [µm]", "5.678")]),
        ]
        widget.update_sample("C0", groups)

        assert not widget._empty_label.isVisible()
        assert widget._scroll.isVisible()
        assert widget._title_label.text() == "C0"
        # One QGroupBox per group, plus the trailing stretch item.
        assert widget._container_layout.count() == len(groups) + 1

        box0 = widget._container_layout.itemAt(0).widget()
        assert box0.title() == "Sample"
        form0 = box0.layout()
        assert form0.rowCount() == 2
        assert form0.itemAt(0, form0.ItemRole.LabelRole).widget().text() == "Filename"
        assert form0.itemAt(0, form0.ItemRole.FieldRole).widget().text() == "a.av"

    def test_group_name_with_ampersand_is_escaped_not_dropped(self, qtbot):
        """A group name containing "&" is escaped, not silently dropped as a mnemonic marker.

        QGroupBox titles treat a lone "&" as a mnemonic marker and would
        otherwise silently drop it (confirmed live: "Folk & Ward (φ)"
        rendered as "Folk_Ward (φ)").
        """
        widget = SampleDetailWidget()
        qtbot.addWidget(widget)
        widget.show()
        widget.update_sample("C0", [("Folk & Ward (φ)", [("FW Mean [φ]", "1.0")])])
        box = widget._container_layout.itemAt(0).widget()
        # box.title() returns the raw stored text (with the "&&" escape
        # still literally in it) -- Qt only interprets it at render time.
        assert box.title().replace("&&", "&") == "Folk & Ward (φ)"

    def test_update_sample_replaces_previous_groups(self, qtbot):
        widget = SampleDetailWidget()
        qtbot.addWidget(widget)
        widget.show()
        widget.update_sample("A", [("G1", [("h", "v")])])
        widget.update_sample("B", [("G2", [("h2", "v2")]), ("G3", [("h3", "v3")])])
        assert widget._container_layout.count() == 3  # 2 groups + stretch
        assert widget._container_layout.itemAt(0).widget().title() == "G2"

    def test_clear_resets_to_empty_state(self, qtbot):
        widget = SampleDetailWidget()
        qtbot.addWidget(widget)
        widget.show()
        widget.update_sample("C0", [("Sample", [("Filename", "a.av")])])
        widget.clear()
        assert widget._empty_label.isVisible()
        assert not widget._scroll.isVisible()
        assert widget._title_label.text() == ""
        assert widget._container_layout.count() == 1  # just the stretch


# ---------------------------------------------------------------------------
# DistributionPlotWidget
# ---------------------------------------------------------------------------


class TestDistributionPlotWidget:
    """DistributionPlotWidget rendering, phi/µm scale toggle, and clear/save behavior."""

    def test_update_plot_renders_two_panels(self, qtbot):
        widget = DistributionPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot(BOUNDARIES, VALUES, "TestSample")
        assert len(widget._figure.get_axes()) == 2

    def test_phi_toggle_changes_xscale(self, qtbot):
        widget = DistributionPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot(BOUNDARIES, VALUES, "S1")
        # Default: µm log scale
        assert widget._figure.get_axes()[0].get_xscale() == "log"
        # Toggle to phi
        widget._phi_check.setChecked(True)
        assert widget._figure.get_axes()[0].get_xscale() == "linear"

    def test_clear_hides_data(self, qtbot):
        widget = DistributionPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot(BOUNDARIES, VALUES, "S1")
        widget.clear()
        assert widget._values == []
        assert widget._classes_um == []

    def test_update_plot_with_empty_values_leaves_placeholder(self, qtbot):
        """update_plot([], [], ...) leaves the constructor's placeholder axis untouched.

        _render()'s `if not self._values: return` guard fires before the
        figure is touched, so the constructor's placeholder axis (from
        _draw_empty) survives an update_plot([], [], ...) call untouched.
        """
        widget = DistributionPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot([], [], "")
        axes = widget._figure.get_axes()
        assert len(axes) == 1
        assert axes[0].get_title() == widget._empty_message

    def test_save_default_name(self, qtbot):
        widget = DistributionPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot(BOUNDARIES, VALUES, "My Sample")
        assert widget._save_default_name() == "My_Sample"

    def test_save_default_name_falls_back_when_label_empty(self, qtbot):
        widget = DistributionPlotWidget()
        qtbot.addWidget(widget)
        assert widget._save_default_name() == "grain_size_plot"


# ---------------------------------------------------------------------------
# StratigraphicPlotWidget
# ---------------------------------------------------------------------------


class TestStratigraphicPlotWidget:
    """StratigraphicPlotWidget rendering with/without depths, and empty/clear states."""

    def test_update_plot_renders(self, qtbot):
        widget = StratigraphicPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot([BOUNDARIES, BOUNDARIES], [VALUES, VALUES], ["S1", "S2"])
        # main axes + colorbar axes = 2
        assert len(widget._figure.get_axes()) == 2

    def test_update_plot_with_empty_values_shows_placeholder(self, qtbot):
        widget = StratigraphicPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot([], [], [])
        ax = widget._figure.get_axes()[0]
        assert "No samples" in ax.get_title()

    def test_save_default_name(self, qtbot):
        widget = StratigraphicPlotWidget()
        qtbot.addWidget(widget)
        assert widget._save_default_name() == "stratigraphic"

    def test_update_plot_with_depths_sets_ylabel(self, qtbot):
        widget = StratigraphicPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot(
            [BOUNDARIES, BOUNDARIES],
            [VALUES, VALUES],
            ["S1", "S2"],
            depths=[1.0, 2.0],
        )
        ax = widget._figure.get_axes()[0]
        assert ax.get_ylabel() == "Depth [m]"

    def test_clear_shows_placeholder(self, qtbot):
        widget = StratigraphicPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot([BOUNDARIES], [VALUES], ["S1"])
        widget.clear()
        ax = widget._figure.get_axes()[0]
        assert "No samples" in ax.get_title()


# ---------------------------------------------------------------------------
# CumulativeOverlayWidget
# ---------------------------------------------------------------------------


class TestCumulativeOverlayWidget:
    """CumulativeOverlayWidget rendering, sample-count truncation, and clear behavior."""

    def test_update_renders(self, qtbot):
        widget = CumulativeOverlayWidget()
        qtbot.addWidget(widget)
        widget.update_plot([BOUNDARIES, BOUNDARIES], [VALUES, VALUES], ["A", "B"])
        assert len(widget._figure.get_axes()) == 1

    def test_update_plot_with_empty_values_shows_placeholder(self, qtbot):
        widget = CumulativeOverlayWidget()
        qtbot.addWidget(widget)
        widget.update_plot([], [], [])
        ax = widget._figure.get_axes()[0]
        assert "No samples" in ax.get_title()

    def test_save_default_name(self, qtbot):
        widget = CumulativeOverlayWidget()
        qtbot.addWidget(widget)
        assert widget._save_default_name() == "cumulative_overlay"

    def test_truncation_at_20_samples(self, qtbot):
        widget = CumulativeOverlayWidget()
        qtbot.addWidget(widget)
        n = 25
        widget.update_plot([BOUNDARIES] * n, [VALUES] * n, [str(i) for i in range(n)])
        # Should render without raising; title mentions truncation
        ax = widget._figure.get_axes()[0]
        assert "20" in ax.get_title()

    def test_clear_shows_placeholder(self, qtbot):
        widget = CumulativeOverlayWidget()
        qtbot.addWidget(widget)
        widget.update_plot([BOUNDARIES], [VALUES], ["A"])
        widget.clear()
        ax = widget._figure.get_axes()[0]
        assert "No samples" in ax.get_title()


# ---------------------------------------------------------------------------
# TernaryPlotWidget
# ---------------------------------------------------------------------------


class TestTernaryPlotWidget:
    """TernaryPlotWidget rendering and clear behavior."""

    def test_update_renders(self, qtbot):
        widget = TernaryPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot([10.0, 50.0], [30.0, 30.0], [60.0, 20.0], ["A", "B"])
        assert len(widget._figure.get_axes()) == 1

    def test_clear_shows_placeholder(self, qtbot):
        widget = TernaryPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot([10.0], [30.0], [60.0], ["A"])
        widget.clear()
        ax = widget._figure.get_axes()[0]
        assert "No samples" in ax.get_title()

    def test_update_plot_with_empty_clay_shows_placeholder(self, qtbot):
        widget = TernaryPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot([], [], [], [])
        ax = widget._figure.get_axes()[0]
        assert "No samples" in ax.get_title()

    def test_save_default_name(self, qtbot):
        widget = TernaryPlotWidget()
        qtbot.addWidget(widget)
        assert widget._save_default_name() == "ternary_diagram"


# ---------------------------------------------------------------------------
# MatplotlibPlotWidget (_base_plot) — save-to-file and context menu
# ---------------------------------------------------------------------------


class TestBasePlotWidget:
    """MatplotlibPlotWidget (_base_plot) save-to-file dialog handling."""

    def test_save_plot_writes_file(self, qtbot, monkeypatch, tmp_path):
        widget = TernaryPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot([10.0], [30.0], [60.0], ["A"])

        out_path = tmp_path / "out.png"
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *a, **k: (str(out_path), "PNG image (*.png)"),
        )
        widget._save_plot()
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_save_plot_cancelled_writes_nothing(self, qtbot, monkeypatch, tmp_path):
        widget = TernaryPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot([10.0], [30.0], [60.0], ["A"])

        out_path = tmp_path / "cancelled.png"
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *a, **k: ("", ""),
        )
        widget._save_plot()
        assert not out_path.exists()

    def test_save_plot_pdf_uses_no_explicit_dpi(self, qtbot, monkeypatch, tmp_path):
        widget = TernaryPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot([10.0], [30.0], [60.0], ["A"])

        out_path = tmp_path / "out.pdf"
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *a, **k: (str(out_path), "PDF document (*.pdf)"),
        )
        widget._save_plot()
        assert out_path.exists()

    def test_context_menu_event_triggers_save(self, qtbot, monkeypatch):
        """The context menu's Save action triggers save, without opening a real modal exec().

        menu.exec() opens a real modal event loop, which would hang the
        test under the offscreen platform and can't be reliably intercepted
        by monkeypatching the compiled QMenu.exec method. Instead, exercise
        the menu-building step directly and trigger its action.
        """
        widget = TernaryPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot([10.0], [30.0], [60.0], ["A"])

        called = {}

        def _fake_save():
            called["yes"] = True

        monkeypatch.setattr(widget, "_save_plot", _fake_save)

        menu = widget._build_context_menu()
        actions = menu.actions()
        assert len(actions) == 2
        assert actions[0].text() == "Save plot…"
        actions[0].trigger()
        assert called.get("yes") is True

    def test_context_menu_export_legend_disabled_when_no_legend(self, qtbot):
        """Below the 12-sample threshold, "Export legend" is disabled since there is no legend.

        Below the 12-sample threshold, ternary uses inline point labels
        instead of a legend -- nothing to export.
        """
        widget = TernaryPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot([10.0], [30.0], [60.0], ["A"])

        menu = widget._build_context_menu()
        export_action = menu.actions()[1]
        assert export_action.text() == "Export legend…"
        assert export_action.isEnabled() is False

    def test_context_menu_export_legend_enabled_when_legend_present(self, qtbot):
        widget = CumulativeOverlayWidget()
        qtbot.addWidget(widget)
        widget.update_plot([BOUNDARIES, BOUNDARIES], [VALUES, VALUES], ["A", "B"])

        menu = widget._build_context_menu()
        export_action = menu.actions()[1]
        assert export_action.isEnabled() is True

    def test_export_legend_writes_file_with_no_data_curves(self, qtbot, monkeypatch, tmp_path):
        widget = CumulativeOverlayWidget()
        qtbot.addWidget(widget)
        widget.update_plot([BOUNDARIES, BOUNDARIES], [VALUES, VALUES], ["A", "B"])

        out_path = tmp_path / "legend.png"
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *a, **k: (str(out_path), "PNG image (*.png)"),
        )
        widget._export_legend()
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_export_legend_default_name_includes_suffix(self, qtbot, monkeypatch, tmp_path):
        widget = CumulativeOverlayWidget()
        qtbot.addWidget(widget)
        widget.update_plot([BOUNDARIES, BOUNDARIES], [VALUES, VALUES], ["A", "B"])

        captured = {}

        def _fake_dialog(*args, **_kwargs):
            captured["default_name"] = args[2]
            return ("", "")

        monkeypatch.setattr(QFileDialog, "getSaveFileName", _fake_dialog)
        widget._export_legend()
        assert captured["default_name"] == "cumulative_overlay_legend"

    def test_export_legend_cancelled_writes_nothing(self, qtbot, monkeypatch, tmp_path):
        widget = CumulativeOverlayWidget()
        qtbot.addWidget(widget)
        widget.update_plot([BOUNDARIES, BOUNDARIES], [VALUES, VALUES], ["A", "B"])

        out_path = tmp_path / "cancelled_legend.png"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
        widget._export_legend()
        assert not out_path.exists()

    def test_export_legend_noop_without_legend(self, qtbot, monkeypatch):
        widget = TernaryPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot([10.0], [30.0], [60.0], ["A"])

        called = {}
        monkeypatch.setattr(
            QFileDialog,
            "getSaveFileName",
            lambda *a, **k: called.setdefault("hit", True),
        )
        widget._export_legend()
        assert "hit" not in called

    def test_context_menu_event_builds_and_execs_menu(self, qtbot, monkeypatch):
        from PySide6.QtCore import QPoint
        from PySide6.QtGui import QContextMenuEvent

        widget = TernaryPlotWidget()
        qtbot.addWidget(widget)
        widget.update_plot([10.0], [30.0], [60.0], ["A"])

        exec_calls = []

        class _FakeMenu:
            def exec(self, pos):
                exec_calls.append(pos)

        monkeypatch.setattr(widget, "_build_context_menu", lambda: _FakeMenu())

        pos = QPoint(5, 5)
        event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, pos)
        widget.contextMenuEvent(event)
        assert exec_calls == [event.globalPos()]

    def test_base_save_default_name(self, qtbot):
        from sedstat.gui.widgets._base_plot import MatplotlibPlotWidget

        widget = MatplotlibPlotWidget()
        qtbot.addWidget(widget)
        assert widget._save_default_name() == "plot"

    def test_canvas_has_minimum_size(self, qtbot):
        from sedstat.gui.widgets._base_plot import MatplotlibPlotWidget

        widget = MatplotlibPlotWidget()
        qtbot.addWidget(widget)
        assert widget._canvas.minimumSize().width() > 0
        assert widget._canvas.minimumSize().height() > 0

    def test_shrinking_does_not_emit_tight_layout_warning(self, qtbot):
        """Shrinking the plot down to its minimum size never triggers matplotlib's layout warning.

        Below a certain canvas size, matplotlib's
        tight_layout() can no longer fit the axes' labels/ticks/title and
        prints "Tight layout not applied... margins cannot be made large
        enough" straight to stderr (confirmed live: shrinking the plot
        panel/window reliably reproduced it). The fix is the canvas
        minimum size asserted above -- this test drives an actual resize
        sweep through what used to be the failing range and checks no
        UserWarning fires, the same way the live repro was verified.
        """
        widget = DistributionPlotWidget()
        qtbot.addWidget(widget)
        widget.show()
        widget.update_plot(BOUNDARIES, VALUES, "sample")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for h in (800, 400, 250, 150, 100, 60, 40):
                widget.resize(400, h)
                widget._canvas.draw()
            for w in (800, 400, 250, 150, 100, 60, 40):
                widget.resize(w, 400)
                widget._canvas.draw()

        tight_layout_warnings = [c for c in caught if "tight layout" in str(c.message).lower()]
        assert tight_layout_warnings == []
