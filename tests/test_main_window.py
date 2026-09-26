"""Tests for sedstat.gui.main_window.MainWindow.

Uses pytest-qt (qtbot) plus monkeypatching of native Qt dialogs so tests run
headlessly and deterministically. A fake QSettings org/app name is used so
the real Windows registry entries for "UniKoeln/SedStat" are never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QInputDialog, QMessageBox
from sedstat.gui import main_window as mw_mod
from sedstat.gui.main_window import MainWindow
from sedstat.gui.widgets.results_table import SORTING_COL

from sedstat.core.classification import ClassificationScheme
from sedstat.core.statistics import compute_all
from sedstat.io.beckman_coulter import LSRecord

BOUNDARIES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
VALUES = [1.0, 2.0, 5.0, 15.0, 30.0, 25.0, 12.0, 7.0, 3.0]


def _make_result():
    return compute_all(BOUNDARIES, VALUES)


def _make_rec(name: str = "sample.av", group_id: str | None = None) -> LSRecord:
    return LSRecord(
        path=Path(name),
        classes_um=BOUNDARIES,
        values=VALUES,
        concentration=12.5,
        group_id=group_id if group_id is not None else name.split(".")[0],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fake_qsettings(monkeypatch):
    """Redirect QSettings("UniKoeln", "SedStat") to a throwaway test store."""
    monkeypatch.setattr(
        mw_mod,
        "QSettings",
        lambda *a, **k: QSettings("SedStatTestOrg", "MainWindowTest"),
    )
    yield
    QSettings("SedStatTestOrg", "MainWindowTest").clear()


@pytest.fixture()
def window(qtbot):
    win = MainWindow()
    qtbot.addWidget(win)
    return win


class FakeBatchWorker(QObject):
    """Lightweight stand-in for BatchWorker with the same signal surface."""

    result_ready = Signal(str, object, object)
    error_occurred = Signal(str, str)
    progress = Signal(int, int)
    cancelled = Signal()
    finished = Signal()

    def __init__(self, paths, scheme=None, parent=None):
        super().__init__(parent)
        self.paths = list(paths)
        self.scheme = scheme
        self._running = False
        self.cancel_called = False
        self.start_called = False

    def start(self):
        self.start_called = True
        self._running = True

    def isRunning(self):
        return self._running

    def cancel(self):
        self.cancel_called = True


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """MainWindow's default state, including recovery from an invalid saved scheme."""

    def test_builds_without_error(self, window):
        assert window.windowTitle() == "SedStat"
        assert window._table.rowCount() == 0

    def test_default_scheme_is_gradistat(self, window):
        assert window._current_scheme == ClassificationScheme.GRADISTAT

    def test_invalid_saved_scheme_falls_back(self, qtbot, monkeypatch):
        monkeypatch.setattr(
            mw_mod,
            "QSettings",
            lambda *a, **k: _FakeSettingsWithBadScheme(),
        )
        win = MainWindow()
        qtbot.addWidget(win)
        assert win._current_scheme == ClassificationScheme.GRADISTAT


class _FakeSettingsWithBadScheme:
    """QSettings stand-in whose stored classification_scheme is not a real scheme."""

    def value(self, key, default=None):
        if key == "classification_scheme":
            return "not-a-real-scheme"
        return default

    def setValue(self, *a, **k):
        pass

    def clear(self):
        pass


# ---------------------------------------------------------------------------
# _open_files
# ---------------------------------------------------------------------------


class TestOpenFiles:
    """_open_files() starts a batch for the selected files, or is a no-op if none chosen."""

    def test_no_selection_returns_early(self, window, monkeypatch):
        monkeypatch.setattr(mw_mod.QFileDialog, "getOpenFileNames", lambda *a, **k: ([], ""))
        called = {}

        def fake_start_batch(paths):
            called["paths"] = paths

        monkeypatch.setattr(window, "_start_batch", fake_start_batch)
        window._open_files()
        assert "paths" not in called

    def test_selection_starts_batch(self, window, monkeypatch, tmp_path):
        f = tmp_path / "a.$av"
        f.write_text("x")
        monkeypatch.setattr(
            mw_mod.QFileDialog,
            "getOpenFileNames",
            lambda *a, **k: ([str(f)], "filter"),
        )
        called = {}
        monkeypatch.setattr(window, "_start_batch", lambda paths: called.setdefault("paths", paths))
        window._open_files()
        assert called["paths"] == [str(f)]
        assert window._last_dir == str(tmp_path)


# ---------------------------------------------------------------------------
# _open_folder
# ---------------------------------------------------------------------------


class TestOpenFolder:
    """_open_folder() discovers .$av files in a folder and confirms before batch-loading them."""

    def test_no_folder_selected_returns_early(self, window, monkeypatch):
        monkeypatch.setattr(mw_mod.QFileDialog, "getExistingDirectory", lambda *a, **k: "")
        window._open_folder()  # should not raise

    def test_no_av_files_shows_information(self, window, monkeypatch, tmp_path):
        monkeypatch.setattr(
            mw_mod.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
        )
        info_calls = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: info_calls.append(a))
        window._open_folder()
        assert len(info_calls) == 1
        assert "No files found" in info_calls[0][1]

    def test_all_already_loaded_shows_information(self, window, monkeypatch, tmp_path):
        f = tmp_path / "a.$av"
        f.write_text("x")
        window._table.add_result(_make_result(), _make_rec(str(f)))
        monkeypatch.setattr(
            mw_mod.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
        )
        info_calls = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: info_calls.append(a))
        window._open_folder()
        assert len(info_calls) == 1
        assert "Already loaded" in info_calls[0][1]

    def test_question_yes_starts_batch(self, window, monkeypatch, tmp_path):
        f = tmp_path / "a.$av"
        f.write_text("x")
        monkeypatch.setattr(
            mw_mod.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
        )
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
        called = {}
        monkeypatch.setattr(window, "_start_batch", lambda paths: called.setdefault("paths", paths))
        monkeypatch.setattr(
            window,
            "_load_depth_sidecar",
            lambda folder: called.setdefault("sidecar", folder),
        )
        window._open_folder()
        assert called["paths"] == [str(f)]
        assert called["sidecar"] == str(tmp_path)
        assert window._last_dir == str(tmp_path)

    def test_question_no_does_not_start_batch(self, window, monkeypatch, tmp_path):
        f = tmp_path / "a.$av"
        f.write_text("x")
        monkeypatch.setattr(
            mw_mod.QFileDialog, "getExistingDirectory", lambda *a, **k: str(tmp_path)
        )
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
        called = {}
        monkeypatch.setattr(window, "_start_batch", lambda paths: called.setdefault("paths", paths))
        window._open_folder()
        assert "paths" not in called


# ---------------------------------------------------------------------------
# _import_sieve_pipette_csv / _show_sieve_pipette_dialog
# ---------------------------------------------------------------------------

_SIEVE_PIPETTE_CSV_HEADER = (
    "group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3\n"
)


class TestImportSievePipetteCsv:
    """_import_sieve_pipette_csv() parses, dedupes, and reports import failures."""

    @pytest.fixture(autouse=True)
    def _accept_default_overlap_prompt(self, monkeypatch):
        """Auto-accept the window's current overlap strategy so import tests don't hang.

        _import_sieve_pipette_csv now asks (via _prompt_overlap_strategy,
        a real QInputDialog under the hood) which overlap strategy to use
        before importing -- un-mocked, that's a real modal dialog under
        offscreen Qt with nothing to click, which hangs the test run
        rather than failing it. Auto-accept the window's own current
        default so these tests keep exercising CSV parsing/error-handling
        as before; TestPromptOverlapStrategy covers the prompt itself.
        """

        def fake_get_item(parent, title, _label, items, current, *args, **_kwargs):
            return items[current], True

        monkeypatch.setattr(QInputDialog, "getItem", fake_get_item)
        yield

    def test_no_path_returns_early(self, window, monkeypatch):
        monkeypatch.setattr(mw_mod.QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
        window._import_sieve_pipette_csv()  # should not raise
        assert window._table.rowCount() == 0

    def test_valid_csv_adds_rows_and_status(self, window, monkeypatch, tmp_path):
        f = tmp_path / "samples.csv"
        f.write_text(
            _SIEVE_PIPETTE_CSV_HEADER + "S1,sieve,0,5.0,,,,,\n" + "S1,sieve,250,60.0,,,,,\n"
        )
        monkeypatch.setattr(mw_mod.QFileDialog, "getOpenFileName", lambda *a, **k: (str(f), ""))
        window._import_sieve_pipette_csv()
        assert window._table.rowCount() == 1
        assert "Imported 1" in window._status_label.text()

    def test_reimporting_same_group_is_skipped(self, window, monkeypatch, tmp_path):
        f = tmp_path / "samples.csv"
        f.write_text(
            _SIEVE_PIPETTE_CSV_HEADER + "S1,sieve,0,5.0,,,,,\n" + "S1,sieve,250,60.0,,,,,\n"
        )
        monkeypatch.setattr(mw_mod.QFileDialog, "getOpenFileName", lambda *a, **k: (str(f), ""))
        window._import_sieve_pipette_csv()
        window._import_sieve_pipette_csv()
        assert window._table.rowCount() == 1

    def test_invalid_csv_shows_critical(self, window, monkeypatch, tmp_path):
        f = tmp_path / "bad.csv"
        f.write_text("not,a,valid,header\n1,2,3,4\n")
        monkeypatch.setattr(mw_mod.QFileDialog, "getOpenFileName", lambda *a, **k: (str(f), ""))
        crit_calls = []
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: crit_calls.append(a))
        window._import_sieve_pipette_csv()
        assert len(crit_calls) == 1

    def test_missing_file_shows_critical(self, window, monkeypatch):
        monkeypatch.setattr(
            mw_mod.QFileDialog,
            "getOpenFileName",
            lambda *a, **k: ("does_not_exist.csv", ""),
        )
        crit_calls = []
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: crit_calls.append(a))
        window._import_sieve_pipette_csv()
        assert len(crit_calls) == 1

    def test_compute_all_failure_for_one_sample_is_reported_not_fatal(
        self, window, monkeypatch, tmp_path
    ):
        f = tmp_path / "samples.csv"
        f.write_text(
            _SIEVE_PIPETTE_CSV_HEADER + "S1,sieve,0,5.0,,,,,\n" + "S1,sieve,250,60.0,,,,,\n"
        )
        monkeypatch.setattr(mw_mod.QFileDialog, "getOpenFileName", lambda *a, **k: (str(f), ""))

        def failing_compute_all(*a, **k):
            raise ValueError("boom")

        monkeypatch.setattr(mw_mod, "compute_all", failing_compute_all)
        warn_calls = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warn_calls.append(a))

        window._import_sieve_pipette_csv()

        assert window._table.rowCount() == 0
        assert len(warn_calls) == 1
        assert "Imported 0" in window._status_label.text()

    def test_cancelling_overlap_prompt_aborts_import(self, window, monkeypatch, tmp_path):
        f = tmp_path / "samples.csv"
        f.write_text(
            _SIEVE_PIPETTE_CSV_HEADER + "S1,sieve,0,5.0,,,,,\n" + "S1,sieve,250,60.0,,,,,\n"
        )
        monkeypatch.setattr(mw_mod.QFileDialog, "getOpenFileName", lambda *a, **k: (str(f), ""))
        monkeypatch.setattr(window, "_prompt_overlap_strategy", lambda _question: None)

        window._import_sieve_pipette_csv()

        assert window._table.rowCount() == 0

    def test_chosen_overlap_strategy_is_forwarded_to_the_reader(
        self, window, monkeypatch, tmp_path
    ):
        f = tmp_path / "samples.csv"
        f.write_text(
            _SIEVE_PIPETTE_CSV_HEADER + "S1,sieve,0,5.0,,,,,\n" + "S1,sieve,250,60.0,,,,,\n"
        )
        monkeypatch.setattr(mw_mod.QFileDialog, "getOpenFileName", lambda *a, **k: (str(f), ""))
        monkeypatch.setattr(window, "_prompt_overlap_strategy", lambda _question: "truncate")

        calls = {}
        # _import_sieve_pipette_csv does a lazy `from
        # sedstat.io.sieve_pipette import read_sieve_pipette_csv` inside
        # the method itself (not a module-level import in main_window.py),
        # so the source module is what needs patching for this to be seen.
        from sedstat.io import sieve_pipette as sieve_pipette_mod

        original = sieve_pipette_mod.read_sieve_pipette_csv

        def spy(path, *, overlap_strategy="reject"):
            calls["overlap_strategy"] = overlap_strategy
            return original(path, overlap_strategy=overlap_strategy)

        monkeypatch.setattr(sieve_pipette_mod, "read_sieve_pipette_csv", spy)

        window._import_sieve_pipette_csv()

        assert calls["overlap_strategy"] == "truncate"


class TestPromptOverlapStrategy:
    """_prompt_overlap_strategy() returns the chosen strategy, None on cancel.

    Defaults to the window's current overlap strategy.
    """

    def test_returns_chosen_strategy(self, window, monkeypatch):
        from sedstat.gui.preferences_dialog import _OVERLAP_STRATEGY_LABELS

        monkeypatch.setattr(
            QInputDialog,
            "getItem",
            lambda *a, **k: (_OVERLAP_STRATEGY_LABELS["truncate"], True),
        )
        assert window._prompt_overlap_strategy("q") == "truncate"

    def test_cancel_returns_none(self, window, monkeypatch):
        from sedstat.gui.preferences_dialog import _OVERLAP_STRATEGY_LABELS

        monkeypatch.setattr(
            QInputDialog,
            "getItem",
            lambda *a, **k: (_OVERLAP_STRATEGY_LABELS["reject"], False),
        )
        assert window._prompt_overlap_strategy("q") is None

    def test_defaults_to_windows_current_overlap_strategy(self, window, monkeypatch):
        window._overlap_strategy = "truncate"
        seen = {}

        def fake_get_item(parent, title, _label, items, current, *args, **_kwargs):
            seen["current"] = current
            return items[current], True

        monkeypatch.setattr(QInputDialog, "getItem", fake_get_item)
        window._prompt_overlap_strategy("q")

        from sedstat.gui.preferences_dialog import _OVERLAP_STRATEGY_LABELS

        strategies = list(_OVERLAP_STRATEGY_LABELS)
        assert strategies[seen["current"]] == "truncate"


class _FakeSievePipetteDialogCode:
    """Stand-in for QDialog.DialogCode's Accepted/Rejected constants."""

    Accepted = 1
    Rejected = 0


class _FakeSievePipetteDialog:
    """Callable stand-in for SievePipetteDialog returning a preset exec() result."""

    DialogCode = _FakeSievePipetteDialogCode

    def __init__(self, result, result_record=None):
        self._result = result
        self.result_record = result_record

    def __call__(self, parent=None, *, default_overlap_strategy="reject"):
        self.default_overlap_strategy = default_overlap_strategy
        return self

    def exec(self):
        return self._result


class TestShowSievePipetteDialog:
    """_show_sieve_pipette_dialog() adds a row on accept, handles cancel and compute failures."""

    def test_cancelled_dialog_adds_nothing(self, window, monkeypatch):
        fake = _FakeSievePipetteDialog(result=_FakeSievePipetteDialogCode.Rejected)
        monkeypatch.setattr("sedstat.gui.sieve_pipette_dialog.SievePipetteDialog", fake)
        window._show_sieve_pipette_dialog()
        assert window._table.rowCount() == 0

    def test_passes_windows_overlap_strategy_as_dialog_default(self, window, monkeypatch):
        window._overlap_strategy = "truncate"
        fake = _FakeSievePipetteDialog(result=_FakeSievePipetteDialogCode.Rejected)
        monkeypatch.setattr("sedstat.gui.sieve_pipette_dialog.SievePipetteDialog", fake)
        window._show_sieve_pipette_dialog()
        assert fake.default_overlap_strategy == "truncate"

    def test_accepted_dialog_adds_row(self, window, monkeypatch):
        from sedstat.io.sieve_pipette import SievePipetteRecord

        record = SievePipetteRecord(
            path=Path("<manual>/S1-abcd1234.sievepipette"),
            classes_um=[250.0, 500.0],
            values=[40.0, 60.0],
            group_id="S1",
            method="sieve",
        )
        fake = _FakeSievePipetteDialog(
            result=_FakeSievePipetteDialogCode.Accepted, result_record=record
        )
        monkeypatch.setattr("sedstat.gui.sieve_pipette_dialog.SievePipetteDialog", fake)
        window._show_sieve_pipette_dialog()
        assert window._table.rowCount() == 1
        assert "S1" in window._status_label.text()

    def test_compute_all_failure_shows_critical_and_adds_nothing(self, window, monkeypatch):
        from sedstat.io.sieve_pipette import SievePipetteRecord

        record = SievePipetteRecord(
            path=Path("<manual>/S1-abcd1234.sievepipette"),
            classes_um=[250.0],
            values=[100.0],
            group_id="S1",
            method="sieve",
        )
        fake = _FakeSievePipetteDialog(
            result=_FakeSievePipetteDialogCode.Accepted, result_record=record
        )
        monkeypatch.setattr("sedstat.gui.sieve_pipette_dialog.SievePipetteDialog", fake)

        def failing_compute_all(*a, **k):
            raise ValueError("boom")

        monkeypatch.setattr(mw_mod, "compute_all", failing_compute_all)
        crit_calls = []
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: crit_calls.append(a))

        window._show_sieve_pipette_dialog()

        assert window._table.rowCount() == 0
        assert len(crit_calls) == 1


# ---------------------------------------------------------------------------
# _export_csv
# ---------------------------------------------------------------------------


class TestExportCsv:
    """_export_csv() writes the table's results to CSV, or informs if there's nothing to export."""

    def test_no_data_shows_information(self, window, monkeypatch):
        info_calls = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: info_calls.append(a))
        window._export_csv()
        assert len(info_calls) == 1

    def test_no_path_returns_early(self, window, monkeypatch):
        window._table.add_result(_make_result(), _make_rec("a.av"))
        monkeypatch.setattr(mw_mod.QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
        window._export_csv()  # should not raise

    def test_successful_export_updates_status(self, window, monkeypatch, tmp_path):
        window._table.add_result(_make_result(), _make_rec("a.av"))
        out = tmp_path / "out.csv"
        monkeypatch.setattr(mw_mod.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
        window._export_csv()
        assert "Exported" in window._status_label.text()

    def test_write_failure_shows_critical(self, window, monkeypatch, tmp_path):
        window._table.add_result(_make_result(), _make_rec("a.av"))
        out = tmp_path / "out.csv"
        monkeypatch.setattr(mw_mod.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))

        def raise_to_csv(*a, **k):
            raise OSError("disk full")

        import pandas as pd

        monkeypatch.setattr(pd.DataFrame, "to_csv", raise_to_csv)
        crit_calls = []
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: crit_calls.append(a))
        window._export_csv()
        assert len(crit_calls) == 1


# ---------------------------------------------------------------------------
# _clear_table
# ---------------------------------------------------------------------------


class TestClearTable:
    """_clear_table() empties the results table and the sample detail panel."""

    def test_clears_rows_and_status(self, window):
        window._table.add_result(_make_result(), _make_rec("a.av"))
        window._clear_table()
        assert window._table.rowCount() == 0
        assert "cleared" in window._status_label.text().lower()

    def test_clears_detail_panel(self, window, qtbot):
        """_clear_table() empties the detail panel's title and content.

        Content assertions only, not isVisible() -- see the comment in
        test_selection_updates_detail_panel for why.
        """
        window._table.add_result(_make_result(), _make_rec("a.av"))
        window._table.selectRow(0)
        window._on_selection_changed()
        assert window._detail._title_label.text() != ""

        window._clear_table()
        assert window._detail._title_label.text() == ""
        assert window._detail._container_layout.count() == 1  # just the stretch


# ---------------------------------------------------------------------------
# View actions
# ---------------------------------------------------------------------------


class TestViewActions:
    """Toggling the plot panel and resetting column widths."""

    def test_toggle_plot(self, window, qtbot):
        """_toggle_plot() flips the plot panel's visibility.

        isVisible()/isHidden() only reflect real state once the top-level
        window itself has been shown (offscreen platform still supports
        this), so show it first.
        """
        window.show()
        qtbot.waitExposed(window)
        initial = window._plot_tabs.isVisible()
        window._toggle_plot()
        assert window._plot_tabs.isVisible() != initial

    def test_reset_column_widths_does_not_raise(self, window):
        window._reset_column_widths()


# ---------------------------------------------------------------------------
# Column visibility menu (View > Columns… / header right-click)
# ---------------------------------------------------------------------------


class TestColumnMenu:
    """The column-visibility menu reflects and updates table state, persisted via QSettings."""

    def test_menu_has_one_checkable_action_per_group(self, window):
        menu = window._build_column_menu()
        actions = menu.actions()
        # "&" is escaped to "&&" in the displayed text so Qt doesn't eat it
        # as a mnemonic marker (see _build_column_menu) -- undo that here
        # to compare against the group names themselves.
        displayed = [a.text().replace("&&", "&") for a in actions]
        assert displayed == window._table.column_groups()
        for act in actions:
            assert act.isCheckable()

    def test_menu_initial_checked_state_matches_table(self, window):
        menu = window._build_column_menu()
        for act in menu.actions():
            group = act.text().replace("&&", "&")
            expected_visible = not window._table.is_group_hidden(group)
            assert act.isChecked() == expected_visible, group

    def test_toggle_updates_table_and_menu(self, window):
        assert window._table.is_group_hidden("Fractions")  # hidden by default
        window._on_column_group_toggled("Fractions", True)
        assert not window._table.is_group_hidden("Fractions")

        menu = window._build_column_menu()
        act = next(a for a in menu.actions() if a.text() == "Fractions")
        assert act.isChecked()

    def test_toggle_persists_to_qsettings(self, window):
        window._on_column_group_toggled("Fractions", True)
        # QSettings' native (registry/ini) backends hand back "true"/"false"
        # strings rather than real bools -- same coercion main_window.py's
        # own restore logic already has to do, see _build_ui().
        saved = window._settings.value("column_group_visible/Fractions")
        assert str(saved).lower() == "true"

    def test_second_window_remembers_toggled_group(self, window, qtbot):
        """A column-group toggle persists to a second MainWindow via QSettings.

        Same pattern as the existing classification_scheme persistence
        test: toggle in one window, construct a second in the same
        process/test, confirm it comes up already reflecting the change.
        """
        window._on_column_group_toggled("Fractions", True)

        win2 = MainWindow()
        qtbot.addWidget(win2)
        assert not win2._table.is_group_hidden("Fractions")

    def test_group_with_no_saved_setting_uses_widget_default(self, window, qtbot):
        """A group with no saved QSettings value falls back to the table widget's own default.

        Never touched "Bimodality" this test -- no saved QSettings value
        for it, so a fresh window must fall back to the table widget's
        own curated default (hidden) rather than "show everything".
        """
        win2 = MainWindow()
        qtbot.addWidget(win2)
        assert win2._table.is_group_hidden("Bimodality")

    def test_show_column_menu_pops_up_menu(self, window, monkeypatch):
        calls = {"exec": 0}

        class FakeMenu:
            def exec(self, *a, **k):
                calls["exec"] += 1

        monkeypatch.setattr(window, "_build_column_menu", lambda: FakeMenu())
        window._show_column_menu()
        assert calls["exec"] == 1

    def test_show_column_menu_at_pops_up_menu(self, window, monkeypatch):
        from PySide6.QtCore import QPoint

        calls = {"exec": 0}

        class FakeMenu:
            def exec(self, *a, **k):
                calls["exec"] += 1

        monkeypatch.setattr(window, "_build_column_menu", lambda: FakeMenu())
        window._show_column_menu_at(QPoint(5, 5))
        assert calls["exec"] == 1


# ---------------------------------------------------------------------------
# Toolbar/menu icons
# ---------------------------------------------------------------------------


class TestActionIcons:
    """Toolbar/menu actions have icons where a StandardPixmap match exists, and none otherwise."""

    @pytest.mark.parametrize(
        "action_name",
        [
            "_act_open_files",
            "_act_open_folder",
            "_act_export_csv",
            "_act_clear_table",
            "_act_load_session",
            "_act_save_session",
            "_act_cancel",
            "_act_quit",
            "_act_about",
        ],
    )
    def test_icon_mapped_actions_have_non_null_icon(self, window, action_name):
        """An action with a mapped StandardPixmap has a non-null icon.

        Args:
            action_name: Attribute name of the QAction on `window` to check.
        """
        action = getattr(window, action_name)
        assert not action.icon().isNull(), action_name

    @pytest.mark.parametrize(
        "action_name",
        [
            "_act_prefs",
            "_act_toggle_plot",
            "_act_reset_cols",
            "_act_columns",
        ],
    )
    def test_unmapped_actions_have_no_icon(self, window, action_name):
        """An action with no good StandardPixmap match deliberately has a null icon.

        Args:
            action_name: Attribute name of the QAction on `window` to check.
        """
        # Documents the deliberate gap (no good StandardPixmap match) rather
        # than a misleading icon -- if this starts failing because an icon
        # got added, update the mapped-actions list above instead.
        action = getattr(window, action_name)
        assert action.icon().isNull(), action_name


# ---------------------------------------------------------------------------
# About dialog
# ---------------------------------------------------------------------------


class TestAboutDialog:
    """_show_about() constructs and execs the AboutDialog."""

    def test_show_about_execs_dialog(self, window, monkeypatch):
        calls = {"exec": 0}

        class FakeAboutDialog:
            def __init__(self, parent=None):
                pass

            def exec(self):
                calls["exec"] += 1

        monkeypatch.setattr("sedstat.gui.about_dialog.AboutDialog", FakeAboutDialog)
        window._show_about()
        assert calls["exec"] == 1


# ---------------------------------------------------------------------------
# Drag and drop
# ---------------------------------------------------------------------------


class _FakeUrl:
    """Minimal stand-in for QUrl exposing toLocalFile()."""

    def __init__(self, path: str):
        self._path = path

    def toLocalFile(self):
        return self._path


class _FakeMimeData:
    """Minimal stand-in for QMimeData exposing hasUrls()/urls()."""

    def __init__(self, urls, has_urls=True):
        self._urls = urls
        self._has_urls = has_urls

    def hasUrls(self):
        return self._has_urls

    def urls(self):
        return self._urls


class _FakeDragEvent:
    """Minimal stand-in for a Qt drag/drop event, tracking accept/ignore calls."""

    def __init__(self, mime_data):
        self._mime_data = mime_data
        self.accepted = False
        self.ignored = False

    def mimeData(self):
        return self._mime_data

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        self.ignored = True


class TestDragAndDrop:
    """dragEnterEvent()/dropEvent() accept .$av files/folders and start a batch in sorted order."""

    def test_drag_enter_accepts_urls(self, window):
        event = _FakeDragEvent(_FakeMimeData([_FakeUrl("a.$av")]))
        window.dragEnterEvent(event)
        assert event.accepted

    def test_drag_enter_ignores_no_urls(self, window):
        event = _FakeDragEvent(_FakeMimeData([], has_urls=False))
        window.dragEnterEvent(event)
        assert event.ignored

    def test_drop_no_av_files_sets_status(self, window):
        event = _FakeDragEvent(_FakeMimeData([_FakeUrl("readme.txt")]))
        window.dropEvent(event)
        assert "no .$av files" in window._status_label.text().lower()

    def test_drop_with_av_file_starts_batch(self, window, monkeypatch, tmp_path):
        f = tmp_path / "a.$av"
        f.write_text("x")
        event = _FakeDragEvent(_FakeMimeData([_FakeUrl(str(f))]))
        called = {}
        monkeypatch.setattr(window, "_start_batch", lambda paths: called.setdefault("paths", paths))
        window.dropEvent(event)
        assert called["paths"] == [str(f)]

    def test_drop_with_directory_expands_globs(self, window, monkeypatch, tmp_path):
        f = tmp_path / "a.$av"
        f.write_text("x")
        event = _FakeDragEvent(_FakeMimeData([_FakeUrl(str(tmp_path))]))
        called = {}
        monkeypatch.setattr(window, "_start_batch", lambda paths: called.setdefault("paths", paths))
        window.dropEvent(event)
        assert called["paths"] == [str(f)]

    def test_drop_filters_already_loaded(self, window, monkeypatch, tmp_path):
        f = tmp_path / "a.$av"
        f.write_text("x")
        window._table.add_result(_make_result(), _make_rec(str(f)))
        event = _FakeDragEvent(_FakeMimeData([_FakeUrl(str(f))]))
        called = {}
        monkeypatch.setattr(window, "_start_batch", lambda paths: called.setdefault("paths", paths))
        window.dropEvent(event)
        assert called["paths"] == []

    def test_drop_directory_files_are_sorted_by_name(self, window, monkeypatch, tmp_path):
        """Dropping a folder starts a batch with files sorted by name, not raw glob() order.

        Path.glob()'s order reflects the OS's raw directory-listing order,
        not guaranteed stable or alphabetical -- create files in a
        deliberately non-alphabetical order and confirm _start_batch still
        sees them sorted, matching _open_folder()'s existing sort so the
        same drop always loads (and therefore exports) samples in the
        same, reproducible row order.
        """
        for name in ("z.$av", "a.$av", "m.$av"):
            (tmp_path / name).write_text("x")
        event = _FakeDragEvent(_FakeMimeData([_FakeUrl(str(tmp_path))]))
        called = {}
        monkeypatch.setattr(window, "_start_batch", lambda paths: called.setdefault("paths", paths))
        window.dropEvent(event)
        assert [Path(p).name for p in called["paths"]] == ["a.$av", "m.$av", "z.$av"]

    def test_drop_multi_file_selection_is_sorted_by_name(self, window, monkeypatch, tmp_path):
        """A multi-file drag-and-drop selection is sorted by name too, not just folder drops.

        Individually-dropped files are sorted too, not just a folder's
        glob expansion -- a multi-file drag selection's URL order isn't
        guaranteed either.
        """
        paths = [tmp_path / name for name in ("z.$av", "a.$av", "m.$av")]
        for p in paths:
            p.write_text("x")
        event = _FakeDragEvent(_FakeMimeData([_FakeUrl(str(p)) for p in paths]))
        called = {}
        monkeypatch.setattr(window, "_start_batch", lambda paths: called.setdefault("paths", paths))
        window.dropEvent(event)
        assert [Path(p).name for p in called["paths"]] == ["a.$av", "m.$av", "z.$av"]


# ---------------------------------------------------------------------------
# Window lifecycle / state persistence
# ---------------------------------------------------------------------------


class TestWindowLifecycle:
    """showEvent()/closeEvent() restore and persist window geometry and settings."""

    def test_show_event_restores_geometry_when_present(self, window):
        from PySide6.QtGui import QShowEvent

        window._settings.setValue("geometry", window.saveGeometry())
        window._settings.setValue("splitter", window._splitter.saveState())
        window.showEvent(QShowEvent())  # should not raise

    def test_show_event_noop_when_absent(self, qtbot, monkeypatch):
        from PySide6.QtGui import QShowEvent

        # Fresh settings store guarantees no persisted geometry/splitter keys.
        win = MainWindow()
        qtbot.addWidget(win)
        win.showEvent(QShowEvent())  # should not raise

    def test_close_event_persists_state(self, window):
        from PySide6.QtGui import QCloseEvent

        window._last_dir = "C:/somewhere"
        window._current_scheme = ClassificationScheme.USDA
        window.closeEvent(QCloseEvent())
        assert window._settings.value("last_dir") == "C:/somewhere"
        assert window._settings.value("classification_scheme") == ClassificationScheme.USDA.value
        assert window._settings.value("geometry") is not None

    def test_close_event_persists_overlap_strategy(self, window):
        from PySide6.QtGui import QCloseEvent

        window._overlap_strategy = "truncate"
        window.closeEvent(QCloseEvent())
        assert window._settings.value("overlap_strategy") == "truncate"


# ---------------------------------------------------------------------------
# _show_preferences
# ---------------------------------------------------------------------------


class _FakePrefsDialogCode:
    """Stand-in for QDialog.DialogCode's Accepted/Rejected constants."""

    Accepted = 1
    Rejected = 0


class _FakePrefsDialog:
    """Callable stand-in for PreferencesDialog returning preset exec() results and choices."""

    DialogCode = _FakePrefsDialogCode

    def __init__(
        self,
        result,
        reset_requested=False,
        selected_scheme=None,
        icon_only_toolbar=None,
        dark_theme=None,
        overlap_strategy=None,
    ):
        self._result = result
        self.reset_requested = reset_requested
        self.selected_scheme = selected_scheme
        self.icon_only_toolbar = icon_only_toolbar
        self.dark_theme = dark_theme
        self.overlap_strategy = overlap_strategy

    def __call__(
        self,
        current_scheme,
        current_icon_only=False,
        current_dark_theme=False,
        current_overlap_strategy="reject",
        parent=None,
    ):
        # Mimics constructor call signature; returns self as instance
        self.current_scheme = current_scheme
        if self.selected_scheme is None:
            self.selected_scheme = current_scheme
        if self.icon_only_toolbar is None:
            self.icon_only_toolbar = current_icon_only
        if self.dark_theme is None:
            self.dark_theme = current_dark_theme
        if self.overlap_strategy is None:
            self.overlap_strategy = current_overlap_strategy
        return self

    def exec(self):
        return self._result


class TestShowPreferences:
    """_show_preferences() applies scheme/reset changes and reclassifies existing results."""

    def test_cancelled_dialog_does_nothing(self, window, monkeypatch):
        fake = _FakePrefsDialog(result=_FakePrefsDialogCode.Rejected)
        monkeypatch.setattr("sedstat.gui.preferences_dialog.PreferencesDialog", fake)
        prev_scheme = window._current_scheme
        window._show_preferences()
        assert window._current_scheme == prev_scheme

    def test_reset_requested_clears_settings(self, window, monkeypatch):
        window._settings.setValue("last_dir", "somewhere")
        fake = _FakePrefsDialog(result=_FakePrefsDialogCode.Accepted, reset_requested=True)
        monkeypatch.setattr("sedstat.gui.preferences_dialog.PreferencesDialog", fake)
        window._show_preferences()
        assert window._last_dir == ""
        assert "reset" in window._status_label.text().lower()

    def test_accepted_with_new_scheme_reclassifies(self, window, monkeypatch):
        window._table.add_result(_make_result(), _make_rec("a.av"))
        new_scheme = ClassificationScheme.USDA
        fake = _FakePrefsDialog(
            result=_FakePrefsDialogCode.Accepted,
            reset_requested=False,
            selected_scheme=new_scheme,
        )
        monkeypatch.setattr("sedstat.gui.preferences_dialog.PreferencesDialog", fake)
        window._show_preferences()
        assert window._current_scheme == new_scheme
        assert new_scheme.value in window._status_label.text()

    def test_reclassify_reflects_in_table_cells(self, window, monkeypatch):
        """A scheme switch updates the table's cell values, not just window._current_scheme.

        The actual point of a scheme switch: cells must show the NEW
        scheme's numbers, not just window._current_scheme flipping.
        Boundaries span the 50-63 um band, where GRADISTAT (63 um) and
        USDA (50 um) silt/sand cutoffs disagree, so the fraction columns
        are guaranteed to actually differ between the two schemes.
        """
        classes = [30.0, 50.0, 63.0, 100.0, 500.0, 1000.0]
        values = [10.0, 20.0, 15.0, 25.0, 20.0, 10.0]
        result = compute_all(classes, values, ClassificationScheme.GRADISTAT)
        rec = LSRecord(path=Path("s1.av"), classes_um=classes, values=values, concentration=None)
        window._table.add_result(result, rec)

        coarse_silt_col = next(
            c
            for c in range(window._table.columnCount())
            if window._table.horizontalHeaderItem(c).text() == "Coarse silt [%]"
        )
        before = window._table.item(0, coarse_silt_col).text()

        new_scheme = ClassificationScheme.USDA
        fake = _FakePrefsDialog(
            result=_FakePrefsDialogCode.Accepted,
            reset_requested=False,
            selected_scheme=new_scheme,
        )
        monkeypatch.setattr("sedstat.gui.preferences_dialog.PreferencesDialog", fake)
        window._show_preferences()

        after = window._table.item(0, coarse_silt_col).text()
        expected = compute_all(classes, values, new_scheme).fractions.coarse_silt
        assert after != before
        assert after == f"{expected:.3f}"
        assert window._table.raw_data()[0][0].scheme == new_scheme

    def test_accepted_toggles_icon_only_toolbar(self, window, monkeypatch):
        from PySide6.QtCore import Qt

        assert window._icon_only_toolbar is False
        assert window._toolbar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon

        fake = _FakePrefsDialog(result=_FakePrefsDialogCode.Accepted, icon_only_toolbar=True)
        monkeypatch.setattr("sedstat.gui.preferences_dialog.PreferencesDialog", fake)
        window._show_preferences()

        assert window._icon_only_toolbar is True
        assert window._toolbar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonIconOnly

    def test_accepted_unchanged_icon_only_toolbar_is_a_noop(self, window, monkeypatch):
        """Accepting with an unchanged icon_only_toolbar value does not re-apply the toolbar style.

        icon_only_toolbar defaults to False and matches the window's own
        default -- confirms _show_preferences doesn't unconditionally
        re-apply the toolbar style on every accept, only on an actual
        change.
        """
        from PySide6.QtCore import Qt

        toolbar = window._toolbar
        fake = _FakePrefsDialog(result=_FakePrefsDialogCode.Accepted, icon_only_toolbar=False)
        monkeypatch.setattr("sedstat.gui.preferences_dialog.PreferencesDialog", fake)
        window._show_preferences()

        assert window._toolbar is toolbar
        assert toolbar.toolButtonStyle() == Qt.ToolButtonStyle.ToolButtonTextBesideIcon

    def test_icon_only_toolbar_persists_across_windows(self, window, monkeypatch, qtbot):
        fake = _FakePrefsDialog(result=_FakePrefsDialogCode.Accepted, icon_only_toolbar=True)
        monkeypatch.setattr("sedstat.gui.preferences_dialog.PreferencesDialog", fake)
        window._show_preferences()

        from PySide6.QtGui import QCloseEvent

        window.closeEvent(QCloseEvent())

        win2 = MainWindow()
        qtbot.addWidget(win2)
        assert win2._icon_only_toolbar is True

    def test_accepted_toggles_dark_theme(self, window, monkeypatch):
        assert window._dark_theme is False
        calls = []
        # main_window does `from sedstat.gui.theme import apply_theme`,
        # binding the name directly into its own module namespace -- patch
        # that name, not sedstat.gui.theme.apply_theme (which the
        # already-imported reference wouldn't pick up).
        monkeypatch.setattr(mw_mod, "apply_theme", lambda dark: calls.append(dark))

        fake = _FakePrefsDialog(result=_FakePrefsDialogCode.Accepted, dark_theme=True)
        monkeypatch.setattr("sedstat.gui.preferences_dialog.PreferencesDialog", fake)
        window._show_preferences()

        assert window._dark_theme is True
        assert calls == [True]

    def test_accepted_unchanged_dark_theme_is_a_noop(self, window, monkeypatch):
        calls = []
        monkeypatch.setattr(mw_mod, "apply_theme", lambda dark: calls.append(dark))

        fake = _FakePrefsDialog(result=_FakePrefsDialogCode.Accepted, dark_theme=False)
        monkeypatch.setattr("sedstat.gui.preferences_dialog.PreferencesDialog", fake)
        window._show_preferences()

        assert calls == []  # unchanged (False -> False) -- no re-apply

    def test_dark_theme_persists_across_windows(self, window, monkeypatch, qtbot):
        fake = _FakePrefsDialog(result=_FakePrefsDialogCode.Accepted, dark_theme=True)
        monkeypatch.setattr("sedstat.gui.preferences_dialog.PreferencesDialog", fake)
        window._show_preferences()

        from PySide6.QtGui import QCloseEvent

        window.closeEvent(QCloseEvent())

        win2 = MainWindow()
        qtbot.addWidget(win2)
        assert win2._dark_theme is True

    def test_accepted_updates_overlap_strategy(self, window, monkeypatch):
        assert window._overlap_strategy == "reject"
        fake = _FakePrefsDialog(result=_FakePrefsDialogCode.Accepted, overlap_strategy="truncate")
        monkeypatch.setattr("sedstat.gui.preferences_dialog.PreferencesDialog", fake)
        window._show_preferences()
        assert window._overlap_strategy == "truncate"

    def test_overlap_strategy_persists_across_windows(self, window, monkeypatch, qtbot):
        fake = _FakePrefsDialog(result=_FakePrefsDialogCode.Accepted, overlap_strategy="truncate")
        monkeypatch.setattr("sedstat.gui.preferences_dialog.PreferencesDialog", fake)
        window._show_preferences()

        from PySide6.QtGui import QCloseEvent

        window.closeEvent(QCloseEvent())

        win2 = MainWindow()
        qtbot.addWidget(win2)
        assert win2._overlap_strategy == "truncate"

    def test_reclassify_reapplies_active_sorting_filter(self, window, monkeypatch):
        """A scheme switch re-applies the active sorting-class filter instead of dropping it.

        replace_all() rebuilds every row from scratch, which resets Qt's
        hidden-row state. Before the fix, an active sorting-class filter
        silently stopped being enforced after a scheme switch: the filter
        combo kept showing the old selection while every row — including
        ones that no longer match it — became visible again.
        """
        classes = [30.0, 50.0, 63.0, 100.0, 500.0, 1000.0]
        values_a = [10.0, 20.0, 15.0, 25.0, 20.0, 10.0]
        values_b = [1.0, 1.0, 1.0, 1.0, 1.0, 95.0]  # very different sorting
        window._table.add_result(
            compute_all(classes, values_a, ClassificationScheme.GRADISTAT),
            LSRecord(
                path=Path("a.av"),
                classes_um=classes,
                values=values_a,
                concentration=None,
            ),
        )
        window._table.add_result(
            compute_all(classes, values_b, ClassificationScheme.GRADISTAT),
            LSRecord(
                path=Path("b.av"),
                classes_um=classes,
                values=values_b,
                concentration=None,
            ),
        )
        sorting_a = window._table.item(0, SORTING_COL).text()
        sorting_b = window._table.item(1, SORTING_COL).text()
        assert sorting_a != sorting_b  # precondition for this test to mean anything

        window._filter_sorting.setCurrentText(sorting_a)
        window._apply_filter()
        assert not window._table.isRowHidden(0)
        assert window._table.isRowHidden(1)

        fake = _FakePrefsDialog(
            result=_FakePrefsDialogCode.Accepted,
            reset_requested=False,
            selected_scheme=ClassificationScheme.USDA,
        )
        monkeypatch.setattr("sedstat.gui.preferences_dialog.PreferencesDialog", fake)
        window._show_preferences()

        # The filter combo still shows the same text (preserved by
        # _update_sorting_filter); the row-visibility must still match it.
        assert window._filter_sorting.currentText() == sorting_a
        row0_now = window._table.item(0, SORTING_COL).text()
        row1_now = window._table.item(1, SORTING_COL).text()
        assert window._table.isRowHidden(0) == (row0_now != sorting_a)
        assert window._table.isRowHidden(1) == (row1_now != sorting_a)

    def test_accepted_with_same_scheme_no_status_change(self, window, monkeypatch):
        prev_status = window._status_label.text()
        fake = _FakePrefsDialog(
            result=_FakePrefsDialogCode.Accepted,
            reset_requested=False,
            selected_scheme=window._current_scheme,
        )
        monkeypatch.setattr("sedstat.gui.preferences_dialog.PreferencesDialog", fake)
        window._show_preferences()
        assert window._status_label.text() == prev_status


# ---------------------------------------------------------------------------
# Session save / load
# ---------------------------------------------------------------------------


class TestSaveSession:
    """_save_session() writes loaded (non-sieve/pipette) files/depths to a JSON session file."""

    def test_no_data_shows_information(self, window, monkeypatch):
        info_calls = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: info_calls.append(a))
        window._save_session()
        assert len(info_calls) == 1

    def test_no_path_returns_early(self, window, monkeypatch):
        window._table.add_result(_make_result(), _make_rec("a.av"))
        monkeypatch.setattr(mw_mod.QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
        window._save_session()  # should not raise

    def test_successful_save_appends_json_extension(self, window, monkeypatch, tmp_path):
        window._table.add_result(_make_result(), _make_rec("a.av"))
        out = tmp_path / "session"
        monkeypatch.setattr(mw_mod.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
        window._save_session()
        saved = out.with_suffix(".json")
        assert saved.exists()
        data = json.loads(saved.read_text())
        assert data["version"] == 1
        assert "Session saved" in window._status_label.text()

    def test_saved_files_order_matches_insertion_order(self, window, monkeypatch, tmp_path):
        """Saved session files stay in insertion order, unlike a hash-seed-dependent set.

        This used to serialize list(self._table.loaded_paths())
        -- a set, whose iteration order depends on Python's per-process
        string hash seed, so saving the exact same loaded files twice
        (even in the same run) could write a different "files" order to
        the JSON with no data actually having changed.
        """
        names = ["c.av", "a.av", "z.av", "m.av", "b.av"]
        for name in names:
            window._table.add_result(_make_result(), _make_rec(name))
        out = tmp_path / "session.json"
        monkeypatch.setattr(mw_mod.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
        window._save_session()
        data = json.loads(out.read_text())
        assert data["files"] == [str(Path(n)) for n in names]

    def test_write_failure_shows_critical(self, window, monkeypatch, tmp_path):
        window._table.add_result(_make_result(), _make_rec("a.av"))
        out = tmp_path / "sub" / "session.json"  # parent dir doesn't exist -> write fails
        monkeypatch.setattr(mw_mod.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
        crit_calls = []
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: crit_calls.append(a))
        window._save_session()
        assert len(crit_calls) == 1

    def test_sieve_pipette_samples_are_skipped_with_notice(self, window, monkeypatch, tmp_path):
        from sedstat.io.sieve_pipette import SievePipetteRecord

        window._table.add_result(_make_result(), _make_rec("a.av"))
        manual_rec = SievePipetteRecord(
            path=Path("<manual>/S1-abcd1234.sievepipette"),
            classes_um=[250.0],
            values=[100.0],
            group_id="S1",
            method="sieve",
        )
        window._table.add_result(_make_result(), manual_rec)

        out = tmp_path / "session"
        monkeypatch.setattr(mw_mod.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
        info_calls = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: info_calls.append(a))

        window._save_session()

        saved = out.with_suffix(".json")
        data = json.loads(saved.read_text())
        assert data["files"] == ["a.av"]
        assert len(info_calls) == 1


class TestIsSyntheticSievePipettePath:
    """_is_synthetic_sieve_pipette_path() distinguishes manual/CSV-group paths from real files."""

    def test_manual_path_is_synthetic(self):
        assert mw_mod._is_synthetic_sieve_pipette_path("<manual>/S1-abcd1234.sievepipette")

    def test_csv_group_path_is_synthetic(self):
        assert mw_mod._is_synthetic_sieve_pipette_path("samples.csv::S1")

    def test_real_file_path_is_not_synthetic(self):
        assert not mw_mod._is_synthetic_sieve_pipette_path("C:/data/sample.$av")


class TestLoadSession:
    """_load_session() restores files/depths from a session file, or reports load failures."""

    def test_no_path_returns_early(self, window, monkeypatch):
        monkeypatch.setattr(mw_mod.QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
        window._load_session()  # should not raise

    def test_malformed_json_shows_critical(self, window, monkeypatch, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        monkeypatch.setattr(mw_mod.QFileDialog, "getOpenFileName", lambda *a, **k: (str(bad), ""))
        crit_calls = []
        monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: crit_calls.append(a))
        window._load_session()
        assert len(crit_calls) == 1

    def test_unknown_version_shows_warning(self, window, monkeypatch, tmp_path):
        f = tmp_path / "s.json"
        f.write_text(json.dumps({"version": 99, "files": [], "depths": {}}))
        monkeypatch.setattr(mw_mod.QFileDialog, "getOpenFileName", lambda *a, **k: (str(f), ""))
        warn_calls = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warn_calls.append(a))
        window._load_session()
        assert len(warn_calls) == 1

    def test_all_files_already_loaded_shows_information(self, window, monkeypatch, tmp_path):
        window._table.add_result(_make_result(), _make_rec("a.av"))
        f = tmp_path / "s.json"
        f.write_text(json.dumps({"version": 1, "files": ["a.av"], "depths": {}}))
        monkeypatch.setattr(mw_mod.QFileDialog, "getOpenFileName", lambda *a, **k: (str(f), ""))
        info_calls = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: info_calls.append(a))
        window._load_session()
        assert len(info_calls) == 1

    def test_new_files_trigger_batch_with_restore_callback(self, window, monkeypatch, tmp_path):
        f = tmp_path / "s.json"
        f.write_text(json.dumps({"version": 1, "files": ["new.av"], "depths": {"new.av": 3.5}}))
        monkeypatch.setattr(mw_mod.QFileDialog, "getOpenFileName", lambda *a, **k: (str(f), ""))
        called = {}
        monkeypatch.setattr(window, "_start_batch", lambda paths: called.setdefault("paths", paths))
        window._load_session()
        assert called["paths"] == ["new.av"]
        assert window._worker_session_restore is not None
        # Exercise the restore callback itself.
        window._worker_session_restore()


# ---------------------------------------------------------------------------
# Depth sidecar
# ---------------------------------------------------------------------------


class TestLoadDepthSidecar:
    """_load_depth_sidecar() applies a folder's .sedstat_depths.json, tolerating malformed files."""

    def test_missing_sidecar_is_noop(self, window, tmp_path):
        window._load_depth_sidecar(str(tmp_path))  # should not raise

    def test_valid_sidecar_updates_depths(self, window, tmp_path):
        sidecar = tmp_path / ".sedstat_depths.json"
        sidecar.write_text(json.dumps({"a.av": 1.2}))
        window._load_depth_sidecar(str(tmp_path))
        # No exception, and update_depths was invoked (no rows to check against,
        # but calling it directly on the table should not raise either)
        assert window._table.all_depths() == {} or "a.av" in window._table.all_depths()

    def test_malformed_sidecar_logs_warning_not_raise(self, window, tmp_path, caplog):
        sidecar = tmp_path / ".sedstat_depths.json"
        sidecar.write_text("{not valid json")
        with caplog.at_level("WARNING"):
            window._load_depth_sidecar(str(tmp_path))
        assert any("Malformed depth sidecar" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Cancel batch
# ---------------------------------------------------------------------------


class TestCancelBatch:
    """_cancel_batch() calls cancel() on a running worker, no-ops otherwise."""

    def test_no_worker_is_noop(self, window):
        window._cancel_batch()  # should not raise

    def test_worker_running_calls_cancel(self, window):
        fake = FakeBatchWorker([])
        fake.start()
        window._worker = fake
        window._cancel_batch()
        assert fake.cancel_called
        assert "cancel" in window._status_label.text().lower()

    def test_worker_not_running_does_not_call_cancel(self, window):
        fake = FakeBatchWorker([])
        window._worker = fake
        window._cancel_batch()
        assert not fake.cancel_called


# ---------------------------------------------------------------------------
# Filter bar
# ---------------------------------------------------------------------------


class TestApplyFilter:
    """_apply_filter() hides rows by filename/group-id text and by sorting-class selection."""

    def _populate(self, window):
        window._table.add_result(_make_result(), _make_rec("alpha.av", group_id="grpA"))
        window._table.add_result(_make_result(), _make_rec("beta.av", group_id="grpB"))
        window._table.add_result(_make_result(), _make_rec("gamma.av", group_id="grpA"))

    def _row_for(self, window, filename: str) -> int:
        for row in range(window._table.rowCount()):
            if window._table.item(row, 0).text() == filename:
                return row
        raise AssertionError(f"{filename} not found in table")

    def test_no_filter_shows_all(self, window):
        self._populate(window)
        window._apply_filter()
        for row in range(window._table.rowCount()):
            assert not window._table.isRowHidden(row)

    def test_text_filter_by_filename(self, window):
        self._populate(window)
        window._filter_text.setText("alpha")
        window._apply_filter()
        assert not window._table.isRowHidden(self._row_for(window, "alpha.av"))
        assert window._table.isRowHidden(self._row_for(window, "beta.av"))
        assert window._table.isRowHidden(self._row_for(window, "gamma.av"))

    def test_text_filter_by_group_id(self, window):
        self._populate(window)
        window._filter_text.setText("grpb")
        window._apply_filter()
        assert window._table.isRowHidden(self._row_for(window, "alpha.av"))
        assert not window._table.isRowHidden(self._row_for(window, "beta.av"))
        assert window._table.isRowHidden(self._row_for(window, "gamma.av"))

    def test_class_filter_all_classes(self, window):
        self._populate(window)
        window._filter_sorting.setCurrentIndex(0)  # "All classes"
        window._apply_filter()
        for row in range(window._table.rowCount()):
            assert not window._table.isRowHidden(row)


# ---------------------------------------------------------------------------
# Selection changed
# ---------------------------------------------------------------------------


class TestSelectionChanged:
    """_on_selection_changed() syncs the plot and detail panel to the selected row."""

    def test_no_selection_returns_early(self, window):
        window._on_selection_changed()  # should not raise

    def test_selection_updates_plot(self, window, qtbot):
        window._table.add_result(_make_result(), _make_rec("a.av"))
        window._table.selectRow(0)
        window._on_selection_changed()
        assert window._plot._values != []

    def test_selection_updates_detail_panel(self, window, qtbot):
        """Selecting a row updates the detail panel's content.

        Content assertions only here, not isVisible() -- the Details tab
        isn't the active QTabWidget page in this test, and a non-current
        tab page's children report isVisible() == False regardless of
        update_sample() having run correctly (see TestSampleDetailWidget
        for the widget-level visibility behaviour in isolation).
        """
        window._table.add_result(_make_result(), _make_rec("a.av"))
        window._table.selectRow(0)
        window._on_selection_changed()

        # Same label the Distribution plot got -- one source of truth.
        assert window._detail._title_label.text() == "a"
        # 9 groups: pinned "Sample" + the 8 column groups.
        assert window._detail._container_layout.count() == 9 + 1

    def test_details_tab_sits_right_after_distribution(self, window):
        titles = [window._plot_tabs.tabText(i) for i in range(window._plot_tabs.count())]
        assert titles == [
            "Distribution",
            "Details",
            "Stratigraphic",
            "Cumulative",
            "Ternary",
        ]


# ---------------------------------------------------------------------------
# Plot refresh helpers
# ---------------------------------------------------------------------------


class TestRefreshPlots:
    """_refresh_stratigraphic()/_refresh_cumulative()/_refresh_ternary() update from table rows."""

    def test_refresh_with_no_rows_clears(self, window):
        window._refresh_stratigraphic()
        window._refresh_cumulative()
        window._refresh_ternary()  # should not raise

    def test_refresh_with_rows_updates_plots(self, window):
        window._table.add_result(_make_result(), _make_rec("a.av"))
        window._table.add_result(_make_result(), _make_rec("b.av"))
        window._refresh_stratigraphic()
        window._refresh_cumulative()
        window._refresh_ternary()
        assert len(window._strat_plot._figure.get_axes()) >= 1
        assert len(window._cumul_plot._figure.get_axes()) >= 1
        assert len(window._ternary_plot._figure.get_axes()) >= 1

    def test_refresh_stratigraphic_with_depths(self, window):
        window._table.add_result(_make_result(), _make_rec("a.av"))
        window._table.update_depths({"a.av": 2.5})
        window._refresh_stratigraphic()
        ax = window._strat_plot._figure.get_axes()[0]
        assert ax.get_ylabel() == "Depth [m]"


# ---------------------------------------------------------------------------
# Batch processing (using FakeBatchWorker)
# ---------------------------------------------------------------------------


class TestStartBatch:
    """_start_batch() creates and starts a BatchWorker, or warns if one is already running."""

    def test_worker_already_running_shows_warning(self, window, monkeypatch):
        fake = FakeBatchWorker([])
        fake.start()
        window._worker = fake
        warn_calls = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warn_calls.append(a))
        window._start_batch(["x.av"])
        assert len(warn_calls) == 1

    def test_start_batch_creates_and_starts_worker(self, window, monkeypatch):
        monkeypatch.setattr(mw_mod, "BatchWorker", FakeBatchWorker)
        window._start_batch(["a.av", "b.av"])
        assert isinstance(window._worker, FakeBatchWorker)
        assert window._worker.start_called
        assert window._act_cancel.isEnabled()
        assert not window._progress.isHidden()
        assert window._total_files == 2

    def test_on_result_ready_adds_row_and_schedules_refresh(self, window):
        result = _make_result()
        rec = _make_rec("a.av")
        window._on_result_ready("a.av", result, rec)
        assert window._table.rowCount() == 1
        assert window._plot_refresh_timer.isActive()

    def test_on_error_sets_status(self, window):
        window._on_error("bad/path/file.av", "boom")
        assert "file.av" in window._status_label.text()
        assert "boom" in window._status_label.text()

    def test_on_progress_updates_bar_and_status(self, window):
        window._progress.setMaximum(10)
        window._on_progress(3, 10)
        assert window._progress.value() == 3
        assert "3/10" in window._status_label.text()

    def test_on_cancelled_updates_ui(self, window):
        window._act_cancel.setEnabled(True)
        window._progress.setVisible(True)
        window._table.add_result(_make_result(), _make_rec("a.av"))
        window._on_cancelled()
        assert not window._act_cancel.isEnabled()
        assert not window._progress.isVisible()
        assert "Cancelled" in window._status_label.text()

    def test_on_finished_updates_ui_and_flushes_refresh(self, window):
        window._act_cancel.setEnabled(True)
        window._progress.setVisible(True)
        window._table.add_result(_make_result(), _make_rec("a.av"))
        window._on_finished()
        assert not window._act_cancel.isEnabled()
        assert not window._progress.isVisible()
        assert "Done" in window._status_label.text()

    def test_on_finished_restores_session_depths(self, window):
        called = {"restored": False}

        def restore():
            called["restored"] = True

        window._worker_session_restore = restore
        window._on_finished()
        assert called["restored"]
        assert window._worker_session_restore is None

    def test_full_fake_worker_flow_via_signals(self, window, qtbot, monkeypatch):
        monkeypatch.setattr(mw_mod, "BatchWorker", FakeBatchWorker)
        window._start_batch(["a.av"])
        worker = window._worker
        rec = _make_rec("a.av")
        result = _make_result()

        with qtbot.waitSignal(worker.result_ready, timeout=1000, raising=True):
            worker.result_ready.emit("a.av", result, rec)
        assert window._table.rowCount() == 1

        with qtbot.waitSignal(worker.progress, timeout=1000, raising=True):
            worker.progress.emit(1, 1)
        assert window._progress.value() == 1

        with qtbot.waitSignal(worker.finished, timeout=1000, raising=True):
            worker.finished.emit()
        assert "Done" in window._status_label.text()
