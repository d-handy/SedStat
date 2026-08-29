"""Tests for sedstat.gui.sieve_pipette_dialog.SievePipetteDialog."""

from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QMessageBox
from sedstat.gui.sieve_pipette_dialog import SievePipetteDialog
from sedstat.io.sieve_pipette import read_sieve_pipette_csv, write_sieve_pipette_csv


def _write_temp_csv(content: str) -> Path:
    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(content)
    return Path(tmp.name)


class TestConstruction:
    """Initial dialog state: empty tables, default tabs, and overlap-strategy preselection."""

    def test_is_qdialog(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        assert isinstance(dlg, QDialog)

    def test_result_record_starts_none(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        assert dlg.result_record is None

    def test_sieve_table_starts_with_pan_row(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        assert dlg._sieve_table.rowCount() == 1
        assert dlg._sieve_table.item(0, 0).text() == "Pan"

    def test_pipette_table_starts_empty(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        assert dlg._pipette_table.rowCount() == 0

    def test_tabs_present(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        titles = [dlg._tabs.tabText(i) for i in range(dlg._tabs.count())]
        assert titles == ["Sieve", "Pipette"]

    def test_overlap_combo_defaults_to_reject(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        assert dlg._overlap_combo.currentData() == "reject"

    def test_overlap_combo_preselects_default_overlap_strategy(self, qtbot):
        dlg = SievePipetteDialog(default_overlap_strategy="truncate")
        qtbot.addWidget(dlg)
        assert dlg._overlap_combo.currentData() == "truncate"


class TestRowManagement:
    """Adding/removing sieve and pipette rows, including the protected pan row."""

    def test_add_sieve_row(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._add_sieve_row()
        assert dlg._sieve_table.rowCount() == 2

    def test_pan_row_cannot_be_removed(self, qtbot, monkeypatch):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
        dlg._sieve_table.setCurrentCell(0, 0)
        dlg._remove_selected_row(dlg._sieve_table)
        assert dlg._sieve_table.rowCount() == 1

    def test_non_pan_sieve_row_can_be_removed(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._add_sieve_row()
        dlg._sieve_table.setCurrentCell(1, 0)
        dlg._remove_selected_row(dlg._sieve_table)
        assert dlg._sieve_table.rowCount() == 1

    def test_remove_with_no_selection_is_noop(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._add_sieve_row()
        dlg._sieve_table.clearSelection()
        dlg._sieve_table.setCurrentCell(-1, -1)
        dlg._remove_selected_row(dlg._sieve_table)
        assert dlg._sieve_table.rowCount() == 2

    def test_add_pipette_row(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._add_pipette_row()
        assert dlg._pipette_table.rowCount() == 1

    def test_pipette_row_can_be_removed(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._add_pipette_row()
        dlg._pipette_table.setCurrentCell(0, 0)
        dlg._remove_selected_row(dlg._pipette_table)
        assert dlg._pipette_table.rowCount() == 0


class TestPipetteModeToggle:
    """Switching a pipette row's mode toggles which cells are editable."""

    def test_default_mode_enables_direct_size_cell(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._add_pipette_row()
        size_item = dlg._pipette_table.item(0, 1)
        assert size_item.flags() & Qt.ItemFlag.ItemIsEditable

    def test_switching_to_raw_mode_disables_size_and_enables_raw_cells(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._add_pipette_row()
        combo = dlg._pipette_table.cellWidget(0, 0)
        combo.setCurrentText("Raw Stokes calc")

        size_item = dlg._pipette_table.item(0, 1)
        time_item = dlg._pipette_table.item(0, 2)
        assert not (size_item.flags() & Qt.ItemFlag.ItemIsEditable)
        assert time_item.flags() & Qt.ItemFlag.ItemIsEditable


class TestReadRows:
    """_read_sieve_rows()/_read_pipette_rows() parse table contents, including raw-mode fields."""

    def test_read_sieve_rows_skips_blank_weight(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._add_sieve_row()  # blank row, no weight
        dlg._sieve_table.item(0, 1).setText("5.0")  # pan weight
        rows = dlg._read_sieve_rows()
        assert len(rows) == 1
        assert rows[0].mesh_um == 0.0
        assert rows[0].weight_g == 5.0

    def test_read_pipette_rows_direct_mode(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._add_pipette_row()
        dlg._pipette_table.item(0, 1).setText("16")
        dlg._pipette_table.item(0, 6).setText("1.8")
        rows = dlg._read_pipette_rows()
        assert len(rows) == 1
        assert rows[0].size_um == 16.0
        assert rows[0].time_s is None

    def test_read_pipette_rows_raw_mode(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._add_pipette_row()
        combo = dlg._pipette_table.cellWidget(0, 0)
        combo.setCurrentText("Raw Stokes calc")
        dlg._pipette_table.item(0, 2).setText("3600")
        dlg._pipette_table.item(0, 3).setText("10")
        dlg._pipette_table.item(0, 4).setText("20")
        dlg._pipette_table.item(0, 6).setText("1.0")
        rows = dlg._read_pipette_rows()
        assert rows[0].size_um is None
        assert rows[0].time_s == 3600.0
        assert rows[0].depth_cm == 10.0
        assert rows[0].temp_c == 20.0


class TestReadRowsSkipsBlankRows:
    """A pipette row with a size but no weight is dropped by _read_pipette_rows()."""

    def test_pipette_row_with_blank_weight_is_skipped(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._add_pipette_row()
        dlg._pipette_table.item(0, 1).setText("16")  # size filled, weight left blank
        rows = dlg._read_pipette_rows()
        assert rows == []


class TestPopulateFromRecordWithoutPanRow:
    """_populate_from_record() backfills a Pan row for pipette-only records."""

    def test_pipette_only_record_still_gets_a_pan_row(self, qtbot):
        from sedstat.io.sieve_pipette import PipetteRow, SievePipetteRecord

        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        record = SievePipetteRecord(
            path=Path("x"),
            classes_um=[16.0],
            values=[100.0],
            group_id="S1",
            method="pipette",
            pipette_rows=[PipetteRow(weight_g=1.0, size_um=16.0)],
        )
        dlg._populate_from_record(record)
        assert dlg._sieve_table.rowCount() == 1
        assert dlg._sieve_table.item(0, 0).text() == "Pan"


class TestAccept:
    """_on_accept() builds a result record from sieve/pipette/combined input, warns when empty."""

    def test_accept_with_sieve_only_builds_record(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._group_id_edit.setText("S1")
        dlg._sieve_table.item(0, 1).setText("5.0")
        dlg._add_sieve_row()
        dlg._sieve_table.item(1, 0).setText("250")
        dlg._sieve_table.item(1, 1).setText("60.0")

        dlg._on_accept()

        assert dlg.result() == QDialog.DialogCode.Accepted
        assert dlg.result_record is not None
        assert dlg.result_record.method == "sieve"
        assert dlg.result_record.group_id == "S1"
        assert str(dlg.result_record.path).startswith("<manual>")

    def test_accept_with_no_data_shows_warning_and_stays_open(self, qtbot, monkeypatch):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))

        dlg._on_accept()

        assert warned
        assert dlg.result_record is None
        assert dlg.result() != QDialog.DialogCode.Accepted

    def test_ok_button_triggers_accept(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._sieve_table.item(0, 1).setText("5.0")
        dlg._add_sieve_row()
        dlg._sieve_table.item(1, 0).setText("250")
        dlg._sieve_table.item(1, 1).setText("60.0")

        box = dlg.findChild(QDialogButtonBox)
        ok_button = box.button(QDialogButtonBox.StandardButton.Ok)
        qtbot.mouseClick(ok_button, Qt.MouseButton.LeftButton)

        assert dlg.result_record is not None

    def test_accept_with_pipette_only_builds_record(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._add_pipette_row()
        dlg._pipette_table.item(0, 1).setText("16")
        dlg._pipette_table.item(0, 6).setText("1.8")

        dlg._on_accept()

        assert dlg.result_record is not None
        assert dlg.result_record.method == "pipette"

    def test_accept_with_sieve_and_pipette_builds_combined_record(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._sieve_table.item(0, 1).setText("5.0")
        dlg._add_sieve_row()
        dlg._sieve_table.item(1, 0).setText("250")
        dlg._sieve_table.item(1, 1).setText("60.0")
        dlg._add_pipette_row()
        dlg._pipette_table.item(0, 1).setText("16")
        dlg._pipette_table.item(0, 6).setText("1.8")

        dlg._on_accept()

        assert dlg.result_record is not None
        assert dlg.result_record.method == "combined"

    def _add_overlapping_sieve_and_pipette(self, dlg):
        """Sieve finest mesh 250; pipette rows at 16 (kept) and 300 (overlaps)."""
        dlg._group_id_edit.setText("OverlapSample")
        dlg._sieve_table.item(0, 1).setText("5.0")
        dlg._add_sieve_row()
        dlg._sieve_table.item(1, 0).setText("250")
        dlg._sieve_table.item(1, 1).setText("60.0")
        dlg._add_pipette_row()
        dlg._pipette_table.item(0, 1).setText("16")
        dlg._pipette_table.item(0, 6).setText("1.8")
        dlg._add_pipette_row()
        dlg._pipette_table.item(1, 1).setText("300")
        dlg._pipette_table.item(1, 6).setText("2.1")

    def test_reject_strategy_blocks_overlapping_accept(self, qtbot, monkeypatch):
        dlg = SievePipetteDialog(default_overlap_strategy="reject")
        qtbot.addWidget(dlg)
        self._add_overlapping_sieve_and_pipette(dlg)
        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))

        dlg._on_accept()

        assert dlg.result_record is None
        assert warned
        assert "overlap" in warned[0][2].lower()

    def test_truncate_strategy_reconciles_overlapping_accept(self, qtbot):
        dlg = SievePipetteDialog(default_overlap_strategy="truncate")
        qtbot.addWidget(dlg)
        self._add_overlapping_sieve_and_pipette(dlg)

        dlg._on_accept()

        assert dlg.result_record is not None
        assert 300.0 not in dlg.result_record.classes_um  # overlapping class dropped
        assert sum(dlg.result_record.values) == 100.0

    def test_overlap_combo_overrides_the_passed_in_default(self, qtbot):
        """Changing the overlap combo before accepting overrides the dialog's default.

        Per-sample override: even though the dialog was opened with
        "reject" as the default, changing the combo before accepting
        uses the newly-selected strategy, not the constructor default.
        """
        dlg = SievePipetteDialog(default_overlap_strategy="reject")
        qtbot.addWidget(dlg)
        self._add_overlapping_sieve_and_pipette(dlg)
        truncate_index = dlg._overlap_combo.findData("truncate")
        dlg._overlap_combo.setCurrentIndex(truncate_index)

        dlg._on_accept()

        assert dlg.result_record is not None
        assert 300.0 not in dlg.result_record.classes_um

    def test_cancel_button_does_not_build_record(self, qtbot):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        box = dlg.findChild(QDialogButtonBox)
        cancel_button = box.button(QDialogButtonBox.StandardButton.Cancel)
        qtbot.mouseClick(cancel_button, Qt.MouseButton.LeftButton)
        assert dlg.result_record is None
        assert dlg.result() == QDialog.DialogCode.Rejected


class TestImportExportCsv:
    """_on_import_csv()/_on_export_csv() round-trip via the CSV file dialogs."""

    def test_import_single_sample_populates_tables(self, qtbot, monkeypatch):
        content = (
            "group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3\n"
            "S1,sieve,0,5.0,,,,,\n"
            "S1,sieve,250,60.0,,,,,\n"
            "S1,pipette,,1.8,16,,,,\n"
        )
        path = _write_temp_csv(content)
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))

        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._on_import_csv()

        assert dlg._group_id_edit.text() == "S1"
        assert dlg._pipette_table.rowCount() == 1

    def test_import_uses_dialogs_own_overlap_combo_setting(self, qtbot, monkeypatch):
        """CSV import honors the dialog's current overlap combo, not a hardcoded default.

        A single-sample CSV whose pipette range (up to 300) overlaps the
        sieve's finest mesh (250) -- _on_import_csv must honor whatever
        the dialog's own overlap combo is currently set to, the same
        per-sample override _on_accept respects, not a hardcoded default.
        """
        content = (
            "group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3\n"
            "S1,sieve,0,5.0,,,,,\n"
            "S1,sieve,250,60.0,,,,,\n"
            "S1,pipette,,1.8,16,,,,\n"
            "S1,pipette,,2.1,300,,,,\n"
        )
        path = _write_temp_csv(content)
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))

        dlg = SievePipetteDialog(default_overlap_strategy="reject")
        qtbot.addWidget(dlg)
        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
        dlg._on_import_csv()
        assert warned  # default "reject" blocks it
        assert dlg._pipette_table.rowCount() == 0  # not populated

        truncate_index = dlg._overlap_combo.findData("truncate")
        dlg._overlap_combo.setCurrentIndex(truncate_index)
        warned.clear()
        dlg._on_import_csv()
        assert not warned  # truncate reconciles it instead of rejecting
        assert dlg._group_id_edit.text() == "S1"
        # _populate_from_record fills the table from the *raw* pipette rows
        # for review/editing, not the already-truncated distribution -- both
        # rows (16 and the overlapping 300) are still shown.
        assert dlg._pipette_table.rowCount() == 2

    def test_import_multi_sample_shows_info_and_does_not_populate(self, qtbot, monkeypatch):
        content = (
            "group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3\n"
            "S1,sieve,0,5.0,,,,,\n"
            "S1,sieve,250,60.0,,,,,\n"
            "S2,sieve,0,3.0,,,,,\n"
            "S2,sieve,500,40.0,,,,,\n"
        )
        path = _write_temp_csv(content)
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
        informed = []
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: informed.append(a))

        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._on_import_csv()

        assert informed
        assert dlg._group_id_edit.text() == ""

    def test_import_cancelled_dialog_is_noop(self, qtbot, monkeypatch):
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._on_import_csv()
        assert dlg._sieve_table.rowCount() == 1  # unchanged (just the pan row)

    def test_import_invalid_csv_shows_warning(self, qtbot, monkeypatch):
        path = _write_temp_csv("not,a,valid,header\n1,2,3,4\n")
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), ""))
        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))

        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._on_import_csv()

        assert warned

    def test_export_writes_csv_reusing_write_function(self, qtbot, monkeypatch, tmp_path):
        out_path = tmp_path / "exported.csv"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(out_path), ""))

        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._group_id_edit.setText("S1")
        dlg._sieve_table.item(0, 1).setText("5.0")
        dlg._add_sieve_row()
        dlg._sieve_table.item(1, 0).setText("250")
        dlg._sieve_table.item(1, 1).setText("60.0")

        dlg._on_export_csv()

        assert out_path.exists()
        records = read_sieve_pipette_csv(out_path)
        assert records[0].group_id == "S1"

    def test_export_cancelled_dialog_is_noop(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._sieve_table.item(0, 1).setText("5.0")
        dlg._on_export_csv()  # should not raise

    def test_export_with_unparseable_mesh_shows_warning(self, qtbot, monkeypatch):
        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._sieve_table.item(0, 1).setText("5.0")
        dlg._add_sieve_row()
        dlg._sieve_table.item(1, 0).setText("not-a-number")
        dlg._sieve_table.item(1, 1).setText("60.0")
        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))

        dlg._on_export_csv()

        assert warned

    def test_export_write_failure_shows_warning(self, qtbot, monkeypatch, tmp_path):
        # A path pointing at an existing directory can't be opened for writing.
        bad_path = tmp_path  # a directory, not a file
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(bad_path), ""))
        warned = []
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))

        dlg = SievePipetteDialog()
        qtbot.addWidget(dlg)
        dlg._sieve_table.item(0, 1).setText("5.0")

        dlg._on_export_csv()

        assert warned


class TestWriteSievePipetteCsvHelperUsedDirectly:
    """write_sieve_pipette_csv()/read_sieve_pipette_csv() round-trip a record's group_id."""

    def test_write_then_read_round_trips_group_id(self, tmp_path):
        from sedstat.io.sieve_pipette import PipetteRow, SieveRow

        out_path = tmp_path / "manual.csv"
        write_sieve_pipette_csv(
            out_path,
            [SieveRow(mesh_um=0, weight_g=5.0), SieveRow(mesh_um=250, weight_g=60.0)],
            [PipetteRow(weight_g=1.8, size_um=16.0)],
            group_id="ManualTest",
        )
        records = read_sieve_pipette_csv(out_path)
        assert records[0].group_id == "ManualTest"
        assert records[0].method == "combined"
