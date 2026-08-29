"""Tests for sedstat.gui.preferences_dialog.PreferencesDialog."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPushButton
from sedstat.core.classification import ClassificationScheme
from sedstat.gui.preferences_dialog import PreferencesDialog


class TestPreferencesDialogConstruction:
    """PreferencesDialog's initial state reflects the constructor's current-value arguments."""

    def test_is_qdialog(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)
        assert isinstance(dlg, QDialog)

    def test_window_title(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)
        assert dlg.windowTitle() == "Preferences"

    def test_minimum_width(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)
        assert dlg.minimumWidth() == 380

    def test_initial_selected_scheme_matches_current(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.USDA)
        qtbot.addWidget(dlg)
        assert dlg.selected_scheme == ClassificationScheme.USDA

    def test_reset_requested_defaults_false(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)
        assert dlg.reset_requested is False

    def test_icon_only_toolbar_defaults_false(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)
        assert dlg.icon_only_toolbar is False
        assert dlg._icon_only_check.isChecked() is False

    def test_icon_only_toolbar_preselects_current_value(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT, current_icon_only=True)
        qtbot.addWidget(dlg)
        assert dlg._icon_only_check.isChecked() is True

    def test_accept_reads_icon_only_checkbox_state(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)
        dlg._icon_only_check.setChecked(True)
        dlg._on_accept()
        assert dlg.icon_only_toolbar is True

    def test_dark_theme_defaults_false(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)
        assert dlg.dark_theme is False
        assert dlg._dark_theme_check.isChecked() is False

    def test_dark_theme_preselects_current_value(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT, current_dark_theme=True)
        qtbot.addWidget(dlg)
        assert dlg._dark_theme_check.isChecked() is True

    def test_accept_reads_dark_theme_checkbox_state(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)
        dlg._dark_theme_check.setChecked(True)
        dlg._on_accept()
        assert dlg.dark_theme is True

    def test_overlap_strategy_defaults_reject(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)
        assert dlg.overlap_strategy == "reject"
        assert dlg._overlap_combo.currentData() == "reject"

    def test_overlap_strategy_preselects_current_value(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT, current_overlap_strategy="truncate")
        qtbot.addWidget(dlg)
        assert dlg._overlap_combo.currentData() == "truncate"

    def test_accept_reads_overlap_combo_state(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)
        truncate_index = dlg._overlap_combo.findData("truncate")
        dlg._overlap_combo.setCurrentIndex(truncate_index)
        dlg._on_accept()
        assert dlg.overlap_strategy == "truncate"

    def test_combo_preselects_current_scheme(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.FOLK_WARD_1957)
        qtbot.addWidget(dlg)
        assert dlg._combo.currentData() == ClassificationScheme.FOLK_WARD_1957

    def test_combo_contains_all_schemes(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)
        combo_data = {dlg._combo.itemData(i) for i in range(dlg._combo.count())}
        assert combo_data == set(ClassificationScheme)


class TestPreferencesDialogAccept:
    """Accepting the dialog commits the selected scheme; cancelling discards it."""

    def test_accept_updates_selected_scheme(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)

        # Switch combo selection to a different scheme.
        idx = dlg._combo.findData(ClassificationScheme.USDA)
        dlg._combo.setCurrentIndex(idx)

        accepted = []
        dlg.accepted.connect(lambda: accepted.append(True))
        dlg._on_accept()

        assert dlg.selected_scheme == ClassificationScheme.USDA
        assert accepted == [True]
        assert dlg.result() == QDialog.DialogCode.Accepted

    def test_ok_button_click_accepts_with_new_scheme(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)

        idx = dlg._combo.findData(ClassificationScheme.FOLK_WARD_1957)
        dlg._combo.setCurrentIndex(idx)

        box = dlg.findChild(QDialogButtonBox)
        ok_button = box.button(QDialogButtonBox.StandardButton.Ok)
        qtbot.mouseClick(ok_button, Qt.MouseButton.LeftButton)

        assert dlg.selected_scheme == ClassificationScheme.FOLK_WARD_1957
        assert dlg.result() == QDialog.DialogCode.Accepted

    def test_cancel_button_rejects_without_changing_scheme(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)

        idx = dlg._combo.findData(ClassificationScheme.USDA)
        dlg._combo.setCurrentIndex(idx)

        rejected = []
        dlg.rejected.connect(lambda: rejected.append(True))

        box = dlg.findChild(QDialogButtonBox)
        cancel_button = box.button(QDialogButtonBox.StandardButton.Cancel)
        qtbot.mouseClick(cancel_button, Qt.MouseButton.LeftButton)

        # selected_scheme is only updated on accept, not on reject.
        assert dlg.selected_scheme == ClassificationScheme.GRADISTAT
        assert rejected == [True]
        assert dlg.result() == QDialog.DialogCode.Rejected


class TestPreferencesDialogReset:
    """The Reset button (and _on_reset()) sets reset_requested and accepts the dialog."""

    def test_reset_button_sets_flag_and_accepts(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)

        reset_button = dlg.findChild(QPushButton)
        assert reset_button is not None
        assert "Reset" in reset_button.text()

        accepted = []
        dlg.accepted.connect(lambda: accepted.append(True))

        qtbot.mouseClick(reset_button, Qt.MouseButton.LeftButton)

        assert dlg.reset_requested is True
        assert accepted == [True]
        assert dlg.result() == QDialog.DialogCode.Accepted

    def test_on_reset_directly(self, qtbot):
        dlg = PreferencesDialog(ClassificationScheme.GRADISTAT)
        qtbot.addWidget(dlg)
        assert dlg.reset_requested is False
        dlg._on_reset()
        assert dlg.reset_requested is True
