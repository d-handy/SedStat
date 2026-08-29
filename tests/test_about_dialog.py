"""Tests for sedstat.gui.about_dialog.AboutDialog."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel
from sedstat.gui.about_dialog import AboutDialog, _pkg_version


class TestAboutDialogConstruction:
    """AboutDialog's basic Qt properties: type, title, minimum width, and parenting."""

    def test_is_qdialog(self, qtbot):
        dlg = AboutDialog()
        qtbot.addWidget(dlg)
        assert isinstance(dlg, QDialog)

    def test_window_title(self, qtbot):
        dlg = AboutDialog()
        qtbot.addWidget(dlg)
        assert dlg.windowTitle() == "About SedStat"

    def test_minimum_width(self, qtbot):
        dlg = AboutDialog()
        qtbot.addWidget(dlg)
        assert dlg.minimumWidth() == 480

    def test_accepts_parent(self, qtbot):
        parent = QDialog()
        qtbot.addWidget(parent)
        dlg = AboutDialog(parent=parent)
        qtbot.addWidget(dlg)
        assert dlg.parent() is parent


class TestAboutDialogContent:
    """AboutDialog shows the title, Python version, citation, and OK button."""

    def test_title_label_present(self, qtbot):
        dlg = AboutDialog()
        qtbot.addWidget(dlg)
        labels = dlg.findChildren(QLabel)
        texts = [lbl.text() for lbl in labels]
        assert any("SedStat" in t for t in texts)

    def test_version_label_contains_python_version(self, qtbot):
        dlg = AboutDialog()
        qtbot.addWidget(dlg)
        labels = dlg.findChildren(QLabel)
        texts = [lbl.text() for lbl in labels]
        py_ver = sys.version.split()[0]
        assert any(py_ver in t for t in texts)

    def test_citation_label_present_and_word_wrapped(self, qtbot):
        dlg = AboutDialog()
        qtbot.addWidget(dlg)
        labels = dlg.findChildren(QLabel)
        citation_labels = [lbl for lbl in labels if "Blott" in lbl.text()]
        assert len(citation_labels) == 1
        assert citation_labels[0].wordWrap() is True
        assert citation_labels[0].openExternalLinks() is True

    def test_ok_button_present(self, qtbot):
        dlg = AboutDialog()
        qtbot.addWidget(dlg)
        box = dlg.findChild(QDialogButtonBox)
        assert box is not None
        ok_button = box.button(QDialogButtonBox.StandardButton.Ok)
        assert ok_button is not None


class TestAboutDialogInteraction:
    """Clicking OK accepts the dialog and emits the accepted signal."""

    def test_ok_button_accepts_dialog(self, qtbot):
        dlg = AboutDialog()
        qtbot.addWidget(dlg)
        box = dlg.findChild(QDialogButtonBox)
        ok_button = box.button(QDialogButtonBox.StandardButton.Ok)

        accepted = []
        dlg.accepted.connect(lambda: accepted.append(True))

        qtbot.mouseClick(ok_button, Qt.MouseButton.LeftButton)

        assert accepted == [True]
        assert dlg.result() == QDialog.DialogCode.Accepted


class TestPkgVersionHelper:
    """_pkg_version() resolves installed package versions and falls back to "unknown"."""

    def test_known_package_returns_string(self):
        result = _pkg_version("PySide6")
        assert isinstance(result, str)
        assert result != "unknown"

    def test_unknown_package_returns_unknown(self):
        result = _pkg_version("this-package-definitely-does-not-exist-xyz")
        assert result == "unknown"
