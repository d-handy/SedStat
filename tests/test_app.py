"""Tests for sedstat.gui.app.main().

main() constructs a real QApplication and calls app.exec(), which would
block indefinitely under pytest. We patch QApplication and MainWindow so
the entry point's wiring (argv handling, app metadata, window show, and
exit code propagation) can be exercised without a real event loop.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sedstat.gui import app as app_module


class TestMain:
    """main()'s argv/app wiring, missing-dependency handling, and exit code propagation."""

    def test_missing_pyside6_raises_clear_systemexit(self):
        """Missing PySide6 raises SystemExit with a message pointing at the gui extra.

        Setting sys.modules[name] = None makes the next `import name` (or
        `from name import ...`) raise ImportError immediately — simulates
        a bare `pip install sedstat` (no `gui` extra) without needing
        PySide6 to actually be absent from this test environment.
        """
        with (
            patch.dict("sys.modules", {"PySide6.QtWidgets": None}),
            pytest.raises(SystemExit, match=r"gui.*extra"),
        ):
            app_module.main()

    def test_creates_application_with_argv(self):
        with (
            patch("PySide6.QtWidgets.QApplication") as mock_qapp_cls,
            patch("sedstat.gui.main_window.MainWindow"),
            patch("sys.argv", ["sedstat"]),
            patch("sys.exit"),
        ):
            mock_app_instance = MagicMock()
            mock_app_instance.exec.return_value = 0
            mock_qapp_cls.return_value = mock_app_instance

            app_module.main()

            mock_qapp_cls.assert_called_once_with(["sedstat"])

    def test_sets_application_metadata(self):
        with (
            patch("PySide6.QtWidgets.QApplication") as mock_qapp_cls,
            patch("sedstat.gui.main_window.MainWindow"),
            patch("sys.exit"),
        ):
            mock_app_instance = MagicMock()
            mock_app_instance.exec.return_value = 0
            mock_qapp_cls.return_value = mock_app_instance

            app_module.main()

            mock_app_instance.setApplicationName.assert_called_once_with("SedStat")
            mock_app_instance.setOrganizationName.assert_called_once_with("Uni Köln")

    def test_shows_main_window(self):
        with (
            patch("PySide6.QtWidgets.QApplication") as mock_qapp_cls,
            patch("sedstat.gui.main_window.MainWindow") as mock_window_cls,
            patch("sys.exit"),
        ):
            mock_app_instance = MagicMock()
            mock_app_instance.exec.return_value = 0
            mock_qapp_cls.return_value = mock_app_instance
            mock_window_instance = MagicMock()
            mock_window_cls.return_value = mock_window_instance

            app_module.main()

            mock_window_cls.assert_called_once_with()
            mock_window_instance.show.assert_called_once()

    def test_exits_with_app_exec_return_code(self):
        with (
            patch("PySide6.QtWidgets.QApplication") as mock_qapp_cls,
            patch("sedstat.gui.main_window.MainWindow"),
            patch("sys.exit") as mock_exit,
        ):
            mock_app_instance = MagicMock()
            mock_app_instance.exec.return_value = 7
            mock_qapp_cls.return_value = mock_app_instance

            app_module.main()

            mock_app_instance.exec.assert_called_once()
            mock_exit.assert_called_once_with(7)
