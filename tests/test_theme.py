"""Tests for sedstat.gui.theme."""

from __future__ import annotations

from sedstat.gui.theme import apply_theme


class TestApplyTheme:
    """apply_theme() sets the Fusion style and switches the palette between dark/light."""

    def test_sets_fusion_style(self, qtbot, qapp):
        apply_theme(dark=False)
        assert qapp.style().objectName().lower() == "fusion"

    def test_dark_and_light_produce_different_window_colors(self, qtbot, qapp):
        apply_theme(dark=False)
        light_window = qapp.palette().color(qapp.palette().ColorRole.Window)

        apply_theme(dark=True)
        dark_window = qapp.palette().color(qapp.palette().ColorRole.Window)

        assert dark_window != light_window

    def test_dark_palette_matches_documented_recipe(self, qtbot, qapp):
        apply_theme(dark=True)
        window = qapp.palette().color(qapp.palette().ColorRole.Window)
        assert window.getRgb()[:3] == (53, 53, 53)

    def test_toggling_back_to_light_restores_fusion_default(self, qtbot, qapp):
        """Switching dark -> light restores the original light palette.

        qapp is session-scoped (shared across tests), so establish a
        known light baseline explicitly rather than trusting whatever
        palette happens to already be applied when this test runs.
        """
        apply_theme(dark=False)
        light_window = qapp.palette().color(qapp.palette().ColorRole.Window)

        apply_theme(dark=True)
        apply_theme(dark=False)
        restored_window = qapp.palette().color(qapp.palette().ColorRole.Window)

        assert restored_window == light_window

    def test_noop_without_a_qapplication(self, monkeypatch):
        """apply_theme() does not raise when no QApplication instance exists.

        Theming is cosmetic -- must never be able to crash a caller that
        (for whatever reason) has no QApplication yet.
        """
        import sedstat.gui.theme as theme_mod

        monkeypatch.setattr(theme_mod.QApplication, "instance", staticmethod(lambda: None))
        apply_theme(dark=True)  # should not raise
