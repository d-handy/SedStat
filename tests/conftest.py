"""Shared pytest configuration — sets Qt to headless offscreen mode."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
