"""Tests for sedstat.gui.session — no Qt/qtbot needed, plain functions."""

from __future__ import annotations

import json

import pytest
from sedstat.gui.session import (
    SessionError,
    SessionVersionError,
    build_session,
    load_session,
    save_session,
)


class TestBuildSession:
    """build_session() assembles the versioned session dict."""

    def test_shape(self):
        session = build_session(["a.$av", "b.$av"], {"a.$av": 1.5})
        assert session == {
            "version": 1,
            "files": ["a.$av", "b.$av"],
            "depths": {"a.$av": 1.5},
        }


class TestSaveLoadRoundTrip:
    """save_session()/load_session() round-trip files and depths, including None depths."""

    def test_round_trip(self, tmp_path):
        path = tmp_path / "session.sedstat.json"
        save_session(path, ["a.$av", "b.$av"], {"a.$av": 2.0, "b.$av": None})
        files, depths = load_session(path)
        assert files == ["a.$av", "b.$av"]
        assert depths == {"a.$av": 2.0, "b.$av": None}


class TestLoadSessionErrors:
    """load_session() raises SessionError/SessionVersionError for malformed session files."""

    def test_missing_file_raises_session_error(self, tmp_path):
        with pytest.raises(SessionError, match="Could not read"):
            load_session(tmp_path / "does_not_exist.json")

    def test_invalid_json_raises_session_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(SessionError, match="invalid JSON"):
            load_session(path)

    def test_non_object_json_raises_session_error(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(SessionError, match="JSON object"):
            load_session(path)

    def test_wrong_version_raises_session_version_error(self, tmp_path):
        """An unsupported session version raises SessionVersionError, not a plain SessionError.

        A distinct subclass — callers show a softer warning for this case
        ("probably a different SedStat version") than for a truly
        malformed file.
        """
        path = tmp_path / "v2.json"
        path.write_text(json.dumps({"version": 2, "files": [], "depths": {}}), encoding="utf-8")
        with pytest.raises(SessionVersionError, match="Unsupported session version"):
            load_session(path)
        # ... and it's still a SessionError, so a broad `except SessionError`
        # (if a caller wants to treat all failures the same way) still works.
        with pytest.raises(SessionError):
            load_session(path)

    def test_files_not_a_list_raises_session_error(self, tmp_path):
        path = tmp_path / "bad_files.json"
        path.write_text(
            json.dumps({"version": 1, "files": "a.$av", "depths": {}}), encoding="utf-8"
        )
        with pytest.raises(SessionError, match="'files'"):
            load_session(path)

    def test_files_with_non_string_entries_raises_session_error(self, tmp_path):
        path = tmp_path / "bad_files2.json"
        path.write_text(
            json.dumps({"version": 1, "files": ["a.$av", 42], "depths": {}}),
            encoding="utf-8",
        )
        with pytest.raises(SessionError, match="'files'"):
            load_session(path)

    def test_depths_not_a_dict_raises_session_error(self, tmp_path):
        path = tmp_path / "bad_depths.json"
        path.write_text(json.dumps({"version": 1, "files": [], "depths": [1, 2]}), encoding="utf-8")
        with pytest.raises(SessionError, match="'depths'"):
            load_session(path)

    def test_depths_with_non_numeric_value_raises_session_error(self, tmp_path):
        path = tmp_path / "bad_depths2.json"
        path.write_text(
            json.dumps({"version": 1, "files": [], "depths": {"a.$av": "deep"}}),
            encoding="utf-8",
        )
        with pytest.raises(SessionError, match="'depths'"):
            load_session(path)

    def test_null_depth_is_valid(self, tmp_path):
        path = tmp_path / "null_depth.json"
        path.write_text(
            json.dumps({"version": 1, "files": [], "depths": {"a.$av": None}}),
            encoding="utf-8",
        )
        _files, depths = load_session(path)
        assert depths == {"a.$av": None}
