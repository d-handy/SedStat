"""Tests for sedstat.io.beckman_coulter."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from sedstat.io.beckman_coulter import LSRecord, read_ls_file

# ---------------------------------------------------------------------------
# Helpers — minimal synthetic .$av content
# ---------------------------------------------------------------------------


def _make_av_content(
    *,
    n_bins: int = 5,
    include_sizestats: bool = True,
    include_sisave: bool = True,
    include_obs: bool = True,
) -> str:
    """Return a minimal .$av file string.

    Args:
        n_bins: Number of size classes (bin boundaries and heights are
            truncated to this many entries).
        include_sizestats: Whether to emit a [SizeStats] section.
        include_sisave: Whether to emit a [SIsave0] section.
        include_obs: Whether to emit [SizeN] Obs= blocks.

    Returns:
        The synthesized .$av file content.
    """
    boundaries = "\n".join(f"{v:.4f}" for v in [2.0, 4.0, 8.0, 16.0, 32.0, 64.0][: n_bins + 1])
    heights = "\n".join(f"{v:.4f}" for v in [5.0, 20.0, 50.0, 20.0, 5.0][:n_bins])

    sizestats = ""
    if include_sizestats:
        sizestats = (
            "[SizeStats]\n"
            "Mean= 18.5\n"
            "Mode= 16.0\n"
            "Median= 17.2\n"
            "SD= 2.4\n"
            "FWMean= 6.3\n"
            "FWMedian= 6.1\n"
            "FWSD= 1.8\n"
            "FWSkew= 0.05\n"
            "FWKurt= 1.1\n"
        )

    sisave = ""
    if include_sisave:
        sisave = "[SIsave0]\nGroupID=TestGroup\nDensity=2.65\nOperator=JD\n"

    obs_blocks = ""
    if include_obs:
        obs_blocks = "[Size0]\nObs= 12.5\n[Size1]\nObs= 11.8\n[Size2]\nObs= 13.1\n"

    return (
        "[common]\n"
        "Program=Beckman Coulter LS\n"
        "\n" + sisave + "\n"
        "[#Bindiam]\n" + boundaries + "\n\n"
        "[#Binheight]\n" + heights + "\n\n" + sizestats + obs_blocks
    )


def _write_temp_av(content: str) -> Path:
    with tempfile.NamedTemporaryFile(
        suffix=".$av", delete=False, mode="w", encoding="latin-1"
    ) as tmp:
        tmp.write(content)
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestReadLsFileSmoke:
    """read_ls_file() returns an LSRecord and accepts both Path and str paths."""

    def test_returns_ls_record(self):
        p = _write_temp_av(_make_av_content())
        rec = read_ls_file(p)
        assert isinstance(rec, LSRecord)

    def test_path_stored(self):
        p = _write_temp_av(_make_av_content())
        rec = read_ls_file(p)
        assert rec.path == p

    def test_accepts_string_path(self):
        p = _write_temp_av(_make_av_content())
        rec = read_ls_file(str(p))
        assert isinstance(rec, LSRecord)


# ---------------------------------------------------------------------------
# Class boundaries and values
# ---------------------------------------------------------------------------


class TestBoundariesAndValues:
    """read_ls_file() parses class boundaries and weight values from [#Bindiam]/[#Binheight].

    Attributes:
        rec: The LSRecord parsed from a 5-bin synthetic .$av file, from setup_method.
    """

    def setup_method(self):
        p = _write_temp_av(_make_av_content(n_bins=5))
        self.rec = read_ls_file(p)

    def test_n_boundaries(self):
        assert len(self.rec.classes_um) == 6

    def test_n_values(self):
        assert len(self.rec.values) == 5

    def test_first_boundary(self):
        assert abs(self.rec.classes_um[0] - 2.0) < 1e-6

    def test_values_sum_near_100(self):
        assert abs(sum(self.rec.values) - 100.0) < 0.1


# ---------------------------------------------------------------------------
# Sample metadata
# ---------------------------------------------------------------------------


class TestSampleMetadata:
    """read_ls_file() parses group_id, density, and operator from [SIsave0].

    Attributes:
        rec: The LSRecord parsed from a synthetic .$av file with [SIsave0], from setup_method.
    """

    def setup_method(self):
        p = _write_temp_av(_make_av_content(include_sisave=True))
        self.rec = read_ls_file(p)

    def test_group_id(self):
        assert self.rec.group_id == "TestGroup"

    def test_density(self):
        assert abs(self.rec.density - 2.65) < 1e-6

    def test_operator(self):
        assert self.rec.operator == "JD"

    def test_other_fields_in_extra_meta(self):
        content = (
            "[#Bindiam]\n2.0\n4.0\n8.0\n[#Binheight]\n40.0\n60.0\n"
            "[SIsave0]\nGroupID=G1\nSampleID=X7\n"
        )
        rec = read_ls_file(_write_temp_av(content))
        assert rec.extra_meta["SIsave0.SampleID"] == "X7"
        assert "SIsave0.GroupID" not in rec.extra_meta


class TestMissingSIsave:
    """Sample metadata is None when the [SIsave0] section is absent."""

    def test_metadata_none_when_section_absent(self):
        p = _write_temp_av(_make_av_content(include_sisave=False))
        rec = read_ls_file(p)
        assert rec.group_id is None
        assert rec.density is None
        assert rec.operator is None


# ---------------------------------------------------------------------------
# Device statistics
# ---------------------------------------------------------------------------


class TestDeviceStats:
    """read_ls_file() parses the device's own reported statistics from [SizeStats].

    Attributes:
        rec: The LSRecord parsed from a synthetic .$av file with [SizeStats], from setup_method.
    """

    def setup_method(self):
        p = _write_temp_av(_make_av_content(include_sizestats=True))
        self.rec = read_ls_file(p)

    def test_mean_parsed(self):
        assert abs(self.rec.device_mean_um - 18.5) < 1e-6

    def test_fw_mean_parsed(self):
        assert abs(self.rec.device_fw_mean_um - 6.3) < 1e-6

    def test_fw_std_parsed(self):
        assert abs(self.rec.device_fw_std - 1.8) < 1e-6


class TestMissingSizeStats:
    """Device statistics are None when the [SizeStats] section is absent."""

    def test_stats_none_when_section_absent(self):
        p = _write_temp_av(_make_av_content(include_sizestats=False))
        rec = read_ls_file(p)
        assert rec.device_mean_um is None
        assert rec.device_fw_mean_um is None


# ---------------------------------------------------------------------------
# Concentration / obscuration
# ---------------------------------------------------------------------------


class TestConcentration:
    """concentration is averaged over [SizeN] Obs= blocks, or None when absent."""

    def test_mean_of_three_obs(self):
        p = _write_temp_av(_make_av_content(include_obs=True))
        rec = read_ls_file(p)
        # Obs values: 12.5, 11.8, 13.1 → mean ≈ 12.467
        assert rec.concentration is not None
        assert abs(rec.concentration - (12.5 + 11.8 + 13.1) / 3.0) < 1e-6

    def test_concentration_none_when_no_obs(self):
        p = _write_temp_av(_make_av_content(include_obs=False))
        rec = read_ls_file(p)
        assert rec.concentration is None


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrors:
    """read_ls_file() raises for a missing file or malformed/non-monotonic bin data."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_ls_file("/nonexistent/path/sample.$av")

    def test_missing_bindiam_raises(self):
        content = "[#Binheight]\n10.0\n20.0\n30.0\n40.0\n"
        p = _write_temp_av(content)
        with pytest.raises(ValueError, match="Bindiam"):
            read_ls_file(p)

    def test_missing_binheight_raises(self):
        content = "[#Bindiam]\n2.0\n4.0\n8.0\n"
        p = _write_temp_av(content)
        with pytest.raises(ValueError, match="Binheight"):
            read_ls_file(p)

    def test_non_monotonic_boundaries_raises(self):
        # Boundaries must be strictly increasing; 16.0 < 8.0 is invalid
        content = "[#Bindiam]\n2.0\n4.0\n16.0\n8.0\n32.0\n[#Binheight]\n10\n20\n30\n40\n"
        p = _write_temp_av(content)
        with pytest.raises(ValueError, match="strictly increasing"):
            read_ls_file(p)

    def test_directory_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_ls_file(tmp_path)

    def test_decimal_comma_in_bindiam_raises(self):
        """A dropped boundary would shift every later bin, so it must raise."""
        content = "[#Bindiam]\n2.0\n4.0\n8,0\n16.0\n[#Binheight]\n10\n20\n70\n"
        p = _write_temp_av(content)
        with pytest.raises(ValueError, match=r"'8,0' in \[#Bindiam\] at line 4"):
            read_ls_file(p)

    def test_non_numeric_binheight_raises(self):
        content = "[#Bindiam]\n2.0\n4.0\n8.0\n[#Binheight]\n40\nabc\n"
        p = _write_temp_av(content)
        with pytest.raises(ValueError, match=r"'abc' in \[#Binheight\]"):
            read_ls_file(p)

    def test_nan_binheight_raises(self):
        content = "[#Bindiam]\n2.0\n4.0\n8.0\n[#Binheight]\n40\nnan\n"
        p = _write_temp_av(content)
        with pytest.raises(ValueError, match="'nan'"):
            read_ls_file(p)

    def test_blank_lines_in_bin_sections_ignored(self):
        content = "[#Bindiam]\n2.0\n\n4.0\n8.0\n\n[#Binheight]\n\n40\n60\n"
        rec = read_ls_file(_write_temp_av(content))
        assert rec.classes_um == [2.0, 4.0, 8.0]
        assert rec.values == [40.0, 60.0]

    def test_length_mismatch_raises(self):
        content = "[#Bindiam]\n2.0\n4.0\n8.0\n16.0\n32.0\n64.0\n[#Binheight]\n30\n30\n40\n"
        p = _write_temp_av(content)
        with pytest.raises(ValueError, match=r"6 values and .* has 3"):
            read_ls_file(p)


# ---------------------------------------------------------------------------
# Integration with compute_all
# ---------------------------------------------------------------------------


class TestIntegrationWithComputeAll:
    """An LSRecord's boundaries/values feed directly into compute_all()."""

    def test_record_feeds_compute_all(self):
        from sedstat.core.results import GrainSizeResult
        from sedstat.core.statistics import compute_all

        p = _write_temp_av(_make_av_content(n_bins=5))
        rec = read_ls_file(p)
        result = compute_all(rec.classes_um, rec.values)
        assert isinstance(result, GrainSizeResult)
        assert np.isfinite(result.folk_ward_log.mean)


# ---------------------------------------------------------------------------
# Malformed-value branches in _handle_size_stats / _handle_size_run
# ---------------------------------------------------------------------------


class TestMalformedSizeStats:
    """Covers the blank-line, unknown-key, and bad-float branches of
    _handle_size_stats (lines 100, 103, 106-107).
    """

    def test_blank_line_ignored(self):
        content = (
            "[#Bindiam]\n2.0\n4.0\n8.0\n16.0\n32.0\n64.0\n"
            "[#Binheight]\n5.0\n20.0\n50.0\n20.0\n5.0\n"
            "[SizeStats]\n"
            "\n"  # blank line inside SizeStats -> kv is None -> return
            "Mean= 18.5\n"
        )
        p = _write_temp_av(content)
        rec = read_ls_file(p)
        assert abs(rec.device_mean_um - 18.5) < 1e-6

    def test_unknown_key_ignored(self):
        content = (
            "[#Bindiam]\n2.0\n4.0\n8.0\n16.0\n32.0\n64.0\n"
            "[#Binheight]\n5.0\n20.0\n50.0\n20.0\n5.0\n"
            "[SizeStats]\n"
            "Mean= 18.5\n"
            "NotAKnownStat= 99.0\n"  # key not in _STAT_KEYS -> skipped
        )
        p = _write_temp_av(content)
        rec = read_ls_file(p)
        assert abs(rec.device_mean_um - 18.5) < 1e-6
        assert not hasattr(rec, "not_a_known_stat")

    def test_bad_float_value_ignored(self):
        content = (
            "[#Bindiam]\n2.0\n4.0\n8.0\n16.0\n32.0\n64.0\n"
            "[#Binheight]\n5.0\n20.0\n50.0\n20.0\n5.0\n"
            "[SizeStats]\n"
            "Mean= not_a_number\n"  # float() raises ValueError -> skipped
            "Mode= 16.0\n"
        )
        p = _write_temp_av(content)
        rec = read_ls_file(p)
        assert rec.device_mean_um is None
        assert abs(rec.device_mode_um - 16.0) < 1e-6


class TestSizeRunBranches:
    """Covers the no-'=' and non-Obs-key branches of _handle_size_run
    (lines 119, 124).
    """

    def test_line_without_equals_ignored(self):
        content = (
            "[#Bindiam]\n2.0\n4.0\n8.0\n16.0\n32.0\n64.0\n"
            "[#Binheight]\n5.0\n20.0\n50.0\n20.0\n5.0\n"
            "[Size0]\n"
            "this line has no equals sign\n"
            "Obs= 12.5\n"
        )
        p = _write_temp_av(content)
        rec = read_ls_file(p)
        assert rec.concentration is not None
        assert abs(rec.concentration - 12.5) < 1e-6

    def test_non_obs_key_stored_in_extra_meta(self):
        content = (
            "[#Bindiam]\n2.0\n4.0\n8.0\n16.0\n32.0\n64.0\n"
            "[#Binheight]\n5.0\n20.0\n50.0\n20.0\n5.0\n"
            "[Size0]\n"
            "SomeField= hello\n"
            "Obs= 12.5\n"
        )
        p = _write_temp_av(content)
        rec = read_ls_file(p)
        assert rec.extra_meta.get("Size0.SomeField") == "hello"


# ---------------------------------------------------------------------------
# _resolve_num_runs / _resolve_density branches
# ---------------------------------------------------------------------------


class TestResolveNumRuns:
    """Covers lines 181-182 (found via extra_meta fallback loop), 185-188
    (successful int parse / ValueError -> None).
    """

    def test_num_runs_from_size_block_extra_meta(self):
        """num_runs falls back to scanning extra_meta when absent from [SIsave0]/common.

        No NumRuns in [SIsave0] or common.NumRuns -> falls back to
        scanning extra_meta for any key ending in ".NumRuns".
        """
        content = (
            "[#Bindiam]\n2.0\n4.0\n8.0\n16.0\n32.0\n64.0\n"
            "[#Binheight]\n5.0\n20.0\n50.0\n20.0\n5.0\n"
            "[SIsave0]\nGroupID=G1\n"
            "[Size0]\nNumRuns=3\n"
        )
        p = _write_temp_av(content)
        rec = read_ls_file(p)
        assert rec.num_runs == 3

    def test_num_runs_malformed_returns_none(self):
        content = (
            "[#Bindiam]\n2.0\n4.0\n8.0\n16.0\n32.0\n64.0\n"
            "[#Binheight]\n5.0\n20.0\n50.0\n20.0\n5.0\n"
            "[SIsave0]\nGroupID=G1\nNumRuns=not_an_int\n"
        )
        p = _write_temp_av(content)
        rec = read_ls_file(p)
        assert rec.num_runs is None

    def test_num_runs_from_sisave0(self):
        content = (
            "[#Bindiam]\n2.0\n4.0\n8.0\n16.0\n32.0\n64.0\n"
            "[#Binheight]\n5.0\n20.0\n50.0\n20.0\n5.0\n"
            "[SIsave0]\nGroupID=G1\nNumRuns=5\n"
        )
        p = _write_temp_av(content)
        rec = read_ls_file(p)
        assert rec.num_runs == 5


class TestResolveDensity:
    """Covers lines 196-197 (Density present but not a valid float)."""

    def test_malformed_density_returns_none(self):
        content = (
            "[#Bindiam]\n2.0\n4.0\n8.0\n16.0\n32.0\n64.0\n"
            "[#Binheight]\n5.0\n20.0\n50.0\n20.0\n5.0\n"
            "[SIsave0]\nGroupID=G1\nDensity=not_a_float\n"
        )
        p = _write_temp_av(content)
        rec = read_ls_file(p)
        assert rec.density is None
