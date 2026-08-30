"""Tests for sedstat.io.sieve_pipette."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sedstat.io.sieve_pipette import (
    PipetteRow,
    SievePipetteRecord,
    SieveRow,
    _build_record,
    combine_sieve_pipette,
    pipette_to_distribution,
    read_sieve_pipette_csv,
    sieve_to_distribution,
    write_sieve_pipette_csv,
)

from sedstat.core.statistics import compute_all
from sedstat.core.stokes import stokes_settling_diameter_um


def _write_temp_csv(content: str) -> Path:
    with tempfile.NamedTemporaryFile(
        suffix=".csv", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(content)
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# sieve_to_distribution
# ---------------------------------------------------------------------------


class TestSieveToDistribution:
    """sieve_to_distribution() converts sieve rows into a normalized class distribution."""

    def test_worked_example(self):
        rows = [
            SieveRow(mesh_um=0, weight_g=5.0),
            SieveRow(mesh_um=250, weight_g=60.0),
            SieveRow(mesh_um=500, weight_g=25.0),
            SieveRow(mesh_um=1000, weight_g=8.0),
            SieveRow(mesh_um=2000, weight_g=2.0),  # oversize, excluded from fit
        ]
        classes_um, values = sieve_to_distribution(rows)
        assert classes_um == [250.0, 500.0, 1000.0, 2000.0]
        # Weights are shifted (pan -> 250, 250's own weight -> 500, ...) and
        # normalized to sum 100 over the fitted (oversize-excluded) total.
        assert sum(values) == pytest.approx(100.0)
        expected_ratios = [5.0, 60.0, 25.0, 8.0]
        total = sum(expected_ratios)
        assert values == [pytest.approx(v / total * 100.0) for v in expected_ratios]

    def test_unsorted_input_is_sorted_internally(self):
        rows = [
            SieveRow(mesh_um=500, weight_g=25.0),
            SieveRow(mesh_um=0, weight_g=5.0),
            SieveRow(mesh_um=250, weight_g=60.0),
        ]
        classes_um, values = sieve_to_distribution(rows)
        assert classes_um == [250.0, 500.0]
        assert sum(values) == pytest.approx(100.0)
        assert values[0] / values[1] == pytest.approx(5.0 / 60.0)

    def test_integrates_with_compute_all(self):
        rows = [
            SieveRow(mesh_um=0, weight_g=5.0),
            SieveRow(mesh_um=63, weight_g=10.0),
            SieveRow(mesh_um=250, weight_g=60.0),
            SieveRow(mesh_um=1000, weight_g=25.0),
        ]
        classes_um, values = sieve_to_distribution(rows)
        result = compute_all(classes_um, values)
        assert np.isfinite(result.folk_ward_log.mean)


class TestSieveToDistributionValidation:
    """sieve_to_distribution() validates pan presence, mesh uniqueness, and weight sums."""

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            sieve_to_distribution([])

    def test_missing_pan_raises(self):
        rows = [
            SieveRow(mesh_um=250, weight_g=60.0),
            SieveRow(mesh_um=500, weight_g=25.0),
        ]
        with pytest.raises(ValueError, match="pan"):
            sieve_to_distribution(rows)

    def test_duplicate_mesh_raises(self):
        rows = [
            SieveRow(mesh_um=0, weight_g=5.0),
            SieveRow(mesh_um=250, weight_g=60.0),
            SieveRow(mesh_um=250, weight_g=10.0),
        ]
        with pytest.raises(ValueError, match="duplicate"):
            sieve_to_distribution(rows)

    def test_single_sieve_plus_pan_is_valid_degenerate_case(self):
        rows = [SieveRow(mesh_um=0, weight_g=5.0), SieveRow(mesh_um=250, weight_g=60.0)]
        classes_um, values = sieve_to_distribution(rows)
        assert classes_um == [250.0]
        assert values == [pytest.approx(100.0)]

    def test_pan_only_raises(self):
        rows = [SieveRow(mesh_um=0, weight_g=5.0)]
        with pytest.raises(ValueError, match="at least one real sieve mesh"):
            sieve_to_distribution(rows)

    def test_all_zero_weights_raises(self):
        rows = [SieveRow(mesh_um=0, weight_g=0.0), SieveRow(mesh_um=250, weight_g=0.0)]
        with pytest.raises(ValueError, match="sum to zero"):
            sieve_to_distribution(rows)

    def test_explicit_is_pan_flag_is_honored(self):
        """is_pan=True is authoritative, independent of mesh_um == 0.

        is_pan=True is the authoritative signal — works even without
        relying on mesh_um == 0 (though mesh_um must still be 0 to sort
        first; see the mismatch test below for what happens otherwise).
        """
        rows = [
            SieveRow(mesh_um=0, weight_g=5.0, is_pan=True),
            SieveRow(mesh_um=250, weight_g=60.0),
        ]
        classes_um, _values = sieve_to_distribution(rows)
        assert classes_um == [250.0]

    def test_multiple_is_pan_flags_raises(self):
        rows = [
            SieveRow(mesh_um=0, weight_g=5.0, is_pan=True),
            SieveRow(mesh_um=10, weight_g=1.0, is_pan=True),
            SieveRow(mesh_um=250, weight_g=60.0),
        ]
        with pytest.raises(ValueError, match="more than one"):
            sieve_to_distribution(rows)

    def test_is_pan_flag_on_non_smallest_mesh_raises(self):
        """A pan-flagged row whose mesh_um doesn't sort first is a genuine inconsistency.

        A row flagged is_pan=True but with a nonzero mesh_um that doesn't
        sort first is a genuine inconsistency — must raise, not silently
        treat whichever row happens to sort first as the pan.
        """
        rows = [
            SieveRow(mesh_um=0, weight_g=5.0),  # unflagged, but sorts first
            SieveRow(mesh_um=500, weight_g=1.0, is_pan=True),  # flagged, wrong mesh
            SieveRow(mesh_um=250, weight_g=60.0),
        ]
        with pytest.raises(ValueError, match="must have the smallest mesh_um"):
            sieve_to_distribution(rows)


# ---------------------------------------------------------------------------
# pipette_to_distribution
# ---------------------------------------------------------------------------


class TestPipetteToDistribution:
    """pipette_to_distribution() resolves direct-size and raw-Stokes rows into a distribution."""

    def test_direct_mode_worked_example(self):
        rows = [
            PipetteRow(weight_g=2.1, size_um=31),
            PipetteRow(weight_g=1.8, size_um=16),
        ]
        classes_um, values = pipette_to_distribution(rows)
        assert classes_um == [16.0, 31.0]
        assert sum(values) == pytest.approx(100.0)
        assert values[0] / values[1] == pytest.approx(1.8 / 2.1)

    def test_raw_mode_matches_direct_stokes_call(self):
        rows = [PipetteRow(weight_g=1.0, time_s=3600, depth_cm=10, temp_c=20)]
        classes_um, _ = pipette_to_distribution(rows)
        expected = stokes_settling_diameter_um(3600, 10, 20)
        assert classes_um[0] == pytest.approx(expected)

    def test_raw_mode_uses_custom_density(self):
        rows = [PipetteRow(weight_g=1.0, time_s=3600, depth_cm=10, temp_c=20, density_g_cm3=2.2)]
        classes_um, _ = pipette_to_distribution(rows)
        expected = stokes_settling_diameter_um(3600, 10, 20, particle_density_g_cm3=2.2)
        assert classes_um[0] == pytest.approx(expected)

    def test_mixed_direct_and_raw_rows(self):
        rows = [
            PipetteRow(weight_g=1.0, size_um=8.0),
            PipetteRow(weight_g=1.0, time_s=3600, depth_cm=10, temp_c=20),
        ]
        classes_um, _values = pipette_to_distribution(rows)
        assert len(classes_um) == 2
        assert classes_um == sorted(classes_um)

    def test_integrates_with_compute_all(self):
        rows = [
            PipetteRow(weight_g=2.1, size_um=31),
            PipetteRow(weight_g=1.8, size_um=16),
            PipetteRow(weight_g=1.0, size_um=8),
            PipetteRow(weight_g=0.5, size_um=2),
        ]
        classes_um, values = pipette_to_distribution(rows)
        result = compute_all(classes_um, values)
        assert np.isfinite(result.folk_ward_log.mean)


class TestPipetteRowValidation:
    """pipette_to_distribution() validates each row's direct/raw input combination."""

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            pipette_to_distribution([])

    def test_neither_diameter_nor_raw_inputs_raises(self):
        rows = [PipetteRow(weight_g=1.0)]
        with pytest.raises(ValueError, match="size_um, or time_s"):
            pipette_to_distribution(rows)

    def test_both_diameter_and_raw_inputs_raises(self):
        rows = [PipetteRow(weight_g=1.0, size_um=16, time_s=3600, depth_cm=10, temp_c=20)]
        with pytest.raises(ValueError, match="ambiguous"):
            pipette_to_distribution(rows)

    def test_partial_raw_inputs_raises(self):
        rows = [PipetteRow(weight_g=1.0, time_s=3600, depth_cm=10)]  # temp_c missing
        with pytest.raises(ValueError, match="size_um, or time_s"):
            pipette_to_distribution(rows)

    def test_duplicate_resolved_diameter_raises(self):
        rows = [
            PipetteRow(weight_g=1.0, size_um=16),
            PipetteRow(weight_g=1.0, size_um=16),
        ]
        with pytest.raises(ValueError, match="duplicate"):
            pipette_to_distribution(rows)


# ---------------------------------------------------------------------------
# combine_sieve_pipette
# ---------------------------------------------------------------------------


class TestCombineSievePipette:
    """combine_sieve_pipette() merges sieve and pipette distributions, preserving ratios."""

    def test_worked_example_preserves_ratios_and_normalizes(self):
        sieve_classes = [250.0, 500.0, 1000.0, 2000.0]
        sieve_values = [5.0, 60.0, 25.0, 8.0]  # pan = 5.0
        pipette_classes = [16.0, 31.0]
        pipette_values = [1.8, 2.1]  # sub-sample of the pan

        classes_um, values = combine_sieve_pipette(
            sieve_classes, sieve_values, pipette_classes, pipette_values
        )

        assert classes_um == [16.0, 31.0, 500.0, 1000.0, 2000.0]
        assert sum(values) == pytest.approx(100.0)
        # Pipette sub-fractions keep their relative ratio to each other.
        assert values[0] / values[1] == pytest.approx(1.8 / 2.1)
        # The pipette's total (rescaled pan) keeps the same ratio to the
        # 500 µm class as the original sieve pan did.
        pipette_total = values[0] + values[1]
        assert pipette_total / values[2] == pytest.approx(5.0 / 60.0)

    def test_integrates_with_compute_all(self):
        classes_um, values = combine_sieve_pipette(
            [250.0, 500.0, 1000.0], [5.0, 60.0, 25.0], [16.0, 31.0], [1.8, 2.1]
        )
        result = compute_all(classes_um, values)
        assert np.isfinite(result.folk_ward_log.mean)

    def test_overlap_defaults_to_reject(self):
        """overlap_strategy defaults to "reject", preserving pre-truncate-option behavior.

        overlap_strategy is a methodological choice, not something to
        default silently to whichever behavior happened to get
        implemented -- "reject" preserves this function's original
        (pre-truncate-option) behavior unless a caller opts in to
        "truncate" explicitly.
        """
        with pytest.raises(ValueError, match="overlaps"):
            combine_sieve_pipette([250.0, 500.0], [5.0, 60.0], [16.0, 300.0], [1.8, 2.1])

    def test_reject_strategy_error_mentions_truncate_alternative(self):
        with pytest.raises(ValueError, match="truncate"):
            combine_sieve_pipette(
                [250.0, 500.0],
                [5.0, 60.0],
                [16.0, 300.0],
                [1.8, 2.1],
                overlap_strategy="reject",
            )

    def test_partial_overlap_is_truncated_and_mass_preserved(self):
        """An overlapping pipette class is truncated and its mass folded into the retained class.

        300.0 exceeds the sieve's finest mesh (250.0) -- would
        double-count material the sieve already measured. With
        overlap_strategy="truncate": dropped, its weight (2.1) folded
        into the coarsest *retained* pipette class (16.0's own 1.8 g)
        instead of silently discarded, so total sample mass is
        unaffected.
        """
        without_overlap_classes, without_overlap_values = combine_sieve_pipette(
            [250.0, 500.0], [5.0, 60.0], [16.0], [1.8 + 2.1]
        )
        classes_um, values = combine_sieve_pipette(
            [250.0, 500.0],
            [5.0, 60.0],
            [16.0, 300.0],
            [1.8, 2.1],
            overlap_strategy="truncate",
        )
        assert classes_um == [16.0, 500.0] == without_overlap_classes
        assert values == pytest.approx(without_overlap_values)
        assert sum(values) == pytest.approx(100.0)

    def test_truncation_handles_unsorted_input(self):
        """Truncation identifies the coarsest retained class correctly even for unsorted input.

        combine_sieve_pipette doesn't require pre-sorted pipette
        classes (pipette_to_distribution's own output always is, but
        this is public API) -- passed out of order, the coarsest
        *retained* class must still be identified correctly, not just
        whichever happens to be last in the list.
        """
        sorted_classes, sorted_values = combine_sieve_pipette(
            [250.0, 500.0],
            [5.0, 60.0],
            [8.0, 16.0, 300.0],
            [0.5, 1.8, 2.1],
            overlap_strategy="truncate",
        )
        unsorted_classes, unsorted_values = combine_sieve_pipette(
            [250.0, 500.0],
            [5.0, 60.0],
            [300.0, 8.0, 16.0],
            [2.1, 0.5, 1.8],
            overlap_strategy="truncate",
        )
        assert unsorted_classes == sorted_classes == [8.0, 16.0, 500.0]
        assert unsorted_values == pytest.approx(sorted_values)

    def test_total_overlap_with_reject_raises_overlap_message(self):
        with pytest.raises(ValueError, match="overlaps"):
            combine_sieve_pipette([250.0, 500.0], [5.0, 60.0], [300.0, 400.0], [1.8, 2.1])

    def test_total_overlap_with_truncate_still_raises(self):
        """If every pipette class overlaps, truncation leaves nothing left to combine.

        Every pipette diameter exceeds the sieve's finest mesh --
        nothing survives truncation, genuinely nothing left to combine
        even with overlap_strategy="truncate".
        """
        with pytest.raises(ValueError, match="nothing left to combine"):
            combine_sieve_pipette(
                [250.0, 500.0],
                [5.0, 60.0],
                [300.0, 400.0],
                [1.8, 2.1],
                overlap_strategy="truncate",
            )

    def test_empty_sieve_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            combine_sieve_pipette([], [], [16.0], [1.0])

    def test_zero_pipette_total_raises(self):
        with pytest.raises(ValueError, match="zero"):
            combine_sieve_pipette([250.0, 500.0], [5.0, 60.0], [16.0, 31.0], [0.0, 0.0])

    def test_zero_combined_total_raises(self):
        """A zero pan weight plus zero sieve classes still yields a zero combined total.

        Pan weight is zero and the sieve's remaining classes are also
        zero, so after rescaling the pipette values the combined total
        is zero too (distinct from the zero-pipette-total case above).
        """
        with pytest.raises(ValueError, match="combined values sum to zero"):
            combine_sieve_pipette([250.0, 500.0], [0.0, 0.0], [16.0, 31.0], [1.0, 1.0])


# ---------------------------------------------------------------------------
# read_sieve_pipette_csv / write_sieve_pipette_csv
# ---------------------------------------------------------------------------


class TestReadSievePipetteCsv:
    """read_sieve_pipette_csv() parses sieve/pipette/combined samples grouped by group_id."""

    def test_sieve_only_sample(self):
        content = (
            "group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3\n"
            "S1,sieve,0,5.0,,,,,\n"
            "S1,sieve,250,60.0,,,,,\n"
            "S1,sieve,500,25.0,,,,,\n"
        )
        path = _write_temp_csv(content)
        records = read_sieve_pipette_csv(path)
        assert len(records) == 1
        rec = records[0]
        assert isinstance(rec, SievePipetteRecord)
        assert rec.method == "sieve"
        assert rec.group_id == "S1"
        assert rec.classes_um == [250.0, 500.0]
        assert sum(rec.values) == pytest.approx(100.0)
        assert rec.values[0] / rec.values[1] == pytest.approx(5.0 / 60.0)

    def test_pipette_only_sample(self):
        content = (
            "group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3\n"
            "S1,pipette,,1.8,16,,,,\n"
            "S1,pipette,,2.1,31,,,,\n"
        )
        path = _write_temp_csv(content)
        records = read_sieve_pipette_csv(path)
        assert records[0].method == "pipette"
        assert records[0].classes_um == [16.0, 31.0]

    def test_combined_sample(self):
        content = (
            "group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3\n"
            "S1,sieve,0,5.0,,,,,\n"
            "S1,sieve,250,60.0,,,,,\n"
            "S1,pipette,,1.8,16,,,,\n"
            "S1,pipette,,2.1,31,,,,\n"
        )
        path = _write_temp_csv(content)
        records = read_sieve_pipette_csv(path)
        assert records[0].method == "combined"

    def test_raw_stokes_pipette_row(self):
        content = (
            "group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3\n"
            "S1,pipette,,1.0,,28800,10,20,2.65\n"
        )
        path = _write_temp_csv(content)
        records = read_sieve_pipette_csv(path)
        assert records[0].classes_um[0] == pytest.approx(
            stokes_settling_diameter_um(28800, 10, 20, 2.65)
        )

    def test_multiple_samples_grouped_by_group_id(self):
        content = (
            "group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3\n"
            "S1,sieve,0,5.0,,,,,\n"
            "S1,sieve,250,60.0,,,,,\n"
            "S2,sieve,0,3.0,,,,,\n"
            "S2,sieve,500,40.0,,,,,\n"
        )
        path = _write_temp_csv(content)
        records = read_sieve_pipette_csv(path)
        assert {r.group_id for r in records} == {"S1", "S2"}

    def test_oversize_weight_recorded_in_extra_meta(self):
        content = (
            "group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3\n"
            "S1,sieve,0,5.0,,,,,\n"
            "S1,sieve,250,60.0,,,,,\n"
            "S1,sieve,2000,3.0,,,,,\n"
        )
        path = _write_temp_csv(content)
        records = read_sieve_pipette_csv(path)
        assert records[0].extra_meta["oversize_excluded_weight_g"] == "3.0"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read_sieve_pipette_csv("does_not_exist.csv")

    def test_header_only_file_raises(self):
        header = "group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3\n"
        path = _write_temp_csv(header)
        with pytest.raises(ValueError, match="No data rows"):
            read_sieve_pipette_csv(path)

    def test_completely_empty_file_raises(self):
        path = _write_temp_csv("")
        with pytest.raises(ValueError, match="No header row"):
            read_sieve_pipette_csv(path)

    def test_zero_weight_oversize_is_not_recorded_in_extra_meta(self):
        content = (
            "group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3\n"
            "S1,sieve,0,5.0,,,,,\n"
            "S1,sieve,250,60.0,,,,,\n"
            "S1,sieve,1000,0.0,,,,,\n"
        )
        path = _write_temp_csv(content)
        records = read_sieve_pipette_csv(path)
        assert "oversize_excluded_weight_g" not in records[0].extra_meta


class TestBuildRecordDefensive:
    """_build_record()/read_sieve_pipette_csv() reject missing or malformed CSV input."""

    def test_no_sieve_or_pipette_rows_raises(self):
        with pytest.raises(ValueError, match="no sieve or pipette rows"):
            _build_record(Path("dummy.csv"), "S1", [], [])

    def test_missing_required_column_raises(self):
        path = _write_temp_csv("type,mesh_um,weight_g\nsieve,0,5.0\n")
        with pytest.raises(ValueError, match="missing required column"):
            read_sieve_pipette_csv(path)

    def test_unknown_type_raises(self):
        content = (
            "group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3\n"
            "S1,laser,0,5.0,,,,,\n"
        )
        path = _write_temp_csv(content)
        with pytest.raises(ValueError, match="unknown type"):
            read_sieve_pipette_csv(path)

    def test_missing_weight_raises(self):
        content = (
            "group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3\n"
            "S1,sieve,0,,,,,,\n"
        )
        path = _write_temp_csv(content)
        with pytest.raises(ValueError, match="weight_g"):
            read_sieve_pipette_csv(path)


class TestWriteSievePipetteCsvRoundTrip:
    """write_sieve_pipette_csv()/read_sieve_pipette_csv() round-trip sieve+pipette rows."""

    def test_round_trip_sieve_and_pipette(self):
        sieve_rows = [
            SieveRow(mesh_um=0, weight_g=5.0),
            SieveRow(mesh_um=250, weight_g=60.0),
        ]
        pipette_rows = [PipetteRow(weight_g=1.8, size_um=16.0)]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            out_path = Path(tmp.name)
        write_sieve_pipette_csv(out_path, sieve_rows, pipette_rows, group_id="S1")

        records = read_sieve_pipette_csv(out_path)
        assert len(records) == 1
        assert records[0].method == "combined"
        assert records[0].group_id == "S1"


class TestSievePipetteRecordDuckTyping:
    """SievePipetteRecord exposes the same path/group_id/concentration attributes as LSRecord."""

    def test_has_path_group_id_concentration(self):
        rec = SievePipetteRecord(
            path=Path("sample::S1"), classes_um=[250.0], values=[100.0], group_id="S1"
        )
        assert rec.path == Path("sample::S1")
        assert rec.group_id == "S1"
        assert rec.concentration is None
