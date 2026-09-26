"""Reader/writer for sieve and pipette grain-size lab data.

This module defines a SedStat CSV template for it and converts sieve
weights / pipette draws into the ``(classes_um, values)`` shape
``sedstat.core.statistics.compute_all`` expects — the same shape
``LSRecord`` already provides for laser-diffraction data.

Sieve analysis
---------------
A stack of sieves (coarsest on top, pan at the bottom) is shaken; material
retained on a given mesh passed the next-coarser mesh above it but not
that mesh itself, so its true size range is ``(mesh, next-coarser-mesh]``.
Converting a sieve stack to the upper-bounds-only form ``compute_all`` accepts
therefore requires shifting each weight by one position relative to its
mesh size; see :func:`sieve_to_distribution`.

Pipette analysis
-----------------
A settling column is sampled at known times/depths; by Stokes' law, each
draw's weight represents the material finer than an equivalent diameter.
Two ways to supply that equivalent diameter are supported per row: directly
(a lab that already worked it out from a published settling-time table), or
as raw time/depth/temperature/particle-density, from which
:func:`sedstat.core.stokes.stokes_settling_diameter_um` computes it.

CSV template
------------
One file may hold multiple samples, grouped by ``group_id``::

    group_id,type,mesh_um,weight_g,size_um,time_s,depth_cm,temp_c,density_g_cm3
    S1,sieve,0,,,,,,          <- pan (mesh_um=0 or blank)
    S1,sieve,63,8.5,,,,,
    S1,sieve,250,60.2,,,,,
    S1,pipette,,2.1,31,,,,
    S1,pipette,,1.8,,28800,10,20,2.65

``type`` is ``sieve`` or ``pipette`` (case-insensitive). Sieve rows use
``mesh_um``/``weight_g`` only. Pipette rows use ``weight_g`` plus *either*
``size_um`` (direct-diameter mode) *or* ``time_s``+``depth_cm``+``temp_c``
(raw-Stokes mode, ``density_g_cm3`` optional, defaults to 2.65).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sedstat.core.stokes import stokes_settling_diameter_um

_DEFAULT_PARTICLE_DENSITY_G_CM3 = 2.65

_CSV_FIELDNAMES = [
    "group_id",
    "type",
    "mesh_um",
    "weight_g",
    "size_um",
    "time_s",
    "depth_cm",
    "temp_c",
    "density_g_cm3",
]


@dataclass
class SieveRow:
    """One weighed sieve fraction.

    ``is_pan`` is the authoritative pan marker — set it explicitly rather
    than relying on ``mesh_um == 0``. The pan's ``mesh_um`` is 0 by
    convention, needed so it sorts first, but callers that *compute*
    ``mesh_um`` should not rely on hitting exactly 0.0.
    ``sieve_to_distribution`` accepts either signal.
    """

    mesh_um: float
    weight_g: float
    group_id: str | None = None
    is_pan: bool = False


@dataclass
class PipetteRow:
    """One weighed pipette draw.

    Exactly one of (``size_um``) or (``time_s``, ``depth_cm``, ``temp_c``)
    must be provided — see :func:`pipette_to_distribution`.
    """

    weight_g: float
    group_id: str | None = None
    size_um: float | None = None
    time_s: float | None = None
    depth_cm: float | None = None
    temp_c: float | None = None
    density_g_cm3: float | None = None


@dataclass
class SievePipetteRecord:
    """Converted sieve/pipette sample, ready for ``compute_all()``.

    ``path``, ``group_id``, and ``concentration`` exist so this record can
    be used anywhere an ``LSRecord`` is used without subclassing it.
    """

    path: Path
    classes_um: list[float]
    values: list[float]
    concentration: float | None = None
    group_id: str | None = None
    method: Literal["sieve", "pipette", "combined"] = "sieve"
    density_g_cm3: float | None = None
    sieve_rows: list[SieveRow] = field(default_factory=list[SieveRow])
    pipette_rows: list[PipetteRow] = field(default_factory=list[PipetteRow])
    extra_meta: dict[str, str] = field(default_factory=dict[str, str])


def _normalize_to_100(values: list[float], context: str) -> list[float]:
    total = sum(values)
    if total == 0:
        raise ValueError(f"{context} weights sum to zero.")
    return [v / total * 100.0 for v in values]


def _validate_sieve_pan(rows: list[SieveRow], sorted_rows: list[SieveRow]) -> None:
    """Raise if the pan-row invariants aren't satisfied.

    The pan row must sort first (mesh_um=0) because sieve_to_distribution's
    shift is purely positional, so a row flagged as the pan that does not
    sort first is an input inconsistency, not something to work around.
    """
    flagged_pan_rows = [r for r in rows if r.is_pan]
    if len(flagged_pan_rows) > 1:
        raise ValueError("more than one sieve row is marked is_pan=True.")

    first = sorted_rows[0]
    if flagged_pan_rows and flagged_pan_rows[0] is not first:
        raise ValueError(
            "the row marked is_pan=True must have the smallest mesh_um "
            "(mesh_um=0 by convention) so it sorts first; got "
            f"mesh_um={flagged_pan_rows[0].mesh_um}, but "
            f"mesh_um={first.mesh_um} sorts first."
        )
    if not (first.is_pan or first.mesh_um == 0):
        raise ValueError(
            "sieve rows must include a pan fraction (mesh_um=0) to close "
            "the distribution at the fine end."
        )


def sieve_to_distribution(rows: list[SieveRow]) -> tuple[list[float], list[float]]:
    """Convert weighed sieve fractions to ``(classes_um, values)``.

    Requires exactly one pan row (``is_pan=True``, or ``mesh_um == 0``). The
    coarsest sieve's own retained weight represents an open-ended ``>mesh``
    oversize tail with no home in a closed distribution and is excluded —
    call sites can recover it from the input rows if needed.

    Returns:
        ``(classes_um, values)`` — ``values`` are normalized to sum to 100
        (percent of the fitted, oversize-excluded distribution), matching
        the convention every other reader in this package already uses.

    Raises:
        ValueError: If *rows* is empty, holds duplicate mesh sizes, has no
            pan row or more than one, carries only the pan, or weighs zero.
    """
    if not rows:
        raise ValueError("sieve rows must not be empty.")

    sorted_rows = sorted(rows, key=lambda r: r.mesh_um)

    mesh_values = [r.mesh_um for r in sorted_rows]
    if len(set(mesh_values)) != len(mesh_values):
        raise ValueError("duplicate mesh_um values in sieve rows.")

    _validate_sieve_pan(rows, sorted_rows)

    if len(sorted_rows) < 2:
        raise ValueError(
            "sieve rows must include at least one real sieve mesh in addition to the pan."
        )

    classes_um = [r.mesh_um for r in sorted_rows[1:]]
    values = _normalize_to_100([r.weight_g for r in sorted_rows[:-1]], "sieve")
    return classes_um, values


def _resolve_pipette_diameter(row: PipetteRow) -> float:
    """Resolve one pipette row's equivalent diameter (µm), direct or Stokes."""
    has_raw = row.time_s is not None and row.depth_cm is not None and row.temp_c is not None

    if row.size_um is not None:
        if has_raw:
            raise ValueError(
                "pipette row has both size_um and raw Stokes inputs "
                "(time_s/depth_cm/temp_c) — ambiguous, provide only one."
            )
        return row.size_um

    if row.time_s is None or row.depth_cm is None or row.temp_c is None:
        raise ValueError("pipette row needs either size_um, or time_s+depth_cm+temp_c.")

    return stokes_settling_diameter_um(
        row.time_s,
        row.depth_cm,
        row.temp_c,
        row.density_g_cm3 or _DEFAULT_PARTICLE_DENSITY_G_CM3,
    )


def pipette_to_distribution(rows: list[PipetteRow]) -> tuple[list[float], list[float]]:
    """Convert weighed pipette draws to ``(classes_um, values)``.

    Each row's equivalent diameter is resolved directly (``size_um``) or
    computed via Stokes' law from ``time_s``/``depth_cm``/``temp_c``
    (``density_g_cm3`` defaults to 2.65 if not given in raw mode).

    Returns:
        ``(classes_um, values)`` sorted ascending by resolved diameter;
        ``values`` are normalized to sum to 100 (percent), matching the
        convention every other reader in this package uses.

    Raises:
        ValueError: If *rows* is empty, a row's diameter inputs are missing
            or ambiguous, two rows resolve to the same diameter, or the
            draws weigh zero.
    """
    if not rows:
        raise ValueError("pipette rows must not be empty.")

    resolved = [(_resolve_pipette_diameter(row), row.weight_g) for row in rows]
    resolved.sort(key=lambda pair: pair[0])

    diam_values = [d for d, _ in resolved]
    if len(set(diam_values)) != len(diam_values):
        raise ValueError("duplicate resolved diameters in pipette rows.")

    classes_um = [d for d, _ in resolved]
    values = _normalize_to_100([w for _, w in resolved], "pipette")
    return classes_um, values


def _truncate_pipette_overlap(
    pipette_classes_um: list[float],
    pipette_values: list[float],
    finest_sieve_mesh: float,
) -> tuple[list[float], list[float]]:
    """Drop pipette classes coarser than finest_sieve_mesh, preserving mass.

    Their weight is folded into the coarsest *retained* pipette class rather
    than discarded — see combine_sieve_pipette's docstring for why. Raises
    if nothing would remain.
    """
    # Public API, so the caller's input is not assumed to be sorted.
    pairs = sorted(zip(pipette_classes_um, pipette_values, strict=True), key=lambda pair: pair[0])
    kept_pairs = [(d, w) for d, w in pairs if d <= finest_sieve_mesh]
    if not kept_pairs:
        raise ValueError(
            f"every pipette diameter (finest {pairs[0][0]} µm) exceeds the "
            f"sieve stack's finest mesh ({finest_sieve_mesh} µm) — nothing "
            "left to combine."
        )
    overflow_weight = sum(w for d, w in pairs if d > finest_sieve_mesh)
    kept_classes = [d for d, _ in kept_pairs]
    kept_values = [w for _, w in kept_pairs]
    kept_values[-1] += overflow_weight
    return kept_classes, kept_values


def combine_sieve_pipette(
    sieve_classes_um: list[float],
    sieve_values: list[float],
    pipette_classes_um: list[float],
    pipette_values: list[float],
    *,
    overlap_strategy: Literal["truncate", "reject"] = "reject",
) -> tuple[list[float], list[float]]:
    """Stitch a pipette breakdown into a sieve distribution's pan fraction.

    The pipette's total weight is rescaled to match the sieve distribution's
    pan-class weight, then the pipette's finer classes replace that single
    pan class.

    Overlap reconciliation: if the pipette's coarsest analyzed diameter
    exceeds the sieve stack's finest mesh, those pipette classes would
    double-count material the sieve stack already measured (draw-condition
    imprecision can push a computed equivalent diameter slightly past the
    finest mesh). A methodological choice, so it is user-selectable via
    ``overlap_strategy`` rather than fixed:

    - ``"reject"`` (default): raise, refusing to combine — the caller must
      clean up the raw data themselves.
    - ``"truncate"``: drop the overlapping pipette classes and fold their
      weight into the coarsest *retained* pipette class, so total sample
      mass is preserved rather than silently dropped. A truncation, not a
      re-interpolation — no assumption is made about how that mass is
      actually distributed within the overlap band, since the raw pipette
      draws do not support one. Still raises if truncation would remove
      every pipette class.

    Returns:
        ``(classes_um, values)``, ``values`` renormalized to sum to 100.

    Raises:
        ValueError: If either distribution is empty, the ranges overlap
            under ``overlap_strategy="reject"``, truncation would leave no
            pipette class, or the draws or the combined distribution weigh
            zero.
    """
    if not sieve_classes_um or not pipette_classes_um:
        raise ValueError("both sieve and pipette distributions must be non-empty.")

    finest_sieve_mesh = sieve_classes_um[0]

    if max(pipette_classes_um) > finest_sieve_mesh:
        if overlap_strategy == "reject":
            coarsest_pipette_diameter = max(pipette_classes_um)
            raise ValueError(
                f"pipette diameter range (up to {coarsest_pipette_diameter} µm) "
                f"overlaps the sieve range (finest mesh {finest_sieve_mesh} µm) "
                "— cannot combine without double-counting. Pass "
                'overlap_strategy="truncate" to reconcile instead of rejecting.'
            )
        pipette_classes_um, pipette_values = _truncate_pipette_overlap(
            pipette_classes_um, pipette_values, finest_sieve_mesh
        )

    pan_weight = sieve_values[0]
    pipette_total = sum(pipette_values)
    if pipette_total == 0:
        raise ValueError("pipette weights sum to zero — cannot rescale.")

    rescaled_pipette_values = [w / pipette_total * pan_weight for w in pipette_values]

    classes_um = [*pipette_classes_um, *sieve_classes_um[1:]]
    values = [*rescaled_pipette_values, *sieve_values[1:]]

    total = sum(values)
    if total == 0:
        raise ValueError("combined values sum to zero.")
    values = [v / total * 100.0 for v in values]

    return classes_um, values


def _parse_float(text: str, column: str, line_no: int) -> float | None:
    """Parse one optional numeric CSV cell; a blank cell means "not given"."""
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"row {line_no}: {column} must be a number; got {text!r}.") from None


def _parse_weight(text: str, line_no: int) -> float:
    weight_g = _parse_float(text, "weight_g", line_no)
    if weight_g is None:
        raise ValueError(f"row {line_no}: weight_g is required.")
    if weight_g < 0:
        raise ValueError(f"row {line_no}: weight_g must not be negative; got {weight_g}.")
    return weight_g


def _row_from_csv_dict(d: dict[str, str], line_no: int) -> SieveRow | PipetteRow:
    row_type = d.get("type", "").strip().lower()
    group_id = d.get("group_id", "").strip() or None
    weight_g = _parse_weight(d.get("weight_g", ""), line_no)

    if row_type == "sieve":
        mesh_um = _parse_float(d.get("mesh_um", ""), "mesh_um", line_no) or 0.0
        return SieveRow(
            mesh_um=mesh_um,
            weight_g=weight_g,
            group_id=group_id,
            is_pan=(mesh_um == 0.0),
        )
    if row_type == "pipette":
        return PipetteRow(
            weight_g=weight_g,
            group_id=group_id,
            size_um=_parse_float(d.get("size_um", ""), "size_um", line_no),
            time_s=_parse_float(d.get("time_s", ""), "time_s", line_no),
            depth_cm=_parse_float(d.get("depth_cm", ""), "depth_cm", line_no),
            temp_c=_parse_float(d.get("temp_c", ""), "temp_c", line_no),
            density_g_cm3=_parse_float(d.get("density_g_cm3", ""), "density_g_cm3", line_no),
        )
    raise ValueError(
        f"row {line_no}: unknown type {d.get('type')!r}; expected 'sieve' or 'pipette'."
    )


def _validate_csv_header(fieldnames: Sequence[str] | None, path: Path) -> None:
    if fieldnames is None:
        raise ValueError(f"No header row found in {path.name}.")
    missing = [c for c in ("group_id", "type", "weight_g") if c not in fieldnames]
    if missing:
        raise ValueError(f"{path.name} is missing required column(s): {', '.join(missing)}.")


def _read_rows_by_group(
    reader: csv.DictReader,
) -> dict[str, list[SieveRow | PipetteRow]]:
    rows_by_group: dict[str, list[SieveRow | PipetteRow]] = defaultdict(list)
    for line_no, raw_row in enumerate(reader, start=2):
        row = _row_from_csv_dict(raw_row, line_no)
        rows_by_group[row.group_id or f"sample_{line_no}"].append(row)
    return rows_by_group


def read_sieve_pipette_csv(
    path: str | Path,
    *,
    overlap_strategy: Literal["truncate", "reject"] = "reject",
) -> list[SievePipetteRecord]:
    """Read a SedStat sieve/pipette CSV template into one record per sample.

    Rows are grouped by ``group_id`` (first-seen order). A group with only
    sieve rows becomes ``method="sieve"``; only pipette rows becomes
    ``method="pipette"``; both present are combined via
    :func:`combine_sieve_pipette` into ``method="combined"``, using
    *overlap_strategy* for every group in this file (a single choice per
    import call, not per group).

    Raises:
        FileNotFoundError: If the path is not an existing file.
        ValueError: If the file is empty, missing required columns, or a row
            fails validation (see the module docstring for the schema).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        _validate_csv_header(reader.fieldnames, path)
        rows_by_group = _read_rows_by_group(reader)

    if not rows_by_group:
        raise ValueError(f"No data rows found in {path.name}.")

    records: list[SievePipetteRecord] = []
    for group_id, rows in rows_by_group.items():
        sieve_rows = [r for r in rows if isinstance(r, SieveRow)]
        pipette_rows = [r for r in rows if isinstance(r, PipetteRow)]
        records.append(_build_record(path, group_id, sieve_rows, pipette_rows, overlap_strategy))
    return records


def resolve_sieve_pipette_distribution(
    sieve_rows: list[SieveRow],
    pipette_rows: list[PipetteRow],
    *,
    no_data_message: str,
    overlap_strategy: Literal["truncate", "reject"] = "reject",
) -> tuple[list[float], list[float], Literal["sieve", "pipette", "combined"]]:
    """Pick sieve/pipette/combined and build the ``(classes_um, values)`` pair.

    Shared by the CSV batch-import path (``_build_record``) and the
    manual-entry dialog in the GUI. Both need the same branching and differ
    only in how "neither" should be worded, hence *no_data_message* rather
    than a hardcoded string here.

    Raises:
        ValueError: With *no_data_message* if neither row list holds data,
            or whatever the underlying conversion raises.
    """
    if sieve_rows and pipette_rows:
        sieve_classes, sieve_values = sieve_to_distribution(sieve_rows)
        pipette_classes, pipette_values = pipette_to_distribution(pipette_rows)
        classes_um, values = combine_sieve_pipette(
            sieve_classes,
            sieve_values,
            pipette_classes,
            pipette_values,
            overlap_strategy=overlap_strategy,
        )
        return classes_um, values, "combined"
    if sieve_rows:
        classes_um, values = sieve_to_distribution(sieve_rows)
        return classes_um, values, "sieve"
    if pipette_rows:
        classes_um, values = pipette_to_distribution(pipette_rows)
        return classes_um, values, "pipette"
    raise ValueError(no_data_message)


def _build_record(
    path: Path,
    group_id: str,
    sieve_rows: list[SieveRow],
    pipette_rows: list[PipetteRow],
    overlap_strategy: Literal["truncate", "reject"] = "reject",
) -> SievePipetteRecord:
    extra_meta: dict[str, str] = {}
    if sieve_rows:
        oversize = max(sieve_rows, key=lambda r: r.mesh_um)
        if oversize.mesh_um > 0 and oversize.weight_g > 0:
            extra_meta["oversize_excluded_weight_g"] = str(oversize.weight_g)

    classes_um, values, method = resolve_sieve_pipette_distribution(
        sieve_rows,
        pipette_rows,
        no_data_message=f"group {group_id!r} has no sieve or pipette rows.",
        overlap_strategy=overlap_strategy,
    )

    return SievePipetteRecord(
        path=Path(f"{path}::{group_id}"),
        classes_um=classes_um,
        values=values,
        group_id=group_id,
        method=method,
        sieve_rows=sieve_rows,
        pipette_rows=pipette_rows,
        extra_meta=extra_meta,
    )


def _csv_cell(value: float | None) -> float | str:
    return value if value is not None else ""


def write_sieve_pipette_csv(
    path: str | Path,
    sieve_rows: list[SieveRow],
    pipette_rows: list[PipetteRow],
    group_id: str,
) -> None:
    """Write sieve/pipette rows to the same CSV template :func:`read_sieve_pipette_csv` reads."""
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for row in sieve_rows:
            writer.writerow(
                {
                    "group_id": group_id,
                    "type": "sieve",
                    "mesh_um": row.mesh_um,
                    "weight_g": row.weight_g,
                    "size_um": "",
                    "time_s": "",
                    "depth_cm": "",
                    "temp_c": "",
                    "density_g_cm3": "",
                }
            )
        for prow in pipette_rows:
            writer.writerow(
                {
                    "group_id": group_id,
                    "type": "pipette",
                    "mesh_um": "",
                    "weight_g": prow.weight_g,
                    "size_um": _csv_cell(prow.size_um),
                    "time_s": _csv_cell(prow.time_s),
                    "depth_cm": _csv_cell(prow.depth_cm),
                    "temp_c": _csv_cell(prow.temp_c),
                    "density_g_cm3": _csv_cell(prow.density_g_cm3),
                }
            )
