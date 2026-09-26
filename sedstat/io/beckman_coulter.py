"""Reader for Beckman Coulter LS laser granulometer export files (.$av).

File format
-----------
Sections are delimited by square-bracket headers, e.g. [#Bindiam].
Relevant sections:

  [#Bindiam]     — class boundaries in µm (one value per line)
  [#Binheight]   — volume % per class (one value per line), averaged over runs
  [SizeStats]    — device-computed statistics as key=value pairs
                   (Mean, Mode, Median, SD, Skew, Kurtosis,
                    FWMean, FWMedian, FWSD, FWSkew, FWKurt)
  [SIsave0]      — sample metadata: GroupID, Density, Operator, etc.;
                   fields without an LSRecord attribute go to extra_meta
  [Size0] …      — per-run metadata; Obs field = obscuration / concentration %

Files are read as latin-1 (common in Coulter LS exports). [#Bindiam] and
[#Binheight] are parsed strictly: a value that is not a finite float is an
error, since silently dropping it would shift every later bin by one class.
Unparseable metadata lines are ignored.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LSRecord:
    """Raw data read from one Beckman Coulter LS .$av file."""

    path: Path
    classes_um: list[float]  # class boundaries in µm, as listed in [#Bindiam]
    values: list[float]  # volume % per class (averaged over runs)
    concentration: float | None  # mean obscuration / concentration [%]

    group_id: str | None = None  # sample group identifier from [SIsave0]
    density: float | None = None  # particle density [g/cm³] from [SIsave0]
    operator: str | None = None  # operator name from [SIsave0]
    num_runs: int | None = None  # number of runs averaged

    # Statistics pre-computed by the device (optional)
    device_mean_um: float | None = None
    device_mode_um: float | None = None
    device_median_um: float | None = None
    device_std_um: float | None = None
    device_skew: float | None = None
    device_kurtosis: float | None = None
    device_fw_mean_um: float | None = None
    device_fw_median_um: float | None = None
    device_fw_std: float | None = None
    device_fw_skew: float | None = None
    device_fw_kurt: float | None = None

    extra_meta: dict[str, str] = field(default_factory=dict[str, str])


_STAT_KEYS = {
    "Mean": "device_mean_um",
    "Mode": "device_mode_um",
    "Median": "device_median_um",
    "SD": "device_std_um",
    "Skew": "device_skew",
    "Kurtosis": "device_kurtosis",
    "FWMean": "device_fw_mean_um",
    "FWMedian": "device_fw_median_um",
    "FWSD": "device_fw_std",
    "FWSkew": "device_fw_skew",
    "FWKurt": "device_fw_kurt",
}

_SISAVE0_RECORD_FIELDS = {"GroupID", "Density", "Operator"}


class _Sections:
    """Accumulates the parsed content of each bracketed section."""

    def __init__(self) -> None:
        self.classes: list[float] = []
        self.values: list[float] = []
        self.concentrations: list[float] = []
        self.stats: dict[str, float] = {}
        self.meta: dict[str, str] = {}
        self.extra_meta: dict[str, str] = {}


def _append_float(target: list[float], text: str) -> None:
    with contextlib.suppress(ValueError):
        target.append(float(text))


def _append_bin_value(target: list[float], line: str, where: str) -> None:
    if not line:
        return
    try:
        value = float(line)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        raise ValueError(f"Unreadable value {line!r} in {where}.")
    target.append(value)


def _split_kv(line: str) -> tuple[str, str] | None:
    if "=" not in line:
        return None
    key, _, val = line.partition("=")
    return key.strip(), val.strip()


def _handle_size_stats(line: str, sections: _Sections) -> None:
    kv = _split_kv(line)
    if kv is None:
        return
    key, val = kv
    if key not in _STAT_KEYS:
        return
    with contextlib.suppress(ValueError):
        sections.stats[_STAT_KEYS[key]] = float(val)


def _handle_sisave0(line: str, sections: _Sections) -> None:
    kv = _split_kv(line)
    if kv is None:
        return
    key, val = kv
    sections.meta[key] = val
    if key not in _SISAVE0_RECORD_FIELDS:
        sections.extra_meta[f"SIsave0.{key}"] = val


def _handle_size_run(line: str, block: str, sections: _Sections) -> None:
    kv = _split_kv(line)
    if kv is None:
        return
    key, val = kv
    if key == "Obs":
        _append_float(sections.concentrations, val)
    else:
        sections.extra_meta[f"{block}.{key}"] = val


def _handle_common(line: str, sections: _Sections) -> None:
    kv = _split_kv(line)
    if kv is not None:
        sections.extra_meta[f"common.{kv[0]}"] = kv[1]


_BLOCK_HANDLERS = {
    "SizeStats": _handle_size_stats,
    "SIsave0": _handle_sisave0,
    "common": _handle_common,
}


def _dispatch_line(line: str, block: str, sections: _Sections, where: str) -> None:
    if block == "#Bindiam":
        _append_bin_value(sections.classes, line, f"[{block}] at {where}")
    elif block == "#Binheight":
        _append_bin_value(sections.values, line, f"[{block}] at {where}")
    elif block in _BLOCK_HANDLERS:
        _BLOCK_HANDLERS[block](line, sections)
    elif block.startswith("Size") and block[4:].isdigit():
        _handle_size_run(line, block, sections)


def _parse_sections(lines: list[str], path: Path) -> _Sections:
    sections = _Sections()
    block: str | None = None

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            block = line[1:-1]
        elif block is not None:
            _dispatch_line(line, block, sections, f"line {line_no} of {path.name}")

    return sections


def _validate_sections(sections: _Sections, path: Path) -> None:
    if not sections.classes:
        raise ValueError(f"No class boundaries ([#Bindiam]) found in {path.name}.")
    if not sections.values:
        raise ValueError(f"No measurement data ([#Binheight]) found in {path.name}.")
    n_classes, n_values = len(sections.classes), len(sections.values)
    if n_classes not in (n_values, n_values + 1):
        raise ValueError(
            f"[#Bindiam] has {n_classes} values and [#Binheight] has {n_values} in "
            f"{path.name}; expected {n_values} or {n_values + 1} class boundaries."
        )
    for i in range(len(sections.classes) - 1):
        if sections.classes[i] >= sections.classes[i + 1]:
            raise ValueError(
                f"Class boundaries must be strictly increasing in {path.name}; "
                f"found {sections.classes[i]} ≥ {sections.classes[i + 1]} at position {i}."
            )


def _resolve_num_runs(meta: dict[str, str], extra_meta: dict[str, str]) -> int | None:
    num_runs_str = meta.get("NumRuns") or extra_meta.get("common.NumRuns")
    if num_runs_str is None:
        # e.g. Size0.NumRuns
        for key, val in extra_meta.items():
            if key.endswith(".NumRuns"):
                num_runs_str = val
                break
    if num_runs_str is None:
        return None
    try:
        return int(num_runs_str)
    except ValueError:
        return None


def _resolve_density(meta: dict[str, str]) -> float | None:
    if "Density" not in meta:
        return None
    try:
        return float(meta["Density"])
    except ValueError:
        return None


def read_ls_file(path: str | Path) -> LSRecord:
    """Parse a Beckman Coulter LS .$av export file.

    Args:
        path: Path to the .$av file.

    Returns:
        Class boundaries, measured values, concentration, sample metadata
        (GroupID, Density, Operator), and device-provided statistics.

    Raises:
        FileNotFoundError: If the path is not an existing file.
        ValueError: If [#Bindiam] or [#Binheight] is missing or holds a value
            that is not a finite number, if the class boundaries are not
            strictly increasing, or if the number of boundaries does not match
            the number of values (N or N+1 boundaries for N values).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open(encoding="latin-1") as fh:
        lines = fh.readlines()

    sections = _parse_sections(lines, path)
    _validate_sections(sections, path)

    concentration = (
        sum(sections.concentrations) / len(sections.concentrations)
        if sections.concentrations
        else None
    )

    return LSRecord(
        path=path,
        classes_um=sections.classes,
        values=sections.values,
        concentration=concentration,
        group_id=sections.meta.get("GroupID"),
        density=_resolve_density(sections.meta),
        operator=sections.meta.get("Operator"),
        num_runs=_resolve_num_runs(sections.meta, sections.extra_meta),
        extra_meta=sections.extra_meta,
        **sections.stats,
    )
