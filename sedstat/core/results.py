"""Dataclasses for grain size statistics results."""

from __future__ import annotations

from dataclasses import dataclass, field

from sedstat.core.classification import ClassificationScheme


@dataclass
class MomentStats:
    """Statistics calculated by one of the three moment methods."""

    mean: float
    std: float
    skewness: float
    kurtosis: float


@dataclass
class FolkWardStats:
    """Statistics calculated by one of the two Folk & Ward graphical methods."""

    mean: float
    std: float
    skewness: float
    kurtosis: float


@dataclass
class Percentiles:
    """Percentile grain diameters in µm interpolated from the cumulative curve."""

    p5: float
    p10: float
    p16: float
    p25: float
    p50: float
    p75: float
    p84: float
    p90: float
    p95: float

    @property
    def d10(self) -> float:
        return self.p10

    @property
    def d50(self) -> float:
        return self.p50

    @property
    def d90(self) -> float:
        return self.p90

    @property
    def d90_d10(self) -> float:
        """D90/D10 ratio — a coarse/fine spread indicator."""
        return self.p90 / self.p10

    @property
    def d90_minus_d10(self) -> float:
        """D90 - D10 (µm), the absolute spread between the coarse and fine tails."""
        return self.p90 - self.p10

    @property
    def d75_d25(self) -> float:
        """D75/D25 ratio — a narrower-interval coarse/fine spread indicator."""
        return self.p75 / self.p25

    @property
    def d75_minus_d25(self) -> float:
        """D75 - D25 (µm), the absolute spread between the quartiles."""
        return self.p75 - self.p25


@dataclass
class Fractions:
    """Volume/weight percentages per Wentworth fraction."""

    clay: float = 0.0
    fine_silt: float = 0.0
    medium_silt: float = 0.0
    coarse_silt: float = 0.0
    fine_sand: float = 0.0
    medium_sand: float = 0.0
    coarse_sand: float = 0.0
    gravel: float = 0.0

    @property
    def total_clay(self) -> float:
        return self.clay

    @property
    def total_silt(self) -> float:
        return self.fine_silt + self.medium_silt + self.coarse_silt

    @property
    def total_sand(self) -> float:
        return self.fine_sand + self.medium_sand + self.coarse_sand


@dataclass
class GrainSizeResult:
    """Full result of a grain size analysis for one sample."""

    # Raw inputs, preserved for plotting
    classes_um: list[float]
    values: list[float]

    # Five statistical methods from Blott & Pye (2001) Table II
    arithmetic: MomentStats  # Table IIa — metric, normal distribution
    geometric: MomentStats  # Table IIb — metric, log-normal
    logarithmic: MomentStats  # Table IIc — phi, log-normal
    folk_ward_log: FolkWardStats  # Table IId — phi, original Folk & Ward (1957)
    folk_ward_geo: FolkWardStats  # Table IIe — metric, modified Folk & Ward

    # Percentiles (in µm) and fractions
    percentiles: Percentiles
    fractions: Fractions

    # Textural descriptions (set by classification module)
    descriptions: dict[str, str] = field(default_factory=dict[str, str])
    # Which classification scheme produced the descriptions
    scheme: ClassificationScheme = field(default=ClassificationScheme.GRADISTAT)

    def to_dict(self) -> dict[str, float | str]:
        """Flat dict suitable for a DataFrame row.

        Every value is a float except the ``desc_*`` entries, which carry the
        classification strings from ``descriptions``.
        """
        return {
            # Arithmetic
            "arith_mean_um": self.arithmetic.mean,
            "arith_std_um": self.arithmetic.std,
            "arith_skewness": self.arithmetic.skewness,
            "arith_kurtosis": self.arithmetic.kurtosis,
            # Geometric moments
            "geo_mean_um": self.geometric.mean,
            "geo_std": self.geometric.std,
            "geo_skewness": self.geometric.skewness,
            "geo_kurtosis": self.geometric.kurtosis,
            # Logarithmic moments
            "log_mean_phi": self.logarithmic.mean,
            "log_std_phi": self.logarithmic.std,
            "log_skewness": self.logarithmic.skewness,
            "log_kurtosis": self.logarithmic.kurtosis,
            # Folk & Ward logarithmic
            "fw_log_mean_phi": self.folk_ward_log.mean,
            "fw_log_std_phi": self.folk_ward_log.std,
            "fw_log_skewness": self.folk_ward_log.skewness,
            "fw_log_kurtosis": self.folk_ward_log.kurtosis,
            # Folk & Ward geometric
            "fw_geo_mean_um": self.folk_ward_geo.mean,
            "fw_geo_std": self.folk_ward_geo.std,
            "fw_geo_skewness": self.folk_ward_geo.skewness,
            "fw_geo_kurtosis": self.folk_ward_geo.kurtosis,
            # Percentiles (full set + d10/d50/d90 aliases for backward compat)
            "p5_um": self.percentiles.p5,
            "p10_um": self.percentiles.p10,
            "p16_um": self.percentiles.p16,
            "p25_um": self.percentiles.p25,
            "p50_um": self.percentiles.p50,
            "p75_um": self.percentiles.p75,
            "p84_um": self.percentiles.p84,
            "p90_um": self.percentiles.p90,
            "p95_um": self.percentiles.p95,
            "d10_um": self.percentiles.p10,
            "d50_um": self.percentiles.p50,
            "d90_um": self.percentiles.p90,
            "d90_d10": self.percentiles.d90_d10,
            "d90_minus_d10": self.percentiles.d90_minus_d10,
            "d75_d25": self.percentiles.d75_d25,
            "d75_minus_d25": self.percentiles.d75_minus_d25,
            # Wentworth fractions (individual + totals)
            "clay_pct": self.fractions.clay,
            "fine_silt_pct": self.fractions.fine_silt,
            "medium_silt_pct": self.fractions.medium_silt,
            "coarse_silt_pct": self.fractions.coarse_silt,
            "fine_sand_pct": self.fractions.fine_sand,
            "medium_sand_pct": self.fractions.medium_sand,
            "coarse_sand_pct": self.fractions.coarse_sand,
            "gravel_pct": self.fractions.gravel,
            "total_silt_pct": self.fractions.total_silt,
            "total_sand_pct": self.fractions.total_sand,
            # Descriptions
            **{f"desc_{k}": v for k, v in self.descriptions.items()},
        }
