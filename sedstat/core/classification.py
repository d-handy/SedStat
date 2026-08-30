"""Textural classification of grain size statistics: sorting, skewness,
kurtosis, and Wentworth grade names.

Four classification schemes are provided — see ``ClassificationScheme`` for
what each one controls. Use ``Classifier(scheme)`` for a single object with
all classification methods for that scheme; the standalone ``classify_*``
functions below remain for backward compatibility.
"""

from __future__ import annotations

import math
from enum import Enum

# ---------------------------------------------------------------------------
# Classification scheme
# ---------------------------------------------------------------------------


class ClassificationScheme(Enum):
    """Grain-size classification scheme: controls fraction boundaries and sorting-class labels.

    GRADISTAT, FOLK_WARD_1957, and ISO14688 share the Wentworth/EU 63 µm
    silt/sand boundary and GRADISTAT sorting labels; USDA uses the 50 µm
    US soil-science boundary.
    """

    GRADISTAT = "GRADISTAT"  # Blott & Pye (2001): silt < 63 µm
    FOLK_WARD_1957 = "Folk & Ward (1957)"  # FW 1957: silt < 63 µm, 6-class sort
    USDA = "USDA"  # US soil science: silt < 50 µm
    ISO14688 = "ISO 14688-1:2017"  # ISO 14688-1:2017 Table 1: silt < 63 µm


# ---------------------------------------------------------------------------
# Internal threshold tables
# ---------------------------------------------------------------------------


def _pick(thresholds: list[tuple[float, str]], value: float) -> str:
    for upper, name in thresholds:
        if value < upper:
            return name
    return thresholds[-1][1]


# ── Sorting (phi / logarithmic scale) ─────────────────────────────────────

# USDA uses the same sorting labels as GRADISTAT (sorting is a statistical
# property unrelated to the grain-size boundary choice).
_SORTING_PHI: dict[ClassificationScheme, list[tuple[float, str]]] = {
    ClassificationScheme.GRADISTAT: [
        (0.35, "Very Well Sorted"),
        (0.50, "Well Sorted"),
        (0.70, "Moderately Well Sorted"),
        (1.00, "Moderately Sorted"),
        (2.00, "Poorly Sorted"),
        (4.00, "Very Poorly Sorted"),
        (math.inf, "Extremely Poorly Sorted"),
    ],
    ClassificationScheme.FOLK_WARD_1957: [
        (0.35, "Well Sorted"),
        (0.50, "Moderately Well Sorted"),
        (0.71, "Moderately Sorted"),
        (1.00, "Poorly Sorted"),
        (2.00, "Very Poorly Sorted"),
        (math.inf, "Extremely Poorly Sorted"),
    ],
}
# USDA and ISO14688 share GRADISTAT sorting thresholds — neither standard
# defines its own statistical sorting classification (sorting classes are a
# Folk & Ward/GRADISTAT statistical concept, unrelated to either boundary
# standard's own particle-size naming).
_SORTING_PHI[ClassificationScheme.USDA] = _SORTING_PHI[ClassificationScheme.GRADISTAT]
_SORTING_PHI[ClassificationScheme.ISO14688] = _SORTING_PHI[ClassificationScheme.GRADISTAT]

# ── Sorting (geometric / metric scale) ────────────────────────────────────
# Thresholds are 2^σ_φ  (e.g. 2^0.35 ≈ 1.274).

_SORTING_GEO: dict[ClassificationScheme, list[tuple[float, str]]] = {
    ClassificationScheme.GRADISTAT: [
        (1.27, "Very Well Sorted"),
        (1.41, "Well Sorted"),
        (1.62, "Moderately Well Sorted"),
        (2.00, "Moderately Sorted"),
        (4.00, "Poorly Sorted"),
        (16.0, "Very Poorly Sorted"),
        (math.inf, "Extremely Poorly Sorted"),
    ],
    ClassificationScheme.FOLK_WARD_1957: [
        (1.27, "Well Sorted"),  # 2^0.35
        (1.41, "Moderately Well Sorted"),  # 2^0.50
        (1.63, "Moderately Sorted"),  # 2^0.71
        (2.00, "Poorly Sorted"),  # 2^1.00
        (4.00, "Very Poorly Sorted"),  # 2^2.00
        (math.inf, "Extremely Poorly Sorted"),
    ],
}
# USDA and ISO14688 share GRADISTAT sorting thresholds (see note above).
_SORTING_GEO[ClassificationScheme.USDA] = _SORTING_GEO[ClassificationScheme.GRADISTAT]
_SORTING_GEO[ClassificationScheme.ISO14688] = _SORTING_GEO[ClassificationScheme.GRADISTAT]

# ── Skewness and kurtosis (identical in both schemes) ─────────────────────

_SKEWNESS_FW_PHI = [
    (-math.inf, "Very Coarse Skewed"),
    (-0.30, "Coarse Skewed"),
    (-0.10, "Symmetrical"),
    (0.10, "Fine Skewed"),
    (0.30, "Very Fine Skewed"),
    (math.inf, "Very Fine Skewed"),
]


def _sk_fw_phi(sk: float) -> str:
    if sk > 0.30:
        return "Very Fine Skewed"
    if sk > 0.10:
        return "Fine Skewed"
    if sk >= -0.10:
        return "Symmetrical"
    if sk >= -0.30:
        return "Coarse Skewed"
    return "Very Coarse Skewed"


def _sk_fw_geo(sk: float) -> str:
    if sk < -0.30:
        return "Very Fine Skewed"
    if sk < -0.10:
        return "Fine Skewed"
    if sk <= 0.10:
        return "Symmetrical"
    if sk <= 0.30:
        return "Coarse Skewed"
    return "Very Coarse Skewed"


def _sk_moments_log(sk: float) -> str:
    if sk > 1.30:
        return "Very Fine Skewed"
    if sk > 0.43:
        return "Fine Skewed"
    if sk >= -0.43:
        return "Symmetrical"
    if sk >= -1.30:
        return "Coarse Skewed"
    return "Very Coarse Skewed"


def _sk_moments_geo(sk: float) -> str:
    if sk <= -1.30:
        return "Very Fine Skewed"
    if sk <= -0.43:
        return "Fine Skewed"
    if sk < 0.43:
        return "Symmetrical"
    if sk < 1.30:
        return "Coarse Skewed"
    return "Very Coarse Skewed"


def _kurtosis_fw(kg: float) -> str:
    if kg < 0.67:
        return "Very Platykurtic"
    if kg < 0.90:
        return "Platykurtic"
    if kg < 1.11:
        return "Mesokurtic"
    if kg < 1.50:
        return "Leptokurtic"
    if kg < 3.00:
        return "Very Leptokurtic"
    return "Extremely Leptokurtic"


def _kurtosis_moments_log(k: float) -> str:
    if k < 1.70:
        return "Very Platykurtic"
    if k < 2.55:
        return "Platykurtic"
    if k < 3.70:
        return "Mesokurtic"
    if k < 7.40:
        return "Leptokurtic"
    return "Very Leptokurtic"


def _kurtosis_moments_geo(k: float) -> str:
    return _kurtosis_moments_log(k)


# ---------------------------------------------------------------------------
# Wentworth grade scale (same for all schemes)
# ---------------------------------------------------------------------------

_WENTWORTH_PHI = [
    (-11.0, "Very Large Boulder"),
    (-10.0, "Large Boulder"),
    (-9.0, "Medium Boulder"),
    (-8.0, "Small Boulder"),
    (-7.0, "Very Small Boulder"),
    (-6.0, "Very Coarse Gravel"),
    (-5.0, "Coarse Gravel"),
    (-4.0, "Medium Gravel"),
    (-3.0, "Fine Gravel"),
    (-2.0, "Very Fine Gravel"),
    (-1.0, "Granule"),
    (0.0, "Very Coarse Sand"),
    (1.0, "Coarse Sand"),
    (2.0, "Medium Sand"),
    (3.0, "Fine Sand"),
    (4.0, "Very Fine Sand"),
    (5.0, "Very Coarse Silt"),
    (6.0, "Coarse Silt"),
    (7.0, "Medium Silt"),
    (8.0, "Fine Silt"),
    (9.0, "Very Fine Silt"),
    (math.inf, "Clay"),
]

_WENTWORTH_UM = [
    (0.002, "Clay"),
    (0.004, "Very Fine Silt"),
    (0.008, "Fine Silt"),
    (0.016, "Medium Silt"),
    (0.031, "Coarse Silt"),
    (0.063, "Very Coarse Silt"),
    (0.125, "Very Fine Sand"),
    (0.250, "Fine Sand"),
    (0.500, "Medium Sand"),
    (1.000, "Coarse Sand"),
    (2.000, "Very Coarse Sand"),
    (4.000, "Very Fine Gravel"),
    (8.000, "Fine Gravel"),
    (16.0, "Medium Gravel"),
    (32.0, "Coarse Gravel"),
    (64.0, "Very Coarse Gravel"),
    (128.0, "Very Small Boulder"),
    (256.0, "Small Boulder"),
    (512.0, "Medium Boulder"),
    (1024.0, "Large Boulder"),
    (2048.0, "Very Large Boulder"),
    (math.inf, "Very Large Boulder+"),
]

# ---------------------------------------------------------------------------
# ISO 14688-1:2017 grade scale — Table 1, European (63 µm) convention.
# Clay/silt/sand boundaries are numerically identical to Wentworth above;
# the standard's distinguishing contribution is standardizing the coarse
# end into fine/medium/coarse gravel, cobbles, boulders, and large boulders
# (no "granule" class, no boulder sub-division beyond "large boulder").
# Values (mm): 0.002 / 0.0063 / 0.02 / 0.063 / 0.2 / 0.63 / 2.0 / 6.3 / 20
# / 63 / 200 / 630.
# ---------------------------------------------------------------------------

_ISO14688_PHI = [
    (-9.30, "Large Boulder"),  # 630 mm
    (-7.64, "Boulder"),  # 200 mm
    (-5.98, "Cobble"),  # 63 mm
    (-4.32, "Coarse Gravel"),  # 20 mm
    (-2.66, "Medium Gravel"),  # 6.3 mm
    (-1.00, "Fine Gravel"),  # 2.0 mm
    (0.67, "Coarse Sand"),  # 0.63 mm
    (2.32, "Medium Sand"),  # 0.2 mm
    (3.99, "Fine Sand"),  # 0.063 mm
    (5.64, "Coarse Silt"),  # 0.02 mm
    (7.31, "Medium Silt"),  # 0.0063 mm
    (8.97, "Fine Silt"),  # 0.002 mm
    (math.inf, "Clay"),
]

_ISO14688_UM = [
    (0.002, "Clay"),
    (0.0063, "Fine Silt"),
    (0.02, "Medium Silt"),
    (0.063, "Coarse Silt"),
    (0.2, "Fine Sand"),
    (0.63, "Medium Sand"),
    (2.0, "Coarse Sand"),
    (6.3, "Fine Gravel"),
    (20.0, "Medium Gravel"),
    (63.0, "Coarse Gravel"),
    (200.0, "Cobble"),
    (630.0, "Boulder"),
    (math.inf, "Large Boulder"),
]


# ---------------------------------------------------------------------------
# Classifier — single object for all classification needs
# ---------------------------------------------------------------------------


class Classifier:
    """Applies all textural classification thresholds for one scheme.

    Args:
        scheme: Which classification scheme to use. Defaults to GRADISTAT.
    """

    def __init__(self, scheme: ClassificationScheme = ClassificationScheme.GRADISTAT) -> None:
        self.scheme = scheme
        self._phi = _SORTING_PHI[scheme]
        self._geo = _SORTING_GEO[scheme]

    # ── Wentworth mean class ───────────────────────────────────────────

    def mean_wentworth_phi(self, mean_phi: float) -> str:
        """Wentworth (or ISO 14688) grade name for a mean grain size in phi units."""
        return classify_mean_wentworth_phi(mean_phi, self.scheme)

    def mean_wentworth_um(self, mean_um: float) -> str:
        """Wentworth (or ISO 14688) grade name for a mean grain size in µm."""
        return classify_mean_wentworth_um(mean_um, self.scheme)

    # ── Sorting ───────────────────────────────────────────────────────

    def sorting_fw_log(self, std: float) -> str:
        """Sorting class name for a Folk & Ward logarithmic (phi) standard deviation."""
        return _pick(self._phi, std)

    def sorting_fw_geo(self, std: float) -> str:
        """Sorting class name for a Folk & Ward geometric standard deviation."""
        return _pick(self._geo, std)

    def sorting_moments_log(self, std: float) -> str:
        """Sorting class name for a logarithmic-moments standard deviation."""
        return _pick(self._phi, std)

    def sorting_moments_geo(self, std: float) -> str:
        """Sorting class name for a geometric-moments standard deviation."""
        return _pick(self._geo, std)

    # ── Skewness ──────────────────────────────────────────────────────

    def skewness_fw_log(self, sk: float) -> str:
        """Skewness class name for a Folk & Ward logarithmic (phi) skewness value."""
        return _sk_fw_phi(sk)

    def skewness_fw_geo(self, sk: float) -> str:
        """Skewness class name for a Folk & Ward geometric skewness value."""
        return _sk_fw_geo(sk)

    def skewness_moments_log(self, sk: float) -> str:
        """Skewness class name for a logarithmic-moments skewness value."""
        return _sk_moments_log(sk)

    def skewness_moments_geo(self, sk: float) -> str:
        """Skewness class name for a geometric-moments skewness value."""
        return _sk_moments_geo(sk)

    # ── Kurtosis ──────────────────────────────────────────────────────

    def kurtosis_fw_log(self, kg: float) -> str:
        """Kurtosis class name for a Folk & Ward logarithmic kurtosis value."""
        return _kurtosis_fw(kg)

    def kurtosis_fw_geo(self, kg: float) -> str:
        """Kurtosis class name for a Folk & Ward geometric kurtosis value."""
        return _kurtosis_fw(kg)

    def kurtosis_moments_log(self, k: float) -> str:
        """Kurtosis class name for a logarithmic-moments kurtosis value."""
        return _kurtosis_moments_log(k)

    def kurtosis_moments_geo(self, k: float) -> str:
        """Kurtosis class name for a geometric-moments kurtosis value."""
        return _kurtosis_moments_geo(k)

    # ── Utility ───────────────────────────────────────────────────────

    @property
    def sorting_class_names(self) -> list[str]:
        """Ordered list of sorting class names (finest → coarsest)."""
        return [name for _, name in self._phi]


# ---------------------------------------------------------------------------
# Standalone functions — kept for backward compatibility and direct use
# ---------------------------------------------------------------------------


def classify_mean_wentworth_phi(
    mean_phi: float, scheme: ClassificationScheme = ClassificationScheme.GRADISTAT
) -> str:
    """Wentworth (or ISO 14688) grade name for a mean grain size in phi units."""
    table = _ISO14688_PHI if scheme == ClassificationScheme.ISO14688 else _WENTWORTH_PHI
    return _pick(table, mean_phi)


def classify_mean_wentworth_um(
    mean_um: float, scheme: ClassificationScheme = ClassificationScheme.GRADISTAT
) -> str:
    """Wentworth (or ISO 14688) grade name for a mean grain size in µm."""
    mean_mm = mean_um / 1000.0
    table = _ISO14688_UM if scheme == ClassificationScheme.ISO14688 else _WENTWORTH_UM
    return _pick(table, mean_mm)


def classify_sorting_fw_log(
    std: float, scheme: ClassificationScheme = ClassificationScheme.GRADISTAT
) -> str:
    """Sorting class name for a Folk & Ward logarithmic (phi) standard deviation."""
    return _pick(_SORTING_PHI[scheme], std)


def classify_skewness_fw_log(skewness: float) -> str:
    """Skewness class name for a Folk & Ward logarithmic (phi) skewness value."""
    return _sk_fw_phi(skewness)


def classify_kurtosis_fw_log(kurtosis: float) -> str:
    """Kurtosis class name for a Folk & Ward logarithmic kurtosis value."""
    return _kurtosis_fw(kurtosis)


def classify_sorting_fw_geo(
    std: float, scheme: ClassificationScheme = ClassificationScheme.GRADISTAT
) -> str:
    """Sorting class name for a Folk & Ward geometric standard deviation."""
    return _pick(_SORTING_GEO[scheme], std)


def classify_skewness_fw_geo(skewness: float) -> str:
    """Skewness class name for a Folk & Ward geometric skewness value."""
    return _sk_fw_geo(skewness)


def classify_kurtosis_fw_geo(kurtosis: float) -> str:
    """Kurtosis class name for a Folk & Ward geometric kurtosis value."""
    return _kurtosis_fw(kurtosis)


def classify_sorting_moments_geo(
    std: float, scheme: ClassificationScheme = ClassificationScheme.GRADISTAT
) -> str:
    """Sorting class name for a geometric-moments standard deviation."""
    return _pick(_SORTING_GEO[scheme], std)


def classify_skewness_moments_geo(skewness: float) -> str:
    """Skewness class name for a geometric-moments skewness value."""
    return _sk_moments_geo(skewness)


def classify_kurtosis_moments_geo(kurtosis: float) -> str:
    """Kurtosis class name for a geometric-moments kurtosis value."""
    return _kurtosis_moments_geo(kurtosis)


def classify_sorting_moments_log(
    std: float, scheme: ClassificationScheme = ClassificationScheme.GRADISTAT
) -> str:
    """Sorting class name for a logarithmic-moments standard deviation."""
    return _pick(_SORTING_PHI[scheme], std)


def classify_skewness_moments_log(skewness: float) -> str:
    """Skewness class name for a logarithmic-moments skewness value."""
    return _sk_moments_log(skewness)


def classify_kurtosis_moments_log(kurtosis: float) -> str:
    """Kurtosis class name for a logarithmic-moments kurtosis value."""
    return _kurtosis_moments_log(kurtosis)
