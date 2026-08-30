"""Tests for sedstat.core.classification."""

import math

import pytest

from sedstat.core.classification import (
    ClassificationScheme,
    Classifier,
    _pick,
    classify_kurtosis_fw_geo,
    classify_kurtosis_fw_log,
    classify_kurtosis_moments_geo,
    classify_kurtosis_moments_log,
    classify_mean_wentworth_phi,
    classify_mean_wentworth_um,
    classify_skewness_fw_geo,
    classify_skewness_fw_log,
    classify_skewness_moments_geo,
    classify_skewness_moments_log,
    classify_sorting_fw_geo,
    classify_sorting_fw_log,
    classify_sorting_moments_geo,
    classify_sorting_moments_log,
)


class TestSortingFwLog:
    """Table IId sorting thresholds."""

    @pytest.mark.parametrize(
        "std, expected",
        [
            (0.30, "Very Well Sorted"),
            (0.35, "Well Sorted"),
            (0.49, "Well Sorted"),
            (0.50, "Moderately Well Sorted"),
            (0.69, "Moderately Well Sorted"),
            (0.70, "Moderately Sorted"),
            (0.99, "Moderately Sorted"),
            (1.00, "Poorly Sorted"),
            (1.99, "Poorly Sorted"),
            (2.00, "Very Poorly Sorted"),
            (3.99, "Very Poorly Sorted"),
            (4.00, "Extremely Poorly Sorted"),
            (9.99, "Extremely Poorly Sorted"),
        ],
    )
    def test_boundaries(self, std, expected):
        """classify_sorting_fw_log() maps a std value to its expected sorting label.

        Args:
            std: Folk & Ward logarithmic (phi) standard deviation to classify.
            expected: The sorting class label classify_sorting_fw_log() must return.
        """
        assert classify_sorting_fw_log(std) == expected


class TestSortingFwGeo:
    """Table IIe sorting thresholds (geometric)."""

    @pytest.mark.parametrize(
        "std, expected",
        [
            (1.26, "Very Well Sorted"),
            (1.27, "Well Sorted"),
            (1.40, "Well Sorted"),
            (1.41, "Moderately Well Sorted"),
            (1.61, "Moderately Well Sorted"),
            (1.62, "Moderately Sorted"),
            (1.99, "Moderately Sorted"),
            (2.00, "Poorly Sorted"),
            (3.99, "Poorly Sorted"),
            (4.00, "Very Poorly Sorted"),
            (15.9, "Very Poorly Sorted"),
            (16.0, "Extremely Poorly Sorted"),
        ],
    )
    def test_boundaries(self, std, expected):
        """classify_sorting_fw_geo() maps a std value to its expected sorting label.

        Args:
            std: Folk & Ward geometric std (dimensionless ratio) to classify.
            expected: The sorting class label classify_sorting_fw_geo() must return.
        """
        assert classify_sorting_fw_geo(std) == expected


class TestSkewnessFwLog:
    """Table IId skewness thresholds (phi, positive = fine skewed)."""

    @pytest.mark.parametrize(
        "sk, expected",
        [
            (0.31, "Very Fine Skewed"),
            (0.30, "Fine Skewed"),  # boundary: > 0.30 required, so 0.30 is Fine Skewed
            (0.11, "Fine Skewed"),
            (0.10, "Symmetrical"),  # boundary: > 0.10 required, so 0.10 is Symmetrical
            (0.00, "Symmetrical"),
            (-0.10, "Symmetrical"),  # >= -0.10
            (-0.11, "Coarse Skewed"),
            (-0.30, "Coarse Skewed"),  # >= -0.30
            (-0.31, "Very Coarse Skewed"),
            (-1.00, "Very Coarse Skewed"),
        ],
    )
    def test_boundaries(self, sk, expected):
        """classify_skewness_fw_log() maps a skewness value to its expected label.

        Args:
            sk: Folk & Ward logarithmic (phi) skewness to classify.
            expected: The skewness class label classify_skewness_fw_log() must return.
        """
        assert classify_skewness_fw_log(sk) == expected


class TestKurtosisFwLog:
    """Table IId kurtosis thresholds."""

    @pytest.mark.parametrize(
        "k, expected",
        [
            (0.66, "Very Platykurtic"),
            (0.67, "Platykurtic"),
            (0.89, "Platykurtic"),
            (0.90, "Mesokurtic"),
            (1.10, "Mesokurtic"),
            (1.11, "Leptokurtic"),
            (1.49, "Leptokurtic"),
            (1.50, "Very Leptokurtic"),
            (2.99, "Very Leptokurtic"),
            (3.00, "Extremely Leptokurtic"),
            (5.00, "Extremely Leptokurtic"),
        ],
    )
    def test_boundaries(self, k, expected):
        """classify_kurtosis_fw_log() maps a kurtosis value to its expected label.

        Args:
            k: Folk & Ward logarithmic (phi) kurtosis to classify.
            expected: The kurtosis class label classify_kurtosis_fw_log() must return.
        """
        assert classify_kurtosis_fw_log(k) == expected


class TestWentworthPhi:
    """Grain size classification using phi scale."""

    @pytest.mark.parametrize(
        "phi, expected",
        [
            (2.455, "Fine Sand"),  # Mablethorpe sample
            (1.5, "Medium Sand"),
            (3.5, "Very Fine Sand"),
            (4.5, "Very Coarse Silt"),  # 4-5 phi = Very Coarse Silt (31-63 µm)
            (5.5, "Coarse Silt"),  # 5-6 phi = Coarse Silt (15.6-31 µm)
            (9.5, "Clay"),
            (-1.5, "Granule"),  # 2–4 mm → granule zone, not sand
        ],
    )
    def test_classification(self, phi, expected):
        """classify_mean_wentworth_phi() maps a phi value to its expected Wentworth grade.

        Args:
            phi: Mean grain size in phi units to classify.
            expected: The Wentworth grade name classify_mean_wentworth_phi() must return.
        """
        assert classify_mean_wentworth_phi(phi) == expected


class TestWentworthUm:
    """Grain size classification using µm scale."""

    @pytest.mark.parametrize(
        "um, expected",
        [
            (182.5, "Fine Sand"),  # Mablethorpe folk_ward_geo mean
            (350.0, "Medium Sand"),
            (50.0, "Very Coarse Silt"),  # 31-63 µm = Very Coarse Silt
            (22.0, "Coarse Silt"),  # 15.6-31 µm = Coarse Silt
            (1.5, "Clay"),
            (800.0, "Coarse Sand"),
        ],
    )
    def test_classification(self, um, expected):
        """classify_mean_wentworth_um() maps a µm value to its expected Wentworth grade.

        Args:
            um: Mean grain size in µm to classify.
            expected: The Wentworth grade name classify_mean_wentworth_um() must return.
        """
        assert classify_mean_wentworth_um(um) == expected


class TestISO14688WentworthPhi:
    """ISO 14688-1:2017 Table 1 grade names (phi scale).

    Same clay/silt/sand naming range as Wentworth but a different,
    officially standardized coarse-end subdivision (no "granule" class;
    cobble/boulder/large boulder instead of Wentworth's five boulder
    sub-classes). Boundaries (mm): 0.002/0.0063/0.02/0.063/0.2/0.63/2.0/
    6.3/20/63/200/630.
    """

    @pytest.mark.parametrize(
        "phi, expected",
        [
            (2.455, "Fine Sand"),  # Mablethorpe sample — same name as Wentworth
            (1.0, "Medium Sand"),  # 0.667 < 1.0 < 2.322
            (-1.5, "Fine Gravel"),  # -2.655 < -1.5 < -1.0 — no "granule" in ISO
            (-5.0, "Coarse Gravel"),  # -5.977 < -5.0 < -4.322
            (-6.5, "Cobble"),  # -7.644 < -6.5 < -5.977
            (9.5, "Clay"),
            (-10.0, "Large Boulder"),  # < -9.30
        ],
    )
    def test_classification(self, phi, expected):
        """classify_mean_wentworth_phi(ISO14688) maps a phi value to its expected ISO grade.

        Args:
            phi: Mean grain size in phi units to classify.
            expected: The ISO 14688-1 grade name that must be returned.
        """
        assert classify_mean_wentworth_phi(phi, ClassificationScheme.ISO14688) == expected

    def test_default_scheme_still_uses_wentworth(self):
        """The default scheme argument still resolves to Wentworth, not ISO14688.

        -1.5 phi is "Granule" under Wentworth but "Fine Gravel" under ISO —
        confirms the default argument didn't change existing behaviour.
        """
        assert classify_mean_wentworth_phi(-1.5) == "Granule"


class TestISO14688WentworthUm:
    """ISO 14688-1:2017 Table 1 grade names (µm scale)."""

    @pytest.mark.parametrize(
        "um, expected",
        [
            (182.5, "Fine Sand"),
            (
                50.0,
                "Coarse Silt",
            ),  # 0.02 < 0.05 < 0.063 mm — ISO has no "very coarse silt"
            (1.5, "Clay"),
            (100_000.0, "Cobble"),  # 100 mm — Wentworth would say "Very Small Boulder"
        ],
    )
    def test_classification(self, um, expected):
        """classify_mean_wentworth_um(ISO14688) maps a µm value to its expected ISO grade.

        Args:
            um: Mean grain size in µm to classify.
            expected: The ISO 14688-1 grade name that must be returned.
        """
        assert classify_mean_wentworth_um(um, ClassificationScheme.ISO14688) == expected

    def test_default_scheme_still_uses_wentworth(self):
        assert classify_mean_wentworth_um(100_000.0) == "Very Small Boulder"


class TestISO14688Classifier:
    """Classifier(ISO14688) end-to-end: naming picks ISO table, sorting reuses GRADISTAT."""

    def test_mean_wentworth_um_uses_iso_table(self):
        clf = Classifier(ClassificationScheme.ISO14688)
        assert clf.mean_wentworth_um(100_000.0) == "Cobble"

    def test_mean_wentworth_phi_uses_iso_table(self):
        clf = Classifier(ClassificationScheme.ISO14688)
        assert clf.mean_wentworth_phi(-6.5) == "Cobble"

    def test_sorting_class_names_match_gradistat(self):
        iso = Classifier(ClassificationScheme.ISO14688)
        gradistat = Classifier(ClassificationScheme.GRADISTAT)
        assert iso.sorting_class_names == gradistat.sorting_class_names

    def test_sorting_fw_log_matches_gradistat(self):
        clf = Classifier(ClassificationScheme.ISO14688)
        assert clf.sorting_fw_log(0.30) == "Very Well Sorted"


# ---------------------------------------------------------------------------
# _pick fallback branch (line 57): value that never satisfies "< upper",
# even against the trailing math.inf threshold — only possible with NaN.
# ---------------------------------------------------------------------------


class TestPickFallback:
    """_pick() falls back to the last threshold entry for a NaN value."""

    def test_nan_value_falls_back_to_last_entry(self):
        thresholds = [(1.0, "a"), (2.0, "b"), (math.inf, "c")]
        assert _pick(thresholds, float("nan")) == "c"


# ---------------------------------------------------------------------------
# Standalone geometric/moments wrapper functions (backward-compat API) —
# each simply delegates to the GRADISTAT scheme, exercised with one value
# from each threshold band.
# ---------------------------------------------------------------------------


class TestStandaloneGeometricAndMomentsWrappers:
    """Standalone geometric/moments classify_* wrappers delegate to the GRADISTAT scheme."""

    def test_skewness_fw_geo(self):
        assert classify_skewness_fw_geo(-0.5) == "Very Fine Skewed"
        assert classify_skewness_fw_geo(0.0) == "Symmetrical"

    def test_kurtosis_fw_geo(self):
        assert classify_kurtosis_fw_geo(0.5) == "Very Platykurtic"

    def test_sorting_moments_geo(self):
        assert classify_sorting_moments_geo(1.0) == "Very Well Sorted"

    def test_skewness_moments_geo(self):
        assert classify_skewness_moments_geo(-2.0) == "Very Fine Skewed"

    def test_kurtosis_moments_geo(self):
        assert classify_kurtosis_moments_geo(1.0) == "Very Platykurtic"

    def test_sorting_moments_log(self):
        assert classify_sorting_moments_log(0.30) == "Very Well Sorted"

    def test_skewness_moments_log(self):
        assert classify_skewness_moments_log(0.0) == "Symmetrical"

    def test_kurtosis_moments_log(self):
        assert classify_kurtosis_moments_log(1.0) == "Very Platykurtic"


class TestStandaloneSortingFunctionsAcceptScheme:
    """The four sorting wrappers take an optional scheme override.

    Bug fix: they used to hardcode GRADISTAT unconditionally, even though
    sorting thresholds genuinely differ for FOLK_WARD_1957 — so it was
    impossible to get correct sorting labels for that scheme through these
    functions. The default stays GRADISTAT for backward compatibility.
    """

    def test_sorting_fw_log_default_is_gradistat(self):
        assert classify_sorting_fw_log(0.30) == "Very Well Sorted"

    def test_sorting_fw_log_folk_ward_1957(self):
        assert classify_sorting_fw_log(0.30, ClassificationScheme.FOLK_WARD_1957) == "Well Sorted"

    def test_sorting_fw_geo_folk_ward_1957(self):
        assert classify_sorting_fw_geo(1.20, ClassificationScheme.FOLK_WARD_1957) == "Well Sorted"

    def test_sorting_moments_geo_folk_ward_1957(self):
        assert (
            classify_sorting_moments_geo(1.20, ClassificationScheme.FOLK_WARD_1957) == "Well Sorted"
        )

    def test_sorting_moments_log_folk_ward_1957(self):
        assert (
            classify_sorting_moments_log(0.30, ClassificationScheme.FOLK_WARD_1957) == "Well Sorted"
        )


# ---------------------------------------------------------------------------
# Internal skewness/kurtosis threshold functions used by the moments
# methods — cover the remaining Coarse/Very-Coarse and Leptokurtic/
# Very-Leptokurtic branches not reached by the wrapper smoke tests above.
# ---------------------------------------------------------------------------


class TestMomentsLogSkewnessBoundaries:
    """classify_skewness_moments_log() threshold boundaries, including the Coarse bands."""

    @pytest.mark.parametrize(
        "sk, expected",
        [
            (1.31, "Very Fine Skewed"),
            (1.30, "Fine Skewed"),
            (0.44, "Fine Skewed"),
            (0.43, "Symmetrical"),
            (0.0, "Symmetrical"),
            (-0.43, "Symmetrical"),
            (-0.44, "Coarse Skewed"),
            (-1.30, "Coarse Skewed"),
            (-1.31, "Very Coarse Skewed"),
        ],
    )
    def test_boundaries(self, sk, expected):
        """classify_skewness_moments_log() maps a skewness value to its expected label.

        Args:
            sk: Moments-method logarithmic skewness to classify.
            expected: The skewness class label classify_skewness_moments_log() must return.
        """
        assert classify_skewness_moments_log(sk) == expected


class TestMomentsGeoSkewnessBoundaries:
    """classify_skewness_moments_geo() threshold boundaries, including the Coarse bands."""

    @pytest.mark.parametrize(
        "sk, expected",
        [
            (-1.31, "Very Fine Skewed"),
            (-1.30, "Very Fine Skewed"),
            (-1.29, "Fine Skewed"),
            (-0.44, "Fine Skewed"),
            (-0.43, "Fine Skewed"),
            (-0.42, "Symmetrical"),
            (0.42, "Symmetrical"),
            (0.43, "Coarse Skewed"),
            (1.29, "Coarse Skewed"),
            (1.30, "Very Coarse Skewed"),
        ],
    )
    def test_boundaries(self, sk, expected):
        """classify_skewness_moments_geo() maps a skewness value to its expected label.

        Args:
            sk: Moments-method geometric skewness to classify.
            expected: The skewness class label classify_skewness_moments_geo() must return.
        """
        assert classify_skewness_moments_geo(sk) == expected


class TestFwGeoSkewnessBoundaries:
    """classify_skewness_fw_geo() threshold boundaries, including the Coarse bands."""

    @pytest.mark.parametrize(
        "sk, expected",
        [
            (-0.31, "Very Fine Skewed"),
            (-0.30, "Fine Skewed"),
            (-0.11, "Fine Skewed"),
            (-0.10, "Symmetrical"),
            (0.10, "Symmetrical"),
            (0.11, "Coarse Skewed"),
            (0.30, "Coarse Skewed"),
            (0.31, "Very Coarse Skewed"),
        ],
    )
    def test_boundaries(self, sk, expected):
        """classify_skewness_fw_geo() maps a skewness value to its expected label.

        Args:
            sk: Folk & Ward geometric skewness to classify.
            expected: The skewness class label classify_skewness_fw_geo() must return.
        """
        assert classify_skewness_fw_geo(sk) == expected


class TestMomentsKurtosisBoundaries:
    """classify_kurtosis_moments_log() threshold boundaries, including the Leptokurtic bands."""

    @pytest.mark.parametrize(
        "k, expected",
        [
            (1.69, "Very Platykurtic"),
            (1.70, "Platykurtic"),
            (2.54, "Platykurtic"),
            (2.55, "Mesokurtic"),
            (3.69, "Mesokurtic"),
            (3.70, "Leptokurtic"),
            (7.39, "Leptokurtic"),
            (7.40, "Very Leptokurtic"),
            (20.0, "Very Leptokurtic"),
        ],
    )
    def test_boundaries(self, k, expected):
        """classify_kurtosis_moments_log() maps a kurtosis value to its expected label.

        Args:
            k: Moments-method logarithmic kurtosis to classify.
            expected: The kurtosis class label classify_kurtosis_moments_log() must return.
        """
        assert classify_kurtosis_moments_log(k) == expected


# ---------------------------------------------------------------------------
# Classifier convenience object
# ---------------------------------------------------------------------------


class TestClassifier:
    """Classifier's scheme-bound convenience API: class names and delegating methods."""

    def test_sorting_class_names_gradistat(self):
        clf = Classifier(ClassificationScheme.GRADISTAT)
        names = clf.sorting_class_names
        assert names[0] == "Very Well Sorted"
        assert names[-1] == "Extremely Poorly Sorted"

    def test_mean_wentworth_phi_delegates(self):
        clf = Classifier()
        assert clf.mean_wentworth_phi(2.455) == "Fine Sand"

    def test_mean_wentworth_um_delegates(self):
        clf = Classifier()
        assert clf.mean_wentworth_um(182.5) == "Fine Sand"
