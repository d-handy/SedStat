"""Tests for sedstat.core.stokes."""

from __future__ import annotations

import pytest
from sedstat.core.stokes import (
    stokes_settling_diameter_um,
    stokes_settling_time_s,
    water_density_kg_m3,
    water_viscosity_pa_s,
)


class TestWaterViscosity:
    """water_viscosity_pa_s() against known reference values and its temperature trend."""

    def test_20c_matches_reference(self):
        # CRC Handbook / NIST reference: ~1.002 mPa*s at 20 C.
        assert water_viscosity_pa_s(20.0) == pytest.approx(1.002e-3, rel=1e-3)

    def test_25c_matches_reference(self):
        # CRC Handbook / NIST reference: ~0.890 mPa*s at 25 C.
        assert water_viscosity_pa_s(25.0) == pytest.approx(0.890e-3, rel=1e-3)

    def test_monotonically_decreases_with_temperature(self):
        temps = [0.0, 10.0, 20.0, 30.0, 40.0]
        viscosities = [water_viscosity_pa_s(t) for t in temps]
        assert viscosities == sorted(viscosities, reverse=True)


class TestWaterViscosityValidation:
    """water_viscosity_pa_s() enforces its [0, 40] °C temp_c range."""

    def test_below_min_raises(self):
        with pytest.raises(ValueError, match="temp_c"):
            water_viscosity_pa_s(-0.01)

    def test_above_max_raises(self):
        with pytest.raises(ValueError, match="temp_c"):
            water_viscosity_pa_s(40.01)

    def test_min_boundary_does_not_raise(self):
        water_viscosity_pa_s(0.0)

    def test_max_boundary_does_not_raise(self):
        water_viscosity_pa_s(40.0)


class TestWaterDensity:
    """water_density_kg_m3() against known reference values, including the 4°C density anomaly."""

    def test_density_maximum_near_4c(self):
        # Water's known density anomaly: maximum density is near 4 C, not 0 C.
        assert water_density_kg_m3(4.0) > water_density_kg_m3(0.0)
        assert water_density_kg_m3(4.0) > water_density_kg_m3(20.0)

    def test_0c_matches_reference(self):
        assert water_density_kg_m3(0.0) == pytest.approx(999.84, abs=0.01)

    def test_20c_matches_reference(self):
        assert water_density_kg_m3(20.0) == pytest.approx(998.21, abs=0.01)

    def test_40c_matches_reference(self):
        assert water_density_kg_m3(40.0) == pytest.approx(992.22, abs=0.01)


class TestWaterDensityValidation:
    """water_density_kg_m3() enforces its [0, 40] °C temp_c range."""

    def test_below_min_raises(self):
        with pytest.raises(ValueError, match="temp_c"):
            water_density_kg_m3(-0.01)

    def test_above_max_raises(self):
        with pytest.raises(ValueError, match="temp_c"):
            water_density_kg_m3(40.01)

    def test_min_boundary_does_not_raise(self):
        water_density_kg_m3(0.0)

    def test_max_boundary_does_not_raise(self):
        water_density_kg_m3(40.0)


class TestStokesSettlingDiameter:
    """stokes_settling_diameter_um()'s response to density, time, and depth.

    Also covers its round-trip with stokes_settling_time_s().
    """

    def test_denser_particle_settles_faster_smaller_diameter(self):
        """A denser particle needs a smaller diameter to match the same settling event.

        At fixed time/depth, a denser particle needs a *smaller* diameter
        to have just settled past depth_cm in time_s (it falls faster per
        unit size), so the equivalent diameter computed for a fixed
        settling event shrinks as density grows.
        """
        d_light = stokes_settling_diameter_um(3600, 10, 20, particle_density_g_cm3=2.2)
        d_heavy = stokes_settling_diameter_um(3600, 10, 20, particle_density_g_cm3=2.65)
        assert d_heavy < d_light

    def test_longer_time_gives_smaller_diameter(self):
        d_short = stokes_settling_diameter_um(1800, 10, 20)
        d_long = stokes_settling_diameter_um(3600, 10, 20)
        assert d_long < d_short

    def test_deeper_draw_gives_larger_diameter(self):
        d_shallow = stokes_settling_diameter_um(3600, 5, 20)
        d_deep = stokes_settling_diameter_um(3600, 20, 20)
        assert d_deep > d_shallow

    def test_typical_clay_silt_boundary_is_plausible(self):
        """A realistic 8-hour pipette draw yields a silt/clay-sized diameter.

        A ~8 hour draw at 10 cm, 20 C, quartz density should land in the
        silt/clay range (order of a few microns), not sand-sized.
        """
        d = stokes_settling_diameter_um(8 * 3600, 10, 20.0)
        assert 0.5 < d < 20.0

    def test_round_trip_with_settling_time(self):
        d = stokes_settling_diameter_um(3600, 10, 20)
        t = stokes_settling_time_s(d, 10, 20)
        assert t == pytest.approx(3600, rel=1e-9)


class TestStokesSettlingDiameterValidation:
    """stokes_settling_diameter_um() validates time, depth, temperature, and particle density."""

    def test_non_positive_time_raises(self):
        with pytest.raises(ValueError, match="time_s"):
            stokes_settling_diameter_um(0, 10, 20)

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="time_s"):
            stokes_settling_diameter_um(-1, 10, 20)

    def test_non_positive_depth_raises(self):
        with pytest.raises(ValueError, match="depth_cm"):
            stokes_settling_diameter_um(3600, 0, 20)

    def test_temp_out_of_range_raises(self):
        with pytest.raises(ValueError, match="temp_c"):
            stokes_settling_diameter_um(3600, 10, 100)

    def test_particle_lighter_than_water_raises(self):
        with pytest.raises(ValueError, match="particle_density_g_cm3"):
            stokes_settling_diameter_um(3600, 10, 20, particle_density_g_cm3=0.9)

    def test_particle_density_equal_to_water_raises(self):
        rho_water = water_density_kg_m3(20.0) / 1000.0
        with pytest.raises(ValueError, match="particle_density_g_cm3"):
            stokes_settling_diameter_um(3600, 10, 20, particle_density_g_cm3=rho_water)


class TestStokesSettlingTimeValidation:
    """stokes_settling_time_s() rejects a non-positive diameter."""

    def test_non_positive_diameter_raises(self):
        with pytest.raises(ValueError, match="diameter_um"):
            stokes_settling_time_s(0, 10, 20)


class TestAgainstKrumbeinPettijohn1938:
    """Cross-check against a classic, independently-sourced reference value.

    Krumbein, W.C. & Pettijohn, F.J. (1938). Manual of Sedimentary
    Petrography, p. 102: a 0.05 mm (50 µm) quartz sphere settles at
    0.196 cm/s at 15 C and 0.223 cm/s at 20 C. Also cited in a University
    of Maryland G342 Sedimentation & Stratigraphy lab handout
    (geol.umd.edu/~kaufman/ppt/G342_06/13Feb06_lab.doc), which gives the
    equivalent simplified formula v[cm/s] = 3.59e4 * r[cm]^2 (20 C,
    particle density 2.65 g/cm^3).

    The module has no settling-velocity function directly, so the check
    runs through stokes_settling_diameter_um: pick a settling depth/time
    pair whose ratio equals the reference velocity, and confirm the
    recovered diameter is 50 µm.
    """

    @pytest.mark.parametrize(
        "temp_c, v_cm_s",
        [
            (15.0, 0.196),
            (20.0, 0.223),
        ],
    )
    def test_reference_velocity_recovers_50um_quartz_sphere(self, temp_c, v_cm_s):
        """The reference settling velocity, run back through the diameter formula, recovers 50 µm.

        Args:
            temp_c: Water temperature (°C) for the reference measurement.
            v_cm_s: Reference settling velocity (cm/s) at temp_c for the 50 µm sphere.
        """
        depth_cm = 10.0
        v_m_s = v_cm_s / 100.0
        time_s = (depth_cm / 100.0) / v_m_s

        diameter_um = stokes_settling_diameter_um(
            time_s, depth_cm, temp_c, particle_density_g_cm3=2.65
        )

        # ~0.5% agreement observed by hand for both temperatures; allow 2%
        # to cover 1938-era table rounding without masking a real bug (a
        # dropped factor of 2 or 4 would miss by 41% or 75%, not 2%).
        assert diameter_um == pytest.approx(50.0, rel=0.02)
