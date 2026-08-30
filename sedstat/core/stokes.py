"""Stokes' law settling-diameter calculation for the pipette method.

Computes the equivalent settling diameter from raw physical inputs (time,
depth, water temperature, particle density), for labs that record draw
conditions rather than a pre-computed diameter from a published
settling-time table. Method: Guy (1969); ISO 13317-2:2001, fixed pipette
method.

Limitation: no drag-coefficient correction. ISO 13317 sets the maximum
Reynolds number for Stokes law at 0.25, corresponding to 65 µm for quartz
in water at 20 C (Goossens, 2008) — the coarse-silt/fine-sand boundary
where pipette analysis normally stops anyway.
"""

from __future__ import annotations

import math

_G = 9.80665  # standard gravity, m/s^2

_TEMP_MIN_C = 0.0
_TEMP_MAX_C = 40.0  # typical sedimentology lab water-bath range

_DEFAULT_PARTICLE_DENSITY_G_CM3 = 2.65  # quartz


def _validate_temp_c(temp_c: float) -> None:
    if not (_TEMP_MIN_C <= temp_c <= _TEMP_MAX_C):
        raise ValueError(
            f"temp_c must be between {_TEMP_MIN_C:.0f} and {_TEMP_MAX_C:.0f} "
            f"(the range the viscosity/density correlations below are valid "
            f"for); got {temp_c}."
        )


def water_viscosity_pa_s(temp_c: float) -> float:
    """Dynamic viscosity of water (Pa*s) at *temp_c* (deg C).

    Vogel/Andrade-type correlation:
        eta(T) = 2.414e-5 * 10^(247.8 / (T + 133.15))
    Matches the Kestin, Sokolov & Wakeham (1978) reference values closely
    from ~10-40 C (e.g. ~1.002 mPa*s at 20 C, ~0.890 mPa*s at 25 C);
    accuracy degrades toward 0 C (~2% low against their ~1.792 mPa*s
    reference value). See REFERENCES.md.
    """
    _validate_temp_c(temp_c)
    return 2.414e-5 * 10 ** (247.8 / (temp_c + 133.15))


def water_density_kg_m3(temp_c: float) -> float:
    """Density of pure water (kg/m^3) at *temp_c* (deg C).

    Millero & Poisson (1981) / UNESCO (1981) pure-water polynomial, valid
    0-40 C. Reproduces the known non-monotonic shape (density maximum
    ~999.97 kg/m^3 near 4 C): density(4) > density(0) and density(4) >
    density(20).
    """
    _validate_temp_c(temp_c)
    t = temp_c
    return (
        999.842594
        + 6.793952e-2 * t
        - 9.095290e-3 * t**2
        + 1.001685e-4 * t**3
        - 1.120083e-6 * t**4
        + 6.536336e-9 * t**5
    )


def _validate_settling_inputs(
    time_s: float, depth_cm: float, temp_c: float, particle_density_g_cm3: float
) -> None:
    if time_s <= 0:
        raise ValueError(f"time_s must be positive; got {time_s}.")
    if depth_cm <= 0:
        raise ValueError(f"depth_cm must be positive; got {depth_cm}.")
    _validate_temp_c(temp_c)
    rho_f_g_cm3 = water_density_kg_m3(temp_c) / 1000.0
    if particle_density_g_cm3 <= rho_f_g_cm3:
        raise ValueError(
            f"particle_density_g_cm3 ({particle_density_g_cm3}) must exceed "
            f"the water density at {temp_c} C ({rho_f_g_cm3:.4f} g/cm3) — "
            "a particle this light would not settle by gravity."
        )


def stokes_settling_diameter_um(
    time_s: float,
    depth_cm: float,
    temp_c: float,
    particle_density_g_cm3: float = _DEFAULT_PARTICLE_DENSITY_G_CM3,
) -> float:
    """Equivalent settling diameter (µm) for a pipette draw.

    Args:
        time_s: Elapsed settling time since the suspension was mixed (s).
        depth_cm: Sampling depth below the suspension surface (cm).
        temp_c: Water temperature (deg C), 0-40.
        particle_density_g_cm3: Particle density (g/cm3); defaults to 2.65
            (quartz).

    Returns:
        The equivalent spherical diameter (µm) of the largest particle
        still in suspension at *depth_cm* after *time_s* — i.e. every
        particle finer than this diameter is present in the draw.
    """
    _validate_settling_inputs(time_s, depth_cm, temp_c, particle_density_g_cm3)

    h_m = depth_cm / 100.0
    rho_s = particle_density_g_cm3 * 1000.0
    rho_f = water_density_kg_m3(temp_c)
    eta = water_viscosity_pa_s(temp_c)

    d_m = math.sqrt(18.0 * eta * h_m / (time_s * (rho_s - rho_f) * _G))
    return d_m * 1e6


def stokes_settling_time_s(
    diameter_um: float,
    depth_cm: float,
    temp_c: float,
    particle_density_g_cm3: float = _DEFAULT_PARTICLE_DENSITY_G_CM3,
) -> float:
    """Elapsed time (s) for a particle of *diameter_um* to settle past *depth_cm*.

    The algebraic inverse of :func:`stokes_settling_diameter_um` — lets a
    lab work out how long to wait before drawing a sample targeting a given
    equivalent diameter, and provides a clean round-trip identity for tests.
    """
    if diameter_um <= 0:
        raise ValueError(f"diameter_um must be positive; got {diameter_um}.")
    _validate_settling_inputs(1.0, depth_cm, temp_c, particle_density_g_cm3)

    h_m = depth_cm / 100.0
    rho_s = particle_density_g_cm3 * 1000.0
    rho_f = water_density_kg_m3(temp_c)
    eta = water_viscosity_pa_s(temp_c)
    d_m = diameter_um / 1e6

    return 18.0 * eta * h_m / (d_m**2 * (rho_s - rho_f) * _G)
