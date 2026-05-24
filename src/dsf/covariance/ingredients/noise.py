"""Noise terms for projected DeltaSigma covariance calculations.

This module defines the discreteness and shape-noise terms that enter simple
DeltaSigma covariance models. The functions distinguish between lens shot noise,
which comes from the finite number of lens galaxies, and source shape noise,
which comes from the intrinsic ellipticity scatter of source galaxies.

The returned quantities are intended to be combined with survey geometry,
redshift-window factors, and matter power spectra when building covariance blocks
for galaxy-matter lensing and related projected observables.
"""

from __future__ import annotations

from dsf.utils.converters import arcmin2_per_steradian

__all__ = [
    "angular_shape_noise",
    "projected_shape_noise",
    "shot_noise",
]


def shot_noise(n_gal: float) -> float:
    """Return lens-galaxy shot noise from a three-dimensional number density.

    Shot noise describes the discreteness noise from representing the lens
    galaxy field with a finite number of galaxies. A denser lens sample has a
    smaller shot-noise contribution, while a sparse lens sample has a larger
    one.

    Args:
        n_gal: Comoving three-dimensional lens galaxy number density in
            ``(h / Mpc)^3``.

    Returns:
        Poisson shot-noise amplitude in ``(Mpc / h)^3``.
    """
    if n_gal <= 0.0:
        raise ValueError("n_gal must be positive.")

    return 1.0 / n_gal


def angular_shape_noise(
    sigma_e: float,
    n_eff_arcmin2: float,
) -> float:
    """Return source shape noise for an angular shear field.

    Shape noise describes the variance from the intrinsic ellipticity scatter of
    source galaxies. This is the angular form used when the source density is
    treated as a number per solid angle.

    Args:
        sigma_e: Per-component intrinsic ellipticity dispersion.
        n_eff_arcmin2: Effective angular source galaxy density in ``arcmin^-2``.

    Returns:
        Angular shape-noise amplitude in steradian units.
    """
    if sigma_e <= 0.0:
        raise ValueError("sigma_e must be positive.")
    if n_eff_arcmin2 <= 0.0:
        raise ValueError("n_eff_arcmin2 must be positive.")

    n_eff_sr = n_eff_arcmin2 * arcmin2_per_steradian()

    return sigma_e**2 / n_eff_sr


def projected_shape_noise(
    sigma_e: float,
    n_eff_arcmin2: float,
    chi_eff: float,
) -> float:
    """Return source shape noise in projected comoving coordinates.

    This is the shape-noise contribution expressed on the transverse comoving
    plane at an effective lens distance. It is useful for DeltaSigma covariance
    calculations written in projected physical or comoving coordinates rather
    than purely angular coordinates.

    Args:
        sigma_e: Per-component intrinsic ellipticity dispersion.
        n_eff_arcmin2: Effective angular source galaxy density in ``arcmin^-2``.
        chi_eff: Effective comoving lens distance in ``Mpc / h``.

    Returns:
        Projected shape-noise amplitude in ``(Mpc / h)^2``.
    """
    if sigma_e <= 0.0:
        raise ValueError("sigma_e must be positive.")
    if n_eff_arcmin2 <= 0.0:
        raise ValueError("n_eff_arcmin2 must be positive.")
    if chi_eff <= 0.0:
        raise ValueError("chi_eff must be positive.")

    n_eff_sr = n_eff_arcmin2 * arcmin2_per_steradian()
    n_eff_projected = n_eff_sr / chi_eff**2

    return sigma_e**2 / n_eff_projected
