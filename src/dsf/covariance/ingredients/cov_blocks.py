r"""Covariance blocks for projected DeltaSigma observables.

This module provides covariance terms for DeltaSigma data vectors built from
galaxy-matter and galaxy-galaxy projected correlations. The functions return
auto- and cross-covariance blocks for lensing-style ``gm`` measurements,
clustering-style ``gg`` measurements, and their joint covariance matrix.

The inputs are assumed to describe a single lens/source bin pair or an
effective lens-bin average: a matter power spectrum, survey-volume factors,
line-of-sight window integrals, noise terms, and the projected-radius binning
used by the DeltaSigma data vector.

Optional tapering applies a smooth window to the low- and high-wavenumber
edges of the input spectra before projection. This reduces numerical ringing
from finite ``k`` coverage and makes the Hankel-projected covariance less
sensitive to abrupt power-spectrum cutoffs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dsf.covariance.projection.hankel_transform import HankelTransform
from dsf.utils.types import ArrayLike
from dsf.utils.validators import (
    as_1d_float_array,
    as_2d_float_array,
    validate_joint_covariance_blocks,
    validate_positive_scalar,
    validate_power_spectrum_inputs,
)

__all__ = [
    "delta_sigma_gm_covariance",
    "delta_sigma_gg_covariance",
    "delta_sigma_gm_gg_cross_covariance",
    "joint_delta_sigma_covariance",
]

DEFAULT_TAPER_KWARGS = {
    "large_k_lower": 10.0,
    "low_k_factor": 1.2,
}


def _covariance_on_requested_radius_grid(
    hankel: HankelTransform,
    r: Any,
    cov: Any,
    rp_bin_edges: ArrayLike | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return covariance on the transform grid or requested radial bins.

    Args:
        hankel: Hankel transform object defining the radial grid and optional
            radial binning.
        r: Projected-radius grid associated with ``cov``.
        cov: Covariance matrix evaluated on ``r``.
        rp_bin_edges: Optional projected-radius bin edges.

    Returns:
        Projected radii and covariance matrix, either on the original transform
        grid or averaged into the requested radial bins.
    """
    r_arr = as_1d_float_array(r, "r")
    cov_arr = as_2d_float_array(cov, "cov")

    if cov_arr.shape != (r_arr.size, r_arr.size):
        raise ValueError(
            "cov must have shape (r.size, r.size). "
            f"Got cov.shape={cov_arr.shape} and r.size={r_arr.size}."
        )

    if rp_bin_edges is None:
        return r_arr, cov_arr

    return hankel.bin_radial_matrix(r_arr, cov_arr, rp_bin_edges)


def delta_sigma_gm_covariance(
    hankel: HankelTransform,
    k: ArrayLike,
    pk: ArrayLike,
    *,
    galaxy_bias: float,
    omega_m: float,
    rho_crit: float,
    delta_pi_gm_squared_window: float,
    sigma_crit_squared_average: float,
    shape_noise: float,
    shot_noise: float,
    volume: float,
    rp_bin_edges: ArrayLike | None = None,
    order: int = 2,
    taper: bool = True,
    taper_kwargs: dict | None = None,
    gkgk_taper: bool = False,
    gkgk_taper_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the auto-covariance of the galaxy-matter DeltaSigma signal.

    This block describes the uncertainty of a DeltaSigma measurement from lens
    galaxy positions and source galaxy shapes. It combines the covariance from
    lens-density and shear fluctuations with the covariance from their shared
    matter fluctuations, including lens shot noise, source shape noise, survey
    volume, and squared line-of-sight window weighting.

    Args:
        hankel: Hankel transform object used to project Fourier-space covariance
            terms into projected-radius space.
        k: Wavenumber grid in ``h / Mpc``.
        pk: Lens-averaged matter power spectrum evaluated on ``k``.
        galaxy_bias: Linear galaxy bias of the lens sample.
        omega_m: Present-day matter density parameter.
        rho_crit: Critical density factor used to convert matter fluctuations into
            DeltaSigma units.
        delta_pi_gm_squared_window: Squared line-of-sight window factor for the
            galaxy-matter auto-covariance.
        sigma_crit_squared_average: Effective average of :math:`\Sigma_c^2` for
            the lens-source bin pair.
        shape_noise: Source shape-noise contribution.
        shot_noise: Lens shot-noise contribution.
        volume: Effective survey volume for the lens bin.
        rp_bin_edges: Optional projected-radius bin edges for the final covariance.
            If omitted, the covariance is returned on the Hankel radial grid.
        order: Bessel order associated with the projected DeltaSigma statistic.
        taper: Whether to taper the lens-density and shear-fluctuation term before
            projection.
        taper_kwargs: Optional taper settings for the lens-density and
            shear-fluctuation term.
        gkgk_taper: Whether to taper the shared matter-fluctuation term before
            projection.
        gkgk_taper_kwargs: Optional taper settings for the shared
            matter-fluctuation term.

    Returns:
        Projected radii and the ``gm x gm`` DeltaSigma covariance matrix.
    """
    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)

    validate_power_spectrum_inputs(k, pk)
    validate_positive_scalar(galaxy_bias, "galaxy_bias")
    validate_positive_scalar(omega_m, "omega_m")
    validate_positive_scalar(rho_crit, "rho_crit")
    validate_positive_scalar(
        delta_pi_gm_squared_window,
        "delta_pi_gm_squared_window",
    )
    validate_positive_scalar(sigma_crit_squared_average, "sigma_crit_squared_average")
    validate_positive_scalar(shape_noise, "shape_noise")
    validate_positive_scalar(shot_noise, "shot_noise")
    validate_positive_scalar(volume, "volume")

    taper_kwargs = _build_taper_kwargs(k, taper_kwargs)
    gkgk_taper_kwargs = _build_taper_kwargs(k, gkgk_taper_kwargs)

    p_g = pk * galaxy_bias**2
    p_kappa = pk * (rho_crit * omega_m) ** 2 * delta_pi_gm_squared_window
    p_gk = pk * galaxy_bias * rho_crit * omega_m

    shape_delta_sigma_noise = shape_noise * sigma_crit_squared_average

    r_ggkk, cov_ggkk = hankel.projected_covariance(
        k_pk=k,
        pk1=p_g + shot_noise,
        pk2=p_kappa + shape_delta_sigma_noise,
        order=order,
        taper=taper,
        taper_kwargs=taper_kwargs,
    )

    r_gkgk, cov_gkgk = hankel.projected_covariance(
        k_pk=k,
        pk1=p_gk,
        pk2=p_gk,
        order=order,
        taper=gkgk_taper,
        taper_kwargs=gkgk_taper_kwargs,
    )

    if r_gkgk.shape != r_ggkk.shape or not np.allclose(r_gkgk, r_ggkk):
        raise ValueError(
            "The ggkk and gkgk covariance terms must be evaluated on the same "
            "radial grid before they can be combined."
        )

    cov = (cov_ggkk + cov_gkgk * delta_pi_gm_squared_window) / volume

    return _covariance_on_requested_radius_grid(hankel, r_ggkk, cov, rp_bin_edges)


def delta_sigma_gg_covariance(
    hankel: HankelTransform,
    k: ArrayLike,
    pk: ArrayLike,
    *,
    galaxy_bias: float,
    rho_crit: float,
    delta_pi_gg: float,
    shot_noise: float,
    volume: float,
    rp_bin_edges: ArrayLike | None = None,
    order: int = 2,
    taper: bool = True,
    taper_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the auto-covariance of the galaxy-galaxy DeltaSigma-like signal.

    This block describes the uncertainty of the clustering-style projected
    galaxy-density measurement written in DeltaSigma units. It is controlled by
    the lens galaxy auto-power spectrum, lens shot noise, the line-of-sight depth
    of the lens bin, and the effective survey volume.

    Args:
        hankel: Hankel transform object used to project Fourier-space covariance
            terms into projected-radius space.
        k: Wavenumber grid in ``h / Mpc``.
        pk: Lens-averaged matter power spectrum evaluated on ``k``.
        galaxy_bias: Linear galaxy bias of the lens sample.
        rho_crit: Critical density factor used to express the projected clustering
            covariance in DeltaSigma-like units.
        delta_pi_gg: Line-of-sight width or window factor for the galaxy-galaxy
            covariance block.
        shot_noise: Lens shot-noise contribution.
        volume: Effective survey volume for the lens bin.
        rp_bin_edges: Optional projected-radius bin edges for the final covariance.
        order: Bessel order associated with the projected statistic.
        taper: Whether to smoothly suppress the edges of the input ``k`` range before
            projection. This reduces artificial oscillations in projected-radius space
            caused by the finite Fourier-space range.
        taper_kwargs: Optional taper window settings. Accepted keys are
            ``low_k_lower``, ``low_k_upper``, ``large_k_lower``, and
            ``large_k_upper``. If not provided, default windows are chosen from the
            first and last values of ``k``.

    Returns:
        Projected radii and the ``gg x gg`` DeltaSigma covariance matrix.
    """
    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)

    validate_power_spectrum_inputs(k, pk)
    validate_positive_scalar(galaxy_bias, "galaxy_bias")
    validate_positive_scalar(rho_crit, "rho_crit")
    validate_positive_scalar(delta_pi_gg, "delta_pi_gg")
    validate_positive_scalar(shot_noise, "shot_noise")
    validate_positive_scalar(volume, "volume")

    taper_kwargs = _build_taper_kwargs(k, taper_kwargs)

    p_g = pk * galaxy_bias**2

    r, cov = hankel.projected_covariance(
        k_pk=k,
        pk1=p_g + shot_noise,
        pk2=p_g + shot_noise,
        order=order,
        taper=taper,
        taper_kwargs=taper_kwargs,
    )

    cov = cov * 2.0 * delta_pi_gg * rho_crit**2 / volume

    return _covariance_on_requested_radius_grid(hankel, r, cov, rp_bin_edges)


def delta_sigma_gm_gg_cross_covariance(
    hankel: HankelTransform,
    k: ArrayLike,
    pk: ArrayLike,
    *,
    galaxy_bias: float,
    omega_m: float,
    rho_crit: float,
    delta_pi_gm_gg: float,
    shot_noise: float,
    volume: float,
    rp_bin_edges: ArrayLike | None = None,
    order: int = 2,
    taper: bool = True,
    taper_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the cross-covariance between galaxy-matter and galaxy-galaxy signals.

    This block describes how fluctuations in the lensing-style DeltaSigma
    measurement covary with fluctuations in the projected galaxy-density
    measurement. It couples the galaxy-matter and galaxy-galaxy power-spectrum
    terms through the shared lens density field and the relevant line-of-sight
    window overlap.

    Args:
        hankel: Hankel transform object used to project Fourier-space covariance
            terms into projected-radius space.
        k: Wavenumber grid in ``h / Mpc``.
        pk: Lens-averaged matter power spectrum evaluated on ``k``.
        galaxy_bias: Linear galaxy bias of the lens sample.
        omega_m: Present-day matter density parameter.
        rho_crit: Critical density factor used to convert matter fluctuations into
            DeltaSigma units.
        delta_pi_gm_gg: Line-of-sight overlap factor between the ``gm`` and ``gg``
            covariance terms.
        shot_noise: Lens shot-noise contribution.
        volume: Effective survey volume for the lens bin.
        rp_bin_edges: Optional projected-radius bin edges for the final covariance.
        order: Bessel order associated with the projected statistic.
        taper: Whether to smoothly suppress the edges of the input ``k`` range before
            projection. This reduces artificial oscillations in projected-radius space
            caused by the finite Fourier-space range.
        taper_kwargs: Optional taper window settings. Accepted keys are
            ``low_k_lower``, ``low_k_upper``, ``large_k_lower``, and
            ``large_k_upper``. If not provided, default windows are chosen from the
            first and last values of ``k``.

    Returns:
        Projected radii and the ``gm x gg`` DeltaSigma cross-covariance matrix.
    """
    k = np.asarray(k, dtype=float)
    pk = np.asarray(pk, dtype=float)

    validate_power_spectrum_inputs(k, pk)
    validate_positive_scalar(galaxy_bias, "galaxy_bias")
    validate_positive_scalar(omega_m, "omega_m")
    validate_positive_scalar(rho_crit, "rho_crit")
    validate_positive_scalar(delta_pi_gm_gg, "delta_pi_gm_gg")
    validate_positive_scalar(shot_noise, "shot_noise")
    validate_positive_scalar(volume, "volume")

    taper_kwargs = _build_taper_kwargs(k, taper_kwargs)

    p_g = pk * galaxy_bias**2
    p_gk = pk * galaxy_bias * rho_crit * omega_m

    r, cov = hankel.projected_covariance(
        k_pk=k,
        pk1=p_gk,
        pk2=p_g + shot_noise,
        order=order,
        taper=taper,
        taper_kwargs=taper_kwargs,
    )

    cov = cov * 2.0 * delta_pi_gm_gg * rho_crit / volume

    return _covariance_on_requested_radius_grid(hankel, r, cov, rp_bin_edges)


def joint_delta_sigma_covariance(
    cov_gm_gm: ArrayLike,
    cov_gg_gg: ArrayLike,
    cov_gm_gg: ArrayLike,
) -> np.ndarray:
    r"""Return the joint covariance matrix for ``gm`` and ``gg`` DeltaSigma data.

    The output matrix is ordered with the galaxy-matter DeltaSigma data vector
    first and the galaxy-galaxy DeltaSigma-like data vector second. The off-diagonal
    blocks contain the cross-covariance between the two projected observables.

    Args:
        cov_gm_gm: Auto-covariance block for the galaxy-matter DeltaSigma signal.
        cov_gg_gg: Auto-covariance block for the galaxy-galaxy DeltaSigma-like
            signal.
        cov_gm_gg: Cross-covariance block between the ``gm`` and ``gg`` signals.

    Returns:
        Joint block covariance matrix ordered as
        ``[DeltaSigma_gm, DeltaSigma_gg]``.
    """
    cov_gm_gm, cov_gg_gg, cov_gm_gg = validate_joint_covariance_blocks(
        cov_gm_gm,
        cov_gg_gg,
        cov_gm_gg,
    )

    return np.block(
        [
            [cov_gm_gm, cov_gm_gg],
            [cov_gm_gg.T, cov_gg_gg],
        ]
    )


def _build_taper_kwargs(
    k: np.ndarray,
    taper_kwargs: dict | None,
) -> dict:
    r"""Return Fourier-space taper settings for covariance projections.

    The default matches the legacy DeltaSigma covariance calculation: suppress
    the low-k edge over a short interval above the minimum sampled wavenumber
    and suppress the high-k edge from k=10 h/Mpc to the maximum sampled
    wavenumber.

    Args:
        k: Wavenumber grid in ``h / Mpc``.
        taper_kwargs: Optional user-supplied taper settings. If supplied, these
            settings are returned unchanged.

    Returns:
        Taper keyword dictionary.
    """
    if taper_kwargs is not None:
        return taper_kwargs

    k_min = float(k[0])
    k_max = float(k[-1])

    return {
        "large_k_lower": DEFAULT_TAPER_KWARGS["large_k_lower"],
        "large_k_upper": k_max,
        "low_k_lower": k_min,
        "low_k_upper": k_min * DEFAULT_TAPER_KWARGS["low_k_factor"],
    }
