"""Geometry ingredients for DeltaSigma covariance calculations.

This module converts tomographic redshift-bin metadata and cosmology into the
geometry quantities needed by the DeltaSigma covariance blocks.

Bin edges and redshift distributions are supplied by the tomography layer.
This module then computes cosmology-dependent distances, line-of-sight depths,
survey volumes, effective distances, and lens number densities.
"""

from __future__ import annotations

import numpy as np
import pyccl as ccl

from src.dsf.utils.converters import (
    comoving_distance_h,
    deg2_to_arcmin2,
    resolve_h,
)
from src.dsf.utils.integrators import trapezoid_integral
from src.dsf.utils.types import FloatArray
from src.dsf.utils.validators import (
    validate_1d_pair,
    validate_finite_scalar,
    validate_positive_scalar,
    validate_redshift_distribution,
    validate_redshift_edges,
)

from .sigma_crit import (
    effective_sigma_crit_squared,
    sigma_crit_inverse_comoving,
)

__all__ = [
    "delta_pi_from_window",
    "delta_pi_gg_from_edges",
    "delta_pi_gm_factors",
    "delta_pi_gm_from_window",
    "delta_pi_gm_gg_from_window",
    "effective_comoving_distance",
    "lens_number_density_3d_from_angular_density",
    "pi_limits_from_lens_edges",
    "survey_volume_from_edges",
    "gm_lensing_window",
]


def effective_comoving_distance(
    cosmo: ccl.Cosmology,
    z: FloatArray,
    nz: FloatArray,
    *,
    h: float | None = None,
) -> float:
    """Return the effective comoving distance of a tomographic sample.

    Args:
        cosmo: CCL cosmology object.
        z: Redshift grid.
        nz: Normalized redshift distribution evaluated on ``z``.
        h: Dimensionless Hubble parameter. If not supplied, read from ``cosmo["h"]``.

    Returns:
        Mean comoving radial distance in ``Mpc / h``.
    """
    h = resolve_h(cosmo, h)

    z_arr, nz_arr = validate_redshift_distribution(z, nz)
    chi = comoving_distance_h(cosmo, z_arr, h=h)

    return float(trapezoid_integral(chi * nz_arr, z_arr))


def delta_pi_gg_from_edges(
    cosmo: ccl.Cosmology,
    *,
    z_min: float,
    z_max: float,
    h: float | None = None,
) -> float:
    """Return the line-of-sight width of a lens redshift bin.

    Args:
        cosmo: CCL cosmology object.
        z_min: Lower lens-bin redshift edge.
        z_max: Upper lens-bin redshift edge.
        h: Dimensionless Hubble parameter. If not supplied, read from ``cosmo["h"]``.

    Returns:
        Comoving line-of-sight width in ``Mpc / h``.
    """
    validate_redshift_edges(z_min, z_max)

    h = resolve_h(cosmo, h)

    chi_min = float(comoving_distance_h(cosmo, z_min, h=h))
    chi_max = float(comoving_distance_h(cosmo, z_max, h=h))

    return chi_max - chi_min


def delta_pi_from_window(
    pi: FloatArray,
    window: FloatArray,
    *,
    squared: bool,
) -> float:
    """Return a line-of-sight window integral.

    Args:
        pi: Line-of-sight separation grid in ``Mpc / h``.
        window: Dimensionless window evaluated on ``pi``.
        squared: Whether to integrate the squared window.

    Returns:
        Integrated line-of-sight window factor in ``Mpc / h``.
    """
    pi_arr = np.asarray(pi, dtype=float)
    window_arr = np.asarray(window, dtype=float)

    validate_1d_pair(pi_arr, window_arr, x_name="pi", y_name="window")

    integrand = window_arr**2 if squared else window_arr

    return float(trapezoid_integral(integrand, pi_arr))


def delta_pi_gm_from_window(
    pi: FloatArray,
    window: FloatArray,
) -> float:
    """Return the squared line-of-sight window factor for ``gm x gm``.

    Args:
        pi: Line-of-sight separation grid in ``Mpc / h``.
        window: Dimensionless lensing window evaluated on ``pi``.

    Returns:
        Squared-window line-of-sight factor in ``Mpc / h``.
    """
    return delta_pi_from_window(pi, window, squared=True)


def delta_pi_gm_gg_from_window(
    pi: FloatArray,
    window: FloatArray,
    *,
    pi_min: float,
    pi_max: float,
) -> float:
    """Return the line-of-sight overlap factor for ``gm x gg``.

    Args:
        pi: Line-of-sight separation grid in ``Mpc / h``.
        window: Dimensionless lensing window evaluated on ``pi``.
        pi_min: Lower line-of-sight limit in ``Mpc / h``.
        pi_max: Upper line-of-sight limit in ``Mpc / h``.

    Returns:
        Cross-window line-of-sight factor in ``Mpc / h``.
    """
    validate_finite_scalar(pi_min, "pi_min")
    validate_finite_scalar(pi_max, "pi_max")

    if pi_max <= pi_min:
        raise ValueError("pi_max must be greater than pi_min.")

    pi_arr = np.asarray(pi, dtype=float)
    window_arr = np.asarray(window, dtype=float)

    validate_1d_pair(pi_arr, window_arr, x_name="pi", y_name="window")

    mask = (pi_arr >= pi_min) & (pi_arr <= pi_max)

    if np.count_nonzero(mask) < 2:
        raise ValueError("The requested pi range must contain at least two points.")

    return float(trapezoid_integral(window_arr[mask], pi_arr[mask]))


def survey_volume_from_edges(
    cosmo: ccl.Cosmology,
    *,
    z_min: float,
    z_max: float,
    area_deg2: float,
    h: float | None = None,
) -> float:
    """Return the comoving survey volume of a redshift shell.

    Args:
        cosmo: CCL cosmology object.
        z_min: Lower redshift edge.
        z_max: Upper redshift edge.
        area_deg2: Survey area in square degrees.
        h: Dimensionless Hubble parameter. If not supplied, read from ``cosmo["h"]``.

    Returns:
        Comoving shell volume in ``(Mpc / h)^3``.
    """
    z_min = float(np.asarray(z_min, dtype=float).squeeze())
    z_max = float(np.asarray(z_max, dtype=float).squeeze())

    validate_positive_scalar(area_deg2, name="area_deg2")
    validate_redshift_edges(z_min, z_max)

    h = resolve_h(cosmo, h)

    chi_min = comoving_distance_h(cosmo, z_min, h=h)
    chi_max = comoving_distance_h(cosmo, z_max, h=h)

    area_sr = area_deg2 * (np.pi / 180.0) ** 2

    return area_sr * (chi_max**3 - chi_min**3) / 3.0


def lens_number_density_3d_from_angular_density(
    cosmo: ccl.Cosmology,
    *,
    n_lens_arcmin2: float,
    z_min: float,
    z_max: float,
    area_deg2: float,
    h: float | None = None,
) -> float:
    """Return the three-dimensional lens number density for a redshift bin.

    Args:
        cosmo: CCL cosmology object.
        n_lens_arcmin2: Angular number density of lenses in ``arcmin^-2``.
        z_min: Lower lens-bin redshift edge.
        z_max: Upper lens-bin redshift edge.
        area_deg2: Survey area in square degrees.
        h: Dimensionless Hubble parameter. If not supplied, read from ``cosmo["h"]``.

    Returns:
        Comoving lens number density in ``(h / Mpc)^3``.
    """
    validate_positive_scalar(n_lens_arcmin2, "n_lens_arcmin2")

    h = resolve_h(cosmo, h)

    volume = survey_volume_from_edges(
        cosmo,
        z_min=z_min,
        z_max=z_max,
        area_deg2=area_deg2,
        h=h,
    )

    area_arcmin2 = deg2_to_arcmin2(area_deg2)
    n_lens_total = n_lens_arcmin2 * area_arcmin2

    return n_lens_total / volume


def pi_limits_from_lens_edges(
    cosmo: ccl.Cosmology,
    *,
    z_min: float,
    z_max: float,
    z_center: float,
    h: float | None = None,
) -> tuple[float, float]:
    """Return line-of-sight limits relative to a central lens redshift.

    Args:
        cosmo: CCL cosmology object.
        z_min: Lower lens-bin redshift edge.
        z_max: Upper lens-bin redshift edge.
        z_center: Effective or central lens redshift.
        h: Dimensionless Hubble parameter. If not supplied, read from ``cosmo["h"]``.

    Returns:
        Lower and upper line-of-sight limits in ``Mpc / h``.
    """
    validate_redshift_edges(z_min, z_max)
    validate_finite_scalar(z_center, "z_center")

    h = resolve_h(cosmo, h)

    if z_center < z_min or z_center > z_max:
        raise ValueError("z_center must lie between z_min and z_max.")

    chi_min = float(comoving_distance_h(cosmo, z_min, h=h))
    chi_max = float(comoving_distance_h(cosmo, z_max, h=h))
    chi_center = float(comoving_distance_h(cosmo, z_center, h=h))

    return chi_min - chi_center, chi_max - chi_center


def gm_lensing_window(
    cosmo: ccl.Cosmology,
    *,
    z_lens: FloatArray,
    nz_lens: FloatArray,
    z_source: FloatArray,
    nz_source: FloatArray,
    sigma_crit_prefactor: float,
    pi: FloatArray | None = None,
    h: float | None = None,
    pi_min: float = 0.1,
    pi_max: float = 3000.0,
    n_pi: int = 4000,
    z_interp_max: float | None = None,
    n_z_interp: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the galaxy-matter lensing line-of-sight window.

    The returned window is the dimensionless ``W(Pi)`` factor used for the
    ``gm x gm`` and ``gm x gg`` DeltaSigma covariance line-of-sight integrals.
    It follows the same structure as the legacy calculation: average
    ``Sigma_crit^{-1}`` over source redshifts at displaced lens positions,
    average that quantity over the lens redshift distribution, and multiply by
    the effective ``Sigma_crit`` normalization for the lens-source pair.

    Args:
        cosmo: CCL cosmology object.
        z_lens: Lens redshift grid.
        nz_lens: Normalized lens redshift distribution.
        z_source: Source redshift grid.
        nz_source: Normalized source redshift distribution.
        sigma_crit_prefactor: Unit-conversion prefactor used by the
            critical-surface-density ingredient.
        pi: Optional line-of-sight separation grid in ``Mpc / h``. If not
            supplied, a symmetric logarithmic grid from ``-pi_max`` to
            ``pi_max`` is built.
        h: Dimensionless Hubble parameter. If not supplied, read from
            ``cosmo["h"]``.
        pi_min: Minimum positive line-of-sight separation used when building
            the default grid.
        pi_max: Maximum positive line-of-sight separation used when building
            the default grid.
        n_pi: Number of positive grid points used when building the default
            grid.
        z_interp_max: Optional maximum redshift used for the distance-redshift
            interpolation.
        n_z_interp: Number of redshift points used for the interpolation.

    Returns:
        Tuple containing the line-of-sight grid in ``Mpc / h`` and the
        dimensionless lensing window evaluated on that grid.
    """
    h = resolve_h(cosmo, h)

    z_lens_arr, nz_lens_arr = validate_redshift_distribution(z_lens, nz_lens)
    z_source_arr, nz_source_arr = validate_redshift_distribution(
        z_source,
        nz_source,
    )

    validate_positive_scalar(sigma_crit_prefactor, "sigma_crit_prefactor")
    validate_positive_scalar(pi_min, "pi_min")
    validate_positive_scalar(pi_max, "pi_max")
    validate_positive_scalar(float(n_pi), "n_pi")
    validate_positive_scalar(float(n_z_interp), "n_z_interp")

    if pi_max <= pi_min:
        raise ValueError("pi_max must be greater than pi_min.")

    if pi is None:
        pi_positive = np.logspace(np.log10(pi_min), np.log10(pi_max), int(n_pi))
        pi_arr = np.concatenate((-pi_positive[::-1], pi_positive))
    else:
        pi_arr = np.asarray(pi, dtype=float)
        if pi_arr.ndim != 1:
            raise ValueError("pi must be one-dimensional.")
        if pi_arr.size < 2:
            raise ValueError("pi must contain at least two points.")

    chi_lens = comoving_distance_h(cosmo, z_lens_arr, h=h)

    max_target_chi = float(np.max(chi_lens) + np.max(np.abs(pi_arr)))
    if z_interp_max is None:
        z_interp_max = max(
            float(np.max(z_lens_arr)),
            float(np.max(z_source_arr)),
            5.0,
        )

    z_interp = np.linspace(0.0, float(z_interp_max), int(n_z_interp))
    chi_interp = comoving_distance_h(cosmo, z_interp, h=h)

    while float(chi_interp[-1]) < max_target_chi:
        z_interp = np.linspace(0.0, 2.0 * float(z_interp[-1]), int(n_z_interp))
        chi_interp = comoving_distance_h(cosmo, z_interp, h=h)

    sigma_crit_average = np.sqrt(
        effective_sigma_crit_squared(
            cosmo,
            z_lens=z_lens_arr,
            nz_lens=nz_lens_arr,
            z_source=z_source_arr,
            nz_source=nz_source_arr,
            h=h,
            sigma_crit_prefactor=sigma_crit_prefactor,
        )
    )

    window = np.zeros_like(pi_arr, dtype=float)

    for pi_index, pi_value in enumerate(pi_arr):
        displaced_chi = chi_lens + pi_value
        valid = displaced_chi > 0.0

        if not np.any(valid):
            continue

        displaced_z = np.zeros_like(displaced_chi)
        displaced_z[valid] = np.interp(
            displaced_chi[valid],
            chi_interp,
            z_interp,
        )

        source_averaged_sigma_crit_inv = np.zeros_like(z_lens_arr)

        for lens_index, z_displaced in enumerate(displaced_z):
            if not valid[lens_index]:
                continue

            sigma_crit_inv = sigma_crit_inverse_comoving(
                cosmo,
                z_lens=float(z_displaced),
                z_source=z_source_arr,
                h=h,
                sigma_crit_prefactor=sigma_crit_prefactor,
            )

            source_averaged_sigma_crit_inv[lens_index] = trapezoid_integral(
                sigma_crit_inv * nz_source_arr,
                z_source_arr,
            )

        lens_averaged_sigma_crit_inv = trapezoid_integral(
            source_averaged_sigma_crit_inv * nz_lens_arr,
            z_lens_arr,
        )

        window[pi_index] = lens_averaged_sigma_crit_inv * sigma_crit_average

    return pi_arr, window


def delta_pi_gm_factors(
    cosmo: ccl.Cosmology,
    *,
    z_lens: FloatArray,
    nz_lens: FloatArray,
    z_source: FloatArray,
    nz_source: FloatArray,
    z_min: float,
    z_max: float,
    z_center: float,
    sigma_crit_prefactor: float,
    pi: FloatArray | None = None,
    gm_window: FloatArray | None = None,
    h: float | None = None,
) -> tuple[float, float]:
    """Return line-of-sight factors for DeltaSigma ``gm`` covariance blocks.

    Args:
        cosmo: CCL cosmology object.
        z_lens: Lens redshift grid.
        nz_lens: Normalized lens redshift distribution.
        z_source: Source redshift grid.
        nz_source: Normalized source redshift distribution.
        z_min: Lower lens-bin redshift edge.
        z_max: Upper lens-bin redshift edge.
        z_center: Effective or central lens redshift.
        sigma_crit_prefactor: Unit-conversion prefactor used by the
            critical-surface-density ingredient.
        pi: Optional line-of-sight separation grid in ``Mpc / h``.
        gm_window: Optional galaxy-matter lensing window evaluated on ``pi``.
        h: Dimensionless Hubble parameter. If not supplied, read from
            ``cosmo["h"]``.

    Returns:
        The ``gm x gm`` and ``gm x gg`` line-of-sight factors in ``Mpc / h``.
    """
    if pi is None and gm_window is None:
        pi_arr, gm_window_arr = gm_lensing_window(
            cosmo,
            z_lens=z_lens,
            nz_lens=nz_lens,
            z_source=z_source,
            nz_source=nz_source,
            h=h,
            sigma_crit_prefactor=sigma_crit_prefactor,
        )
    elif pi is None or gm_window is None:
        raise ValueError("pi and gm_window must either both be supplied or both be None.")
    else:
        pi_arr = np.asarray(pi, dtype=float)
        gm_window_arr = np.asarray(gm_window, dtype=float)

    pi_min, pi_max = pi_limits_from_lens_edges(
        cosmo,
        z_min=z_min,
        z_max=z_max,
        z_center=z_center,
        h=h,
    )

    delta_pi_gm = delta_pi_gm_from_window(
        pi_arr,
        gm_window_arr,
    )

    delta_pi_gm_gg = delta_pi_gm_gg_from_window(
        pi_arr,
        gm_window_arr,
        pi_min=pi_min,
        pi_max=pi_max,
    )

    return delta_pi_gm, delta_pi_gm_gg
