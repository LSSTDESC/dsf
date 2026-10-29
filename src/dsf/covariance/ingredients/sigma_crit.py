"""Critical-surface-density ingredients for Delta Sigma covariance calculations.

This module computes inverse comoving critical surface densities and effective
source-noise conversion factors for lens-source tomographic bin pairs.
"""

from __future__ import annotations

import numpy as np
import pyccl as ccl

from dsf.utils.converters import (
    comoving_distance_h,
    sigma_crit_prefactor_msun_h_pc2,
)
from dsf.utils.integrators import trapezoid_integral
from dsf.utils.types import FloatArray
from dsf.utils.validators import (
    validate_finite_scalar,
    validate_nonnegative_1d_array,
    validate_redshift_distribution,
)

__all__ = [
    "effective_sigma_crit_squared",
    "sigma_crit_inverse_comoving",
    "sigma_crit_inverse_source_average",
    "sigma_crit_squared_inverse_source_average",
    "sigma_crit_squared_average",
    "effective_squared_sigma_crit",
]


_SIGMA_CRIT_PREFACTOR = sigma_crit_prefactor_msun_h_pc2()


def sigma_crit_inverse_comoving(
    cosmo: ccl.Cosmology,
    z_lens: float,
    z_source: FloatArray,
) -> FloatArray:
    """Return the inverse comoving critical surface density for one lens redshift.

    Sources behind the lens receive non-zero lensing efficiency. Sources at or
    in front of the lens are assigned zero contribution.

    Args:
        cosmo: CCL cosmology object.
        z_lens: Lens redshift.
        z_source: Source redshift grid.

    Returns:
        Inverse comoving critical surface density in
        ``pc^2 / (h Msun)`` evaluated on ``z_source``.
    """
    validate_finite_scalar(z_lens, "z_lens")

    if z_lens < 0.0:
        raise ValueError("z_lens must be non-negative.")

    z_source_arr = validate_nonnegative_1d_array(
        z_source,
        "z_source",
        min_size=1,
    )

    chi_lens = comoving_distance_h(cosmo, z_lens)
    chi_source = comoving_distance_h(cosmo, z_source_arr)

    sigma_crit_inv = np.zeros_like(z_source_arr, dtype=float)
    behind = chi_source > chi_lens

    sigma_crit_inv[behind] = (
        chi_lens
        * (chi_source[behind] - chi_lens)
        / chi_source[behind]
        / _SIGMA_CRIT_PREFACTOR
        * (1.0 + z_lens)
    )

    return sigma_crit_inv


def sigma_crit_inverse_source_average(
    cosmo: ccl.Cosmology,
    *,
    z_lens: float,
    z_source: FloatArray,
    nz_source: FloatArray,
) -> float:
    """Return the source-averaged inverse comoving critical surface density.

    This averages the lensing efficiency for a fixed lens redshift over the
    source redshift distribution.

    Args:
        cosmo: CCL cosmology object.
        z_lens: Lens redshift.
        z_source: Source redshift grid.
        nz_source: Normalized source redshift distribution.

    Returns:
        Source-averaged inverse comoving critical surface density.
    """
    z_source_arr, nz_source_arr = validate_redshift_distribution(
        z_source,
        nz_source,
        name="nz_source",
    )

    sigma_inv = sigma_crit_inverse_comoving(cosmo, z_lens, z_source_arr)

    return float(trapezoid_integral(sigma_inv * nz_source_arr, z_source_arr))


def sigma_crit_squared_inverse_source_average(
    cosmo: ccl.Cosmology,
    *,
    z_lens: float,
    z_source: FloatArray,
    nz_source: FloatArray,
) -> float:
    """Return the source-averaged squared inverse comoving critical surface density.

    This averages the lensing efficiency squared for a fixed lens redshift over the
    source redshift distribution.

    Args:
        cosmo: CCL cosmology object.
        z_lens: Lens redshift.
        z_source: Source redshift grid.
        nz_source: Normalized source redshift distribution.

    Returns:
        Source-averaged squared inverse comoving critical surface density.
    """
    z_source_arr, nz_source_arr = validate_redshift_distribution(
        z_source,
        nz_source,
        name="nz_source",
    )

    sigma_inv = sigma_crit_inverse_comoving(cosmo, z_lens, z_source_arr)

    return float(trapezoid_integral(sigma_inv**2 * nz_source_arr, z_source_arr))


def _sigma_crit_inverse_grid(
    cosmo: ccl.Cosmology,
    z_lens: FloatArray,
    z_source: FloatArray,
) -> FloatArray:
    """Compute inverse critical surface densities on a lens-source grid.

    Args:
        cosmo: CCL cosmology object.
        z_lens: Lens redshift grid.
        z_source: Source redshift grid.

    Returns:
        Inverse critical surface densities with shape
        ``(n_lens, n_source)``.
    """
    chi_lens = np.asarray(
        comoving_distance_h(cosmo, z_lens),
        dtype=float,
    )

    chi_source = np.asarray(
        comoving_distance_h(cosmo, z_source),
        dtype=float,
    )

    chi_l = chi_lens[:, None]
    chi_s = chi_source[None, :]

    efficiency = np.zeros(
        (z_lens.size, z_source.size),
        dtype=float,
    )

    np.divide(
        chi_s - chi_l,
        chi_s,
        out=efficiency,
        where=chi_s > chi_l,
    )

    return (
        chi_l
        * efficiency
        * (1.0 + z_lens[:, None])
        / _SIGMA_CRIT_PREFACTOR
    )


def effective_sigma_crit_squared(
    cosmo: ccl.Cosmology,
    *,
    z_lens: FloatArray,
    nz_lens: FloatArray,
    z_source: FloatArray,
    nz_source: FloatArray,
) -> float:
    r"""Return the inverse squared mean inverse critical surface density.

    This quantity converts source shape noise into Delta Sigma covariance units
    for a lens-source tomographic bin pair.

    Args:
        cosmo: CCL cosmology object.
        z_lens: Lens redshift grid.
        nz_lens: Normalized lens redshift distribution.
        z_source: Source redshift grid.
        nz_source: Normalized source redshift distribution.

    Returns:
        Effective :math:`\Sigma_c^2` factor for the lens-source bin pair.
    """
    z_lens_arr, nz_lens_arr = validate_redshift_distribution(
        z_lens,
        nz_lens,
        name="nz_lens",
    )

    z_source_arr, nz_source_arr = validate_redshift_distribution(
        z_source,
        nz_source,
        name="nz_source",
    )

    sigma_inv = _sigma_crit_inverse_grid(
        cosmo,
        z_lens_arr,
        z_source_arr,
    )

    source_average = np.trapezoid(
        sigma_inv * nz_source_arr[None, :],
        x=z_source_arr,
        axis=1,
    )

    sigma_inv_avg = float(
        np.trapezoid(
            source_average * nz_lens_arr,
            x=z_lens_arr,
        )
    )

    if not np.isfinite(sigma_inv_avg) or sigma_inv_avg <= 0.0:
        raise ValueError("Average SigmaCrit inverse must be finite and positive.")

    return 1.0 / sigma_inv_avg**2


def effective_squared_sigma_crit(
    cosmo: ccl.Cosmology,
    *,
    z_lens: FloatArray,
    nz_lens: FloatArray,
    z_source: FloatArray,
    nz_source: FloatArray,
) -> float:
    r"""Return the inverse of the averaged squared inverse critical surface density.

    This computes the inverse of the lens- and source-averaged
    :math:`\Sigma_{\mathrm{crit}}^{-2}`.

    This quantity converts source shape noise into DeltaSigma covariance units
    for a lens-source tomographic bin pair.

    Args:
        cosmo: CCL cosmology object.
        z_lens: Lens redshift grid.
        nz_lens: Normalized lens redshift distribution.
        z_source: Source redshift grid.
        nz_source: Normalized source redshift distribution.

    Returns:
        Effective :math:`\Sigma_c^2` factor for the lens-source bin pair.
    """
    z_lens_arr, nz_lens_arr = validate_redshift_distribution(
        z_lens,
        nz_lens,
        name="nz_lens",
    )

    z_source_arr, nz_source_arr = validate_redshift_distribution(
        z_source,
        nz_source,
        name="nz_source",
    )

    sigma_inv = _sigma_crit_inverse_grid(
        cosmo,
        z_lens_arr,
        z_source_arr,
    )

    source_average = np.trapezoid(
        sigma_inv**2 * nz_source_arr[None, :],
        x=z_source_arr,
        axis=1,
    )

    sigma_sq_inv_avg = float(
        np.trapezoid(
            source_average * nz_lens_arr,
            x=z_lens_arr,
        )
    )

    if not np.isfinite(sigma_sq_inv_avg) or sigma_sq_inv_avg <= 0.0:
        raise ValueError("Average SigmaCrit inverse must be finite and positive.")

    return 1.0 / sigma_sq_inv_avg


def sigma_crit_squared_average(
    cosmo: ccl.Cosmology,
    *,
    z_lens: FloatArray,
    nz_lens: FloatArray,
    z_source: FloatArray,
    nz_source: FloatArray,
) -> float:
    """Return the effective critical-surface-density squared factor.

    This alias preserves the previous public function name.
    """
    return effective_sigma_crit_squared(
        cosmo,
        z_lens=z_lens,
        nz_lens=nz_lens,
        z_source=z_source,
        nz_source=nz_source,
    )
