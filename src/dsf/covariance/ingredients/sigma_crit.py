"""Critical-surface-density ingredients for DeltaSigma covariance calculations.

This module computes inverse comoving critical surface densities and effective
source-noise conversion factors for lens-source tomographic bin pairs.
"""

from __future__ import annotations

import numpy as np
import pyccl as ccl

from src.dsf.utils.converters import comoving_distance_h, resolve_h
from src.dsf.utils.integrators import trapezoid_integral
from src.dsf.utils.types import FloatArray
from src.dsf.utils.validators import (
    validate_finite_scalar,
    validate_nonnegative_1d_array,
    validate_positive_scalar,
    validate_redshift_distribution,
)

__all__ = [
    "effective_sigma_crit_squared",
    "sigma_crit_inverse_comoving",
    "sigma_crit_inverse_source_average",
    "sigma_crit_squared_average",
]


def sigma_crit_inverse_comoving(
    cosmo: ccl.Cosmology,
    z_lens: float,
    z_source: FloatArray,
    *,
    h: float | None = None,
    sigma_crit_prefactor: float,
) -> FloatArray:
    """Return the inverse comoving critical surface density for one lens redshift.

    Sources behind the lens receive non-zero lensing efficiency. Sources at or
    in front of the lens are assigned zero contribution.

    Args:
        cosmo: CCL cosmology object.
        z_lens: Lens redshift.
        z_source: Source redshift grid.
        h: Dimensionless Hubble parameter. If not supplied, read from ``cosmo["h"]``.
        sigma_crit_prefactor: Unit-conversion prefactor defining the desired
            critical-surface-density convention.

    Returns:
        Inverse comoving critical surface density evaluated on ``z_source``.
    """
    h = resolve_h(cosmo, h)

    validate_positive_scalar(sigma_crit_prefactor, "sigma_crit_prefactor")
    validate_finite_scalar(z_lens, "z_lens")

    if z_lens < 0.0:
        raise ValueError("z_lens must be non-negative.")

    z_source_arr = validate_nonnegative_1d_array(
        z_source,
        "z_source",
        min_size=1,
    )

    chi_lens = comoving_distance_h(cosmo, z_lens, h=h)
    chi_source = comoving_distance_h(cosmo, z_source_arr, h=h)

    sigma_crit_inv = np.zeros_like(z_source_arr, dtype=float)
    behind = chi_source > chi_lens

    sigma_crit_inv[behind] = (
        chi_lens
        * (chi_source[behind] - chi_lens)
        / chi_source[behind]
        / sigma_crit_prefactor
        * (1.0 + z_lens)
    )

    return sigma_crit_inv


def sigma_crit_inverse_source_average(
    cosmo: ccl.Cosmology,
    *,
    z_lens: float,
    z_source: FloatArray,
    nz_source: FloatArray,
    h: float | None = None,
    sigma_crit_prefactor: float,
) -> float:
    """Return the source-averaged inverse comoving critical surface density.

    This averages the lensing efficiency for a fixed lens redshift over the
    source redshift distribution.

    Args:
        cosmo: CCL cosmology object.
        z_lens: Lens redshift.
        z_source: Source redshift grid.
        nz_source: Normalized source redshift distribution.
        h: Dimensionless Hubble parameter. If not supplied, read from ``cosmo["h"]``.
        sigma_crit_prefactor: Unit-conversion prefactor defining the desired
            critical-surface-density convention.

    Returns:
        Source-averaged inverse comoving critical surface density.
    """
    h = resolve_h(cosmo, h)

    z_source_arr, nz_source_arr = validate_redshift_distribution(
        z_source,
        nz_source,
        name="nz_source",
    )

    sigma_inv = sigma_crit_inverse_comoving(
        cosmo,
        z_lens,
        z_source_arr,
        h=h,
        sigma_crit_prefactor=sigma_crit_prefactor,
    )

    return float(trapezoid_integral(sigma_inv * nz_source_arr, z_source_arr))


def effective_sigma_crit_squared(
    cosmo: ccl.Cosmology,
    *,
    z_lens: FloatArray,
    nz_lens: FloatArray,
    z_source: FloatArray,
    nz_source: FloatArray,
    h: float | None = None,
    sigma_crit_prefactor: float,
) -> float:
    r"""Return the effective critical-surface-density squared factor.

    This quantity converts source shape noise into DeltaSigma covariance units
    for a lens-source tomographic bin pair.

    Args:
        cosmo: CCL cosmology object.
        z_lens: Lens redshift grid.
        nz_lens: Normalized lens redshift distribution.
        z_source: Source redshift grid.
        nz_source: Normalized source redshift distribution.
        h: Dimensionless Hubble parameter. If not supplied, read from ``cosmo["h"]``.
        sigma_crit_prefactor: Unit-conversion prefactor defining the desired
            critical-surface-density convention.

    Returns:
        Effective :math:`\Sigma_c^2` factor for the lens-source bin pair.
    """
    h = resolve_h(cosmo, h)

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

    sigma_inv_source_avg = np.asarray(
        [
            sigma_crit_inverse_source_average(
                cosmo,
                z_lens=z_value,
                z_source=z_source_arr,
                nz_source=nz_source_arr,
                h=h,
                sigma_crit_prefactor=sigma_crit_prefactor,
            )
            for z_value in z_lens_arr
        ],
        dtype=float,
    )

    sigma_inv_avg = float(
        trapezoid_integral(
            sigma_inv_source_avg * nz_lens_arr,
            z_lens_arr,
        )
    )

    if not np.isfinite(sigma_inv_avg) or sigma_inv_avg <= 0.0:
        raise ValueError("Average SigmaCrit inverse must be finite and positive.")

    return 1.0 / sigma_inv_avg**2


def sigma_crit_squared_average(
    cosmo: ccl.Cosmology,
    *,
    z_lens: FloatArray,
    nz_lens: FloatArray,
    z_source: FloatArray,
    nz_source: FloatArray,
    h: float | None = None,
    sigma_crit_prefactor: float,
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
        h=h,
        sigma_crit_prefactor=sigma_crit_prefactor,
    )
