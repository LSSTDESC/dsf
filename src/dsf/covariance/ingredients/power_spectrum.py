"""Power-spectrum ingredients for DeltaSigma covariance calculations.

This module computes effective matter power spectra used by DeltaSigma
covariance blocks. Redshift-dependent power spectra are evaluated with CCL and
averaged over tomographic lens-bin redshift distributions.
"""

from __future__ import annotations

import numpy as np
import pyccl as ccl

from src.dsf.utils.converters import redshift_to_scale_factor, resolve_h
from src.dsf.utils.integrators import weighted_trapezoid_average
from src.dsf.utils.types import FloatArray
from src.dsf.utils.validators import (
    validate_positive_strictly_increasing_1d_array,
    validate_redshift_distribution,
)

__all__ = [
    "lens_averaged_matter_power",
]


def lens_averaged_matter_power(
    cosmo: ccl.Cosmology,
    k: FloatArray,
    z_lens: FloatArray,
    nz_lens: FloatArray,
    *,
    h: float | None = None,
    nonlinear: bool = True,
) -> FloatArray:
    """Return the matter power spectrum averaged over a lens redshift distribution.

    Args:
        cosmo: CCL cosmology object.
        k: Wavenumber grid in ``h / Mpc``.
        z_lens: Lens redshift grid.
        nz_lens: Normalized lens redshift distribution evaluated on ``z_lens``.
        h: Dimensionless Hubble parameter. If not supplied, read from ``cosmo["h"]``.
        nonlinear: Whether to use the nonlinear matter power spectrum.

    Returns:
        Lens-redshift-averaged matter power spectrum in ``(Mpc / h)^3``.
    """
    h = resolve_h(cosmo, h)

    k_arr = validate_positive_strictly_increasing_1d_array(k, "k")
    z_arr, nz_arr = validate_redshift_distribution(
        z_lens,
        nz_lens,
        name="nz_lens",
    )

    power_function = ccl.nonlin_matter_power if nonlinear else ccl.linear_matter_power

    pk_of_z = np.asarray(
        [
            power_function(
                cosmo,
                k_arr * h,
                redshift_to_scale_factor(z_value),
            )
            * h**3
            for z_value in z_arr
        ]
    )

    pk_avg = weighted_trapezoid_average(
        z_arr,
        pk_of_z,
        nz_arr,
        axis=0,
        normalize_weights=False,
    )

    return np.asarray(pk_avg, dtype=float)
