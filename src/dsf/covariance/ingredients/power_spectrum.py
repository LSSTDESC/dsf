"""Power-spectrum ingredients for Delta Sigma covariance calculations.

This module computes effective matter power spectra used by Delta Sigma
covariance blocks. Redshift-dependent power spectra are evaluated with CCL and
averaged over tomographic lens-bin redshift distributions.
"""

from __future__ import annotations

import numpy as np
import pyccl as ccl

from dsf.utils.converters import redshift_to_scale_factor, resolve_h
from dsf.utils.types import FloatArray
from dsf.utils.validators import (
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
) -> FloatArray:
    """Return the matter power spectrum averaged over a lens redshift distribution.

    Args:
        cosmo: CCL cosmology object. The matter power spectrum prescription
            is determined by its ``matter_power_spectrum`` configuration.
        k: Wavenumber grid in ``h / Mpc``.
        z_lens: Lens redshift grid.
        nz_lens: Normalized lens redshift distribution evaluated on ``z_lens``.
        h: Dimensionless Hubble parameter. If not supplied, read from
            ``cosmo["h"]``.

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

    dz = np.diff(z_arr)

    integration_weights = np.empty_like(z_arr)
    integration_weights[0] = dz[0] / 2.0
    integration_weights[-1] = dz[-1] / 2.0
    integration_weights[1:-1] = (dz[:-1] + dz[1:]) / 2.0

    integration_weights *= nz_arr

    active = integration_weights != 0.0

    pk_avg = np.zeros_like(k_arr, dtype=float)
    k_physical = k_arr * h

    for z_value, weight in zip(
        z_arr[active],
        integration_weights[active],
        strict=True,
    ):
        pk_avg += weight * ccl.nonlin_matter_power(
            cosmo,
            k_physical,
            redshift_to_scale_factor(z_value),
        )

    return np.asarray(pk_avg * h**3, dtype=float)
