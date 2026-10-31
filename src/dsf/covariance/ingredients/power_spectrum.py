"""Power-spectrum ingredients for Delta Sigma covariance calculations.

This module computes lens-redshift-averaged power spectra used by Delta Sigma
covariance blocks. Power spectra are evaluated with CCL and averaged over
tomographic lens-bin redshift distributions.
"""

from __future__ import annotations

import numpy as np
import pyccl as ccl

from dsf.utils.converters import redshift_to_scale_factor
from dsf.utils.types import FloatArray
from dsf.utils.validators import (
    validate_positive_strictly_increasing_1d_array,
    validate_redshift_distribution,
)

__all__ = ["lens_averaged_power_spectrum"]


def lens_averaged_power_spectrum(
    cosmo: ccl.Cosmology,
    pk: ccl.Pk2D,
    k: FloatArray,
    z_lens: FloatArray,
    nz_lens: FloatArray,
) -> FloatArray:
    """Return a power spectrum averaged over a lens redshift distribution.

    Args:
        cosmo: CCL cosmology object.
        pk: CCL 2D power spectrum object.
        k: Wavenumber grid in ``h / Mpc``.
        z_lens: Lens redshift grid.
        nz_lens: Normalized lens redshift distribution evaluated on ``z_lens``.

    Returns:
        Lens-redshift-averaged power spectrum in ``(Mpc / h)^3``.
    """
    h = cosmo["h"]
    k_arr = validate_positive_strictly_increasing_1d_array(k, "k")
    z_arr, nz_arr = validate_redshift_distribution(
        z_lens,
        nz_lens,
        name="nz_lens",
    )

    k_physical = k_arr * h
    a_lens = redshift_to_scale_factor(z_arr)

    pk_values = np.stack([
        np.asarray(pk(k_physical, a, cosmo=cosmo), dtype=float)
        for a in a_lens
    ])

    pk_avg = np.trapezoid(pk_values * nz_arr[:, None], x=z_arr, axis=0)

    return np.asarray(pk_avg * h**3, dtype=float)
