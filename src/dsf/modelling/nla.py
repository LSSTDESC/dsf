"""Nonlinear alignment power-spectrum models."""

from __future__ import annotations

import numpy as np
import pyccl as ccl
from numpy.typing import NDArray

from dsf.data_vector.profiles import density_weighted_power_spectrum
from dsf.utils.validators import validate_finite_scalar, validate_pk2d_grids

__all__ = ["pk2d_nla"]


def pk2d_nla(
    cosmo: ccl.Cosmology,
    *,
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
    A_IA: float,
    C1rhocrit: float = 0.0134,
    b_g: float = 1.0,
) -> dict[str, ccl.Pk2D]:
    """Construct nonlinear alignment galaxy-intrinsic power spectra.

    The galaxy-intrinsic spectrum follows the nonlinear alignment model,
    with a constant intrinsic alignment amplitude and galaxy bias.

    Args:
        cosmo: CCL cosmology.
        k_array: Wavenumber grid.
        a_array: Scale-factor grid.
        A_IA: Intrinsic alignment amplitude.
        C1rhocrit: Intrinsic alignment normalization.
        b_g: Linear galaxy bias.

    Returns:
        Dictionary containing ``pk_deltasigma`` and ``pk_gi``.
    """
    k_arr, a_arr = validate_pk2d_grids(k_array, a_array)

    validate_finite_scalar(A_IA, "A_IA")
    validate_finite_scalar(C1rhocrit, "C1rhocrit")
    validate_finite_scalar(b_g, "b_g")

    growth = np.asarray(cosmo.growth_factor(a_arr), dtype=float)

    if np.any(~np.isfinite(growth)) or np.any(growth <= 0.0):
        raise ValueError("growth_factor(a_array) must be finite and positive.")

    pk_mm = np.asarray(
        cosmo.get_nonlin_power()(k_arr, a_arr, cosmo=cosmo),
        dtype=float,
    )

    if pk_mm.shape != (a_arr.size, k_arr.size):
        raise ValueError("Nonlinear matter power must have shape (len(a_array), len(k_array)).")

    if np.any(~np.isfinite(pk_mm)):
        raise ValueError("nonlinear matter power must contain only finite values.")

    amplitude = float(A_IA) * float(C1rhocrit) * float(cosmo["Omega_m"]) / growth

    pk_gi = -float(b_g) * amplitude[:, None] * pk_mm

    pk2d = ccl.Pk2D(
        a_arr=a_arr,
        lk_arr=np.log(k_arr),
        pk_arr=pk_gi,
        is_logp=False,
    )

    def pk_gi_power(cosmo, k, a):
        """Evaluate the galaxy-intrinsic power spectrum."""
        return pk2d(k, a, cosmo=cosmo)

    return {
        "pk_deltasigma": density_weighted_power_spectrum(cosmo, pk_gi_power),
        "pk_gi": pk2d,
    }
