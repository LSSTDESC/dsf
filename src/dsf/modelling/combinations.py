"""Combined HOD, baryonic, and intrinsic alignment power-spectrum models."""

from __future__ import annotations

from typing import Any

import pyccl as ccl
from numpy.typing import NDArray
import numpy as np

from dsf.modelling.baryons import pk2d_hod_baryonified
from dsf.modelling.hod import hod_galaxy_bias, pk2d_hod
from dsf.modelling.nla import pk2d_nla
from dsf.utils.validators import validate_finite_scalar

__all__ = [
    "pk2d_hod_with_nla",
    "pk2d_hod_baryonified_with_nla",
]


def _add_nla_to_hod(
    cosmo: ccl.Cosmology,
    pk_hod: dict[str, ccl.Pk2D],
    *,
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
    A_IA: float,
    C1rhocrit: float,
    b_g: float | None,
    a_bias: float,
    concentration: ccl.halos.Concentration | None,
    **hod_kwargs: Any,
) -> dict[str, ccl.Pk2D]:
    """Add NLA to the Delta Sigma spectrum, preserving HOD covariance spectra.

    Args:
        cosmo: CCL cosmology.
        pk_hod: HOD power-spectrum dictionary.
        k_array: Wavenumber grid.
        a_array: Scale-factor grid.
        A_IA: Intrinsic alignment amplitude.
        C1rhocrit: NLA normalization.
        b_g: Galaxy bias, computed from the HOD if ``None``.
        a_bias: Scale factor for the HOD bias calculation.
        concentration: Optional CCL concentration model.
        **hod_kwargs: HOD profile parameters.

    Returns:
        HOD spectra with NLA added to ``pk_deltasigma``.
    """
    validate_finite_scalar(A_IA, "A_IA")

    if float(A_IA) == 0.0:
        return pk_hod

    if b_g is None:
        b_g = hod_galaxy_bias(
            cosmo,
            a=a_bias,
            concentration=concentration,
            **hod_kwargs,
        )

    pk_ia = pk2d_nla(
        cosmo,
        k_array=k_array,
        a_array=a_array,
        A_IA=A_IA,
        C1rhocrit=C1rhocrit,
        b_g=float(b_g),
    )

    return {
        **pk_hod,
        "pk_deltasigma": pk_hod["pk_deltasigma"] + pk_ia["pk_deltasigma"],
    }


def pk2d_hod_with_nla(
    cosmo: ccl.Cosmology,
    *,
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
    A_IA: float = 0.0,
    C1rhocrit: float = 0.0134,
    b_g: float | None = None,
    a_bias: float = 1.0,
    concentration: ccl.halos.Concentration | None = None,
    **hod_kwargs: Any,
) -> dict[str, ccl.Pk2D]:
    """Construct HOD spectra with an optional NLA contribution.

    Args:
        cosmo: CCL cosmology.
        k_array: Wavenumber grid.
        a_array: Scale-factor grid.
        A_IA: Intrinsic alignment amplitude.
        C1rhocrit: NLA normalization.
        b_g: Galaxy bias, computed from the HOD if ``None``.
        a_bias: Scale factor for the HOD bias calculation.
        concentration: Optional CCL concentration model.
        **hod_kwargs: HOD profile parameters.

    Returns:
        Delta Sigma spectrum including NLA and HOD-only covariance spectra.
    """
    pk_hod = pk2d_hod(
        cosmo,
        k_array=k_array,
        a_array=a_array,
        concentration=concentration,
        **hod_kwargs,
    )

    return _add_nla_to_hod(
        cosmo,
        pk_hod,
        k_array=k_array,
        a_array=a_array,
        A_IA=A_IA,
        C1rhocrit=C1rhocrit,
        b_g=b_g,
        a_bias=a_bias,
        concentration=concentration,
        **hod_kwargs,
    )


def pk2d_hod_baryonified_with_nla(
    cosmo: ccl.Cosmology,
    *,
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
    f_c: float = 1.0,
    A_IA: float = 0.0,
    C1rhocrit: float = 0.0134,
    b_g: float | None = None,
    a_bias: float = 1.0,
    concentration: ccl.halos.Concentration | None = None,
    **hod_kwargs: Any,
) -> dict[str, ccl.Pk2D]:
    """Construct baryonified HOD spectra with an optional NLA contribution.

    Args:
        cosmo: CCL cosmology.
        k_array: Wavenumber grid.
        a_array: Scale-factor grid.
        f_c: Baryonic concentration rescaling.
        A_IA: Intrinsic alignment amplitude.
        C1rhocrit: NLA normalization.
        b_g: Galaxy bias, computed from the HOD if ``None``.
        a_bias: Scale factor for the HOD bias calculation.
        concentration: Optional CCL concentration model.
        **hod_kwargs: HOD profile parameters.

    Returns:
        Delta Sigma spectrum including NLA and baryonified HOD covariance spectra.
    """
    pk_hod = pk2d_hod_baryonified(
        cosmo,
        k_array=k_array,
        a_array=a_array,
        f_c=f_c,
        concentration=concentration,
        **hod_kwargs,
    )

    return _add_nla_to_hod(
        cosmo,
        pk_hod,
        k_array=k_array,
        a_array=a_array,
        A_IA=A_IA,
        C1rhocrit=C1rhocrit,
        b_g=b_g,
        a_bias=a_bias,
        concentration=concentration,
        **hod_kwargs,
    )
