"""Halo occupation distribution power spectrum models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyccl as ccl
from numpy.typing import NDArray

from dsf.data_vector.profiles import density_weighted_power_spectrum
from dsf.utils.validators import validate_pk2d_grids, validate_scale_factor

__all__ = [
    "MASS_DEF",
    "CONCENTRATION",
    "MASS_FUNCTION",
    "HALO_BIAS",
    "MATTER_PROFILE",
    "HM_CALCULATOR",
    "HOD_2PT",
    "hod_galaxy_bias",
    "pk2d_hod",
]


MASS_DEF = ccl.halos.MassDef200c
CONCENTRATION = ccl.halos.ConcentrationDuffy08(mass_def=MASS_DEF)
MASS_FUNCTION = ccl.halos.MassFuncTinker10(mass_def=MASS_DEF)
HALO_BIAS = ccl.halos.HaloBiasTinker10(mass_def=MASS_DEF)

MATTER_PROFILE = ccl.halos.HaloProfileNFW(
    mass_def=MASS_DEF,
    concentration=CONCENTRATION,
    fourier_analytic=True,
)

HM_CALCULATOR = ccl.halos.HMCalculator(
    mass_function=MASS_FUNCTION,
    halo_bias=HALO_BIAS,
    mass_def=MASS_DEF,
)

HOD_2PT = ccl.halos.Profile2ptHOD()


def _pk2d_hod(
    cosmo: ccl.Cosmology,
    *,
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
    concentration: ccl.halos.Concentration,
    matter_profile: ccl.halos.HaloProfileNFW | None = None,
    **hod_kwargs: Any,
) -> dict[str, ccl.Pk2D]:
    """Construct HOD galaxy-matter and galaxy-galaxy power spectra.

    Args:
        cosmo: CCL cosmology.
        k_array: Wavenumber grid.
        a_array: Scale-factor grid.
        concentration: CCL concentration model used by the galaxy profile.
        matter_profile: Optional matter profile. If omitted, constructs an
            NFW profile using the supplied concentration model.
        **hod_kwargs: HOD profile parameters.

    Returns:
        Dictionary containing ``pk_deltasigma``, ``pk_gm``, and ``pk_gg``.
    """
    k_arr, a_arr = validate_pk2d_grids(k_array, a_array)

    galaxy_profile = ccl.halos.HaloProfileHOD(
        mass_def=MASS_DEF,
        concentration=concentration,
        **hod_kwargs,
    )

    if matter_profile is None:
        matter_profile = ccl.halos.HaloProfileNFW(
            mass_def=MASS_DEF,
            concentration=concentration,
            fourier_analytic=True,
        )

    pk_gm = ccl.halos.halomod_Pk2D(
        cosmo,
        HM_CALCULATOR,
        galaxy_profile,
        prof2=matter_profile,
        lk_arr=np.log(k_arr),
        a_arr=a_arr,
        p_of_k_a=cosmo.get_nonlin_power(),
    )

    pk_gg = ccl.halos.halomod_Pk2D(
        cosmo,
        HM_CALCULATOR,
        galaxy_profile,
        prof_2pt=HOD_2PT,
        lk_arr=np.log(k_arr),
        a_arr=a_arr,
        p_of_k_a=cosmo.get_nonlin_power(),
    )

    def pk_gm_power(cosmo, k, a):
        """Evaluate the galaxy-matter power spectrum."""
        return pk_gm(k, a, cosmo=cosmo)

    return {
        "pk_deltasigma": density_weighted_power_spectrum(cosmo, pk_gm_power),
        "pk_gm": pk_gm,
        "pk_gg": pk_gg,
    }


def pk2d_hod(
    cosmo: ccl.Cosmology,
    *,
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
    concentration: ccl.halos.Concentration | None = None,
    **hod_kwargs: Any,
) -> dict[str, ccl.Pk2D]:
    """Construct standard HOD power spectra.

    Args:
        cosmo: CCL cosmology.
        k_array: Wavenumber grid.
        a_array: Scale-factor grid.
        concentration: Optional CCL concentration model. Defaults to Duffy08.
            The model must be compatible with the 200c halo mass definition.
        **hod_kwargs: HOD profile parameters.

    Returns:
        Dictionary containing ``pk_deltasigma``, ``pk_gm``, and ``pk_gg``.
    """
    if concentration is None:
        concentration = CONCENTRATION

    return _pk2d_hod(
        cosmo,
        k_array=k_array,
        a_array=a_array,
        concentration=concentration,
        matter_profile=MATTER_PROFILE if concentration is CONCENTRATION else None,
        **hod_kwargs,
    )


def hod_galaxy_bias(
    cosmo: ccl.Cosmology,
    *,
    a: float,
    concentration: ccl.halos.Concentration | None = None,
    **hod_kwargs: Any,
) -> float:
    """Calculate the occupation-weighted linear galaxy bias.

    The effective bias is computed by averaging the halo bias over the
    central and satellite occupations, weighted by the halo mass function.

    Args:
        cosmo: CCL cosmology.
        a: Scale factor.
        concentration: Optional CCL concentration model. Defaults to Duffy08.
            The model must be compatible with the 200c halo mass definition.
        **hod_kwargs: HOD profile parameters.

    Returns:
        Effective linear galaxy bias.

    Raises:
        ValueError: If the scale factor is invalid or the galaxy
            normalization is nonfinite or nonpositive.
    """
    validate_scale_factor(a)

    if concentration is None:
        concentration = CONCENTRATION

    galaxy_profile = ccl.halos.HaloProfileHOD(
        mass_def=MASS_DEF,
        concentration=concentration,
        **hod_kwargs,
    )

    n_g = galaxy_profile.get_normalization(cosmo, a, hmc=HM_CALCULATOR)

    if not np.isfinite(n_g) or n_g <= 0.0:
        raise ValueError("HOD galaxy normalization must be finite and positive.")

    def integrand(mass: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluate the occupation-weighted halo bias."""
        bias = HM_CALCULATOR.halo_bias(cosmo, mass, a)
        occupation = galaxy_profile._Nc(mass, a) + galaxy_profile._Ns(mass, a)
        return bias * occupation

    weighted_bias = HM_CALCULATOR.integrate_over_massfunc(integrand, cosmo, a)

    return float(weighted_bias / n_g)
