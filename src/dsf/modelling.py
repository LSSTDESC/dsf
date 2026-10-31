"""HOD and intrinsic alignment power-spectrum models for Delta Sigma.

Construct density-weighted power spectra for Delta Sigma predictions and
galaxy-matter and galaxy-galaxy spectra for covariance calculations.

Combined HOD and NLA models include intrinsic alignment in the data vector
while retaining HOD-only spectra for the current covariance approximation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pyccl as ccl
from numpy.typing import NDArray

from dsf.data_vector.profiles import density_weighted_power_spectrum
from dsf.utils.validators import (
    validate_finite_scalar,
    validate_positive_scalar,
    validate_positive_strictly_increasing_1d_array,
    validate_scale_factor,
)

__all__ = [
    "MASS_DEF",
    "CONCENTRATION",
    "MASS_FUNCTION",
    "HALO_BIAS",
    "MATTER_PROFILE",
    "HM_CALCULATOR",
    "HOD_2PT",
    "baryonified_duffy_concentration",
    "hod_galaxy_bias",
    "make_ccl_cosmology",
    "pk2d_hod",
    "pk2d_hod_baryonified",
    "pk2d_hod_baryonified_with_nla",
    "pk2d_hod_with_nla",
    "pk2d_nla",
]


MASS_DEF = ccl.halos.MassDef200c
CONCENTRATION = ccl.halos.ConcentrationDuffy08(mass_def=MASS_DEF)
MASS_FUNCTION = ccl.halos.MassFuncTinker10(mass_def=MASS_DEF)
HALO_BIAS = ccl.halos.HaloBiasTinker10(mass_def=MASS_DEF)
MATTER_PROFILE = ccl.halos.HaloProfileNFW(
    mass_def=MASS_DEF, concentration=CONCENTRATION, fourier_analytic=True
)
HM_CALCULATOR = ccl.halos.HMCalculator(
    mass_function=MASS_FUNCTION, halo_bias=HALO_BIAS, mass_def=MASS_DEF
)
HOD_2PT = ccl.halos.Profile2ptHOD()


def make_ccl_cosmology(
    cosmology: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ccl.Cosmology:
    """Construct a CCL cosmology, accepting ``Omega_m`` in place of ``Omega_c``.

    Args:
        cosmology: Optional mapping of cosmological parameters.
        **kwargs: Additional CCL cosmology parameters.

    Returns:
        CCL cosmology object.
    """
    if cosmology is None:
        return ccl.CosmologyVanillaLCDM(**kwargs)

    values = dict(cosmology)
    values.update(kwargs)

    if "Omega_m" in values:
        values["Omega_c"] = values.pop("Omega_m") - values["Omega_b"]

    return ccl.Cosmology(**values)


def _validate_pk2d_grids(
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate wavenumber and scale-factor grids.

    Args:
        k_array: Wavenumber grid.
        a_array: Scale-factor grid.

    Returns:
        Validated arrays.
    """
    k_arr = validate_positive_strictly_increasing_1d_array(
        k_array, "k_array", min_size=2
    )
    a_arr = validate_positive_strictly_increasing_1d_array(
        a_array, "a_array", min_size=2
    )

    for a in a_arr:
        validate_scale_factor(float(a))

    return k_arr, a_arr


def baryonified_duffy_concentration(f_c: float) -> ccl.halos.ConcentrationDuffy08:
    """Construct a Duffy concentration model with baryonic rescaling.

    Args:
        f_c: Multiplicative concentration rescaling.

    Returns:
        Rescaled Duffy concentration model.
    """
    validate_positive_scalar(f_c, "f_c")
    return ccl.halos.ConcentrationDuffy08(mass_def=MASS_DEF, fc_bar=float(f_c))


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
        concentration: HOD concentration model.
        matter_profile: Optional matter profile using the same concentration.
        **hod_kwargs: HOD profile parameters.

    Returns:
        Dictionary containing ``pk_deltasigma``, ``pk_gm``, and ``pk_gg``.
    """
    k_arr, a_arr = _validate_pk2d_grids(k_array, a_array)

    galaxy_profile = ccl.halos.HaloProfileHOD(
        mass_def=MASS_DEF, concentration=concentration, **hod_kwargs
    )

    if matter_profile is None:
        matter_profile = ccl.halos.HaloProfileNFW(
            mass_def=MASS_DEF, concentration=concentration, fourier_analytic=True
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
    **hod_kwargs: Any,
) -> dict[str, ccl.Pk2D]:
    """Construct standard HOD power spectra.

    Args:
        cosmo: CCL cosmology.
        k_array: Wavenumber grid.
        a_array: Scale-factor grid.
        **hod_kwargs: HOD profile parameters.

    Returns:
        Density-weighted galaxy-matter, galaxy-matter, and galaxy-galaxy spectra.
    """
    return _pk2d_hod(
        cosmo,
        k_array=k_array,
        a_array=a_array,
        concentration=CONCENTRATION,
        matter_profile=MATTER_PROFILE,
        **hod_kwargs,
    )


def pk2d_hod_baryonified(
    cosmo: ccl.Cosmology,
    *,
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
    f_c: float = 1.0,
    **hod_kwargs: Any,
) -> dict[str, ccl.Pk2D]:
    """Construct HOD power spectra with baryonic concentration rescaling.

    Args:
        cosmo: CCL cosmology.
        k_array: Wavenumber grid.
        a_array: Scale-factor grid.
        f_c: Baryonic concentration rescaling.
        **hod_kwargs: HOD profile parameters.

    Returns:
        Density-weighted galaxy-matter, galaxy-matter, and galaxy-galaxy spectra.
    """
    return _pk2d_hod(
        cosmo,
        k_array=k_array,
        a_array=a_array,
        concentration=baryonified_duffy_concentration(f_c),
        **hod_kwargs,
    )


def hod_galaxy_bias(
    cosmo: ccl.Cosmology,
    *,
    a: float,
    **hod_kwargs: Any,
) -> float:
    """Calculate the occupation-weighted linear galaxy bias.

    Args:
        cosmo: CCL cosmology.
        a: Scale factor.
        **hod_kwargs: HOD profile parameters.

    Returns:
        Effective linear galaxy bias.
    """
    validate_scale_factor(a)

    galaxy_profile = ccl.halos.HaloProfileHOD(
        mass_def=MASS_DEF, concentration=CONCENTRATION, **hod_kwargs
    )
    n_g = galaxy_profile.get_normalization(cosmo, a, hmc=HM_CALCULATOR)

    if not np.isfinite(n_g) or n_g <= 0.0:
        raise ValueError("HOD galaxy normalization must be finite and positive.")

    def integrand(mass: NDArray[np.float64]) -> NDArray[np.float64]:
        bias = HM_CALCULATOR.halo_bias(cosmo, mass, a)
        return bias * (galaxy_profile._Nc(mass, a) + galaxy_profile._Ns(mass, a))

    weighted_bias = HM_CALCULATOR.integrate_over_massfunc(integrand, cosmo, a)
    return float(weighted_bias / n_g)


def pk2d_nla(
    cosmo: ccl.Cosmology,
    *,
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
    A_IA: float,
    C1rhocrit: float = 0.0134,
    b_g: float = 1.0,
) -> dict[str, ccl.Pk2D]:
    """Construct the NLA galaxy-intrinsic power spectrum.

    Args:
        cosmo: CCL cosmology.
        k_array: Wavenumber grid.
        a_array: Scale-factor grid.
        A_IA: Intrinsic alignment amplitude.
        C1rhocrit: NLA normalization.
        b_g: Galaxy bias.

    Returns:
        Dictionary containing density-weighted ``pk_deltasigma`` and ``pk_gi``.
    """
    k_arr, a_arr = _validate_pk2d_grids(k_array, a_array)

    validate_finite_scalar(A_IA, "A_IA")
    validate_finite_scalar(C1rhocrit, "C1rhocrit")
    validate_finite_scalar(b_g, "b_g")

    growth = np.asarray(cosmo.growth_factor(a_arr), dtype=float)

    if np.any(~np.isfinite(growth)) or np.any(growth <= 0.0):
        raise ValueError("growth_factor(a_array) must be finite and positive.")

    amplitude = float(A_IA) * float(C1rhocrit) * float(cosmo["Omega_m"]) / growth
    pk_mm = np.asarray(cosmo.nonlin_matter_power(k_arr, a_arr), dtype=float)

    if pk_mm.shape != (a_arr.size, k_arr.size):
        pk_mm = np.reshape(pk_mm, (a_arr.size, k_arr.size))

    if np.any(~np.isfinite(pk_mm)):
        raise ValueError("nonlinear matter power must contain only finite values.")

    pk_gi = -amplitude[:, None] * float(b_g) * pk_mm

    pk2d = ccl.Pk2D(
        a_arr=a_arr, lk_arr=np.log(k_arr), pk_arr=pk_gi, is_logp=False
    )

    def pk_gi_power(cosmo, k, a):
        return pk2d(k, a, cosmo=cosmo)

    return {
        "pk_deltasigma": density_weighted_power_spectrum(cosmo, pk_gi_power),
        "pk_gi": pk2d,
    }


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
        **hod_kwargs: HOD profile parameters.

    Returns:
        HOD spectra with NLA added to ``pk_deltasigma``.
    """
    validate_finite_scalar(A_IA, "A_IA")

    if float(A_IA) == 0.0:
        return pk_hod

    if b_g is None:
        b_g = hod_galaxy_bias(cosmo, a=a_bias, **hod_kwargs)

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
        **hod_kwargs: HOD profile parameters.

    Returns:
        Delta Sigma spectrum including NLA and HOD-only covariance spectra.
    """
    pk_hod = pk2d_hod(
        cosmo, k_array=k_array, a_array=a_array, **hod_kwargs
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
        **hod_kwargs: HOD profile parameters.

    Returns:
        Delta Sigma spectrum including NLA and baryonified HOD covariance spectra.
    """
    pk_hod = pk2d_hod_baryonified(
        cosmo, k_array=k_array, a_array=a_array, f_c=f_c, **hod_kwargs
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
        **hod_kwargs,
    )
