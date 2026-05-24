"""Power-spectrum-backed HOD and IA models for Delta Sigma forecasts.

This module contains reusable ``Pk2D`` model components that can be passed to
``DeltaSigmaForecastBuilder`` through its ``pk2d_func`` argument.

The basic two-halo density profiles live in ``dsf.data_vector.profiles``.
Additive DeltaSigma-level terms, such as stellar point mass and lens
magnification bias, live in the data-vector and magnification-bias modules.

The functions here are for model components that naturally enter through a CCL
``Pk2D`` object, such as HOD galaxy-matter spectra, baryonified HOD spectra,
and NLA galaxy-intrinsic spectra.
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
    "baryonified_duffy_concentration",
    "hod_galaxy_bias",
    "make_ccl_cosmology",
    "pk2d_hod",
    "pk2d_hod_baryonified",
    "pk2d_hod_baryonified_with_nla",
    "pk2d_hod_with_nla",
    "pk2d_nla",
]


def make_ccl_cosmology(
    cosmology: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ccl.Cosmology:
    """Return a CCL cosmology.

    If no cosmology mapping is supplied, this returns
    ``ccl.CosmologyVanillaLCDM(**kwargs)``. This is useful for quick tests and
    examples.

    If a cosmology mapping is supplied, the values are passed to
    ``ccl.Cosmology``. The only DSF convenience handled here is allowing
    ``Omega_m`` instead of ``Omega_c``. In that case ``Omega_c`` is computed as

    ``Omega_c = Omega_m - Omega_b``.

    Args:
        cosmology: Optional mapping of cosmological parameters.
        **kwargs: Extra keyword arguments passed to the CCL cosmology
            constructor.

    Returns:
        CCL cosmology object.
    """
    if cosmology is None:
        return ccl.CosmologyVanillaLCDM(**kwargs)

    values = dict(cosmology)
    values.update(kwargs)

    if "Omega_m" in values:
        omega_m = values.pop("Omega_m")
        omega_b = values["Omega_b"]
        values["Omega_c"] = omega_m - omega_b

    return ccl.Cosmology(**values)


MASS_DEF = ccl.halos.MassDef200c

CONCENTRATION = ccl.halos.ConcentrationDuffy08(
    mass_def=MASS_DEF,
)

MASS_FUNCTION = ccl.halos.MassFuncTinker10(
    mass_def=MASS_DEF,
)

HALO_BIAS = ccl.halos.HaloBiasTinker10(
    mass_def=MASS_DEF,
)

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


def _validate_pk2d_grids(
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and return grids used to construct a ``Pk2D`` object.

    Args:
        k_array: Wavenumber grid.
        a_array: Scale-factor grid.

    Returns:
        Validated ``k`` and ``a`` arrays.
    """
    k_arr = validate_positive_strictly_increasing_1d_array(
        k_array,
        "k_array",
        min_size=2,
    )
    a_arr = validate_positive_strictly_increasing_1d_array(
        a_array,
        "a_array",
        min_size=2,
    )

    for a in a_arr:
        validate_scale_factor(float(a))

    return k_arr, a_arr


def baryonified_duffy_concentration(f_c: float) -> ccl.halos.ConcentrationDuffy08:
    """Return a Duffy concentration model with baryonic rescaling.

    Args:
        f_c: Multiplicative baryonic concentration rescaling.

    Returns:
        CCL Duffy concentration model.
    """
    validate_positive_scalar(f_c, "f_c")

    return ccl.halos.ConcentrationDuffy08(
        mass_def=MASS_DEF,
        fc_bar=float(f_c),
    )


def pk2d_hod(
    cosmo: ccl.Cosmology,
    *,
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
    **hod_kwargs: Any,
) -> ccl.Pk2D:
    """Return the HOD galaxy-matter power spectrum for Delta Sigma.

    Args:
        cosmo: CCL cosmology object.
        k_array: Wavenumber grid used for the halo-model ``Pk2D`` spline.
        a_array: Scale-factor grid used for the halo-model ``Pk2D`` spline.
        **hod_kwargs: Keyword arguments passed to ``ccl.halos.HaloProfileHOD``.

    Returns:
        Density-weighted galaxy-matter ``Pk2D`` object.
    """
    k_arr, a_arr = _validate_pk2d_grids(k_array, a_array)

    galaxy_profile = ccl.halos.HaloProfileHOD(
        mass_def=MASS_DEF,
        concentration=CONCENTRATION,
        **hod_kwargs,
    )

    pk_gm = ccl.halos.halomod_Pk2D(
        cosmo,
        HM_CALCULATOR,
        galaxy_profile,
        prof2=MATTER_PROFILE,
        lk_arr=np.log(k_arr),
        a_arr=a_arr,
        p_of_k_a=cosmo.get_nonlin_power(),
    )

    def pk_gm_power(cosmo, k, a):
        return pk_gm(k, a, cosmo=cosmo)

    return density_weighted_power_spectrum(cosmo, pk_gm_power)


def pk2d_hod_baryonified(
    cosmo: ccl.Cosmology,
    *,
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
    f_c: float = 1.0,
    **hod_kwargs: Any,
) -> ccl.Pk2D:
    """Return a baryonified HOD galaxy-matter power spectrum.

    Args:
        cosmo: CCL cosmology object.
        k_array: Wavenumber grid used for the halo-model ``Pk2D`` spline.
        a_array: Scale-factor grid used for the halo-model ``Pk2D`` spline.
        f_c: Baryonic rescaling of the Duffy concentration relation.
        **hod_kwargs: Keyword arguments passed to ``ccl.halos.HaloProfileHOD``.

    Returns:
        Density-weighted baryonified HOD galaxy-matter ``Pk2D`` object.
    """
    k_arr, a_arr = _validate_pk2d_grids(k_array, a_array)
    concentration = baryonified_duffy_concentration(f_c)

    galaxy_profile = ccl.halos.HaloProfileHOD(
        mass_def=MASS_DEF,
        concentration=concentration,
        **hod_kwargs,
    )

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

    def pk_gm_power(cosmo, k, a):
        return pk_gm(k, a, cosmo=cosmo)

    return density_weighted_power_spectrum(cosmo, pk_gm_power)


def hod_galaxy_bias(
    cosmo: ccl.Cosmology,
    *,
    a: float,
    **hod_kwargs: Any,
) -> float:
    """Return the effective linear galaxy bias of the HOD sample.

    Args:
        cosmo: CCL cosmology object.
        a: Scale factor where the HOD bias is evaluated.
        **hod_kwargs: Keyword arguments passed to ``ccl.halos.HaloProfileHOD``.

    Returns:
        Effective HOD galaxy bias.
    """
    validate_scale_factor(a)

    galaxy_profile = ccl.halos.HaloProfileHOD(
        mass_def=MASS_DEF,
        concentration=CONCENTRATION,
        **hod_kwargs,
    )

    n_g = galaxy_profile.get_normalization(
        cosmo,
        a,
        hmc=HM_CALCULATOR,
    )

    if not np.isfinite(n_g) or n_g <= 0.0:
        raise ValueError("HOD galaxy normalization must be finite and positive.")

    def integrand(mass: NDArray[np.float64]) -> NDArray[np.float64]:
        bias = HM_CALCULATOR.halo_bias(cosmo, mass, a)
        centrals = galaxy_profile._Nc(mass, a)
        satellites = galaxy_profile._Ns(mass, a)

        return bias * (centrals + satellites)

    weighted_bias = HM_CALCULATOR.integrate_over_massfunc(
        integrand,
        cosmo,
        a,
    )

    return float(weighted_bias / n_g)


def pk2d_nla(
    cosmo: ccl.Cosmology,
    *,
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
    A_IA: float,
    C1rhocrit: float = 0.0134,
    b_g: float = 1.0,
) -> ccl.Pk2D:
    """Return the NLA galaxy-intrinsic power spectrum contribution.

    Args:
        cosmo: CCL cosmology object.
        k_array: Wavenumber grid used for the ``Pk2D`` spline.
        a_array: Scale-factor grid used for the ``Pk2D`` spline.
        A_IA: Intrinsic-alignment amplitude.
        C1rhocrit: NLA normalization.
        b_g: Galaxy bias multiplying the galaxy-intrinsic term.

    Returns:
        Density-weighted NLA galaxy-intrinsic ``Pk2D`` object.
    """
    k_arr, a_arr = _validate_pk2d_grids(k_array, a_array)

    validate_finite_scalar(A_IA, "A_IA")
    validate_finite_scalar(C1rhocrit, "C1rhocrit")
    validate_finite_scalar(b_g, "b_g")

    growth = np.asarray(cosmo.growth_factor(a_arr), dtype=float)

    if np.any(~np.isfinite(growth)) or np.any(growth <= 0.0):
        raise ValueError("growth_factor(a_array) must be finite and positive.")

    amplitude = float(A_IA) * float(C1rhocrit) * float(cosmo["Omega_m"]) / growth

    pk_mm = np.asarray(
        cosmo.nonlin_matter_power(k_arr, a_arr),
        dtype=float,
    )

    if pk_mm.shape != (a_arr.size, k_arr.size):
        pk_mm = np.reshape(pk_mm, (a_arr.size, k_arr.size))

    if np.any(~np.isfinite(pk_mm)):
        raise ValueError("nonlinear matter power must contain only finite values.")

    pk_gi = -amplitude[:, None] * float(b_g) * pk_mm

    pk2d = ccl.Pk2D(
        a_arr=a_arr,
        lk_arr=np.log(k_arr),
        pk_arr=pk_gi,
        is_logp=False,
    )

    def pk_gi_power(cosmo, k, a):
        return pk2d(k, a, cosmo=cosmo)

    return density_weighted_power_spectrum(cosmo, pk_gi_power)


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
) -> ccl.Pk2D:
    """Return HOD galaxy-matter power plus optional NLA contribution.

    Args:
        cosmo: CCL cosmology object.
        k_array: Wavenumber grid used for the ``Pk2D`` spline.
        a_array: Scale-factor grid used for the ``Pk2D`` spline.
        A_IA: Intrinsic-alignment amplitude. If zero, only the HOD
            galaxy-matter contribution is returned.
        C1rhocrit: NLA normalization.
        b_g: Galaxy bias multiplying the galaxy-intrinsic term. If ``None``,
            the effective HOD galaxy bias is computed at ``a_bias``.
        a_bias: Scale factor used when deriving ``b_g`` from the HOD model.
        **hod_kwargs: Keyword arguments passed to ``ccl.halos.HaloProfileHOD``.

    Returns:
        Density-weighted HOD galaxy-matter ``Pk2D`` object, optionally including
        the NLA galaxy-intrinsic contribution.
    """
    validate_finite_scalar(A_IA, "A_IA")

    pk_hod = pk2d_hod(
        cosmo,
        k_array=k_array,
        a_array=a_array,
        **hod_kwargs,
    )

    if float(A_IA) == 0.0:
        return pk_hod

    if b_g is None:
        b_g = hod_galaxy_bias(
            cosmo,
            a=a_bias,
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

    return pk_hod + pk_ia


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
) -> ccl.Pk2D:
    """Return baryonified HOD galaxy-matter power plus optional NLA contribution.

    Args:
        cosmo: CCL cosmology object.
        k_array: Wavenumber grid used for the ``Pk2D`` spline.
        a_array: Scale-factor grid used for the ``Pk2D`` spline.
        f_c: Baryonic rescaling of the Duffy concentration relation.
        A_IA: Intrinsic-alignment amplitude. If zero, only the baryonified HOD
            galaxy-matter contribution is returned.
        C1rhocrit: NLA normalization.
        b_g: Galaxy bias multiplying the galaxy-intrinsic term. If ``None``,
            the effective HOD galaxy bias is computed at ``a_bias``.
        a_bias: Scale factor used when deriving ``b_g`` from the HOD model.
        **hod_kwargs: Keyword arguments passed to ``ccl.halos.HaloProfileHOD``.

    Returns:
        Density-weighted baryonified HOD galaxy-matter ``Pk2D`` object,
        optionally including the NLA galaxy-intrinsic contribution.
    """
    validate_finite_scalar(A_IA, "A_IA")

    pk_hod = pk2d_hod_baryonified(
        cosmo,
        k_array=k_array,
        a_array=a_array,
        f_c=f_c,
        **hod_kwargs,
    )

    if float(A_IA) == 0.0:
        return pk_hod

    if b_g is None:
        b_g = hod_galaxy_bias(
            cosmo,
            a=a_bias,
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

    return pk_hod + pk_ia
