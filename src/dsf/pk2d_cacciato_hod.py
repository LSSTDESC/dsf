import numpy as np
import pyccl as ccl
from numpy.typing import NDArray

from dsf.data_vector.profiles import density_weighted_power_spectrum
from dsf.hod_cacciato import CacciatoHOD
from dsf.modelling import (
    _validate_pk2d_grids,
    make_ccl_cosmology,
)

__all__ = [
    "MASS_DEF",
    "CONCENTRATION",
    "MASS_FUNCTION",
    "HALO_BIAS",
    "MATTER_PROFILE",
    "HM_CALCULATOR",
    "make_ccl_cosmology",
    "pk2d_cacciato_hod",
    "ScaledConcentration"
]


class ScaledConcentration(ccl.halos.Concentration):
    """ A CCL concentration class wrapper that scales a baseline
    concentration-Mass relation via a free parameter `eta` as (1+eta)."""

    name = 'Scaled_cM_Cacciato2013'

    def __init__(self, baseline_cM, *, eta=1.0, rho_type="matter", **baseline_cM_kwargs):
        """
        baseline_cM: a pyccl.halos.concentration class
        baseline_cM_kwargs: Arguments for a pyccl.halos.concentration class
        eta: This multiplies the baseline c-M relation in the form (1+eta)
        """
        self._eta = eta
        self.mass_def = baseline_cM_kwargs.pop("mass_def")
        
        assert self.mass_def.rho_type == rho_type, "Inconsistent mass definition in concentration"

        print(f"Extra baseline concentration parameters: -- {baseline_cM_kwargs} --")
        self.baseline_relation = baseline_cM(mass_def=self.mass_def, **baseline_cM_kwargs)
        assert rho_type == self.baseline_relation.mass_def.rho_type 

        super().__init__(mass_def=self.mass_def)

    def _check_mass_def_strict(self, mass_def):
        """
        To exactly work with Cacciato+2013 conventions, mandate 200m
        """
        return mass_def.name not in ["200m"]

    def _concentration(self, cosmo, M, a):
        return (1.0+self.eta) * self.baseline_relation(cosmo, M, a)
    
    @property
    def eta(self):
        """float: The concentration-Mass relation modifier parameter that
        scales it by an overall factor of (1+eta)"""
        return self._eta
    @eta.setter
    def eta(self, eta):
        """Updates the scaling factor (1+eta) of the c-M relation.""" 
        self._eta = eta

MASS_DEF = ccl.halos.MassDef200m

CONCENTRATION = ScaledConcentration(
    ccl.halos.ConcentrationDuffy08,
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

def pk2d_cacciato_hod(
    cosmo: ccl.Cosmology,
    *,
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
    **hod_kwargs,
) -> ccl.Pk2D:
    """Return the HOD galaxy-matter power spectrum for Delta Sigma.

    Args:
        cosmo: CCL cosmology object.
        k_array: Wavenumber grid used for the halo-model ``Pk2D`` spline.
        a_array: Scale-factor grid used for the halo-model ``Pk2D`` spline.
        **hod_kwargs: Keyword arguments passed to ``CacciatoHOD``.

    Returns:
        Density-weighted galaxy-matter ``Pk2D`` object.
    """
    k_arr, a_arr = _validate_pk2d_grids(k_array, a_array)

    galaxy_profile = CacciatoHOD(
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


