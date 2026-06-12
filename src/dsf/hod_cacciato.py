import numpy as np
import pyccl as ccl
from numpy.typing import NDArray
from scipy.special import erf

from dsf.utils.special_func import safe_upper_inc_gamma

__all__ = [
    "CacciatoHOD"
]

class CacciatoHOD(ccl.halos.HaloProfileHOD):
    r"""
    This class implements the galaxy-halo connection in the HOD framework based
    on the Conditional Luminosity Function (CLF) model of Cacciato et al. 2013.
    This framework is designed to model galaxy samples defined by luminosity
    bins.
    
    Args:
        mass_def: Halo mass definition (e.g., '200m' or a MassDef object).

        concentration: Concentration-mass relation (e.g., 'Duffy08' or a
            Concentration object). To apply the `eta` multiplicative factor,
            `concentration` should be a class wrapped around
            `pyccl.halos.concentration` that manages the multiplication.

        log_L1/log_L2: :math:`\log_{10}` (luminosity bin edge) of the lens
            sample being modelled. The edges are defined s.t. :math:`(\log_{10}L_2
            > \log_{10}L_1)`

    HOD model based on Cacciato+2013 has 9 free parameters as listed below:

    .. rubric:: The CLF Parameters

    * **log_L0**: 
      :math:`\log_{10}` of the normalization factor :math:`(L_0)`
      of the central galaxy luminosity-halo mass scaling relation, :math:`L_c(M)`.

    * **log_M1**: 
      :math:`\log_{10}` of the characteristic halo mass scale :math:`(M_1)`,
      such that :math:`L_c(M) \propto M^{\gamma_1}` for :math:`{\rm halo-mass},
      M \ll M_2`.

    * **gamma_1**: 
      slope of the central galaxy luminosity-halo mass scaling relation,
      :math:`L_c(M)` at the low-mass end :math:`(M \ll M1)`.

    * **gamma_2**: 
      slope of the central galaxy luminosity-halo mass scaling relation,
      :math:`L_c(M)` at the high-mass end :math:`(M \gg M1)`.

    * **sigma_c**: 
      scatter in the luminosities of central galaxies in the sample populating
      a fixed halo mass, :math:`M`.

    * **alpha_s**: 
      faint end of the slope of the satellite galaxy occupation number.
      Currently, assumed independent of (M,z), but can be generalised.

    * **p_cs**:
      The coupling parameter between the characteristic central luminosity :math:`L_c(M)` and
      the cut-off satellite luminosity :math:`L_s^{*}(M)` for a halo of mass M

    * **b_0, b_1, b_2**: 
      parameters affecting the overall scaling of the the satellite galaxy
      HOD/CLF/LF.

    Additional nusance parameters in the CLF model:

    * **eta**: 
      A multiplicative factor, as described in Cacciato+2013, which modifies
      the parent halo concentration as follows:

      .. math::

          c_\mathrm{halo}(M) = (1+\eta) \times c_\mathrm{baseline}(M)

      Here, :math:`c_\mathrm{baseline}` is the chosen c-M relation
      which gets marginalised-over by `eta`. `eta` acts via the c-M wrapper class
      passed above to the `concentration` parameter.

    * **R_s**: 
      A multiplicative factor, alters the concentration of the satellite
      distribution. R_s=1 corresponds to fiducial case, where satellites
      density distribution follows the NFW profile of the parent halo.

    Note: 
        HODs are unitless. But the characteristic masses and luminosities hold
        the information of units and scalings involved. Here, we require the
        masses and luminosities in :math:`M_\odot/h` and :math:`L_\odot/h^2` respectively to make
        sure that the Cacciato+13 CLF parameters can be directly passed.

        CCL HMF is given as dn/dlog10(m[:math:`M_\odot`]), so make sure to work with
        masses in :math:`M_\odot` units out of the HOD definition and use logarithmic base
        10 mass bins for integration.

        To use the Cacciato+2013 HOD parameters as is, all the methods in this class expect
        the mass to be in :math:`M_\odot/h`. So an internal conversion of pyCCL's :math:`M_\odot`
        mass is implemented where necessary.

        For Cacciato specific usecase, we do not vary free parameters in
        redshift dependent manner.

    """

    def __init__(
        self,
        *,
        # halo mass def
        mass_def: str | ccl.halos.MassDef, 
        # mass profile
        concentration: str | ccl.halos.Concentration, 
        # sample bin def
        log_L1, log_L2, 
        # hubble const
        h=0.739, 
        # CLF pars
        log_L0=9.95, log_M1=11.24, 
        gamma_1=3.18, gamma_2=0.245, p_cs = 0.562,
        sigma_c=0.157, alpha_s=-1.18, 
        b_0=-1.17, b_1=1.53, b_2=-0.217,
        # central prof pars
        eta=0.,
        # satellite prof pars
        R_s=1., 
    ):
        super().__init__(mass_def=mass_def, concentration=concentration)
        # disable (set to None) the vanilla HOD parameters not applicable to this CLF.
        self._disable_vanilla_hod_parameters()
        
        # define inputs that don't change throughout the analysis.
        self.h = h
        #self.cM = concentration #use if needed
        self.ns_independent = False
        # sample luminosity bin definition
        self.log_L1 = log_L1
        self.log_L2 = log_L2
        # update/define the CLF free parameters
        self.R_s = R_s
        bg_0 = 1./self.R_s
        # followings are to be fixed for Cacciato like work
        bg_p=0.
        bmax_0=1.
        bmax_p=0.
        self.update_parameters(
            log_L0=log_L0,log_M1=log_M1, gamma_1=gamma_1, gamma_2=gamma_2,
            sigma_c=sigma_c, alpha_s=alpha_s, b_0=b_0, b_1=b_1, b_2=b_2,
            bg_0=bg_0, bg_p=bg_p, bmax_0=bmax_0, bmax_p=bmax_p, eta=eta,
            p_cs=p_cs,
        )

    def _disable_vanilla_hod_parameters(self):
        """Nullifies the standard HOD parameters not utilized by the Cacciato CLF framework."""
        unused_attrs = [
            "log10Mmin_0", "log10Mmin_p", "log10M0_0", "log10M0_p",
            "log10M1_0", "log10M1_p", "siglnM_0", "siglnM_p", "alpha_0", "alpha_p"
        ]
        for attr in unused_attrs:
            setattr(self, attr, None)

    @property
    def log_L1(self):
        """float: The faint-end luminosity bin boundary in log10(Lsun/h^2)."""
        return self._log_L1

    @log_L1.setter
    def log_L1(self, log_L1):
        """Updates the faint-end luminosity bin boundary and triggers updates."""
        self._log_L1 = log_L1

    @property
    def log_L2(self):
        """float: The bright-end luminosity bin boundary in log10(Lsun/h^2)."""
        return self._log_L2

    @log_L2.setter
    def log_L2(self, log_L2):
        """Updates the bright-end luminosity bin boundary and triggers updates."""
        self._log_L2 = log_L2

    @property
    def h(self):
        """float: The Hubble parameter value"""
        return self._h
    @h.setter
    def h(self, h):
        """Updates the value of the Hubble parameter thoughout the analysis."""
        self._h = h
        # cache log10(h) whenever h is altered.
        self._log10_h = np.log10(h)

    def update_parameters(self, **kwargs):
        r"""Updates a selective set of CLF parameters or resets them to defaults.

        If an empty dictionary or no arguments are provided, the parameters will fall
        back to the default CLF parameters of Cacciato+13. Passing a subset of
        parameters will only update those specific values, keeping the remaining
        parameters at their previously instantiated values.

        This update does not affect the sample luminosity bin edges and
        hubble-constant definitions.

        Args: 
            kwargs: a dict of CLF free parameters (e.g., log_L0, log_M1, gamma_1)

        Note:
            To match the Cacciato parametrization exactly, set:
                * bg = bg_0 (i.e., bg_p = 0)
                * bmax = 1 (i.e., bmax_0 = 1, bmax_p = 0)
            
            And redefine:
                * 1 / b_g = R_s (where R_s is a multiplicative factor to the
                  fiducial c-M relation as defined in Cacciato, not a CCL
                  definition).
        """

        # Luminosity thresholds (calibrated as h-dependent: log10(L / [h^-2 Lsun]))
        self.log_L0 = kwargs.get('log_L0', getattr(self, 'log_L0', 9.95))
        # Mass scales (calibrated as h-dependent: log10(M / [h^-1 Msun]))
        self.log_M1 = kwargs.get('log_M1', getattr(self, 'log_M1', 11.24))
        # Dimensionless polynomial and shape parameters
        self.gamma_1 = kwargs.get('gamma_1', getattr(self, 'gamma_1', 3.18))
        self.gamma_2 = kwargs.get('gamma_2', getattr(self, 'gamma_2', 0.245))
        self.sigma_c = kwargs.get('sigma_c', getattr(self, 'sigma_c', 0.157))
        self.alpha_s = kwargs.get('alpha_s', getattr(self, 'alpha_s', -1.18))
        #self.b_0 = kwargs.get('b_0', getattr(self, 'b_0', -1.17))
        self.b_0 = kwargs.get('b_0', getattr(self, 'b_0', -1.17))
        self.b_1 = kwargs.get('b_1', getattr(self, 'b_1', 1.53))
        self.b_2 = kwargs.get('b_2', getattr(self, 'b_2', -0.217))
        self.p_cs = kwargs.get('p_cs', getattr(self, 'p_cs', 0.562))
        # satellite concentration bias parameters 
        self.R_s = kwargs.get('R_s', getattr(self, 'R_s', 1))
        self.bg_0 = 1.0 / self.R_s
        # central incompleteness modeling, as a constant value
        self.fc_0 = kwargs.get('fc_0', getattr(self, 'fc_0', 1.))
        self.fc_p = kwargs.get('fc_p', getattr(self, 'fc_p', 0.))
        self.a_pivot = kwargs.get('a_pivot', getattr(self, 'a_pivot', 0.))
        # parent halo concentration scaling injected into concentration-Mass relation
        self.concentration.eta = kwargs.get('eta', getattr(self, 'eta', 0.))

    def _log_Lc(self, M_h: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Calculates \log_{10} characteristic central luminosity as a
        function of halo mass M_h

        Args:
            M_h: Halo mass in units of Msun/h.

        Returns:
            The $\log_{10}$ of the characteristic central luminosity in units of 
            Lsun/h^2.
        """

        return (
            self.log_L0
            + self.gamma_1 * (np.log10(M_h) - self.log_M1)
            - (self.gamma_1 - self.gamma_2) * np.log10(1 + M_h / 10 ** self.log_M1)
        )

    def _log_L_star_s(self, M_h: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Calculates \log_{10} of characteristic satellite luminosity as
        a function of halo mass M_h

        This represents the cut-off luminosity of the satellite galaxies
        populating a halo of mass M_h.

        Args:
            M_h: Halo mass in Msun/h

        Returns:
            The $\log_{10}$ of the characteristic satellite luminosity in units
            of Lsun/h^2.
        """
        return np.log10(self.p_cs) + self._log_Lc(M_h)

    def _log_phi_star_s(self, M_h: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Calculates \log_{10} of the amplitude of satellite CLF as a function of halo
        mass M_h

        Args:
            M_h: Mass in Msun/h; Msun/h unit is required to extract the correct
            response from this function using Cacciato+2013 parameters

        Returns:
            \log_{10} of the amplitude of satellite CLF
        """
        # the dimensionless Cacciato scaled-mass parameter: log10( M / [10^12 * Msun/h] )
        log_M12 = np.log10(M_h) - 12.0
        return self.b_0 + self.b_1 * log_M12 + self.b_2 * log_M12 ** 2

    def _Nc(self, M: NDArray[np.float64], a=None) -> NDArray[np.float64]:
        """Calculates the central HOD: The number of central galaxies with
        luminosities between [L-dL, L+dL] populated in a halo of mass M.

        Args:
            M: Mass in Msun; internally converted to Msun/h

        Note:
            Because of the internal conversion of Msun to Msun/h units,
            constraints on the mass parameters will still be in units of
            Msun/h.
            
        Returns:
            Central HOD for the given CLF parameters
        """
        # internal conversion to Msun/h
        M = M * self.h

        log_Lc = self._log_Lc(M)
        denom = np.sqrt(2) * self.sigma_c
        term_max = (self.log_L2 - log_Lc) / denom
        term_min = (self.log_L1 - log_Lc) / denom
        return 0.5 * (erf(term_max) - erf(term_min))

    def _Ns(self, M: NDArray[np.float64], a=None) -> NDArray[np.float64]:
        # Here, adding`a` as a kwarg, but are not needed for this work. Putting
        # just to be able to run the delta_sigma_builder. Remove it in the
        # future.
        """Calculates the satellite HOD: The number of satellite galaxies with
        luminosities between [L-dL, L+dL] populated in a halo of mass M.

        Args:
            M: Halo mass in Msun; internally converted to Msun/h

        Note:
            Because of the internal conversion of Msun to Msun/h units,
            constraints on the mass parameters will still be in units of
            Msun/h.
            
        Returns:
            Satellite HOD for the given CLF parameters
        """
        # internal conversion to Msun/h
        M = M * self.h

        phi_star_s = 10 ** self._log_phi_star_s(M)
        log_Ls = self._log_L_star_s(M) #Lsun/h^2
        shape = (self.alpha_s + 1) / 2.0
        x_min = 2.0 * (self.log_L1 - log_Ls)
        x_max = 2.0 * (self.log_L2 - log_Ls)
        integral_min = safe_upper_inc_gamma(shape, 10 ** x_min)
        integral_max = safe_upper_inc_gamma(shape, 10 ** x_max)
        return (phi_star_s / 2.0) * (integral_min - integral_max)
