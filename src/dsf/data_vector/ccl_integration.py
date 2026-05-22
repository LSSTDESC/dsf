"""CCL integration scripts for generic Delta Sigma profiles."""

from __future__ import annotations

import numpy as np
import pyccl as ccl
from numpy.typing import NDArray


class HaloProfileGeneric(ccl.halos.profiles.profile_base.HaloProfile):
    """CCL halo-profile wrapper around a generic Pk2D.

    This class uses CCL's HaloProfile API only as a projection backend.
    The mass definition, concentration, and input mass are dummy values
    required by the API and are not physically used by this wrapper.
    """

    def __init__(
        self,
        *,
        pk2d: ccl.Pk2D,
        padding_hi_fftlog: float = 1.0e3,
        padding_lo_fftlog: float = 1.0e-2,
        n_per_decade: int = 5000,
    ) -> None:
        """Initialize the generic CCL profile.

        Args:
            pk2d: CCL ``Pk2D`` object used as the Fourier-space profile.
            padding_hi_fftlog: High-k FFTLog padding used by CCL.
            padding_lo_fftlog: Low-k FFTLog padding used by CCL.
            n_per_decade: FFTLog sampling density.
        """
        self.pk2d = pk2d

        self.fourier_analytic = True
        self._fourier = self._fourier_analytic
        self.projected_analytic = False
        self.cumul2d_analytic = False

        super().__init__(
            mass_def="200m",
            concentration=ccl.halos.concentration.ConcentrationConstant(
                5.0,
                mass_def="200m",
            ),
        )

        self.update_precision_fftlog(
            padding_hi_fftlog=padding_hi_fftlog,
            padding_lo_fftlog=padding_lo_fftlog,
            n_per_decade=n_per_decade,
        )

    def _fourier_analytic(
        self,
        cosmo: ccl.Cosmology,
        k: float | NDArray[np.float64],
        M: float | NDArray[np.float64],
        a: float,
    ) -> NDArray[np.float64]:
        """Evaluate the stored ``Pk2D`` object as a Fourier-space profile.

        Args:
            cosmo: CCL cosmology object. Included for compatibility with the
                CCL ``HaloProfile`` API.
            k: Wavenumber in 1/Mpc.
            M: Halo mass in solar masses. This wrapper only supports a single
                dummy mass value.
            a: Scale factor.

        Returns:
            Fourier-space profile values with shape ``(1, n_k)``.
        """
        _ = cosmo  # Required by the CCL HaloProfile API, unused here.

        if np.size(M) != 1:
            raise ValueError("M must be a single value for HaloProfileGeneric.")

        k_arr = np.atleast_1d(np.asarray(k, dtype=float))
        values = np.asarray(self.pk2d(k_arr, a), dtype=float)

        return values.reshape((1, k_arr.size))
