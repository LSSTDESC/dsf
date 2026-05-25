"""1D Hankel transform utilities.

This module provides functions for performing 1D Hankel transforms using the FFTLog algorithm. 

The public functions include:

1) ``hankel_j0``: Convert a 3D power spectrum to a 3D correlation function 
using the J0 Bessel function.

2) ``hankel_J2``: Convert a projected GGL power spectrum to a 2D tangential 
shear correlation function using the J2 Bessel function.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.fft import fhtoffset, ifht

from dsf.utils.types import FloatArray
from dsf.utils.validators import validate_hankel_1d_grid_spacing

__all__ = [
    "hankel_j0",
    "hankel_J2",
]


def hankel_j0(
    P_k: FloatArray,
    k: FloatArray,
    offset: bool = True,
) -> Callable[[FloatArray], FloatArray]:
    """Convert power spectrum to 3D correlation function using FFTLog:
    
    :math:`\\xi(r) = \\int \\frac{k^2 dk}{2\\pi^2} P(k) j_0(kr)`.

    Args:
        power_spectrum: Power spectrum to transform.
        k: Wavenumber array (must be uniform in logspace).
        offset: Optional flag to apply an offset to the lgoarithmic spacing 
            of the output. Can reduce numerical ringing.

    Returns:
        Function returning :math:`\\xi(r)` evaluated at the requested radii.
    """
    k_arr = validate_hankel_1d_grid_spacing(k, "k")
    
    r = 1.0 / k_arr[::-1]
    dln_k = float(np.log(k_arr[1] / k_arr[0]))
    offset = fhtoffset(dln=dln_k, mu=0.5) if offset else 0.0
    
    transformed_power = ifht(
        k_arr**1.5 * P_k,
        dln=dln_k,
        mu=0.5,
        offset=offset,
    )
    
    prefactor = 1.0 / (2.0 * np.pi * r) ** 1.5
    xi = np.asarray(prefactor * transformed_power, dtype=float)
    
    def correlation(r_eval: FloatArray) -> FloatArray:
        """Return the correlation function at the requested radii."""
        r_eval_arr = np.asarray(r_eval, dtype=float)

        return np.asarray(
            np.interp(
                r_eval_arr,
                r,
                xi,
                left=xi[0],
                right=xi[-1],
            ),
            dtype=float,
        )

    return correlation


def hankel_J2(
    C_ell: FloatArray,
    ell: FloatArray,
    offset: bool = True,
) -> Callable[[FloatArray], FloatArray]:
    """Convert projected GGL power spectrum to 2D correlation function using FFTLog:
    
    :math:`\\gamma_t(r) = \\int \\frac{\\ell d\\ell}{2\\pi} C(\\ell) J_2(\\ell \\theta)`.

    Args:
        C_ell: Power spectrum to transform.
        ell: ell array (must be uniform in logspace).
        offset: Optional flag to apply an offset to the lgoarithmic spacing 
            of the output. Can reduce numerical ringing.

    Returns:
        Function returning :math:`\\gamma_t(r)` evaluated at the requested radii.
    """
    ell_arr = validate_hankel_1d_grid_spacing(ell, "ell")
    
    theta = 1.0 / ell_arr[::-1]
    dln_ell = float(np.log(ell_arr[1] / ell_arr[0]))
    offset = fhtoffset(dln=dln_ell, mu=2) if offset else 0.0
    
    gammat = ifht(C_ell * ell_arr, 
                  dln=dln_ell, 
                  mu=2, 
                  offset=offset) / theta
    
    def gamma_t(r_eval: FloatArray) -> FloatArray:
        """Return the tangential shear correlation function at the requested radii."""
        r_eval_arr = np.asarray(r_eval, dtype=float)
        return np.asarray(
            np.interp(
                r_eval_arr,
                theta,
                gammat,
                left=gammat[0],
                right=gammat[-1],
            ),
            dtype=float,
        )

    return gamma_t