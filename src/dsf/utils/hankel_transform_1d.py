"""1D Hankel transform utilities.

This module provides functions for performing 1D Hankel transforms using the FFTLog algorithm. 

The public functions include:

1) ``hankel_spherical_order_0``: Convert a 3D power spectrum to a 3D correlation function 
using the J0 Bessel function.

2) ``hankel_projected_order_2``: Convert a projected GGL power spectrum to a 2D tangential 
shear correlation function using the J2 Bessel function.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.fft import fhtoffset, ifht

from dsf.utils.types import FloatArray
from dsf.utils.validators import (
    as_1d_float_array,
    validate_hankel_1d_grid_spacing,
    validate_interpolation_within_bounds,
    validate_power_spectrum_inputs,
)

__all__ = [
    "hankel_spherical_order_0",
    "hankel_projected_order_2",
]


def hankel_spherical_order_0(
    k: FloatArray,
    pk: FloatArray,
    use_offset: bool = True,
) -> Callable[[FloatArray], FloatArray]:
    """Convert power spectrum to 3D correlation function using FFTLog:
    
    :math:`\\xi(r) = \\int \\frac{k^2 dk}{2\\pi^2} P(k) j_0(kr)`.

    Args:
        k: Wavenumber array (must be uniform in logspace).
        pk: Power spectrum to transform.
        use_offset: Optional flag to apply an offset to the logarithmic spacing 
            of the output. Can reduce numerical ringing.

    Returns:
        Function returning :math:`\\xi(r)` evaluated at the requested radii.
    """
    k_arr = validate_hankel_1d_grid_spacing(k, "k")
    pk_arr = as_1d_float_array(pk, "pk", min_size=2)
    validate_power_spectrum_inputs(k_arr, pk_arr)

    r = 1.0 / k_arr[::-1]
    dln_k = float(np.log(k_arr[1] / k_arr[0]))
    offset = fhtoffset(dln=dln_k, mu=0.5) if use_offset else 0.0
    
    transformed_power = ifht(
        k_arr**1.5 * pk_arr,
        dln=dln_k,
        mu=0.5,
        offset=offset,
    )
    
    prefactor = 1.0 / (2.0 * np.pi * r) ** 1.5
    xi = np.asarray(prefactor * transformed_power, dtype=float)
    
    def correlation(r_eval: FloatArray) -> FloatArray:
        """Return the correlation function at the requested radii."""
        r_eval_arr = validate_interpolation_within_bounds(r_eval, r, "r")

        return np.asarray(
            np.interp(
                r_eval_arr,
                r,
                xi,
            ),
            dtype=float,
        )

    return correlation


def hankel_projected_order_2(
    ell: FloatArray,
    c_ell: FloatArray,
    use_offset: bool = True,
) -> Callable[[FloatArray], FloatArray]:
    """Convert projected GGL power spectrum to 2D correlation function using FFTLog:
    
    :math:`\\gamma_t(\\theta) = \\int \\frac{\\ell d\\ell}{2\\pi} C(\\ell) J_2(\\ell \\theta)`.

    Args:
        ell: ell array (must be uniform in logspace).
        c_ell: Power spectrum to transform.
        use_offset: Optional flag to apply an offset to the logarithmic spacing 
            of the output. Can reduce numerical ringing.

    Returns:
        Function returning :math:`\\gamma_t(r)` evaluated at the requested radii.
    """
    ell_arr = validate_hankel_1d_grid_spacing(ell, "ell")
    c_ell_arr = as_1d_float_array(c_ell, "c_ell", min_size=2)
    validate_power_spectrum_inputs(ell_arr, c_ell_arr, k_name='ell', pk_name='c_ell')

    theta = 1.0 / ell_arr[::-1]
    dln_ell = float(np.log(ell_arr[1] / ell_arr[0]))
    offset = fhtoffset(dln=dln_ell, mu=2) if use_offset else 0.0
    
    gammat = ifht(c_ell_arr * ell_arr, 
                  dln=dln_ell, 
                  mu=2, 
                  offset=offset) / theta / (2.0 * np.pi)
    
    def gamma_t(theta_eval: FloatArray) -> FloatArray:
        """Return the tangential shear correlation function at the requested radii."""
        theta_eval_arr = validate_interpolation_within_bounds(theta_eval, theta, "theta")
        
        return np.asarray(
            np.interp(
                theta_eval_arr,
                theta,
                gammat,
            ),
            dtype=float,
        )

    return gamma_t