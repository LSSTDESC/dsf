"""Special functions and their integrals"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, gammaincc

__all__ = [
    "upper_inc_gamma",
    "safe_upper_inc_gamma",
]

def upper_inc_gamma(a, x):
    r"""Computes the upper incomplete gamma function :math:`\Gamma(a, x)`
    :math:`\forall a \in  \mathbb{R}` and :math:`x \ge 0`.

    .. math::

        \Gamma(a, x) = \int_x^\infty t^{a - 1}e^{-t} dt.

    This function uses the recurrence relation,

    .. math::
    
        \Gamma(a, x) = \frac{\Gamma(a+1, x) - x^a \cdot e^{-x}}{a}
    
    to handle :math:`a \lt 0` cases, where scipy's implementation returns nan.

    Parameters:
        a (float): The shape parameter, can be any real number.
        x (float): The lower limit of upper-gamma function integral.
        
    Returns: 
        a float
    """
    if x <= 0 and a <= 0:
        return np.inf
    if a > 0:
        return gammaincc(a, x) * gamma(a)
    return (upper_inc_gamma(a + 1, x) - (x ** a) * np.exp(-x)) / a

safe_upper_inc_gamma = np.vectorize(upper_inc_gamma, excluded=["a"])
