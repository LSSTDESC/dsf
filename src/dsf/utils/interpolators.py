"""Interpolation scripts for Delta Sigma calculations."""

from __future__ import annotations

import numpy as np

from .types import FloatArray
from .validators import (
    as_1d_float_array,
    validate_interpolation_within_bounds,
    validate_positive_1d_array,
    validate_positive_strictly_increasing_1d_array,
    validate_strictly_increasing,
)

__all__ = [
    "interpolate_linear",
    "interpolate_loglog",
]


def interpolate_linear(
    x: FloatArray,
    xp: FloatArray,
    fp: FloatArray,
    *,
    x_name: str = "x",
    xp_name: str = "xp",
    fp_name: str = "fp",
) -> FloatArray:
    """Interpolate a 1D function linearly between tabulated points.

    Args:
        x: requested interpolation points of the independent variable.
        xp: tabulated points of the independent variable.
        fp: tabulated points of the dependent variable.
        x_name: Name of the independent variable for error messages.
        xp_name: Name of the independent variable grid for error messages.
        fp_name: Name of the dependent variable grid for error messages.

    Returns:
        Interpolated values of the dependent variable at the requested points.
    """
    xp_arr = validate_strictly_increasing(xp, xp_name)
    x_arr = validate_interpolation_within_bounds(x, xp, x_name)
    fp_arr = as_1d_float_array(fp, fp_name)

    return np.interp(x_arr, xp_arr, fp_arr)


def interpolate_loglog(
    x: FloatArray,
    xp: FloatArray,
    fp: FloatArray,
    *,
    x_name: str = "x",
    xp_name: str = "xp",
    fp_name: str = "fp",
) -> FloatArray:
    """Interpolate a 1D function on a log-log scale between tabulated points.

    Args:
        x: requested interpolation points of the independent variable.
        xp: tabulated points of the independent variable.
        fp: tabulated points of the dependent variable.
        x_name: Name of the independent variable for error messages.
        xp_name: Name of the independent variable grid for error messages.
        fp_name: Name of the dependent variable grid for error messages.

    Returns:
        Interpolated values of the dependent variable at the requested points.
    """
    xp_arr = validate_positive_strictly_increasing_1d_array(xp, xp_name)
    x_arr = validate_interpolation_within_bounds(x, xp, x_name)
    fp_arr = validate_positive_1d_array(fp, fp_name)

    log_x = np.log(x_arr)
    log_xp = np.log(xp_arr)
    log_fp = np.log(fp_arr)

    log_interp = np.interp(log_x, log_xp, log_fp)

    return np.exp(log_interp)
