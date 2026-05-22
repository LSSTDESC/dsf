"""Galaxy-bias scripts for covariance calculations.

This module provides small utilities for constructing simple galaxy-bias
inputs from tomographic lens-bin redshifts. The functions are intentionally
independent of survey presets so they can be used with any tomography builder
or user-supplied bin centers.
"""

from __future__ import annotations

import numpy as np
import pyccl as ccl

from src.dsf.utils.converters import redshift_to_scale_factor
from src.dsf.utils.types import ArrayLike, FloatArray
from src.dsf.utils.validators import (
    validate_nonnegative_1d_array,
    validate_positive_scalar,
)

__all__ = [
    "linear_galaxy_bias",
]


def linear_galaxy_bias(
    cosmo: ccl.Cosmology,
    z_bin_centers: ArrayLike,
    *,
    bias_prefactor: float = 1.0,
    round_decimals: int | None = 4,
) -> FloatArray:
    """Return linear galaxy bias evaluated at lens-bin centers.

    The model uses a constant prefactor divided by the linear growth factor,
    evaluated at the scale factor corresponding to each lens-bin center.

    Args:
        cosmo: CCL cosmology object.
        z_bin_centers: Lens-bin center redshifts.
        bias_prefactor: Bias normalization prefactor. Defaults to 1.0.
        round_decimals: Optional number of decimals used to round the output.
            Set to ``None`` to return unrounded values.

    Returns:
        Galaxy-bias values evaluated at the input redshifts.
    """
    validate_positive_scalar(bias_prefactor, "bias_prefactor")

    z = validate_nonnegative_1d_array(z_bin_centers, "z_bin_centers")
    scale_factor = redshift_to_scale_factor(z)

    growth_factor = np.asarray(ccl.growth_factor(cosmo, scale_factor), dtype=float)

    if np.any(growth_factor <= 0.0):
        raise ValueError("growth_factor must contain only positive values.")

    galaxy_bias = bias_prefactor / growth_factor

    if round_decimals is not None:
        galaxy_bias = np.round(galaxy_bias, int(round_decimals))

    return np.asarray(galaxy_bias, dtype=float)
