"""Validation scripts."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from dsf.utils.types import FloatArray

__all__ = [
    "as_1d_float_array",
    "as_2d_float_array",
    "is_non_negative_integer",
    "normalize_axis",
    "occupied_redshift_range_from_nz",
    "redshift_window_mask",
    "validate_1d_pair",
    "validate_finite_scalar",
    "validate_forecast_vector_and_covariance",
    "validate_hankel_1d_grid_spacing",
    "validate_integration_axis",
    "validate_integration_params",
    "validate_interpolation_within_bounds",
    "validate_joint_covariance_blocks",
    "validate_nonnegative_1d_array",
    "validate_nonnegative_scalar",
    "validate_parameter_names",
    "validate_positive_1d_array",
    "validate_positive_scalar",
    "validate_positive_strictly_increasing_1d_array",
    "validate_power_spectrum_inputs",
    "validate_redshift_distribution",
    "validate_redshift_edges",
    "validate_redshift_distribution_support",
    "validate_redshift_pair",
    "validate_scale_factor",
    "validate_strictly_increasing",
]


def validate_finite_scalar(value: float, name: str) -> None:
    """Validate that a scalar is finite.

    Args:
        value: Scalar value to validate.
        name: Name used in error messages.

    Raises:
        ValueError: If ``value`` is not finite.
    """
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def validate_positive_scalar(value: float, name: str) -> None:
    """Validate that a scalar is finite and positive.

    Args:
        value: Scalar value to validate.
        name: Name used in error messages.

    Raises:
        ValueError: If ``value`` is not finite or is not positive.
    """
    validate_finite_scalar(value, name)

    if value <= 0.0:
        raise ValueError(f"{name} must be positive.")


def validate_nonnegative_scalar(value: float, name: str) -> None:
    """Validate that a scalar is finite and non-negative.

    Args:
        value: Scalar value to validate.
        name: Name used in error messages.

    Raises:
        ValueError: If ``value`` is not finite or is negative.
    """
    validate_finite_scalar(value, name)

    if value < 0.0:
        raise ValueError(f"{name} must be non-negative.")


def validate_scale_factor(a: float) -> None:
    """Validate a cosmological scale factor.

    Args:
        a: Scale factor.

    Raises:
        ValueError: If ``a`` is not in the interval ``0 < a <= 1``.
    """
    validate_finite_scalar(a, "a")

    if a <= 0.0 or a > 1.0:
        raise ValueError("a must satisfy 0 < a <= 1.")


def validate_redshift_pair(z_lens: float, z_source: float) -> None:
    """Validate a lens/source redshift pair.

    Args:
        z_lens: Lens redshift.
        z_source: Source redshift.

    Raises:
        ValueError: If redshifts are invalid or the source is not behind the lens.
    """
    validate_nonnegative_scalar(z_lens, "z_lens")
    validate_finite_scalar(z_source, "z_source")

    if z_source <= z_lens:
        raise ValueError("z_source must be greater than z_lens.")


def as_1d_float_array(
    values: Any,
    name: str,
    *,
    min_size: int = 1,
) -> NDArray[np.float64]:
    """Return values as a finite one-dimensional float array.

    Args:
        values: Input values.
        name: Name used in error messages.
        min_size: Minimum allowed number of values.

    Returns:
        Finite one-dimensional float array.

    Raises:
        ValueError: If the input is not one-dimensional, too short, or non-finite.
    """
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional. Got shape {array.shape}.")
    if array.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} values.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def as_2d_float_array(
    values: Any,
    name: str,
) -> NDArray[np.float64]:
    """Return values as a finite two-dimensional float array.

    Args:
        values: Input values.
        name: Name used in error messages.

    Returns:
        Finite two-dimensional float array.

    Raises:
        ValueError: If the input is not two-dimensional or contains non-finite
            values.
    """
    array = np.asarray(values, dtype=float)

    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional. Got shape {array.shape}.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")

    return array


def validate_positive_1d_array(
    values: Any,
    name: str,
    *,
    min_size: int = 1,
) -> NDArray[np.float64]:
    """Return values as a positive one-dimensional float array.

    Args:
        values: Input values.
        name: Name used in error messages.
        min_size: Minimum allowed number of values.

    Returns:
        Positive one-dimensional float array.

    Raises:
        ValueError: If the input is invalid or contains non-positive values.
    """
    array = as_1d_float_array(values, name, min_size=min_size)

    if np.any(array <= 0.0):
        raise ValueError(f"{name} must contain only positive values.")

    return array


def validate_nonnegative_1d_array(
    values: Any,
    name: str,
    *,
    min_size: int = 1,
) -> NDArray[np.float64]:
    """Return values as a non-negative one-dimensional float array.

    Args:
        values: Input values.
        name: Name used in error messages.
        min_size: Minimum allowed number of values.

    Returns:
        Non-negative one-dimensional float array.

    Raises:
        ValueError: If the input is invalid or contains negative values.
    """
    array = as_1d_float_array(values, name, min_size=min_size)

    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative.")

    return array


def validate_strictly_increasing(
    values: Any,
    name: str,
    *,
    min_size: int = 2,
) -> NDArray[np.float64]:
    """Return values as a strictly increasing one-dimensional float array.

    Args:
        values: Input values.
        name: Name used in error messages.
        min_size: Minimum allowed number of values.

    Returns:
        Strictly increasing one-dimensional float array.

    Raises:
        ValueError: If the input is invalid or is not strictly increasing.
    """
    array = as_1d_float_array(values, name, min_size=min_size)

    if np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")

    return array


def validate_positive_strictly_increasing_1d_array(
    values: Any,
    name: str,
    *,
    min_size: int = 2,
) -> NDArray[np.float64]:
    """Return values as a positive strictly increasing one-dimensional array.

    This is useful for physical coordinate grids such as wavenumber, radius, or
    frequency grids where values must be finite, positive, and ordered.

    Args:
        values: Input coordinate values.
        name: Name used in error messages.
        min_size: Minimum allowed number of values.

    Returns:
        Positive strictly increasing one-dimensional float array.

    Raises:
        ValueError: If the input is invalid, non-positive, or not strictly
            increasing.
    """
    array = validate_positive_1d_array(values, name, min_size=min_size)

    if np.any(np.diff(array) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")

    return array


def validate_1d_pair(
    x: Any,
    y: Any,
    *,
    x_name: str = "x",
    y_name: str = "y",
) -> None:
    """Validate two matching one-dimensional arrays.

    Args:
        x: First array. This array must be strictly increasing.
        y: Second array.
        x_name: Name of the first array for error messages.
        y_name: Name of the second array for error messages.

    Raises:
        ValueError: If arrays are not finite, one-dimensional, matching, or if
            ``x`` is not strictly increasing.
    """
    x_arr = validate_strictly_increasing(x, x_name, min_size=1)
    y_arr = as_1d_float_array(y, y_name, min_size=1)

    if x_arr.shape != y_arr.shape:
        raise ValueError(
            f"{x_name} and {y_name} must have matching shapes. Got {x_arr.shape} and {y_arr.shape}."
        )


def validate_joint_covariance_blocks(
    cov_gm_gm: Any,
    cov_gg_gg: Any,
    cov_gm_gg: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Validate and return covariance blocks for a joint Delta Sigma covariance.

    Args:
        cov_gm_gm: Auto-covariance block for the galaxy-matter signal.
        cov_gg_gg: Auto-covariance block for the galaxy-galaxy signal.
        cov_gm_gg: Cross-covariance block between the two signals.

    Returns:
        Validated covariance blocks as finite float arrays.

    Raises:
        ValueError: If any block has an invalid shape.
    """
    cov_gm_gm_arr = as_2d_float_array(cov_gm_gm, "cov_gm_gm")
    cov_gg_gg_arr = as_2d_float_array(cov_gg_gg, "cov_gg_gg")
    cov_gm_gg_arr = as_2d_float_array(cov_gm_gg, "cov_gm_gg")

    if cov_gm_gm_arr.shape[0] != cov_gm_gm_arr.shape[1]:
        raise ValueError("cov_gm_gm must be square.")
    if cov_gg_gg_arr.shape[0] != cov_gg_gg_arr.shape[1]:
        raise ValueError("cov_gg_gg must be square.")

    expected_cross_shape = (
        cov_gm_gm_arr.shape[0],
        cov_gg_gg_arr.shape[0],
    )

    if cov_gm_gg_arr.shape != expected_cross_shape:
        raise ValueError(
            "cov_gm_gg has incompatible shape. "
            f"Expected {expected_cross_shape}, got {cov_gm_gg_arr.shape}."
        )

    return cov_gm_gm_arr, cov_gg_gg_arr, cov_gm_gg_arr


def normalize_axis(axis: int, ndim: int) -> int:
    """Return a non-negative axis index for an array with ``ndim`` dimensions.

    Args:
        axis: Axis index, allowing negative indexing.
        ndim: Number of dimensions of the target array.

    Returns:
        Equivalent non-negative axis index.

    Raises:
        ValueError: If ``ndim`` is not positive or if ``axis`` is out of bounds.
    """
    if ndim <= 0:
        raise ValueError("ndim must be positive.")
    if not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} is out of bounds for an array with {ndim} dimensions.")

    return axis % ndim


def validate_integration_axis(
    values: Any,
    x: Any,
    *,
    axis: int = -1,
) -> None:
    """Validate that ``x`` matches an integration axis of ``values``.

    Args:
        values: Array of values to integrate.
        x: Coordinate array for the integration axis.
        axis: Integration axis.

    Raises:
        ValueError: If ``x`` is invalid or does not match the chosen axis.
    """
    values_arr = np.asarray(values, dtype=float)
    x_arr = validate_strictly_increasing(x, "x", min_size=2)

    if values_arr.ndim == 0:
        raise ValueError("values must have at least one dimension.")

    axis = normalize_axis(axis, values_arr.ndim)

    if values_arr.shape[axis] != x_arr.size:
        raise ValueError(
            "x length must match the integration axis of values. "
            f"Got x.size={x_arr.size} and values.shape[{axis}]="
            f"{values_arr.shape[axis]}."
        )


def validate_power_spectrum_inputs(
    k: Any,
    pk: Any,
    *,
    k_name: str = "k",
    pk_name: str = "pk",
) -> None:
    """Validate a tabulated one-dimensional power spectrum.

    Args:
        k: Wavenumber grid.
        pk: Power-spectrum values evaluated on ``k``.
        k_name: Name of the wavenumber array for error messages.
        pk_name: Name of the power-spectrum array for error messages.

    Raises:
        ValueError: If arrays are not finite, one-dimensional, matching, positive
            in ``k``, or strictly increasing in ``k``.
    """
    k_arr = validate_positive_strictly_increasing_1d_array(
        k,
        k_name,
        min_size=2,
    )
    pk_arr = as_1d_float_array(pk, pk_name, min_size=2)

    if pk_arr.shape != k_arr.shape:
        raise ValueError(
            f"{k_name} and {pk_name} must have matching shapes. "
            f"Got {k_arr.shape} and {pk_arr.shape}."
        )


def validate_redshift_edges(
    z_min: float,
    z_max: float,
    *,
    z_min_name: str = "z_min",
    z_max_name: str = "z_max",
) -> None:
    """Validate lower and upper redshift edges.

    Args:
        z_min: Lower redshift edge.
        z_max: Upper redshift edge.
        z_min_name: Name of the lower edge for error messages.
        z_max_name: Name of the upper edge for error messages.

    Raises:
        ValueError: If the redshift edges are invalid.
    """
    validate_nonnegative_scalar(z_min, z_min_name)
    validate_finite_scalar(z_max, z_max_name)

    if z_max <= z_min:
        raise ValueError(f"{z_max_name} must be greater than {z_min_name}.")


def validate_redshift_distribution(
    z: Any,
    nz: Any,
    *,
    name: str = "nz",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and return a redshift distribution.

    Args:
        z: Redshift grid.
        nz: Redshift distribution evaluated on ``z``.
        name: Name used in error messages.

    Returns:
        Redshift grid and redshift distribution as float arrays.

    Raises:
        ValueError: If the inputs are invalid.
    """
    z_arr = validate_strictly_increasing(z, "z", min_size=2)
    nz_arr = validate_nonnegative_1d_array(nz, name, min_size=2)

    if z_arr.shape != nz_arr.shape:
        raise ValueError(
            f"z and {name} must have matching shapes. Got {z_arr.shape} and {nz_arr.shape}."
        )
    if np.any(z_arr < 0.0):
        raise ValueError("z must be non-negative.")

    return z_arr, nz_arr


def validate_integration_params(params: dict[str, Any]) -> None:
    """Validate lens-magnification integration settings.

    Args:
        params: Integration settings dictionary.

    Raises:
        ValueError: If any setting is invalid.
    """
    if int(params["n_ell"]) <= 0:
        raise ValueError("n_ell must be positive.")
    if float(params["ell_min"]) <= 0.0:
        raise ValueError("ell_min must be positive.")
    if float(params["ell_max"]) <= float(params["ell_min"]):
        raise ValueError("ell_max must be greater than ell_min.")
    if float(params["z_stepsize"]) <= 0.0:
        raise ValueError("z_stepsize must be positive.")
    if float(params["z_min"]) < 0.0:
        raise ValueError("z_min must be non-negative.")
    if float(params["delta_z_source"]) <= 0.0:
        raise ValueError("delta_z_source must be positive.")


def is_non_negative_integer(value: float | int) -> bool:
    """Return whether a value is a non-negative integer."""
    value_float = float(value)
    return value_float >= 0.0 and value_float.is_integer()


def occupied_redshift_range_from_nz(
    z: Any,
    nz: Any,
    *,
    threshold: float = 0.0,
    name: str = "nz",
) -> tuple[float, float]:
    """Return the redshift range occupied by a redshift distribution.

    The occupied range is defined by the first and last redshift values where
    the redshift distribution is above ``threshold``. This is useful when an
    approximate tomographic-bin extent is needed but explicit bin edges are not
    available.

    Args:
        z: Redshift grid.
        nz: Redshift distribution evaluated on ``z``.
        threshold: Minimum ``n(z)`` value used to define the occupied range.
        name: Name of the redshift distribution used in error messages.

    Returns:
        Lower and upper occupied redshift values.

    Raises:
        ValueError: If the redshift distribution is invalid, if ``threshold`` is
            negative, or if fewer than two redshift cells are occupied.
    """
    validate_nonnegative_scalar(threshold, "threshold")

    z_arr, nz_arr = validate_redshift_distribution(
        z,
        nz,
        name=name,
    )

    occupied = nz_arr > threshold

    if np.count_nonzero(occupied) < 2:
        raise ValueError(f"{name} must be non-zero in at least two redshift cells.")

    z_used = z_arr[occupied]

    return float(z_used[0]), float(z_used[-1])


def validate_forecast_vector_and_covariance(
    data_vector: Any,
    covariance: Any,
    *,
    data_vector_name: str = "data_vector",
    covariance_name: str = "covariance",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate and return a forecast data vector and matching covariance.

    Args:
        data_vector: Forecast data vector.
        covariance: Forecast covariance matrix.
        data_vector_name: Name of the data vector used in error messages.
        covariance_name: Name of the covariance matrix used in error messages.

    Returns:
        Validated one-dimensional data vector and two-dimensional covariance
        matrix.

    Raises:
        ValueError: If the data vector or covariance is invalid, or if the
            covariance shape does not match the data-vector length.
    """
    data_vector_arr = as_1d_float_array(data_vector, data_vector_name, min_size=1)
    covariance_arr = as_2d_float_array(covariance, covariance_name)

    expected_shape = (data_vector_arr.size, data_vector_arr.size)

    if covariance_arr.shape != expected_shape:
        raise ValueError(
            f"{covariance_name} shape does not match {data_vector_name}. "
            f"Expected {expected_shape}, got {covariance_arr.shape}."
        )

    return data_vector_arr, covariance_arr


def validate_parameter_names(
    parameter_names: Any,
    theta0: Any,
    *,
    parameter_names_name: str = "parameter_names",
    theta0_name: str = "theta0",
) -> tuple[str, ...]:
    """Validate parameter names against a fiducial parameter vector.

    Args:
        parameter_names: Names associated with the forecast parameters.
        theta0: Fiducial parameter vector.
        parameter_names_name: Name of the parameter-name sequence used in error
            messages.
        theta0_name: Name of the fiducial parameter vector used in error
            messages.

    Returns:
        Parameter names as a tuple of strings.

    Raises:
        ValueError: If the number of names does not match the number of
            fiducial parameters.
        TypeError: If any parameter name is not a string.
    """
    theta0_arr = as_1d_float_array(theta0, theta0_name, min_size=1)
    names = tuple(parameter_names or ())

    if any(not isinstance(name, str) for name in names):
        raise TypeError(f"{parameter_names_name} must contain only strings.")

    if names and len(names) != theta0_arr.size:
        raise ValueError(
            f"{parameter_names_name} must have the same length as {theta0_name}. "
            f"Got {len(names)} names for {theta0_arr.size} parameters."
        )

    return names


def redshift_window_mask(
    z: NDArray[np.float64],
    *,
    z_min: float | None = None,
    z_max: float | None = None,
) -> NDArray[np.bool_]:
    """Return a mask selecting redshifts inside optional window limits."""
    z_arr = validate_nonnegative_1d_array(z, "z", min_size=2)

    if z_min is not None:
        validate_nonnegative_scalar(z_min, "z_min")

    if z_max is not None:
        validate_nonnegative_scalar(z_max, "z_max")

    if z_min is not None and z_max is not None:
        validate_redshift_edges(z_min, z_max)

    mask = np.ones(z_arr.shape, dtype=bool)

    if z_min is not None:
        mask &= z_arr >= z_min

    if z_max is not None:
        mask &= z_arr <= z_max

    return mask


def validate_redshift_distribution_support(
    z,
    nz,
    *,
    name="redshift_distribution",
    trim_edge_points=0,
):
    """Validate and select the positive support of a redshift distribution.

    Leading and trailing zero-density points are removed. Interior non-positive
    values inside the selected support are rejected.
    """
    z_arr, nz_arr = validate_redshift_distribution(z, nz, name=name)

    if not is_non_negative_integer(trim_edge_points):
        raise ValueError("trim_edge_points must be a non-negative integer.")

    trim_edge_points = int(trim_edge_points)

    positive_support = np.where(nz_arr > 0.0)[0]

    if positive_support.size == 0:
        raise ValueError(f"{name} normalization must be finite and positive.")

    first_support = int(positive_support[0])
    last_support = int(positive_support[-1]) + 1

    z_use = z_arr[first_support:last_support]
    nz_use = nz_arr[first_support:last_support]

    if z_use.size < 2:
        raise ValueError(
            f"{name} support must contain at least two redshift values "
            "after support/window filtering."
        )

    if np.any(nz_use <= 0.0):
        raise ValueError(f"{name} normalization must be finite and positive.")

    if trim_edge_points > 0:
        if z_use.size <= 2 * trim_edge_points:
            raise ValueError(
                f"trim_edge_points removes all {name} support. "
                f"support size = {z_use.size}, "
                f"trim_edge_points = {trim_edge_points}."
            )

        z_use = z_use[trim_edge_points:-trim_edge_points]
        nz_use = nz_use[trim_edge_points:-trim_edge_points]

    if z_use.size < 2:
        raise ValueError(
            f"{name} support must contain at least two redshift values "
            "after support/window filtering."
        )

    norm = np.trapezoid(nz_use, z_use)

    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} normalization must be finite and positive.")

    return z_use, nz_use, norm


def validate_hankel_1d_grid_spacing(
    k: FloatArray,
    name: str = "k",
) -> FloatArray:
    """Validate that the Hankel transform input grid has logarithmic spacing.

    Args:
        k: Wavenumber grid for the Hankel transform.
        name: Name of the wavenumber grid used in error messages.
    """
    
    k_arr = as_1d_float_array(k, "k", min_size=3)
    lnk = np.log(k_arr)
    dlnk = np.diff(lnk)
    if not np.allclose(dlnk, dlnk[0]):
        raise ValueError(f"{name} must have uniform logarithmic spacing for Hankel transforms.")
        
    return k_arr


def validate_interpolation_within_bounds(
    x_eval: Any,
    x_data: Any,
    name: str,
) -> FloatArray:
    """Validate that interpolation points are within the bounds of the data grid.

    Args:
        x_eval: Points at which to evaluate the interpolation.
        x_data: Data points for the interpolation.
        name: Name of the interpolation variable used in error messages.

    Returns:
        Validated interpolation points as a float array.

    Raises:
        ValueError: If the inputs are invalid.
    """
    x_eval_arr = as_1d_float_array(x_eval, f"{name}_eval", min_size=1)
    x_data_arr = as_1d_float_array(x_data, f"{name}_data", min_size=2)

    if not np.all((x_eval_arr >= x_data_arr[0]) & (x_eval_arr <= x_data_arr[-1])):
        raise ValueError(f"Requested interpolation values for {name} lie outside the data grid.")

    return x_eval_arr