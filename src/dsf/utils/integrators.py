"""Numerical integration scripts for Delta Sigma calculations."""

from __future__ import annotations

import numpy as np

from .types import FloatArray, FloatLike
from .validators import validate_1d_pair, validate_integration_axis

__all__ = [
    "normalize_distribution",
    "trapezoid_integral",
    "weighted_trapezoid_average",
]


def trapezoid_integral(
    y: FloatArray,
    x: FloatArray,
    *,
    axis: int = -1,
) -> FloatLike:
    """Integrate values using the trapezoid rule.

    Args:
        y: Values to integrate.
        x: Coordinates along the integration axis.
        axis: Axis over which to integrate. If ``y`` is one-dimensional,
            the result is a scalar. If ``y`` has extra dimensions, the
            integration removes the selected axis and returns an array.

    Returns:
        Integral of ``y`` with respect to ``x``. Returns a scalar for a
        one-dimensional integral and an array for batched integrals.
    """
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)

    validate_integration_axis(y_arr, x_arr, axis=axis)

    result = np.trapezoid(y_arr, x_arr, axis=axis)

    if np.ndim(result) == 0:
        return float(result)

    return np.asarray(result, dtype=float)


def normalize_distribution(
    x: FloatArray,
    y: FloatArray,
) -> FloatArray:
    """Normalize a one-dimensional distribution.

    Args:
        x: Coordinate array.
        y: Distribution values evaluated on ``x``.

    Returns:
        Distribution values normalized so that their trapezoid integral over
        ``x`` is one.

    Raises:
        ValueError: If the distribution integral is not positive.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    validate_1d_pair(x_arr, y_arr, x_name="x", y_name="y")

    norm = np.trapezoid(y_arr, x_arr)

    if norm <= 0.0:
        raise ValueError("Distribution integral must be positive.")

    return np.asarray(y_arr / norm, dtype=float)


def weighted_trapezoid_average(
    x: FloatArray,
    values: FloatArray,
    weights: FloatArray,
    *,
    axis: int = 0,
    normalize_weights: bool = True,
) -> FloatLike:
    """Compute a weighted trapezoid average along one axis.

    Args:
        x: Coordinate array for the integration axis.
        values: Values to average.
        weights: One-dimensional weights defined on ``x``.
        axis: Axis of ``values`` corresponding to ``x``.
        normalize_weights: Whether to normalize ``weights`` before averaging.
            If ``True``, the returned value is a true weighted average. If
            ``False``, the returned value is the weighted integral.

    Returns:
        Weighted average over ``axis``. Returns a scalar if all dimensions are
        integrated away, otherwise returns an array.
    """
    x_arr = np.asarray(x, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)

    validate_integration_axis(values_arr, x_arr, axis=axis)
    validate_1d_pair(x_arr, weights_arr, x_name="x", y_name="weights")

    if normalize_weights:
        weights_arr = normalize_distribution(x_arr, weights_arr)

    weight_shape = [1] * values_arr.ndim
    weight_shape[axis] = weights_arr.size

    weighted_values = values_arr * weights_arr.reshape(weight_shape)

    return trapezoid_integral(weighted_values, x_arr, axis=axis)
