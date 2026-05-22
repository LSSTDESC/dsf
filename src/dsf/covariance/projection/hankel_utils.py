"""Numerical scripts for projected radial statistics.

This module provides small utilities used when converting Fourier-space
power spectra into radial-space projected statistics. The scripts cover
Bessel-function roots, smooth spectrum tapering, radial bin centers,
radial integration weights, covariance-to-correlation conversion, and
bin-averaging of radial matrices or tensors.

The functions assume their inputs have already been validated by the public
calling layer.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import jn_zeros, jv

from src.dsf.utils.types import FloatArray

__all__ = [
    "bessel_zeros",
    "apply_taper_spectrum",
    "compute_bin_radial_matrix",
    "compute_correlation_matrix",
    "compute_diagonal_error",
    "radial_bin_centers",
    "radial_weights",
]


def _is_non_negative_integer(value: float | int) -> bool:
    """Return whether a value is a valid integer Bessel order."""
    value_float = float(value)
    return value_float >= 0.0 and value_float.is_integer()


def bessel_zeros(order: float | int, n_zeros: int) -> FloatArray:
    """Return positive roots of the Bessel function for a given order.

    Args:
        order: Bessel-function order.
        n_zeros: Number of positive roots to return.

    Returns:
        Positive roots of :math:`J_\\nu(x)` for the requested order.
    """
    if n_zeros <= 0:
        raise ValueError("n_zeros must be positive.")

    if _is_non_negative_integer(order):
        return np.asarray(jn_zeros(int(order), n_zeros), dtype=float)

    order_float = float(order)
    roots = np.empty(n_zeros, dtype=float)

    def bessel_order_value(x: float) -> float:
        value = np.real_if_close(jv(order_float, x))
        return float(np.asarray(value).item())

    for i in range(n_zeros):
        zero_number = i + 1
        center = (zero_number + 0.5 * order_float - 0.25) * np.pi
        width = 0.55 * np.pi
        lower = max(1.0e-10, center - width)
        upper = center + width

        f_lower = bessel_order_value(lower)
        f_upper = bessel_order_value(upper)

        for _ in range(25):
            if f_lower * f_upper <= 0.0:
                break

            width *= 1.5
            lower = max(1.0e-10, center - width)
            upper = center + width
            f_lower = bessel_order_value(lower)
            f_upper = bessel_order_value(upper)
        else:
            raise RuntimeError(
                f"Could not bracket Bessel zero {zero_number} for order={order_float}."
            )

        roots[i] = brentq(bessel_order_value, lower, upper)

    return roots


def apply_taper_spectrum(
    k: FloatArray,
    pk: FloatArray,
    large_k_lower: float = 10.0,
    large_k_upper: float = 100.0,
    low_k_lower: float = 0.0,
    low_k_upper: float = 1.0e-5,
) -> FloatArray:
    """Return a smoothly tapered power spectrum.

    The taper suppresses power outside the trusted wavenumber range so that
    projected radial statistics are less sensitive to sharp spectrum cutoffs.

    Args:
        k: Wavenumber grid.
        pk: Power-spectrum values evaluated on ``k``.
        large_k_lower: Wavenumber where high-k suppression begins.
        large_k_upper: Wavenumber above which the spectrum is set to zero.
        low_k_lower: Wavenumber below which the spectrum is set to zero.
        low_k_upper: Wavenumber where low-k suppression ends.

    Returns:
        Power spectrum with smooth low-k and high-k suppression applied.
    """
    pk_out = np.copy(pk)

    high = k > large_k_lower
    pk_out[high] *= np.cos(
        (k[high] - large_k_lower) / (large_k_upper - large_k_lower) * np.pi / 2.0
    )
    pk_out[k > large_k_upper] = 0.0

    low = k < low_k_upper
    pk_out[low] *= np.cos((k[low] - low_k_upper) / (low_k_upper - low_k_lower) * np.pi / 2.0)
    pk_out[k < low_k_lower] = 0.0

    return pk_out


def compute_correlation_matrix(covariance: FloatArray) -> FloatArray:
    """Return the correlation matrix associated with a covariance matrix.

    Args:
        covariance: Covariance matrix.

    Returns:
        Dimensionless correlation matrix with entries normalized by the
        corresponding covariance standard deviations.
    """
    diag = np.diagonal(covariance)
    denom = np.sqrt(np.outer(diag, diag))

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / denom

    return np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)


def compute_diagonal_error(covariance: FloatArray) -> FloatArray:
    """Return one-sigma errors from a covariance matrix.

    Args:
        covariance: Covariance matrix.

    Returns:
        Square root of the covariance diagonal.
    """
    return np.sqrt(np.diagonal(covariance))


def radial_bin_centers(r_bins: FloatArray) -> FloatArray:
    """Return geometric centers of radial bins.

    Args:
        r_bins: Radial bin edges.

    Returns:
        Geometric mean of each pair of neighboring radial bin edges.
    """
    return np.sqrt(r_bins[1:] * r_bins[:-1])


def radial_weights(
    r: FloatArray,
    r_bins: FloatArray | None = None,
) -> FloatArray:
    """Return radial averaging weights.

    The weights represent the radial measure used when averaging projected
    radial quantities over annular bins.

    Args:
        r: Radial grid.
        r_bins: Optional radial bin edges.

    Returns:
        Radial weights proportional to :math:`r\\,dr`.
    """
    if r_bins is None:
        return r * np.gradient(r)

    r_union = np.union1d(r, r_bins)
    dr_union = np.gradient(r_union)
    r_positions = np.searchsorted(r_union, r)
    dr = dr_union[r_positions]

    return r * dr


def _outer_product(values: FloatArray, ndim: int) -> FloatArray:
    """Return radial weights for a matrix or higher-order tensor.

    Args:
        values: One-dimensional radial weights.
        ndim: Number of radial axes.

    Returns:
        Repeated outer product of ``values`` with one factor per radial axis.
    """
    result = values

    for _ in range(ndim - 1):
        result = np.multiply.outer(result, values)

    return result


def compute_bin_radial_matrix(
    r: FloatArray,
    matrix: FloatArray,
    r_bins: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Average a radial matrix or tensor into radial bins.

    Each axis of ``matrix`` is interpreted as a radial coordinate sampled on
    the same grid ``r``. The returned quantity is the annular-bin average of
    the input matrix or tensor.

    Args:
        r: Radial grid associated with each axis of ``matrix``.
        matrix: Radial matrix or tensor to bin.
        r_bins: Radial bin edges.

    Returns:
        Radial bin centers and the binned matrix or tensor.
    """
    ndim = matrix.ndim
    centers = radial_bin_centers(r_bins)
    n_bins = centers.size

    bin_index = np.digitize(r, r_bins) - 1
    valid = (bin_index >= 0) & (bin_index < n_bins)

    weights = radial_weights(r, r_bins=r_bins)

    bin_weight_sums = np.zeros(n_bins, dtype=float)
    np.add.at(bin_weight_sums, bin_index[valid], weights[valid])

    weighted_matrix = matrix * _outer_product(weights, ndim)
    binned_sum = np.zeros(tuple([n_bins] * ndim), dtype=float)

    bin_grids = np.meshgrid(*([bin_index] * ndim), indexing="ij")
    valid_grids = np.meshgrid(*([valid] * ndim), indexing="ij")
    valid_matrix = np.logical_and.reduce(valid_grids)

    output_indices = tuple(grid[valid_matrix] for grid in bin_grids)
    np.add.at(binned_sum, output_indices, weighted_matrix[valid_matrix])

    norm = _outer_product(bin_weight_sums, ndim)
    binned = np.zeros_like(binned_sum)

    nonzero = norm != 0.0
    binned[nonzero] = binned_sum[nonzero] / norm[nonzero]

    return centers, binned
