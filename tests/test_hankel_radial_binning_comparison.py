"""Compare radial binning methods against independent annular integrals."""

from itertools import combinations

import numpy as np
import pytest

from dsf.hankel.hankel_utils import compute_bin_radial_matrix

METHODS = ("gradient", "voronoi", "polynomial")
RTOL = 5.0e-3


def annular_power_average(edges, power):
    """Analytic integral of r**power against r dr, divided by bin area."""
    lower, upper = edges[:-1], edges[1:]
    if power == -2:
        integral = np.log(upper / lower)
    else:
        integral = (upper ** (power + 2) - lower ** (power + 2)) / (power + 2)
    return integral / (0.5 * (upper**2 - lower**2))


def assert_methods_agree(r, values, edges, expected):
    """Require accuracy against the integral and agreement for every pair."""
    results = {}
    for method in METHODS:
        centers, result = compute_bin_radial_matrix(r, values, edges, radial_weight_method=method)
        np.testing.assert_allclose(centers, np.sqrt(edges[:-1] * edges[1:]))
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=0.0, err_msg=method)
        results[method] = result

    for first, second in combinations(METHODS, 2):
        np.testing.assert_allclose(
            results[first],
            results[second],
            rtol=RTOL,
            atol=0.0,
            err_msg=f"{first} versus {second}",
        )


@pytest.mark.parametrize("spacing", ["linear", "log"])
@pytest.mark.parametrize("power", [-2, -1, 0, 1, 2])
def test_radial_binning_methods_agree_for_smooth_power_laws(spacing, power):
    """Exercise varying profiles and bin edges inserted between samples."""
    r = np.linspace(1.0, 10.0, 1001) if spacing == "linear" else np.geomspace(1.0, 10.0, 1001)
    edges = np.array([1.13, 1.9, 3.7, 7.4, 9.6])
    expected = annular_power_average(edges, power)
    assert_methods_agree(r, r**power, edges, expected)


@pytest.mark.parametrize("spacing", ["linear", "log"])
def test_radial_binning_methods_agree_for_covariance_kernel(spacing):
    """Compare nonconstant covariance diagonals and off-diagonal entries."""
    r = np.linspace(1.0, 10.0, 801) if spacing == "linear" else np.geomspace(1.0, 10.0, 801)
    edges = np.array([1.13, 1.9, 3.7, 7.4, 9.6])
    # Sum of positive outer products provides a symmetric PSD test kernel.
    covariance = np.outer(r**-1, r**-1) + 0.01 * np.outer(r, r)
    inverse_mean = annular_power_average(edges, -1)
    linear_mean = annular_power_average(edges, 1)
    expected = np.outer(inverse_mean, inverse_mean) + 0.01 * np.outer(linear_mean, linear_mean)
    assert_methods_agree(r, covariance, edges, expected)
