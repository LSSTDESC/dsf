"""Unit tests for ``src.dsf.covariance.projection.hankel_utils``."""

import numpy as np
import pytest
from scipy.special import jv

from src.dsf.covariance.projection.hankel_utils import (
    apply_taper_spectrum,
    bessel_zeros,
    compute_bin_radial_matrix,
    compute_correlation_matrix,
    compute_diagonal_error,
    radial_bin_centers,
    radial_weights,
)


def test_bessel_zeros_integer_order_matches_known_j0_roots():
    """Tests that integer-order Bessel roots match known J0 zeros."""
    roots = bessel_zeros(order=0, n_zeros=3)

    expected = np.array(
        [
            2.404825557695773,
            5.520078110286311,
            8.653727912911013,
        ]
    )

    np.testing.assert_allclose(roots, expected, rtol=1.0e-12, atol=1.0e-12)


def test_bessel_zeros_non_integer_order_are_actual_roots():
    """Tests that non-integer-order roots evaluate to zero."""
    roots = bessel_zeros(order=0.5, n_zeros=4)

    assert roots.shape == (4,)
    assert np.all(roots > 0.0)
    assert np.all(np.diff(roots) > 0.0)

    values = jv(0.5, roots)
    np.testing.assert_allclose(values, np.zeros_like(values), atol=1.0e-10)


def test_bessel_zeros_rejects_non_positive_number_of_roots():
    """Tests that requesting zero or fewer Bessel roots fails."""
    with pytest.raises(ValueError, match="n_zeros must be positive"):
        bessel_zeros(order=0, n_zeros=0)


def test_apply_taper_spectrum_preserves_input_array():
    """Tests that tapering returns a copy without mutating the input."""
    k = np.array([1.0e-6, 1.0e-4, 1.0, 20.0, 200.0])
    pk = np.ones_like(k)

    tapered = apply_taper_spectrum(
        k,
        pk,
        large_k_lower=10.0,
        large_k_upper=100.0,
        low_k_lower=0.0,
        low_k_upper=1.0e-5,
    )

    np.testing.assert_allclose(pk, np.ones_like(pk))
    assert tapered is not pk


def test_apply_taper_spectrum_sets_power_to_zero_outside_limits():
    """Tests that the taper zeros power outside the trusted limits."""
    k = np.array([-1.0e-6, 1.0e-6, 1.0e-4, 10.0, 50.0, 150.0])
    pk = np.ones_like(k)

    tapered = apply_taper_spectrum(
        k,
        pk,
        large_k_lower=10.0,
        large_k_upper=100.0,
        low_k_lower=0.0,
        low_k_upper=1.0e-5,
    )

    assert tapered[0] == 0.0
    assert tapered[-1] == 0.0
    assert tapered[2] == 1.0
    assert tapered[3] == 1.0
    assert 0.0 < tapered[1] < 1.0
    assert 0.0 < tapered[4] < 1.0


def test_apply_taper_spectrum_uses_expected_cosine_taper_values():
    """Tests that low-k and high-k taper factors follow the cosine form."""
    k = np.array([5.0e-6, 55.0])
    pk = np.array([2.0, 4.0])

    tapered = apply_taper_spectrum(
        k,
        pk,
        large_k_lower=10.0,
        large_k_upper=100.0,
        low_k_lower=0.0,
        low_k_upper=1.0e-5,
    )

    expected_low = 2.0 * np.cos((5.0e-6 - 1.0e-5) / (1.0e-5 - 0.0) * np.pi / 2.0)
    expected_high = 4.0 * np.cos((55.0 - 10.0) / (100.0 - 10.0) * np.pi / 2.0)

    np.testing.assert_allclose(tapered, [expected_low, expected_high])


def test_compute_correlation_matrix_normalizes_covariance():
    """Tests that covariance entries are normalized by standard deviations."""
    covariance = np.array(
        [
            [4.0, 2.0, -1.0],
            [2.0, 9.0, 3.0],
            [-1.0, 3.0, 16.0],
        ]
    )

    correlation = compute_correlation_matrix(covariance)

    expected = np.array(
        [
            [1.0, 1.0 / 3.0, -1.0 / 8.0],
            [1.0 / 3.0, 1.0, 0.25],
            [-1.0 / 8.0, 0.25, 1.0],
        ]
    )

    np.testing.assert_allclose(correlation, expected)


def test_compute_correlation_matrix_handles_zero_variance_entries():
    """Tests that zero-variance rows and columns produce finite correlations."""
    covariance = np.array(
        [
            [4.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 9.0],
        ]
    )

    correlation = compute_correlation_matrix(covariance)

    expected = np.array(
        [
            [1.0, 0.0, 1.0 / 3.0],
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(correlation, expected)


def test_compute_diagonal_error_returns_sqrt_of_diagonal():
    """Tests that diagonal errors are square roots of covariance diagonals."""
    covariance = np.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 9.0, 2.0],
            [0.0, 2.0, 25.0],
        ]
    )

    errors = compute_diagonal_error(covariance)

    np.testing.assert_allclose(errors, [2.0, 3.0, 5.0])


def test_radial_bin_centers_are_geometric_centers():
    """Tests that radial bin centers are geometric means of bin edges."""
    r_bins = np.array([1.0, 4.0, 16.0, 64.0])

    centers = radial_bin_centers(r_bins)

    np.testing.assert_allclose(centers, [2.0, 8.0, 32.0])


def test_radial_weights_without_bins_use_grid_gradient():
    """Tests that unbinned radial weights use r times the grid gradient."""
    r = np.array([1.0, 2.0, 4.0, 8.0])

    weights = radial_weights(r)

    expected = r * np.gradient(r)

    np.testing.assert_allclose(weights, expected)


def test_radial_weights_with_bins_use_union_grid_spacing():
    """Tests that binned radial weights use spacing from the union grid."""
    r = np.array([1.0, 2.0, 4.0])
    r_bins = np.array([1.0, 3.0, 5.0])

    weights = radial_weights(r, r_bins=r_bins)

    r_union = np.union1d(r, r_bins)
    dr_union = np.gradient(r_union)
    positions = np.searchsorted(r_union, r)
    expected = r * dr_union[positions]

    np.testing.assert_allclose(weights, expected)


def test_compute_bin_radial_matrix_returns_geometric_bin_centers():
    """Tests that radial binning returns geometric bin centers."""
    r = np.array([1.0, 2.0, 4.0, 8.0])
    r_bins = np.array([1.0, 3.0, 9.0])
    matrix = np.ones((4, 4))

    centers, _ = compute_bin_radial_matrix(r, matrix, r_bins)

    np.testing.assert_allclose(centers, np.sqrt([3.0, 27.0]))


def test_compute_bin_radial_matrix_constant_matrix_stays_constant_in_populated_bins():
    """Tests that bin-averaging preserves a constant matrix."""
    r = np.array([1.0, 2.0, 4.0, 8.0])
    r_bins = np.array([1.0, 3.0, 9.0])
    matrix = np.full((4, 4), 7.0)

    _, binned = compute_bin_radial_matrix(r, matrix, r_bins)

    np.testing.assert_allclose(binned, np.full((2, 2), 7.0))


def test_compute_bin_radial_matrix_matches_manual_weighted_average():
    """Test that radial matrix binning against a manual weighted average."""
    r = np.array([1.0, 2.0, 4.0, 8.0])
    r_bins = np.array([1.0, 3.0, 9.0])
    matrix = np.add.outer(r, 10.0 * r)

    _, binned = compute_bin_radial_matrix(r, matrix, r_bins)

    weights = radial_weights(r, r_bins=r_bins)
    bin_index = np.digitize(r, r_bins) - 1

    expected = np.zeros((2, 2))
    for i in range(2):
        rows = bin_index == i
        row_weights = weights[rows]

        for j in range(2):
            cols = bin_index == j
            col_weights = weights[cols]

            submatrix = matrix[np.ix_(rows, cols)]
            weight_matrix = np.multiply.outer(row_weights, col_weights)
            expected[i, j] = np.sum(submatrix * weight_matrix) / np.sum(weight_matrix)

    np.testing.assert_allclose(binned, expected)


def test_compute_bin_radial_matrix_ignores_points_outside_bins():
    """Tests that radial points outside the bin edges do not affect averages."""
    r = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 10.0])
    r_bins = np.array([1.0, 3.0, 9.0])
    matrix = np.ones((6, 6))

    _, binned = compute_bin_radial_matrix(r, matrix, r_bins)

    np.testing.assert_allclose(binned, np.ones((2, 2)))


def test_compute_bin_radial_matrix_empty_bins_remain_zero():
    """Tests that bins with no radial support remain zero."""
    r = np.array([1.0, 2.0, 8.0])
    r_bins = np.array([1.0, 3.0, 5.0, 9.0])
    matrix = np.ones((3, 3))

    _, binned = compute_bin_radial_matrix(r, matrix, r_bins)

    assert binned.shape == (3, 3)
    np.testing.assert_allclose(binned[1, :], np.zeros(3))
    np.testing.assert_allclose(binned[:, 1], np.zeros(3))
    assert binned[0, 0] == 1.0
    assert binned[0, 2] == 1.0
    assert binned[2, 0] == 1.0
    assert binned[2, 2] == 1.0


def test_compute_bin_radial_matrix_supports_three_dimensional_tensors():
    """Tests that radial binning supports higher-dimensional tensors."""
    r = np.array([1.0, 2.0, 4.0, 8.0])
    r_bins = np.array([1.0, 3.0, 9.0])
    tensor = np.ones((4, 4, 4))

    centers, binned = compute_bin_radial_matrix(r, tensor, r_bins)

    np.testing.assert_allclose(centers, np.sqrt([3.0, 27.0]))
    assert binned.shape == (2, 2, 2)
    np.testing.assert_allclose(binned, np.ones((2, 2, 2)))
