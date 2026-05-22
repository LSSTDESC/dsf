"""Tests for ``src.dsf.utils.integrators.py``."""

import numpy as np
import pytest

from src.dsf.utils.integrators import (
    normalize_distribution,
    trapezoid_integral,
    weighted_trapezoid_average,
)


def test_trapezoid_integral_returns_scalar_for_1d_input():
    """Tests that one-dimensional integration returns a scalar."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])

    assert trapezoid_integral(y, x) == pytest.approx(2.0)


def test_trapezoid_integral_integrates_along_requested_axis():
    """Tests that batched integration removes the requested axis."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])

    actual = trapezoid_integral(y, x, axis=1)

    np.testing.assert_allclose(actual, np.array([2.0, 4.0]))


def test_trapezoid_integral_rejects_mismatched_axis_length():
    """Tests that mismatched integration coordinates are rejected."""
    x = np.array([0.0, 1.0])
    y = np.array([[0.0, 1.0, 2.0]])

    with pytest.raises(ValueError):
        trapezoid_integral(y, x, axis=1)


def test_normalize_distribution_returns_unit_integral():
    """Tests that distributions are normalized to unit integral."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 1.0, 1.0])

    actual = normalize_distribution(x, y)

    assert np.trapezoid(actual, x) == pytest.approx(1.0)


def test_normalize_distribution_preserves_relative_shape():
    """Tests that normalization preserves relative distribution values."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([1.0, 2.0, 1.0])

    actual = normalize_distribution(x, y)

    np.testing.assert_allclose(actual / actual[1], y / y[1])


def test_normalize_distribution_rejects_non_positive_integral():
    """Tests that distributions with non-positive integrals are rejected."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="Distribution integral must be positive"):
        normalize_distribution(x, y)


def test_weighted_trapezoid_average_returns_scalar_for_1d_values():
    """Tests that weighted averaging returns a scalar for one-dimensional values."""
    x = np.array([0.0, 1.0, 2.0])
    values = np.array([2.0, 4.0, 6.0])
    weights = np.array([1.0, 1.0, 1.0])

    actual = weighted_trapezoid_average(x, values, weights)

    assert actual == pytest.approx(4.0)


def test_weighted_trapezoid_average_integrates_along_requested_axis():
    """Tests that weighted averaging works along the requested axis."""
    x = np.array([0.0, 1.0, 2.0])
    values = np.array([[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]])
    weights = np.array([1.0, 1.0, 1.0])

    actual = weighted_trapezoid_average(x, values, weights, axis=0)

    np.testing.assert_allclose(actual, np.array([4.0, 8.0]))


def test_weighted_trapezoid_average_can_return_weighted_integral():
    """Tests that disabling weight normalization returns the weighted integral."""
    x = np.array([0.0, 1.0, 2.0])
    values = np.array([2.0, 4.0, 6.0])
    weights = np.array([1.0, 1.0, 1.0])

    actual = weighted_trapezoid_average(
        x,
        values,
        weights,
        normalize_weights=False,
    )

    assert actual == pytest.approx(8.0)


def test_weighted_trapezoid_average_rejects_mismatched_weights():
    """Tests that weights must match the integration grid."""
    x = np.array([0.0, 1.0, 2.0])
    values = np.array([2.0, 4.0, 6.0])
    weights = np.array([1.0, 1.0])

    with pytest.raises(ValueError):
        weighted_trapezoid_average(x, values, weights)
