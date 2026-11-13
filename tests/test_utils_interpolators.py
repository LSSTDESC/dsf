"""Tests for ``dsf.utils.interpolators.py``."""

from __future__ import annotations

import numpy as np
import pytest

from dsf.utils.interpolators import interpolate_linear, interpolate_loglog

XP_LINEAR = np.array([0.0, 1.0, 2.0])  # monotonic, positive
FP_LINEAR = np.array([0.0, 10.0, 20.0])  # simple linear relationship

XP_LOGLOG = np.array([1.0, 10.0, 100.0])  # strictly increasing, > 0
FP_LOGLOG = np.array([1.0, 10.0, 100.0])  # same values → identity on log‑log scale


def test_interpolate_linear_vectorised():
    """A 1D array of points must be interpolated element-wise."""
    xs = np.array([0.0, 0.25, 1.5, 2.0])
    expected = 10.0 * xs
    result = interpolate_linear(xs, XP_LINEAR, FP_LINEAR)
    assert np.allclose(result, expected)


def test_interpolate_linear_out_of_bounds():
    """Values outside the xp range should trigger a validation error."""
    xs = np.array([-0.1, 2.1])
    with pytest.raises(ValueError):
        interpolate_linear(xs, XP_LINEAR, FP_LINEAR)


def test_interpolate_linear_non_monotonic_xp():
    """xp that is not strictly increasing must raise an error."""
    bad_xp = np.array([0.0, 0.0, 1.0])
    xs = np.array([0.5])
    with pytest.raises(ValueError):
        interpolate_linear(xs, bad_xp, FP_LINEAR)


def test_interpolate_loglog_vectorised():
    """Vectorised version must return a NumPy array of correct values."""
    xs = np.array([1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    expected = xs
    result = interpolate_loglog(xs, XP_LOGLOG, FP_LOGLOG)
    assert np.allclose(result, expected)


def test_interpolate_loglog_negative_input():
    """Test that log of a non-positive number is undefined."""
    xs = np.array([-1.0, 0.0, 0.5])
    with pytest.raises(ValueError):
        interpolate_loglog(xs, XP_LOGLOG, FP_LOGLOG)


def test_interpolate_loglog_non_monotonic_xp():
    """Test that xp must be strictly increasing and positive for log-log."""
    bad_xp = np.array([1.0, 1.0, 10.0])
    xs = np.array([5.0])
    with pytest.raises(ValueError):
        interpolate_loglog(xs, bad_xp, FP_LOGLOG)
