"""Unit tests for ``src.dsf.covariance.ingredients.noise``."""

import numpy as np
import pytest

from src.dsf.covariance.ingredients.noise import (
    angular_shape_noise,
    projected_shape_noise,
    shot_noise,
)
from src.dsf.utils.converters import arcmin2_per_steradian


def test_shot_noise_returns_inverse_number_density():
    """Test that shot noise is the inverse lens number density."""
    n_gal = 2.5e-4

    result = shot_noise(n_gal)

    np.testing.assert_allclose(result, 1.0 / n_gal)


@pytest.mark.parametrize("n_gal", [0.0, -1.0, -1.0e-4])
def test_shot_noise_requires_positive_number_density(n_gal):
    """Test that shot noise rejects non-positive lens number densities."""
    with pytest.raises(ValueError, match="n_gal must be positive"):
        shot_noise(n_gal)


def test_angular_shape_noise_uses_steradian_density():
    """Test that angular shape noise converts source density to steradians."""
    sigma_e = 0.26
    n_eff_arcmin2 = 10.0

    n_eff_sr = n_eff_arcmin2 * arcmin2_per_steradian()
    expected = sigma_e**2 / n_eff_sr

    result = angular_shape_noise(sigma_e, n_eff_arcmin2)

    np.testing.assert_allclose(result, expected)


def test_angular_shape_noise_decreases_for_denser_source_sample():
    """Test that angular shape noise decreases for denser source samples."""
    sigma_e = 0.26

    sparse = angular_shape_noise(sigma_e, n_eff_arcmin2=5.0)
    dense = angular_shape_noise(sigma_e, n_eff_arcmin2=20.0)

    assert dense < sparse


def test_angular_shape_noise_scales_as_sigma_e_squared():
    """Test that angular shape noise scales quadratically with sigma_e."""
    n_eff_arcmin2 = 10.0

    noise_low = angular_shape_noise(sigma_e=0.2, n_eff_arcmin2=n_eff_arcmin2)
    noise_high = angular_shape_noise(sigma_e=0.4, n_eff_arcmin2=n_eff_arcmin2)

    np.testing.assert_allclose(noise_high / noise_low, 4.0)


@pytest.mark.parametrize(
    ("sigma_e", "n_eff_arcmin2", "message"),
    [
        (0.0, 10.0, "sigma_e must be positive"),
        (-0.1, 10.0, "sigma_e must be positive"),
        (0.26, 0.0, "n_eff_arcmin2 must be positive"),
        (0.26, -5.0, "n_eff_arcmin2 must be positive"),
    ],
)
def test_angular_shape_noise_requires_positive_inputs(
    sigma_e,
    n_eff_arcmin2,
    message,
):
    """Test that angular shape noise rejects non-positive inputs."""
    with pytest.raises(ValueError, match=message):
        angular_shape_noise(sigma_e, n_eff_arcmin2)


def test_projected_shape_noise_uses_projected_source_density():
    """Test that projected shape noise uses transverse projected density."""
    sigma_e = 0.26
    n_eff_arcmin2 = 10.0
    chi_eff = 1500.0

    n_eff_sr = n_eff_arcmin2 * arcmin2_per_steradian()
    n_eff_projected = n_eff_sr / chi_eff**2
    expected = sigma_e**2 / n_eff_projected

    result = projected_shape_noise(sigma_e, n_eff_arcmin2, chi_eff)

    np.testing.assert_allclose(result, expected)


def test_projected_shape_noise_matches_angular_noise_times_chi_squared():
    """Test that projected shape noise equals angular noise times chi_eff squared."""
    sigma_e = 0.26
    n_eff_arcmin2 = 10.0
    chi_eff = 1500.0

    angular = angular_shape_noise(sigma_e, n_eff_arcmin2)
    projected = projected_shape_noise(sigma_e, n_eff_arcmin2, chi_eff)

    np.testing.assert_allclose(projected, angular * chi_eff**2)


def test_projected_shape_noise_scales_as_chi_squared():
    """Test that projected shape noise scales quadratically with chi_eff."""
    sigma_e = 0.26
    n_eff_arcmin2 = 10.0

    noise_low = projected_shape_noise(sigma_e, n_eff_arcmin2, chi_eff=1000.0)
    noise_high = projected_shape_noise(sigma_e, n_eff_arcmin2, chi_eff=2000.0)

    np.testing.assert_allclose(noise_high / noise_low, 4.0)


def test_projected_shape_noise_decreases_for_denser_source_sample():
    """Test that projected shape noise decreases for denser source samples."""
    sigma_e = 0.26
    chi_eff = 1500.0

    sparse = projected_shape_noise(sigma_e, n_eff_arcmin2=5.0, chi_eff=chi_eff)
    dense = projected_shape_noise(sigma_e, n_eff_arcmin2=20.0, chi_eff=chi_eff)

    assert dense < sparse


@pytest.mark.parametrize(
    ("sigma_e", "n_eff_arcmin2", "chi_eff", "message"),
    [
        (0.0, 10.0, 1500.0, "sigma_e must be positive"),
        (-0.1, 10.0, 1500.0, "sigma_e must be positive"),
        (0.26, 0.0, 1500.0, "n_eff_arcmin2 must be positive"),
        (0.26, -5.0, 1500.0, "n_eff_arcmin2 must be positive"),
        (0.26, 10.0, 0.0, "chi_eff must be positive"),
        (0.26, 10.0, -100.0, "chi_eff must be positive"),
    ],
)
def test_projected_shape_noise_requires_positive_inputs(
    sigma_e,
    n_eff_arcmin2,
    chi_eff,
    message,
):
    """Test that projected shape noise rejects non-positive inputs."""
    with pytest.raises(ValueError, match=message):
        projected_shape_noise(sigma_e, n_eff_arcmin2, chi_eff)
