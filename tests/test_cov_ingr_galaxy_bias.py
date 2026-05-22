"""Unit tests for ``src.dsf.covariance.ingredients.galaxy_bias``."""

import numpy as np
import pytest

from src.dsf.covariance.ingredients.galaxy_bias import linear_galaxy_bias


def test_linear_galaxy_bias_returns_prefactor_over_growth(monkeypatch):
    """Tests that linear galaxy bias is computed as prefactor over growth."""
    z_bin_centers = np.asarray([0.0, 0.5, 1.0])
    growth = np.asarray([1.0, 0.8, 0.5])

    def fake_growth_factor(cosmo, scale_factor):
        return growth

    monkeypatch.setattr(
        "src.dsf.covariance.ingredients.galaxy_bias.ccl.growth_factor",
        fake_growth_factor,
    )

    galaxy_bias = linear_galaxy_bias(
        cosmo=object(),
        z_bin_centers=z_bin_centers,
        bias_prefactor=2.0,
        round_decimals=None,
    )

    expected = np.asarray([2.0, 2.5, 4.0])

    np.testing.assert_allclose(galaxy_bias, expected)


def test_linear_galaxy_bias_rounds_output(monkeypatch):
    """Tests that linear galaxy bias rounds values when requested."""
    z_bin_centers = np.asarray([0.0, 0.5, 1.0])
    growth = np.asarray([1.0, 0.81, 0.49])

    def fake_growth_factor(cosmo, scale_factor):
        return growth

    monkeypatch.setattr(
        "src.dsf.covariance.ingredients.galaxy_bias.ccl.growth_factor",
        fake_growth_factor,
    )

    galaxy_bias = linear_galaxy_bias(
        cosmo=object(),
        z_bin_centers=z_bin_centers,
        bias_prefactor=1.0,
        round_decimals=2,
    )

    expected = np.round(1.0 / growth, 2)

    np.testing.assert_allclose(galaxy_bias, expected)


def test_linear_galaxy_bias_does_not_round_when_round_decimals_is_none(monkeypatch):
    """Tests that linear galaxy bias can return unrounded values."""
    z_bin_centers = np.asarray([0.5])
    growth = np.asarray([0.81])

    def fake_growth_factor(cosmo, scale_factor):
        return growth

    monkeypatch.setattr(
        "src.dsf.covariance.ingredients.galaxy_bias.ccl.growth_factor",
        fake_growth_factor,
    )

    galaxy_bias = linear_galaxy_bias(
        cosmo=object(),
        z_bin_centers=z_bin_centers,
        bias_prefactor=1.0,
        round_decimals=None,
    )

    np.testing.assert_allclose(galaxy_bias, [1.0 / 0.81])


@pytest.mark.parametrize(
    "bad_prefactor",
    [0.0, -1.0],
)
def test_linear_galaxy_bias_rejects_non_positive_prefactor(
    monkeypatch,
    bad_prefactor,
):
    """Tests that linear galaxy bias rejects non-positive bias prefactors."""

    def fake_growth_factor(cosmo, scale_factor):
        return np.asarray([1.0])

    monkeypatch.setattr(
        "src.dsf.covariance.ingredients.galaxy_bias.ccl.growth_factor",
        fake_growth_factor,
    )

    with pytest.raises(ValueError):
        linear_galaxy_bias(
            cosmo=object(),
            z_bin_centers=[0.5],
            bias_prefactor=bad_prefactor,
        )


@pytest.mark.parametrize(
    "bad_z_bin_centers",
    [
        [-0.1, 0.2],
        [[0.1, 0.2], [0.3, 0.4]],
    ],
)
def test_linear_galaxy_bias_rejects_invalid_redshift_inputs(
    monkeypatch,
    bad_z_bin_centers,
):
    """Tests that linear galaxy bias rejects invalid redshift inputs."""

    def fake_growth_factor(cosmo, scale_factor):
        return np.asarray([1.0])

    monkeypatch.setattr(
        "src.dsf.covariance.ingredients.galaxy_bias.ccl.growth_factor",
        fake_growth_factor,
    )

    with pytest.raises(ValueError):
        linear_galaxy_bias(
            cosmo=object(),
            z_bin_centers=bad_z_bin_centers,
        )


@pytest.mark.parametrize(
    "bad_growth",
    [
        np.asarray([0.0, 1.0]),
        np.asarray([-0.5, 1.0]),
    ],
)
def test_linear_galaxy_bias_rejects_non_positive_growth_factor(
    monkeypatch,
    bad_growth,
):
    """Tests that linear galaxy bias rejects non-positive growth factors."""

    def fake_growth_factor(cosmo, scale_factor):
        return bad_growth

    monkeypatch.setattr(
        "src.dsf.covariance.ingredients.galaxy_bias.ccl.growth_factor",
        fake_growth_factor,
    )

    with pytest.raises(ValueError, match="growth_factor"):
        linear_galaxy_bias(
            cosmo=object(),
            z_bin_centers=[0.0, 0.5],
        )


def test_linear_galaxy_bias_passes_expected_scale_factors_to_ccl(monkeypatch):
    """Tests that linear galaxy bias evaluates growth at expected scale factors."""
    recorded = {}

    def fake_growth_factor(cosmo, scale_factor):
        recorded["cosmo"] = cosmo
        recorded["scale_factor"] = np.asarray(scale_factor, dtype=float)

        return np.asarray([1.0, 0.8, 0.5])

    monkeypatch.setattr(
        "src.dsf.covariance.ingredients.galaxy_bias.ccl.growth_factor",
        fake_growth_factor,
    )

    cosmo = object()
    z_bin_centers = np.asarray([0.0, 0.5, 1.0])

    linear_galaxy_bias(
        cosmo=cosmo,
        z_bin_centers=z_bin_centers,
    )

    expected_scale_factor = 1.0 / (1.0 + z_bin_centers)

    assert recorded["cosmo"] is cosmo
    np.testing.assert_allclose(recorded["scale_factor"], expected_scale_factor)
