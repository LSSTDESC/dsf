"""Unit tests for ``dsf.covariance.ingredients.power_spectrum``."""

import numpy as np
import pytest

from dsf.covariance.ingredients.power_spectrum import lens_averaged_matter_power

POWER_SPECTRUM_MODULE = "dsf.covariance.ingredients.power_spectrum"


def test_lens_averaged_matter_power_uses_weighted_redshift_average(monkeypatch):
    """Test that the lens-averaged power spectrum is weighted over lens redshift."""
    cosmo = {"h": 0.7}
    k = np.array([0.1, 1.0, 10.0])
    z_lens = np.array([0.0, 1.0, 2.0])
    nz_lens = np.array([1.0, 2.0, 1.0])

    def fake_nonlin_matter_power(cosmo, k_mpc, scale_factor):
        z = 1.0 / scale_factor - 1.0
        return (1.0 + z) * k_mpc**2

    monkeypatch.setattr(
        f"{POWER_SPECTRUM_MODULE}.ccl.nonlin_matter_power",
        fake_nonlin_matter_power,
    )

    result = lens_averaged_matter_power(
        cosmo,
        k,
        z_lens,
        nz_lens,
        nonlinear=True,
    )

    h = cosmo["h"]
    pk_z0 = (1.0 + z_lens[0]) * (k * h) ** 2 * h**3
    pk_z1 = (1.0 + z_lens[1]) * (k * h) ** 2 * h**3
    pk_z2 = (1.0 + z_lens[2]) * (k * h) ** 2 * h**3

    pk_of_z = np.array([pk_z0, pk_z1, pk_z2])
    expected = np.trapezoid(pk_of_z * nz_lens[:, None], z_lens, axis=0)

    np.testing.assert_allclose(result, expected)


def test_lens_averaged_matter_power_uses_supplied_h(monkeypatch):
    """Test that a supplied h value overrides the value stored on the cosmology."""
    cosmo = {"h": 0.5}
    k = np.array([0.1, 1.0, 10.0])
    z_lens = np.array([0.0, 1.0])
    nz_lens = np.array([1.0, 1.0])
    supplied_h = 0.8

    called_k_values = []

    def fake_nonlin_matter_power(cosmo, k_mpc, scale_factor):
        called_k_values.append(np.array(k_mpc, copy=True))
        return np.ones_like(k_mpc)

    monkeypatch.setattr(
        f"{POWER_SPECTRUM_MODULE}.ccl.nonlin_matter_power",
        fake_nonlin_matter_power,
    )

    result = lens_averaged_matter_power(
        cosmo,
        k,
        z_lens,
        nz_lens,
        h=supplied_h,
        nonlinear=True,
    )

    for called_k in called_k_values:
        np.testing.assert_allclose(called_k, k * supplied_h)

    expected = np.trapezoid(
        np.ones((z_lens.size, k.size)) * supplied_h**3 * nz_lens[:, None],
        z_lens,
        axis=0,
    )

    np.testing.assert_allclose(result, expected)


def test_lens_averaged_matter_power_converts_power_to_mpc_over_h_units(monkeypatch):
    """Test that CCL power values are multiplied by h cubed."""
    cosmo = {"h": 0.7}
    k = np.array([0.1, 1.0, 10.0])
    z_lens = np.array([0.0, 1.0])
    nz_lens = np.array([1.0, 1.0])

    def fake_nonlin_matter_power(cosmo, k_mpc, scale_factor):
        return np.full_like(k_mpc, 2.0)

    monkeypatch.setattr(
        f"{POWER_SPECTRUM_MODULE}.ccl.nonlin_matter_power",
        fake_nonlin_matter_power,
    )

    result = lens_averaged_matter_power(
        cosmo,
        k,
        z_lens,
        nz_lens,
        nonlinear=True,
    )

    expected = np.full_like(k, 2.0 * cosmo["h"] ** 3)

    np.testing.assert_allclose(result, expected)


def test_lens_averaged_matter_power_uses_nonlinear_power_by_default(monkeypatch):
    """Test that nonlinear matter power is used by default."""
    cosmo = {"h": 0.7}
    k = np.array([0.1, 1.0])
    z_lens = np.array([0.0, 1.0])
    nz_lens = np.array([1.0, 1.0])

    calls = {"linear": 0, "nonlinear": 0}

    def fake_nonlin_matter_power(cosmo, k_mpc, scale_factor):
        calls["nonlinear"] += 1
        return np.ones_like(k_mpc)

    def fake_linear_matter_power(cosmo, k_mpc, scale_factor):
        calls["linear"] += 1
        return np.ones_like(k_mpc)

    monkeypatch.setattr(
        f"{POWER_SPECTRUM_MODULE}.ccl.nonlin_matter_power",
        fake_nonlin_matter_power,
    )
    monkeypatch.setattr(
        f"{POWER_SPECTRUM_MODULE}.ccl.linear_matter_power",
        fake_linear_matter_power,
    )

    lens_averaged_matter_power(cosmo, k, z_lens, nz_lens)

    assert calls["nonlinear"] == z_lens.size
    assert calls["linear"] == 0


def test_lens_averaged_matter_power_uses_linear_power_when_requested(monkeypatch):
    """Test that linear matter power is used when nonlinear is false."""
    cosmo = {"h": 0.7}
    k = np.array([0.1, 1.0])
    z_lens = np.array([0.0, 1.0])
    nz_lens = np.array([1.0, 1.0])

    calls = {"linear": 0, "nonlinear": 0}

    def fake_nonlin_matter_power(cosmo, k_mpc, scale_factor):
        calls["nonlinear"] += 1
        return np.ones_like(k_mpc)

    def fake_linear_matter_power(cosmo, k_mpc, scale_factor):
        calls["linear"] += 1
        return np.ones_like(k_mpc)

    monkeypatch.setattr(
        f"{POWER_SPECTRUM_MODULE}.ccl.nonlin_matter_power",
        fake_nonlin_matter_power,
    )
    monkeypatch.setattr(
        f"{POWER_SPECTRUM_MODULE}.ccl.linear_matter_power",
        fake_linear_matter_power,
    )

    lens_averaged_matter_power(
        cosmo,
        k,
        z_lens,
        nz_lens,
        nonlinear=False,
    )

    assert calls["linear"] == z_lens.size
    assert calls["nonlinear"] == 0


def test_lens_averaged_matter_power_returns_one_value_per_k_mode(monkeypatch):
    """Test that the averaged power spectrum has the same shape as the k grid."""
    cosmo = {"h": 0.7}
    k = np.array([0.1, 0.3, 1.0, 3.0, 10.0])
    z_lens = np.array([0.0, 0.5, 1.0])
    nz_lens = np.array([1.0, 2.0, 1.0])

    def fake_nonlin_matter_power(cosmo, k_mpc, scale_factor):
        return np.ones_like(k_mpc)

    monkeypatch.setattr(
        f"{POWER_SPECTRUM_MODULE}.ccl.nonlin_matter_power",
        fake_nonlin_matter_power,
    )

    result = lens_averaged_matter_power(cosmo, k, z_lens, nz_lens)

    assert result.shape == k.shape


def test_lens_averaged_matter_power_passes_correct_scale_factors(monkeypatch):
    """Test that lens redshifts are converted to CCL scale factors."""
    cosmo = {"h": 0.7}
    k = np.array([0.1, 1.0])
    z_lens = np.array([0.0, 1.0, 3.0])
    nz_lens = np.array([1.0, 1.0, 1.0])

    called_scale_factors = []

    def fake_nonlin_matter_power(cosmo, k_mpc, scale_factor):
        called_scale_factors.append(scale_factor)
        return np.ones_like(k_mpc)

    monkeypatch.setattr(
        f"{POWER_SPECTRUM_MODULE}.ccl.nonlin_matter_power",
        fake_nonlin_matter_power,
    )

    lens_averaged_matter_power(cosmo, k, z_lens, nz_lens)

    expected = 1.0 / (1.0 + z_lens)

    np.testing.assert_allclose(called_scale_factors, expected)


def test_lens_averaged_matter_power_does_not_renormalize_lens_weights(monkeypatch):
    """Test that lens weights are integrated as supplied without renormalization."""
    cosmo = {"h": 1.0}
    k = np.array([0.1, 1.0])
    z_lens = np.array([0.0, 1.0])
    nz_lens = np.array([2.0, 2.0])

    def fake_nonlin_matter_power(cosmo, k_mpc, scale_factor):
        return np.ones_like(k_mpc)

    monkeypatch.setattr(
        f"{POWER_SPECTRUM_MODULE}.ccl.nonlin_matter_power",
        fake_nonlin_matter_power,
    )

    result = lens_averaged_matter_power(
        cosmo,
        k,
        z_lens,
        nz_lens,
        nonlinear=True,
    )

    expected = np.full_like(k, 2.0)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "k",
    [
        np.array([0.0, 1.0]),
        np.array([-0.1, 1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 0.1]),
        np.array([[0.1, 1.0]]),
    ],
)
def test_lens_averaged_matter_power_rejects_invalid_k_grid(k):
    """Test that invalid k grids are rejected."""
    cosmo = {"h": 0.7}
    z_lens = np.array([0.0, 1.0])
    nz_lens = np.array([1.0, 1.0])

    with pytest.raises(ValueError):
        lens_averaged_matter_power(cosmo, k, z_lens, nz_lens)


@pytest.mark.parametrize(
    ("z_lens", "nz_lens"),
    [
        (np.array([0.0, 1.0]), np.array([1.0])),
        (np.array([[0.0, 1.0]]), np.array([1.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([[1.0, 1.0]])),
        (np.array([1.0, 0.0]), np.array([1.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([1.0, -1.0])),
    ],
)
def test_lens_averaged_matter_power_rejects_invalid_lens_distribution(
    z_lens,
    nz_lens,
):
    """Test that invalid lens redshift distributions are rejected."""
    cosmo = {"h": 0.7}
    k = np.array([0.1, 1.0])

    with pytest.raises(ValueError):
        lens_averaged_matter_power(cosmo, k, z_lens, nz_lens)
