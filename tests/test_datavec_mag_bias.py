"""Unit tests for ``dsf.data_vector.mag_bias``."""

import numpy as np
import pytest

from dsf.data_vector import mag_bias
from dsf.data_vector.mag_bias import (
    _inner_redshift_integrand,
    _lens_mag_distance_kernel,
    _lens_mag_lss_shear,
    _multipole_integrand,
    delta_sigma_lens_mag_correction,
    get_lens_mag_integ_params,
    set_lens_mag_integ_params,
)


@pytest.fixture(autouse=True)
def restore_lens_mag_integ_params():
    """Restore lens-magnification integration parameters after each test."""
    original = get_lens_mag_integ_params()

    yield

    current = get_lens_mag_integ_params()
    reset = {key: original[key] for key in current}
    set_lens_mag_integ_params(**reset)


@pytest.fixture
def cosmo():
    """Return a small dictionary-like cosmology object for unit tests."""
    return {
        "h": 0.7,
        "Omega_m": 0.3,
    }


def test_get_lens_mag_integ_params_returns_copy():
    """Tests that integration parameters are returned as a copy."""
    params = get_lens_mag_integ_params()

    params["n_ell"] = 12

    assert get_lens_mag_integ_params()["n_ell"] != 12


def test_set_lens_mag_integ_params_updates_known_parameters():
    """Tests that known integration parameters can be updated."""
    set_lens_mag_integ_params(
        n_ell=12,
        ell_min=1.0e-3,
        ell_max=1.0e3,
        z_stepsize=0.05,
        z_min=1.0e-4,
        delta_z_source=0.5,
    )

    params = get_lens_mag_integ_params()

    assert params["n_ell"] == 12
    assert params["ell_min"] == 1.0e-3
    assert params["ell_max"] == 1.0e3
    assert params["z_stepsize"] == 0.05
    assert params["z_min"] == 1.0e-4
    assert params["delta_z_source"] == 0.5


def test_set_lens_mag_integ_params_rejects_unknown_parameter():
    """Tests that unknown integration parameters are rejected."""
    with pytest.raises(KeyError, match="Unknown lens-magnification"):
        set_lens_mag_integ_params(not_a_parameter=1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_ell": 0},
        {"ell_min": 0.0},
        {"ell_max": 1.0e-5},
        {"z_stepsize": 0.0},
        {"z_min": -0.1},
        {"delta_z_source": 0.0},
    ],
)
def test_set_lens_mag_integ_params_rejects_invalid_values(kwargs):
    """Tests that invalid integration parameters are rejected."""
    with pytest.raises(ValueError):
        set_lens_mag_integ_params(**kwargs)


def test_lens_mag_distance_kernel_uses_expected_ccl_distances(monkeypatch, cosmo):
    """Tests that the distance kernel combines angular distances correctly."""

    def fake_angular_diameter_distance(cosmo, a1, a2=None):
        """Return deterministic one- or two-scale-factor distances."""
        a1_arr = np.asarray(a1, dtype=float)

        if a2 is None:
            return 10.0 * a1_arr

        return 100.0 * (np.asarray(a2, dtype=float) - a1_arr)

    monkeypatch.setattr(
        mag_bias.ccl,
        "angular_diameter_distance",
        fake_angular_diameter_distance,
    )

    a_inner = np.array([0.8, 0.9])
    a_lens = np.array([0.7, 0.7])
    a_source = np.array([0.5, 0.5])

    result = _lens_mag_distance_kernel(
        cosmo=cosmo,
        a_inner=a_inner,
        a_lens=a_lens,
        a_source=a_source,
    )

    expected = (
        fake_angular_diameter_distance(cosmo, a_inner, a_lens)
        * fake_angular_diameter_distance(cosmo, a_inner, a_source)
        / (
            fake_angular_diameter_distance(cosmo, a_lens)
            * fake_angular_diameter_distance(cosmo, a_source)
        )
    )

    np.testing.assert_allclose(result, expected)


def test_inner_redshift_integrand_returns_expected_shape(monkeypatch, cosmo):
    """Tests that the inner redshift integrand has shape ``(n_z, n_ell)``."""
    z_inner = np.array([0.1, 0.2, 0.3])
    ell = np.array([10.0, 20.0])

    monkeypatch.setattr(
        mag_bias,
        "_lens_mag_distance_kernel",
        lambda *args: np.ones(z_inner.size),
    )
    monkeypatch.setattr(
        mag_bias.ccl,
        "h_over_h0",
        lambda cosmo, a: np.ones_like(np.asarray(a, dtype=float)),
    )
    monkeypatch.setattr(
        mag_bias.ccl,
        "angular_diameter_distance",
        lambda cosmo, a: np.ones_like(np.asarray(a, dtype=float)) * 2.0,
    )
    monkeypatch.setattr(
        mag_bias.ccl,
        "nonlin_matter_power",
        lambda cosmo, k, a: np.full_like(np.asarray(k, dtype=float), 3.0),
    )

    result = _inner_redshift_integrand(
        z_inner=z_inner,
        ell=ell,
        cosmo=cosmo,
        z_lens=0.5,
        z_source=1.0,
    )

    expected = ((1.0 + z_inner) ** 2)[:, None] * np.full((z_inner.size, ell.size), 3.0)

    assert result.shape == (z_inner.size, ell.size)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "z_inner",
    [
        np.array([[0.1, 0.2]]),
        np.array([0.1]),
        np.array([0.1, np.nan]),
        np.array([-0.1, 0.1]),
    ],
)
def test_inner_redshift_integrand_rejects_invalid_redshift_grid(z_inner, cosmo):
    """Tests that invalid inner-redshift grids are rejected."""
    with pytest.raises(ValueError):
        _inner_redshift_integrand(
            z_inner=z_inner,
            ell=np.array([10.0, 20.0]),
            cosmo=cosmo,
            z_lens=0.5,
            z_source=1.0,
        )


def test_inner_redshift_integrand_rejects_invalid_redshift_pair(cosmo):
    """Tests that invalid lens/source redshift ordering is rejected."""
    with pytest.raises(ValueError, match="z_source must be greater"):
        _inner_redshift_integrand(
            z_inner=np.array([0.1, 0.2]),
            ell=np.array([10.0, 20.0]),
            cosmo=cosmo,
            z_lens=1.0,
            z_source=0.5,
        )


def test_inner_redshift_integrand_rejects_invalid_ell(cosmo):
    """Tests that invalid multipoles are rejected."""
    with pytest.raises(ValueError, match="ell"):
        _inner_redshift_integrand(
            z_inner=np.array([0.1, 0.2]),
            ell=np.array([0.0, 20.0]),
            cosmo=cosmo,
            z_lens=0.5,
            z_source=1.0,
        )


def test_multipole_integrand_returns_expected_shape(monkeypatch, cosmo):
    """Tests that the multipole integrand has shape ``(n_ell, n_theta)``."""
    set_lens_mag_integ_params(
        n_ell=4,
        ell_min=1.0,
        ell_max=10.0,
        z_stepsize=0.1,
        z_min=0.1,
        delta_z_source=0.5,
    )

    ell = np.array([2.0, 4.0, 8.0])
    theta = np.array([0.01, 0.02])

    monkeypatch.setattr(
        mag_bias,
        "_inner_redshift_integrand",
        lambda z_inner, ell, cosmo, z_lens, z_source: np.ones(
            (z_inner.size, ell.size),
            dtype=float,
        ),
    )

    result = _multipole_integrand(
        ell=ell,
        theta=theta,
        cosmo=cosmo,
        z_lens=0.5,
        z_source=1.0,
    )

    z_arr = np.arange(0.1, 0.5, step=0.1)
    inner_integral = np.trapezoid(np.ones((z_arr.size, ell.size)), z_arr, axis=0)
    expected = mag_bias.jv(2, ell[:, None] * theta[None, :]) * ell[:, None]
    expected = expected * inner_integral[:, None]

    assert result.shape == (ell.size, theta.size)
    np.testing.assert_allclose(result, expected)


def test_multipole_integrand_rejects_too_short_inner_redshift_grid(cosmo):
    """Tests that an under-sampled generated inner-redshift grid is rejected."""
    set_lens_mag_integ_params(
        z_min=0.49,
        z_stepsize=0.2,
    )

    with pytest.raises(ValueError, match="inner redshift grid"):
        _multipole_integrand(
            ell=np.array([2.0, 4.0]),
            theta=np.array([0.01, 0.02]),
            cosmo=cosmo,
            z_lens=0.5,
            z_source=1.0,
        )


@pytest.mark.parametrize(
    ("ell", "theta"),
    [
        (np.array([0.0, 1.0]), np.array([0.01, 0.02])),
        (np.array([1.0, 2.0]), np.array([0.0, 0.02])),
    ],
)
def test_multipole_integrand_rejects_invalid_grids(ell, theta, cosmo):
    """Tests that invalid multipole or angular grids are rejected."""
    with pytest.raises(ValueError):
        _multipole_integrand(
            ell=ell,
            theta=theta,
            cosmo=cosmo,
            z_lens=0.5,
            z_source=1.0,
        )


def test_lens_mag_lss_shear_integrates_multipole_integrand(monkeypatch, cosmo):
    """Tests that LSS shear integrates the multipole integrand with prefactor."""
    set_lens_mag_integ_params(
        n_ell=3,
        ell_min=1.0,
        ell_max=4.0,
        z_stepsize=0.1,
        z_min=0.1,
        delta_z_source=0.5,
    )

    theta = np.array([0.01, 0.02])

    def fake_multipole_integrand(ell, theta, cosmo, z_lens, z_source):
        """Return a deterministic multipole integrand."""
        return ell[:, None] * np.ones((ell.size, theta.size))

    monkeypatch.setattr(
        mag_bias,
        "_multipole_integrand",
        fake_multipole_integrand,
    )
    monkeypatch.setattr(
        mag_bias,
        "hubble_over_c_cubed",
        lambda h: 2.0,
    )

    result = _lens_mag_lss_shear(
        cosmo=cosmo,
        theta=theta,
        z_lens=0.5,
        z_source=1.0,
    )

    ell = np.geomspace(1.0, 4.0, 3)
    integral = np.trapezoid(ell[:, None] * np.ones((ell.size, theta.size)), ell, axis=0)
    prefactor = 9.0 * 2.0 * cosmo["Omega_m"] ** 2 / (8.0 * np.pi)

    np.testing.assert_allclose(result, prefactor * integral)


def test_lens_mag_lss_shear_rejects_invalid_theta(cosmo):
    """Tests that invalid angular separations are rejected."""
    with pytest.raises(ValueError, match="theta"):
        _lens_mag_lss_shear(
            cosmo=cosmo,
            theta=np.array([0.0, 0.1]),
            z_lens=0.5,
            z_source=1.0,
        )


def test_delta_sigma_lens_mag_correction_matches_expected_formula(monkeypatch, cosmo):
    """Tests that the public correction applies the expected prefactors."""
    set_lens_mag_integ_params(delta_z_source=0.5)

    r = np.array([1.0, 2.0, 3.0])
    a = 0.5
    alpha_lens = 2.0

    monkeypatch.setattr(
        mag_bias.ccl,
        "angular_diameter_distance",
        lambda cosmo, a: 100.0,
    )
    monkeypatch.setattr(
        mag_bias,
        "_lens_mag_lss_shear",
        lambda cosmo, theta, z_lens, z_source: np.asarray(theta, dtype=float) + 1.0,
    )
    monkeypatch.setattr(
        mag_bias.ccl,
        "sigma_critical",
        lambda cosmo, a_lens, a_source: 4.0e12,
    )

    result = delta_sigma_lens_mag_correction(
        r=r,
        a=a,
        cosmo=cosmo,
        alpha_lens=alpha_lens,
    )

    theta = r * a / 100.0
    expected = 2.0 * a**2 * 4.0e12 * (alpha_lens - 1.0) * (theta + 1.0) / 1.0e12

    np.testing.assert_allclose(result, expected)


def test_delta_sigma_lens_mag_correction_passes_expected_redshifts(
    monkeypatch,
    cosmo,
):
    """Tests that the public correction derives lens and source redshifts."""
    set_lens_mag_integ_params(delta_z_source=0.25)

    calls = []

    monkeypatch.setattr(
        mag_bias.ccl,
        "angular_diameter_distance",
        lambda cosmo, a: 100.0,
    )

    def fake_lss_shear(cosmo, theta, z_lens, z_source):
        """Record lens/source redshift inputs and return deterministic shear."""
        calls.append((z_lens, z_source))
        return np.ones_like(theta)

    monkeypatch.setattr(
        mag_bias,
        "_lens_mag_lss_shear",
        fake_lss_shear,
    )
    monkeypatch.setattr(
        mag_bias.ccl,
        "sigma_critical",
        lambda cosmo, a_lens, a_source: 1.0e12,
    )

    delta_sigma_lens_mag_correction(
        r=np.array([1.0, 2.0]),
        a=0.5,
        cosmo=cosmo,
        alpha_lens=1.5,
    )

    assert calls == [(1.0, 1.25)]


@pytest.mark.parametrize(
    ("r", "a", "alpha_lens"),
    [
        (np.array([0.0, 1.0]), 0.5, 1.0),
        (np.array([1.0, 2.0]), 0.0, 1.0),
        (np.array([1.0, 2.0]), 0.5, np.nan),
    ],
)
def test_delta_sigma_lens_mag_correction_rejects_invalid_inputs(
    r,
    a,
    alpha_lens,
    cosmo,
):
    """Tests that the public correction rejects invalid inputs."""
    with pytest.raises(ValueError):
        delta_sigma_lens_mag_correction(
            r=r,
            a=a,
            cosmo=cosmo,
            alpha_lens=alpha_lens,
        )
