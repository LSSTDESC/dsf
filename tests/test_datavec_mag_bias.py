"""Unit tests for ``dsf.data_vector.mag_bias``."""

import numpy as np
import pyccl as ccl
import pytest

from dsf.utils import validators
from dsf.data_vector import mag_bias
from dsf.data_vector.mag_bias import (
    _inner_redshift_integrand,
    _lens_mag_distance_kernel,
    _lens_mag_lss_shear,
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
    class _FakeNonlin:
        def __init__(self, val=3.0):
            self.psp = object()

    class _FakeCosmo(dict):
        def __init__(self, h, Omega_m, pk_value=3.0):
            super().__init__(h=h, Omega_m=Omega_m)
            self._nonlin = _FakeNonlin(pk_value)
            self.cosmo = object()

        def get_nonlin_power(self):
            return self._nonlin

    return _FakeCosmo(0.7, 0.3)


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
    )

    params = get_lens_mag_integ_params()

    assert params["n_ell"] == 12
    assert params["ell_min"] == 1.0e-3
    assert params["ell_max"] == 1.0e3
    assert params["z_stepsize"] == 0.05
    assert params["z_min"] == 1.0e-4


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
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        mag_bias.ccl.lib,
        "pk2d_eval_multi",
        lambda psp, k_arr, a_use, cosmo_cosmo, n, zero: (
            np.full_like(np.asarray(k_arr, dtype=float), 3.0),
        ),
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


def test_lens_mag_lss_shear_returns_expected_shape(monkeypatch, cosmo):
    """Tests that the LSS shear has shape ``(n_theta)``."""
    set_lens_mag_integ_params(
        n_ell=4,
        ell_min=1.0,
        ell_max=10.0,
        z_stepsize=0.1,
        z_min=0.1,
    )

    theta = np.array([0.2, 0.9])

    monkeypatch.setattr(
        mag_bias,
        "_inner_redshift_integrand",
        lambda z_inner, ell, cosmo, z_lens, z_source: np.ones(
            (z_inner.size, ell.size),
            dtype=float,
        ),
    )

    lss_shear = _lens_mag_lss_shear(
        cosmo=cosmo,
        theta=theta,
        z_lens=0.5,
        z_source=1.0,
    )
    result = lss_shear.shape
    expected = theta.shape

    np.testing.assert_allclose(result, expected)


def test_lens_mag_lss_shear_rejects_too_short_inner_redshift_grid(cosmo):
    """Tests that an under-sampled generated inner-redshift grid is rejected."""
    set_lens_mag_integ_params(
        z_min=0.49,
        z_stepsize=0.2,
    )

    with pytest.raises(ValueError, match="z must contain at least 2 values"):
        _lens_mag_lss_shear(
            cosmo=cosmo,
            theta=np.array([0.01, 0.02]),
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
def test_lens_mag_lss_shear_rejects_invalid_grids(monkeypatch, ell, theta, cosmo):
    """Tests that invalid multipole or angular grids are rejected."""
    monkeypatch.setattr(
        validators,
        'validate_integration_params',
        lambda: None,
    )
    
    with pytest.raises(ValueError):
        set_lens_mag_integ_params(
            n_ell=len(ell),
            ell_min=np.min(ell),
            ell_max=np.max(ell),
        )
        _lens_mag_lss_shear(
            cosmo=cosmo,
            theta=theta,
            z_lens=0.5,
            z_source=1.0,
        )


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
    r = np.array([1.0, 2.0, 3.0])
    a_lens = 0.5
    a_source = 0.3
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
        a_lens=a_lens,
        a_source=a_source,
        cosmo=cosmo,
        alpha_lens=alpha_lens,
    )

    theta = r * a_lens / 100.0
    expected = 2.0 * a_lens**2 * 4.0e12 * (alpha_lens - 1.0) * (theta + 1.0) / 1.0e12

    np.testing.assert_allclose(result, expected)


def test_delta_sigma_lens_mag_correction_passes_expected_redshifts(
    monkeypatch,
    cosmo,
):
    """Tests that the public correction derives lens and source redshifts."""
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
        a_lens=0.5,
        a_source=0.4,
        cosmo=cosmo,
        alpha_lens=1.5,
    )

    assert calls == [(1.0, 1.5)]


@pytest.mark.parametrize(
    ("r", "a_lens", "a_source", "alpha_lens"),
    [
        (np.array([0.0, 1.0]), 0.5, 0.3, 1.0),
        (np.array([1.0, 2.0]), 0.0, 0.3, 1.0),
        (np.array([1.0, 2.0]), 0.5, 0.6, 1.0),
        (np.array([1.0, 2.0]), 0.5, 0.3, np.nan),
    ],
)
def test_delta_sigma_lens_mag_correction_rejects_invalid_inputs(
    r,
    a_lens,
    a_source,
    alpha_lens,
    cosmo,
):
    """Tests that the public correction rejects invalid inputs."""
    with pytest.raises(ValueError):
        delta_sigma_lens_mag_correction(
            r=r,
            a_lens=a_lens,
            a_source=a_source,
            cosmo=cosmo,
            alpha_lens=alpha_lens,
        )

@pytest.mark.slow
def test_delta_sigma_lens_mag_correction_matches_ccl():
    """Tests that delta_sigma_lens_mag_correction agrees with the CCL prediction."""
    cosmo = ccl.cosmology.CosmologyVanillaLCDM()
    Z_LENS = 0.3
    Z_SOURCE = 1.3
    SIGMA_NZ = 0.005
    ALPHA = 1.5

    ell_ccl = np.geomspace(1e-5, 1e6, 5000)
    r = np.geomspace(1e0, 1e2)
    theta = np.degrees(r / (1+Z_LENS) / ccl.angular_diameter_distance(cosmo, 1/(1+Z_LENS)))

    z_lens_ccl = np.linspace(0.01, 1.0, 500)
    nz_lens_ccl = np.exp(-0.5 * ((z_lens_ccl - Z_LENS)/SIGMA_NZ)**2)
    z_source_ccl = np.linspace(0.5, 1.5, 500)
    nz_source_ccl = np.exp(-0.5 * ((z_source_ccl - Z_SOURCE)/SIGMA_NZ)**2)

    t_g = ccl.NumberCountsTracer(cosmo, dndz=(z_lens_ccl, nz_lens_ccl), 
                                bias=None, 
                                mag_bias=(z_lens_ccl, ALPHA/2.5 * np.ones_like(nz_lens_ccl)), 
                                has_rsd=False)
    t_m = ccl.WeakLensingTracer(cosmo,
                                dndz=(z_source_ccl, nz_source_ccl),
                                has_shear=True)
    c_ell_ccl = ccl.angular_cl(cosmo, t_g, t_m, ell_ccl)
    gammat_ccl = ccl.correlation(cosmo, 
                                ell=ell_ccl, 
                                C_ell=c_ell_ccl, 
                                theta=theta, 
                                method='FFTLog', 
                                type='NG')
    correction_ccl = gammat_ccl / 1e12 * ((1/(1+Z_LENS))**2) * ccl.sigma_critical(cosmo, 
                                                                        a_lens=1/(1+Z_LENS), 
                                                                        a_source=1/(1+Z_SOURCE))

    set_lens_mag_integ_params(
        ell_min=1e-5,
        ell_max=1e6,
        n_ell=5000,
    )
    correction_dsf = delta_sigma_lens_mag_correction(r, 
                                                     1/(1+Z_LENS), 
                                                     1/(1+Z_SOURCE), 
                                                     cosmo, 
                                                     alpha_lens=ALPHA)
    
    # This is just a rough comparison, so require a match only within 3%.
    assert np.allclose(correction_ccl, correction_dsf, rtol=0.03, atol=0)