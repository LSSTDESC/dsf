"""Tests for ``dsf.utils.converters.py``."""

import numpy as np
import pyccl as ccl
import pytest

from dsf.utils.converters import (
    arcmin2_per_steradian,
    comoving_delta_sigma_to_proper,
    comoving_distance_h,
    deg2_to_arcmin2,
    hubble_constant_per_s_per_mpc,
    hubble_over_c_cubed,
    redshift_to_scale_factor,
    resolve_h,
    resolve_omega_m,
    rho_critical_comoving_msun_mpc3,
    rho_critical_projected_msun_pc2_per_mpc,
    scale_factor_to_redshift,
    sigma_crit_prefactor_msun_h_pc2,
    speed_of_light_mpc_per_s,
)


def test_arcmin2_per_steradian_matches_geometric_factor():
    """Tests that steradians are converted to square arcminutes."""
    expected = (180.0 * 60.0 / np.pi) ** 2

    assert arcmin2_per_steradian() == pytest.approx(expected)


def test_deg2_to_arcmin2_multiplies_by_sixty_squared():
    """Tests that square degrees are converted to square arcminutes."""
    assert deg2_to_arcmin2(2.5) == pytest.approx(2.5 * 60.0**2)


def test_scale_factor_to_redshift_handles_arrays():
    """Tests that scale factors are converted to redshifts."""
    actual = scale_factor_to_redshift(np.array([1.0, 0.5, 0.25]))

    np.testing.assert_allclose(actual, np.array([0.0, 1.0, 3.0]))


def test_redshift_to_scale_factor_handles_arrays():
    """Tests that redshifts are converted to scale factors."""
    actual = redshift_to_scale_factor(np.array([0.0, 1.0, 3.0]))

    np.testing.assert_allclose(actual, np.array([1.0, 0.5, 0.25]))


def test_redshift_and_scale_factor_conversions_roundtrip():
    """Tests that redshift and scale-factor conversions are inverse operations."""
    redshift = np.array([0.0, 0.3, 1.0, 2.0])

    actual = scale_factor_to_redshift(redshift_to_scale_factor(redshift))

    np.testing.assert_allclose(actual, redshift)


def test_speed_of_light_mpc_per_s_is_positive():
    """Tests that the speed of light conversion returns a positive scalar."""
    assert speed_of_light_mpc_per_s() > 0.0


def test_hubble_constant_per_s_per_mpc_uses_reduced_hubble_parameter():
    """Tests that H0 is returned as 100 h km/s/Mpc."""
    actual = hubble_constant_per_s_per_mpc(0.7)

    assert actual.value == pytest.approx(70.0)
    assert str(actual.unit) == "km / (Mpc s)"


def test_hubble_over_c_cubed_is_positive():
    """Tests that the H0 over c cubed conversion is positive."""
    assert hubble_over_c_cubed(0.7) > 0.0


def test_comoving_delta_sigma_to_proper_rescales_radius_and_amplitude():
    """Tests that proper Delta Sigma wraps a comoving Delta Sigma function."""
    calls = {}

    def func(r, a, amplitude=1.0):
        calls["r"] = np.asarray(r)
        calls["a"] = a
        return amplitude * np.asarray(r)

    wrapper = comoving_delta_sigma_to_proper(func)

    actual = wrapper(np.array([1.0, 2.0]), 0.5, amplitude=3.0)

    np.testing.assert_allclose(calls["r"], np.array([2.0, 4.0]))
    assert calls["a"] == pytest.approx(0.5)
    np.testing.assert_allclose(actual, np.array([24.0, 48.0]))


def test_rho_critical_projected_divides_comoving_density_by_pc2_factor():
    """Tests that projected critical density uses Msun per pc squared per Mpc."""
    cosmo = ccl.CosmologyVanillaLCDM()

    rho_crit = rho_critical_comoving_msun_mpc3(cosmo)
    projected = rho_critical_projected_msun_pc2_per_mpc(cosmo)

    assert projected == pytest.approx(rho_crit / 1.0e12)


def test_rho_critical_comoving_allows_explicit_h_override():
    """Tests that critical density can use an explicit Hubble parameter."""
    cosmo = ccl.CosmologyVanillaLCDM()

    actual = rho_critical_comoving_msun_mpc3(cosmo, h=0.5)
    expected = float(cosmo.rho_x(1.0, "critical", is_comoving=True) / 0.5**2)

    assert actual == pytest.approx(expected)


def test_comoving_distance_h_returns_scalar_for_scalar_redshift():
    """Tests that scalar redshifts return scalar comoving distances."""
    cosmo = ccl.CosmologyVanillaLCDM()

    actual = comoving_distance_h(cosmo, 0.5, h=0.7)

    assert isinstance(actual, float)
    assert actual > 0.0


def test_comoving_distance_h_preserves_array_shape():
    """Tests that array redshifts preserve their input shape."""
    cosmo = ccl.CosmologyVanillaLCDM()
    redshift = np.array([[0.1, 0.2], [0.3, 0.4]])

    actual = comoving_distance_h(cosmo, redshift, h=0.7)

    assert actual.shape == redshift.shape
    assert np.all(actual > 0.0)


def test_resolve_h_uses_supplied_value():
    """Tests that explicit h values are returned unchanged."""
    cosmo = ccl.CosmologyVanillaLCDM()

    assert resolve_h(cosmo, 0.71) == pytest.approx(0.71)


def test_resolve_h_reads_value_from_cosmology():
    """Tests that missing h values are read from the cosmology."""
    cosmo = ccl.CosmologyVanillaLCDM()

    assert resolve_h(cosmo, None) == pytest.approx(float(cosmo["h"]))


@pytest.mark.parametrize("bad_h", [0.0, -0.1, np.nan])
def test_resolve_h_rejects_invalid_values(bad_h):
    """Tests that invalid Hubble parameters are rejected."""
    cosmo = ccl.CosmologyVanillaLCDM()

    with pytest.raises(ValueError, match="h"):
        resolve_h(cosmo, bad_h)


def test_resolve_omega_m_uses_supplied_value():
    """Tests that explicit Omega_m values are returned unchanged."""
    cosmo = ccl.CosmologyVanillaLCDM()

    assert resolve_omega_m(cosmo, 0.31) == pytest.approx(0.31)


def test_resolve_omega_m_reads_value_from_cosmology():
    """Tests that missing Omega_m values are read from the cosmology."""
    cosmo = ccl.CosmologyVanillaLCDM()

    assert resolve_omega_m(cosmo, None) == pytest.approx(float(cosmo["Omega_m"]))


@pytest.mark.parametrize("bad_omega_m", [0.0, -0.1, np.nan])
def test_resolve_omega_m_rejects_invalid_values(bad_omega_m):
    """Tests that invalid matter densities are rejected."""
    cosmo = ccl.CosmologyVanillaLCDM()

    with pytest.raises(ValueError, match="omega_m"):
        resolve_omega_m(cosmo, bad_omega_m)


def test_sigma_crit_prefactor_is_positive():
    """Tests that the Sigma_crit prefactor is positive."""
    assert sigma_crit_prefactor_msun_h_pc2() > 0.0
