"""Benchmarking tests for the Hankel transform module."""

import numpy as np
import pyccl as ccl
import pytest
from scipy.special import jv

from dsf.hankel.hankel import HankelTransform
from dsf.utils.integrators import trapezoid_integral


@pytest.mark.slow
def test_hankel_spherical_fftlog_matches_ccl():
    """Tests that fftlog hankel_spherical agrees with the CCL transform."""
    cosmo = ccl.cosmology.CosmologyVanillaLCDM()

    k_arr = np.geomspace(1.0e-5, 1.0e5, 1000)
    r_arr = np.geomspace(0.1, 100, 100)
    z = 0.3

    ht_fft = HankelTransform(backend="fftlog")
    _, xi_dsf = ht_fft.spherical_correlation_interpolated(
        r_arr, k_pk=k_arr, pk=cosmo.nonlin_matter_power(k_arr, 1 / (1 + z)), use_offset=False
    )
    xi_ccl = ccl.correlation_3d(cosmo, r=r_arr, a=1 / (1 + z), p_of_k_a=cosmo.get_nonlin_power())

    assert np.allclose(xi_dsf, xi_ccl, rtol=0.005, atol=0)


@pytest.mark.slow
def test_hankel_projected_fftlog_matches_direct_integration():
    """Tests that fftlog hankel_projected agrees with direct Bessel integration."""
    cosmo = ccl.cosmology.CosmologyVanillaLCDM()

    ell_arr = np.geomspace(1e-6, 1e6, 30000)
    theta_arr = np.radians(np.geomspace(0.1, 100, 100))
    c_ell_arr = cosmo.nonlin_power(ell_arr, 1)

    bessel_kernel = jv(2, ell_arr[:, None] * theta_arr[None, :]) * ell_arr[:, None]
    direct_integ_result = trapezoid_integral(
        bessel_kernel * c_ell_arr[:, None],
        ell_arr,
        axis=0,
    ) / (2 * np.pi)

    ht_fft = HankelTransform(backend="fftlog")
    _, dsf_result = ht_fft.projected_correlation_interpolated(
        theta_arr, ell=ell_arr, c_ell=c_ell_arr, order=2, use_offset=False
    )

    assert np.allclose(dsf_result, direct_integ_result, rtol=0.005, atol=0)


@pytest.mark.slow
def test_hankel_projected_matrix_zeros_matches_direct_integration():
    """Tests that matrix_zeros hankel_projected agrees with direct Bessel integration."""
    cosmo = ccl.cosmology.CosmologyVanillaLCDM()

    ell_arr = np.geomspace(1e-6, 1e6, 30000)
    theta_arr = np.radians(np.geomspace(3, 100, 100))
    c_ell_arr = cosmo.nonlin_power(ell_arr, 1)

    bessel_kernel = jv(2, ell_arr[:, None] * theta_arr[None, :]) * ell_arr[:, None]
    direct_integ_result = trapezoid_integral(
        bessel_kernel * c_ell_arr[:, None],
        ell_arr,
        axis=0,
    ) / (2 * np.pi)

    ht_matz = HankelTransform(
        backend="matrix_zeros",
        r_min=np.min(theta_arr),
        r_max=np.max(theta_arr),
        k_min=8.0e-2,
        k_max=1.0e3,
        max_iterations=300,
        n_zeros=25000,
        n_zeros_step=5000,
        prune_r=None,
        verbose=True,
        orders=[2],
    )

    _, dsf_result = ht_matz.projected_correlation_interpolated(
        theta_arr, ell=ell_arr, c_ell=c_ell_arr, order=2, use_offset=False
    )

    assert np.allclose(dsf_result, direct_integ_result, rtol=0.005, atol=0)


@pytest.mark.slow
def test_hankel_projected_matrix_direct_matches_direct_integration():
    """Tests that matrix_direct hankel_projected agrees with direct Bessel integration."""
    cosmo = ccl.cosmology.CosmologyVanillaLCDM()

    ell_arr = np.geomspace(1e-6, 1e6, 30000)
    theta_arr = np.radians(np.geomspace(3, 100, 100))
    c_ell_arr = cosmo.nonlin_power(ell_arr, 1)

    bessel_kernel = jv(2, ell_arr[:, None] * theta_arr[None, :]) * ell_arr[:, None]
    direct_integ_result = trapezoid_integral(
        bessel_kernel * c_ell_arr[:, None],
        ell_arr,
        axis=0,
    ) / (2 * np.pi)

    ht_matd = HankelTransform(
        backend="matrix_direct",
        r_min=np.min(theta_arr),
        r_max=np.max(theta_arr),
        k_min=8.0e-2,
        k_max=1.0e3,
        n_k=5000,
        n_r=2000,
        orders=[2],
    )

    _, dsf_result = ht_matd.projected_correlation_interpolated(
        theta_arr, ell=ell_arr, c_ell=c_ell_arr, order=2, use_offset=False
    )

    assert np.allclose(dsf_result, direct_integ_result, rtol=0.005, atol=0)


@pytest.mark.slow
def test_hankel_projected_covariance_agrees_between_zeros_and_direct():
    """Tests that the projected covariance from matrix_zeros and matrix_direct backends agree."""
    cosmo = ccl.cosmology.CosmologyVanillaLCDM()

    theta_arr = np.radians(np.geomspace(3, 100, 100))
    r_out = np.geomspace(np.radians(5), np.radians(90), 30)
    a = 0.5

    ht_matz = HankelTransform(
        backend="matrix_zeros",
        r_min=np.min(theta_arr),
        r_max=np.max(theta_arr),
        k_min=8.0e-2,
        k_max=1.0e3,
        max_iterations=300,
        n_zeros=25000,
        n_zeros_step=5000,
        prune_r=None,
        verbose=True,
        orders=[2],
    )

    ell_z = ht_matz.backend.k[2]
    r_gkgk_z, cov_gkgk_z = ht_matz.projected_covariance(
        k_pk=ell_z,
        pk1=cosmo.nonlin_power(ell_z, a),
        pk2=cosmo.nonlin_power(ell_z, a),
        order=2,
    )
    r_matz, cov_matz = ht_matz.bin_radial_matrix(r_gkgk_z, cov_gkgk_z, r_out * cosmo["h"])

    ht_matd = HankelTransform(
        backend="matrix_direct",
        r_min=np.min(theta_arr),
        r_max=np.max(theta_arr),
        k_min=5.0e-2,
        k_max=1.0e3,
        n_k=5000,
        n_r=2000,
        orders=[2],
    )

    ell_d = ht_matd.backend.k[2]
    r_gkgk_d, cov_gkgk_d = ht_matd.projected_covariance(
        k_pk=ell_d,
        pk1=cosmo.nonlin_power(ell_d, a),
        pk2=cosmo.nonlin_power(ell_d, a),
        order=2,
    )
    r_matd, cov_matd = ht_matd.bin_radial_matrix(r_gkgk_d, cov_gkgk_d, r_out * cosmo["h"])

    assert np.allclose(cov_matz, cov_matd, rtol=0.05, atol=0)
    assert np.all(r_matz == r_matd)
