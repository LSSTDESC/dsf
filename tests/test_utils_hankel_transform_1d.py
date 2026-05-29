"""Unit tests for ``dsf.utils.hankel_transform_1d``."""

import numpy as np
import pytest
from pytest import param

from dsf.utils.hankel_transform_1d import hankel_projected_order_2, hankel_spherical_order_0


@pytest.mark.parametrize(
    "k,pk,r_eval,expected_len",
    [
        param(
            np.geomspace(1.0e-2, 10.0, 16),
            np.ones(16),
            np.geomspace(0.2, 50, 7),
            7,
            id="hankel_j0_output_length_basic",
        ),
        param(
            np.geomspace(1.0e-2, 10.0, 8),
            np.linspace(0.1, 2.0, 8),
            np.array([0.2, 50]),
            2,
            id="hankel_j0_output_length_two_points",
        ),
    ],
)
def test_hankel_spherical_order_0_output_exists_and_correct_length(
    k,
    pk,
    r_eval,
    expected_len,
):
    """Tests that hankel_spherical_order_0 has shape ``len(r_eval)``."""
    xi_func = hankel_spherical_order_0(k=k, pk=pk, use_offset=False)
    xi_vals = xi_func(r_eval)

    assert isinstance(xi_vals, np.ndarray)
    assert len(xi_vals) == expected_len


@pytest.mark.parametrize(
    "ell,C_ell,theta_eval,expected_len",
    [
        param(
            np.geomspace(1.0e-2, 10.0, 16),
            np.ones(16),
            np.geomspace(0.2, 50, 7),
            7,
            id="hankel_projected_order_2_output_length_basic",
        ),
        param(
            np.geomspace(1.0e-2, 10.0, 8),
            np.geomspace(0.01, 1.0, 8),
            np.array([0.2, 50]),
            2,
            id="hankel_projected_order_2_output_length_two_points",
        ),
    ],
)
def test_hankel_projected_order_2_output_exists_and_correct_length(
    ell,
    C_ell,
    theta_eval,
    expected_len,
):
    """Tests that hankel_projected_order_2 has shape ``len(theta_eval)``."""
    gamma_t_func = hankel_projected_order_2(ell=ell, C_ell=C_ell, use_offset=False)
    gamma_vals = gamma_t_func(theta_eval)

    assert isinstance(gamma_vals, np.ndarray)
    assert len(gamma_vals) == expected_len


@pytest.mark.parametrize(
    "transform_func",
    [
        param(hankel_spherical_order_0, id="invalid_spacing_k_hankel_spherical_order_0"),
        param(hankel_projected_order_2, id="invalid_spacing_ell_hankel_projected_order_2"),
    ],
)
def test_hankel_invalid_spacing_raises(transform_func):
    """Tests that non-logspaced input arrays are rejected."""
    P_or_C = np.ones(4)
    non_logspaced = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError):
        transform_func(non_logspaced, P_or_C, use_offset=False)
        
        
@pytest.mark.parametrize(
    "k,pk,r_eval",
    [
        param(
            np.geomspace(1.0e-2, 10.0, 8),
            np.geomspace(0.01, 1.0, 8),
            np.array([0.01, 50]),
        ),
    ],
)
def test_hankel_spherical_order_0_rejects_interpolation_outside_bounds(
    k,pk,r_eval,
):
    """Tests that hankel_spherical_order_0 rejects r outside the interpolation grid."""
    with pytest.raises(ValueError):
        xi_func = hankel_spherical_order_0(k=k, pk=pk, use_offset=False)
        xi_func(r_eval)
        
        
@pytest.mark.parametrize(
    "ell,C_ell,theta_eval",
    [
        param(
            np.geomspace(1.0e-2, 10.0, 8),
            np.geomspace(0.01, 1.0, 8),
            np.array([0.01, 50]),
        ),
    ],
)
def test_hankel_projected_order_2_rejects_interpolation_outside_bounds(
    ell,
    C_ell,
    theta_eval,
):
    """Tests that hankel_projected_order_2 rejects theta outside the interpolation grid."""
    with pytest.raises(ValueError):
        gamma_t_func = hankel_projected_order_2(ell=ell, C_ell=C_ell, use_offset=False)
        gamma_t_func(theta_eval)
        

@pytest.mark.slow
def test_hankel_spherical_order_0_matches_ccl():
    """Tests that hankel_spherical_order_0 agrees with the CCL transform."""
    import pyccl as ccl

    cosmo = ccl.cosmology.CosmologyVanillaLCDM()

    k_arr = np.geomspace(1.0e-5, 1.0e5, 1000)
    r_arr = np.geomspace(0.1, 100, 100)
    z = 0.3

    xi_dsf = hankel_spherical_order_0(k_arr,
                                      cosmo.nonlin_matter_power(k_arr, 1/(1+z)), 
                                      use_offset=False)(r_arr)
    xi_ccl = ccl.correlation_3d(cosmo,r=r_arr, a=1/(1+z), p_of_k_a=cosmo.get_nonlin_power())

    assert np.allclose(xi_dsf, xi_ccl, rtol=0.005, atol=0)