"""Unit tests for ``dsf.utils.hankel_transform_1d``."""

import numpy as np
import pytest
from pytest import param

from dsf.utils.hankel_transform_1d import hankel_projected_order_2, hankel_spherical_order_0


@pytest.mark.parametrize(
    "P_k,k,r_eval,expected_len",
    [
        param(
            np.ones(16),
            np.logspace(-2, 1, 16),
            np.logspace(-1, 0.5, 7),
            7,
            id="hankel_j0_output_length_basic",
        ),
        param(
            np.linspace(0.1, 2.0, 8),
            np.logspace(-3, -1, 8),
            np.array([0.1, 0.2]),
            2,
            id="hankel_j0_output_length_two_points",
        ),
    ],
)
def test_hankel_spherical_order_0_output_exists_and_correct_length(
    P_k,
    k,
    r_eval,
    expected_len,
):
    xi_func = hankel_spherical_order_0(P_k=P_k, k=k, use_offset=True)
    xi_vals = xi_func(r_eval)

    assert isinstance(xi_vals, np.ndarray)
    assert len(xi_vals) == expected_len


@pytest.mark.parametrize(
    "C_ell,ell,theta_eval,expected_len",
    [
        param(
            np.ones(16),
            np.logspace(1, 3, 16),
            np.logspace(0, 2, 7),
            7,
            id="hankel_projected_order_2_output_length_basic",
        ),
        param(
            np.linspace(0.01, 1.0, 10),
            np.logspace(0, 2, 10),
            np.array([0.01, 0.02, 0.05]),
            3,
            id="hankel_projected_order_2_output_length_three_points",
        ),
    ],
)
def test_hankel_projected_order_2_output_exists_and_correct_length(
    C_ell,
    ell,
    theta_eval,
    expected_len,
):
    gamma_t_func = hankel_projected_order_2(C_ell=C_ell, ell=ell, use_offset=True)
    gamma_vals = gamma_t_func(theta_eval)

    assert isinstance(gamma_vals, np.ndarray)
    assert len(gamma_vals) == expected_len


@pytest.mark.parametrize(
    "transform_func,grid_name",
    [
        param(hankel_spherical_order_0, "k", id="invalid_spacing_k_hankel_spherical_order_0"),
        param(hankel_projected_order_2, "ell", id="invalid_spacing_ell_hankel_projected_order_2"),
    ],
)
def test_hankel_invalid_spacing_raises(transform_func, grid_name):
    # Use non-logspaced arrays: linear spacing should fail validation
    P_or_C = np.ones(4)
    non_logspaced = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError):
        transform_func(P_or_C, non_logspaced, use_offset=True)

@pytest.mark.slow
def test_hankel_spherical_order_0_matches_ccl():
    import pyccl as ccl

    cosmo = ccl.cosmology.CosmologyVanillaLCDM()

    k_arr = np.logspace(-5, 5, 1000)
    r_arr = np.geomspace(0.1, 100, 100)
    z = 0.3

    xi_dsf = hankel_spherical_order_0(cosmo.nonlin_matter_power(k_arr, 1/(1+z)), 
                                      k_arr, 
                                      use_offset=False)(r_arr)
    xi_ccl = ccl.correlation_3d(cosmo,r=r_arr, a=1/(1+z), p_of_k_a=cosmo.get_nonlin_power())

    assert np.allclose(xi_dsf, xi_ccl, rtol=0.005, atol=0)