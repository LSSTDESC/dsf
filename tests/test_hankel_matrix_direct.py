"""Companion unit tests for the matrix-direct Hankel backend."""

import numpy as np

from dsf.hankel.hankel_transform_matrix_direct import (
    HankelTransformMatrixDirect,
)


def test_matrix_direct_builds_expected_grid_shapes():
    """Test that matrix-direct constructs 1D grids and matching kernels."""
    transform = HankelTransformMatrixDirect(
        r_min=1.0,
        r_max=4.0,
        k_min=1.0,
        k_max=4.0,
        n_r=4,
        n_k=5,
        orders=(0,),
    )

    assert transform.k[0].shape == (5,)
    assert transform.r[0].shape == (4,)
    assert transform.j[0].shape == (4, 5)
    assert transform.j_t[0].shape == (5, 4)
    assert transform.weights[0].shape == (5,)


def test_matrix_direct_uses_log_space_trapezoidal_weights():
    """Test that integration weights give half weight to the endpoints."""
    transform = HankelTransformMatrixDirect(
        r_min=1.0,
        r_max=4.0,
        k_min=1.0,
        k_max=16.0,
        n_r=2,
        n_k=5,
        orders=(0,),
    )

    k = transform.k[0]
    log_spacing = np.log(k[1] / k[0])
    expected_dlnk = log_spacing * np.array([0.5, 1.0, 1.0, 1.0, 0.5])
    expected = k**2 * expected_dlnk / (2.0 * np.pi)

    np.testing.assert_allclose(transform.weights[0], expected)


def test_matrix_direct_projected_covariance_returns_square_matrix():
    """Test that matrix-direct projects two spectra into a square matrix."""
    transform = HankelTransformMatrixDirect(
        r_min=1.0,
        r_max=4.0,
        k_min=1.0,
        k_max=4.0,
        n_r=3,
        n_k=4,
        orders=(0,),
    )

    k = transform.k[0]
    pk1 = np.ones_like(k)
    pk2 = 2.0 * np.ones_like(k)

    r, cov = transform.projected_covariance(k_pk=k, pk1=pk1, pk2=pk2, order=0)

    assert r.shape == (3,)
    assert cov.shape == (3, 3)
    assert np.all(np.isfinite(cov))
