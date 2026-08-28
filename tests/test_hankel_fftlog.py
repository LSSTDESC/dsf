"""Unit tests for ``dsf.utils.hankel_transform_1d``."""

import numpy as np
import pytest
from pytest import param

from dsf.hankel.hankel import HankelTransform
from dsf.hankel.hankel_transform_fftlog import (
    HankelTransformFFTLog,
    hankel_projected,
    hankel_spherical,
)


@pytest.mark.parametrize(
    "k_pk,pk,expected_len",
    [
        param(
            np.geomspace(1.0e-2, 10.0, 16),
            np.ones(16),
            16,
            id="hankel_spherical_output_length_basic",
        ),
        param(
            np.geomspace(1.0e-2, 10.0, 8),
            np.linspace(0.1, 2.0, 8),
            8,
            id="hankel_spherical_output_length_two_points",
        ),
    ],
)
def test_spherical_correlation_output_exists_and_correct_length(
    k_pk,
    pk,
    expected_len,
):
    """Tests that spherical_correlation has shape ``(len(k), len(k))``."""
    ht_fft = HankelTransform(backend="fftlog")
    r_vals, xi_vals = ht_fft.spherical_correlation(
        k_pk=k_pk,
        pk=pk,
        use_offset=False,
    )

    assert isinstance(r_vals, np.ndarray)
    assert isinstance(xi_vals, np.ndarray)
    assert len(r_vals) == expected_len
    assert len(xi_vals) == expected_len
    assert np.all(np.isfinite(r_vals))
    assert np.all(np.isfinite(xi_vals))


@pytest.mark.parametrize(
    "k_pk,pk,r_eval,expected_len",
    [
        param(
            np.geomspace(1.0e-2, 10.0, 16),
            np.ones(16),
            np.geomspace(0.2, 50, 7),
            7,
            id="hankel_spherical_output_length_basic",
        ),
        param(
            np.geomspace(1.0e-2, 10.0, 8),
            np.linspace(0.1, 2.0, 8),
            np.array([0.2, 50]),
            2,
            id="hankel_spherical_output_length_two_points",
        ),
    ],
)
def test_spherical_correlation_interpolated_output_exists_and_correct_length(
    k_pk,
    pk,
    r_eval,
    expected_len,
):
    """Tests that spherical_correlation_interpolated has shape ``(len(r_eval), len(r_eval))``."""
    ht_fft = HankelTransform(backend="fftlog")
    r_vals, xi_vals = ht_fft.spherical_correlation_interpolated(
        r_eval,
        k_pk=k_pk,
        pk=pk,
        use_offset=False,
    )

    assert isinstance(r_vals, np.ndarray)
    assert isinstance(xi_vals, np.ndarray)
    assert len(r_vals) == expected_len
    assert len(xi_vals) == expected_len
    assert np.all(np.isfinite(r_vals))
    assert np.all(np.isfinite(xi_vals))


@pytest.mark.parametrize(
    "ell,c_ell,expected_len",
    [
        param(
            np.geomspace(1.0e-2, 10.0, 16),
            np.ones(16),
            16,
            id="hankel_projected_output_length_basic",
        ),
        param(
            np.geomspace(1.0e-2, 10.0, 8),
            np.geomspace(0.01, 1.0, 8),
            8,
            id="hankel_projected_output_length_two_points",
        ),
    ],
)
def test_projected_correlation_output_exists_and_correct_length(
    ell,
    c_ell,
    expected_len,
):
    """Tests that projected_correlation has shape ``(len(ell), len(ell))``."""
    ht_fft = HankelTransform(backend="fftlog")
    theta_vals, gamma_vals = ht_fft.projected_correlation(
        ell=ell,
        c_ell=c_ell,
        use_offset=False,
    )

    assert isinstance(theta_vals, np.ndarray)
    assert isinstance(gamma_vals, np.ndarray)
    assert len(theta_vals) == expected_len
    assert len(gamma_vals) == expected_len
    assert np.all(np.isfinite(theta_vals))
    assert np.all(np.isfinite(gamma_vals))


@pytest.mark.parametrize(
    "ell,c_ell,theta_eval,expected_len",
    [
        param(
            np.geomspace(1.0e-2, 10.0, 16),
            np.ones(16),
            np.geomspace(0.2, 50, 7),
            7,
            id="hankel_projected_output_length_basic",
        ),
        param(
            np.geomspace(1.0e-2, 10.0, 8),
            np.geomspace(0.01, 1.0, 8),
            np.array([0.2, 50]),
            2,
            id="hankel_projected_output_length_two_points",
        ),
    ],
)
def test_projected_correlation_interpolated_output_exists_and_correct_length(
    ell,
    c_ell,
    theta_eval,
    expected_len,
):
    """Tests that projected_correlation_interpolated has shape ``(len(theta), len(theta))``."""
    ht_fft = HankelTransform(backend="fftlog")
    theta_vals, gamma_vals = ht_fft.projected_correlation_interpolated(
        theta_eval,
        ell=ell,
        c_ell=c_ell,
        use_offset=False,
    )

    assert isinstance(theta_vals, np.ndarray)
    assert isinstance(gamma_vals, np.ndarray)
    assert len(theta_vals) == expected_len
    assert len(gamma_vals) == expected_len
    assert np.all(np.isfinite(theta_vals))
    assert np.all(np.isfinite(gamma_vals))


@pytest.mark.parametrize(
    "transform_func",
    [
        param(
            HankelTransform(backend="fftlog").spherical_correlation_interpolated,
            id="invalid_spacing_k_hankel_spherical",
        ),
        param(
            HankelTransform(backend="fftlog").projected_correlation_interpolated,
            id="invalid_spacing_ell_hankel_projected",
        ),
    ],
)
def test_hankel_invalid_spacing_raises(transform_func):
    """Tests that non-logspaced input arrays are rejected."""
    P_or_C = np.ones(4)
    non_logspaced = np.array([1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError):
        transform_func(non_logspaced, P_or_C, P_or_C, use_offset=False)


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
def test_spherical_correlation_interpolated_rejects_interpolation_outside_bounds(
    k,
    pk,
    r_eval,
):
    """Tests that spherical_correlation_interpolated rejects r outside grid."""
    with pytest.raises(ValueError):
        ht_fft = HankelTransform(backend="fftlog")
        _, xi_vals = ht_fft.spherical_correlation_interpolated(
            r=r_eval,
            k_pk=k,
            pk=pk,
            use_offset=False,
        )


@pytest.mark.parametrize(
    "ell,c_ell,theta_eval",
    [
        param(
            np.geomspace(1.0e-2, 10.0, 8),
            np.geomspace(0.01, 1.0, 8),
            np.array([0.01, 50]),
        ),
    ],
)
def test_projected_correlation_interpolated_rejects_interpolation_outside_bounds(
    ell,
    c_ell,
    theta_eval,
):
    """Tests that projected_correlation_interpolated rejects theta outside grid."""
    with pytest.raises(ValueError):
        ht_fft = HankelTransform(backend="fftlog")
        _, gamma_vals = ht_fft.projected_correlation_interpolated(
            theta=theta_eval,
            ell=ell,
            c_ell=c_ell,
            use_offset=False,
        )


def test_fftlog_projected_correlation_requires_spectrum():
    """Test that the FFTLog backend requires a projected spectrum input."""
    transform = HankelTransformFFTLog()

    with pytest.raises(ValueError, match="ell must be supplied"):
        transform.projected_correlation(order=2)


def test_fftlog_spherical_correlation_requires_spectrum():
    """Test that the FFTLog backend requires a spherical spectrum input."""
    transform = HankelTransformFFTLog()

    with pytest.raises(ValueError, match="pk must be supplied"):
        transform.spherical_correlation(order=0)


def test_hankel_projected_returns_reversed_reciprocal_grid():
    """Test the standalone projected FFTLog helper returns a reciprocal radial grid."""
    ell = np.geomspace(1.0e-2, 1.0e2, 8)
    c_ell = np.ones_like(ell)

    theta, xi = hankel_projected(ell, c_ell, order=2, use_offset=False)

    np.testing.assert_allclose(theta, 1.0 / ell[::-1])
    assert xi.shape == theta.shape


def test_hankel_spherical_rejects_nonzero_order():
    """Test that the standalone spherical helper only supports order zero."""
    k = np.geomspace(1.0e-2, 1.0e2, 8)
    pk = np.ones_like(k)

    with pytest.raises(NotImplementedError, match="Only order 0"):
        hankel_spherical(k, pk, order=2, use_offset=False)


def test_projected_covariance_returns_not_implemented():
    """Test that the FFTLog backend does not support projected_covariance."""
    transform = HankelTransformFFTLog()

    with pytest.raises(
        NotImplementedError, match="does not support projected_covariance"
    ):
        transform.projected_covariance(order=2)


def test_projected_skewness_returns_not_implemented():
    """Test that the FFTLog backend does not support projected_skewness."""
    transform = HankelTransformFFTLog()

    with pytest.raises(
        NotImplementedError, match="does not support projected_skewness"
    ):
        transform.projected_skewness(order=2)
