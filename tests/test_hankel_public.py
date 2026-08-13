"""Unit tests for the public HankelTransform façade."""

import numpy as np
import pytest

from dsf.hankel.hankel import HankelTransform


def test_hankel_transform_accepts_matrix_alias(monkeypatch):
    """Test that the public factory accepts the matrix alias."""
    captured = {}

    class DummyBackend:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setitem(
        HankelTransform.__init__.__globals__["_BACKENDS"], "matrix_zeros", DummyBackend
    )

    transform = HankelTransform(backend="matrix_zeros", foo=1)

    assert isinstance(transform.backend, DummyBackend)
    assert captured["kwargs"] == {"foo": 1}


def test_public_projected_covariance_delegates_to_backend():
    """Test that the public covariance interface delegates to the backend."""
    transform = HankelTransform.__new__(HankelTransform)

    expected_r = np.array([1.0, 2.0])
    expected_cov = np.array([[1.0, 0.5], [0.5, 1.0]])

    class DummyBackend:
        def projected_covariance(self, **kwargs):
            assert kwargs["order"] == 2
            return expected_r, expected_cov

        def bin_radial_matrix(self, **kwargs):
            return kwargs["r"], kwargs["matrix"]

    transform.backend = DummyBackend()

    r, cov = transform.projected_covariance(
        k_pk=np.array([1.0, 2.0]),
        pk1=np.array([1.0, 2.0]),
        pk2=np.array([2.0, 3.0]),
        order=2,
    )

    np.testing.assert_allclose(r, expected_r)
    np.testing.assert_allclose(cov, expected_cov)


def test_public_bin_radial_matrix_delegates_to_backend():
    """Test that the public binning interface delegates to the backend."""
    transform = HankelTransform.__new__(HankelTransform)

    class DummyBackend:
        def projected_covariance(self, **kwargs):
            raise AssertionError("Not expected")

        def bin_radial_matrix(self, **kwargs):
            return kwargs["r"], kwargs["matrix"]

    transform.backend = DummyBackend()

    r = np.array([1.0, 2.0, 3.0])
    matrix = np.arange(9.0).reshape(3, 3)
    r_bins = np.array([1.0, 2.0, 4.0])

    out_r, out_matrix = transform.bin_radial_matrix(r, matrix, r_bins)

    np.testing.assert_allclose(out_r, r)
    np.testing.assert_allclose(out_matrix, matrix)


def test_unsupported_public_covariance_methods_raise_not_implemented():
    """Test that unsupported backends fail with a clear exception."""
    transform = HankelTransform(backend="fftlog")

    with pytest.raises(NotImplementedError, match="projected_covariance"):
        transform.projected_covariance(
            k_pk=np.array([1.0, 2.0]),
            pk1=np.array([1.0, 2.0]),
            pk2=np.array([1.0, 2.0]),
        )

    with pytest.raises(NotImplementedError, match="bin_radial_matrix"):
        transform.bin_radial_matrix(
            np.array([1.0, 2.0]),
            np.ones((2, 2)),
            np.array([1.0, 2.0, 3.0]),
        )
