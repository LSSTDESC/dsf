"""Unit tests for ``dsf.data_vector.ccl_integration``."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import jv

from dsf.data_vector.ccl_integration import DeltaSigma


class DummyPk2D:
    """Small callable stand-in for a CCL Pk2D object."""

    def __init__(self):
        """Initialize an empty call log."""
        self.calls = []

    def __call__(self, k, a):
        """Return a Gaussian spectrum while recording inputs."""
        k = np.asarray(k, dtype=float)
        self.calls.append((k.copy(), a))
        return a * np.exp(-k**2)


def test_delta_sigma_stores_pk2d_object():
    """Tests that the projection stores the supplied Pk2D object."""
    pk2d = DummyPk2D()

    projection = DeltaSigma(pk2d=pk2d)

    assert projection.pk2d is pk2d


def test_delta_sigma_uses_fftlog_backend():
    """Tests that the projection uses the FFTLog Hankel backend."""
    projection = DeltaSigma(pk2d=DummyPk2D())

    assert projection.hankel.backend_name == "fftlog"


@pytest.mark.parametrize(
    ("k_min", "k_max", "n_k"),
    [
        (1.0e-3, 1.0e3, 100),
        (1.0e-4, 1.0e4, 250),
        (1.0e-6, 1.0e6, 5000),
    ],
)
def test_delta_sigma_configures_wavenumber_grid(k_min, k_max, n_k):
    """Tests that the requested logarithmic wavenumber grid is constructed."""
    projection = DeltaSigma(
        pk2d=DummyPk2D(),
        k_min=k_min,
        k_max=k_max,
        n_k=n_k,
    )

    assert projection.k.shape == (n_k,)
    np.testing.assert_allclose(projection.k[0], k_min)
    np.testing.assert_allclose(projection.k[-1], k_max)
    np.testing.assert_allclose(
        np.diff(np.log(projection.k)),
        np.log(k_max / k_min) / (n_k - 1),
    )


@pytest.mark.parametrize(
    ("k_min", "k_max"),
    [
        (0.0, 10.0),
        (-1.0, 10.0),
        (10.0, 1.0),
        (1.0, 1.0),
        (np.nan, 10.0),
        (1.0, np.inf),
    ],
)
def test_delta_sigma_rejects_invalid_wavenumber_bounds(k_min, k_max):
    """Tests that invalid wavenumber bounds raise a ValueError."""
    with pytest.raises(ValueError):
        DeltaSigma(pk2d=DummyPk2D(), k_min=k_min, k_max=k_max)


@pytest.mark.parametrize("n_k", [0, 1, -1])
def test_delta_sigma_rejects_invalid_grid_size(n_k):
    """Tests that the wavenumber grid requires at least two samples."""
    with pytest.raises(ValueError, match="n_k must be at least 2"):
        DeltaSigma(pk2d=DummyPk2D(), n_k=n_k)


def test_delta_sigma_forwards_wavenumbers_and_scale_factor():
    """Tests that the stored Pk2D receives the configured grid and scale factor."""
    pk2d = DummyPk2D()
    projection = DeltaSigma(pk2d=pk2d, k_min=1.0e-3, k_max=1.0e3, n_k=256)

    projection.delta_sigma(cosmo=None, r=np.array([0.5, 1.0]), a=0.7)

    called_k, called_a = pk2d.calls[-1]

    np.testing.assert_array_equal(called_k, projection.k)
    assert called_k.dtype == float
    assert called_a == 0.7


def test_delta_sigma_returns_expected_shape():
    """Tests that the projected profile matches the requested radial grid."""
    projection = DeltaSigma(pk2d=DummyPk2D(), n_k=1024)
    r = np.geomspace(0.1, 10.0, 20)

    result = projection.delta_sigma(cosmo=None, r=r, a=0.8)

    assert result.shape == r.shape
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("a", [0.25, 0.5, 1.0])
def test_delta_sigma_scales_with_pk2d_amplitude(a):
    """Tests that the projected profile scales linearly with the input spectrum."""
    projection = DeltaSigma(pk2d=DummyPk2D(), n_k=1024)
    r = np.geomspace(0.1, 10.0, 20)

    reference = projection.delta_sigma(cosmo=None, r=r, a=1.0)
    result = projection.delta_sigma(cosmo=None, r=r, a=a)

    np.testing.assert_allclose(result, a * reference, rtol=1.0e-10, atol=1.0e-12)


@pytest.mark.parametrize(
    "r",
    [
        np.array([0.0, 1.0]),
        np.array([-1.0, 1.0]),
        np.array([np.nan, 1.0]),
        np.array([np.inf, 1.0]),
    ],
)
def test_delta_sigma_rejects_invalid_radii(r):
    """Tests that projected radii must be positive and finite."""
    projection = DeltaSigma(pk2d=DummyPk2D())

    with pytest.raises(ValueError):
        projection.delta_sigma(cosmo=None, r=r, a=1.0)


@pytest.mark.parametrize("a", [-0.5, 0.0, 1.5, np.nan])
def test_delta_sigma_rejects_invalid_scale_factor(a):
    """Tests that the scale factor must lie in the valid range."""
    projection = DeltaSigma(pk2d=DummyPk2D())

    with pytest.raises(ValueError):
        projection.delta_sigma(cosmo=None, r=np.array([1.0, 2.0]), a=a)


def test_delta_sigma_matches_direct_hankel_integration():
    r"""Tests the FFTLog projection against direct numerical integration.

    For a Gaussian Fourier-space profile,

    .. math::

        P(k) = \exp(-k^2),

    the excess surface density is

    .. math::

        \Delta\Sigma(R) =
        \int_0^\infty \frac{k\,\mathrm{d}k}{2\pi}
        P(k) J_2(kR).

    The reference is evaluated independently using adaptive quadrature.
    """
    projection = DeltaSigma(
        pk2d=DummyPk2D(),
        k_min=1.0e-4,
        k_max=1.0e3,
        n_k=4096,
    )

    r = np.geomspace(0.5, 5.0, 10)
    result = projection.delta_sigma(cosmo=None, r=r, a=1.0)

    expected = np.array([
        quad(
            lambda k, radius=radius: k * np.exp(-k**2) * jv(2, k * radius) / (2.0 * np.pi),
            0.0,
            20.0,
            args=(),
            epsabs=1.0e-11,
            epsrel=1.0e-11,
        )[0]
        for radius in r
    ])

    np.testing.assert_allclose(result, expected, rtol=1.0e-2, atol=1.0e-6)


def test_delta_sigma_matches_analytic_gaussian_transform():
    r"""Tests the FFTLog projection against the analytic Gaussian transform.

    For a Gaussian Fourier-space profile,

    .. math::

        P(k) = \exp(-\alpha k^2),

    the second-order Hankel transform is

    .. math::

        \Delta\Sigma(R) =
        \frac{1}{\pi R^2}
        \left[
            1 -
            \left(1 + \frac{R^2}{4\alpha}\right)
            \exp\left(-\frac{R^2}{4\alpha}\right)
        \right].
    """
    alpha = 1.0

    def gaussian_pk(k, a):
        """Evaluate the Gaussian Fourier-space profile."""
        return a * np.exp(-alpha * np.asarray(k, dtype=float) ** 2)

    projection = DeltaSigma(
        pk2d=gaussian_pk,
        k_min=1.0e-4,
        k_max=1.0e3,
        n_k=4096,
    )

    r = np.geomspace(0.5, 5.0, 20)
    result = projection.delta_sigma(cosmo=None, r=r, a=1.0)

    x = r**2 / (4.0 * alpha)
    expected = (1.0 - (1.0 + x) * np.exp(-x)) / (np.pi * r**2)

    np.testing.assert_allclose(result, expected, rtol=1.0e-2, atol=1.0e-6)


def test_delta_sigma_is_linear_in_input_spectrum():
    """Tests that the Hankel projection is linear under spectrum addition."""

    def first_pk(k, a):
        """Return the first Gaussian spectrum."""
        return a * np.exp(-np.asarray(k, dtype=float) ** 2)

    def second_pk(k, a):
        """Return the second Gaussian spectrum."""
        return a * np.exp(-0.5 * np.asarray(k, dtype=float) ** 2)

    def combined_pk(k, a):
        """Return the sum of both spectra."""
        return first_pk(k, a) + second_pk(k, a)

    kwargs = {"k_min": 1.0e-4, "k_max": 1.0e3, "n_k": 2048}

    first = DeltaSigma(pk2d=first_pk, **kwargs)
    second = DeltaSigma(pk2d=second_pk, **kwargs)
    combined = DeltaSigma(pk2d=combined_pk, **kwargs)

    r = np.geomspace(0.5, 5.0, 20)

    result_first = first.delta_sigma(cosmo=None, r=r, a=1.0)
    result_second = second.delta_sigma(cosmo=None, r=r, a=1.0)
    result_combined = combined.delta_sigma(cosmo=None, r=r, a=1.0)

    np.testing.assert_allclose(
        result_combined,
        result_first + result_second,
        rtol=1.0e-8,
        atol=1.0e-10,
    )


def test_delta_sigma_preserves_radius_order():
    """Tests that the projection preserves the ordering of requested radii."""
    projection = DeltaSigma(
        pk2d=DummyPk2D(),
        k_min=1.0e-4,
        k_max=1.0e3,
        n_k=2048,
    )

    r_sorted = np.array([0.5, 1.0, 2.0, 5.0])
    r_unsorted = np.array([5.0, 0.5, 2.0, 1.0])

    sorted_result = projection.delta_sigma(cosmo=None, r=r_sorted, a=1.0)
    unsorted_result = projection.delta_sigma(cosmo=None, r=r_unsorted, a=1.0)

    expected = sorted_result[[3, 0, 2, 1]]

    np.testing.assert_allclose(unsorted_result, expected, rtol=1.0e-10)


def test_delta_sigma_is_independent_of_cosmology_argument():
    """Tests that the projection depends only on the supplied Pk2D object."""
    projection = DeltaSigma(pk2d=DummyPk2D(), n_k=1024)
    r = np.geomspace(0.5, 5.0, 10)

    result_none = projection.delta_sigma(cosmo=None, r=r, a=0.7)
    result_object = projection.delta_sigma(cosmo=object(), r=r, a=0.7)

    np.testing.assert_allclose(result_none, result_object, rtol=1.0e-12)


def test_delta_sigma_converges_with_wavenumber_resolution():
    """Tests numerical stability as the FFTLog wavenumber grid is refined."""
    r = np.geomspace(0.5, 5.0, 20)

    coarse = DeltaSigma(
        pk2d=DummyPk2D(),
        k_min=1.0e-4,
        k_max=1.0e3,
        n_k=2048,
    )

    fine = DeltaSigma(
        pk2d=DummyPk2D(),
        k_min=1.0e-4,
        k_max=1.0e3,
        n_k=4096,
    )

    coarse_result = coarse.delta_sigma(cosmo=None, r=r, a=1.0)
    fine_result = fine.delta_sigma(cosmo=None, r=r, a=1.0)

    np.testing.assert_allclose(
        coarse_result,
        fine_result,
        rtol=1.0e-2,
        atol=1.0e-6,
    )
