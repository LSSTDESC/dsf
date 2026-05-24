"""Unit tests for ``dsf.data_vector.ccl_integration``."""

import numpy as np
import pytest

from dsf.data_vector.ccl_integration import HaloProfileGeneric


class DummyPk2D:
    """Small callable stand-in for a CCL Pk2D object."""

    def __init__(self):
        """Initialize an empty call log."""
        self.calls = []

    def __call__(self, k, a):
        """Return a deterministic profile while recording inputs."""
        k = np.asarray(k, dtype=float)
        self.calls.append((k.copy(), a))
        return a * (1.0 + k**2)


@pytest.fixture
def record_fftlog_precision(monkeypatch):
    """Patch FFTLog precision updates with a lightweight recorder."""

    def update_precision_fftlog(self, **kwargs):
        """Record FFTLog precision options on the profile instance."""
        self.recorded_fftlog_precision = kwargs

    monkeypatch.setattr(
        HaloProfileGeneric,
        "update_precision_fftlog",
        update_precision_fftlog,
    )


def test_halo_profile_generic_stores_pk2d_object(record_fftlog_precision):
    """Tests that the generic profile stores the supplied Pk2D object."""
    pk2d = DummyPk2D()

    profile = HaloProfileGeneric(pk2d=pk2d)

    assert profile.pk2d is pk2d


def test_halo_profile_generic_sets_expected_profile_flags(record_fftlog_precision):
    """Tests that the generic profile configures CCL projection flags."""
    profile = HaloProfileGeneric(pk2d=DummyPk2D())

    assert profile.fourier_analytic is True
    assert profile.projected_analytic is False
    assert profile.cumul2d_analytic is False
    assert profile._fourier.__self__ is profile
    assert profile._fourier.__func__ is profile._fourier_analytic.__func__


@pytest.mark.parametrize(
    ("padding_hi_fftlog", "padding_lo_fftlog", "n_per_decade"),
    [
        (1.0e3, 1.0e-2, 5000),
        (1.0e2, 1.0e-3, 100),
        (5.0e1, 1.0e-4, 250),
    ],
)
def test_halo_profile_generic_passes_fftlog_precision_options(
    record_fftlog_precision,
    padding_hi_fftlog,
    padding_lo_fftlog,
    n_per_decade,
):
    """Tests that FFTLog precision options are forwarded to the profile."""
    profile = HaloProfileGeneric(
        pk2d=DummyPk2D(),
        padding_hi_fftlog=padding_hi_fftlog,
        padding_lo_fftlog=padding_lo_fftlog,
        n_per_decade=n_per_decade,
    )

    assert profile.recorded_fftlog_precision == {
        "padding_hi_fftlog": padding_hi_fftlog,
        "padding_lo_fftlog": padding_lo_fftlog,
        "n_per_decade": n_per_decade,
    }


@pytest.mark.parametrize(
    ("k", "a", "expected_shape", "expected_values"),
    [
        (
            np.array([0.1, 1.0, 10.0]),
            0.5,
            (1, 3),
            np.array([[0.5 * (1.0 + 0.1**2), 0.5 * 2.0, 0.5 * 101.0]]),
        ),
        (
            2.0,
            0.25,
            (1, 1),
            np.array([[0.25 * (1.0 + 2.0**2)]]),
        ),
        (
            [1, 2, 3],
            1.0,
            (1, 3),
            np.array([[2.0, 5.0, 10.0]]),
        ),
    ],
)
def test_fourier_analytic_returns_expected_row_vector(
    record_fftlog_precision,
    k,
    a,
    expected_shape,
    expected_values,
):
    """Tests that Fourier evaluation returns the expected row-vector output."""
    profile = HaloProfileGeneric(pk2d=DummyPk2D())

    result = profile._fourier_analytic(
        cosmo=object(),
        k=k,
        M=1.0,
        a=a,
    )

    assert result.shape == expected_shape
    np.testing.assert_allclose(result, expected_values)


@pytest.mark.parametrize(
    "mass",
    [
        1.0,
        np.array([1.0e14]),
        [1.0e14],
    ],
)
def test_fourier_analytic_accepts_single_dummy_mass(
    record_fftlog_precision,
    mass,
):
    """Tests that scalar and one-element mass inputs are accepted."""
    profile = HaloProfileGeneric(pk2d=DummyPk2D())

    result = profile._fourier_analytic(
        cosmo=object(),
        k=np.array([0.5, 1.5]),
        M=mass,
        a=1.0,
    )

    assert result.shape == (1, 2)


@pytest.mark.parametrize(
    "mass",
    [
        np.array([1.0e14, 2.0e14]),
        [1.0e14, 2.0e14],
    ],
)
def test_fourier_analytic_rejects_multiple_mass_values(
    record_fftlog_precision,
    mass,
):
    """Tests that multiple mass values are rejected by the generic wrapper."""
    profile = HaloProfileGeneric(pk2d=DummyPk2D())

    with pytest.raises(ValueError, match="M must be a single value"):
        profile._fourier_analytic(
            cosmo=object(),
            k=np.array([0.5, 1.5]),
            M=mass,
            a=1.0,
        )


def test_fourier_analytic_forwards_k_and_scale_factor_to_pk2d(
    record_fftlog_precision,
):
    """Tests that k and scale factor are forwarded to the stored Pk2D object."""
    pk2d = DummyPk2D()
    profile = HaloProfileGeneric(pk2d=pk2d)

    k = np.array([0.2, 0.4])
    profile._fourier_analytic(
        cosmo=object(),
        k=k,
        M=1.0,
        a=0.7,
    )

    called_k, called_a = pk2d.calls[-1]

    np.testing.assert_allclose(called_k, k)
    assert called_k.dtype == float
    assert called_a == 0.7


def test_fourier_method_alias_calls_fourier_analytic(record_fftlog_precision):
    """Tests that the CCL Fourier hook evaluates the analytic profile method."""
    profile = HaloProfileGeneric(pk2d=DummyPk2D())

    result = profile._fourier(
        cosmo=None,
        k=np.array([1.0, 2.0]),
        M=1.0,
        a=1.0,
    )

    np.testing.assert_allclose(result, [[2.0, 5.0]])


def test_fourier_analytic_ignores_cosmology_argument(record_fftlog_precision):
    """Tests that Fourier evaluation does not require a real cosmology object."""
    profile = HaloProfileGeneric(pk2d=DummyPk2D())

    result = profile._fourier_analytic(
        cosmo=None,
        k=np.array([1.0]),
        M=1.0,
        a=1.0,
    )

    np.testing.assert_allclose(result, [[2.0]])
