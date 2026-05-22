"""Unit tests for ``src.dsf.data_vector.profiles``."""

import numpy as np

from src.dsf.data_vector import profiles
from src.dsf.data_vector.profiles import (
    density_weighted_power_spectrum,
    linear_twohalo_density_profile,
    nonlinear_twohalo_density_profile,
)


class DummyCosmology:
    """Small stand-in for a CCL cosmology object."""

    def __init__(self):
        """Initialize an empty call log."""
        self.rho_x_calls = []

    def rho_x(self, a, species, *, is_comoving):
        """Return a deterministic comoving matter density."""
        self.rho_x_calls.append((a, species, is_comoving))
        return 10.0 * a


class DummyPk2D:
    """Small stand-in for a CCL Pk2D object."""

    def __init__(
        self,
        function,
        *,
        is_logp,
        extrap_order_lok,
        extrap_order_hik,
    ):
        """Store the Pk2D construction inputs."""
        self.function = function
        self.is_logp = is_logp
        self.extrap_order_lok = extrap_order_lok
        self.extrap_order_hik = extrap_order_hik


def test_density_weighted_power_spectrum_builds_pk2d_from_function(monkeypatch):
    """Tests that density-weighted spectra are wrapped in a Pk2D object."""
    calls = []

    def fake_from_function(
        function,
        *,
        is_logp,
        extrap_order_lok,
        extrap_order_hik,
    ):
        """Record Pk2D construction arguments."""
        calls.append(
            {
                "function": function,
                "is_logp": is_logp,
                "extrap_order_lok": extrap_order_lok,
                "extrap_order_hik": extrap_order_hik,
            }
        )
        return DummyPk2D(
            function,
            is_logp=is_logp,
            extrap_order_lok=extrap_order_lok,
            extrap_order_hik=extrap_order_hik,
        )

    monkeypatch.setattr(
        profiles.ccl.Pk2D,
        "from_function",
        fake_from_function,
    )

    result = density_weighted_power_spectrum(
        cosmo=DummyCosmology(),
        power_spectrum=lambda cosmo, k, a: np.ones_like(k),
    )

    assert isinstance(result, DummyPk2D)
    assert result.function is calls[0]["function"]
    assert result.is_logp is False
    assert result.extrap_order_lok == 1
    assert result.extrap_order_hik == 2


def test_density_weighted_power_spectrum_multiplies_pk_by_comoving_density(
    monkeypatch,
):
    """Tests that the wrapped power spectrum returns rho_m times P_mm."""
    stored = {}

    def fake_from_function(
        function,
        *,
        is_logp,
        extrap_order_lok,
        extrap_order_hik,
    ):
        """Store the wrapped function and return it in a dummy object."""
        stored["function"] = function
        return DummyPk2D(
            function,
            is_logp=is_logp,
            extrap_order_lok=extrap_order_lok,
            extrap_order_hik=extrap_order_hik,
        )

    def power_spectrum(cosmo, k, a):
        """Return a deterministic matter power spectrum."""
        return np.asarray(k, dtype=float) + a

    monkeypatch.setattr(
        profiles.ccl.Pk2D,
        "from_function",
        fake_from_function,
    )

    cosmo = DummyCosmology()

    density_weighted_power_spectrum(
        cosmo=cosmo,
        power_spectrum=power_spectrum,
    )

    k = np.array([1.0, 2.0, 4.0])
    a = 0.5

    result = stored["function"](k, a)

    expected_density = 10.0 * a
    expected_pk = k + a
    expected = expected_density * expected_pk

    np.testing.assert_allclose(result, expected)
    assert cosmo.rho_x_calls == [(a, "matter", True)]


def test_density_weighted_power_spectrum_accepts_scalar_power_output(monkeypatch):
    """Tests that scalar-like power-spectrum outputs are converted to arrays."""
    stored = {}

    def fake_from_function(
        function,
        *,
        is_logp,
        extrap_order_lok,
        extrap_order_hik,
    ):
        """Store the wrapped function and return it in a dummy object."""
        stored["function"] = function
        return DummyPk2D(
            function,
            is_logp=is_logp,
            extrap_order_lok=extrap_order_lok,
            extrap_order_hik=extrap_order_hik,
        )

    monkeypatch.setattr(
        profiles.ccl.Pk2D,
        "from_function",
        fake_from_function,
    )

    cosmo = DummyCosmology()

    density_weighted_power_spectrum(
        cosmo=cosmo,
        power_spectrum=lambda cosmo, k, a: 2.0,
    )

    result = stored["function"](np.array([1.0, 2.0]), 0.5)

    np.testing.assert_allclose(result, 10.0)


def test_linear_twohalo_density_profile_uses_linear_matter_power(monkeypatch):
    """Tests that the linear profile uses CCL linear matter power."""
    calls = []

    def fake_density_weighted_power_spectrum(cosmo, power_spectrum):
        """Record the power-spectrum function passed by the wrapper."""
        calls.append((cosmo, power_spectrum))
        return "linear-pk2d"

    monkeypatch.setattr(
        profiles,
        "density_weighted_power_spectrum",
        fake_density_weighted_power_spectrum,
    )

    result = linear_twohalo_density_profile(cosmo="cosmo")

    assert result == "linear-pk2d"
    assert calls == [("cosmo", profiles.ccl.linear_matter_power)]


def test_nonlinear_twohalo_density_profile_uses_nonlinear_matter_power(monkeypatch):
    """Tests that the nonlinear profile uses CCL nonlinear matter power."""
    calls = []

    def fake_density_weighted_power_spectrum(cosmo, power_spectrum):
        """Record the power-spectrum function passed by the wrapper."""
        calls.append((cosmo, power_spectrum))
        return "nonlinear-pk2d"

    monkeypatch.setattr(
        profiles,
        "density_weighted_power_spectrum",
        fake_density_weighted_power_spectrum,
    )

    result = nonlinear_twohalo_density_profile(cosmo="cosmo")

    assert result == "nonlinear-pk2d"
    assert calls == [("cosmo", profiles.ccl.nonlin_matter_power)]


def test_density_weighted_power_spectrum_passes_cosmology_to_power_spectrum(
    monkeypatch,
):
    """Tests that the wrapped function passes the original cosmology through."""
    stored = {}
    power_calls = []

    def fake_from_function(
        function,
        *,
        is_logp,
        extrap_order_lok,
        extrap_order_hik,
    ):
        """Store the wrapped function and return it in a dummy object."""
        stored["function"] = function
        return DummyPk2D(
            function,
            is_logp=is_logp,
            extrap_order_lok=extrap_order_lok,
            extrap_order_hik=extrap_order_hik,
        )

    def power_spectrum(cosmo, k, a):
        """Record power-spectrum inputs."""
        power_calls.append((cosmo, np.asarray(k, dtype=float), a))
        return np.ones_like(k, dtype=float)

    monkeypatch.setattr(
        profiles.ccl.Pk2D,
        "from_function",
        fake_from_function,
    )

    cosmo = DummyCosmology()

    density_weighted_power_spectrum(
        cosmo=cosmo,
        power_spectrum=power_spectrum,
    )

    k = np.array([0.1, 1.0])
    a = 0.8
    stored["function"](k, a)

    assert power_calls[0][0] is cosmo
    np.testing.assert_allclose(power_calls[0][1], k)
    assert power_calls[0][2] == a
