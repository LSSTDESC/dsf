"""Unit tests for ``dsf.covariance.ingredients.sigma_crit``."""

import numpy as np
import pytest

from dsf.covariance.ingredients.sigma_crit import (
    effective_sigma_crit_squared,
    sigma_crit_inverse_comoving,
    sigma_crit_inverse_source_average,
    sigma_crit_squared_average,
)

SIGMA_CRIT_MODULE = "dsf.covariance.ingredients.sigma_crit"


def fake_comoving_distance_h(cosmo, z, *, h=None):
    """Return a simple monotonic comoving distance for tests."""
    return 1000.0 * np.asarray(z, dtype=float)


def test_sigma_crit_inverse_comoving_masks_sources_in_front_of_lens(monkeypatch):
    """Test that only sources behind the lens contribute to SigmaCrit inverse."""
    cosmo = {"h": 0.7}
    z_lens = 0.5
    z_source = np.array([0.1, 0.5, 1.0, 2.0])
    sigma_crit_prefactor = 10.0

    monkeypatch.setattr(
        f"{SIGMA_CRIT_MODULE}.comoving_distance_h",
        fake_comoving_distance_h,
    )

    result = sigma_crit_inverse_comoving(
        cosmo,
        z_lens,
        z_source,
        sigma_crit_prefactor=sigma_crit_prefactor,
    )

    chi_lens = 1000.0 * z_lens
    chi_source = 1000.0 * z_source

    expected = np.zeros_like(z_source, dtype=float)
    behind = chi_source > chi_lens
    expected[behind] = (
        chi_lens
        * (chi_source[behind] - chi_lens)
        / chi_source[behind]
        / sigma_crit_prefactor
        * (1.0 + z_lens)
    )

    np.testing.assert_allclose(result, expected)


def test_sigma_crit_inverse_comoving_returns_zero_for_sources_at_lens(monkeypatch):
    """Test that sources exactly at the lens redshift get zero efficiency."""
    cosmo = {"h": 0.7}
    z_lens = 0.5
    z_source = np.array([0.5])

    monkeypatch.setattr(
        f"{SIGMA_CRIT_MODULE}.comoving_distance_h",
        fake_comoving_distance_h,
    )

    result = sigma_crit_inverse_comoving(
        cosmo,
        z_lens,
        z_source,
        sigma_crit_prefactor=10.0,
    )

    np.testing.assert_allclose(result, np.array([0.0]))


def test_sigma_crit_inverse_comoving_passes_supplied_h_to_distance(monkeypatch):
    """Test that supplied h is passed to comoving distance evaluations."""
    cosmo = {"h": 0.5}
    z_lens = 0.5
    z_source = np.array([1.0, 2.0])
    supplied_h = 0.8
    called_h_values = []

    def recording_comoving_distance_h(cosmo, z, *, h=None):
        called_h_values.append(h)
        return 1000.0 * np.asarray(z, dtype=float)

    monkeypatch.setattr(
        f"{SIGMA_CRIT_MODULE}.comoving_distance_h",
        recording_comoving_distance_h,
    )

    sigma_crit_inverse_comoving(
        cosmo,
        z_lens,
        z_source,
        h=supplied_h,
        sigma_crit_prefactor=10.0,
    )

    assert called_h_values == [supplied_h, supplied_h]


def test_sigma_crit_inverse_source_average_integrates_over_source_distribution(
    monkeypatch,
):
    """Test that source-averaged SigmaCrit inverse integrates over n(z_source)."""
    cosmo = {"h": 0.7}
    z_lens = 0.5
    z_source = np.array([0.0, 1.0, 2.0])
    nz_source = np.array([1.0, 2.0, 1.0])
    sigma_crit_prefactor = 10.0

    monkeypatch.setattr(
        f"{SIGMA_CRIT_MODULE}.comoving_distance_h",
        fake_comoving_distance_h,
    )

    result = sigma_crit_inverse_source_average(
        cosmo,
        z_lens=z_lens,
        z_source=z_source,
        nz_source=nz_source,
        sigma_crit_prefactor=sigma_crit_prefactor,
    )

    sigma_inv = sigma_crit_inverse_comoving(
        cosmo,
        z_lens,
        z_source,
        sigma_crit_prefactor=sigma_crit_prefactor,
    )
    expected = np.trapezoid(sigma_inv * nz_source, z_source)

    np.testing.assert_allclose(result, expected)


def test_effective_sigma_crit_squared_averages_over_lens_distribution(monkeypatch):
    """Test that effective SigmaCrit squared averages over lens n(z)."""
    cosmo = {"h": 0.7}
    z_lens = np.array([0.2, 0.5, 0.8])
    nz_lens = np.array([1.0, 2.0, 1.0])
    z_source = np.array([0.0, 1.0, 2.0])
    nz_source = np.array([1.0, 2.0, 1.0])
    sigma_crit_prefactor = 10.0

    monkeypatch.setattr(
        f"{SIGMA_CRIT_MODULE}.comoving_distance_h",
        fake_comoving_distance_h,
    )

    result = effective_sigma_crit_squared(
        cosmo,
        z_lens=z_lens,
        nz_lens=nz_lens,
        z_source=z_source,
        nz_source=nz_source,
        sigma_crit_prefactor=sigma_crit_prefactor,
    )

    sigma_inv_source_avg = np.array(
        [
            sigma_crit_inverse_source_average(
                cosmo,
                z_lens=z_value,
                z_source=z_source,
                nz_source=nz_source,
                sigma_crit_prefactor=sigma_crit_prefactor,
            )
            for z_value in z_lens
        ]
    )
    sigma_inv_avg = np.trapezoid(sigma_inv_source_avg * nz_lens, z_lens)
    expected = 1.0 / sigma_inv_avg**2

    np.testing.assert_allclose(result, expected)


def test_effective_sigma_crit_squared_uses_supplied_h_in_nested_calls(monkeypatch):
    """Test that supplied h is propagated through effective SigmaCrit calculation."""
    cosmo = {"h": 0.5}
    z_lens = np.array([0.2, 0.5])
    nz_lens = np.array([1.0, 1.0])
    z_source = np.array([1.0, 2.0])
    nz_source = np.array([1.0, 1.0])
    supplied_h = 0.8
    called_h_values = []

    def recording_comoving_distance_h(cosmo, z, *, h=None):
        called_h_values.append(h)
        return 1000.0 * np.asarray(z, dtype=float)

    monkeypatch.setattr(
        f"{SIGMA_CRIT_MODULE}.comoving_distance_h",
        recording_comoving_distance_h,
    )

    effective_sigma_crit_squared(
        cosmo,
        z_lens=z_lens,
        nz_lens=nz_lens,
        z_source=z_source,
        nz_source=nz_source,
        h=supplied_h,
        sigma_crit_prefactor=10.0,
    )

    assert called_h_values
    assert all(h_value == supplied_h for h_value in called_h_values)


def test_effective_sigma_crit_squared_raises_when_no_sources_are_behind_lenses(
    monkeypatch,
):
    """Test that effective SigmaCrit squared fails for zero lensing efficiency."""
    cosmo = {"h": 0.7}
    z_lens = np.array([1.0, 2.0])
    nz_lens = np.array([1.0, 1.0])
    z_source = np.array([0.0, 0.5])
    nz_source = np.array([1.0, 1.0])

    monkeypatch.setattr(
        f"{SIGMA_CRIT_MODULE}.comoving_distance_h",
        fake_comoving_distance_h,
    )

    with pytest.raises(
        ValueError,
        match="Average SigmaCrit inverse must be finite and positive",
    ):
        effective_sigma_crit_squared(
            cosmo,
            z_lens=z_lens,
            nz_lens=nz_lens,
            z_source=z_source,
            nz_source=nz_source,
            sigma_crit_prefactor=10.0,
        )


def test_sigma_crit_squared_average_is_alias_for_effective_sigma_crit_squared(
    monkeypatch,
):
    """Test that sigma_crit_squared_average preserves the effective function result."""
    cosmo = {"h": 0.7}
    z_lens = np.array([0.2, 0.5])
    nz_lens = np.array([1.0, 1.0])
    z_source = np.array([1.0, 2.0])
    nz_source = np.array([1.0, 1.0])

    monkeypatch.setattr(
        f"{SIGMA_CRIT_MODULE}.comoving_distance_h",
        fake_comoving_distance_h,
    )

    direct = effective_sigma_crit_squared(
        cosmo,
        z_lens=z_lens,
        nz_lens=nz_lens,
        z_source=z_source,
        nz_source=nz_source,
        sigma_crit_prefactor=10.0,
    )
    alias = sigma_crit_squared_average(
        cosmo,
        z_lens=z_lens,
        nz_lens=nz_lens,
        z_source=z_source,
        nz_source=nz_source,
        sigma_crit_prefactor=10.0,
    )

    np.testing.assert_allclose(alias, direct)


@pytest.mark.parametrize("z_lens", [-1.0, -0.1])
def test_sigma_crit_inverse_comoving_rejects_negative_lens_redshift(z_lens):
    """Test that SigmaCrit inverse rejects negative lens redshifts."""
    cosmo = {"h": 0.7}
    z_source = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="z_lens must be non-negative"):
        sigma_crit_inverse_comoving(
            cosmo,
            z_lens,
            z_source,
            sigma_crit_prefactor=10.0,
        )


@pytest.mark.parametrize("sigma_crit_prefactor", [0.0, -1.0])
def test_sigma_crit_inverse_comoving_rejects_nonpositive_prefactor(
    sigma_crit_prefactor,
):
    """Test that SigmaCrit inverse rejects non-positive prefactors."""
    cosmo = {"h": 0.7}
    z_source = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="sigma_crit_prefactor"):
        sigma_crit_inverse_comoving(
            cosmo,
            0.5,
            z_source,
            sigma_crit_prefactor=sigma_crit_prefactor,
        )


@pytest.mark.parametrize(
    "z_source",
    [
        np.array([-0.1, 1.0]),
        np.array([[0.0, 1.0]]),
    ],
)
def test_sigma_crit_inverse_comoving_rejects_invalid_source_grid(z_source):
    """Test that SigmaCrit inverse rejects invalid source redshift grids."""
    cosmo = {"h": 0.7}

    with pytest.raises(ValueError):
        sigma_crit_inverse_comoving(
            cosmo,
            0.5,
            z_source,
            sigma_crit_prefactor=10.0,
        )


@pytest.mark.parametrize(
    ("z_source", "nz_source"),
    [
        (np.array([0.0, 1.0]), np.array([1.0])),
        (np.array([[0.0, 1.0]]), np.array([1.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([[1.0, 1.0]])),
        (np.array([1.0, 0.0]), np.array([1.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([1.0, -1.0])),
    ],
)
def test_sigma_crit_inverse_source_average_rejects_invalid_source_distribution(
    z_source,
    nz_source,
):
    """Test that source-averaged SigmaCrit inverse rejects invalid source n(z)."""
    cosmo = {"h": 0.7}

    with pytest.raises(ValueError):
        sigma_crit_inverse_source_average(
            cosmo,
            z_lens=0.5,
            z_source=z_source,
            nz_source=nz_source,
            sigma_crit_prefactor=10.0,
        )


@pytest.mark.parametrize(
    ("z_lens", "nz_lens"),
    [
        (np.array([0.0, 1.0]), np.array([1.0])),
        (np.array([[0.0, 1.0]]), np.array([1.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([[1.0, 1.0]])),
        (np.array([1.0, 0.0]), np.array([1.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([1.0, -1.0])),
    ],
)
def test_effective_sigma_crit_squared_rejects_invalid_lens_distribution(
    z_lens,
    nz_lens,
):
    """Test that effective SigmaCrit squared rejects invalid lens n(z)."""
    cosmo = {"h": 0.7}
    z_source = np.array([1.0, 2.0])
    nz_source = np.array([1.0, 1.0])

    with pytest.raises(ValueError):
        effective_sigma_crit_squared(
            cosmo,
            z_lens=z_lens,
            nz_lens=nz_lens,
            z_source=z_source,
            nz_source=nz_source,
            sigma_crit_prefactor=10.0,
        )


@pytest.mark.parametrize(
    ("z_source", "nz_source"),
    [
        (np.array([0.0, 1.0]), np.array([1.0])),
        (np.array([[0.0, 1.0]]), np.array([1.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([[1.0, 1.0]])),
        (np.array([1.0, 0.0]), np.array([1.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([1.0, -1.0])),
    ],
)
def test_effective_sigma_crit_squared_rejects_invalid_source_distribution(
    z_source,
    nz_source,
):
    """Test that effective SigmaCrit squared rejects invalid source n(z)."""
    cosmo = {"h": 0.7}
    z_lens = np.array([0.2, 0.5])
    nz_lens = np.array([1.0, 1.0])

    with pytest.raises(ValueError):
        effective_sigma_crit_squared(
            cosmo,
            z_lens=z_lens,
            nz_lens=nz_lens,
            z_source=z_source,
            nz_source=nz_source,
            sigma_crit_prefactor=10.0,
        )
