"""Unit tests for ``dsf.modelling.nla``."""

import numpy as np
import pyccl as ccl
import pytest

import dsf.modelling.nla as nla


@pytest.fixture
def cosmo():
    """Construct a CCL cosmology for NLA calculations."""
    return ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        transfer_function="eisenstein_hu",
        matter_power_spectrum="halofit",
    )


@pytest.fixture
def grids():
    """Construct wavenumber and scale-factor grids."""
    return {
        "k_array": np.geomspace(1.0e-3, 5.0, 30),
        "a_array": np.linspace(0.5, 1.0, 15),
    }


@pytest.fixture
def capture_density_weighted(monkeypatch):
    """Replace the density-weighted calculation with an argument recorder."""
    calls = []

    def fake_density_weighted(cosmo, power):
        """Record the supplied cosmology and power-spectrum callback."""
        calls.append((cosmo, power))
        return ("density_weighted", cosmo, power)

    monkeypatch.setattr(nla, "density_weighted_power_spectrum", fake_density_weighted)

    return calls


def test_pk2d_nla_returns_expected_keys(cosmo, grids, capture_density_weighted):
    """Tests that NLA returns the expected power-spectrum keys."""
    result = nla.pk2d_nla(cosmo, **grids, A_IA=1.0)

    assert set(result) == {"pk_deltasigma", "pk_gi"}


def test_pk2d_nla_returns_pk2d(cosmo, grids, capture_density_weighted):
    """Tests that the galaxy-intrinsic spectrum is a CCL Pk2D."""
    result = nla.pk2d_nla(cosmo, **grids, A_IA=1.0)

    assert isinstance(result["pk_gi"], ccl.Pk2D)


def test_pk2d_nla_matches_analytic_model(
    cosmo,
    grids,
    capture_density_weighted,
):
    """Tests numerical agreement with the nonlinear alignment model."""
    A_IA = 1.5
    C1rhocrit = 0.0134
    b_g = 1.8

    result = nla.pk2d_nla(cosmo, **grids, A_IA=A_IA, C1rhocrit=C1rhocrit, b_g=b_g)

    k_array = grids["k_array"]
    a_array = grids["a_array"]

    growth = cosmo.growth_factor(a_array)
    pk_mm = cosmo.get_nonlin_power()(k_array, a_array, cosmo=cosmo)

    amplitude = A_IA * C1rhocrit * cosmo["Omega_m"] / growth
    expected = -b_g * amplitude[:, None] * pk_mm

    actual = result["pk_gi"](k_array, a_array, cosmo=cosmo)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-8)


@pytest.mark.parametrize("A_IA", [0.0, 0.5, 1.0, 2.0, -1.0])
def test_pk2d_nla_amplitude_scaling(
    cosmo,
    grids,
    capture_density_weighted,
    A_IA,
):
    """Tests linear scaling with intrinsic alignment amplitude."""
    reference = nla.pk2d_nla(cosmo, **grids, A_IA=1.0)
    result = nla.pk2d_nla(cosmo, **grids, A_IA=A_IA)

    k = grids["k_array"]
    a = grids["a_array"]

    expected = A_IA * reference["pk_gi"](k, a, cosmo=cosmo)
    actual = result["pk_gi"](k, a, cosmo=cosmo)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-8, atol=1.0e-12)


@pytest.mark.parametrize("b_g", [0.0, 0.5, 1.0, 2.0, -1.0])
def test_pk2d_nla_galaxy_bias_scaling(
    cosmo,
    grids,
    capture_density_weighted,
    b_g,
):
    """Tests linear scaling with galaxy bias."""
    reference = nla.pk2d_nla(cosmo, **grids, A_IA=1.0, b_g=1.0)
    result = nla.pk2d_nla(cosmo, **grids, A_IA=1.0, b_g=b_g)

    k = grids["k_array"]
    a = grids["a_array"]

    expected = b_g * reference["pk_gi"](k, a, cosmo=cosmo)
    actual = result["pk_gi"](k, a, cosmo=cosmo)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-8, atol=1.0e-12)


@pytest.mark.parametrize("C1rhocrit", [0.0, 0.005, 0.0134, 0.02, -0.01])
def test_pk2d_nla_normalization_scaling(
    cosmo,
    grids,
    capture_density_weighted,
    C1rhocrit,
):
    """Tests linear scaling with the NLA normalization."""
    reference = nla.pk2d_nla(cosmo, **grids, A_IA=1.0, C1rhocrit=0.0134)
    result = nla.pk2d_nla(cosmo, **grids, A_IA=1.0, C1rhocrit=C1rhocrit)

    k = grids["k_array"]
    a = grids["a_array"]

    expected = (C1rhocrit / 0.0134) * reference["pk_gi"](k, a, cosmo=cosmo)

    actual = result["pk_gi"](k, a, cosmo=cosmo)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-8, atol=1.0e-12)


def test_pk2d_nla_negative_sign(cosmo, grids, capture_density_weighted):
    """Tests that positive IA amplitude produces negative GI power."""
    result = nla.pk2d_nla(cosmo, **grids, A_IA=1.0)

    pk_gi = result["pk_gi"](grids["k_array"], grids["a_array"], cosmo=cosmo)

    assert np.all(pk_gi < 0.0)


def test_pk2d_nla_negative_amplitude(cosmo, grids, capture_density_weighted):
    """Tests that negative IA amplitude produces positive GI power."""
    result = nla.pk2d_nla(cosmo, **grids, A_IA=-1.0)

    pk_gi = result["pk_gi"](grids["k_array"], grids["a_array"], cosmo=cosmo)

    assert np.all(pk_gi > 0.0)


def test_pk2d_nla_zero_amplitude(cosmo, grids, capture_density_weighted):
    """Tests that zero IA amplitude produces zero GI power."""
    result = nla.pk2d_nla(cosmo, **grids, A_IA=0.0)

    pk_gi = result["pk_gi"](grids["k_array"], grids["a_array"], cosmo=cosmo)

    np.testing.assert_allclose(pk_gi, 0.0, atol=1.0e-12)


def test_pk2d_nla_default_normalization(
    cosmo,
    grids,
    capture_density_weighted,
):
    """Tests that the default NLA normalization is 0.0134."""
    default = nla.pk2d_nla(cosmo, **grids, A_IA=1.0)
    explicit = nla.pk2d_nla(cosmo, **grids, A_IA=1.0, C1rhocrit=0.0134)

    k = grids["k_array"]
    a = grids["a_array"]

    np.testing.assert_allclose(
        default["pk_gi"](k, a, cosmo=cosmo),
        explicit["pk_gi"](k, a, cosmo=cosmo),
    )


def test_pk2d_nla_default_galaxy_bias(
    cosmo,
    grids,
    capture_density_weighted,
):
    """Tests that the default galaxy bias is unity."""
    default = nla.pk2d_nla(cosmo, **grids, A_IA=1.0)
    explicit = nla.pk2d_nla(cosmo, **grids, A_IA=1.0, b_g=1.0)

    k = grids["k_array"]
    a = grids["a_array"]

    np.testing.assert_allclose(
        default["pk_gi"](k, a, cosmo=cosmo),
        explicit["pk_gi"](k, a, cosmo=cosmo),
    )


def test_pk2d_nla_density_weighted_called(
    cosmo,
    grids,
    capture_density_weighted,
):
    """Tests that the density-weighted spectrum is constructed."""
    nla.pk2d_nla(cosmo, **grids, A_IA=1.0)

    assert len(capture_density_weighted) == 1
    assert capture_density_weighted[0][0] is cosmo


def test_pk2d_nla_density_weighted_callback(
    cosmo,
    grids,
    capture_density_weighted,
):
    """Tests that the callback evaluates the galaxy-intrinsic spectrum."""
    result = nla.pk2d_nla(cosmo, **grids, A_IA=1.0)

    callback = capture_density_weighted[0][1]

    k = grids["k_array"]
    a = 0.8

    expected = result["pk_gi"](k, a, cosmo=cosmo)
    actual = callback(cosmo, k, a)

    np.testing.assert_allclose(actual, expected)


def test_pk2d_nla_density_weighted_return_value(
    cosmo,
    grids,
    capture_density_weighted,
):
    """Tests that the density-weighted result is returned unchanged."""
    result = nla.pk2d_nla(cosmo, **grids, A_IA=1.0)

    assert result["pk_deltasigma"][0] == "density_weighted"
    assert result["pk_deltasigma"][1] is cosmo


@pytest.mark.parametrize("A_IA", [np.nan, np.inf, -np.inf])
def test_pk2d_nla_rejects_invalid_amplitude(
    cosmo,
    grids,
    capture_density_weighted,
    A_IA,
):
    """Tests that nonfinite IA amplitudes are rejected."""
    with pytest.raises(ValueError):
        nla.pk2d_nla(cosmo, **grids, A_IA=A_IA)


@pytest.mark.parametrize("C1rhocrit", [np.nan, np.inf, -np.inf])
def test_pk2d_nla_rejects_invalid_normalization(
    cosmo,
    grids,
    capture_density_weighted,
    C1rhocrit,
):
    """Tests that nonfinite NLA normalizations are rejected."""
    with pytest.raises(ValueError):
        nla.pk2d_nla(cosmo, **grids, A_IA=1.0, C1rhocrit=C1rhocrit)


@pytest.mark.parametrize("b_g", [np.nan, np.inf, -np.inf])
def test_pk2d_nla_rejects_invalid_galaxy_bias(
    cosmo,
    grids,
    capture_density_weighted,
    b_g,
):
    """Tests that nonfinite galaxy biases are rejected."""
    with pytest.raises(ValueError):
        nla.pk2d_nla(cosmo, **grids, A_IA=1.0, b_g=b_g)


@pytest.mark.parametrize(
    "k_array",
    [
        np.array([0.0, 0.1, 1.0]),
        np.array([-1.0, 0.1, 1.0]),
        np.array([0.1, np.nan, 1.0]),
        np.array([0.1, np.inf, 1.0]),
        np.array([1.0, 0.1, 10.0]),
    ],
)
def test_pk2d_nla_rejects_invalid_wavenumbers(
    cosmo,
    grids,
    capture_density_weighted,
    k_array,
):
    """Tests that invalid wavenumber grids are rejected."""
    with pytest.raises((ValueError, TypeError)):
        nla.pk2d_nla(
            cosmo,
            k_array=k_array,
            a_array=grids["a_array"],
            A_IA=1.0,
        )


@pytest.mark.parametrize(
    "a_array",
    [
        np.array([0.0, 0.5, 1.0]),
        np.array([-0.1, 0.5, 1.0]),
        np.array([0.5, np.nan, 1.0]),
        np.array([0.5, np.inf, 1.0]),
        np.array([0.9, 0.7, 1.0]),
        np.array([0.5, 0.8, 1.1]),
    ],
)
def test_pk2d_nla_rejects_invalid_scale_factors(
    cosmo,
    grids,
    capture_density_weighted,
    a_array,
):
    """Tests that invalid scale-factor grids are rejected."""
    with pytest.raises((ValueError, TypeError)):
        nla.pk2d_nla(
            cosmo,
            k_array=grids["k_array"],
            a_array=a_array,
            A_IA=1.0,
        )
