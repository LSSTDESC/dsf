"""Unit tests for ``dsf.modelling``."""

from types import SimpleNamespace

import numpy as np
import pytest

import dsf.modelling as modelling


class DummyCosmology(dict):
    """Small mapping-style cosmology stand-in."""

    def growth_factor(self, a_array):
        """Return deterministic positive growth factors."""
        return np.asarray(a_array, dtype=float) + 1.0

    def nonlin_matter_power(self, k_array, a_array):
        """Return deterministic nonlinear matter power values."""
        k_arr = np.asarray(k_array, dtype=float)
        a_arr = np.asarray(a_array, dtype=float)
        return a_arr[:, None] + k_arr[None, :]

    def get_nonlin_power(self):
        """Return a dummy nonlinear power object."""
        return object()


class DummyPk2D:
    """Small Pk2D stand-in supporting calls and addition."""

    instances = []

    def __init__(self, *, a_arr, lk_arr, pk_arr, is_logp):
        """Store Pk2D initialization arrays."""
        self.a_arr = np.asarray(a_arr, dtype=float)
        self.lk_arr = np.asarray(lk_arr, dtype=float)
        self.pk_arr = np.asarray(pk_arr, dtype=float)
        self.is_logp = is_logp
        self.__class__.instances.append(self)

    def __call__(self, k, a, cosmo=None):
        """Return a deterministic callable power value."""
        return np.asarray(k, dtype=float) + np.asarray(a, dtype=float)

    def __add__(self, other):
        """Return a tagged combined Pk2D object."""
        return SimpleNamespace(left=self, right=other, combined=True)


class DummyConcentrationDuffy08:
    """Small Duffy concentration stand-in."""

    instances = []

    def __init__(self, *, mass_def=None, fc_bar=None):
        """Store concentration initialization arguments."""
        self.mass_def = mass_def
        self.fc_bar = fc_bar
        self.__class__.instances.append(self)


def patch_density_weighting(monkeypatch):
    """Patch density weighting to expose the supplied power callable."""
    calls = []

    def fake_density_weighted_power_spectrum(cosmo, power_spectrum):
        calls.append((cosmo, power_spectrum))
        return SimpleNamespace(cosmo=cosmo, power_spectrum=power_spectrum)

    monkeypatch.setattr(
        modelling,
        "density_weighted_power_spectrum",
        fake_density_weighted_power_spectrum,
    )
    return calls


def test_make_ccl_cosmology_uses_vanilla_lcdm_without_mapping(monkeypatch):
    """Tests that make_ccl_cosmology uses VanillaLCDM when no mapping is supplied."""
    calls = []

    def fake_vanilla(**kwargs):
        """Return a tagged vanilla cosmology."""
        calls.append(kwargs)
        return {"kind": "vanilla", **kwargs}

    monkeypatch.setattr(modelling.ccl, "CosmologyVanillaLCDM", fake_vanilla)

    cosmo = modelling.make_ccl_cosmology(extra="value")

    assert cosmo == {"kind": "vanilla", "extra": "value"}
    assert calls == [{"extra": "value"}]


def test_make_ccl_cosmology_converts_omega_m_to_omega_c(monkeypatch):
    """Tests that make_ccl_cosmology converts Omega_m into Omega_c."""
    calls = []

    def fake_cosmology(**kwargs):
        """Return the received cosmology keyword arguments."""
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(modelling.ccl, "Cosmology", fake_cosmology)

    cosmo = modelling.make_ccl_cosmology(
        {
            "Omega_m": 0.31,
            "Omega_b": 0.05,
            "h": 0.7,
            "sigma8": 0.8,
            "n_s": 0.96,
        },
        transfer_function="bbks",
    )

    assert cosmo["Omega_c"] == pytest.approx(0.26)
    assert "Omega_m" not in cosmo
    assert cosmo["Omega_b"] == pytest.approx(0.05)
    assert cosmo["transfer_function"] == "bbks"
    assert calls == [cosmo]


def test_validate_pk2d_grids_accepts_valid_grids():
    """Tests that _validate_pk2d_grids accepts positive increasing grids."""
    k_array = np.array([0.1, 1.0, 10.0], dtype=float)
    a_array = np.array([0.4, 0.7, 1.0], dtype=float)

    k_result, a_result = modelling._validate_pk2d_grids(k_array, a_array)

    np.testing.assert_allclose(k_result, k_array)
    np.testing.assert_allclose(a_result, a_array)


@pytest.mark.parametrize(
    ("k_array", "a_array", "message"),
    [
        (np.array([0.1, 0.1]), np.array([0.5, 1.0]), "k_array"),
        (np.array([0.1, 1.0]), np.array([0.5, 0.5]), "a_array"),
        (np.array([-0.1, 1.0]), np.array([0.5, 1.0]), "k_array"),
        (np.array([0.1, 1.0]), np.array([0.0, 1.0]), "a_array"),
        (np.array([0.1, 1.0]), np.array([0.5, 1.2]), "0 < a <= 1"),
    ],
)
def test_validate_pk2d_grids_rejects_invalid_grids(k_array, a_array, message):
    """Tests that _validate_pk2d_grids rejects invalid k and scale-factor grids."""
    with pytest.raises(ValueError, match=message):
        modelling._validate_pk2d_grids(k_array, a_array)


def test_baryonified_duffy_concentration_passes_fc_bar(monkeypatch):
    """Tests that baryonified_duffy_concentration passes fc_bar to CCL."""
    DummyConcentrationDuffy08.instances = []

    monkeypatch.setattr(
        modelling.ccl.halos,
        "ConcentrationDuffy08",
        DummyConcentrationDuffy08,
    )

    concentration = modelling.baryonified_duffy_concentration(1.5)

    assert isinstance(concentration, DummyConcentrationDuffy08)
    assert concentration.mass_def is modelling.MASS_DEF
    assert concentration.fc_bar == pytest.approx(1.5)


def test_baryonified_duffy_concentration_rejects_non_positive_fc():
    """Tests that baryonified_duffy_concentration rejects non-positive rescaling."""
    with pytest.raises(ValueError, match="f_c"):
        modelling.baryonified_duffy_concentration(0.0)


def test_pk2d_nla_constructs_expected_negative_gi_power(monkeypatch):
    """Tests that pk2d_nla constructs the expected negative galaxy-intrinsic power."""
    DummyPk2D.instances = []
    density_calls = patch_density_weighting(monkeypatch)
    monkeypatch.setattr(modelling.ccl, "Pk2D", DummyPk2D)

    cosmo = DummyCosmology(Omega_m=0.3)
    k_array = np.array([0.1, 1.0], dtype=float)
    a_array = np.array([0.5, 1.0], dtype=float)

    result = modelling.pk2d_nla(
        cosmo,
        k_array=k_array,
        a_array=a_array,
        A_IA=2.0,
        C1rhocrit=0.5,
        b_g=3.0,
    )

    pk2d = DummyPk2D.instances[0]
    growth = cosmo.growth_factor(a_array)
    pk_mm = cosmo.nonlin_matter_power(k_array, a_array)
    expected = -(2.0 * 0.5 * 0.3 / growth)[:, None] * 3.0 * pk_mm

    np.testing.assert_allclose(pk2d.a_arr, a_array)
    np.testing.assert_allclose(pk2d.lk_arr, np.log(k_array))
    np.testing.assert_allclose(pk2d.pk_arr, expected)
    assert pk2d.is_logp is False

    assert result.cosmo is cosmo
    assert density_calls[0][0] is cosmo


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"A_IA": np.nan, "C1rhocrit": 0.0134, "b_g": 1.0}, "A_IA"),
        ({"A_IA": 1.0, "C1rhocrit": np.inf, "b_g": 1.0}, "C1rhocrit"),
        ({"A_IA": 1.0, "C1rhocrit": 0.0134, "b_g": np.nan}, "b_g"),
    ],
)
def test_pk2d_nla_rejects_non_finite_scalars(monkeypatch, kwargs, message):
    """Tests that pk2d_nla rejects non-finite scalar inputs."""
    monkeypatch.setattr(modelling.ccl, "Pk2D", DummyPk2D)
    cosmo = DummyCosmology(Omega_m=0.3)

    with pytest.raises(ValueError, match=message):
        modelling.pk2d_nla(
            cosmo,
            k_array=np.array([0.1, 1.0], dtype=float),
            a_array=np.array([0.5, 1.0], dtype=float),
            **kwargs,
        )


def test_pk2d_nla_rejects_non_positive_growth(monkeypatch):
    """Tests that pk2d_nla rejects non-positive growth factors."""

    class BadGrowthCosmology(DummyCosmology):
        """Cosmology stand-in with invalid growth factors."""

        def growth_factor(self, a_array):
            """Return a non-positive growth factor."""
            return np.array([1.0, 0.0], dtype=float)

    monkeypatch.setattr(modelling.ccl, "Pk2D", DummyPk2D)

    with pytest.raises(ValueError, match="growth_factor"):
        modelling.pk2d_nla(
            BadGrowthCosmology(Omega_m=0.3),
            k_array=np.array([0.1, 1.0], dtype=float),
            a_array=np.array([0.5, 1.0], dtype=float),
            A_IA=1.0,
        )


def test_pk2d_nla_rejects_non_finite_matter_power(monkeypatch):
    """Tests that pk2d_nla rejects non-finite nonlinear matter power."""

    class BadPowerCosmology(DummyCosmology):
        """Cosmology stand-in with invalid nonlinear matter power."""

        def nonlin_matter_power(self, k_array, a_array):
            """Return nonlinear matter power containing NaN."""
            return np.array(
                [
                    [1.0, np.nan],
                    [2.0, 3.0],
                ],
                dtype=float,
            )

    monkeypatch.setattr(modelling.ccl, "Pk2D", DummyPk2D)

    with pytest.raises(ValueError, match="nonlinear matter power"):
        modelling.pk2d_nla(
            BadPowerCosmology(Omega_m=0.3),
            k_array=np.array([0.1, 1.0], dtype=float),
            a_array=np.array([0.5, 1.0], dtype=float),
            A_IA=1.0,
        )


def test_pk2d_hod_with_nla_returns_hod_only_when_amplitude_is_zero(monkeypatch):
    """Tests that pk2d_hod_with_nla returns only HOD power when A_IA is zero."""
    pk_hod = SimpleNamespace(name="hod")

    def fake_pk2d_hod(cosmo, *, k_array, a_array, **hod_kwargs):
        """Return a tagged HOD power object."""
        return pk_hod

    monkeypatch.setattr(modelling, "pk2d_hod", fake_pk2d_hod)

    result = modelling.pk2d_hod_with_nla(
        DummyCosmology(Omega_m=0.3),
        k_array=np.array([0.1, 1.0], dtype=float),
        a_array=np.array([0.5, 1.0], dtype=float),
        A_IA=0.0,
    )

    assert result is pk_hod


def test_pk2d_hod_with_nla_uses_supplied_galaxy_bias(monkeypatch):
    """Tests that pk2d_hod_with_nla adds NLA power using supplied galaxy bias."""
    calls = {}
    pk_hod = DummyPk2D(
        a_arr=np.array([0.5, 1.0]),
        lk_arr=np.log([0.1, 1.0]),
        pk_arr=np.ones((2, 2)),
        is_logp=False,
    )
    pk_ia = DummyPk2D(
        a_arr=np.array([0.5, 1.0]),
        lk_arr=np.log([0.1, 1.0]),
        pk_arr=np.ones((2, 2)),
        is_logp=False,
    )

    def fake_pk2d_hod(cosmo, *, k_array, a_array, **hod_kwargs):
        """Return a tagged HOD power object."""
        calls["hod_kwargs"] = dict(hod_kwargs)
        return pk_hod

    def fake_pk2d_nla(cosmo, *, k_array, a_array, A_IA, C1rhocrit, b_g):
        """Return a tagged NLA power object."""
        calls["nla"] = {
            "A_IA": A_IA,
            "C1rhocrit": C1rhocrit,
            "b_g": b_g,
        }
        return pk_ia

    def fail_hod_galaxy_bias(*args, **kwargs):
        """Fail if automatic HOD bias is unexpectedly used."""
        raise AssertionError("hod_galaxy_bias should not be called")

    monkeypatch.setattr(modelling, "pk2d_hod", fake_pk2d_hod)
    monkeypatch.setattr(modelling, "pk2d_nla", fake_pk2d_nla)
    monkeypatch.setattr(modelling, "hod_galaxy_bias", fail_hod_galaxy_bias)

    result = modelling.pk2d_hod_with_nla(
        DummyCosmology(Omega_m=0.3),
        k_array=np.array([0.1, 1.0], dtype=float),
        a_array=np.array([0.5, 1.0], dtype=float),
        A_IA=1.2,
        C1rhocrit=0.02,
        b_g=1.7,
        log10Mmin_0=12.5,
    )

    assert result.combined is True
    assert result.left is pk_hod
    assert result.right is pk_ia
    assert calls["hod_kwargs"] == {"log10Mmin_0": 12.5}
    assert calls["nla"] == {
        "A_IA": 1.2,
        "C1rhocrit": 0.02,
        "b_g": 1.7,
    }


def test_pk2d_hod_with_nla_computes_galaxy_bias_when_missing(monkeypatch):
    """Tests that pk2d_hod_with_nla computes galaxy bias when b_g is absent."""
    calls = {}
    pk_hod = DummyPk2D(
        a_arr=np.array([0.5, 1.0]),
        lk_arr=np.log([0.1, 1.0]),
        pk_arr=np.ones((2, 2)),
        is_logp=False,
    )
    pk_ia = DummyPk2D(
        a_arr=np.array([0.5, 1.0]),
        lk_arr=np.log([0.1, 1.0]),
        pk_arr=np.ones((2, 2)),
        is_logp=False,
    )

    def fake_pk2d_hod(cosmo, *, k_array, a_array, **hod_kwargs):
        """Return a tagged HOD power object."""
        return pk_hod

    def fake_hod_galaxy_bias(cosmo, *, a, **hod_kwargs):
        """Return a deterministic HOD galaxy bias."""
        calls["bias"] = {"a": a, "hod_kwargs": dict(hod_kwargs)}
        return 2.3

    def fake_pk2d_nla(cosmo, *, k_array, a_array, A_IA, C1rhocrit, b_g):
        """Return a tagged NLA power object."""
        calls["nla_b_g"] = b_g
        return pk_ia

    monkeypatch.setattr(modelling, "pk2d_hod", fake_pk2d_hod)
    monkeypatch.setattr(modelling, "hod_galaxy_bias", fake_hod_galaxy_bias)
    monkeypatch.setattr(modelling, "pk2d_nla", fake_pk2d_nla)

    result = modelling.pk2d_hod_with_nla(
        DummyCosmology(Omega_m=0.3),
        k_array=np.array([0.1, 1.0], dtype=float),
        a_array=np.array([0.5, 1.0], dtype=float),
        A_IA=1.0,
        a_bias=0.67,
        log10Mmin_0=12.5,
    )

    assert result.combined is True
    assert calls["bias"] == {
        "a": 0.67,
        "hod_kwargs": {"log10Mmin_0": 12.5},
    }
    assert calls["nla_b_g"] == pytest.approx(2.3)


def test_pk2d_hod_baryonified_with_nla_returns_baryonified_only_when_amplitude_is_zero(
    monkeypatch,
):
    """Tests that pk2d_hod_baryonified_with_nla returns baryonified power for zero IA."""
    pk_hod = SimpleNamespace(name="baryonified")

    def fake_pk2d_hod_baryonified(cosmo, *, k_array, a_array, f_c, **hod_kwargs):
        """Return a tagged baryonified HOD power object."""
        return pk_hod

    monkeypatch.setattr(
        modelling,
        "pk2d_hod_baryonified",
        fake_pk2d_hod_baryonified,
    )

    result = modelling.pk2d_hod_baryonified_with_nla(
        DummyCosmology(Omega_m=0.3),
        k_array=np.array([0.1, 1.0], dtype=float),
        a_array=np.array([0.5, 1.0], dtype=float),
        f_c=1.4,
        A_IA=0.0,
    )

    assert result is pk_hod


def test_pk2d_hod_baryonified_with_nla_passes_fc_and_adds_nla(monkeypatch):
    """Tests that pk2d_hod_baryonified_with_nla passes f_c and adds NLA power."""
    calls = {}
    pk_hod = DummyPk2D(
        a_arr=np.array([0.5, 1.0]),
        lk_arr=np.log([0.1, 1.0]),
        pk_arr=np.ones((2, 2)),
        is_logp=False,
    )
    pk_ia = DummyPk2D(
        a_arr=np.array([0.5, 1.0]),
        lk_arr=np.log([0.1, 1.0]),
        pk_arr=np.ones((2, 2)),
        is_logp=False,
    )

    def fake_pk2d_hod_baryonified(cosmo, *, k_array, a_array, f_c, **hod_kwargs):
        """Return a tagged baryonified HOD power object."""
        calls["baryonified"] = {"f_c": f_c, "hod_kwargs": dict(hod_kwargs)}
        return pk_hod

    def fake_pk2d_nla(cosmo, *, k_array, a_array, A_IA, C1rhocrit, b_g):
        """Return a tagged NLA power object."""
        calls["nla"] = {
            "A_IA": A_IA,
            "C1rhocrit": C1rhocrit,
            "b_g": b_g,
        }
        return pk_ia

    monkeypatch.setattr(
        modelling,
        "pk2d_hod_baryonified",
        fake_pk2d_hod_baryonified,
    )
    monkeypatch.setattr(modelling, "pk2d_nla", fake_pk2d_nla)

    result = modelling.pk2d_hod_baryonified_with_nla(
        DummyCosmology(Omega_m=0.3),
        k_array=np.array([0.1, 1.0], dtype=float),
        a_array=np.array([0.5, 1.0], dtype=float),
        f_c=1.4,
        A_IA=0.8,
        C1rhocrit=0.01,
        b_g=1.9,
        log10M1_0=13.5,
    )

    assert result.combined is True
    assert calls["baryonified"] == {
        "f_c": 1.4,
        "hod_kwargs": {"log10M1_0": 13.5},
    }
    assert calls["nla"] == {
        "A_IA": 0.8,
        "C1rhocrit": 0.01,
        "b_g": 1.9,
    }
