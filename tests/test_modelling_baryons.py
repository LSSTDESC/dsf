"""Unit tests for ``dsf.modelling.baryons``."""

import numpy as np
import pyccl as ccl
import pytest

import dsf.modelling.baryons as baryons
from dsf.modelling.hod import MASS_DEF

CONCENTRATION_MODELS = [
    (ccl.halos.ConcentrationBhattacharya13, "200c", {}),
    (ccl.halos.ConcentrationConstant, "200c", {"c": 5.0}),
    (ccl.halos.ConcentrationDiemer15, "200c", {}),
    (ccl.halos.ConcentrationDuffy08, "200c", {}),
    (ccl.halos.ConcentrationIshiyama21, "500c", {}),
    (ccl.halos.ConcentrationKlypin11, "vir", {}),
    (ccl.halos.ConcentrationPrada12, "200c", {}),
]


@pytest.fixture
def cosmo():
    """Construct a CCL cosmology for concentration calculations."""
    return ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        transfer_function="eisenstein_hu",
        matter_power_spectrum="linear",
    )


@pytest.fixture
def grids():
    """Construct wavenumber and scale-factor grids."""
    return {
        "k_array": np.geomspace(1.0e-3, 10.0, 20),
        "a_array": np.linspace(0.5, 1.0, 10),
    }


@pytest.fixture
def capture_hod(monkeypatch):
    """Replace the HOD calculation with an argument recorder."""
    calls = []

    def fake_hod(cosmo, **kwargs):
        """Record the supplied HOD arguments."""
        calls.append((cosmo, kwargs))
        return {"pk_gm": "gm", "pk_gg": "gg"}

    monkeypatch.setattr(baryons, "_pk2d_hod", fake_hod)

    return calls


def test_pk2d_hod_baryonified_uses_duffy_by_default(cosmo, grids, capture_hod):
    """Tests that Duffy08 is the default concentration model."""
    baryons.pk2d_hod_baryonified(cosmo, **grids)

    concentration = capture_hod[0][1]["concentration"]

    assert isinstance(concentration, ccl.halos.ConcentrationDuffy08)
    assert concentration.fc_bar == pytest.approx(1.0)
    assert concentration.mass_def == MASS_DEF


def test_pk2d_hod_baryonified_rescales_default_duffy(cosmo, grids, capture_hod):
    """Tests that the default Duffy08 model supports rescaling."""
    baryons.pk2d_hod_baryonified(cosmo, **grids, f_c=1.2)

    concentration = capture_hod[0][1]["concentration"]

    assert isinstance(concentration, baryons._RescaledConcentration)
    assert isinstance(concentration.concentration, ccl.halos.ConcentrationDuffy08)
    assert concentration.f_c == pytest.approx(1.2)
    assert concentration.concentration.fc_bar == pytest.approx(1.0)


def test_pk2d_hod_baryonified_forwards_cosmology(cosmo, grids, capture_hod):
    """Tests that the original cosmology is forwarded to the HOD model."""
    baryons.pk2d_hod_baryonified(cosmo, **grids)

    assert capture_hod[0][0] is cosmo


def test_pk2d_hod_baryonified_forwards_grids(cosmo, grids, capture_hod):
    """Tests that the input grids are forwarded unchanged."""
    baryons.pk2d_hod_baryonified(cosmo, **grids)

    kwargs = capture_hod[0][1]

    assert kwargs["k_array"] is grids["k_array"]
    assert kwargs["a_array"] is grids["a_array"]


def test_pk2d_hod_baryonified_forwards_hod_kwargs(
    cosmo, grids, capture_hod,
):
    """Tests that additional HOD parameters are forwarded."""
    baryons.pk2d_hod_baryonified(
        cosmo,
        **grids,
        f_c=1.2,
        log10Mmin_0=12.5,
        siglnM_0=0.4,
    )

    kwargs = capture_hod[0][1]

    assert kwargs["log10Mmin_0"] == 12.5
    assert kwargs["siglnM_0"] == 0.4


def test_pk2d_hod_baryonified_returns_hod_spectra(cosmo, grids, capture_hod):
    """Tests that the HOD result is returned unchanged."""
    result = baryons.pk2d_hod_baryonified(cosmo, **grids)

    assert result == {"pk_gm": "gm", "pk_gg": "gg"}


@pytest.mark.parametrize(
    ("model", "mass_def", "parameters"),
    CONCENTRATION_MODELS,
)
def test_pk2d_hod_baryonified_accepts_all_ccl_models(
    cosmo, grids, capture_hod, model, mass_def, parameters,
):
    """Tests that every available CCL concentration model is accepted."""
    concentration = model(mass_def=mass_def, **parameters)

    baryons.pk2d_hod_baryonified(
        cosmo,
        **grids,
        concentration=concentration,
    )

    result = capture_hod[0][1]["concentration"]

    assert result is concentration
    assert isinstance(result, model)


@pytest.mark.parametrize(
    ("model", "mass_def", "parameters"),
    CONCENTRATION_MODELS,
)
def test_pk2d_hod_baryonified_rescales_all_ccl_models(
    cosmo, grids, capture_hod, model, mass_def, parameters,
):
    """Tests that every CCL concentration model supports rescaling."""
    concentration = model(mass_def=mass_def, **parameters)

    baryons.pk2d_hod_baryonified(
        cosmo,
        **grids,
        f_c=1.2,
        concentration=concentration,
    )

    result = capture_hod[0][1]["concentration"]

    assert isinstance(result, baryons._RescaledConcentration)
    assert result.concentration is concentration
    assert result.f_c == pytest.approx(1.2)
    assert result.mass_def == concentration.mass_def


@pytest.mark.parametrize(
    ("model", "mass_def", "parameters"),
    CONCENTRATION_MODELS,
)
def test_rescaled_concentration_matches_ccl(
    cosmo, model, mass_def, parameters,
):
    """Tests numerical rescaling against the original CCL concentration."""
    concentration = model(mass_def=mass_def, **parameters)
    rescaled = baryons._RescaledConcentration(concentration, 1.2)

    masses = np.geomspace(1.0e12, 1.0e15, 5)
    scale_factor = 0.8

    expected = 1.2 * concentration(cosmo, masses, scale_factor)
    result = rescaled(cosmo, masses, scale_factor)

    np.testing.assert_allclose(result, expected, rtol=1.0e-12)


@pytest.mark.parametrize(
    ("model", "mass_def", "parameters"),
    CONCENTRATION_MODELS,
)
def test_rescaled_concentration_identity(
    cosmo, model, mass_def, parameters,
):
    """Tests that unit rescaling preserves the original concentration."""
    concentration = model(mass_def=mass_def, **parameters)
    rescaled = baryons._RescaledConcentration(concentration, 1.0)

    masses = np.geomspace(1.0e12, 1.0e15, 5)

    expected = concentration(cosmo, masses, 0.8)
    result = rescaled(cosmo, masses, 0.8)

    np.testing.assert_allclose(result, expected, rtol=1.0e-12)


@pytest.mark.parametrize("f_c", [0.5, 0.8, 1.0, 1.2, 2.0])
def test_rescaled_duffy_concentration_factors(cosmo, f_c):
    """Tests Duffy08 concentration rescaling for different factors."""
    concentration = ccl.halos.ConcentrationDuffy08(mass_def=MASS_DEF)
    rescaled = baryons._RescaledConcentration(concentration, f_c)

    masses = np.geomspace(1.0e12, 1.0e15, 5)

    expected = f_c * concentration(cosmo, masses, 0.8)
    result = rescaled(cosmo, masses, 0.8)

    np.testing.assert_allclose(result, expected, rtol=1.0e-12)


def test_rescaled_concentration_preserves_mass_definition():
    """Tests that the wrapper preserves the original mass definition."""
    concentration = ccl.halos.ConcentrationKlypin11(mass_def="vir")
    rescaled = baryons._RescaledConcentration(concentration, 1.2)

    assert rescaled.mass_def == concentration.mass_def


def test_rescaled_concentration_preserves_original_model(cosmo):
    """Tests that rescaling does not modify the supplied CCL model."""
    concentration = ccl.halos.ConcentrationDuffy08(
        mass_def=MASS_DEF,
        fc_bar=1.3,
    )

    masses = np.array([1.0e13, 1.0e14])
    original = concentration(cosmo, masses, 0.8)

    rescaled = baryons._RescaledConcentration(concentration, 1.2)

    result = concentration(cosmo, masses, 0.8)

    np.testing.assert_allclose(result, original, rtol=1.0e-12)
    assert concentration.fc_bar == pytest.approx(1.3)
    assert rescaled.concentration is concentration


def test_rescaled_duffy_preserves_existing_fc_bar(cosmo):
    """Tests that an existing Duffy08 rescaling is not overwritten."""
    concentration = ccl.halos.ConcentrationDuffy08(
        mass_def=MASS_DEF,
        fc_bar=1.3,
    )
    rescaled = baryons._RescaledConcentration(concentration, 1.2)

    masses = np.array([1.0e13, 1.0e14])

    standard = ccl.halos.ConcentrationDuffy08(mass_def=MASS_DEF)
    expected = 1.2 * 1.3 * standard(cosmo, masses, 0.8)
    result = rescaled(cosmo, masses, 0.8)

    np.testing.assert_allclose(result, expected, rtol=1.0e-12)


def test_pk2d_hod_baryonified_preserves_custom_model(cosmo, grids, capture_hod):
    """Tests that a custom concentration model is not modified."""
    concentration = ccl.halos.ConcentrationConstant(
        c=5.0,
        mass_def=MASS_DEF,
    )

    baryons.pk2d_hod_baryonified(
        cosmo,
        **grids,
        f_c=1.5,
        concentration=concentration,
    )

    result = capture_hod[0][1]["concentration"]

    assert result.concentration is concentration
    assert concentration.c == pytest.approx(5.0)


@pytest.mark.parametrize("f_c", [0.0, -0.1, -1.0, np.nan, np.inf, -np.inf])
def test_pk2d_hod_baryonified_rejects_invalid_f_c(cosmo, grids, f_c):
    """Tests that invalid concentration rescaling factors are rejected."""
    with pytest.raises((ValueError, TypeError)):
        baryons.pk2d_hod_baryonified(cosmo, **grids, f_c=f_c)


def test_pk2d_hod_baryonified_rejects_non_scalar_f_c(cosmo, grids):
    """Tests that concentration rescaling must be scalar."""
    with pytest.raises((ValueError, TypeError)):
        baryons.pk2d_hod_baryonified(
            cosmo,
            **grids,
            f_c=np.array([1.0, 1.2]),
        )


def test_rescaled_concentration_derivative(cosmo):
    """Tests that concentration depends linearly on the rescaling factor."""
    concentration = ccl.halos.ConcentrationDuffy08(mass_def=MASS_DEF)

    masses = np.array([1.0e13, 1.0e14])
    scale_factor = 0.8
    step = 1.0e-5

    def evaluate(f_c):
        """Evaluate the concentration at the supplied rescaling factor."""
        model = baryons._RescaledConcentration(concentration, f_c)
        return model(cosmo, masses, scale_factor)

    derivative = (
        evaluate(1.0 + step) - evaluate(1.0 - step)
    ) / (2.0 * step)

    expected = concentration(cosmo, masses, scale_factor)

    np.testing.assert_allclose(derivative, expected, rtol=1.0e-8)


def test_pk2d_hod_baryonified_f_c_changes_concentration(
    cosmo, grids, capture_hod,
):
    """Tests that varying f_c changes the concentration passed to HOD."""
    baryons.pk2d_hod_baryonified(cosmo, **grids, f_c=0.8)
    baryons.pk2d_hod_baryonified(cosmo, **grids, f_c=1.2)

    first = capture_hod[0][1]["concentration"]
    second = capture_hod[1][1]["concentration"]

    masses = np.array([1.0e13, 1.0e14])

    first_values = first(cosmo, masses, 0.8)
    second_values = second(cosmo, masses, 0.8)

    np.testing.assert_allclose(
        second_values / first_values,
        1.2 / 0.8,
        rtol=1.0e-12,
    )


@pytest.mark.parametrize(
    ("model", "mass_def", "parameters"),
    CONCENTRATION_MODELS,
)
def test_rescaled_concentration_scalar_mass(
    cosmo, model, mass_def, parameters,
):
    """Tests concentration rescaling for scalar halo masses."""
    concentration = model(mass_def=mass_def, **parameters)
    rescaled = baryons._RescaledConcentration(concentration, 1.2)

    mass = 1.0e14
    scale_factor = 0.8

    expected = 1.2 * concentration(cosmo, mass, scale_factor)
    result = rescaled(cosmo, mass, scale_factor)

    np.testing.assert_allclose(result, expected, rtol=1.0e-12)


@pytest.mark.parametrize(
    ("model", "mass_def", "parameters"),
    CONCENTRATION_MODELS,
)
def test_rescaled_concentration_scale_factor_dependence(
    cosmo, model, mass_def, parameters,
):
    """Tests concentration rescaling at different scale factors."""
    concentration = model(mass_def=mass_def, **parameters)
    rescaled = baryons._RescaledConcentration(concentration, 1.2)

    masses = np.geomspace(1.0e12, 1.0e15, 5)

    for scale_factor in (0.5, 0.7, 0.9, 1.0):
        expected = 1.2 * concentration(cosmo, masses, scale_factor)
        result = rescaled(cosmo, masses, scale_factor)

        np.testing.assert_allclose(result, expected, rtol=1.0e-12)


@pytest.mark.parametrize(
    ("model", "mass_def", "parameters"),
    CONCENTRATION_MODELS,
)
def test_rescaled_concentration_mass_definition_validation(
    model, mass_def, parameters,
):
    """Tests that the wrapper preserves CCL mass-definition restrictions."""
    concentration = model(mass_def=mass_def, **parameters)
    rescaled = baryons._RescaledConcentration(concentration, 1.2)

    assert rescaled.mass_def == concentration.mass_def

    for candidate in ("200c", "200m", "500c", "vir"):
        mass_definition = ccl.halos.MassDef.from_name(candidate)

        assert (
            rescaled._check_mass_def_strict(mass_definition)
            == concentration._check_mass_def_strict(mass_definition)
        )


def test_rescaled_concentration_matches_native_duffy_fc_bar(cosmo):
    """Tests that generic rescaling reproduces native Duffy08 rescaling."""
    base = ccl.halos.ConcentrationDuffy08(mass_def=MASS_DEF)
    native = ccl.halos.ConcentrationDuffy08(
        mass_def=MASS_DEF,
        fc_bar=1.2,
    )
    rescaled = baryons._RescaledConcentration(base, 1.2)

    masses = np.geomspace(1.0e12, 1.0e15, 10)

    for scale_factor in (0.5, 0.8, 1.0):
        expected = native(cosmo, masses, scale_factor)
        result = rescaled(cosmo, masses, scale_factor)

        np.testing.assert_allclose(result, expected, rtol=1.0e-12)


def test_pk2d_hod_baryonified_default_matches_explicit_duffy(
    cosmo, grids, capture_hod,
):
    """Tests equivalence between default and explicit Duffy08 models."""
    baryons.pk2d_hod_baryonified(cosmo, **grids, f_c=1.2)

    baryons.pk2d_hod_baryonified(
        cosmo,
        **grids,
        f_c=1.2,
        concentration=ccl.halos.ConcentrationDuffy08(mass_def=MASS_DEF),
    )

    first = capture_hod[0][1]["concentration"]
    second = capture_hod[1][1]["concentration"]

    masses = np.geomspace(1.0e12, 1.0e15, 5)

    np.testing.assert_allclose(
        first(cosmo, masses, 0.8),
        second(cosmo, masses, 0.8),
        rtol=1.0e-12,
    )
