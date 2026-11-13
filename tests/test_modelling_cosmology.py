"""Unit tests for ``dsf.modelling.cosmology``."""

import numpy as np
import pyccl as ccl
import pytest

import dsf.modelling.cosmology as cosmology


def test_make_ccl_cosmology_uses_vanilla_lcdm_without_mapping(monkeypatch):
    """Tests that the default construction uses VanillaLCDM."""
    calls = []

    def fake_vanilla(**kwargs):
        """Return a tagged vanilla cosmology."""
        calls.append(kwargs)
        return {"kind": "vanilla", **kwargs}

    monkeypatch.setattr(cosmology.ccl, "CosmologyVanillaLCDM", fake_vanilla)

    result = cosmology.make_ccl_cosmology()

    assert result == {"kind": "vanilla"}
    assert calls == [{}]


def test_make_ccl_cosmology_forwards_vanilla_kwargs(monkeypatch):
    """Tests that configuration arguments are forwarded to VanillaLCDM."""
    calls = []

    def fake_vanilla(**kwargs):
        """Record the supplied cosmology parameters."""
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "CosmologyVanillaLCDM", fake_vanilla)

    result = cosmology.make_ccl_cosmology(
        transfer_function="eisenstein_hu",
        matter_power_spectrum="halofit",
    )

    expected = {
        "transfer_function": "eisenstein_hu",
        "matter_power_spectrum": "halofit",
    }

    assert result == expected
    assert calls == [expected]


def test_make_ccl_cosmology_uses_standard_cosmology_with_mapping(monkeypatch):
    """Tests that a parameter mapping constructs a standard CCL cosmology."""
    calls = []

    def fake_cosmology(**kwargs):
        """Record the supplied cosmology parameters."""
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    parameters = {
        "Omega_c": 0.25,
        "Omega_b": 0.05,
        "h": 0.7,
        "sigma8": 0.8,
        "n_s": 0.96,
    }

    result = cosmology.make_ccl_cosmology(parameters)

    assert result == parameters
    assert calls == [parameters]


def test_make_ccl_cosmology_converts_omega_m_to_omega_c(monkeypatch):
    """Tests that Omega_m is converted to Omega_c using Omega_b."""
    calls = []

    def fake_cosmology(**kwargs):
        """Record the converted cosmology parameters."""
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    parameters = {
        "Omega_m": 0.3,
        "Omega_b": 0.05,
        "h": 0.7,
    }

    result = cosmology.make_ccl_cosmology(parameters)

    assert result["Omega_c"] == pytest.approx(0.25)
    assert result["Omega_b"] == 0.05
    assert "Omega_m" not in result
    assert calls == [result]


def test_make_ccl_cosmology_preserves_omega_c_without_omega_m(monkeypatch):
    """Tests that Omega_c is preserved when Omega_m is absent."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology(
        {
            "Omega_c": 0.26,
            "Omega_b": 0.04,
        }
    )

    assert result["Omega_c"] == 0.26
    assert result["Omega_b"] == 0.04
    assert "Omega_m" not in result


def test_make_ccl_cosmology_kwargs_override_mapping(monkeypatch):
    """Tests that keyword arguments take precedence over mapping values."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology(
        {"Omega_c": 0.25, "Omega_b": 0.05, "h": 0.7},
        h=0.8,
    )

    assert result["Omega_c"] == 0.25
    assert result["Omega_b"] == 0.05
    assert result["h"] == 0.8


def test_make_ccl_cosmology_converts_overridden_omega_m(monkeypatch):
    """Tests that Omega_m overrides are applied before conversion."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology(
        {"Omega_m": 0.3, "Omega_b": 0.05},
        Omega_m=0.35,
    )

    assert result["Omega_c"] == pytest.approx(0.30)
    assert result["Omega_b"] == 0.05
    assert "Omega_m" not in result


def test_make_ccl_cosmology_does_not_modify_input_mapping(monkeypatch):
    """Tests that cosmology construction does not mutate the input mapping."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    parameters = {
        "Omega_m": 0.3,
        "Omega_b": 0.05,
        "h": 0.7,
    }
    original = parameters.copy()

    cosmology.make_ccl_cosmology(parameters, h=0.8)

    assert parameters == original


def test_make_ccl_cosmology_accepts_empty_mapping(monkeypatch):
    """Tests that an empty mapping uses the standard CCL constructor."""
    calls = []

    def fake_cosmology(**kwargs):
        """Record the supplied cosmology parameters."""
        calls.append(kwargs)
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology({})

    assert result == {}
    assert calls == [{}]


def test_make_ccl_cosmology_rejects_omega_m_without_omega_b():
    """Tests that Omega_m conversion requires Omega_b."""
    with pytest.raises(KeyError, match="Omega_b"):
        cosmology.make_ccl_cosmology({"Omega_m": 0.3})


def test_make_ccl_cosmology_converts_s8_with_omega_m(monkeypatch):
    """Tests that S8 is converted to sigma8 using Omega_m."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology(
        {
            "Omega_m": 0.25,
            "Omega_b": 0.05,
            "S8": 0.8,
        }
    )

    assert result["sigma8"] == pytest.approx(0.8 * (0.3 / 0.25) ** 0.5)
    assert result["Omega_c"] == pytest.approx(0.20)
    assert "S8" not in result
    assert "Omega_m" not in result


def test_make_ccl_cosmology_converts_s8_with_omega_c(monkeypatch):
    """Tests that S8 uses Omega_c and Omega_b when Omega_m is absent."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology(
        {
            "Omega_c": 0.20,
            "Omega_b": 0.05,
            "S8": 0.8,
        }
    )

    assert result["sigma8"] == pytest.approx(0.8 * (0.3 / 0.25) ** 0.5)
    assert result["Omega_c"] == 0.20
    assert result["Omega_b"] == 0.05
    assert "S8" not in result


def test_make_ccl_cosmology_converts_s8_at_fiducial_omega_m(monkeypatch):
    """Tests that S8 equals sigma8 when Omega_m is 0.3."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology(
        {
            "Omega_m": 0.3,
            "Omega_b": 0.05,
            "S8": 0.82,
        }
    )

    assert result["sigma8"] == pytest.approx(0.82)


def test_make_ccl_cosmology_converts_overridden_s8(monkeypatch):
    """Tests that keyword overrides are applied before S8 conversion."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology(
        {"Omega_m": 0.3, "Omega_b": 0.05, "S8": 0.8},
        Omega_m=0.25,
        S8=0.85,
    )

    assert result["sigma8"] == pytest.approx(0.85 * (0.3 / 0.25) ** 0.5)
    assert result["Omega_c"] == pytest.approx(0.20)
    assert "S8" not in result


@pytest.mark.parametrize("amplitude", ["sigma8", "A_s"])
def test_make_ccl_cosmology_rejects_conflicting_amplitudes(amplitude):
    """Tests that a single mapping cannot contain competing amplitudes."""
    parameters = {
        "Omega_m": 0.3,
        "Omega_b": 0.05,
        "S8": 0.8,
        amplitude: 0.81,
    }

    with pytest.raises(ValueError, match="S8, sigma8, and A_s"):
        cosmology.make_ccl_cosmology(parameters)


def test_make_ccl_cosmology_overrides_s8_with_sigma8(monkeypatch):
    """Tests that sigma8 overrides a fiducial S8."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology(
        {"Omega_m": 0.3, "Omega_b": 0.05, "S8": 0.8},
        sigma8=0.85,
    )

    assert result["sigma8"] == pytest.approx(0.85)
    assert "S8" not in result


def test_make_ccl_cosmology_overrides_sigma8_with_s8(monkeypatch):
    """Tests that S8 overrides a fiducial sigma8."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology(
        {"Omega_m": 0.25, "Omega_b": 0.05, "sigma8": 0.8},
        S8=0.85,
    )

    assert result["sigma8"] == pytest.approx(0.85 * (0.3 / 0.25) ** 0.5)
    assert "S8" not in result


def test_make_ccl_cosmology_overrides_as_with_s8(monkeypatch):
    """Tests that S8 overrides a fiducial primordial amplitude."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology(
        {"Omega_m": 0.3, "Omega_b": 0.05, "A_s": 2.1e-9},
        S8=0.82,
    )

    assert result["sigma8"] == pytest.approx(0.82)
    assert "A_s" not in result


def test_make_ccl_cosmology_overrides_s8_with_as(monkeypatch):
    """Tests that A_s overrides a fiducial S8."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology(
        {"Omega_m": 0.3, "Omega_b": 0.05, "S8": 0.8},
        A_s=2.1e-9,
    )

    assert result["A_s"] == pytest.approx(2.1e-9)
    assert "S8" not in result
    assert "sigma8" not in result


def test_make_ccl_cosmology_overrides_omega_c_with_omega_m(monkeypatch):
    """Tests that Omega_m overrides a fiducial Omega_c."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology(
        {"Omega_c": 0.25, "Omega_b": 0.05},
        Omega_m=0.35,
    )

    assert result["Omega_c"] == pytest.approx(0.30)
    assert "Omega_m" not in result


def test_make_ccl_cosmology_overrides_omega_m_with_omega_c(monkeypatch):
    """Tests that Omega_c overrides a fiducial Omega_m."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology(
        {"Omega_m": 0.3, "Omega_b": 0.05},
        Omega_c=0.28,
    )

    assert result["Omega_c"] == pytest.approx(0.28)
    assert "Omega_m" not in result


def test_make_ccl_cosmology_varies_omega_m_at_fixed_s8(monkeypatch):
    """Tests that varying Omega_m at fixed S8 changes sigma8."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    fiducial = {
        "Omega_m": 0.3,
        "Omega_b": 0.05,
        "S8": 0.8,
    }

    first = cosmology.make_ccl_cosmology(fiducial, Omega_m=0.3)
    second = cosmology.make_ccl_cosmology(fiducial, Omega_m=0.35)

    assert first["sigma8"] == pytest.approx(0.8)
    assert second["sigma8"] == pytest.approx(0.8 * (0.3 / 0.35) ** 0.5)
    assert second["Omega_c"] == pytest.approx(0.30)
    assert first["sigma8"] != second["sigma8"]


def test_make_ccl_cosmology_varies_s8_at_fixed_omega_m(monkeypatch):
    """Tests that varying S8 changes sigma8 at fixed Omega_m."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    fiducial = {
        "Omega_m": 0.3,
        "Omega_b": 0.05,
        "S8": 0.8,
    }

    first = cosmology.make_ccl_cosmology(fiducial, S8=0.8)
    second = cosmology.make_ccl_cosmology(fiducial, S8=0.85)

    assert first["sigma8"] == pytest.approx(0.8)
    assert second["sigma8"] == pytest.approx(0.85)
    assert first["Omega_c"] == second["Omega_c"]


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("h", 0.75),
        ("n_s", 0.98),
        ("w0", -0.9),
        ("wa", 0.2),
        ("Omega_b", 0.06),
    ],
)
def test_make_ccl_cosmology_varies_standard_parameters(monkeypatch, parameter, value):
    """Tests that standard cosmological parameters can be varied."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    fiducial = {
        "Omega_m": 0.3,
        "Omega_b": 0.05,
        "h": 0.7,
        "n_s": 0.96,
        "S8": 0.8,
        "w0": -1.0,
        "wa": 0.0,
    }

    result = cosmology.make_ccl_cosmology(fiducial, **{parameter: value})

    assert result[parameter] == pytest.approx(value)


def test_make_ccl_cosmology_preserves_input_mapping_with_s8(monkeypatch):
    """Tests that S8 conversion does not mutate the original mapping."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    parameters = {
        "Omega_m": 0.25,
        "Omega_b": 0.05,
        "S8": 0.8,
    }
    original = parameters.copy()

    cosmology.make_ccl_cosmology(parameters)

    assert parameters == original
    assert "S8" in parameters


def test_make_ccl_cosmology_preserves_sigma8_without_s8(monkeypatch):
    """Tests that sigma8 is unchanged when S8 is not supplied."""

    def fake_cosmology(**kwargs):
        """Return the supplied cosmology parameters."""
        return kwargs

    monkeypatch.setattr(cosmology.ccl, "Cosmology", fake_cosmology)

    result = cosmology.make_ccl_cosmology(
        {
            "Omega_m": 0.25,
            "Omega_b": 0.05,
            "sigma8": 0.82,
        }
    )

    assert result["sigma8"] == 0.82
    assert result["Omega_c"] == pytest.approx(0.20)


@pytest.mark.parametrize(
    "parameters",
    [
        {"Omega_m": 0.3, "Omega_c": 0.25, "Omega_b": 0.05},
        {"Omega_m": 0.3, "Omega_b": 0.05, "S8": 0.8, "sigma8": 0.8},
        {"Omega_m": 0.3, "Omega_b": 0.05, "sigma8": 0.8, "A_s": 2.1e-9},
    ],
)
def test_make_ccl_cosmology_rejects_conflicting_mapping_parameters(parameters):
    """Tests that ambiguous parameterizations are rejected."""
    with pytest.raises(ValueError):
        cosmology.make_ccl_cosmology(parameters)


@pytest.mark.parametrize(
    "overrides",
    [
        {"Omega_m": 0.3, "Omega_c": 0.25},
        {"S8": 0.8, "sigma8": 0.8},
        {"sigma8": 0.8, "A_s": 2.1e-9},
    ],
)
def test_make_ccl_cosmology_rejects_conflicting_overrides(overrides):
    """Tests that ambiguous keyword overrides are rejected."""
    with pytest.raises(ValueError):
        cosmology.make_ccl_cosmology({}, **overrides)


@pytest.mark.parametrize("omega_m", [0.0, -0.1])
def test_make_ccl_cosmology_rejects_nonpositive_omega_m_for_s8(omega_m):
    """Tests that S8 conversion requires positive Omega_m."""
    with pytest.raises(ValueError, match="Omega_m must be positive"):
        cosmology.make_ccl_cosmology(
            {
                "Omega_m": omega_m,
                "Omega_b": 0.05,
                "S8": 0.8,
            }
        )


def test_make_ccl_cosmology_constructs_real_ccl_object():
    """Tests that cosmology construction returns a real CCL object."""
    parameters = {
        "Omega_m": 0.3,
        "Omega_b": 0.05,
        "h": 0.7,
        "n_s": 0.96,
        "S8": 0.8,
    }

    result = cosmology.make_ccl_cosmology(parameters)

    assert isinstance(result, ccl.Cosmology)
    assert result["Omega_c"] == pytest.approx(0.25)
    assert result["Omega_b"] == pytest.approx(0.05)
    assert result["sigma8"] == pytest.approx(0.8)


def test_make_ccl_cosmology_constructs_real_ccl_with_overrides():
    """Tests that parameter variations produce valid CCL cosmologies."""
    fiducial = {
        "Omega_m": 0.3,
        "Omega_b": 0.05,
        "h": 0.7,
        "n_s": 0.96,
        "S8": 0.8,
    }

    first = cosmology.make_ccl_cosmology(fiducial)
    second = cosmology.make_ccl_cosmology(fiducial, Omega_m=0.32, S8=0.85)

    assert isinstance(first, ccl.Cosmology)
    assert isinstance(second, ccl.Cosmology)

    assert first["Omega_c"] == pytest.approx(0.25)
    assert second["Omega_c"] == pytest.approx(0.27)

    assert first["sigma8"] == pytest.approx(0.8)
    assert second["sigma8"] == pytest.approx(0.85 * (0.3 / 0.32) ** 0.5)


def test_make_ccl_cosmology_derivative_parameter_dependence():
    """Tests that S8 and Omega_m variations produce the expected derivatives."""
    fiducial = {
        "Omega_m": 0.3,
        "Omega_b": 0.05,
        "h": 0.7,
        "n_s": 0.96,
        "S8": 0.8,
    }

    def sigma8_model(omega_m, s8):
        """Return the CCL amplitude for the supplied forecast parameters."""
        cosmo = cosmology.make_ccl_cosmology(
            fiducial,
            Omega_m=omega_m,
            S8=s8,
        )
        return cosmo["sigma8"]

    omega_m = 0.3
    s8 = 0.8
    step = 1.0e-5

    derivative_omega_m = (sigma8_model(omega_m + step, s8) - sigma8_model(omega_m - step, s8)) / (
        2.0 * step
    )

    derivative_s8 = (sigma8_model(omega_m, s8 + step) - sigma8_model(omega_m, s8 - step)) / (
        2.0 * step
    )

    expected_omega_m = -0.5 * s8 / omega_m
    expected_s8 = 1.0

    np.testing.assert_allclose(derivative_omega_m, expected_omega_m, rtol=1.0e-6)
    np.testing.assert_allclose(derivative_s8, expected_s8, rtol=1.0e-6)
