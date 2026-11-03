"""Unit tests for ``dsf.modelling.combinations``."""

import numpy as np
import pyccl as ccl
import pytest

import dsf.modelling.combinations as combinations


@pytest.fixture
def cosmo():
    """Construct a CCL cosmology for combined power-spectrum calculations."""
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
        "k_array": np.geomspace(1.0e-3, 5.0, 20),
        "a_array": np.linspace(0.5, 1.0, 10),
    }


@pytest.fixture
def concentration():
    """Construct a custom CCL concentration model."""
    return ccl.halos.ConcentrationConstant(
        c=5.0, mass_def=ccl.halos.MassDef200c
    )


@pytest.fixture
def mock_models(monkeypatch):
    """Replace the HOD, baryonic, NLA, and galaxy-bias calculations."""
    calls = {
        "hod": [],
        "baryons": [],
        "nla": [],
        "bias": [],
    }

    pk_hod = {
        "pk_deltasigma": 10.0,
        "pk_gm": object(),
        "pk_gg": object(),
    }

    pk_baryons = {
        "pk_deltasigma": 20.0,
        "pk_gm": object(),
        "pk_gg": object(),
    }

    pk_ia = {
        "pk_deltasigma": -2.0,
        "pk_gi": object(),
    }

    def fake_hod(cosmo, **kwargs):
        """Record standard HOD model arguments."""
        calls["hod"].append((cosmo, kwargs))
        return pk_hod

    def fake_baryons(cosmo, **kwargs):
        """Record baryonified HOD model arguments."""
        calls["baryons"].append((cosmo, kwargs))
        return pk_baryons

    def fake_nla(cosmo, **kwargs):
        """Record NLA model arguments."""
        calls["nla"].append((cosmo, kwargs))
        return pk_ia

    def fake_bias(cosmo, **kwargs):
        """Record galaxy-bias calculation arguments."""
        calls["bias"].append((cosmo, kwargs))
        return 1.8

    monkeypatch.setattr(combinations, "pk2d_hod", fake_hod)
    monkeypatch.setattr(combinations, "pk2d_hod_baryonified", fake_baryons)
    monkeypatch.setattr(combinations, "pk2d_nla", fake_nla)
    monkeypatch.setattr(combinations, "hod_galaxy_bias", fake_bias)

    return calls, pk_hod, pk_baryons, pk_ia


def test_hod_with_nla_returns_expected_keys(cosmo, grids, mock_models):
    """Tests that standard HOD with NLA preserves the HOD spectrum keys."""
    result = combinations.pk2d_hod_with_nla(
        cosmo, **grids, A_IA=1.0, b_g=1.5
    )

    assert set(result) == {"pk_deltasigma", "pk_gm", "pk_gg"}


def test_baryonified_with_nla_returns_expected_keys(cosmo, grids, mock_models):
    """Tests that baryonified HOD with NLA preserves the spectrum keys."""
    result = combinations.pk2d_hod_baryonified_with_nla(
        cosmo, **grids, A_IA=1.0, b_g=1.5
    )

    assert set(result) == {"pk_deltasigma", "pk_gm", "pk_gg"}


@pytest.mark.parametrize(
    "function_name,model_name",
    [
        ("pk2d_hod_with_nla", "hod"),
        ("pk2d_hod_baryonified_with_nla", "baryons"),
    ],
)
def test_combination_calls_correct_hod_model(
    cosmo, grids, mock_models, function_name, model_name,
):
    """Tests that each public function calls the corresponding HOD model."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids)

    assert len(calls[model_name]) == 1
    assert len(calls["baryons" if model_name == "hod" else "hod"]) == 0


@pytest.mark.parametrize(
    "function_name,model_name",
    [
        ("pk2d_hod_with_nla", "hod"),
        ("pk2d_hod_baryonified_with_nla", "baryons"),
    ],
)
def test_combination_forwards_cosmology(
    cosmo, grids, mock_models, function_name, model_name,
):
    """Tests that the original cosmology is forwarded to the HOD model."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids)

    assert calls[model_name][0][0] is cosmo


@pytest.mark.parametrize(
    "function_name,model_name",
    [
        ("pk2d_hod_with_nla", "hod"),
        ("pk2d_hod_baryonified_with_nla", "baryons"),
    ],
)
def test_combination_forwards_grids(
    cosmo, grids, mock_models, function_name, model_name,
):
    """Tests that wavenumber and scale-factor grids are forwarded unchanged."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids)

    kwargs = calls[model_name][0][1]

    assert kwargs["k_array"] is grids["k_array"]
    assert kwargs["a_array"] is grids["a_array"]


@pytest.mark.parametrize(
    "function_name,model_name",
    [
        ("pk2d_hod_with_nla", "hod"),
        ("pk2d_hod_baryonified_with_nla", "baryons"),
    ],
)
def test_combination_forwards_custom_concentration(
    cosmo, grids, concentration, mock_models, function_name, model_name,
):
    """Tests that custom concentration models are forwarded to the HOD model."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids, concentration=concentration)

    assert calls[model_name][0][1]["concentration"] is concentration


@pytest.mark.parametrize(
    "function_name,model_name",
    [
        ("pk2d_hod_with_nla", "hod"),
        ("pk2d_hod_baryonified_with_nla", "baryons"),
    ],
)
def test_combination_default_concentration(
    cosmo, grids, mock_models, function_name, model_name,
):
    """Tests that the default concentration argument is None."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids)

    assert calls[model_name][0][1]["concentration"] is None


@pytest.mark.parametrize(
    "function_name,model_name",
    [
        ("pk2d_hod_with_nla", "hod"),
        ("pk2d_hod_baryonified_with_nla", "baryons"),
    ],
)
def test_combination_forwards_hod_parameters(
    cosmo, grids, mock_models, function_name, model_name,
):
    """Tests that HOD profile parameters are forwarded."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(
        cosmo,
        **grids,
        log10Mmin_0=12.5,
        siglnM_0=0.4,
    )

    kwargs = calls[model_name][0][1]

    assert kwargs["log10Mmin_0"] == pytest.approx(12.5)
    assert kwargs["siglnM_0"] == pytest.approx(0.4)


def test_baryonified_with_nla_forwards_rescaling(cosmo, grids, mock_models):
    """Tests that baryonic concentration rescaling is forwarded."""
    calls, _, _, _ = mock_models

    combinations.pk2d_hod_baryonified_with_nla(
        cosmo, **grids, f_c=1.4
    )

    assert calls["baryons"][0][1]["f_c"] == pytest.approx(1.4)


def test_baryonified_with_nla_default_rescaling(cosmo, grids, mock_models):
    """Tests that baryonic concentration rescaling defaults to unity."""
    calls, _, _, _ = mock_models

    combinations.pk2d_hod_baryonified_with_nla(cosmo, **grids)

    assert calls["baryons"][0][1]["f_c"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "function_name,model_name",
    [
        ("pk2d_hod_with_nla", "hod"),
        ("pk2d_hod_baryonified_with_nla", "baryons"),
    ],
)
def test_combination_zero_ia_returns_original_spectra(
    cosmo, grids, mock_models, function_name, model_name,
):
    """Tests that zero IA amplitude returns the original HOD dictionary."""
    _, pk_hod, pk_baryons, _ = mock_models

    function = getattr(combinations, function_name)
    result = function(cosmo, **grids, A_IA=0.0)

    expected = pk_hod if model_name == "hod" else pk_baryons

    assert result is expected


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_zero_ia_skips_nla_and_bias(
    cosmo, grids, mock_models, function_name,
):
    """Tests that zero IA amplitude skips NLA and galaxy-bias calculations."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids, A_IA=0.0)

    assert len(calls["nla"]) == 0
    assert len(calls["bias"]) == 0


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_default_ia_is_zero(
    cosmo, grids, mock_models, function_name,
):
    """Tests that the default IA amplitude disables the NLA contribution."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids)

    assert len(calls["nla"]) == 0
    assert len(calls["bias"]) == 0


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_adds_nla_to_deltasigma(
    cosmo, grids, mock_models, function_name,
):
    """Tests that NLA is added to the Delta Sigma power spectrum."""
    function = getattr(combinations, function_name)
    result = function(cosmo, **grids, A_IA=1.0, b_g=1.5)

    expected = 8.0 if function_name == "pk2d_hod_with_nla" else 18.0

    assert result["pk_deltasigma"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "function_name,model_name",
    [
        ("pk2d_hod_with_nla", "hod"),
        ("pk2d_hod_baryonified_with_nla", "baryons"),
    ],
)
def test_combination_preserves_covariance_spectra(
    cosmo, grids, mock_models, function_name, model_name,
):
    """Tests that adding NLA preserves the HOD covariance spectra."""
    _, pk_hod, pk_baryons, _ = mock_models

    function = getattr(combinations, function_name)
    result = function(cosmo, **grids, A_IA=1.0, b_g=1.5)

    original = pk_hod if model_name == "hod" else pk_baryons

    assert result["pk_gm"] is original["pk_gm"]
    assert result["pk_gg"] is original["pk_gg"]


@pytest.mark.parametrize(
    "function_name,model_name",
    [
        ("pk2d_hod_with_nla", "hod"),
        ("pk2d_hod_baryonified_with_nla", "baryons"),
    ],
)
def test_combination_does_not_modify_original_spectra(
    cosmo, grids, mock_models, function_name, model_name,
):
    """Tests that adding NLA does not mutate the original HOD dictionary."""
    _, pk_hod, pk_baryons, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids, A_IA=1.0, b_g=1.5)

    original = pk_hod if model_name == "hod" else pk_baryons
    expected = 10.0 if model_name == "hod" else 20.0

    assert original["pk_deltasigma"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_explicit_bias_skips_hod_bias(
    cosmo, grids, mock_models, function_name,
):
    """Tests that explicit galaxy bias bypasses the HOD bias calculation."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids, A_IA=1.0, b_g=2.5)

    assert len(calls["bias"]) == 0
    assert calls["nla"][0][1]["b_g"] == pytest.approx(2.5)


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_automatic_bias(
    cosmo, grids, mock_models, function_name,
):
    """Tests that galaxy bias is computed from the HOD when omitted."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids, A_IA=1.0)

    assert len(calls["bias"]) == 1
    assert calls["nla"][0][1]["b_g"] == pytest.approx(1.8)


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_automatic_bias_forwards_scale_factor(
    cosmo, grids, mock_models, function_name,
):
    """Tests that the bias calculation receives the requested scale factor."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids, A_IA=1.0, a_bias=0.7)

    assert calls["bias"][0][1]["a"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_automatic_bias_default_scale_factor(
    cosmo, grids, mock_models, function_name,
):
    """Tests that galaxy bias is evaluated at unit scale factor by default."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids, A_IA=1.0)

    assert calls["bias"][0][1]["a"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_automatic_bias_forwards_concentration(
    cosmo, grids, concentration, mock_models, function_name,
):
    """Tests that automatic galaxy bias receives the concentration model."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(
        cosmo,
        **grids,
        A_IA=1.0,
        concentration=concentration,
    )

    assert calls["bias"][0][1]["concentration"] is concentration


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_automatic_bias_forwards_hod_parameters(
    cosmo, grids, mock_models, function_name,
):
    """Tests that automatic galaxy bias receives the HOD parameters."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(
        cosmo,
        **grids,
        A_IA=1.0,
        log10Mmin_0=12.5,
        siglnM_0=0.4,
    )

    kwargs = calls["bias"][0][1]

    assert kwargs["log10Mmin_0"] == pytest.approx(12.5)
    assert kwargs["siglnM_0"] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_nla_forwards_cosmology(
    cosmo, grids, mock_models, function_name,
):
    """Tests that the NLA calculation receives the original cosmology."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids, A_IA=1.0, b_g=1.5)

    assert calls["nla"][0][0] is cosmo


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_nla_forwards_grids(
    cosmo, grids, mock_models, function_name,
):
    """Tests that the NLA calculation receives the original grids."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids, A_IA=1.0, b_g=1.5)

    kwargs = calls["nla"][0][1]

    assert kwargs["k_array"] is grids["k_array"]
    assert kwargs["a_array"] is grids["a_array"]


@pytest.mark.parametrize("A_IA", [-2.0, -1.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_nla_forwards_amplitude(
    cosmo, grids, mock_models, function_name, A_IA,
):
    """Tests that positive and negative IA amplitudes are forwarded."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids, A_IA=A_IA, b_g=1.5)

    assert calls["nla"][0][1]["A_IA"] == pytest.approx(A_IA)


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_nla_forwards_normalization(
    cosmo, grids, mock_models, function_name,
):
    """Tests that the NLA normalization is forwarded."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids, A_IA=1.0, C1rhocrit=0.02, b_g=1.5)

    assert calls["nla"][0][1]["C1rhocrit"] == pytest.approx(0.02)


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_nla_default_normalization(
    cosmo, grids, mock_models, function_name,
):
    """Tests that the default NLA normalization is 0.0134."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids, A_IA=1.0, b_g=1.5)

    assert calls["nla"][0][1]["C1rhocrit"] == pytest.approx(0.0134)


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_automatic_bias_forwards_cosmology(
    cosmo, grids, mock_models, function_name,
):
    """Tests that the bias calculation receives the original cosmology."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)
    function(cosmo, **grids, A_IA=1.0)

    assert calls["bias"][0][0] is cosmo


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
@pytest.mark.parametrize("A_IA", [np.nan, np.inf, -np.inf])
def test_combination_rejects_invalid_amplitude(
    cosmo, grids, mock_models, function_name, A_IA,
):
    """Tests that nonfinite IA amplitudes are rejected."""
    function = getattr(combinations, function_name)

    with pytest.raises(ValueError):
        function(cosmo, **grids, A_IA=A_IA)


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_invalid_amplitude_skips_nla(
    cosmo, grids, mock_models, function_name,
):
    """Tests that invalid IA amplitudes never reach the NLA calculation."""
    calls, _, _, _ = mock_models

    function = getattr(combinations, function_name)

    with pytest.raises(ValueError):
        function(cosmo, **grids, A_IA=np.nan)

    assert len(calls["nla"]) == 0
    assert len(calls["bias"]) == 0


@pytest.mark.parametrize(
    "function_name",
    [
        "pk2d_hod_with_nla",
        "pk2d_hod_baryonified_with_nla",
    ],
)
def test_combination_nla_does_not_add_gi_key(
    cosmo, grids, mock_models, function_name,
):
    """Tests that the GI spectrum is not added to the HOD result dictionary."""
    function = getattr(combinations, function_name)
    result = function(cosmo, **grids, A_IA=1.0, b_g=1.5)

    assert "pk_gi" not in result


def test_add_nla_to_hod_zero_amplitude(cosmo, grids, mock_models):
    """Tests that the internal helper returns HOD spectra for zero IA."""
    _, pk_hod, _, _ = mock_models

    result = combinations._add_nla_to_hod(
        cosmo,
        pk_hod,
        **grids,
        A_IA=0.0,
        C1rhocrit=0.0134,
        b_g=None,
        a_bias=1.0,
        concentration=None,
    )

    assert result is pk_hod


def test_add_nla_to_hod_explicit_bias(cosmo, grids, mock_models):
    """Tests that the internal helper uses an explicitly supplied galaxy bias."""
    calls, pk_hod, _, _ = mock_models

    result = combinations._add_nla_to_hod(
        cosmo,
        pk_hod,
        **grids,
        A_IA=1.0,
        C1rhocrit=0.0134,
        b_g=2.0,
        a_bias=1.0,
        concentration=None,
    )

    assert result["pk_deltasigma"] == pytest.approx(8.0)
    assert len(calls["bias"]) == 0
    assert calls["nla"][0][1]["b_g"] == pytest.approx(2.0)


def test_add_nla_to_hod_automatic_bias(cosmo, grids, mock_models):
    """Tests that the internal helper computes galaxy bias when omitted."""
    calls, pk_hod, _, _ = mock_models

    combinations._add_nla_to_hod(
        cosmo,
        pk_hod,
        **grids,
        A_IA=1.0,
        C1rhocrit=0.0134,
        b_g=None,
        a_bias=0.8,
        concentration=None,
    )

    assert len(calls["bias"]) == 1
    assert calls["bias"][0][1]["a"] == pytest.approx(0.8)
    assert calls["nla"][0][1]["b_g"] == pytest.approx(1.8)


def test_add_nla_to_hod_preserves_original_dictionary(
    cosmo, grids, mock_models,
):
    """Tests that the internal helper does not mutate the HOD dictionary."""
    _, pk_hod, _, _ = mock_models

    result = combinations._add_nla_to_hod(
        cosmo,
        pk_hod,
        **grids,
        A_IA=1.0,
        C1rhocrit=0.0134,
        b_g=1.5,
        a_bias=1.0,
        concentration=None,
    )

    assert result is not pk_hod
    assert pk_hod["pk_deltasigma"] == pytest.approx(10.0)
    assert result["pk_gm"] is pk_hod["pk_gm"]
    assert result["pk_gg"] is pk_hod["pk_gg"]


@pytest.mark.parametrize("A_IA", [np.nan, np.inf, -np.inf])
def test_add_nla_to_hod_rejects_invalid_amplitude(
    cosmo, grids, mock_models, A_IA,
):
    """Tests that the internal helper rejects nonfinite IA amplitudes."""
    _, pk_hod, _, _ = mock_models

    with pytest.raises(ValueError):
        combinations._add_nla_to_hod(
            cosmo,
            pk_hod,
            **grids,
            A_IA=A_IA,
            C1rhocrit=0.0134,
            b_g=None,
            a_bias=1.0,
            concentration=None,
        )
