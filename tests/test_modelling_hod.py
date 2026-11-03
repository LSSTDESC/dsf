"""Unit tests for ``dsf.modelling.hod``."""

import numpy as np
import pyccl as ccl
import pytest

import dsf.modelling.hod as hod


@pytest.fixture
def cosmo():
    """Construct a CCL cosmology for HOD calculations."""
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
        "k_array": np.geomspace(1.0e-3, 10.0, 20),
        "a_array": np.linspace(0.5, 1.0, 10),
    }


@pytest.fixture
def capture_pk2d(monkeypatch):
    """Replace the HOD power-spectrum calculation with an argument recorder."""
    calls = []
    spectra = {
        "pk_deltasigma": object(),
        "pk_gm": object(),
        "pk_gg": object(),
    }

    def fake_pk2d(cosmo, **kwargs):
        """Record the supplied power-spectrum arguments."""
        calls.append((cosmo, kwargs))
        return spectra

    monkeypatch.setattr(hod, "_pk2d_hod", fake_pk2d)

    return calls, spectra


@pytest.fixture
def capture_halomod(monkeypatch):
    """Replace CCL halo-model calculations with argument recorders."""
    calls = []
    spectra = []

    class FakePk2D:
        """Represent a halo-model power spectrum."""

        def __init__(self, value):
            self.value = value

        def __call__(self, k, a, cosmo=None):
            return np.full_like(np.asarray(k, dtype=float), self.value)

    def fake_halomod(cosmo, hmc, profile, **kwargs):
        """Record a halo-model calculation."""
        calls.append((cosmo, hmc, profile, kwargs))
        spectrum = FakePk2D(len(calls))
        spectra.append(spectrum)
        return spectrum

    def fake_density_weighted(cosmo, power):
        """Return a marker for the density-weighted spectrum."""
        return ("density_weighted", cosmo, power)

    monkeypatch.setattr(hod.ccl.halos, "halomod_Pk2D", fake_halomod)
    monkeypatch.setattr(
        hod, "density_weighted_power_spectrum", fake_density_weighted
    )

    return calls, spectra


def test_hod_mass_definition():
    """Tests that the HOD model uses the 200c mass definition."""
    assert hod.MASS_DEF == ccl.halos.MassDef200c


def test_hod_default_concentration():
    """Tests that Duffy08 is the default concentration model."""
    assert isinstance(hod.CONCENTRATION, ccl.halos.ConcentrationDuffy08)
    assert hod.CONCENTRATION.mass_def == hod.MASS_DEF


def test_hod_default_mass_function():
    """Tests that Tinker10 is the default halo mass function."""
    assert isinstance(hod.MASS_FUNCTION, ccl.halos.MassFuncTinker10)
    assert hod.MASS_FUNCTION.mass_def == hod.MASS_DEF


def test_hod_default_halo_bias():
    """Tests that Tinker10 is the default halo bias model."""
    assert isinstance(hod.HALO_BIAS, ccl.halos.HaloBiasTinker10)
    assert hod.HALO_BIAS.mass_def == hod.MASS_DEF


def test_hod_default_matter_profile():
    """Tests that the default matter profile uses NFW."""
    assert isinstance(hod.MATTER_PROFILE, ccl.halos.HaloProfileNFW)
    assert hod.MATTER_PROFILE.mass_def == hod.MASS_DEF


def test_hod_default_halo_model_calculator():
    """Tests that the default halo-model calculator is configured."""
    assert isinstance(hod.HM_CALCULATOR, ccl.halos.HMCalculator)
    assert hod.HM_CALCULATOR.mass_def == hod.MASS_DEF


def test_hod_default_two_point_profile():
    """Tests that the HOD two-point profile is configured."""
    assert isinstance(hod.HOD_2PT, ccl.halos.Profile2ptHOD)


def test_pk2d_hod_uses_default_concentration(cosmo, grids, capture_pk2d):
    """Tests that the default concentration is forwarded unchanged."""
    calls, _ = capture_pk2d

    hod.pk2d_hod(cosmo, **grids)

    assert calls[0][1]["concentration"] is hod.CONCENTRATION


def test_pk2d_hod_uses_default_matter_profile(cosmo, grids, capture_pk2d):
    """Tests that the default matter profile is reused."""
    calls, _ = capture_pk2d

    hod.pk2d_hod(cosmo, **grids)

    assert calls[0][1]["matter_profile"] is hod.MATTER_PROFILE


def test_pk2d_hod_accepts_custom_concentration(cosmo, grids, capture_pk2d):
    """Tests that a custom concentration model is forwarded."""
    calls, _ = capture_pk2d
    concentration = ccl.halos.ConcentrationConstant(
        c=5.0, mass_def=hod.MASS_DEF
    )

    hod.pk2d_hod(cosmo, **grids, concentration=concentration)

    assert calls[0][1]["concentration"] is concentration


def test_pk2d_hod_custom_concentration_rebuilds_matter_profile(
    cosmo, grids, capture_pk2d,
):
    """Tests that custom concentrations do not reuse the default matter profile."""
    calls, _ = capture_pk2d
    concentration = ccl.halos.ConcentrationConstant(
        c=5.0, mass_def=hod.MASS_DEF
    )

    hod.pk2d_hod(cosmo, **grids, concentration=concentration)

    assert calls[0][1]["matter_profile"] is None


def test_pk2d_hod_explicit_default_concentration(cosmo, grids, capture_pk2d):
    """Tests that explicitly supplying the default preserves its matter profile."""
    calls, _ = capture_pk2d

    hod.pk2d_hod(cosmo, **grids, concentration=hod.CONCENTRATION)

    assert calls[0][1]["concentration"] is hod.CONCENTRATION
    assert calls[0][1]["matter_profile"] is hod.MATTER_PROFILE


def test_pk2d_hod_forwards_cosmology(cosmo, grids, capture_pk2d):
    """Tests that the original cosmology is forwarded."""
    calls, _ = capture_pk2d

    hod.pk2d_hod(cosmo, **grids)

    assert calls[0][0] is cosmo


def test_pk2d_hod_forwards_grids(cosmo, grids, capture_pk2d):
    """Tests that the input grids are forwarded unchanged."""
    calls, _ = capture_pk2d

    hod.pk2d_hod(cosmo, **grids)

    assert calls[0][1]["k_array"] is grids["k_array"]
    assert calls[0][1]["a_array"] is grids["a_array"]


def test_pk2d_hod_forwards_hod_kwargs(cosmo, grids, capture_pk2d):
    """Tests that HOD parameters are forwarded."""
    calls, _ = capture_pk2d

    hod.pk2d_hod(
        cosmo, **grids, log10Mmin_0=12.5, siglnM_0=0.4
    )

    assert calls[0][1]["log10Mmin_0"] == 12.5
    assert calls[0][1]["siglnM_0"] == 0.4


def test_pk2d_hod_returns_spectra(cosmo, grids, capture_pk2d):
    """Tests that the underlying HOD spectra are returned unchanged."""
    _, spectra = capture_pk2d

    result = hod.pk2d_hod(cosmo, **grids)

    assert result is spectra


def test_pk2d_hod_custom_concentration_does_not_modify_default(
    cosmo, grids, capture_pk2d,
):
    """Tests that custom concentration models leave global defaults unchanged."""
    concentration = ccl.halos.ConcentrationConstant(
        c=5.0, mass_def=hod.MASS_DEF
    )

    hod.pk2d_hod(cosmo, **grids, concentration=concentration)

    assert isinstance(hod.CONCENTRATION, ccl.halos.ConcentrationDuffy08)
    assert isinstance(hod.MATTER_PROFILE, ccl.halos.HaloProfileNFW)


def test_internal_hod_constructs_galaxy_profile(cosmo, grids, capture_halomod):
    """Tests that the internal HOD calculation constructs a galaxy profile."""
    calls, _ = capture_halomod

    hod._pk2d_hod(cosmo, **grids, concentration=hod.CONCENTRATION)

    assert isinstance(calls[0][2], ccl.halos.HaloProfileHOD)
    assert calls[0][2].mass_def == hod.MASS_DEF


def test_internal_hod_constructs_matter_profile(cosmo, grids, capture_halomod):
    """Tests that the internal HOD calculation constructs an NFW profile."""
    calls, _ = capture_halomod

    hod._pk2d_hod(cosmo, **grids, concentration=hod.CONCENTRATION)

    assert isinstance(
        calls[0][3]["prof2"], ccl.halos.HaloProfileNFW
    )


def test_internal_hod_uses_supplied_matter_profile(
    cosmo, grids, capture_halomod,
):
    """Tests that a supplied matter profile is forwarded unchanged."""
    calls, _ = capture_halomod
    matter_profile = ccl.halos.HaloProfileNFW(
        mass_def=hod.MASS_DEF,
        concentration=hod.CONCENTRATION,
        fourier_analytic=True,
    )

    hod._pk2d_hod(
        cosmo,
        **grids,
        concentration=hod.CONCENTRATION,
        matter_profile=matter_profile,
    )

    assert calls[0][3]["prof2"] is matter_profile


def test_internal_hod_forwards_cosmology(cosmo, grids, capture_halomod):
    """Tests that both halo-model calculations use the original cosmology."""
    calls, _ = capture_halomod

    hod._pk2d_hod(cosmo, **grids, concentration=hod.CONCENTRATION)

    assert len(calls) == 2
    assert all(call[0] is cosmo for call in calls)


def test_internal_hod_uses_halo_model_calculator(
    cosmo, grids, capture_halomod,
):
    """Tests that both halo-model calculations use the configured calculator."""
    calls, _ = capture_halomod

    hod._pk2d_hod(cosmo, **grids, concentration=hod.CONCENTRATION)

    assert all(call[1] is hod.HM_CALCULATOR for call in calls)


def test_internal_hod_uses_hod_two_point_profile(
    cosmo, grids, capture_halomod,
):
    """Tests that galaxy-galaxy spectra use the HOD two-point profile."""
    calls, _ = capture_halomod

    hod._pk2d_hod(cosmo, **grids, concentration=hod.CONCENTRATION)

    assert calls[1][3]["prof_2pt"] is hod.HOD_2PT


def test_internal_hod_uses_nonlinear_power(cosmo, grids, capture_halomod):
    """Tests that both halo-model calculations receive nonlinear matter power."""
    calls, _ = capture_halomod

    hod._pk2d_hod(cosmo, **grids, concentration=hod.CONCENTRATION)

    nonlinear_power = cosmo.get_nonlin_power()

    assert all(
        call[3]["p_of_k_a"] is nonlinear_power for call in calls
    )


def test_internal_hod_uses_log_wavenumbers(cosmo, grids, capture_halomod):
    """Tests that halo-model calculations receive logarithmic wavenumbers."""
    calls, _ = capture_halomod

    hod._pk2d_hod(cosmo, **grids, concentration=hod.CONCENTRATION)

    expected = np.log(grids["k_array"])

    for call in calls:
        np.testing.assert_allclose(call[3]["lk_arr"], expected)


def test_internal_hod_uses_scale_factor_grid(cosmo, grids, capture_halomod):
    """Tests that both halo-model calculations receive the scale-factor grid."""
    calls, _ = capture_halomod

    hod._pk2d_hod(cosmo, **grids, concentration=hod.CONCENTRATION)

    for call in calls:
        np.testing.assert_allclose(call[3]["a_arr"], grids["a_array"])


def test_internal_hod_returns_expected_keys(cosmo, grids, capture_halomod):
    """Tests that the internal HOD calculation returns all three spectra."""
    hod_spectra = hod._pk2d_hod(
        cosmo, **grids, concentration=hod.CONCENTRATION
    )

    assert set(hod_spectra) == {"pk_deltasigma", "pk_gm", "pk_gg"}


def test_internal_hod_returns_correct_spectra(cosmo, grids, capture_halomod):
    """Tests that the returned spectra match the halo-model calculations."""
    _, spectra = capture_halomod

    result = hod._pk2d_hod(
        cosmo, **grids, concentration=hod.CONCENTRATION
    )

    assert result["pk_gm"] is spectra[0]
    assert result["pk_gg"] is spectra[1]


def test_internal_hod_constructs_density_weighted_spectrum(
    cosmo, grids, capture_halomod,
):
    """Tests that the density-weighted spectrum is constructed."""
    result = hod._pk2d_hod(
        cosmo, **grids, concentration=hod.CONCENTRATION
    )

    assert result["pk_deltasigma"][0] == "density_weighted"
    assert result["pk_deltasigma"][1] is cosmo


def test_internal_hod_density_weighted_callback(
    cosmo, grids, capture_halomod,
):
    """Tests that the density-weighted callback evaluates galaxy-matter power."""
    result = hod._pk2d_hod(
        cosmo, **grids, concentration=hod.CONCENTRATION
    )

    callback = result["pk_deltasigma"][2]
    masses = np.array([0.1, 1.0, 10.0])

    np.testing.assert_allclose(
        callback(cosmo, masses, 0.8),
        np.full(3, 1.0),
    )


def test_internal_hod_forwards_hod_parameters(cosmo, grids, capture_halomod):
    """Tests that HOD parameters are passed to the galaxy profile."""
    calls, _ = capture_halomod

    hod._pk2d_hod(
        cosmo,
        **grids,
        concentration=hod.CONCENTRATION,
        log10Mmin_0=12.5,
        siglnM_0=0.4,
    )

    galaxy_profile = calls[0][2]

    assert isinstance(galaxy_profile, ccl.halos.HaloProfileHOD)
    assert galaxy_profile.log10Mmin_0 == pytest.approx(12.5)
    assert galaxy_profile.siglnM_0 == pytest.approx(0.4)


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
def test_internal_hod_rejects_invalid_wavenumbers(cosmo, grids, k_array):
    """Tests that invalid wavenumber grids are rejected."""
    with pytest.raises((ValueError, TypeError)):
        hod._pk2d_hod(
            cosmo,
            k_array=k_array,
            a_array=grids["a_array"],
            concentration=hod.CONCENTRATION,
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
def test_internal_hod_rejects_invalid_scale_factor_grids(
    cosmo, grids, a_array,
):
    """Tests that invalid scale-factor grids are rejected."""
    with pytest.raises((ValueError, TypeError)):
        hod._pk2d_hod(
            cosmo,
            k_array=grids["k_array"],
            a_array=a_array,
            concentration=hod.CONCENTRATION,
        )


@pytest.mark.parametrize("a", [0.0, -0.1, 1.1, np.nan, np.inf, -np.inf])
def test_hod_galaxy_bias_rejects_invalid_scale_factors(cosmo, a):
    """Tests that invalid scale factors are rejected."""
    with pytest.raises((ValueError, TypeError)):
        hod.hod_galaxy_bias(cosmo, a=a)


def test_hod_galaxy_bias_uses_default_concentration(monkeypatch, cosmo):
    """Tests that galaxy bias uses the default concentration model."""
    captured = {}

    class FakeHOD:
        """Represent an HOD profile with fixed occupations."""

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def get_normalization(self, cosmo, a, hmc=None):
            return 2.0

        def _Nc(self, mass, a):
            return np.ones_like(mass)

        def _Ns(self, mass, a):
            return np.ones_like(mass)

    monkeypatch.setattr(hod.ccl.halos, "HaloProfileHOD", FakeHOD)
    monkeypatch.setattr(
        hod.HM_CALCULATOR,
        "integrate_over_massfunc",
        lambda func, cosmo, a: 4.0,
    )

    hod.hod_galaxy_bias(cosmo, a=0.8)

    assert captured["concentration"] is hod.CONCENTRATION


def test_hod_galaxy_bias_accepts_custom_concentration(monkeypatch, cosmo):
    """Tests that galaxy bias accepts a custom concentration model."""
    captured = {}
    concentration = ccl.halos.ConcentrationConstant(
        c=5.0, mass_def=hod.MASS_DEF
    )

    class FakeHOD:
        """Represent an HOD profile with fixed occupations."""

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def get_normalization(self, cosmo, a, hmc=None):
            return 2.0

        def _Nc(self, mass, a):
            return np.ones_like(mass)

        def _Ns(self, mass, a):
            return np.ones_like(mass)

    monkeypatch.setattr(hod.ccl.halos, "HaloProfileHOD", FakeHOD)
    monkeypatch.setattr(
        hod.HM_CALCULATOR,
        "integrate_over_massfunc",
        lambda func, cosmo, a: 4.0,
    )

    hod.hod_galaxy_bias(
        cosmo, a=0.8, concentration=concentration
    )

    assert captured["concentration"] is concentration


@pytest.mark.parametrize("normalization", [0.0, -1.0, np.nan, np.inf])
def test_hod_galaxy_bias_rejects_invalid_normalization(
    monkeypatch, cosmo, normalization,
):
    """Tests that invalid galaxy normalizations are rejected."""

    class FakeHOD:
        """Represent an HOD profile with a configurable normalization."""

        def __init__(self, **kwargs):
            pass

        def get_normalization(self, cosmo, a, hmc=None):
            return normalization

    monkeypatch.setattr(hod.ccl.halos, "HaloProfileHOD", FakeHOD)

    with pytest.raises(ValueError, match="normalization"):
        hod.hod_galaxy_bias(cosmo, a=0.8)


def test_hod_galaxy_bias_computes_weighted_bias(monkeypatch, cosmo):
    """Tests the occupation-weighted galaxy-bias calculation."""
    masses = np.array([1.0e12, 1.0e13, 1.0e14])
    occupations = np.array([1.0, 2.0, 3.0])
    halo_bias = np.array([1.0, 1.5, 2.0])

    class FakeHOD:
        """Represent an HOD profile with specified occupations."""

        def __init__(self, **kwargs):
            pass

        def get_normalization(self, cosmo, a, hmc=None):
            return 6.0

        def _Nc(self, mass, a):
            return occupations / 2.0

        def _Ns(self, mass, a):
            return occupations / 2.0

    monkeypatch.setattr(hod.ccl.halos, "HaloProfileHOD", FakeHOD)
    monkeypatch.setattr(
        hod.HM_CALCULATOR,
        "integrate_over_massfunc",
        lambda func, cosmo, a: np.sum(func(masses)),
    )
    monkeypatch.setattr(
        hod.HM_CALCULATOR,
        "halo_bias",
        lambda cosmo, mass, a: halo_bias,
    )

    result = hod.hod_galaxy_bias(cosmo, a=0.8)
    expected = np.sum(halo_bias * occupations) / 6.0

    assert result == pytest.approx(expected)


def test_hod_galaxy_bias_returns_float(monkeypatch, cosmo):
    """Tests that the effective galaxy bias is returned as a float."""

    class FakeHOD:
        """Represent an HOD profile with constant occupations."""

        def __init__(self, **kwargs):
            pass

        def get_normalization(self, cosmo, a, hmc=None):
            return 2.0

        def _Nc(self, mass, a):
            return np.ones_like(mass)

        def _Ns(self, mass, a):
            return np.ones_like(mass)

    monkeypatch.setattr(hod.ccl.halos, "HaloProfileHOD", FakeHOD)
    monkeypatch.setattr(
        hod.HM_CALCULATOR,
        "integrate_over_massfunc",
        lambda func, cosmo, a: np.float64(4.0),
    )

    result = hod.hod_galaxy_bias(cosmo, a=0.8)

    assert isinstance(result, float)
