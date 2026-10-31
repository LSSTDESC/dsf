"""Unit tests for ``dsf.covariance.cov_builder``."""

from types import SimpleNamespace

import numpy as np
import pyccl as ccl
import pytest

from dsf.covariance.cov_builder import DeltaSigmaCovarianceBuilder


def test_selected_pairs_returns_stored_pairs_as_integer_tuples():
    """Tests that stored bin pairs are returned as integer tuples."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 2), (1, 3)]

    assert builder.selected_pairs() == [(0, 2), (1, 3)]


def test_selected_pairs_casts_requested_pairs_to_integer_tuples():
    """Tests that requested bin pairs are cast to integer tuples."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 2), (1, 3)]

    actual = builder.selected_pairs([(0.0, 2.0), ("1", "3")])

    assert actual == [(0, 2), (1, 3)]


def test_selected_pairs_rejects_unselected_pairs():
    """Tests that requested pairs must belong to the selected tomography."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 4)]

    with pytest.raises(ValueError, match="not among the selected bin pairs"):
        builder.selected_pairs([(0, 0)])


def test_bin_value_returns_scalar_value_for_all_bins():
    """Tests that scalar bin values are returned unchanged."""
    assert DeltaSigmaCovarianceBuilder.bin_value(1.5, 3) == 1.5


def test_bin_value_selects_sequence_value_by_bin_index():
    """Tests that sequence bin values are selected by index."""
    assert DeltaSigmaCovarianceBuilder.bin_value([1.0, 2.0, 3.0], 1) == 2.0


def test_bin_value_selects_mapping_value_by_bin_index():
    """Tests that mapping bin values are selected by index."""
    assert DeltaSigmaCovarianceBuilder.bin_value({0: 1.0, 2: 3.0}, 2) == 3.0


def test_block_diagonal_from_outputs_preserves_pair_order():
    """Tests that block-diagonal assembly preserves output pair order."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    outputs = {
        (0, 2): {"cov": np.array([[1.0, 0.1], [0.1, 2.0]])},
        (1, 3): {"cov": np.array([[3.0]])},
    }

    pairs, covariance = builder.block_diagonal_from_outputs(outputs, "cov")

    assert pairs == [(0, 2), (1, 3)]
    np.testing.assert_allclose(
        covariance,
        np.array(
            [
                [1.0, 0.1, 0.0],
                [0.1, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ),
    )


def test_correlation_matrix_delegates_to_hankel():
    """Tests that correlation matrices are delegated to the Hankel object."""

    class DummyHankel:
        def correlation_matrix(self, covariance):
            return covariance + 1.0

    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.hankel = DummyHankel()

    actual = builder.correlation_matrix(np.eye(2))

    np.testing.assert_allclose(actual, np.eye(2) + 1.0)


def test_diagonal_error_delegates_to_hankel():
    """Tests that diagonal errors are delegated to the Hankel object."""

    class DummyHankel:
        def diagonal_error(self, covariance):
            return np.sqrt(np.diag(covariance))

    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.hankel = DummyHankel()

    actual = builder.diagonal_error(np.diag([4.0, 9.0]))

    np.testing.assert_allclose(actual, np.array([2.0, 3.0]))


def test_gm_block_diagonal_uses_gm_outputs():
    """Tests that gm block diagonal uses gm covariance outputs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)

    def gm_covariance_for_pairs(bin_pairs=None):
        return {
            (0, 2): {"cov_gm_gm": np.eye(2)},
            (1, 3): {"cov_gm_gm": 2.0 * np.eye(1)},
        }

    builder.gm_covariance_for_pairs = gm_covariance_for_pairs

    pairs, covariance = builder.gm_block_diagonal()

    assert pairs == [(0, 2), (1, 3)]
    np.testing.assert_allclose(covariance, np.diag([1.0, 1.0, 2.0]))


def test_gg_block_diagonal_uses_gg_outputs():
    """Tests that gg block diagonal uses gg covariance outputs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)

    def gg_covariance_for_pairs(bin_pairs=None):
        return {
            (0, 2): {"cov_gg_gg": np.eye(1)},
            (1, 3): {"cov_gg_gg": 3.0 * np.eye(2)},
        }

    builder.gg_covariance_for_pairs = gg_covariance_for_pairs

    pairs, covariance = builder.gg_block_diagonal()

    assert pairs == [(0, 2), (1, 3)]
    np.testing.assert_allclose(covariance, np.diag([1.0, 3.0, 3.0]))


def test_cross_block_diagonal_uses_cross_outputs():
    """Tests that cross block diagonal uses gm-gg covariance outputs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)

    def cross_covariance_for_pairs(bin_pairs=None):
        return {
            (0, 2): {"cov_gm_gg": np.array([[1.0]])},
            (1, 3): {"cov_gm_gg": np.array([[2.0, 0.1], [0.1, 3.0]])},
        }

    builder.cross_covariance_for_pairs = cross_covariance_for_pairs

    pairs, covariance = builder.cross_block_diagonal()

    assert pairs == [(0, 2), (1, 3)]
    np.testing.assert_allclose(
        covariance,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.1],
                [0.0, 0.1, 3.0],
            ]
        ),
    )


def test_joint_block_diagonal_uses_joint_outputs():
    """Tests that joint block diagonal uses joint covariance outputs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)

    def covariance_for_pairs(bin_pairs=None):
        return {
            (0, 2): {"cov_joint": np.eye(2)},
            (1, 3): {"cov_joint": 4.0 * np.eye(1)},
        }

    builder.covariance_for_pairs = covariance_for_pairs

    pairs, covariance = builder.joint_block_diagonal()

    assert pairs == [(0, 2), (1, 3)]
    np.testing.assert_allclose(covariance, np.diag([1.0, 1.0, 4.0]))


def test_gm_covariance_for_pairs_uses_selected_pairs():
    """Tests that gm covariance is evaluated for selected pairs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 2), (1, 3)]

    def gm_covariance_for_pair(lens_bin, source_bin):
        return {"pair": (lens_bin, source_bin)}

    builder.gm_covariance_for_pair = gm_covariance_for_pair

    actual = builder.gm_covariance_for_pairs()

    assert actual == {
        (0, 2): {"pair": (0, 2)},
        (1, 3): {"pair": (1, 3)},
    }


def test_gg_covariance_for_pairs_uses_selected_pairs():
    """Tests that gg covariance is evaluated for selected pairs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 2), (1, 3)]

    def gg_covariance_for_pair(lens_bin, source_bin):
        return {"pair": (lens_bin, source_bin)}

    builder.gg_covariance_for_pair = gg_covariance_for_pair

    actual = builder.gg_covariance_for_pairs()

    assert actual == {
        (0, 2): {"pair": (0, 2)},
        (1, 3): {"pair": (1, 3)},
    }


def test_cross_covariance_for_pairs_uses_selected_pairs():
    """Tests that cross covariance is evaluated for selected pairs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 2), (1, 3)]

    def cross_covariance_for_pair(lens_bin, source_bin):
        return {"pair": (lens_bin, source_bin)}

    builder.cross_covariance_for_pair = cross_covariance_for_pair

    actual = builder.cross_covariance_for_pairs()

    assert actual == {
        (0, 2): {"pair": (0, 2)},
        (1, 3): {"pair": (1, 3)},
    }


def test_covariance_for_pairs_uses_selected_pairs():
    """Tests that full covariance is evaluated for selected pairs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 2), (1, 3)]

    def covariance_for_pair(lens_bin, source_bin):
        return {"pair": (lens_bin, source_bin)}

    builder.covariance_for_pair = covariance_for_pair

    actual = builder.covariance_for_pairs()

    assert actual == {
        (0, 2): {"pair": (0, 2)},
        (1, 3): {"pair": (1, 3)},
    }


def test_pair_ingredients_rejects_unselected_pair():
    """Tests that covariance ingredients reject unselected tomographic pairs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 4)]

    with pytest.raises(ValueError, match="not among the selected bin pairs"):
        builder.pair_ingredients(0, 0)


def test_covariance_for_pair_rejects_unselected_pair():
    """Tests that joint covariance rejects unselected tomographic pairs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 4)]

    with pytest.raises(ValueError, match="not among the selected bin pairs"):
        builder.covariance_for_pair(0, 0)


@pytest.mark.parametrize(
    "method_name",
    [
        "gm_covariance_for_pair",
        "gg_covariance_for_pair",
        "cross_covariance_for_pair",
    ],
)
def test_covariance_blocks_reject_unselected_pair(method_name):
    """Tests that individual covariance blocks reject unselected pairs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 4)]

    method = getattr(builder, method_name)

    with pytest.raises(ValueError, match="not among the selected bin pairs"):
        method(0, 0)


def test_pair_ingredients_accepts_selected_pair(monkeypatch):
    """Tests that selected pairs use their original tomographic indices."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 4)]

    def lens_ingredients(index):
        assert index == 0
        raise RuntimeError("lens ingredients reached")

    monkeypatch.setattr(builder, "lens_ingredients", lens_ingredients)

    with pytest.raises(RuntimeError, match="lens ingredients reached"):
        builder.pair_ingredients(0, 4)


@pytest.fixture
def covariance_builder_inputs(monkeypatch):
    """Minimal constructor inputs that avoid computing a Hankel transform."""
    monkeypatch.setattr(
        DeltaSigmaCovarianceBuilder,
        "hankel_transform",
        lambda self, config: None,
    )
    metadata = {
        "bins": {
            "bin_edges": [0.1, 0.3],
            "indices": [0],
            "truez_summary": {0: {"z_mean": 0.2}},
        },
    }
    result = SimpleNamespace(tomo_meta=metadata, z=np.array([0.1, 0.3]))
    return {
        "cosmo": ccl.CosmologyVanillaLCDM(),
        "tomography": {
            "lens_result": result,
            "source_result": result,
            "bin_pairs": [(0, 0)],
            "lens_population_stats": {},
            "source_population_stats": {},
        },
        "rp_bin_edges": [1.0, 2.0],
        "area_deg2": 100.0,
        "sigma_e": 0.26,
    }


@pytest.mark.parametrize("supply_galaxy_spectra", [False, True])
def test_builder_selects_spectra_and_bias(
    covariance_builder_inputs,
    supply_galaxy_spectra,
):
    """Test that default and custom spectra use the correct bias convention."""
    pk_gg, pk_gm = object(), object()
    spectra = {"pk_gg": pk_gg, "pk_gm": pk_gm} if supply_galaxy_spectra else {}
    bias = {} if supply_galaxy_spectra else {"galaxy_bias": 2.3}

    builder = DeltaSigmaCovarianceBuilder(
        **covariance_builder_inputs, **spectra, **bias
    )

    assert builder.galaxy_bias == (None if supply_galaxy_spectra else 2.3)
    assert builder.pk_gg is (pk_gg if supply_galaxy_spectra else None)
    assert builder.pk_gm is (pk_gm if supply_galaxy_spectra else None)


def test_builder_rejects_supplied_spectra_with_galaxy_bias(covariance_builder_inputs):
    """Explicit galaxy spectra cannot also have a separate bias applied."""
    with pytest.raises(ValueError, match="galaxy_bias cannot be combined"
                                         " with custom power spectra"):
        DeltaSigmaCovarianceBuilder(
            **covariance_builder_inputs,
            pk_gg=object(),
            pk_gm=object(),
            galaxy_bias=2.3,
        )


@pytest.mark.parametrize("galaxy_bias", [1.0, 2.3])
def test_default_lens_spectra_match_nonlinear_matter_power(
    covariance_builder_inputs, galaxy_bias
):
    """Test that default spectra apply galaxy bias exactly once."""
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
    inputs = dict(covariance_builder_inputs, cosmo=cosmo)
    lens_result = inputs["tomography"]["lens_result"]
    lens_result.bins = {0: np.array([5.0, 5.0])}
    lens_result.spec = {
        "sample_properties": {
            "number_density": {
                "n_gal_comoving_h3_mpc3": 1e-4,
            }
        }
    }

    k = np.array([0.05, 0.2, 1.0])
    builder = DeltaSigmaCovarianceBuilder(
        **inputs, galaxy_bias=galaxy_bias, k=k
    )

    ingredients = builder.lens_ingredients(0)

    z = lens_result.z
    h = cosmo["h"]
    power_at_redshifts = np.array(
        [
            ccl.nonlin_matter_power(cosmo, k * h, 1.0 / (1.0 + redshift)) * h**3
            for redshift in z
        ]
    )
    expected_mm = np.trapezoid(
        power_at_redshifts * lens_result.bins[0][:, None],
        z,
        axis=0,
    )

    np.testing.assert_allclose(ingredients["pk_mm"], expected_mm, rtol=1e-12)
    np.testing.assert_allclose(
        ingredients["pk_gm"], galaxy_bias * expected_mm, rtol=1e-12
    )
    np.testing.assert_allclose(
        ingredients["pk_gg"], galaxy_bias**2 * expected_mm, rtol=1e-12
    )


def test_custom_lens_spectra_are_used_without_additional_galaxy_bias(
    covariance_builder_inputs,
):
    """Test that custom galaxy spectra are used without additional bias."""
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
    inputs = dict(covariance_builder_inputs, cosmo=cosmo)
    lens_result = inputs["tomography"]["lens_result"]
    lens_result.bins = {0: np.array([5.0, 5.0])}
    lens_result.spec = {
        "sample_properties": {
            "number_density": {
                "n_gal_comoving_h3_mpc3": 1e-4,
            }
        }
    }

    k = np.array([0.05, 0.2, 1.0])
    pk_mm = cosmo.get_nonlin_power()

    a_arr = np.linspace(0.5, 1.0, 10)
    lk_arr = np.linspace(np.log(0.01), np.log(10.0), 20)

    pk_gg = ccl.Pk2D(
        a_arr=a_arr,
        lk_arr=lk_arr,
        pk_arr=np.full((a_arr.size, lk_arr.size), 7.0),
        is_logp=False,
    )
    pk_gm = ccl.Pk2D(
        a_arr=a_arr,
        lk_arr=lk_arr,
        pk_arr=np.full((a_arr.size, lk_arr.size), 3.0),
        is_logp=False,
    )

    builder = DeltaSigmaCovarianceBuilder(
        **inputs, pk_gg=pk_gg, pk_gm=pk_gm, k=k
    )
    ingredients = builder.lens_ingredients(0)

    z = lens_result.z
    h = cosmo["h"]
    lens_weights = lens_result.bins[0]
    weight_integral = np.trapezoid(lens_weights, z)

    expected_gg = np.full_like(k, 7.0 * h**3 * weight_integral)
    expected_gm = np.full_like(k, 3.0 * h**3 * weight_integral)

    power_at_redshifts = np.array(
        [
            pk_mm(k * h, 1.0 / (1.0 + redshift), cosmo=cosmo) * h**3
            for redshift in z
        ]
    )
    expected_mm = np.trapezoid(
        power_at_redshifts * lens_weights[:, None],
        z,
        axis=0,
    )

    assert builder.galaxy_bias is None
    np.testing.assert_allclose(ingredients["pk_gg"], expected_gg, rtol=1e-12)
    np.testing.assert_allclose(ingredients["pk_gm"], expected_gm, rtol=1e-12)
    np.testing.assert_allclose(ingredients["pk_mm"], expected_mm, rtol=1e-12)


@pytest.mark.parametrize("spectrum_name", ["pk_gg", "pk_gm"])
def test_builder_rejects_incomplete_custom_spectra(
    covariance_builder_inputs, spectrum_name
):
    """Test that custom galaxy spectra must be supplied together."""
    with pytest.raises(ValueError):
        DeltaSigmaCovarianceBuilder(
            **covariance_builder_inputs,
            **{spectrum_name: object()},
        )


def test_custom_spectra_match_equivalent_galaxy_bias(covariance_builder_inputs):
    """Test that equivalent custom spectra reproduce default galaxy bias."""
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
    inputs = dict(covariance_builder_inputs, cosmo=cosmo)
    lens_result = inputs["tomography"]["lens_result"]
    lens_result.bins = {0: np.array([5.0, 5.0])}
    lens_result.spec = {
        "sample_properties": {
            "number_density": {
                "n_gal_comoving_h3_mpc3": 1e-4,
            }
        }
    }

    k = np.array([0.05, 0.2, 1.0])
    galaxy_bias = 2.3
    pk_mm = cosmo.get_nonlin_power()

    pk_gg = galaxy_bias**2 * pk_mm
    pk_gm = galaxy_bias * pk_mm

    default_builder = DeltaSigmaCovarianceBuilder(
        **inputs, galaxy_bias=galaxy_bias, k=k
    )
    custom_builder = DeltaSigmaCovarianceBuilder(
        **inputs, pk_gg=pk_gg, pk_gm=pk_gm, k=k
    )

    default = default_builder.lens_ingredients(0)
    custom = custom_builder.lens_ingredients(0)

    for spectrum in ("pk_gg", "pk_gm", "pk_mm"):
        np.testing.assert_allclose(
            custom[spectrum], default[spectrum], rtol=1e-12
        )


def test_custom_lens_spectra_preserve_scale_dependence(covariance_builder_inputs):
    """Test that custom galaxy spectra retain their scale dependence."""
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
    inputs = dict(covariance_builder_inputs, cosmo=cosmo)
    lens_result = inputs["tomography"]["lens_result"]
    lens_result.bins = {0: np.array([5.0, 5.0])}
    lens_result.spec = {
        "sample_properties": {
            "number_density": {
                "n_gal_comoving_h3_mpc3": 1e-4,
            }
        }
    }

    k = np.array([0.05, 0.2, 1.0])
    h = cosmo["h"]

    a_arr = np.linspace(0.5, 1.0, 10)
    k_arr = np.geomspace(0.01, 10.0, 20)
    lk_arr = np.log(k_arr)

    pk_gg_values = np.tile(7.0 * k_arr**2, (a_arr.size, 1))
    pk_gm_values = np.tile(3.0 * k_arr, (a_arr.size, 1))

    pk_gg = ccl.Pk2D(
        a_arr=a_arr,
        lk_arr=lk_arr,
        pk_arr=pk_gg_values,
        is_logp=False,
    )
    pk_gm = ccl.Pk2D(
        a_arr=a_arr,
        lk_arr=lk_arr,
        pk_arr=pk_gm_values,
        is_logp=False,
    )

    builder = DeltaSigmaCovarianceBuilder(
        **inputs, pk_gg=pk_gg, pk_gm=pk_gm, k=k
    )
    ingredients = builder.lens_ingredients(0)

    z = lens_result.z
    lens_weights = lens_result.bins[0]
    weight_integral = np.trapezoid(lens_weights, z)

    expected_gg = pk_gg(k * h, 1.0, cosmo=cosmo) * h**3 * weight_integral
    expected_gm = pk_gm(k * h, 1.0, cosmo=cosmo) * h**3 * weight_integral

    np.testing.assert_allclose(ingredients["pk_gg"], expected_gg, rtol=1e-12)
    np.testing.assert_allclose(ingredients["pk_gm"], expected_gm, rtol=1e-12)


def test_custom_galaxy_spectra_change_lens_ingredients(covariance_builder_inputs):
    """Test that changing custom P_gg leaves P_gm and P_mm unchanged."""
    cosmo = ccl.CosmologyVanillaLCDM(transfer_function="bbks")
    inputs = dict(covariance_builder_inputs, cosmo=cosmo)
    inputs["tomography"] = {
        **inputs["tomography"],
        "source_population_stats": {
            "density_per_bin": {0: 10.0},
        },
    }
    lens_result = inputs["tomography"]["lens_result"]
    lens_result.bins = {0: np.array([5.0, 5.0])}
    lens_result.spec = {
        "sample_properties": {
            "number_density": {
                "n_gal_comoving_h3_mpc3": 1e-4,
            }
        }
    }

    k = np.geomspace(0.01, 10.0, 100)
    a_arr = np.linspace(0.5, 1.0, 10)
    lk_arr = np.linspace(np.log(0.001), np.log(100.0), 100)

    def make_pk(amplitude):
        return ccl.Pk2D(
            a_arr=a_arr,
            lk_arr=lk_arr,
            pk_arr=np.full((a_arr.size, lk_arr.size), amplitude),
            is_logp=False,
        )

    pk_gm = make_pk(3.0)

    builder_low = DeltaSigmaCovarianceBuilder(
        **inputs, pk_gg=make_pk(5.0), pk_gm=pk_gm, k=k
    )
    builder_high = DeltaSigmaCovarianceBuilder(
        **inputs, pk_gg=make_pk(10.0), pk_gm=pk_gm, k=k
    )

    low = builder_low.lens_ingredients(0)
    high = builder_high.lens_ingredients(0)

    np.testing.assert_allclose(low["pk_gm"], high["pk_gm"])
    np.testing.assert_allclose(low["pk_mm"], high["pk_mm"])
    assert not np.allclose(low["pk_gg"], high["pk_gg"])
