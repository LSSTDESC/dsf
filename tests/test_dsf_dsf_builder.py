"""Unit tests for ``src.dsf.deta_sigma_forecast_builder``."""

from types import SimpleNamespace

import numpy as np
import pytest

import src.dsf.delta_sigma_forecast_builder as forecast_module
from src.dsf.delta_sigma_forecast_builder import DeltaSigmaForecastBuilder


def dummy_pk2d_func(*args, **kwargs):
    """Return a dummy object in place of a Pk2D profile."""
    return object()


def make_tomography():
    """Return deterministic tomography products for forecast-builder tests."""
    lens_result = SimpleNamespace(
        z=np.array([0.1, 0.2, 0.3, 0.4], dtype=float),
        bins={
            0: np.array([1.0, 2.0, 2.0, 1.0], dtype=float),
            1: np.array([0.5, 1.0, 1.0, 0.5], dtype=float),
        },
        tomo_meta={
            "bins": {
                "bin_edges": np.array([0.1, 0.3, 0.5], dtype=float),
            },
        },
    )

    source_result = SimpleNamespace(
        z=np.array([0.4, 0.6, 0.8], dtype=float),
        bins={
            2: np.array([1.0, 1.0, 0.0], dtype=float),
            3: np.array([0.0, 1.0, 1.0], dtype=float),
            4: np.array([0.5, 1.0, 0.5], dtype=float),
        },
        tomo_meta={},
    )

    return {
        "lens_result": lens_result,
        "source_result": source_result,
        "lens_population_stats": {"n_eff": 1.0},
        "source_population_stats": {"n_eff": 10.0},
        "bin_pairs": [(0, 2), (1, 3), (0, 4)],
        "lens_bin_centers": {0: 0.2, 1: 0.4},
        "source_bin_centers": {2: 0.5, 3: 0.7, 4: 0.9},
        "lens_shape_stats": {"ok": True},
        "source_shape_stats": {"ok": True},
    }


class DummyTomographyBuilder:
    """Minimal tomography-builder stand-in."""

    instances = []

    def __init__(self, **kwargs):
        """Store initialization keyword arguments."""
        self.kwargs = kwargs
        self.__class__.instances.append(self)

    def prepare_bins(self):
        """Return deterministic tomography products."""
        return make_tomography()


class DummyDeltaSigmaCovarianceBuilder:
    """Minimal covariance-builder stand-in."""

    instances = []
    covariance_pairs = None

    def __init__(self, **kwargs):
        """Store initialization keyword arguments and gm calls."""
        self.kwargs = kwargs
        self.gm_calls = []
        self.__class__.instances.append(self)

    def gm_block_diagonal(self, bin_pairs):
        """Return a deterministic block-diagonal gm covariance."""
        self.gm_calls.append(list(bin_pairs))

        if self.__class__.covariance_pairs is None:
            cov_pairs = list(bin_pairs)
        else:
            cov_pairs = list(self.__class__.covariance_pairs)

        size = 3 * len(cov_pairs)
        return cov_pairs, np.eye(size, dtype=float)


class DummyDeltaSigmaCalculator:
    """Minimal DeltaSigma calculator stand-in."""

    instances = []

    def __init__(self, pk2d_func):
        """Store the supplied Pk2D function."""
        self.pk2d_func = pk2d_func
        self.calls = []
        self.__class__.instances.append(self)

    def delta_sigma_lens_bin(
        self,
        r_mpc,
        *,
        lens_dndz,
        cosmo,
        pk2d_kwargs,
        z_min,
        z_max,
        trim_edge_points,
    ):
        """Return a deterministic DeltaSigma block for one lens bin."""
        self.calls.append(
            {
                "r_mpc": np.asarray(r_mpc, dtype=float).copy(),
                "lens_dndz": (
                    np.asarray(lens_dndz[0], dtype=float).copy(),
                    np.asarray(lens_dndz[1], dtype=float).copy(),
                ),
                "cosmo": cosmo,
                "pk2d_kwargs": dict(pk2d_kwargs),
                "z_min": float(z_min),
                "z_max": float(z_max),
                "trim_edge_points": int(trim_edge_points),
            }
        )

        lens_id = int(round(10.0 * float(z_min)))
        amplitude = float(pk2d_kwargs.get("amplitude", 0.0))

        return np.full(np.size(r_mpc), lens_id + amplitude, dtype=float)


@pytest.fixture
def patched_dependencies(monkeypatch):
    """Patch forecast-builder dependencies with deterministic dummies."""
    DummyTomographyBuilder.instances = []
    DummyDeltaSigmaCovarianceBuilder.instances = []
    DummyDeltaSigmaCovarianceBuilder.covariance_pairs = None
    DummyDeltaSigmaCalculator.instances = []

    monkeypatch.setattr(
        forecast_module,
        "TomographyBuilder",
        DummyTomographyBuilder,
    )
    monkeypatch.setattr(
        forecast_module,
        "DeltaSigmaCovarianceBuilder",
        DummyDeltaSigmaCovarianceBuilder,
    )
    monkeypatch.setattr(
        forecast_module,
        "DeltaSigmaCalculator",
        DummyDeltaSigmaCalculator,
    )


def make_builder(**overrides):
    """Return a forecast builder with deterministic default inputs."""
    kwargs = {
        "cosmo": {"h": 0.7},
        "pk2d_func": dummy_pk2d_func,
        "theta0": np.array([1.0, 2.0], dtype=float),
        "parameter_names": ["alpha", "beta"],
        "r": np.array([7.0, 14.0, 21.0], dtype=float),
        "rp_bin_edges": np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        "area_deg2": 100.0,
        "verbose": False,
    }
    kwargs.update(overrides)
    return DeltaSigmaForecastBuilder(**kwargs)


def test_prepare_builds_computed_covariance_and_stacked_data_vector(
    patched_dependencies,
):
    """Tests that prepare builds tomography, covariance, and the stacked data vector."""
    builder = make_builder(trim_edge_points=1)

    forecast = builder.prepare()

    expected_pairs = [(0, 2), (1, 3), (0, 4)]
    expected_data_vector = np.array(
        [
            1.0,
            1.0,
            1.0,
            3.0,
            3.0,
            3.0,
            1.0,
            1.0,
            1.0,
        ],
        dtype=float,
    )

    assert forecast["bin_pairs"] == expected_pairs
    assert forecast["cov_pairs"] == expected_pairs

    np.testing.assert_allclose(forecast["data_vector"], expected_data_vector)
    np.testing.assert_allclose(
        forecast["fiducial_data_vector"],
        expected_data_vector,
    )
    np.testing.assert_allclose(forecast["cov"], np.eye(9))

    cov_builder = DummyDeltaSigmaCovarianceBuilder.instances[0]
    assert cov_builder.gm_calls == [expected_pairs]

    calculator = DummyDeltaSigmaCalculator.instances[0]
    assert len(calculator.calls) == 2

    np.testing.assert_allclose(calculator.calls[0]["r_mpc"], [10.0, 20.0, 30.0])
    assert calculator.calls[0]["z_min"] == pytest.approx(0.1)
    assert calculator.calls[0]["z_max"] == pytest.approx(0.3)
    assert calculator.calls[0]["trim_edge_points"] == 1

    np.testing.assert_allclose(calculator.calls[1]["r_mpc"], [10.0, 20.0, 30.0])
    assert calculator.calls[1]["z_min"] == pytest.approx(0.3)
    assert calculator.calls[1]["z_max"] == pytest.approx(0.5)
    assert calculator.calls[1]["trim_edge_points"] == 1


def test_prepare_reuses_supplied_covariance_without_computing_covariance(
    patched_dependencies,
):
    """Tests that prepare reuses a supplied covariance matrix."""
    builder = make_builder()
    supplied_covariance = 2.0 * np.eye(9, dtype=float)

    forecast = builder.prepare(covariance=supplied_covariance)

    np.testing.assert_allclose(forecast["cov"], supplied_covariance)
    assert forecast["cov_pairs"] == [(0, 2), (1, 3), (0, 4)]

    cov_builder = DummyDeltaSigmaCovarianceBuilder.instances[0]
    assert cov_builder.gm_calls == []


@pytest.mark.parametrize(
    ("bad_covariance", "message"),
    [
        (np.ones(9, dtype=float), "2D matrix"),
        (np.ones((8, 9), dtype=float), "square"),
        (
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.nan, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            ),
            "non-finite",
        ),
    ],
)
def test_prepare_rejects_invalid_supplied_covariance(
    patched_dependencies,
    bad_covariance,
    message,
):
    """Tests that prepare rejects invalid supplied covariance matrices."""
    builder = make_builder()

    with pytest.raises(ValueError, match=message):
        builder.prepare(covariance=bad_covariance)


def test_prepare_rejects_covariance_pair_order_mismatch(patched_dependencies):
    """Tests that prepare rejects covariance pair ordering mismatches."""
    DummyDeltaSigmaCovarianceBuilder.covariance_pairs = [(0, 2), (0, 4), (1, 3)]

    builder = make_builder()

    with pytest.raises(RuntimeError, match="ordering does not match"):
        builder.prepare()


def test_forecast_uses_cached_products(patched_dependencies):
    """Tests that forecast returns cached products after the first preparation."""
    builder = make_builder()

    first = builder.forecast()
    second = builder.forecast()

    assert second is first
    assert len(DummyTomographyBuilder.instances) == 1
    assert len(DummyDeltaSigmaCovarianceBuilder.instances) == 1


def test_forecast_rejects_new_covariance_after_cache_exists(patched_dependencies):
    """Tests that forecast rejects a new covariance once products are cached."""
    builder = make_builder()
    builder.forecast()

    with pytest.raises(RuntimeError, match="already cached"):
        builder.forecast(covariance=np.eye(9, dtype=float))


def test_forecast_inputs_returns_model_theta0_and_covariance(patched_dependencies):
    """Tests that forecast_inputs returns the model, theta0, and covariance."""
    builder = make_builder()

    model, theta0, covariance = builder.forecast_inputs()

    assert model.__self__ is builder
    assert model.__func__ is DeltaSigmaForecastBuilder.model
    np.testing.assert_allclose(theta0, [1.0, 2.0])
    np.testing.assert_allclose(covariance, np.eye(9))


def test_context_returns_shallow_copy_of_cached_context(patched_dependencies):
    """Tests that context returns a shallow copy of the cached context."""
    builder = make_builder()
    forecast = builder.prepare()

    context = builder.context()

    assert context is not forecast["context"]

    context["observable"] = "changed"

    assert forecast["context"]["observable"] == "delta_sigma"


def test_model_uses_theta_mapper_pk2d_kwargs(patched_dependencies):
    """Tests that model passes theta-mapped pk2d keyword arguments."""

    def theta_mapper(theta, context):
        """Return profile keyword arguments from the parameter vector."""
        return {
            "pk2d_kwargs": {
                "amplitude": float(theta[0]),
            },
        }

    builder = make_builder(
        theta0=np.array([2.0], dtype=float),
        parameter_names=["amplitude"],
        theta_mapper=theta_mapper,
    )

    context = builder.prepare()["context"]
    data_vector = builder.model(np.array([5.0], dtype=float), context=context)

    expected = np.array(
        [
            6.0,
            6.0,
            6.0,
            8.0,
            8.0,
            8.0,
            6.0,
            6.0,
            6.0,
        ],
        dtype=float,
    )

    np.testing.assert_allclose(data_vector, expected)

    calculator = DummyDeltaSigmaCalculator.instances[0]
    assert calculator.calls[-1]["pk2d_kwargs"] == {"amplitude": 5.0}


def test_model_inputs_defaults_missing_mapper_outputs(patched_dependencies):
    """Tests that _model_inputs fills missing mapper outputs with defaults."""

    def theta_mapper(theta, context):
        """Return an intentionally incomplete mapper output."""
        context["mutated"] = True
        return {}

    builder = make_builder(theta_mapper=theta_mapper)
    context = {"tomography": make_tomography()}

    model_inputs = builder._model_inputs(builder.theta0, context)

    assert model_inputs["cosmo"] is builder.cosmo
    assert model_inputs["pk2d_kwargs"] == {}
    assert "mutated" not in context


def test_model_rejects_non_mapping_pk2d_kwargs(patched_dependencies):
    """Tests that model rejects non-mapping pk2d keyword arguments."""

    def theta_mapper(theta, context):
        """Return an invalid pk2d_kwargs value."""
        return {
            "pk2d_kwargs": 3.0,
        }

    builder = make_builder(theta_mapper=theta_mapper)

    with pytest.raises(TypeError, match="pk2d_kwargs"):
        builder.model(builder.theta0, context={"tomography": make_tomography()})


def test_init_rejects_non_gm_covariance_kind(patched_dependencies):
    """Tests that initialization rejects unsupported covariance kinds."""
    with pytest.raises(NotImplementedError, match="covariance_kind='gm'"):
        make_builder(covariance_kind="joint")


def test_init_rejects_unsupported_observable(patched_dependencies):
    """Tests that initialization rejects unsupported observables."""
    with pytest.raises(NotImplementedError, match="observable='gamma_t'"):
        make_builder(observable="gamma_t")
