"""Unit tests for ``dsf.data_vector.tomography``."""

from types import SimpleNamespace

import numpy as np
import pytest

from dsf.tomography import tomo_builder
from dsf.tomography.tomo_builder import TomographyBuilder


class DummyNZTomography:
    """Small stand-in for Binny's NZTomography builder."""

    instances = []
    next_result = None

    def __init__(self):
        """Initialize call logs and default return values."""
        self.build_survey_bins_calls = []
        self.shape_stats_calls = []
        self.population_stats_calls = []
        self.result = DummyNZTomography.next_result
        self.shape_stats_result = None
        self.population_stats_result = None

        DummyNZTomography.instances.append(self)

    def build_survey_bins(
        self,
        survey,
        *,
        role,
        year,
        scenario,
        sample,
        overrides,
        include_tomo_metadata,
    ):
        """Record Binny build inputs and return a configured result."""
        self.build_survey_bins_calls.append(
            {
                "survey": survey,
                "role": role,
                "year": year,
                "scenario": scenario,
                "sample": sample,
                "overrides": overrides,
                "include_tomo_metadata": include_tomo_metadata,
            }
        )

        if self.result is None:
            raise RuntimeError("DummyNZTomography result was not configured.")

        return self.result

    def shape_stats(self, *, center_method, decimal_places):
        """Record shape-stat inputs and return configured bin centers."""
        self.shape_stats_calls.append(
            {
                "center_method": center_method,
                "decimal_places": decimal_places,
            }
        )

        if self.shape_stats_result is None:
            raise RuntimeError("DummyNZTomography shape stats were not configured.")

        return self.shape_stats_result

    def population_stats(self, **kwargs):
        """Record population-stat inputs and return configured stats."""
        self.population_stats_calls.append(kwargs)

        if self.population_stats_result is None:
            raise RuntimeError("DummyNZTomography population stats were not configured.")

        return self.population_stats_result


def make_result(
    *,
    z,
    bins,
    number_density=None,
    tomo_meta=None,
):
    """Return a small Binny-like tomography result object."""
    sample_properties = {}

    if number_density is not None:
        sample_properties["number_density"] = number_density

    return SimpleNamespace(
        z=np.asarray(z, dtype=float),
        bins={key: np.asarray(value, dtype=float) for key, value in bins.items()},
        spec={"sample_properties": sample_properties},
        tomo_meta={} if tomo_meta is None else tomo_meta,
    )


@pytest.fixture
def patch_nz_tomography(monkeypatch):
    """Patch Binny's NZTomography with a lightweight dummy builder."""
    DummyNZTomography.instances = []
    DummyNZTomography.next_result = None
    monkeypatch.setattr(tomo_builder, "NZTomography", DummyNZTomography)
    return DummyNZTomography.instances


def configure_two_builders(instances, lens_result, source_result):
    """Configure the two dummy Binny builders created by prepare_bins."""
    lens_builder, source_builder = instances

    lens_builder.result = lens_result
    lens_builder.shape_stats_result = {"centers": {0: 0.4, 1: 0.8}}
    lens_builder.population_stats_result = {"lens": "population"}

    source_builder.result = source_result
    source_builder.shape_stats_result = {"centers": {0: 0.6, 1: 1.2}}
    source_builder.population_stats_result = {"source": "population"}

    return lens_builder, source_builder


def test_tomography_builder_stores_configuration():
    """Tests that the tomography builder stores user configuration."""
    builder = TomographyBuilder(
        lens_survey="desi",
        source_survey="lsst",
        lens_sample="lrg",
        source_sample="gold",
        lens_year="2025",
        source_year="1",
        lens_scenario="lens-scenario",
        source_scenario="source-scenario",
        lens_role="foreground",
        source_role="background",
        overlap_threshold=0.25,
        source_behind_lens=False,
        center_method="median",
        decimal_places=3,
        shared_overrides={"z_grid": {"n": 50}},
        lens_overrides={"lens": True},
        source_overrides={"source": True},
    )

    assert builder.lens_survey == "desi"
    assert builder.source_survey == "lsst"
    assert builder.lens_sample == "lrg"
    assert builder.source_sample == "gold"
    assert builder.lens_year == "2025"
    assert builder.source_year == "1"
    assert builder.lens_scenario == "lens-scenario"
    assert builder.source_scenario == "source-scenario"
    assert builder.lens_role == "foreground"
    assert builder.source_role == "background"
    assert builder.overlap_threshold == 0.25
    assert builder.source_behind_lens is False
    assert builder.center_method == "median"
    assert builder.decimal_places == 3
    assert builder.shared_overrides == {"z_grid": {"n": 50}}
    assert builder.lens_overrides == {"lens": True}
    assert builder.source_overrides == {"source": True}


def test_sample_passes_expected_arguments_to_binny(patch_nz_tomography):
    """Tests that one sample is built with the expected Binny arguments."""
    result = make_result(
        z=np.array([0.0, 0.5, 1.0]),
        bins={0: np.array([1.0, 0.0, 0.0])},
    )
    DummyNZTomography.next_result = result

    builder, sample_result = TomographyBuilder._sample(
        survey="desi",
        role="lens",
        sample="lrg",
        year="1",
        scenario="baseline",
        shared_overrides={"shared": 1},
        sample_overrides={"sample": 2},
    )

    assert sample_result is result
    assert builder is patch_nz_tomography[0]
    assert builder.build_survey_bins_calls == [
        {
            "survey": "desi",
            "role": "lens",
            "year": "1",
            "scenario": "baseline",
            "sample": "lrg",
            "overrides": {"shared": 1, "sample": 2},
            "include_tomo_metadata": True,
        }
    ]
