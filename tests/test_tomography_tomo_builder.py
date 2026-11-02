"""Unit tests for ``dsf.tomography.tomo_builder``."""

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
        self.result = DummyNZTomography.next_result
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
        self.build_survey_bins_calls.append({
            "survey": survey,
            "role": role,
            "year": year,
            "scenario": scenario,
            "sample": sample,
            "overrides": overrides,
            "include_tomo_metadata": include_tomo_metadata,
        })

        if self.result is None:
            raise RuntimeError("DummyNZTomography result was not configured.")

        return self.result


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
        bins={
            key: np.asarray(value, dtype=float) for key, value in bins.items()
        },
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


def test_tomography_builder_stores_configuration():
    """Tests that the tomography builder stores user configuration."""
    lens = {
        "survey": "desi",
        "sample": "lrg",
        "year": "2025",
        "scenario": "lens-scenario",
        "role": "foreground",
        "overrides": {"lens": True},
    }
    source = {
        "survey": "lsst",
        "sample": "gold",
        "year": "1",
        "scenario": "source-scenario",
        "role": "background",
        "overrides": {"source": True},
    }

    builder = TomographyBuilder(
        lens=lens,
        source=source,
        overlap_threshold=0.25,
        source_behind_lens=False,
    )

    assert builder.lens == lens
    assert builder.source == source
    assert builder.overlap_threshold == 0.25
    assert builder.source_behind_lens is False


def test_sample_passes_expected_arguments_to_binny(patch_nz_tomography):
    """Tests that one sample is built with the expected Binny arguments."""
    result = make_result(
        z=np.array([0.0, 0.5, 1.0]),
        bins={0: np.array([1.0, 0.0, 0.0])},
    )
    DummyNZTomography.next_result = result

    builder, sample_result = TomographyBuilder._build_sample({
        "survey": "desi",
        "role": "lens",
        "sample": "lrg",
        "year": "1",
        "scenario": "baseline",
        "overrides": {"sample": 2},
    })

    assert sample_result is result
    assert builder is patch_nz_tomography[0]
    assert builder.build_survey_bins_calls == [
        {
            "survey": "desi",
            "role": "lens",
            "year": "1",
            "scenario": "baseline",
            "sample": "lrg",
            "overrides": {"sample": 2},
            "include_tomo_metadata": True,
        }
    ]


def test_default_tomography_configuration():
    """Tests the default lens and source configurations."""
    builder = TomographyBuilder()

    assert builder.lens == {"survey": "desi", "sample": "lrg"}
    assert builder.source == {"survey": "lsst"}
    assert builder.overlap_threshold == 0.10
    assert builder.source_behind_lens is True


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_overlap_threshold(threshold):
    """Tests that invalid overlap thresholds are rejected."""
    with pytest.raises(ValueError, match="overlap_threshold"):
        TomographyBuilder(overlap_threshold=threshold)


def test_pair_selection_overlap_threshold():
    """Tests that pair selection delegates overlap filtering to Binny."""
    calls = []

    def bin_combo_filter(spec, *, other):
        calls.append((spec, other))
        return [(0, 1)]

    lens = SimpleNamespace(bin_combo_filter=bin_combo_filter)
    source = SimpleNamespace()

    builder = TomographyBuilder(
        overlap_threshold=0.25,
        source_behind_lens=False,
    )

    pairs = builder._selected_pairs(lens, source)

    assert pairs == [(0, 1)]
    assert calls == [
        (
            {
                "topology": {"name": "pairs_cartesian"},
                "filters": [
                    {
                        "name": "overlap_fraction",
                        "threshold": 0.25,
                        "compare": "le",
                    }
                ],
            },
            source,
        )
    ]


def test_pair_selection_source_ordering():
    """Tests that source ordering is included when requested."""
    specs = []

    def bin_combo_filter(spec, *, other):
        specs.append(spec)
        return [(0, 1)]

    lens = SimpleNamespace(bin_combo_filter=bin_combo_filter)
    source = SimpleNamespace()

    builder = TomographyBuilder(source_behind_lens=True)
    pairs = builder._selected_pairs(lens, source)

    assert pairs == [(0, 1)]
    assert specs[0]["filters"] == [
        {
            "name": "overlap_fraction",
            "threshold": 0.10,
            "compare": "le",
        },
        {
            "name": "score_relation",
            "score": "mean",
            "relation": "gt",
        },
    ]


def test_prepare_bins_no_pairs(monkeypatch):
    """Tests that empty lens-source pair selection raises an error."""
    result = make_result(
        z=np.array([0.0, 0.5, 1.0]),
        bins={0: np.array([0.0, 1.0, 0.0])},
    )

    result.shape_stats = lambda **kwargs: {"centers": {0: 0.5}}
    result.population_stats = lambda **kwargs: {"density_per_bin": {0: 1.0}}

    monkeypatch.setattr(
        TomographyBuilder,
        "_build_sample",
        classmethod(lambda cls, config: (object(), result)),
    )
    monkeypatch.setattr(
        TomographyBuilder,
        "_selected_pairs",
        lambda self, lens, source: [],
    )

    with pytest.raises(RuntimeError, match="No lens-source bin pairs"):
        TomographyBuilder().prepare_bins()


def test_prepare_bins_output_structure(monkeypatch):
    """Tests that prepared tomography contains the expected products."""
    result = make_result(
        z=np.array([0.0, 0.5, 1.0]),
        bins={0: np.array([0.0, 1.0, 0.0])},
    )

    result.shape_stats = lambda **kwargs: {"centers": {0: 0.5}}
    result.population_stats = lambda **kwargs: {
        "density_per_bin": {0: 1.0},
        "density_unit": "arcmin^-2",
    }

    monkeypatch.setattr(
        TomographyBuilder,
        "_build_sample",
        classmethod(lambda cls, config: (object(), result)),
    )
    monkeypatch.setattr(
        TomographyBuilder,
        "_selected_pairs",
        lambda self, lens, source: [(0, 0)],
    )

    prepared = TomographyBuilder().prepare_bins()

    assert prepared["bin_pairs"] == [(0, 0)]
    assert prepared["lens_result"] is result
    assert prepared["source_result"] is result
    assert prepared["lens_bin_centers"] == {0: 0.5}
    assert prepared["source_bin_centers"] == {0: 0.5}
    assert prepared["lens_population_stats"]["density_per_bin"] == {0: 1.0}
    assert prepared["source_population_stats"]["density_unit"] == "arcmin^-2"
    assert prepared["lens_overrides"] == {}
    assert prepared["source_overrides"] == {}
