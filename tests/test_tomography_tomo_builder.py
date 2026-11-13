"""Unit tests for ``dsf.tomography.tomo_builder``."""

from types import SimpleNamespace

import numpy as np
import pytest

from dsf.tomography import tomo_builder
from dsf.tomography.tomo_builder import TomographyBuilder


def make_result(
    *,
    z,
    bins,
    number_density=None,
    metadata=None,
):
    """Return a lightweight Binny-compatible tomography result."""
    sample_properties = {}

    if number_density is not None:
        sample_properties["number_density"] = number_density

    return SimpleNamespace(
        z=np.asarray(z, dtype=float),
        bins={key: np.asarray(value, dtype=float) for key, value in bins.items()},
        spec={"sample_properties": sample_properties},
        metadata={} if metadata is None else metadata,
    )


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


def test_sample_passes_expected_arguments_to_binny(monkeypatch):
    """Tests that one sample is built with the expected Binny arguments."""
    calls = []
    result = make_result(
        z=np.array([0.0, 0.5, 1.0]),
        bins={0: np.array([1.0, 0.0, 0.0])},
    )

    def fake_build_survey_tomography(survey, **kwargs):
        calls.append({"survey": survey, **kwargs})
        return result

    monkeypatch.setattr(
        tomo_builder,
        "build_survey_tomography",
        fake_build_survey_tomography,
    )

    sample_result = TomographyBuilder._build_sample(
        {
            "survey": "desi",
            "role": "lens",
            "sample": "lrg",
            "year": "1",
            "scenario": "baseline",
            "overrides": {"sample": 2},
        }
    )

    assert sample_result is result
    assert calls == [
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
    """Tests that default lens and source configurations are correct."""
    builder = TomographyBuilder()

    assert builder.lens == {"survey": "desi", "sample": "lrg"}
    assert builder.source == {"survey": "lsst", "year": "1", "role": "source"}
    assert builder.overlap_threshold == 0.10
    assert builder.source_behind_lens is True


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_overlap_threshold(threshold):
    """Tests that invalid overlap thresholds are rejected."""
    with pytest.raises(ValueError, match="overlap_threshold"):
        TomographyBuilder(overlap_threshold=threshold)


def test_pair_selection_overlap_threshold(monkeypatch):
    """Tests that pair selection delegates overlap filtering to Binny."""
    calls = []
    lens = object()
    source = object()

    class DummyFilter:
        def select(self, spec):
            calls.append(("select", spec))
            return self

        def values(self):
            calls.append(("values",))
            return [(0, 1)]

    def fake_make_bin_combo_filter(tomo, other):
        calls.append(("factory", tomo, other))
        return DummyFilter()

    monkeypatch.setattr(
        tomo_builder,
        "make_bin_combo_filter",
        fake_make_bin_combo_filter,
    )

    builder = TomographyBuilder(
        overlap_threshold=0.25,
        source_behind_lens=False,
    )

    pairs = builder._selected_pairs(lens, source)

    assert pairs == [(0, 1)]
    assert calls == [
        ("factory", lens, source),
        (
            "select",
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
        ),
        ("values",),
    ]


def test_pair_selection_source_ordering(monkeypatch):
    """Tests that source ordering is included when requested."""
    specs = []

    class DummyFilter:
        def select(self, spec):
            specs.append(spec)
            return self

        def values(self):
            return [(0, 1)]

    monkeypatch.setattr(
        tomo_builder,
        "make_bin_combo_filter",
        lambda tomo, other: DummyFilter(),
    )

    builder = TomographyBuilder(source_behind_lens=True)
    pairs = builder._selected_pairs(object(), object())

    assert pairs == [(0, 1)]
    assert specs == [
        {
            "topology": {"name": "pairs_cartesian"},
            "filters": [
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
            ],
        }
    ]


def test_prepare_bins_no_pairs(monkeypatch):
    """Tests that empty lens-source pair selection raises an error."""
    result = make_result(
        z=np.array([0.0, 0.5, 1.0]),
        bins={0: np.array([0.0, 1.0, 0.0])},
    )

    monkeypatch.setattr(
        TomographyBuilder,
        "_build_sample",
        classmethod(lambda cls, config: result),
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
        number_density={"n_gal_arcmin2": 1.0},
        metadata={"frac_per_bin": {0: 1.0}},
    )

    monkeypatch.setattr(
        TomographyBuilder,
        "_build_sample",
        classmethod(lambda cls, config: result),
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
    assert prepared["lens_population_stats"]["fractions"] == {0: 1.0}
    assert prepared["source_population_stats"]["density_total"] == 1.0
    assert prepared["lens_population_stats"]["density_per_bin"] == {0: 1.0}
    assert prepared["source_population_stats"]["density_per_bin"] == {0: 1.0}
    assert prepared["lens_metadata"] is result.metadata
    assert prepared["source_metadata"] is result.metadata
    assert prepared["lens_overrides"] == {}
    assert prepared["source_overrides"] == {}


def test_predefined_tomography_preserves_population_fractions(monkeypatch):
    """Tests that predefined bins preserve normalization and population fractions."""
    calls = []
    z = np.array([0.0, 0.5, 1.0])
    bins = {
        0: np.array([0.0, 1.0, 0.0]),
        1: np.array([0.0, 0.0, 1.0]),
    }
    fractions = {0: 0.6, 1: 0.4}
    original_build = tomo_builder.build_predefined_tomography

    def fake_build_predefined_tomography(**kwargs):
        calls.append(kwargs)
        return original_build(**kwargs)

    monkeypatch.setattr(
        tomo_builder,
        "build_predefined_tomography",
        fake_build_predefined_tomography,
    )

    sample = TomographyBuilder._build_sample(
        {
            "z": z,
            "bins": bins,
            "config": {
                "normalize_bins": False,
                "sample_properties": {
                    "number_density": {"n_gal_arcmin2": 10.0},
                },
                "population_fractions": fractions,
            },
        }
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["z"], z)
    assert calls[0]["normalize"] is False

    for key in bins:
        np.testing.assert_array_equal(calls[0]["bins"][key], bins[key])
        np.testing.assert_array_equal(sample.bins[key], bins[key])

    assert sample.metadata["frac_per_bin"] == fractions
    assert sample.spec["sample_properties"]["number_density"] == {"n_gal_arcmin2": 10.0}


def test_parent_nz_delegates_to_binny_spec(monkeypatch):
    """Tests that parent redshift distributions use the functional Binny API."""
    calls = []
    z = np.array([0.0, 0.5, 1.0])
    nz = np.array([0.0, 2.0, 0.0])
    spec = {
        "kind": "edges",
        "bins": {"edges": [0.0, 0.5, 1.0]},
    }
    result = make_result(z=z, bins={0: nz})

    def fake_build_tomography_from_spec(**kwargs):
        calls.append(kwargs)
        return result

    monkeypatch.setattr(
        tomo_builder,
        "build_tomography_from_spec",
        fake_build_tomography_from_spec,
    )

    sample = TomographyBuilder._build_sample({"z": z, "nz": nz, "config": spec})

    assert sample is result
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0]["z"], z)
    np.testing.assert_array_equal(calls[0]["nz"], nz)
    assert calls[0]["tomo_spec"] == spec
    assert calls[0]["include_tomo_metadata"] is True


def test_population_stats_preserves_comoving_density(monkeypatch):
    """Tests that comoving number density is passed to Binny correctly."""
    calls = []
    result = make_result(
        z=np.array([0.0, 0.5, 1.0]),
        bins={0: np.array([0.0, 1.0, 0.0])},
        number_density={"n_gal_mpc3": 0.0005},
        metadata={"frac_per_bin": {0: 1.0}},
    )

    def fake_population_stats(bins, metadata, **kwargs):
        calls.append((bins, metadata, kwargs))
        return {"density_per_bin": {0: 0.0005}}

    monkeypatch.setattr(
        tomo_builder,
        "population_stats",
        fake_population_stats,
    )

    stats = TomographyBuilder._population_stats(result)

    assert stats["density_per_bin"] == {0: 0.0005}
    assert len(calls) == 1
    assert calls[0][0] is result.bins
    assert calls[0][1] is result.metadata


def test_default_tomography_with_real_binny():
    """Tests that real Binny survey presets produce valid tomography inputs."""
    prepared = TomographyBuilder().prepare_bins()

    lens = prepared["lens_result"]
    source = prepared["source_result"]

    assert lens.bins
    assert source.bins
    assert prepared["bin_pairs"]

    for result in (lens, source):
        assert np.all(np.isfinite(result.z))
        assert np.all(np.diff(result.z) > 0)

        for nz in result.bins.values():
            assert len(nz) == len(result.z)
            assert np.all(np.isfinite(nz))
            assert np.all(nz >= 0.0)
            assert np.trapezoid(nz, result.z) == pytest.approx(1.0, abs=1e-3)

    for lens_bin, source_bin in prepared["bin_pairs"]:
        assert lens_bin in lens.bins
        assert source_bin in source.bins
