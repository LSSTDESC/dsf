"""Tomography inputs for Delta Sigma forecasts.

This module provides a thin Binny-facing adapter for Delta Sigma forecasts.
Binny owns the survey presets, redshift distributions, tomographic binning,
bin summaries, densities, and pair filtering. DSF only chooses the lens and
source samples needed by the forecast and optionally passes forecast-specific
overrides into Binny.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from binny import NZTomography

from dsf.utils.types import BinPairs, FloatArray, TomographyInputs

__all__ = [
    "TomographyBuilder",
]


class TomographyBuilder:
    """Build Binny tomography inputs for a Delta Sigma forecast."""

    def __init__(
        self,
        *,
        lens_survey: str = "desi",
        source_survey: str = "lsst",
        lens_sample: str | None = "lrg",
        source_sample: str | None = None,
        lens_year: str | None = None,
        source_year: str | None = "1",
        lens_scenario: str | None = None,
        source_scenario: str | None = None,
        lens_role: str = "lens",
        source_role: str = "source",
        overlap_threshold: float = 0.10,
        source_behind_lens: bool = True,
        center_method: str = "mean",
        decimal_places: int = 4,
        shared_overrides: Mapping[str, Any] | None = None,
        lens_overrides: Mapping[str, Any] | None = None,
        source_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        """Set the survey samples, pair selection, and forecast overrides.

        Args:
            lens_survey: Built-in Binny survey preset used for the foreground
                lens sample.
            source_survey: Built-in Binny survey preset used for the background
                source sample.
            lens_sample: Optional lens sample label, for example ``"lrg"`` or
                ``"elg"`` for DESI-style presets.
            source_sample: Optional source sample label for presets that select
                samples by name.
            lens_year: Optional lens survey year label.
            source_year: Optional source survey year label.
            lens_scenario: Optional lens survey scenario label.
            source_scenario: Optional source survey scenario label.
            lens_role: Survey role used for the foreground sample.
            source_role: Survey role used for the background sample.
            overlap_threshold: Maximum allowed lens-source redshift-distribution
                overlap fraction.
            source_behind_lens: Whether to keep only pairs whose source-bin
                center is larger than the lens-bin center.
            center_method: Binny bin-center definition used for pair ordering.
            decimal_places: Number of decimal places used by Binny summaries.
            shared_overrides: Binny tomography-spec overrides applied to both
                lens and source samples.
            lens_overrides: Binny tomography-spec overrides applied only to the
                lens sample.
            source_overrides: Binny tomography-spec overrides applied only to the
                source sample.
        """
        self.lens_survey = lens_survey
        self.source_survey = source_survey

        self.lens_sample = lens_sample
        self.source_sample = source_sample

        self.lens_year = lens_year
        self.source_year = source_year

        self.lens_scenario = lens_scenario
        self.source_scenario = source_scenario

        self.lens_role = lens_role
        self.source_role = source_role

        self.overlap_threshold = float(overlap_threshold)
        self.source_behind_lens = bool(source_behind_lens)
        self.center_method = center_method
        self.decimal_places = int(decimal_places)

        self.shared_overrides = dict(shared_overrides or {})
        self.lens_overrides = dict(lens_overrides or {})
        self.source_overrides = dict(source_overrides or {})

    def prepare_bins(self) -> TomographyInputs:
        """Return Binny tomography products for Delta Sigma forecasts.

        Returns:
            Dictionary containing the Binny builders, Binny results, bin-shape
            summaries, population summaries, tomography metadata, and selected
            ``(lens_bin, source_bin)`` pairs.

        Raises:
            RuntimeError: If no lens-source bin pair survives the requested
                overlap and source-behind-lens cuts.
        """
        lens_builder, lens_result = self._lens_sample()
        source_builder, source_result = self._source_sample()

        lens_shape_stats = lens_builder.shape_stats(
            center_method=self.center_method,
            decimal_places=self.decimal_places,
        )
        source_shape_stats = source_builder.shape_stats(
            center_method=self.center_method,
            decimal_places=self.decimal_places,
        )

        lens_population_stats = self._population_stats(
            builder=lens_builder,
            result=lens_result,
        )
        source_population_stats = self._population_stats(
            builder=source_builder,
            result=source_result,
        )

        lens_metadata = lens_result.tomo_meta
        source_metadata = source_result.tomo_meta

        bin_pairs = self._selected_pairs(
            lens_result=lens_result,
            source_result=source_result,
            lens_centers=lens_shape_stats["centers"],
            source_centers=source_shape_stats["centers"],
        )

        if not bin_pairs:
            raise RuntimeError(
                "No galaxy-galaxy-lensing lens-source pairs survived the "
                f"selection cuts: overlap <= {self.overlap_threshold}"
                + (" and source_bin_center > lens_bin_center." if self.source_behind_lens else ".")
            )

        return {
            "lens_builder": lens_builder,
            "source_builder": source_builder,
            "lens_result": lens_result,
            "source_result": source_result,
            "lens_shape_stats": lens_shape_stats,
            "source_shape_stats": source_shape_stats,
            "lens_bin_centers": lens_shape_stats["centers"],
            "source_bin_centers": source_shape_stats["centers"],
            "lens_population_stats": lens_population_stats,
            "source_population_stats": source_population_stats,
            "lens_metadata": lens_metadata,
            "source_metadata": source_metadata,
            "bin_pairs": bin_pairs,
            "shared_overrides": self.shared_overrides,
            "lens_overrides": self.lens_overrides,
            "source_overrides": self.source_overrides,
        }

    def prepare_tomography(self) -> TomographyInputs:
        """Return Binny tomography products for Delta Sigma forecasts.

        This alias keeps older forecast scripts working while the preferred
        public method name is ``prepare_bins``.
        """
        return self.prepare_bins()

    def _lens_sample(self) -> tuple[NZTomography, Any]:
        """Return the foreground lens sample requested by the forecast."""
        return self._sample(
            survey=self.lens_survey,
            role=self.lens_role,
            sample=self.lens_sample,
            year=self.lens_year,
            scenario=self.lens_scenario,
            shared_overrides=self.shared_overrides,
            sample_overrides=self.lens_overrides,
        )

    def _source_sample(self) -> tuple[NZTomography, Any]:
        """Return the background source sample requested by the forecast."""
        return self._sample(
            survey=self.source_survey,
            role=self.source_role,
            sample=self.source_sample,
            year=self.source_year,
            scenario=self.source_scenario,
            shared_overrides=self.shared_overrides,
            sample_overrides=self.source_overrides,
        )

    @staticmethod
    def _sample(
        *,
        survey: str,
        role: str,
        sample: str | None = None,
        year: str | None = None,
        scenario: str | None = None,
        shared_overrides: Mapping[str, Any] | None = None,
        sample_overrides: Mapping[str, Any] | None = None,
    ) -> tuple[NZTomography, Any]:
        """Return one Binny survey sample with optional forecast overrides."""
        builder = NZTomography()

        overrides = {
            **dict(shared_overrides or {}),
            **dict(sample_overrides or {}),
        }

        result = builder.build_survey_bins(
            survey,
            role=role,
            year=year,
            scenario=scenario,
            sample=sample,
            overrides=overrides or None,
            include_tomo_metadata=True,
        )

        return builder, result

    def _population_stats(
        self,
        *,
        builder: NZTomography,
        result: Any,
    ) -> dict[str, Any]:
        """Return population statistics using preset density when available."""
        density = result.spec.get("sample_properties", {}).get("number_density", {})

        kwargs: dict[str, Any] = {
            "decimal_places": self.decimal_places,
        }

        if "n_gal_arcmin2" in density:
            kwargs["density_total"] = float(density["n_gal_arcmin2"])

        return builder.population_stats(**kwargs)

    def _selected_pairs(
        self,
        *,
        lens_result: Any,
        source_result: Any,
        lens_centers: dict[int, float],
        source_centers: dict[int, float],
    ) -> BinPairs:
        """Return the lens-source bin pairs kept for the forecast.

        The lens and source samples may be built on different parent redshift
        grids. For pair selection, each source bin is interpolated onto the lens
        redshift grid before the overlap fraction is evaluated. The original
        Binny tomography outputs are left unchanged.
        """
        pairs: BinPairs = []

        lens_z: FloatArray = np.asarray(lens_result.z, dtype=float)
        source_z: FloatArray = np.asarray(source_result.z, dtype=float)

        for lens_bin, lens_nz_raw in lens_result.bins.items():
            lens_index = int(lens_bin)
            lens_nz: FloatArray = np.asarray(lens_nz_raw, dtype=float)

            for source_bin, source_nz_raw in source_result.bins.items():
                source_index = int(source_bin)

                if self.source_behind_lens:
                    if float(source_centers[source_index]) <= float(lens_centers[lens_index]):
                        continue

                source_nz: FloatArray = np.asarray(source_nz_raw, dtype=float)

                overlap = self._overlap_fraction_on_lens_grid(
                    lens_z=lens_z,
                    lens_nz=lens_nz,
                    source_z=source_z,
                    source_nz=source_nz,
                )

                if overlap <= self.overlap_threshold:
                    pairs.append((lens_index, source_index))

        return pairs

    @staticmethod
    def _overlap_fraction_on_lens_grid(
        *,
        lens_z: FloatArray,
        lens_nz: FloatArray,
        source_z: FloatArray,
        source_nz: FloatArray,
    ) -> float:
        """Measure the fractional redshift overlap of a lens and source bin.

        This helper compares the lens and source redshift distributions on the
        lens redshift grid. It is useful when lens and source samples are defined
        on different grids, but the overlap calculation needs both distributions
        evaluated at the same redshift values.

        Binny provides this type of overlap calculation natively, but expects the
        lens and source distributions to be evaluated on a shared redshift grid.
        This helper handles the case where the two samples are defined on different
        redshift grids by evaluating the source distribution on the lens grid before
        computing the fractional overlap.

        Args:
            lens_z: Redshift grid for the lens bin.
            lens_nz: Lens redshift distribution evaluated on ``lens_z``.
            source_z: Redshift grid for the source bin.
            source_nz: Source redshift distribution evaluated on ``source_z``.

        Returns:
            Fractional overlap between the lens and source redshift distributions,
            normalized by the smaller integrated distribution. Returns zero if
            either distribution has a non-positive integral on the comparison grid.
        """
        source_on_lens_grid = np.interp(
            lens_z,
            source_z,
            source_nz,
            left=0.0,
            right=0.0,
        )

        lens_integral = float(np.trapezoid(lens_nz, lens_z))
        source_integral = float(np.trapezoid(source_on_lens_grid, lens_z))

        denominator = min(lens_integral, source_integral)

        if denominator <= 0.0:
            return 0.0

        overlap_integral = float(
            np.trapezoid(
                np.minimum(lens_nz, source_on_lens_grid),
                lens_z,
            )
        )

        return overlap_integral / denominator
