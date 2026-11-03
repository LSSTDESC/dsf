"""Binny tomography inputs for Delta Sigma forecasts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from binny import NZTomography

from dsf.utils.types import BinPairs, TomographyInputs

__all__ = ["TomographyBuilder"]


class TomographyBuilder:
    """Prepare lens and source tomography using Binny."""

    def __init__(
        self,
        lens: Mapping[str, Any] | None = None,
        source: Mapping[str, Any] | None = None,
        *,
        overlap_threshold: float = 0.10,
        source_behind_lens: bool = True,
    ) -> None:
        """Configure lens and source tomography.

        Args:
            lens: Lens configuration using a Binny survey preset, parent n(z),
                or predefined tomographic bins.
            source: Source configuration using the same input conventions.
            overlap_threshold: Maximum allowed lens-source distribution overlap.
            source_behind_lens: Require source-bin centers above lens-bin centers.
        """
        self.lens = dict(lens) if lens is not None else {
            "survey": "desi", "sample": "lrg"
        }
        self.source = dict(source) if source is not None else {
            "survey": "lsst", "role": "source", "year": "1"
        }

        self.overlap_threshold = float(overlap_threshold)
        self.source_behind_lens = bool(source_behind_lens)

        if not 0.0 <= self.overlap_threshold <= 1.0:
            raise ValueError("overlap_threshold must be between 0 and 1.")

    @staticmethod
    def _read_config(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
        """Read tomography metadata from a Python dictionary or YAML file."""
        if isinstance(config, Mapping):
            data = dict(config)
        else:
            with Path(config).open("r", encoding="utf-8") as stream:
                data = yaml.safe_load(stream)

            if not isinstance(data, Mapping):
                raise ValueError("Tomography YAML must contain a mapping.")

            data = dict(data)

        if "tomography" in data:
            entries = data["tomography"]

            if not isinstance(entries, list) or len(entries) != 1:
                raise ValueError(
                    "Custom tomography configuration must contain "
                    "exactly one tomography entry."
                )

            data = dict(entries[0])

        return data

    @classmethod
    def _build_sample(cls, config: Mapping[str, Any]) -> tuple[NZTomography, Any]:
        """Build a survey, parent distribution, or predefined bins with Binny."""
        config = dict(config)
        builder = NZTomography()

        if "survey" in config:
            result = builder.build_survey_bins(
                config["survey"],
                role=config.get("role"),
                year=config.get("year"),
                scenario=config.get("scenario"),
                sample=config.get("sample"),
                overrides=config.get("overrides"),
                include_tomo_metadata=True,
            )
            return builder, result

        if "config" not in config:
            raise ValueError(
                "Custom tomography requires a 'config' dictionary or YAML path."
            )

        spec = cls._read_config(config["config"])

        if "bins" in config:
            if "z" not in config:
                raise ValueError("Predefined bins require a redshift grid 'z'.")

            properties = spec.get("sample_properties")
            fractions = spec.get("population_fractions")

            if not isinstance(properties, Mapping):
                raise ValueError("Predefined bins require sample_properties.")

            if not isinstance(fractions, Mapping):
                raise ValueError("Predefined bins require population_fractions.")

            density = properties.get("number_density")

            if not isinstance(density, Mapping) or not density:
                raise ValueError(
                    "Predefined bins require sample_properties.number_density."
                )

            bins = config["bins"]
            indices = {int(i) for i in bins}
            fractions = {int(i): float(f) for i, f in fractions.items()}

            if not indices or set(fractions) != indices:
                raise ValueError(
                    "population_fractions keys must match the bin indices."
                )

            if any(not np.isfinite(f) or f < 0.0 for f in fractions.values()):
                raise ValueError(
                    "population_fractions must be finite and nonnegative."
                )

            if not np.isclose(sum(fractions.values()), 1.0, rtol=1e-6, atol=1e-8):
                raise ValueError("population_fractions must sum to one.")

            result = builder.build_predefined_bins(
                z=config["z"],
                bins=bins,
                normalize_bins=spec.get("normalize_bins", False),
            )

            result = type(result)(
                z=result.z,
                nz=result.nz,
                spec={**result.spec, "sample_properties": dict(properties)},
                bins=result.bins,
                tomo_meta={"frac_per_bin": fractions},
                survey_meta=result.survey_meta,
                survey=result.survey,
            )

            return builder, result

        if "z" in config and "nz" in config:
            result = builder.build_bins(
                z=config["z"],
                nz=config["nz"],
                tomo_spec=spec,
                include_tomo_metadata=True,
            )
            return builder, result

        if "z" in config or "nz" in config:
            raise ValueError("Custom parent n(z) requires both 'z' and 'nz'.")

        raise ValueError(
            "Custom tomography requires 'z' and 'nz', or 'z' and 'bins'."
        )

    @staticmethod
    def _population_stats(result: Any) -> dict[str, Any]:
        """Return Binny population statistics using survey number densities."""
        density = result.spec.get("sample_properties", {}).get("number_density", {})

        kwargs: dict[str, Any] = {"decimal_places": 4}

        if "n_gal_arcmin2" in density:
            kwargs["density_total"] = float(density["n_gal_arcmin2"])

        stats = dict(result.population_stats(**kwargs))

        if "density_per_bin" in stats:
            return stats

        if "n_gal_comoving_h3_mpc3" in density:
            nbar = float(density["n_gal_comoving_h3_mpc3"])
            stats["density_per_bin"] = {
                int(i): nbar * float(f)
                for i, f in stats.get("fractions", {}).items()
            }
            stats["density_unit"] = "h^3 Mpc^-3"
            return stats

        if "total_count_per_bin" in stats and "volume_per_bin" in stats:
            stats["density_per_bin"] = {
                int(i): float(stats["total_count_per_bin"][i])
                        / float(stats["volume_per_bin"][i])
                for i in stats["total_count_per_bin"]
            }
            stats["density_unit"] = "h^3 Mpc^-3"

        return stats

    def prepare_bins(self) -> TomographyInputs:
        """Return Binny tomography products for Delta Sigma forecasts."""
        lens_builder, lens_result = self._build_sample(self.lens)
        source_builder, source_result = self._build_sample(self.source)

        lens_shape = lens_result.shape_stats(
            center_method="mean", decimal_places=4
        )
        source_shape = source_result.shape_stats(
            center_method="mean", decimal_places=4
        )

        pairs = self._selected_pairs(lens_result, source_result)

        if not pairs:
            raise RuntimeError(
                "No lens-source bin pairs survived the selection cuts."
            )

        return {
            "lens_builder": lens_builder,
            "source_builder": source_builder,
            "lens_result": lens_result,
            "source_result": source_result,
            "lens_shape_stats": lens_shape,
            "source_shape_stats": source_shape,
            "lens_bin_centers": lens_shape["centers"],
            "source_bin_centers": source_shape["centers"],
            "lens_population_stats": self._population_stats(lens_result),
            "source_population_stats": self._population_stats(source_result),
            "lens_metadata": lens_result.tomo_meta,
            "source_metadata": source_result.tomo_meta,
            "bin_pairs": pairs,
            "shared_overrides": {},
            "lens_overrides": dict(self.lens.get("overrides") or {}),
            "source_overrides": dict(self.source.get("overrides") or {}),
        }

    def prepare_tomography(self) -> TomographyInputs:
        """Backward-compatible alias for prepare_bins."""
        return self.prepare_bins()

    def _selected_pairs(
        self,
        lens_result: Any,
        source_result: Any,
    ) -> BinPairs:
        """Select lens-source bin pairs using Binny."""
        filters = [
            {
                "name": "overlap_fraction",
                "threshold": self.overlap_threshold,
                "compare": "le",
            }
        ]

        if self.source_behind_lens:
            filters.append({
                "name": "score_relation",
                "score": "mean",
                "relation": "gt",
            })

        spec = {
            "topology": {"name": "pairs_cartesian"},
            "filters": filters,
        }

        return list(lens_result.bin_combo_filter(spec, other=source_result))
