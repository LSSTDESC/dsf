"""Tomography diagnostics for DSF scenario sweeps."""

from __future__ import annotations

import numpy as np


def safe_float(value, default=0.0):
    """Return a finite float or a default value."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(value):
        return default

    return value


def get_nested(dictionary, keys, default=None):
    """Return a nested dictionary value.

    Integer and string versions of keys are both supported. This is useful for
    handling Binny/DSF outputs where bin indices may appear as either integers
    or strings after serialization.
    """
    value = dictionary

    for key in keys:
        if not isinstance(value, dict):
            return default

        if key in value:
            value = value[key]
        elif str(key) in value:
            value = value[str(key)]
        else:
            return default

    return value


def values_from_mapping(mapping):
    """Return finite numeric values from a mapping."""
    if not isinstance(mapping, dict):
        return np.asarray([], dtype=float)

    values = [safe_float(value, np.nan) for value in mapping.values()]
    values = [value for value in values if np.isfinite(value)]

    return np.asarray(values, dtype=float)


def balance(mapping):
    """Return the min/max balance of values in a mapping.

    A value close to one means the bins are similarly populated. A value close
    to zero means at least one bin is much less populated than the most populated
    bin.
    """
    values = values_from_mapping(mapping)

    if values.size == 0:
        return 0.0

    maximum = float(np.max(values))

    if maximum <= 0.0:
        return 0.0

    return float(np.min(values) / maximum)


def get_population_fractions(population_stats):
    """Return per-bin population fractions from Binny/DSF population stats."""
    fractions = population_stats.get("fractions")

    if fractions is None:
        fractions = population_stats.get("frac_per_bin")

    if fractions is None:
        fractions = population_stats.get("fraction_per_bin")

    if fractions is None:
        return {}

    return {int(key): safe_float(value) for key, value in fractions.items()}


def get_widths(shape_stats):
    """Return 68 percent redshift widths from Binny/DSF shape stats."""
    widths = []

    for stats in shape_stats.get("per_bin", {}).values():
        width = get_nested(stats, ["moments", "width_68"], np.nan)
        width = safe_float(width, np.nan)

        if np.isfinite(width):
            widths.append(width)

    return np.asarray(widths, dtype=float)


def get_second_peak_ratios(shape_stats):
    """Return second-peak ratios from Binny/DSF shape stats."""
    ratios = []

    for stats in shape_stats.get("per_bin", {}).values():
        ratio = get_nested(stats, ["peaks", "second_peak_ratio"], np.nan)
        ratio = safe_float(ratio, np.nan)

        if np.isfinite(ratio):
            ratios.append(ratio)

    return np.asarray(ratios, dtype=float)


def selected_overlaps(builder, lens_result, source_result, bin_pairs):
    """Return redshift-overlap fractions for selected lens-source bin pairs."""
    overlaps = []

    lens_z = np.asarray(lens_result.z, dtype=float)
    source_z = np.asarray(source_result.z, dtype=float)

    for lens_key, source_key in bin_pairs:
        overlap = builder._overlap_fraction_on_lens_grid(
            lens_z=lens_z,
            lens_nz=np.asarray(lens_result.bins[lens_key], dtype=float),
            source_z=source_z,
            source_nz=np.asarray(source_result.bins[source_key], dtype=float),
        )

        overlap = safe_float(overlap, np.nan)

        if np.isfinite(overlap):
            overlaps.append(overlap)

    return np.asarray(overlaps, dtype=float)


def pair_separations(bin_pairs, lens_centers, source_centers):
    """Return source-center minus lens-center values for selected bin pairs."""
    separations = []

    for lens_key, source_key in bin_pairs:
        lens_center = get_nested(lens_centers, [lens_key])
        source_center = get_nested(source_centers, [source_key])

        if lens_center is None or source_center is None:
            continue

        separations.append(float(source_center) - float(lens_center))

    return np.asarray(separations, dtype=float)


def pair_fraction(n_pairs, lens_n_bins, source_n_bins):
    """Return selected-pair fraction out of all lens-source combinations."""
    n_possible = int(lens_n_bins) * int(source_n_bins)

    if n_possible <= 0:
        return 0.0

    return float(n_pairs / n_possible)


def valid_tomography(row):
    """Return whether a scenario built the requested bins and selected pairs."""
    if row["failed"]:
        return False

    if row["lens_n_bins_built"] != row["lens_n_bins_requested"]:
        return False

    if row["source_n_bins_built"] != row["source_n_bins_requested"]:
        return False

    if row["n_pairs"] <= 0:
        return False

    return True


def summarize_tomography_result(
    *,
    label,
    scenario_name,
    tomo_level,
    range_label,
    lens_n_bins,
    source_n_bins,
    z_range,
    builder,
    result,
    plot_path,
):
    """Return one compact row summarizing a tomography result."""
    lens_result = result["lens_result"]
    source_result = result["source_result"]
    bin_pairs = result["bin_pairs"]

    lens_centers = result["lens_bin_centers"]
    source_centers = result["source_bin_centers"]

    lens_fractions = get_population_fractions(result["lens_population_stats"])
    source_fractions = get_population_fractions(result["source_population_stats"])

    lens_widths = get_widths(result["lens_shape_stats"])
    source_widths = get_widths(result["source_shape_stats"])

    lens_second_peaks = get_second_peak_ratios(result["lens_shape_stats"])
    source_second_peaks = get_second_peak_ratios(result["source_shape_stats"])

    overlaps = selected_overlaps(
        builder,
        lens_result,
        source_result,
        bin_pairs,
    )
    separations = pair_separations(
        bin_pairs,
        lens_centers,
        source_centers,
    )

    lens_n_bins_built = len(lens_result.bins)
    source_n_bins_built = len(source_result.bins)

    row = {
        "label": label,
        "scenario": scenario_name,
        "tomo_level": tomo_level,
        "lens_range_label": range_label,
        "lens_z_min": z_range[0],
        "lens_z_max": z_range[1],
        "lens_n_bins_requested": lens_n_bins,
        "source_n_bins_requested": source_n_bins,
        "lens_n_bins_built": lens_n_bins_built,
        "source_n_bins_built": source_n_bins_built,
        "n_pairs": len(bin_pairs),
        "pair_fraction": pair_fraction(
            len(bin_pairs),
            lens_n_bins_built,
            source_n_bins_built,
        ),
        "bin_pairs": bin_pairs,
        "lens_centers": lens_centers,
        "source_centers": source_centers,
        "lens_fractions": lens_fractions,
        "source_fractions": source_fractions,
        "lens_fraction_balance": balance(lens_fractions),
        "source_fraction_balance": balance(source_fractions),
        "lens_mean_width_68": float(np.mean(lens_widths)) if lens_widths.size else 0.0,
        "source_mean_width_68": float(np.mean(source_widths)) if source_widths.size else 0.0,
        "lens_max_second_peak_ratio": (
            float(np.max(lens_second_peaks)) if lens_second_peaks.size else 0.0
        ),
        "source_max_second_peak_ratio": (
            float(np.max(source_second_peaks)) if source_second_peaks.size else 0.0
        ),
        "mean_selected_overlap": float(np.mean(overlaps)) if overlaps.size else 0.0,
        "max_selected_overlap": float(np.max(overlaps)) if overlaps.size else 0.0,
        "behind_fraction": float(np.mean(separations > 0.0)) if separations.size else 0.0,
        "mean_pair_separation": float(np.mean(separations)) if separations.size else 0.0,
        "min_pair_separation": float(np.min(separations)) if separations.size else 0.0,
        "plot_path": str(plot_path),
        "failed": False,
        "error": "",
    }

    row["valid_tomography"] = valid_tomography(row)

    return row


def failed_tomography_row(
    *,
    label,
    scenario_name,
    tomo_level,
    range_label,
    lens_n_bins,
    source_n_bins,
    z_range,
    error,
):
    """Return a compact tomography summary row for a failed scenario."""
    return {
        "label": label,
        "scenario": scenario_name,
        "tomo_level": tomo_level,
        "lens_range_label": range_label,
        "lens_z_min": z_range[0],
        "lens_z_max": z_range[1],
        "lens_n_bins_requested": lens_n_bins,
        "source_n_bins_requested": source_n_bins,
        "lens_n_bins_built": None,
        "source_n_bins_built": None,
        "n_pairs": 0,
        "pair_fraction": 0.0,
        "bin_pairs": [],
        "lens_centers": {},
        "source_centers": {},
        "lens_fractions": {},
        "source_fractions": {},
        "lens_fraction_balance": 0.0,
        "source_fraction_balance": 0.0,
        "lens_mean_width_68": 0.0,
        "source_mean_width_68": 0.0,
        "lens_max_second_peak_ratio": 0.0,
        "source_max_second_peak_ratio": 0.0,
        "mean_selected_overlap": 0.0,
        "max_selected_overlap": 0.0,
        "behind_fraction": 0.0,
        "mean_pair_separation": 0.0,
        "min_pair_separation": 0.0,
        "valid_tomography": False,
        "plot_path": "",
        "failed": True,
        "error": str(error),
    }
