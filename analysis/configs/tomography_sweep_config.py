"""Reusable tomography sweep setup for DSF analysis scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUN_NAME = "tomography_scenarios"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "runs_output" / RUN_NAME
PLOT_DIR = OUTPUT_DIR / "plots"
ARRAY_DIR = OUTPUT_DIR / "arrays"

LENS_SURVEY = "desi"
SOURCE_SURVEY = "lsst"
LENS_SAMPLE = "lrg"
SOURCE_YEAR = "1"

OVERLAP_THRESHOLD = 0.10
SOURCE_BEHIND_LENS = True
CENTER_METHOD = "mean"


TOMO_LEVELS = {
    "coarse": (2, 4),
    "fiducial": (3, 5),
    "fine": (5, 6),
    "stress": (6, 7),
}


LENS_Z_RANGES = {
    "low": [0.1, 0.8],
    "mid": [0.3, 0.9],
    "high": [0.5, 1.1],
    "wide": [0.1, 1.0],
}


SCENARIOS = {
    # Reference setup: fiducial binning and middle lens redshift range.
    "baseline": ("fiducial", "mid"),
    # Same fiducial binning, but move the lens sample to lower redshift.
    "low_z_baseline": ("fiducial", "low"),
    # Same fiducial binning, but move the lens sample to higher redshift.
    "high_z_baseline": ("fiducial", "high"),
    # Same fiducial binning, but use a broader lens redshift range.
    "wide_z_baseline": ("fiducial", "wide"),
    # Fewer lens/source bins at low redshift: checks a simple conservative setup.
    "coarse_low_z": ("coarse", "low"),
    # Fewer lens/source bins over a wide range: checks broad bins with more overlap risk.
    "coarse_wide_z": ("coarse", "wide"),
    # More lens/source bins at the middle redshift range: checks higher tomography resolution.
    "fine_mid_z": ("fine", "mid"),
    # More lens/source bins over a wide range: checks finer slicing across broad coverage.
    "fine_wide_z": ("fine", "wide"),
    # Most aggressive binning at the middle redshift range: isolates bin-count stress.
    "stress_mid_z": ("stress", "mid"),
    # Most aggressive binning over a wide range: strongest safety/stability stress test.
    "stress_wide_z": ("stress", "wide"),
}

SCENARIO_LABELS = {
    "baseline": r"Fiducial: mid-$z$",
    "low_z_baseline": r"Fiducial: low-$z$",
    "high_z_baseline": r"Fiducial: high-$z$",
    "wide_z_baseline": r"Fiducial: wide-$z$",
    "coarse_low_z": r"Coarse: low-$z$",
    "coarse_wide_z": r"Coarse: wide-$z$",
    "fine_mid_z": r"Fine: mid-$z$",
    "fine_wide_z": r"Fine: wide-$z$",
    "stress_mid_z": r"Stress: mid-$z$",
    "stress_wide_z": r"Stress: wide-$z$",
}

SCENARIO_MARKERS = {
    "baseline": "o",
    "fiducial": "o",
    "coarse": "^",
    "fine": "D",
    "stress": "X",
}

Z_SCENARIO_HATCHES = {
    "low": "/////",
    "mid": "....",
    "high": "\\\\\\",
    "wide": "xxxx",
}

SCENARIO_COLOR_ORDER = [
    "baseline",
    "low_z_baseline",
    "high_z_baseline",
    "wide_z_baseline",
    "coarse_low_z",
    "coarse_wide_z",
    "fine_mid_z",
    "fine_wide_z",
    "stress_mid_z",
    "stress_wide_z",
]


def scenario_color_map(colors):
    """Return a fixed scenario-to-color mapping."""
    return {scenario: color for scenario, color in zip(SCENARIO_COLOR_ORDER, colors, strict=True)}


def marker_for_scenario(scenario_name):
    """Return the marker style for one tomography scenario."""
    scenario_name = scenario_name.lower()

    for name, marker in SCENARIO_MARKERS.items():
        if name in scenario_name:
            return marker

    return "o"


def z_scenario_for_scenario(scenario_name):
    """Return the lens-redshift scenario label for one tomography scenario."""
    scenario_name = scenario_name.lower()

    for name in Z_SCENARIO_HATCHES:
        if name in scenario_name:
            return name

    if scenario_name == "baseline":
        return "mid"

    return "mid"


def hatch_for_scenario(scenario_name):
    """Return the hatch style for one tomography scenario."""
    z_scenario = z_scenario_for_scenario(scenario_name)

    return Z_SCENARIO_HATCHES[z_scenario]


def lens_overrides(n_bins, z_range):
    """Return lens-bin overrides for one tomography scenario."""
    return {
        "bins": {
            "scheme": "equidistant",
            "n_bins": int(n_bins),
            "range": z_range,
            "edges": np.linspace(z_range[0], z_range[1], int(n_bins) + 1),
        }
    }


def source_overrides(n_bins):
    """Return source-bin overrides for one tomography scenario."""
    return {
        "bins": {
            "scheme": "equal_number",
            "n_bins": int(n_bins),
        }
    }


def scenario_values(scenario_name):
    """Return expanded setup values for one scenario."""
    tomo_level, range_label = SCENARIOS[scenario_name]
    lens_n_bins, source_n_bins = TOMO_LEVELS[tomo_level]
    z_range = LENS_Z_RANGES[range_label]

    return tomo_level, range_label, lens_n_bins, source_n_bins, z_range


def case_label(scenario_name, lens_n_bins, source_n_bins, z_range):
    """Return a filesystem-safe label for one scenario."""
    zmin, zmax = z_range

    label = f"{scenario_name}_lens{lens_n_bins}_source{source_n_bins}_z{zmin:g}_{zmax:g}"

    return label.replace(".", "p")


def scenario_plot_label(scenario_name):
    """Return a readable plot label for one scenario."""
    return SCENARIO_LABELS.get(scenario_name, scenario_name.replace("_", " "))


def detailed_scenario_plot_label(scenario_name):
    """Return a readable plot label including bin counts and lens redshift range."""
    tomo_level, range_label, lens_n_bins, source_n_bins, z_range = scenario_values(scenario_name)
    zmin, zmax = z_range

    return (
        f"{scenario_plot_label(scenario_name)} "
        f"({lens_n_bins}L/{source_n_bins}S, "
        f"{zmin:g} < z_l < {zmax:g})"
    )


def scenario_name_from_row(row):
    """Return the tomography scenario name stored in a summary row."""
    if "scenario" in row:
        return row["scenario"]

    label = str(row["label"])

    for scenario_name in sorted(SCENARIOS, key=len, reverse=True):
        if label == scenario_name or label.startswith(f"{scenario_name}_"):
            return scenario_name

    for scenario_name, plot_label in SCENARIO_LABELS.items():
        if label == plot_label or label.startswith(f"{plot_label} "):
            return scenario_name

    return label
