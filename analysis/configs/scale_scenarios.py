"""Reusable radial-scale sweep setup for DSF analysis scripts.

This module defines projected-radius scale choices for Delta Sigma SNR tests.
It is intended to be used after choosing one fixed tomography setup. The sweep
then varies only the radial scale range and binning, so the impact of scale
cuts on SNR can be checked cleanly.

The returned dictionaries match the YAML structure used under:

    radial_bins:
      radius_eval:
        min: ...
        max: ...
        n: ...

      rp_bin_edges:
        min: ...
        max: ...
        n: ...
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RUN_NAME = "scale_scenarios"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "runs_output" / RUN_NAME
PLOT_DIR = OUTPUT_DIR / "plots"
ARRAY_DIR = OUTPUT_DIR / "arrays"

# This is the tomography setup you decided to keep fixed while sweeping scales.
# Your runner can import this and apply the corresponding tomography overrides
# from tomography_scenarios.py.
TOMOGRAPHY_SCENARIO = "baseline"


SCALE_SCENARIOS = {
    # Conservative large-scale-only setup.
    "very_conservative": {
        "radius_eval": (10.0, 50.0, 6),
        "rp_bin_edges": (10.0, 60.0, 7),
    },
    # Slightly more radial information, still avoiding very small scales.
    "conservative": {
        "radius_eval": (5.0, 50.0, 7),
        "rp_bin_edges": (5.0, 60.0, 8),
    },
    # Reference setup with moderate inclusion of smaller projected radii.
    "fiducial": {
        "radius_eval": (2.0, 50.0, 8),
        "rp_bin_edges": (2.0, 60.0, 9),
    },
    # Tests whether SNR becomes sensitive to small projected scales.
    "small_scales": {
        "radius_eval": (1.0, 50.0, 9),
        "rp_bin_edges": (1.0, 60.0, 10),
    },
    # Keeps the same minimum scale but extends to larger projected radius.
    "extended_large_scales": {
        "radius_eval": (5.0, 100.0, 8),
        "rp_bin_edges": (5.0, 120.0, 9),
    },
    # Broadest diagnostic range, useful as a stress test.
    "wide_scales": {
        "radius_eval": (1.0, 100.0, 10),
        "rp_bin_edges": (1.0, 120.0, 11),
    },
}


SCENARIO_LABELS = {
    "very_conservative": r"Very conservative scales",
    "conservative": r"Conservative scales",
    "fiducial": r"Fiducial scales",
    "small_scales": r"Small-scale test",
    "extended_large_scales": r"Extended large scales",
    "wide_scales": r"Wide scale range",
}


def geomspace_overrides(minimum, maximum, n_values):
    """Return radial-bin settings matching the YAML geomspace format."""
    return {
        "min": float(minimum),
        "max": float(maximum),
        "n": int(n_values),
    }


def radial_bin_overrides(radius_eval, rp_bin_edges):
    """Return radial-bin overrides for one scale scenario."""
    radius_min, radius_max, radius_n = radius_eval
    rp_min, rp_max, rp_n = rp_bin_edges

    return {
        "radius_eval": geomspace_overrides(
            minimum=radius_min,
            maximum=radius_max,
            n_values=radius_n,
        ),
        "rp_bin_edges": geomspace_overrides(
            minimum=rp_min,
            maximum=rp_max,
            n_values=rp_n,
        ),
    }


def scenario_values(scenario_name):
    """Return expanded radial-bin values for one scale scenario."""
    scenario = SCALE_SCENARIOS[scenario_name]

    radius_eval = scenario["radius_eval"]
    rp_bin_edges = scenario["rp_bin_edges"]

    return radius_eval, rp_bin_edges


def scenario_overrides(scenario_name):
    """Return radial-bin overrides for one scale scenario."""
    radius_eval, rp_bin_edges = scenario_values(scenario_name)

    return radial_bin_overrides(
        radius_eval=radius_eval,
        rp_bin_edges=rp_bin_edges,
    )


def case_label(scenario_name):
    """Return a filesystem-safe label for one scale scenario."""
    radius_eval, rp_bin_edges = scenario_values(scenario_name)

    radius_min, radius_max, radius_n = radius_eval
    rp_min, rp_max, rp_n = rp_bin_edges

    label = (
        f"{scenario_name}_"
        f"r{radius_min:g}_{radius_max:g}_n{radius_n}_"
        f"rp{rp_min:g}_{rp_max:g}_n{rp_n}"
    )

    return label.replace(".", "p")


def scenario_plot_label(scenario_name):
    """Return a readable plot label for one scale scenario."""
    return SCENARIO_LABELS.get(scenario_name, scenario_name.replace("_", " "))


def detailed_scenario_plot_label(scenario_name):
    """Return a readable plot label including radial scale information."""
    radius_eval, rp_bin_edges = scenario_values(scenario_name)

    radius_min, radius_max, radius_n = radius_eval
    rp_min, rp_max, rp_n = rp_bin_edges

    return (
        f"{scenario_plot_label(scenario_name)} "
        f"($R={radius_min:g}$-{radius_max:g}, "
        f"$N_R={radius_n}$; "
        f"$R_p={rp_min:g}$-{rp_max:g}, "
        f"$N_{{edges}}={rp_n}$)"
    )
