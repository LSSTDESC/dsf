"""Best tomography scenario setup for DSF analysis scripts.

This config keeps only the two best-performing tomography scenarios plus the
clean baseline reference setup. It reuses the helper functions from the full
tomography sweep config instead of redefining them.
"""

from __future__ import annotations

from analysis.configs import tomography_sweep_config as full

PROJECT_ROOT = full.PROJECT_ROOT

RUN_NAME = "tomography_best_scenarios"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "runs_output" / RUN_NAME
PLOT_DIR = OUTPUT_DIR / "plots"
ARRAY_DIR = OUTPUT_DIR / "arrays"

LENS_SURVEY = full.LENS_SURVEY
SOURCE_SURVEY = full.SOURCE_SURVEY
LENS_SAMPLE = full.LENS_SAMPLE
SOURCE_YEAR = full.SOURCE_YEAR

OVERLAP_THRESHOLD = full.OVERLAP_THRESHOLD
SOURCE_BEHIND_LENS = full.SOURCE_BEHIND_LENS
CENTER_METHOD = full.CENTER_METHOD

TOMO_LEVELS = full.TOMO_LEVELS
LENS_Z_RANGES = full.LENS_Z_RANGES

SCENARIOS = {
    "baseline": full.SCENARIOS["baseline"],
    "fine_wide_z": full.SCENARIOS["fine_wide_z"],
    "stress_wide_z": full.SCENARIOS["stress_wide_z"],
}

SCENARIO_LABELS = {
    "baseline": full.SCENARIO_LABELS["baseline"],
    "fine_wide_z": full.SCENARIO_LABELS["fine_wide_z"],
    "stress_wide_z": full.SCENARIO_LABELS["stress_wide_z"],
}

SCENARIO_MARKERS = full.SCENARIO_MARKERS
Z_SCENARIO_HATCHES = full.Z_SCENARIO_HATCHES

SCENARIO_COLOR_ORDER = [
    "baseline",
    "fine_wide_z",
    "stress_wide_z",
]


scenario_color_map = full.scenario_color_map
marker_for_scenario = full.marker_for_scenario
z_scenario_for_scenario = full.z_scenario_for_scenario
hatch_for_scenario = full.hatch_for_scenario

lens_overrides = full.lens_overrides
source_overrides = full.source_overrides
case_label = full.case_label
scenario_values = full.scenario_values
detailed_scenario_plot_label = full.detailed_scenario_plot_label
