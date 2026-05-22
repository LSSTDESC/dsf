"""Redo tomography SNR plots from saved forecast arrays.

This script reloads saved arrays from the tomography scenario outputs and
regenerates the plots without rebuilding forecasts or covariances.

Run from the project root with:

    python analysis/redo_tomography_snr_plots.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.configs.tomography_sweep_config import scenario_name_from_row
from analysis.plot_scripts.plot_snr import (
    plot_cumulative_signal_to_noise,
    plot_signal_to_noise_by_pair,
    plot_signal_to_noise_by_radius,
    plot_snr_by_radius_for_scenarios,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "runs_output" / "tomography_scenarios"
SUMMARY_PATH = OUTPUT_DIR / "arrays" / "tomography_snr_summary.tsv"
PLOT_DIR = OUTPUT_DIR / "plots"


def load_array(path):
    """Load one saved numpy array."""
    return np.load(path, allow_pickle=True)


def load_saved_dict(path):
    """Load one saved dictionary-like numpy array."""
    value = np.load(path, allow_pickle=True)

    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()

    return value


def load_label_map():
    """Return run-name to label mapping from the saved summary table."""
    if not SUMMARY_PATH.exists():
        return {}

    label_map = {}

    with SUMMARY_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            label_map[row["run_name"]] = row["label"]

    return label_map


def scenario_dirs():
    """Return tomography scenario directories with saved arrays."""
    skip = {"arrays", "plots"}

    return sorted(
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_dir() and path.name not in skip and (path / "arrays").exists()
    )


def save_plot(fig, path):
    """Save one plot and close the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def scenario_from_run_dir(run_dir):
    """Return the tomography scenario name encoded in one run directory."""
    return scenario_name_from_row({"label": run_dir.name})


def redo_one_scenario(run_dir):
    """Regenerate the individual SNR plots for one scenario."""
    array_dir = run_dir / "arrays"
    plot_dir = run_dir / "plots"

    scenario = scenario_from_run_dir(run_dir)

    r = load_array(array_dir / "r.npy")
    data_vector = load_array(array_dir / "data_vector.npy")
    covariance = load_array(array_dir / "covariance.npy")
    bin_pairs = load_array(array_dir / "bin_pairs.npy")

    n_radius = len(np.asarray(r).reshape(-1))

    fig, _ = plot_signal_to_noise_by_pair(
        data_vector=data_vector,
        covariance=covariance,
        bin_pairs=bin_pairs,
        n_radius=n_radius,
        scenario=scenario,
    )
    save_plot(fig, plot_dir / "snr_by_pair.png")

    fig, _ = plot_signal_to_noise_by_radius(
        r=r,
        data_vector=data_vector,
        covariance=covariance,
        bin_pairs=bin_pairs,
        scenario=scenario,
    )
    save_plot(fig, plot_dir / "snr_by_radius.png")

    fig, _ = plot_cumulative_signal_to_noise(
        data_vector=data_vector,
        covariance=covariance,
        bin_pairs=bin_pairs,
        r=r,
        scenario=scenario,
    )
    save_plot(fig, plot_dir / "cumulative_snr.png")


def load_combined_radius_row(run_dir, label):
    """Load saved radius SNR arrays for the combined scenario plot."""
    array_dir = run_dir / "arrays"
    scenario = scenario_from_run_dir(run_dir)

    return {
        "scenario": scenario,
        "label": label,
        "r": load_array(array_dir / "r.npy"),
        "snr_by_radius": load_saved_dict(array_dir / "snr_by_radius.npy"),
    }


def main():
    """Redo all tomography SNR plots from saved arrays."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map()
    rows = []

    for run_dir in scenario_dirs():
        label = label_map.get(run_dir.name, run_dir.name.replace("_", " "))

        print(f"[DSF] Redrawing plots for {run_dir.name}")

        redo_one_scenario(run_dir)

        rows.append(
            load_combined_radius_row(
                run_dir=run_dir,
                label=label,
            )
        )

    fig, _ = plot_snr_by_radius_for_scenarios(rows)
    if fig is not None:
        save_plot(fig, PLOT_DIR / "snr_by_radius_all_scenarios.png")

    print("\n[DSF] Redrawn plots from saved arrays.")
    print(f"[DSF] Combined plot: {PLOT_DIR / 'snr_by_radius_all_scenarios.png'}")


if __name__ == "__main__":
    main()
