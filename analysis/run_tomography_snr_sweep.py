"""Run a DSF Delta Sigma SNR sweep over tomography scenarios.

This script loads one baseline YAML config and applies the tomography scenarios
defined in ``analysis.scripts.tomography_scenarios``. Each scenario changes the
lens/source binning and lens redshift range, then runs the same Delta Sigma SNR
calculation so the resulting signal-to-noise values can be compared.

Run from the project root with:

    python analysis/run_tomography_snr_sweep.py

The baseline config is currently set at the bottom of this file.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from analysis.configs.tomography_sweep_config import (
    ARRAY_DIR,
    OUTPUT_DIR,
    PLOT_DIR,
    SCENARIOS,
    case_label,
    detailed_scenario_plot_label,
    lens_overrides,
    scenario_values,
    source_overrides,
)
from analysis.run_snr_check import (
    build_forecast,
    make_snr_diagnostics,
    save_forecast_outputs,
    save_snr_plots,
)
from analysis.scripts.config import load_config
from analysis.scripts.io import make_run_output

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_output_directories():
    """Create output directories for the tomography sweep."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    ARRAY_DIR.mkdir(parents=True, exist_ok=True)


def config_for_scenario(base_config, scenario_name):
    """Return a config dictionary for one tomography scenario."""
    config = deepcopy(base_config)

    _, _, lens_n_bins, source_n_bins, z_range = scenario_values(scenario_name)
    label = case_label(
        scenario_name=scenario_name,
        lens_n_bins=lens_n_bins,
        source_n_bins=source_n_bins,
        z_range=z_range,
    )

    config["run_name"] = label

    config.setdefault("tomography", {})
    config["tomography"].setdefault("lens", {})
    config["tomography"].setdefault("source", {})

    config["tomography"]["lens"]["overrides"] = lens_overrides(
        n_bins=lens_n_bins,
        z_range=z_range,
    )
    config["tomography"]["source"]["overrides"] = source_overrides(
        n_bins=source_n_bins,
    )

    config.setdefault("output", {})
    config["output"]["directory"] = str(OUTPUT_DIR)

    return config


def run_one_scenario(base_config, scenario_name):
    """Run one tomography scenario and return its SNR summary."""
    config = config_for_scenario(
        base_config=base_config,
        scenario_name=scenario_name,
    )

    run_output = make_run_output(
        run_name=config["run_name"],
        base_dir=config["output"]["directory"],
    )

    forecast = build_forecast(config)
    diagnostics = make_snr_diagnostics(forecast)

    if config.get("output", {}).get("save_arrays", True):
        save_forecast_outputs(
            run_output=run_output,
            forecast=forecast,
            diagnostics=diagnostics,
        )

    if config.get("output", {}).get("save_plots", True):
        save_snr_plots(
            run_output=run_output,
            forecast=forecast,
        )

    return {
        "scenario": scenario_name,
        "label": detailed_scenario_plot_label(scenario_name),
        "run_name": config["run_name"],
        "snr": diagnostics["snr"],
        "snr2": diagnostics["snr2"],
        "n_pairs": len(forecast["bin_pairs"]),
        "run_dir": run_output.run_dir,
    }


def save_summary_table(results):
    """Save the tomography sweep SNR summary as a text table."""
    summary_path = ARRAY_DIR / "tomography_snr_summary.tsv"

    header = [
        "scenario",
        "label",
        "run_name",
        "snr",
        "snr2",
        "n_pairs",
        "run_dir",
    ]

    rows = []
    for result in results:
        rows.append(
            [
                result["scenario"],
                result["label"],
                result["run_name"],
                f"{result['snr']:.8e}",
                f"{result['snr2']:.8e}",
                str(result["n_pairs"]),
                str(result["run_dir"]),
            ]
        )

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")

    return summary_path


def print_summary(results):
    """Print a compact SNR comparison table."""
    print("\n[DSF] Tomography SNR sweep summary")
    print("-" * 92)
    print(f"{'scenario':28s} {'n_pairs':>8s} {'SNR':>14s} {'SNR^2':>14s}")
    print("-" * 92)

    for result in results:
        print(
            f"{result['scenario']:28s} "
            f"{result['n_pairs']:8d} "
            f"{result['snr']:14.6f} "
            f"{result['snr2']:14.6e}"
        )

    print("-" * 92)


def main(path):
    """Run the tomography SNR sweep from one baseline YAML config."""
    ensure_output_directories()

    base_config = load_config(path)

    results = []
    for scenario_name in SCENARIOS:
        print(f"\n[DSF] Running tomography scenario: {scenario_name}")
        result = run_one_scenario(
            base_config=base_config,
            scenario_name=scenario_name,
        )
        results.append(result)

        print(f"[DSF]   SNR   = {result['snr']:.6f}")
        print(f"[DSF]   SNR^2 = {result['snr2']:.6e}")
        print(f"[DSF]   outputs: {result['run_dir']}")

    print_summary(results)
    summary_path = save_summary_table(results)

    print(f"\n[DSF] Sweep outputs written to {OUTPUT_DIR}")
    print(f"[DSF] Summary table written to {summary_path}")

    return results


if __name__ == "__main__":
    main(PROJECT_ROOT / "analysis/configs/dsf_desi_lrg_lsst_y1_hod.yaml")
