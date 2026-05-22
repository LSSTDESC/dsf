"""Run a DSF Delta Sigma SNR sweep over radial scale choices.

This script keeps one tomography setup fixed and varies only the radial scale
choices defined in ``analysis.configs.scale_sweep_config``. It is meant to
answer whether the total SNR is stable under changes in the minimum/maximum
projected radius and the number of radial bins.

Run from the project root with:

    python analysis/run_scale_snr_sweep.py

The baseline YAML config is set at the bottom of this file.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from analysis.configs.scale_scenarios import (
    ARRAY_DIR,
    OUTPUT_DIR,
    RUN_NAME,
    SCALE_SCENARIOS,
    TOMOGRAPHY_SCENARIO,
)
from analysis.configs.scale_scenarios import (
    case_label as scale_case_label,
)
from analysis.configs.scale_scenarios import (
    detailed_scenario_plot_label as detailed_scale_plot_label,
)
from analysis.configs.scale_scenarios import (
    scenario_overrides as scale_overrides,
)
from analysis.configs.scale_scenarios import (
    scenario_values as scale_values,
)
from analysis.configs.tomography_sweep_config import (
    case_label as tomography_case_label,
)
from analysis.configs.tomography_sweep_config import (
    detailed_scenario_plot_label as detailed_tomography_plot_label,
)
from analysis.configs.tomography_sweep_config import (
    lens_overrides,
    source_overrides,
)
from analysis.configs.tomography_sweep_config import (
    scenario_values as tomography_values,
)
from analysis.run_snr_check import (
    build_forecast,
    make_snr_diagnostics,
    save_forecast_outputs,
    save_snr_plots,
)
from analysis.scripts.config import (
    load_config,
)
from analysis.scripts.io import make_run_output

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def apply_fixed_tomography(config, scenario_name):
    """Apply one selected tomography scenario to a config."""
    _, _, lens_n_bins, source_n_bins, z_range = tomography_values(scenario_name)

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

    return lens_n_bins, source_n_bins, z_range


def config_for_scale_scenario(base_config, tomography_scenario, scale_name):
    """Return one config with fixed tomography and one radial scale choice."""
    config = deepcopy(base_config)

    lens_n_bins, source_n_bins, z_range = apply_fixed_tomography(
        config=config,
        scenario_name=tomography_scenario,
    )

    tomo_label = tomography_case_label(
        scenario_name=tomography_scenario,
        lens_n_bins=lens_n_bins,
        source_n_bins=source_n_bins,
        z_range=z_range,
    )

    scale_label = scale_case_label(scale_name)

    config["run_name"] = f"{tomo_label}_{scale_label}"
    config["radial_bins"] = scale_overrides(scale_name)

    config.setdefault("output", {})
    config["output"]["directory"] = str(OUTPUT_DIR)

    return config


def run_one_scale_scenario(base_config, tomography_scenario, scale_name):
    """Run one fixed-tomography radial scale scenario."""
    config = config_for_scale_scenario(
        base_config=base_config,
        tomography_scenario=tomography_scenario,
        scale_name=scale_name,
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

    radius_eval, rp_bin_edges = scale_values(scale_name)
    radius_min, radius_max, radius_n = radius_eval
    rp_min, rp_max, rp_n = rp_bin_edges

    return {
        "scale_name": scale_name,
        "run_name": config["run_name"],
        "snr": diagnostics["snr"],
        "snr2": diagnostics["snr2"],
        "n_pairs": len(forecast["bin_pairs"]),
        "radius_min": radius_min,
        "radius_max": radius_max,
        "radius_n": radius_n,
        "rp_min": rp_min,
        "rp_max": rp_max,
        "n_rp_bins": rp_n - 1,
        "rp_n_edges": rp_n,
        "run_dir": run_output.run_dir,
    }


def save_summary_table(results):
    """Save the scale-sweep SNR summary as a TSV table."""
    ARRAY_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = ARRAY_DIR / "scale_snr_summary.tsv"

    header = [
        "scale_name",
        "run_name",
        "snr",
        "snr2",
        "n_pairs",
        "radius_min",
        "radius_max",
        "radius_n",
        "rp_min",
        "rp_max",
        "n_rp_bins",
        "rp_n_edges",
        "run_dir",
    ]

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")

        for result in results:
            row = [
                result["scale_name"],
                result["run_name"],
                f"{result['snr']:.8e}",
                f"{result['snr2']:.8e}",
                str(result["n_pairs"]),
                f"{result['radius_min']:.8e}",
                f"{result['radius_max']:.8e}",
                str(result["radius_n"]),
                f"{result['rp_min']:.8e}",
                f"{result['rp_max']:.8e}",
                str(result["n_rp_bins"]),
                str(result["rp_n_edges"]),
                str(result["run_dir"]),
            ]
            f.write("\t".join(row) + "\n")

    return summary_path


def print_summary(results, tomography_scenario):
    """Print a compact scale-sweep SNR comparison."""
    print("\n[DSF] Radial scale SNR sweep")
    print(f"[DSF] Fixed tomography: {detailed_tomography_plot_label(tomography_scenario)}")
    print("-" * 116)
    print(
        f"{'scale':24s} {'radius':>20s} {'rp bins':>8s} {'n_pairs':>8s} {'SNR':>14s} {'SNR^2':>14s}"
    )
    print("-" * 116)

    for result in results:
        radius_range = f"{result['radius_min']:g}-{result['radius_max']:g}"
        print(
            f"{result['scale_name']:24s} "
            f"{radius_range:>20s} "
            f"{result['n_rp_bins']:8d} "
            f"{result['n_pairs']:8d} "
            f"{result['snr']:14.6f} "
            f"{result['snr2']:14.6e}"
        )

    print("-" * 116)


def main(path, tomography_scenario=TOMOGRAPHY_SCENARIO):
    """Run a radial scale SNR sweep for one fixed tomography setup."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_config = load_config(path)

    print(f"[DSF] Scale sweep run name: {RUN_NAME}")
    print(f"[DSF] Fixed tomography: {detailed_tomography_plot_label(tomography_scenario)}")

    results = []
    for scale_name in SCALE_SCENARIOS:
        print(f"\n[DSF] Running scale scenario: {scale_name}")
        print(f"[DSF]   {detailed_scale_plot_label(scale_name)}")

        result = run_one_scale_scenario(
            base_config=base_config,
            tomography_scenario=tomography_scenario,
            scale_name=scale_name,
        )
        results.append(result)

        print(f"[DSF]   SNR   = {result['snr']:.6f}")
        print(f"[DSF]   SNR^2 = {result['snr2']:.6e}")
        print(f"[DSF]   outputs: {result['run_dir']}")

    print_summary(
        results=results,
        tomography_scenario=tomography_scenario,
    )
    summary_path = save_summary_table(results)

    print(f"\n[DSF] Scale sweep outputs written to {OUTPUT_DIR}")
    print(f"[DSF] Summary table written to {summary_path}")

    return results


if __name__ == "__main__":
    main(
        PROJECT_ROOT / "analysis/configs/dsf_desi_lrg_lsst_y1_hod.yaml",
        tomography_scenario=TOMOGRAPHY_SCENARIO,
    )
