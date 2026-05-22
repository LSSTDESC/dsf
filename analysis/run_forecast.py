"""Run DSF Delta Sigma forecasts from a YAML configuration file.

This script can be used in two modes.

Run one forecast directly from the YAML defaults:

    python analysis/run_forecast.py \
        analysis/configs/dsf_desi_lrg_lsst_y1_hod_baryons.yaml

Run the same YAML through tomography overrides from a Python config module:

    python analysis/run_forecast.py \
        analysis/configs/dsf_desi_lrg_lsst_y1_hod_baryons.yaml \
        --scenario-config analysis.configs.tomography_best_scenarios_config

Run only selected scenarios from that config module:

    python analysis/run_forecast.py \
        analysis/configs/dsf_desi_lrg_lsst_y1_hod_baryons.yaml \
        --scenario-config analysis.configs.tomography_best_scenarios_config \
        --scenarios baseline fine_wide_z stress_wide_z

If Python cannot find the local ``dsf`` package and raises::

    ModuleNotFoundError: No module named 'dsf'

then run from the repository root with ``src`` on ``PYTHONPATH``::

    PYTHONPATH=src python analysis/run_forecast.py \
        analysis/configs/dsf_desi_lrg_lsst_y1_hod_baryons.yaml \
        --scenario-config analysis.configs.tomography_best_scenarios_config

or, equivalently::

    PYTHONPATH=src python -m analysis.run_forecast \
        analysis/configs/dsf_desi_lrg_lsst_y1_hod_baryons.yaml \
        --scenario-config analysis.configs.tomography_best_scenarios_config

The YAML controls the physical model, cosmology, parameters, HOD settings,
optional IA and baryonic parameters, radial bins, covariance settings, Fisher
settings, DALI settings, and default output options.

When a scenario config is passed, the base YAML is deep-copied for each
scenario. The scenario config only overrides tomography choices, bin-pair
selection, output directory, and the run name suffix.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np

from analysis.run_snr_check import make_snr_diagnostics
from analysis.scripts.config import (
    geomspace_from_config,
    load_config,
)
from analysis.scripts.forecast import (
    run_dali,
    run_fisher,
    sample_dali,
)
from analysis.scripts.io import make_run_output
from analysis.scripts.model_setup import (
    active_parameter_labels,
    active_parameter_names,
    active_theta0,
    make_fiducial_cosmology,
    make_theta_mapper,
    pk2d_model_func,
)
from analysis.scripts.plot_run_outputs import make_run_plots
from dsf.delta_sigma_forecast_builder import DeltaSigmaForecastBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "analysis/configs/dsf_desi_lrg_lsst_y1_hod.yaml"


def build_forecast(config, covariance=None):
    """Build the DSF Delta Sigma forecast ingredients."""
    cosmo = make_fiducial_cosmology(config)

    r = geomspace_from_config(config["radial_bins"]["radius_eval"])
    rp_bin_edges = geomspace_from_config(config["radial_bins"]["rp_bin_edges"])

    theta_fiducial = active_theta0(config)
    names = active_parameter_names(config)

    survey = config["survey"]
    tomography = config["tomography"]
    selection = config.get("bin_pair_selection", {})
    covariance_config = config.get("covariance", {})

    builder = DeltaSigmaForecastBuilder(
        cosmo=cosmo,
        pk2d_func=pk2d_model_func(config),
        theta0=theta_fiducial,
        theta_mapper=make_theta_mapper(config),
        parameter_names=names,
        r=r,
        rp_bin_edges=rp_bin_edges,
        area_deg2=survey["area_deg2"],
        sigma_e=survey.get("sigma_e", 0.26),
        lens_survey=survey.get("lens_survey", "lsst"),
        source_survey=survey.get("source_survey", "lsst"),
        lens_role=tomography["lens"].get("role", "lens"),
        source_role=tomography["source"].get("role", "source"),
        lens_sample=tomography["lens"].get("sample"),
        source_sample=tomography["source"].get("sample"),
        lens_year=tomography["lens"].get("year"),
        source_year=tomography["source"].get("year"),
        lens_overrides=tomography["lens"].get("overrides"),
        source_overrides=tomography["source"].get("overrides"),
        overlap_threshold=selection.get("overlap_threshold", 0.10),
        source_behind_lens=selection.get("source_behind_lens", True),
        center_method=selection.get("center_method", "mean"),
        covariance_kind=covariance_config.get("kind", "gm"),
        nonlinear=covariance_config.get("nonlinear", True),
        galaxy_bias=covariance_config.get("galaxy_bias"),
        galaxy_bias_prefactor=covariance_config.get("galaxy_bias_prefactor", 1.0),
    )

    return builder.forecast(covariance=covariance)


def load_saved_covariance_if_requested(run_output, output_config):
    """Load a saved covariance matrix when covariance reuse is requested."""
    if not output_config.get("reuse_covariance", False):
        return None

    covariance_path = run_output.array_dir / "covariance.npy"

    if not covariance_path.exists():
        print(f"[DSF] No saved covariance found at {covariance_path}; computing covariance.")
        return None

    print(f"[DSF] Reusing saved covariance from {covariance_path}")
    return np.load(covariance_path, allow_pickle=True)


def save_forecast_outputs(run_output, forecast, diagnostics=None):
    """Save DSF forecast and optional SNR diagnostic arrays."""
    saved = run_output.save_arrays(
        theta0=forecast["theta0"],
        data_vector=forecast["data_vector"],
        covariance=forecast["cov"],
        r=forecast["r"],
        rp_bin_edges=forecast["rp_bin_edges"],
        bin_pairs=np.asarray(forecast["bin_pairs"], dtype=int),
    )

    if diagnostics is not None:
        saved.update(
            run_output.save_arrays(
                snr=diagnostics["snr"],
                snr2=diagnostics["snr2"],
                snr_by_pair=diagnostics["snr_by_pair"],
                snr2_by_pair=diagnostics["snr2_by_pair"],
                snr_by_radius=diagnostics["snr_by_radius"],
                snr2_by_radius=diagnostics["snr2_by_radius"],
                cumulative_snr=diagnostics["cumulative_snr"],
                cumulative_snr2=diagnostics["cumulative_snr2"],
            )
        )

    return saved


def save_forecast_results(run_output, fisher_result=None, dali_result=None):
    """Save Fisher and DALI forecast results."""
    saved = {}

    if fisher_result is not None:
        saved["fisher"] = run_output.save_array("fisher", fisher_result)

    if dali_result is not None:
        saved["dali"] = run_output.save_array("dali", dali_result)

    return saved


def run_one_config(config):
    """Run one forecast from an already-loaded config dictionary."""
    names = active_parameter_names(config)
    labels = active_parameter_labels(config)

    output_config = config.get("output", {})
    output_directory = output_config.get("directory")

    if output_directory is None:
        run_output = make_run_output(run_name=config["run_name"])
    else:
        run_output = make_run_output(
            run_name=config["run_name"],
            base_dir=output_directory,
        )

    saved_covariance = load_saved_covariance_if_requested(
        run_output=run_output,
        output_config=output_config,
    )

    forecast = build_forecast(config, covariance=saved_covariance)
    diagnostics = make_snr_diagnostics(forecast)

    if output_config.get("save_arrays", True):
        save_forecast_outputs(
            run_output=run_output,
            forecast=forecast,
            diagnostics=diagnostics,
        )

    fisher_kit = None
    fisher_result = None

    fisher_output = run_fisher(
        model=forecast["model"],
        theta0=forecast["theta0"],
        covariance=forecast["cov"],
        fisher_config=config.get("fisher", {}),
        parameter_names=names,
    )

    if fisher_output is not None:
        fisher_kit, fisher_result = fisher_output

    dali_result = None
    dali_samples = None

    dali_output = run_dali(
        model=forecast["model"],
        theta0=forecast["theta0"],
        covariance=forecast["cov"],
        dali_config=config.get("dali", {}),
        parameter_names=names,
    )

    if dali_output is not None:
        dali_kit, dali_result = dali_output

        dali_samples = sample_dali(
            config=config,
            kit=dali_kit,
            dali=dali_result,
            names=names,
            labels=labels,
        )

    if output_config.get("save_arrays", True):
        save_forecast_results(
            run_output,
            fisher_result=fisher_result,
            dali_result=dali_result,
        )

        if dali_samples is not None:
            run_output.save_samples("dali_samples", dali_samples)

    if output_config.get("make_plots", True):
        make_run_plots(
            run_output,
            config,
            forecast,
            fisher_kit=fisher_kit,
            fisher_result=fisher_result,
            dali_samples=dali_samples,
        )

    print(f"[DSF] Outputs written to {run_output.run_dir}")

    return {
        "forecast": forecast,
        "diagnostics": diagnostics,
        "fisher": fisher_result,
        "dali": dali_result,
        "dali_samples": dali_samples,
        "run_output": run_output,
    }


def load_python_config(config_reference):
    """Load a Python config module from a dotted module path or file path."""
    path = Path(config_reference)

    if path.exists():
        module_name = path.stem
        spec = importlib.util.spec_from_file_location(module_name, path)

        if spec is None or spec.loader is None:
            msg = f"Could not load scenario config from {path}"
            raise ImportError(msg)

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return importlib.import_module(config_reference)


def apply_scenario_overrides(*, base_config, scenario_name, scenario_config):
    """Return a forecast config with one tomography scenario applied."""
    config = deepcopy(base_config)

    _, _, lens_n_bins, source_n_bins, z_range = scenario_config.scenario_values(scenario_name)

    label = scenario_config.case_label(
        scenario_name=scenario_name,
        lens_n_bins=lens_n_bins,
        source_n_bins=source_n_bins,
        z_range=z_range,
    )

    config["run_name"] = label

    config.setdefault("tomography", {})
    config["tomography"].setdefault("lens", {})
    config["tomography"].setdefault("source", {})

    config["tomography"]["lens"]["overrides"] = scenario_config.lens_overrides(
        n_bins=lens_n_bins,
        z_range=z_range,
    )

    config["tomography"]["source"]["overrides"] = scenario_config.source_overrides(
        n_bins=source_n_bins,
    )

    return config


def run_scenario_configs(base_config, scenario_config, scenario_names=None):
    """Run one base YAML through scenarios from a Python config module."""
    if not hasattr(scenario_config, "SCENARIOS"):
        msg = "Scenario config must define SCENARIOS."
        raise AttributeError(msg)

    available_scenarios = scenario_config.SCENARIOS

    if scenario_names is None or len(scenario_names) == 0:
        scenario_names = list(available_scenarios.keys())

    results = {}

    for scenario_name in scenario_names:
        if scenario_name not in available_scenarios:
            available = ", ".join(available_scenarios.keys())
            msg = f"Unknown scenario '{scenario_name}'. Available scenarios are: {available}"
            raise KeyError(msg)

        scenario = available_scenarios[scenario_name]
        labels = getattr(scenario_config, "SCENARIO_LABELS", {})
        label = labels.get(scenario_name, scenario_name)

        print("")
        print(f"[DSF] Running scenario: {scenario_name}")
        print(f"[DSF] Label: {label}")

        config = apply_scenario_overrides(
            base_config=base_config,
            scenario_name=scenario_name,
            scenario_config=scenario_config,
        )

        results[scenario_name] = run_one_config(config)

    return results


def default_output_directory_from_config_path(path):
    """Return the default run-output directory for one YAML config."""
    config_stem = Path(path).stem
    return PROJECT_ROOT / "analysis" / "runs_output" / config_stem


def main(path, scenario_config=None, scenarios=None, run_dali_cli=None):
    """Run one YAML directly or through optional scenario overrides."""
    base_config = load_config(path)

    base_config.setdefault("output", {})

    config_stem = Path(path).stem
    base_config["output"]["directory"] = str(
        PROJECT_ROOT / "analysis" / "runs_output" / config_stem
    )

    if run_dali_cli is not None:
        base_config.setdefault("dali", {})
        base_config["dali"]["run"] = bool(run_dali_cli)

    if scenario_config is None:
        return run_one_config(base_config)

    scenario_config_module = load_python_config(scenario_config)

    return run_scenario_configs(
        base_config=base_config,
        scenario_config=scenario_config_module,
        scenario_names=scenarios,
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run DSF Delta Sigma forecasts from a YAML config."
    )

    parser.add_argument(
        "config",
        nargs="?",
        default=DEFAULT_CONFIG,
        type=Path,
        help="Path to the YAML forecast config.",
    )

    parser.add_argument(
        "--scenario-config",
        default=None,
        help=(
            "Optional Python config module or file path containing SCENARIOS, "
            "lens_overrides, source_overrides, and selection settings."
        ),
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help=(
            "Optional list of scenario names to run from the scenario config. "
            "If omitted, all scenarios in SCENARIOS are run."
        ),
    )

    parser.add_argument(
        "--run-dali",
        action="store_true",
        default=None,
        help="Run DALI forecasts, overriding dali.run in the YAML.",
    )

    parser.add_argument(
        "--no-dali",
        action="store_false",
        dest="run_dali",
        help="Skip DALI forecasts, overriding dali.run in the YAML.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        path=args.config,
        scenario_config=args.scenario_config,
        scenarios=args.scenarios,
        run_dali_cli=args.run_dali,
    )
