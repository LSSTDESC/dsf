"""Run one DSF Delta Sigma SNR check from a YAML config.

The Pk2D model is selected from the YAML ``pk2d`` block through
``analysis.scripts.model_setup.pk2d_model_func``. This allows the same SNR
runner to work for HOD-only, HOD+NLA, baryonified HOD, and other configured
model choices.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_scripts.plot_snr import (
    plot_cumulative_signal_to_noise,
    plot_signal_to_noise_by_pair,
    plot_signal_to_noise_by_radius,
)
from analysis.scripts.config import (
    geomspace_from_config,
    load_config,
)
from analysis.scripts.diagnostic_metrics import (
    cumulative_signal_to_noise,
    signal_to_noise,
    signal_to_noise_by_pair,
    signal_to_noise_by_radius,
)
from analysis.scripts.io import make_run_output
from analysis.scripts.model_setup import (
    active_parameter_names,
    active_theta0,
    make_fiducial_cosmology,
    make_theta_mapper,
    pk2d_model_func,
)
from dsf.delta_sigma_forecast_builder import DeltaSigmaForecastBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_forecast(config):
    """Build the DSF Delta Sigma forecast ingredients."""
    cosmo = make_fiducial_cosmology(config)

    r = geomspace_from_config(config["radial_bins"]["radius_eval"])
    rp_bin_edges = geomspace_from_config(config["radial_bins"]["rp_bin_edges"])

    survey = config["survey"]
    tomography = config["tomography"]
    selection = config.get("bin_pair_selection", {})
    covariance = config.get("covariance", {})

    builder = DeltaSigmaForecastBuilder(
        cosmo=cosmo,
        pk2d_func=pk2d_model_func(config),
        theta0=active_theta0(config),
        theta_mapper=make_theta_mapper(config),
        parameter_names=active_parameter_names(config),
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
        covariance_kind=covariance.get("kind", "gm"),
        nonlinear=covariance.get("nonlinear", True),
        galaxy_bias=covariance.get("galaxy_bias"),
        galaxy_bias_prefactor=covariance.get("galaxy_bias_prefactor", 1.0),
        hankel_kwargs=covariance.get("hankel_kwargs"),
    )

    return builder.forecast()


def save_forecast_outputs(run_output, forecast, diagnostics):
    """Save DSF forecast arrays and SNR diagnostics."""
    return run_output.save_arrays(
        theta0=forecast["theta0"],
        data_vector=forecast["data_vector"],
        covariance=forecast["cov"],
        r=forecast["r"],
        rp_bin_edges=forecast["rp_bin_edges"],
        bin_pairs=np.asarray(forecast["bin_pairs"], dtype=int),
        snr=np.asarray(diagnostics["snr"]),
        snr2=np.asarray(diagnostics["snr2"]),
        cumulative_snr=diagnostics["cumulative_snr"],
        cumulative_snr2=diagnostics["cumulative_snr2"],
        snr_by_pair=np.asarray(list(diagnostics["snr_by_pair"].values())),
        snr2_by_pair=np.asarray(list(diagnostics["snr2_by_pair"].values())),
        snr_by_radius=np.asarray(list(diagnostics["snr_by_radius"].values())),
        snr2_by_radius=np.asarray(list(diagnostics["snr2_by_radius"].values())),
    )


def make_snr_diagnostics(forecast):
    """Return SNR diagnostics for one DSF forecast."""
    data_vector = forecast["data_vector"]
    covariance = forecast["cov"]
    bin_pairs = forecast["bin_pairs"]
    r = forecast["r"]

    n_pairs = len(bin_pairs)
    n_radius = len(r)

    snr, snr2 = signal_to_noise(data_vector, covariance)

    snr_by_pair, snr2_by_pair = signal_to_noise_by_pair(
        data_vector=data_vector,
        covariance=covariance,
        bin_pairs=bin_pairs,
        n_radius=n_radius,
    )

    snr_by_radius, snr2_by_radius = signal_to_noise_by_radius(
        data_vector=data_vector,
        covariance=covariance,
        n_pairs=n_pairs,
        n_radius=n_radius,
    )

    cumulative_snr, cumulative_snr2 = cumulative_signal_to_noise(
        data_vector=data_vector,
        covariance=covariance,
    )

    return {
        "snr": snr,
        "snr2": snr2,
        "snr_by_pair": snr_by_pair,
        "snr2_by_pair": snr2_by_pair,
        "snr_by_radius": snr_by_radius,
        "snr2_by_radius": snr2_by_radius,
        "cumulative_snr": cumulative_snr,
        "cumulative_snr2": cumulative_snr2,
    }


def save_snr_plots(run_output, forecast):
    """Save SNR diagnostic plots for one forecast."""
    data_vector = forecast["data_vector"]
    covariance = forecast["cov"]
    bin_pairs = forecast["bin_pairs"]
    r = forecast["r"]

    n_radius = len(r)

    fig, _ = plot_signal_to_noise_by_pair(
        data_vector=data_vector,
        covariance=covariance,
        bin_pairs=bin_pairs,
        n_radius=n_radius,
    )
    fig.savefig(run_output.plot_dir / "snr_by_pair.png", dpi=200)
    plt.close(fig)

    fig, _ = plot_signal_to_noise_by_radius(
        r=r,
        data_vector=data_vector,
        covariance=covariance,
        bin_pairs=bin_pairs,
    )
    fig.savefig(run_output.plot_dir / "snr_by_radius.png", dpi=200)
    plt.close(fig)

    fig, _ = plot_cumulative_signal_to_noise(
        data_vector=data_vector,
        covariance=covariance,
    )
    fig.savefig(run_output.plot_dir / "cumulative_snr.png", dpi=200)
    plt.close(fig)


def main(path):
    """Run one HOD-backed DSF Delta Sigma SNR check from a YAML config."""
    config = load_config(path)

    output_config = config.get("output", {})
    output_directory = output_config.get("directory")

    if output_directory is None:
        run_output = make_run_output(run_name=config["run_name"])
    else:
        run_output = make_run_output(
            run_name=config["run_name"],
            base_dir=output_directory,
        )

    forecast = build_forecast(config)
    diagnostics = make_snr_diagnostics(forecast)

    print("\n[DSF] SNR summary")
    print(f"  SNR^2 = {diagnostics['snr2']:.6e}")
    print(f"  SNR   = {diagnostics['snr']:.6f}")

    if output_config.get("save_arrays", True):
        save_forecast_outputs(
            run_output=run_output,
            forecast=forecast,
            diagnostics=diagnostics,
        )

    if output_config.get("save_plots", True):
        save_snr_plots(
            run_output=run_output,
            forecast=forecast,
        )

    print(f"\n[DSF] Outputs written to {run_output.run_dir}")

    return {
        "forecast": forecast,
        "diagnostics": diagnostics,
        "run_output": run_output,
    }


if __name__ == "__main__":
    main(PROJECT_ROOT / "analysis/configs/dsf_desi_lrg_lsst_y1_hod.yaml")
