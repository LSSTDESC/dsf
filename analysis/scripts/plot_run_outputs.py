"""Plot helpers for completed DSF forecast runs."""

import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_scripts.plot_contours import (
    fisher_to_gaussian,
    plot_dali_contours,
    plot_fisher_contours,
    plot_forecast_contours,
)
from analysis.plot_scripts.plot_covariance import (
    plot_covariance_correlation,
    plot_covariance_diagonal,
)
from analysis.plot_scripts.plot_datavector import (
    plot_delta_sigma_data_vector,
    plot_delta_sigma_fractional_errors,
)
from analysis.plot_scripts.plot_snr import (
    plot_cumulative_signal_to_noise,
    plot_signal_to_noise_by_pair,
    plot_signal_to_noise_by_radius,
)
from analysis.plot_scripts.plot_tomobins import plot_lens_and_source_bins
from analysis.scripts.config import parameter_labels, parameter_names


def _save_and_close(fig, save_path):
    """Save and close one Matplotlib figure."""
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _theta0_from_config(config):
    """Return the fiducial parameter vector from the forecast config."""
    return np.array(
        [parameter["fiducial"] for parameter in config["parameters"]],
        dtype=float,
    )


def _fisher_matrix_from_result(fisher_result, *, with_prior=False):
    """Return the numeric Fisher matrix from an array or Fisher result dictionary."""
    if fisher_result is None:
        return None

    if not isinstance(fisher_result, dict):
        return fisher_result

    if with_prior:
        for key in ("fisher_with_prior", "fisher_prior", "with_prior"):
            if key in fisher_result:
                return fisher_result[key]

    for key in ("fisher", "fisher_matrix", "matrix"):
        if key in fisher_result:
            return fisher_result[key]

    raise KeyError(
        "Could not find a Fisher matrix in fisher_result. "
        f"Available keys are: {list(fisher_result.keys())}"
    )


def plot_run_tomography(run_output, forecast):
    """Plot lens and source tomography when available."""
    lens_result = forecast.get("lens_result")
    source_result = forecast.get("source_result")

    if lens_result is None or source_result is None:
        return

    fig, _ = plot_lens_and_source_bins(
        lens_result,
        source_result,
        save_path=run_output.plot_path("tomography_lens_source_bins.png"),
        figsize=(8.0, 5.0),
    )
    plt.close(fig)


def plot_run_diagnostics(run_output, forecast):
    """Make all single-run Delta Sigma diagnostic plots."""
    r = forecast["r"]
    data_vector = forecast["data_vector"]
    covariance = forecast["cov"]
    bin_pairs = forecast["bin_pairs"]
    n_radius = len(np.asarray(r).reshape(-1))

    fig, _ = plot_delta_sigma_data_vector(
        r,
        data_vector,
        covariance,
        bin_pairs,
        save_path=run_output.plot_path("delta_sigma_data_vector.png"),
    )
    plt.close(fig)

    fig, _ = plot_delta_sigma_fractional_errors(
        r,
        data_vector,
        covariance,
        bin_pairs,
        save_path=run_output.plot_path("fractional_uncertainty.png"),
    )
    plt.close(fig)

    fig, _ = plot_signal_to_noise_by_pair(
        data_vector,
        covariance,
        bin_pairs,
        n_radius,
    )
    _save_and_close(fig, run_output.plot_path("snr_by_pair.png"))

    fig, _ = plot_signal_to_noise_by_radius(
        r,
        data_vector,
        covariance,
        bin_pairs,
    )
    _save_and_close(fig, run_output.plot_path("snr_by_radius.png"))

    fig, _ = plot_cumulative_signal_to_noise(
        data_vector,
        covariance,
        bin_pairs=bin_pairs,
        r=r,
    )
    _save_and_close(fig, run_output.plot_path("cumulative_snr.png"))

    fig, _, _ = plot_covariance_correlation(
        covariance,
        save_path=run_output.plot_path("covariance_correlation_matrix.png"),
        figsize=(5.5, 5.0),
    )
    plt.close(fig)

    fig, _ = plot_covariance_diagonal(
        covariance,
        save_path=run_output.plot_path("covariance_diagonal.png"),
        figsize=(6.0, 4.5),
    )
    plt.close(fig)


def plot_run_contours(
    run_output,
    config,
    *,
    fisher_kit=None,
    fisher_result=None,
    dali_samples=None,
):
    """Make Fisher and DALI triangle plots for one run."""
    names = parameter_names(config)
    labels = parameter_labels(config)
    theta0 = _theta0_from_config(config)

    fisher_gaussian = None

    if fisher_result is not None:
        fisher_matrix = _fisher_matrix_from_result(
            fisher_result,
            with_prior=False,
        )

        fisher_gaussian = fisher_to_gaussian(
            fisher_matrix,
            theta0,
            names,
            labels=labels,
            label="Fisher",
        )

        plotter = plot_fisher_contours(
            [fisher_gaussian],
            names,
            labels=["Fisher"],
            save_stem=run_output.plot_path("fisher_triangle"),
            width_inch=5.0,
            line_width=2.0,
        )
        plt.close(plotter.fig)

    if dali_samples is not None:
        plotter = plot_dali_contours(
            [dali_samples],
            names,
            labels=["DALI"],
            save_stem=run_output.plot_path("dali_triangle"),
            width_inch=5.0,
            line_width=2.0,
        )
        plt.close(plotter.fig)

    if fisher_gaussian is not None and dali_samples is not None:
        plotter = plot_forecast_contours(
            [fisher_gaussian, dali_samples],
            names,
            labels=["Fisher", "DALI"],
            save_stem=run_output.plot_path("fisher_dali_triangle"),
            width_inch=5.0,
            line_width=2.0,
            filled=[False, False],
        )
        plt.close(plotter.fig)


def make_run_plots(
    run_output,
    config,
    forecast,
    *,
    fisher_kit=None,
    fisher_result=None,
    dali_samples=None,
):
    """Make all plots for one completed forecast run."""
    plot_run_tomography(run_output, forecast)
    plot_run_diagnostics(run_output, forecast)
    plot_run_contours(
        run_output,
        config,
        fisher_kit=fisher_kit,
        fisher_result=fisher_result,
        dali_samples=dali_samples,
    )
