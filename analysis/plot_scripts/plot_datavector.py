"""Diagnostic plot for Delta Sigma data vectors."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

from analysis.plot_scripts.plot_style import apply_dsf_plot_style, binny_colors


def plot_delta_sigma_data_vector(
    rp,
    data_vector,
    cov,
    bin_pairs,
    *,
    save_path=None,
    figsize=None,
    dpi=300,
):
    """Plot the fiducial Delta Sigma data vector with covariance errors.

    Args:
        rp: Projected-radius values.
        data_vector: Flattened Delta Sigma data vector.
        cov: Covariance matrix matching the flattened data vector.
        bin_pairs: Lens-source bin pairs corresponding to the data-vector blocks.
        save_path: Optional output path for saving the figure.
        figsize: Optional Matplotlib figure size.
        dpi: Resolution used when saving the figure.

    Returns:
        Matplotlib figure and axis.
    """
    apply_dsf_plot_style()

    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    rp = np.asarray(rp).reshape(-1)
    data_vector = np.asarray(data_vector).reshape(-1)
    errors = np.sqrt(np.diag(cov))

    n_rp = len(rp)
    n_pairs = len(bin_pairs)

    dv_by_pair = data_vector.reshape(n_pairs, n_rp)
    err_by_pair = errors.reshape(n_pairs, n_rp)

    colors = binny_colors(
        n_pairs,
        cmap="viridis",
        cmap_range=(0.15, 0.95),
    )

    for pair, dv, err, color in zip(
        bin_pairs,
        dv_by_pair,
        err_by_pair,
        colors,
        strict=True,
    ):
        ax.errorbar(
            rp,
            rp * dv,
            yerr=rp * err,
            marker="o",
            markersize=5,
            linewidth=2.2,
            elinewidth=1.5,
            capsize=3,
            color=color,
            markeredgecolor="k",
            markeredgewidth=0.8,
            label=rf"lens {pair[0]}, source {pair[1]}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=80))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))

    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=8))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=80))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

    ax.tick_params(axis="x", which="major", length=6)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.tick_params(axis="y", which="major", length=6)
    ax.tick_params(axis="y", which="minor", length=3)

    ax.set_xlabel(r"$R_p\,[h^{-1}{\rm Mpc}]$")
    ax.set_ylabel(r"$R_p\,\Delta\Sigma(R_p)$")
    ax.set_title(r"Fiducial $\Delta\Sigma$ data vector")
    ax.legend(frameon=False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")

    return fig, ax


def plot_delta_sigma_fractional_errors(
    rp,
    data_vector,
    cov,
    bin_pairs,
    *,
    save_path=None,
    figsize=None,
    dpi=300,
):
    """Plot fractional covariance errors for the Delta Sigma data vector."""
    apply_dsf_plot_style()

    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    rp = np.asarray(rp).reshape(-1)
    data_vector = np.asarray(data_vector).reshape(-1)
    errors = np.sqrt(np.diag(cov))

    n_rp = len(rp)
    n_pairs = len(bin_pairs)

    dv_by_pair = data_vector.reshape(n_pairs, n_rp)
    err_by_pair = errors.reshape(n_pairs, n_rp)

    colors = binny_colors(
        n_pairs,
        cmap="viridis",
        cmap_range=(0.15, 0.95),
    )

    for pair, dv, err, color in zip(
        bin_pairs,
        dv_by_pair,
        err_by_pair,
        colors,
        strict=True,
    ):
        frac_err = err / np.abs(dv)

        ax.plot(
            rp,
            frac_err,
            marker="o",
            markersize=5,
            linewidth=2.2,
            color=color,
            markeredgecolor="k",
            markeredgewidth=0.8,
            label=rf"lens {pair[0]}, source {pair[1]}",
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"$R_p\,[h^{-1}{\rm Mpc}]$")
    ax.set_ylabel(r"$\sigma_{\Delta\Sigma}/|\Delta\Sigma|$")
    ax.set_title(r"Fractional $\Delta\Sigma$ covariance errors")
    ax.legend(frameon=False)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")

    return fig, ax
