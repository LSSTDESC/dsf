"""Triangle plots for Fisher and DALI forecast comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from getdist import plots as getdist_plots
from getdist.gaussian_mixtures import GaussianND

from analysis.plot_scripts.plot_style import binny_colors


def fisher_to_gaussian(
    fisher,
    theta0,
    names,
    *,
    labels=None,
    label=None,
):
    """Convert one Fisher matrix to a GetDist Gaussian approximation."""
    fisher = np.asarray(fisher, dtype=float)
    theta0 = np.asarray(theta0, dtype=float)

    if fisher.ndim != 2:
        raise ValueError(f"Fisher matrix must be 2D, got shape {fisher.shape}.")

    if fisher.shape[0] != fisher.shape[1]:
        raise ValueError(f"Fisher matrix must be square, got shape {fisher.shape}.")

    if fisher.shape[0] != theta0.size:
        raise ValueError(
            "Fisher matrix size does not match theta0. "
            f"Got Fisher shape {fisher.shape} and theta0 size {theta0.size}."
        )

    if fisher.shape[0] != len(names):
        raise ValueError(
            "Fisher matrix size does not match parameter names. "
            f"Got Fisher shape {fisher.shape} and {len(names)} names."
        )

    if labels is None:
        labels = names

    covariance = np.linalg.inv(fisher)

    return GaussianND(
        mean=theta0,
        cov=covariance,
        names=names,
        labels=labels,
        label=label,
    )


def make_forecast_plotter(width_inch=8.0, line_width=1.2):
    """Return a configured GetDist subplot plotter."""
    plotter = getdist_plots.get_subplot_plotter(width_inch=width_inch)

    plotter.settings.linewidth_contour = line_width
    plotter.settings.linewidth = line_width
    plotter.settings.figure_legend_frame = False
    plotter.settings.legend_rect_border = False

    return plotter


def save_getdist_plot(plotter, save_stem, *, close=True):
    """Save a GetDist figure as PNG and PDF, then optionally close it."""
    if save_stem is None:
        if close and plotter is not None and plotter.fig is not None:
            plt.close(plotter.fig)
        return

    save_stem = Path(save_stem)
    save_stem.parent.mkdir(parents=True, exist_ok=True)

    plotter.fig.savefig(save_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plotter.fig.savefig(save_stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")

    if close:
        plt.close(plotter.fig)


def plot_forecast_contours(
    contours,
    names,
    *,
    save_stem=None,
    labels=None,
    contour_colors=None,
    width_inch=8.0,
    line_width=1.2,
    contour_ls=None,
    close=True,
):
    """Plot Fisher Gaussian objects and/or DALI GetDist samples."""
    contours = [contour for contour in contours if contour is not None]

    if len(contours) == 0:
        return None

    n_contours = len(contours)

    if contour_colors is None:
        contour_colors = binny_colors(
            n_contours,
            cmap="viridis",
            cmap_range=(0.20, 0.80),
        )

    if contour_ls is None:
        contour_ls = ["-"] * n_contours

    plotter = make_forecast_plotter(
        width_inch=width_inch,
        line_width=line_width,
    )

    plotter.triangle_plot(
        contours,
        params=names,
        legend_labels=labels,
        filled=[False] * n_contours,
        contour_colors=contour_colors,
        contour_lws=[line_width] * n_contours,
        contour_ls=contour_ls,
    )

    save_getdist_plot(plotter, save_stem, close=close)

    return plotter


def plot_fisher_contours(
    gaussians,
    names,
    *,
    save_stem=None,
    labels=None,
    contour_colors=None,
    width_inch=8.0,
    line_width=1.2,
    contour_ls=None,
    close=True,
):
    """Plot one or more Fisher Gaussian approximations."""
    return plot_forecast_contours(
        gaussians,
        names,
        save_stem=save_stem,
        labels=labels,
        contour_colors=contour_colors,
        width_inch=width_inch,
        line_width=line_width,
        contour_ls=contour_ls,
        close=close,
    )


def plot_dali_contours(
    samples,
    names,
    *,
    save_stem=None,
    labels=None,
    contour_colors=None,
    width_inch=8.0,
    line_width=1.2,
    contour_ls=None,
    close=True,
):
    """Plot one or more DerivKit/GetDist DALI sample objects."""
    return plot_forecast_contours(
        samples,
        names,
        save_stem=save_stem,
        labels=labels,
        contour_colors=contour_colors,
        width_inch=width_inch,
        line_width=line_width,
        contour_ls=contour_ls,
        close=close,
    )
