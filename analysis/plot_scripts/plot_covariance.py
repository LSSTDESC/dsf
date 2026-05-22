"""Diagnostic plot for covariance correlation matrices."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_scripts.plot_style import apply_dsf_plot_style


def plot_covariance_correlation(
    cov,
    *,
    save_path=None,
    figsize=(5, 5),
    dpi=300,
):
    """Plot the correlation matrix implied by a covariance matrix.

    Args:
        cov: Covariance matrix.
        save_path: Optional output path for saving the figure.
        figsize: Optional Matplotlib figure size.
        dpi: Resolution used when saving the figure.

    Returns:
        Matplotlib figure, axis, and image object.
    """
    apply_dsf_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    cov = np.asarray(cov, dtype=float)
    diag = np.sqrt(np.diag(cov))
    corr = cov / np.outer(diag, diag)

    image = ax.imshow(
        corr,
        origin="lower",
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
        interpolation="none",
        alpha=0.85,
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Correlation coefficient")

    ax.set_title("Covariance correlation matrix")
    ax.set_xlabel("Data-vector index")
    ax.set_ylabel("Data-vector index")

    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(2.0)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")

    return fig, ax, image


def plot_covariance_diagonal(
    cov,
    *,
    save_path=None,
    figsize=None,
    dpi=300,
):
    """Plot the diagonal entries of a covariance matrix."""
    apply_dsf_plot_style()

    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    cov = np.asarray(cov, dtype=float)
    diagonal = np.diag(cov)
    index = np.arange(1, len(diagonal) + 1)

    ax.plot(
        index,
        diagonal,
        marker="o",
        markersize=4.5,
        linewidth=2.0,
        markeredgecolor="k",
        markeredgewidth=0.7,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Data-vector index")
    ax.set_ylabel("Covariance diagonal")
    ax.set_title("Covariance diagonal")

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")

    return fig, ax
