"""Tomography plotting scripts for Delta Sigma forecast runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.plot_scripts.plot_style import apply_dsf_plot_style, binny_colors


def plot_lens_and_source_bins(
    lens_result,
    source_result,
    *,
    save_path=None,
    figsize=None,
    dpi=300,
):
    """Plot lens and source tomographic redshift distributions."""
    apply_dsf_plot_style()

    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    z_lens = np.asarray(lens_result.z).reshape(-1)
    z_source = np.asarray(source_result.z).reshape(-1)

    lens_keys = sorted(lens_result.bins)
    source_keys = sorted(source_result.bins)

    lens_colors = binny_colors(
        len(lens_keys),
        cmap="viridis",
        cmap_range=(0.10, 0.75),
    )
    source_colors = binny_colors(
        len(source_keys),
        cmap="viridis",
        cmap_range=(0.25, 1.00),
    )

    for i, (key, color) in enumerate(zip(lens_keys, lens_colors, strict=True)):
        nz = np.asarray(lens_result.bins[key]).reshape(-1)

        ax.fill_between(
            z_lens,
            0.0,
            nz,
            facecolor=color,
            edgecolor=color,
            hatch="///",
            alpha=0.45,
            linewidth=0.0,
            zorder=10 + i,
            label=f"lens bin {key + 1}",
        )

        ax.plot(
            z_lens,
            nz,
            color="k",
            linewidth=1.8,
            linestyle="--",
            zorder=20 + i,
        )

    for i, (key, color) in enumerate(zip(source_keys, source_colors, strict=True)):
        nz = np.asarray(source_result.bins[key]).reshape(-1)

        ax.plot(
            z_source,
            nz,
            color=color,
            linewidth=2.4,
            zorder=40 + i,
            label=f"source bin {key + 1}",
        )

        ax.plot(
            z_source,
            nz,
            color="k",
            linewidth=0.8,
            alpha=0.55,
            zorder=39 + i,
        )

    z_min = min(np.min(z_lens), np.min(z_source))
    z_max = max(np.max(z_lens), np.max(z_source))

    ax.plot(
        [z_min, z_max],
        [0.0, 0.0],
        color="k",
        linewidth=2.0,
        zorder=1000,
    )

    ax.set_title("Lens and source tomography")
    ax.set_xlabel(r"Redshift $z$")
    ax.set_ylabel(r"Normalized $n_i(z)$")
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")

    return fig, ax
