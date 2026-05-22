import cmasher as cmr
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

__all__ = [
    "apply_dsf_plot_style",
    "binny_colors",
    "transparent_facecolor",
    "ALPHA",
]


ALPHA = 0.65


def apply_dsf_plot_style() -> None:
    """Set a compact Binny-like plotting style for DSF sanity plots."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.0, 5.0),
            "axes.linewidth": 2,
            "axes.grid": False,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 14,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
        }
    )


def binny_colors(n_colors, cmap="viridis", cmap_range=(0.10, 0.90)):
    """Return Binny-style colors sampled from a colormap."""
    return cmr.take_cmap_colors(
        cmap,
        n_colors,
        cmap_range=cmap_range,
        return_fmt="hex",
    )


def transparent_facecolor(color, alpha=0.55):
    """Return a color with transparency applied only to the face."""
    return mcolors.to_rgba(color, alpha=alpha)
