"""Define the DSF color palette and continuous and discrete Matplotlib colormaps."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (
    BoundaryNorm,
    LinearSegmentedColormap,
    ListedColormap,
    to_hex,
    to_rgb,
)

__all__ = [
    "DSF_COLORS",
    "DSF_CMAP",
    "DSF_CMAP_DISCRETE",
    "DSF_NORM",
]

# Pastel strength: 0.0 = original colors, 1.0 = white.
PASTEL_AMOUNT = 0.1

# Original DSF colors, ordered from cool to warm.
_BASE_COLORS = [
    "#294653",  # Deep slate blue
    "#168C83",  # Teal
    "#68C8B8",  # Mint
    "#F2C75C",  # Golden yellow
    "#F6A35C",  # Apricot
    "#ED765E",  # Coral
    "#D95C48",  # Terracotta
]


def pastelize(color, amount):
    """Blend a color with white to produce a pastel shade.

    Args:
        color: Matplotlib-compatible color.
        amount: Fraction of white to mix in, between 0 and 1.

    Returns:
        Hexadecimal color string.
    """
    rgb = np.asarray(to_rgb(color))
    pastel = (1.0 - amount) * rgb + amount * np.ones(3)
    return to_hex(pastel)


# DSF pastel colors.
DSF_COLORS = [
    pastelize(color, PASTEL_AMOUNT)
    for color in _BASE_COLORS
]

# Continuous colormap.
DSF_CMAP = LinearSegmentedColormap.from_list(
    "dsf_colors",
    DSF_COLORS,
    N=256,
)

# Discrete colormap.
DSF_CMAP_DISCRETE = ListedColormap(
    DSF_COLORS,
    name="dsf_colors_discrete",
)

# Discrete normalization.
DSF_NORM = BoundaryNorm(
    np.arange(len(DSF_COLORS) + 1),
    DSF_CMAP_DISCRETE.N,
)


def main():
    """Preview the DSF palette and colormaps."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 5))

    # Individual discrete colors.
    for i, color in enumerate(DSF_COLORS):
        axes[0].barh(0, 1, left=i, color=color, height=1)

    axes[0].set_xlim(0, len(DSF_COLORS))
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].set_title("DSF Colors — Discrete Palette")
    axes[0].axis("off")

    # Continuous colormap.
    gradient = np.linspace(0, 1, 512)[None, :]

    axes[1].imshow(
        gradient,
        aspect="auto",
        cmap=DSF_CMAP,
        extent=[0, 1, 0, 1],
    )
    axes[1].set_title("DSF Colors — Continuous Colormap")
    axes[1].axis("off")

    # Discrete colormap.
    discrete = np.arange(len(DSF_COLORS))[None, :]

    axes[2].imshow(
        discrete,
        aspect="auto",
        cmap=DSF_CMAP_DISCRETE,
        interpolation="nearest",
    )
    axes[2].set_title("DSF Colors — Discrete Colormap")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig("dsf_colors.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()