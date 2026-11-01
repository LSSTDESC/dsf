"""Define the DSF color palettes and continuous and discrete Matplotlib colormaps."""

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
    "DSF_DIVERGING_COLORS",
    "DSF_DIVERGING_CMAP",
    "DSF_DIVERGING_CMAP_DISCRETE",
    "DSF_DIVERGING_NORM",
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

# Diverging DSF colors, ordered from dark blue to dark orange.
_DIVERGING_BASE_COLORS = [
    "#123247",  # Dark navy
    "#316A83",  # Deep blue
    "#95C5DD",  # Light blue
    "#F7F4E8",  # Pale cream
    "#F7B83E",  # Golden yellow
    "#F3A039",  # Orange
    "#F28A32",  # Dark orange
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

# DSF diverging colors.
DSF_DIVERGING_COLORS = _DIVERGING_BASE_COLORS.copy()

# Continuous diverging colormap.
DSF_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "dsf_diverging",
    DSF_DIVERGING_COLORS,
    N=256,
)

# Discrete diverging colormap.
DSF_DIVERGING_CMAP_DISCRETE = ListedColormap(
    DSF_DIVERGING_COLORS,
    name="dsf_diverging_discrete",
)

# Discrete diverging normalization.
DSF_DIVERGING_NORM = BoundaryNorm(
    np.arange(len(DSF_DIVERGING_COLORS) + 1),
    DSF_DIVERGING_CMAP_DISCRETE.N,
)


def main():
    """Preview the DSF palettes and colormaps."""
    fig, axes = plt.subplots(6, 1, figsize=(12, 9))

    palettes = [
        ("DSF", DSF_COLORS, DSF_CMAP, DSF_CMAP_DISCRETE),
        (
            "DSF Diverging",
            DSF_DIVERGING_COLORS,
            DSF_DIVERGING_CMAP,
            DSF_DIVERGING_CMAP_DISCRETE,
        ),
    ]

    gradient = np.linspace(0, 1, 512)[None, :]

    for row, (name, colors, cmap, cmap_discrete) in enumerate(palettes):
        offset = 3 * row

        # Individual discrete colors.
        for i, color in enumerate(colors):
            axes[offset].barh(0, 1, left=i, color=color, height=1)

        axes[offset].set_xlim(0, len(colors))
        axes[offset].set_ylim(-0.5, 0.5)
        axes[offset].set_title(f"{name} — Discrete Palette")
        axes[offset].axis("off")

        # Continuous colormap.
        axes[offset + 1].imshow(
            gradient,
            aspect="auto",
            cmap=cmap,
            extent=[0, 1, 0, 1],
        )
        axes[offset + 1].set_title(f"{name} — Continuous Colormap")
        axes[offset + 1].axis("off")

        # Discrete colormap.
        discrete = np.arange(len(colors))[None, :]

        axes[offset + 2].imshow(
            discrete,
            aspect="auto",
            cmap=cmap_discrete,
            interpolation="nearest",
        )
        axes[offset + 2].set_title(f"{name} — Discrete Colormap")
        axes[offset + 2].axis("off")

    fig.tight_layout()
    fig.savefig("dsf_colors.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
