"""Define the DSF color palette and continuous and discrete Matplotlib colormaps."""

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
    "pastelize",
]

PASTEL_AMOUNT = 0.25

_BASE_COLORS = [
    "#294653",  # Deep slate blue
    "#168C83",  # Teal
    "#68C8B8",  # Mint
    "#F2C75C",  # Golden yellow
    "#F6A35C",  # Apricot
    "#ED765E",  # Coral
    "#D95C48",  # Terracotta
]


def pastelize(color: str, amount: float) -> str:
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


DSF_COLORS = [
    pastelize(color, PASTEL_AMOUNT)
    for color in _BASE_COLORS
]

DSF_CMAP = LinearSegmentedColormap.from_list(
    "dsf_colors",
    DSF_COLORS,
    N=256,
)

DSF_CMAP_DISCRETE = ListedColormap(
    DSF_COLORS,
    name="dsf_colors_discrete",
)

DSF_NORM = BoundaryNorm(
    np.arange(len(DSF_COLORS) + 1),
    DSF_CMAP_DISCRETE.N,
)
