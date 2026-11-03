"""Preview the DSF color palette and colormaps."""

import matplotlib.pyplot as plt
import numpy as np

from dsf.utils.colors import DSF_CMAP, DSF_CMAP_DISCRETE, DSF_COLORS


def main():
    """Preview the DSF palette and colormaps."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 5))

    gradient = np.linspace(0, 1, 512)[None, :]

    for i, color in enumerate(DSF_COLORS):
        axes[0].barh(0, 1, left=i, color=color, height=1)

    axes[0].set_xlim(0, len(DSF_COLORS))
    axes[0].set_ylim(-0.5, 0.5)
    axes[0].set_title("DSF — Discrete Palette")
    axes[0].axis("off")

    axes[1].imshow(
        gradient,
        aspect="auto",
        cmap=DSF_CMAP,
        extent=[0, 1, 0, 1],
    )
    axes[1].set_title("DSF — Continuous Colormap")
    axes[1].axis("off")

    discrete = np.arange(len(DSF_COLORS))[None, :]

    axes[2].imshow(
        discrete,
        aspect="auto",
        cmap=DSF_CMAP_DISCRETE,
        interpolation="nearest",
    )
    axes[2].set_title("DSF — Discrete Colormap")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig("dsf_colors.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
