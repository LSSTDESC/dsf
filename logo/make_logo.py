"""Generate the DSF Delta-Sigma logo and save it for documentation."""

from pathlib import Path

import matplotlib.pyplot as plt

from dsf_colors import DSF_COLORS

fig, ax = plt.subplots(figsize=(5, 5))

# Large Delta.
ax.text(0, 0, r"$\Delta$", fontsize=250, color=DSF_COLORS[0],
        ha="center", va="center")

# Smaller Sigma inside Delta.
ax.text(0, -0.12, r"$\Sigma$", fontsize=85, color=DSF_COLORS[5],
        ha="center", va="center")

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect("equal")
ax.axis("off")

# Save in the current folder.
plt.savefig("dsf_logo.svg", transparent=True, bbox_inches="tight", pad_inches=0.05)
plt.savefig("dsf_logo.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.05)

# Save in the documentation static folder.
static_dir = Path(__file__).resolve().parents[1] / "docs" / "_static"
static_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(static_dir / "dsf_logo.svg", transparent=True, bbox_inches="tight", pad_inches=0.0)
plt.savefig(static_dir / "dsf_logo.png", dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.0)

plt.show()