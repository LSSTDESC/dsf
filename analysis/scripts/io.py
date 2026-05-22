"""Small I/O helpers for analysis run outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "run_outputs"


class RunOutput:
    """Paths and save helpers for one analysis run."""

    def __init__(
        self,
        run_name: str,
        base_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.run_dir = Path(base_dir) / run_name
        self.plot_dir = self.run_dir / "plots"
        self.array_dir = self.run_dir / "arrays"

        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.array_dir.mkdir(parents=True, exist_ok=True)

    def array_path(self, name: str) -> Path:
        """Return an array path inside the run output directory."""
        if not name.endswith(".npy"):
            name = f"{name}.npy"
        return self.array_dir / name

    def plot_path(self, name: str) -> Path:
        """Return a plot path inside the run output directory."""
        return self.plot_dir / name

    def sample_path(self, name: str) -> Path:
        """Return a GetDist sample root path inside the array directory."""
        return self.array_dir / name

    def save_array(self, name: str, array: Any) -> Path:
        """Save one array to the run output directory."""
        path = self.array_path(name)
        np.save(path, array)
        return path

    def save_arrays(self, **arrays: Any) -> dict[str, Path]:
        """Save multiple arrays to the run output directory."""
        return {name: self.save_array(name, array) for name, array in arrays.items()}

    def save_figure(self, name: str, fig: Any, *, dpi: int = 300) -> Path:
        """Save one figure to the run output directory."""
        path = self.plot_path(name)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return path

    def save_samples(self, name: str, samples: Any) -> Path:
        """Save GetDist samples to the run output directory."""
        path = self.sample_path(name)
        samples.saveAsText(str(path))
        return path


def make_run_output(
    run_name: str,
    base_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> RunOutput:
    """Create the standard output object for one analysis run."""
    return RunOutput(run_name, base_dir=base_dir)
