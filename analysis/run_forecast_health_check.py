"""Run numerical health checks on saved DSF forecast outputs.

This script scans an analysis output directory, looks for saved data vectors,
covariances, Fisher matrices, and parameter-error arrays, and reports basic
numerical health diagnostics.

Run from the project root with:

    python analysis/run_forecast_health_check.py

or with a custom output directory:

    python analysis/run_forecast_health_check.py analysis/runs_output/tomography_scenarios
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from analysis.scripts.forecast_health import (
    covariance_health,
    data_vector_health,
    fisher_health,
    health_flag_summary,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "runs_output"


DATA_VECTOR_NAMES = [
    "data_vector.npy",
    "fiducial_data_vector.npy",
    "delta_sigma.npy",
    "model_vector.npy",
]

COVARIANCE_NAMES = [
    "covariance.npy",
    "cov.npy",
    "total_covariance.npy",
]

FISHER_NAMES = [
    "fisher.npy",
    "fisher_matrix.npy",
    "fisher_with_priors.npy",
]


def load_array(path):
    """Load one numpy array."""
    return np.load(path, allow_pickle=True)


def find_first_existing(directory, names):
    """Return the first matching file in a directory."""
    for name in names:
        path = directory / name

        if path.exists():
            return path

    return None


def find_array_dir(run_dir):
    """Return the array directory for one run."""
    array_dir = run_dir / "arrays"

    if array_dir.exists():
        return array_dir

    return run_dir


def summarize_health_flags(health):
    """Return health flags as a semicolon-separated string."""
    flags = health_flag_summary(health)

    if len(flags) == 0:
        return ""

    return ";".join(flags)


def health_row_for_run(run_dir):
    """Return one health-check row for a saved run directory."""
    array_dir = find_array_dir(run_dir)

    data_vector_path = find_first_existing(array_dir, DATA_VECTOR_NAMES)
    covariance_path = find_first_existing(array_dir, COVARIANCE_NAMES)
    fisher_path = find_first_existing(array_dir, FISHER_NAMES)

    row = {
        "run": run_dir.name,
        "array_dir": str(array_dir),
        "data_vector_path": "",
        "covariance_path": "",
        "fisher_path": "",
        "data_vector_finite": "",
        "data_vector_nonfinite": "",
        "data_vector_zeros": "",
        "covariance_finite": "",
        "covariance_symmetric": "",
        "covariance_positive_definite": "",
        "covariance_min_eigenvalue": "",
        "covariance_condition_number": "",
        "covariance_flags": "",
        "fisher_finite": "",
        "fisher_symmetric": "",
        "fisher_positive_definite": "",
        "fisher_min_eigenvalue": "",
        "fisher_condition_number": "",
        "fisher_flags": "",
        "status": "ok",
        "error": "",
    }

    try:
        if data_vector_path is not None:
            data_vector = load_array(data_vector_path)
            health = data_vector_health(data_vector)

            row["data_vector_path"] = str(data_vector_path)
            row["data_vector_finite"] = health["finite"]
            row["data_vector_nonfinite"] = health["n_nonfinite"]
            row["data_vector_zeros"] = health["n_zero"]

        if covariance_path is not None:
            covariance = load_array(covariance_path)
            health = covariance_health(covariance)

            row["covariance_path"] = str(covariance_path)
            row["covariance_finite"] = health["finite"]
            row["covariance_symmetric"] = health["symmetric"]
            row["covariance_positive_definite"] = health["positive_definite"]
            row["covariance_min_eigenvalue"] = health["min_eigenvalue"]
            row["covariance_condition_number"] = health["condition_number"]
            row["covariance_flags"] = summarize_health_flags(health)

        if fisher_path is not None:
            fisher_matrix = load_array(fisher_path)
            health = fisher_health(fisher_matrix)

            row["fisher_path"] = str(fisher_path)
            row["fisher_finite"] = health["finite"]
            row["fisher_symmetric"] = health["symmetric"]
            row["fisher_positive_definite"] = health["positive_definite"]
            row["fisher_min_eigenvalue"] = health["min_eigenvalue"]
            row["fisher_condition_number"] = health["condition_number"]
            row["fisher_flags"] = summarize_health_flags(health)

    except Exception as error:
        row["status"] = "failed"
        row["error"] = str(error)

    return row


def discover_run_dirs(output_dir):
    """Return run directories under an output directory."""
    output_dir = Path(output_dir)

    if (output_dir / "arrays").exists():
        return [output_dir]

    return sorted(path for path in output_dir.iterdir() if path.is_dir())


def save_csv(rows, path):
    """Save health-check rows to CSV."""
    if len(rows) == 0:
        return

    columns = list(rows[0].keys())

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def print_row(row):
    """Print one compact health-check summary."""
    pieces = [row["run"]]

    if row["status"] != "ok":
        print(f"{row['run']}: FAILED: {row['error']}")
        return

    if row["data_vector_path"]:
        pieces.append(f"data_finite={row['data_vector_finite']}")

    if row["covariance_path"]:
        pieces.append(f"cov_pd={row['covariance_positive_definite']}")
        pieces.append(f"cov_cond={float(row['covariance_condition_number']):.3e}")

        if row["covariance_flags"]:
            pieces.append(f"cov_flags={row['covariance_flags']}")

    if row["fisher_path"]:
        pieces.append(f"fisher_pd={row['fisher_positive_definite']}")
        pieces.append(f"fisher_cond={float(row['fisher_condition_number']):.3e}")

        if row["fisher_flags"]:
            pieces.append(f"fisher_flags={row['fisher_flags']}")

    print(": ".join([pieces[0], ", ".join(pieces[1:])]))


def main():
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    run_dirs = discover_run_dirs(output_dir)

    rows = []

    for run_dir in run_dirs:
        row = health_row_for_run(run_dir)
        rows.append(row)
        print_row(row)

    summary_path = output_dir / "forecast_health_summary.csv"
    save_csv(rows, summary_path)

    print()
    print(f"checked {len(rows)} run directories")
    print(f"saved health summary to {summary_path}")


if __name__ == "__main__":
    main()
