"""Summarize SNR and Fisher outputs from saved tomography scenario arrays.

This script scans the tomography scenario run directories, reads the saved
SNR arrays and optional Fisher arrays, and writes a compact TSV summary table.

It does not rebuild forecasts or covariances.

Run from the project root with:

    python analysis/summarize_tomography_snr.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "runs_output" / "tomography_scenarios"
SUMMARY_PATH = OUTPUT_DIR / "arrays" / "tomography_snr_only_summary.tsv"

OMEGA_M_INDEX = 0
SIGMA8_INDEX = 1


def load_array(path):
    """Load one numpy array."""
    return np.load(path, allow_pickle=True)


def load_array_or_none(path):
    """Load one numpy array if it exists."""
    if not path.exists():
        return None

    return np.load(path, allow_pickle=True)


def scalar_from_array(path):
    """Load a scalar value from a saved numpy array."""
    value = load_array(path)
    return float(np.asarray(value).squeeze())


def scenario_directories():
    """Return tomography scenario directories that contain saved arrays."""
    run_dirs = []

    for path in sorted(OUTPUT_DIR.iterdir()):
        if not path.is_dir():
            continue

        if path.name in {"arrays", "plots"}:
            continue

        array_dir = path / "arrays"

        if not array_dir.exists():
            continue

        if (array_dir / "snr.npy").exists():
            run_dirs.append(path)

    return run_dirs


def parameter_covariance_from_fisher(fisher):
    """Return marginalized parameter covariance from a Fisher matrix."""
    fisher = np.asarray(fisher, dtype=float)

    return np.linalg.inv(fisher)


def omega_m_sigma8_fom_from_fisher(fisher):
    """Return marginalized Omega_m-sigma8 Fisher figure of merit.

    The full Fisher matrix is inverted first, so the 2D covariance includes
    marginalization over all other varied parameters. The FoM is defined as
    1 / sqrt(det(C_2d)), so larger values mean tighter joint constraints.
    """
    covariance = parameter_covariance_from_fisher(fisher)

    indices = np.asarray([OMEGA_M_INDEX, SIGMA8_INDEX], dtype=int)
    covariance_2d = covariance[np.ix_(indices, indices)]

    return float(1.0 / np.sqrt(np.linalg.det(covariance_2d)))


def omega_m_sigma8_metrics_from_fisher(fisher):
    """Return marginalized Omega_m-sigma8 diagnostics from a Fisher matrix."""
    if fisher is None:
        return {
            "valid": False,
            "sigma_omega_m": np.nan,
            "sigma8_error": np.nan,
            "corr_omega_m_sigma8": np.nan,
            "omega_m_sigma8_fom": np.nan,
            "fisher_condition_number": np.nan,
        }

    fisher = np.asarray(fisher, dtype=float)

    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        return {
            "valid": False,
            "sigma_omega_m": np.nan,
            "sigma8_error": np.nan,
            "corr_omega_m_sigma8": np.nan,
            "omega_m_sigma8_fom": np.nan,
            "fisher_condition_number": np.nan,
        }

    if fisher.shape[0] <= max(OMEGA_M_INDEX, SIGMA8_INDEX):
        return {
            "valid": False,
            "sigma_omega_m": np.nan,
            "sigma8_error": np.nan,
            "corr_omega_m_sigma8": np.nan,
            "omega_m_sigma8_fom": np.nan,
            "fisher_condition_number": np.nan,
        }

    if not np.all(np.isfinite(fisher)):
        return {
            "valid": False,
            "sigma_omega_m": np.nan,
            "sigma8_error": np.nan,
            "corr_omega_m_sigma8": np.nan,
            "omega_m_sigma8_fom": np.nan,
            "fisher_condition_number": np.nan,
        }

    try:
        covariance = parameter_covariance_from_fisher(fisher)

        sigma_omega_m = float(np.sqrt(covariance[OMEGA_M_INDEX, OMEGA_M_INDEX]))
        sigma8_error = float(np.sqrt(covariance[SIGMA8_INDEX, SIGMA8_INDEX]))

        corr_omega_m_sigma8 = float(
            covariance[OMEGA_M_INDEX, SIGMA8_INDEX] / (sigma_omega_m * sigma8_error)
        )

        omega_m_sigma8_fom = omega_m_sigma8_fom_from_fisher(fisher)
        fisher_condition_number = float(np.linalg.cond(fisher))

    except np.linalg.LinAlgError:
        return {
            "valid": False,
            "sigma_omega_m": np.nan,
            "sigma8_error": np.nan,
            "corr_omega_m_sigma8": np.nan,
            "omega_m_sigma8_fom": np.nan,
            "fisher_condition_number": np.nan,
        }

    return {
        "valid": True,
        "sigma_omega_m": sigma_omega_m,
        "sigma8_error": sigma8_error,
        "corr_omega_m_sigma8": corr_omega_m_sigma8,
        "omega_m_sigma8_fom": omega_m_sigma8_fom,
        "fisher_condition_number": fisher_condition_number,
    }


def add_prefixed_metrics(row, prefix, metrics):
    """Add metric values to a row using a prefix."""
    for key, value in metrics.items():
        row[f"{prefix}_{key}"] = value


def summarize_scenario(run_dir):
    """Return one SNR and Fisher summary row for a tomography scenario."""
    array_dir = run_dir / "arrays"

    snr = scalar_from_array(array_dir / "snr.npy")
    snr2 = scalar_from_array(array_dir / "snr2.npy")

    bin_pairs = load_array(array_dir / "bin_pairs.npy")
    snr_by_pair = np.asarray(load_array(array_dir / "snr_by_pair.npy"), dtype=float)
    snr_by_radius = np.asarray(load_array(array_dir / "snr_by_radius.npy"), dtype=float)
    cumulative_snr = np.asarray(load_array(array_dir / "cumulative_snr.npy"), dtype=float)
    r = np.asarray(load_array(array_dir / "r.npy"), dtype=float)

    fisher = load_array_or_none(array_dir / "fisher.npy")
    fisher_with_priors = load_array_or_none(array_dir / "fisher_with_priors.npy")

    n_pairs = len(bin_pairs)

    if n_pairs > 0:
        snr_per_pair = snr / n_pairs
    else:
        snr_per_pair = np.nan

    if len(snr_by_pair) > 0:
        max_pair_snr = float(np.nanmax(snr_by_pair))
        mean_pair_snr = float(np.nanmean(snr_by_pair))
        median_pair_snr = float(np.nanmedian(snr_by_pair))
    else:
        max_pair_snr = np.nan
        mean_pair_snr = np.nan
        median_pair_snr = np.nan

    if len(snr_by_radius) > 0:
        max_radius_snr = float(np.nanmax(snr_by_radius))
        max_radius_index = int(np.nanargmax(snr_by_radius))
        radius_at_max_snr = float(r[max_radius_index])
    else:
        max_radius_snr = np.nan
        radius_at_max_snr = np.nan

    if len(cumulative_snr) > 0:
        final_cumulative_snr = float(cumulative_snr[-1])
    else:
        final_cumulative_snr = np.nan

    row = {
        "scenario": run_dir.name,
        "total_snr": snr,
        "total_snr2": snr2,
        "n_pairs": n_pairs,
        "snr_per_pair": snr_per_pair,
        "mean_pair_snr": mean_pair_snr,
        "median_pair_snr": median_pair_snr,
        "max_pair_snr": max_pair_snr,
        "max_radius_snr": max_radius_snr,
        "radius_at_max_snr": radius_at_max_snr,
        "final_cumulative_snr": final_cumulative_snr,
    }

    fisher_metrics = omega_m_sigma8_metrics_from_fisher(fisher)
    fisher_with_priors_metrics = omega_m_sigma8_metrics_from_fisher(fisher_with_priors)

    add_prefixed_metrics(row, "fisher", fisher_metrics)
    add_prefixed_metrics(row, "fisher_with_priors", fisher_with_priors_metrics)

    return row


def write_summary(rows):
    """Write the SNR and Fisher summary table."""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "scenario",
        "total_snr",
        "total_snr2",
        "n_pairs",
        "snr_per_pair",
        "mean_pair_snr",
        "median_pair_snr",
        "max_pair_snr",
        "max_radius_snr",
        "radius_at_max_snr",
        "final_cumulative_snr",
        "fisher_valid",
        "fisher_sigma_omega_m",
        "fisher_sigma8_error",
        "fisher_corr_omega_m_sigma8",
        "fisher_omega_m_sigma8_fom",
        "fisher_fisher_condition_number",
        "fisher_with_priors_valid",
        "fisher_with_priors_sigma_omega_m",
        "fisher_with_priors_sigma8_error",
        "fisher_with_priors_corr_omega_m_sigma8",
        "fisher_with_priors_omega_m_sigma8_fom",
        "fisher_with_priors_fisher_condition_number",
    ]

    with SUMMARY_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def print_summary(rows):
    """Print a compact SNR and Fisher ranking."""
    rows = sorted(rows, key=lambda row: row["total_snr"], reverse=True)

    print()
    print("SNR and Fisher summary:")
    print(
        f"{'scenario':45s} "
        f"{'SNR':>10s} "
        f"{'pairs':>6s} "
        f"{'SNR/pair':>10s} "
        f"{'max_pair':>10s} "
        f"{'max_R':>10s} "
        f"{'FoM':>12s} "
        f"{'FoM+prior':>12s}"
    )

    for row in rows:
        print(
            f"{row['scenario']:45s} "
            f"{row['total_snr']:10.3f} "
            f"{row['n_pairs']:6d} "
            f"{row['snr_per_pair']:10.3f} "
            f"{row['max_pair_snr']:10.3f} "
            f"{row['max_radius_snr']:10.3f} "
            f"{row['fisher_omega_m_sigma8_fom']:12.3e} "
            f"{row['fisher_with_priors_omega_m_sigma8_fom']:12.3e}"
        )

    print()
    print(f"Wrote summary to: {SUMMARY_PATH}")


def main():
    """Build the SNR and Fisher summary table."""
    run_dirs = scenario_directories()

    if not run_dirs:
        raise RuntimeError(f"No scenario array directories found under {OUTPUT_DIR}")

    rows = [summarize_scenario(run_dir) for run_dir in run_dirs]

    rows = sorted(rows, key=lambda row: row["total_snr"], reverse=True)

    write_summary(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
