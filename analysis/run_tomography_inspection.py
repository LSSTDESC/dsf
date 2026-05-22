"""Small tomography scenario sweep for DSF."""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from analysis.configs.tomography_sweep_config import (
    ARRAY_DIR,
    CENTER_METHOD,
    LENS_SAMPLE,
    LENS_SURVEY,
    OUTPUT_DIR,
    OVERLAP_THRESHOLD,
    PLOT_DIR,
    SCENARIOS,
    SOURCE_BEHIND_LENS,
    SOURCE_SURVEY,
    SOURCE_YEAR,
    case_label,
    lens_overrides,
    scenario_values,
    source_overrides,
)
from analysis.plot_scripts.plot_tomobins import plot_lens_and_source_bins
from analysis.plot_scripts.plot_tomography_summaries import (
    plot_tomography_bin_widths,
    plot_tomography_overlap_vs_separation,
    plot_tomography_pair_yield,
    plot_tomography_population_balance,
    plot_tomography_summary_dashboard,
    sort_rows_for_tomography,
)
from analysis.scripts.tomography_metrics import (
    failed_tomography_row,
    summarize_tomography_result,
)
from src.dsf.tomography.tomo_builder import TomographyBuilder

PLOT_DIR.mkdir(parents=True, exist_ok=True)
ARRAY_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(fig, path):
    """Save one figure and close it."""
    if fig is None:
        return

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_scenario(scenario_name):
    """Build, plot, and summarize one tomography scenario."""
    tomo_level, range_label, lens_n_bins, source_n_bins, z_range = scenario_values(
        scenario_name,
    )

    label = case_label(scenario_name, lens_n_bins, source_n_bins, z_range)
    plot_path = PLOT_DIR / f"tomography_{label}.png"

    builder = TomographyBuilder(
        lens_survey=LENS_SURVEY,
        source_survey=SOURCE_SURVEY,
        lens_sample=LENS_SAMPLE,
        source_year=SOURCE_YEAR,
        overlap_threshold=OVERLAP_THRESHOLD,
        source_behind_lens=SOURCE_BEHIND_LENS,
        center_method=CENTER_METHOD,
        lens_overrides=lens_overrides(lens_n_bins, z_range),
        source_overrides=source_overrides(source_n_bins),
    )

    result = builder.prepare_bins()

    fig, _ = plot_lens_and_source_bins(
        result["lens_result"],
        result["source_result"],
        save_path=plot_path,
    )
    plt.close(fig)

    row = summarize_tomography_result(
        label=label,
        scenario_name=scenario_name,
        tomo_level=tomo_level,
        range_label=range_label,
        lens_n_bins=lens_n_bins,
        source_n_bins=source_n_bins,
        z_range=z_range,
        builder=builder,
        result=result,
        plot_path=plot_path,
    )

    print_row(row)

    return row


def print_row(row):
    """Print one compact scenario summary."""
    if row["failed"]:
        print(f"{row['label']} FAILED: {row['error']}")
        return

    status = "ok" if row["valid_tomography"] else "check"

    print(
        f"{row['scenario']}: "
        f"{status}, "
        f"level={row['tomo_level']}, "
        f"range={row['lens_range_label']}, "
        f"lens={row['lens_n_bins_built']}, "
        f"source={row['source_n_bins_built']}, "
        f"pairs={row['n_pairs']}, "
        f"pair_frac={row['pair_fraction']:.3f}, "
        f"max_overlap={row['max_selected_overlap']:.3f}, "
        f"min_sep={row['min_pair_separation']:.3f}"
    )


def save_csv(rows, path):
    """Save scenario rows to CSV."""
    columns = [
        "scenario",
        "tomo_level",
        "lens_range_label",
        "lens_z_min",
        "lens_z_max",
        "lens_n_bins_requested",
        "source_n_bins_requested",
        "lens_n_bins_built",
        "source_n_bins_built",
        "n_pairs",
        "pair_fraction",
        "behind_fraction",
        "mean_pair_separation",
        "min_pair_separation",
        "lens_fraction_balance",
        "source_fraction_balance",
        "lens_mean_width_68",
        "source_mean_width_68",
        "lens_max_second_peak_ratio",
        "source_max_second_peak_ratio",
        "mean_selected_overlap",
        "max_selected_overlap",
        "valid_tomography",
        "bin_pairs",
        "lens_centers",
        "source_centers",
        "lens_fractions",
        "source_fractions",
        "failed",
        "error",
        "plot_path",
    ]

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()

        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def print_best(rows):
    """Print ranked scenario table using direct tomography diagnostics."""
    rows = [row for row in rows if not row["failed"]]
    rows = sort_rows_for_tomography(rows)

    print()
    print("tomography scenarios:")
    print(
        "scenario".ljust(22),
        "valid".rjust(7),
        "level".rjust(10),
        "range".rjust(8),
        "pairs".rjust(7),
        "pair_fr".rjust(9),
        "max_ov".rjust(8),
        "min_sep".rjust(9),
        "lens_bal".rjust(9),
        "src_bal".rjust(9),
    )

    for row in rows:
        print(
            row["scenario"].ljust(22),
            str(row["valid_tomography"]).rjust(7),
            row["tomo_level"].rjust(10),
            row["lens_range_label"].rjust(8),
            f"{row['n_pairs']}".rjust(7),
            f"{row['pair_fraction']:.3f}".rjust(9),
            f"{row['max_selected_overlap']:.3f}".rjust(8),
            f"{row['min_pair_separation']:.3f}".rjust(9),
            f"{row['lens_fraction_balance']:.3f}".rjust(9),
            f"{row['source_fraction_balance']:.3f}".rjust(9),
        )


def main():
    rows = []

    for scenario_name in SCENARIOS:
        tomo_level, range_label, lens_n_bins, source_n_bins, z_range = scenario_values(
            scenario_name,
        )
        label = case_label(scenario_name, lens_n_bins, source_n_bins, z_range)

        try:
            row = run_scenario(scenario_name)
        except Exception as error:
            row = failed_tomography_row(
                label=label,
                scenario_name=scenario_name,
                tomo_level=tomo_level,
                range_label=range_label,
                lens_n_bins=lens_n_bins,
                source_n_bins=source_n_bins,
                z_range=z_range,
                error=error,
            )
            print_row(row)

        rows.append(row)

    rows_ranked = sort_rows_for_tomography(rows)

    np.save(
        ARRAY_DIR / "tomography_scenario_rows.npy",
        np.asarray(rows, dtype=object),
        allow_pickle=True,
    )
    np.save(
        ARRAY_DIR / "tomography_scenario_ranked_rows.npy",
        np.asarray(rows_ranked, dtype=object),
        allow_pickle=True,
    )

    save_csv(rows_ranked, ARRAY_DIR / "tomography_scenario_summary.csv")

    fig, _ = plot_tomography_summary_dashboard(
        rows_ranked,
        overlap_threshold=OVERLAP_THRESHOLD,
    )
    save_plot(fig, PLOT_DIR / "summary_tomography_dashboard.png")

    fig, _ = plot_tomography_pair_yield(rows_ranked)
    save_plot(fig, PLOT_DIR / "summary_pair_yield.png")

    fig, _ = plot_tomography_overlap_vs_separation(
        rows_ranked,
        overlap_threshold=OVERLAP_THRESHOLD,
    )
    save_plot(fig, PLOT_DIR / "summary_overlap_vs_separation.png")

    fig, _ = plot_tomography_population_balance(rows_ranked)
    save_plot(fig, PLOT_DIR / "summary_population_balance.png")

    fig, _ = plot_tomography_bin_widths(rows_ranked)
    save_plot(fig, PLOT_DIR / "summary_bin_widths.png")

    print()
    print(f"saved rows to {ARRAY_DIR / 'tomography_scenario_rows.npy'}")
    print(f"saved ranked rows to {ARRAY_DIR / 'tomography_scenario_ranked_rows.npy'}")
    print(f"saved CSV summary to {ARRAY_DIR / 'tomography_scenario_summary.csv'}")
    print(f"saved plots to {PLOT_DIR}")
    print(f"output directory: {OUTPUT_DIR}")

    print_best(rows_ranked)


if __name__ == "__main__":
    main()
