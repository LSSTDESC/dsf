"""Tomography summary plotting scripts for Delta Sigma forecast runs."""

import matplotlib.pyplot as plt
import numpy as np

from analysis.configs.tomography_sweep_config import (
    SCENARIO_COLOR_ORDER,
    hatch_for_scenario,
    marker_for_scenario,
    scenario_color_map,
    scenario_name_from_row,
    scenario_plot_label,
)
from analysis.plot_scripts.plot_style import (
    ALPHA,
    apply_dsf_plot_style,
    binny_colors,
    transparent_facecolor,
)


def sort_rows_for_tomography(rows):
    """Sort rows by direct tomography quality diagnostics."""
    return sorted(
        rows,
        key=lambda row: (
            row["valid_tomography"],
            row["n_pairs"],
            row["pair_fraction"],
            row["min_pair_separation"],
            -row["max_selected_overlap"],
            row["lens_fraction_balance"],
            row["source_fraction_balance"],
        ),
        reverse=True,
    )


def valid_rows(rows):
    """Return rows that did not fail."""
    return [row for row in rows if not row["failed"]]


def row_colors(rows, cmap="viridis", cmap_range=(0.15, 0.90)):
    """Return one Binny-style color per row."""
    return binny_colors(
        len(rows),
        cmap=cmap,
        cmap_range=cmap_range,
    )


def scenario_colors():
    """Return a fixed color lookup for tomography scenarios."""
    colors = row_colors(SCENARIO_COLOR_ORDER)

    return scenario_color_map(colors)


def style_for_scenario(scenario):
    """Return the color, marker, and hatch for one tomography scenario."""
    colors_by_scenario = scenario_colors()

    return (
        colors_by_scenario[scenario],
        marker_for_scenario(scenario),
        hatch_for_scenario(scenario),
    )


def scenario_styles_for_rows(rows):
    """Return labels, facecolors, hatches, and scenarios for plotted rows."""
    colors_by_scenario = scenario_colors()

    scenarios = [scenario_name_from_row(row) for row in rows]
    labels = [scenario_plot_label(scenario) for scenario in scenarios]

    colors = [
        transparent_facecolor(colors_by_scenario[scenario], alpha=ALPHA) for scenario in scenarios
    ]
    hatches = [hatch_for_scenario(scenario) for scenario in scenarios]

    return scenarios, labels, colors, hatches


def apply_bar_hatches(bars, hatches):
    """Apply one hatch pattern to each bar."""
    for bar, hatch in zip(bars, hatches, strict=True):
        bar.set_hatch(hatch)


def label_lens_source_bars(ax, lens_bars, source_bars):
    """Label paired horizontal bars directly with L and S."""
    all_widths = [bar.get_width() for bar in list(lens_bars) + list(source_bars)]
    label_offset = 0.015 * max(all_widths)

    for bar in lens_bars:
        ax.text(
            bar.get_width() + label_offset,
            bar.get_y() + bar.get_height() / 2.0,
            "L",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
        )

    for bar in source_bars:
        ax.text(
            bar.get_width() + label_offset,
            bar.get_y() + bar.get_height() / 2.0,
            "S",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
        )


def plot_tomography_summary_dashboard(
    rows,
    *,
    overlap_threshold=0.10,
):
    """Plot direct tomography diagnostics in one dashboard."""
    apply_dsf_plot_style()

    rows = sort_rows_for_tomography(valid_rows(rows))

    if not rows:
        return None, None

    _, labels, colors, hatches = scenario_styles_for_rows(rows)

    y = np.arange(len(rows))

    columns = [
        ("n_pairs", "pairs", None, None),
        ("pair_fraction", "pair fraction", 0.0, 1.0),
        ("max_selected_overlap", "max overlap", 0.0, None),
        ("min_pair_separation", "min separation", None, None),
        ("lens_fraction_balance", "lens balance", 0.0, 1.05),
        ("source_fraction_balance", "source balance", 0.0, 1.05),
    ]

    fig, axes = plt.subplots(
        ncols=len(columns),
        figsize=(17.0, 0.45 * len(rows) + 2.2),
        sharey=True,
        gridspec_kw={"wspace": 0.10},
    )

    for ax, (column, title, xmin, xmax) in zip(axes, columns, strict=True):
        values = np.asarray([row[column] for row in rows], dtype=float)

        bars = ax.barh(
            y,
            values,
            color=colors,
            edgecolor="k",
            linewidth=0.8,
        )
        apply_bar_hatches(bars, hatches)

        ax.set_xlabel(title)
        ax.set_title(title)

        if column == "max_selected_overlap":
            ax.axvline(
                overlap_threshold,
                color="k",
                linestyle="--",
                linewidth=1.2,
            )

        if column == "min_pair_separation":
            ax.axvline(
                0.0,
                color="k",
                linestyle="--",
                linewidth=1.2,
            )

        if xmin is not None or xmax is not None:
            ax.set_xlim(xmin, xmax)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()

    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)

    fig.suptitle("Tomography-only scenario diagnostics", y=0.995)
    fig.tight_layout()

    return fig, axes


def plot_tomography_pair_yield(rows):
    """Plot selected-pair yield across scenarios."""
    apply_dsf_plot_style()

    rows = valid_rows(rows)

    if not rows:
        return None, None

    rows = sorted(rows, key=lambda row: row["n_pairs"], reverse=True)

    _, labels, colors, hatches = scenario_styles_for_rows(rows)

    y = np.arange(len(rows))
    values = np.asarray([row["n_pairs"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(rows) + 2.0))
    bars = ax.barh(
        y,
        values,
        color=colors,
        edgecolor="k",
        linewidth=0.8,
    )
    apply_bar_hatches(bars, hatches)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("selected lens-source pairs")
    ax.set_title("Tomography pair yield")

    fig.tight_layout()

    return fig, ax


def plot_tomography_overlap_vs_separation(
    rows,
    *,
    overlap_threshold=0.10,
):
    """Plot selected-pair overlap against source-lens separation."""
    apply_dsf_plot_style()

    rows = valid_rows(rows)

    if not rows:
        return None, None

    colors_by_scenario = scenario_colors()

    fig, ax = plt.subplots(figsize=(7, 5))

    for row in rows:
        scenario = scenario_name_from_row(row)
        color = colors_by_scenario[scenario]

        ax.scatter(
            row["max_selected_overlap"],
            row["min_pair_separation"],
            s=200,
            marker=marker_for_scenario(scenario),
            facecolor=transparent_facecolor(color, alpha=ALPHA),
            edgecolor="k",
            linewidth=0.8,
            hatch=hatch_for_scenario(scenario),
            zorder=20,
            label=scenario_plot_label(scenario),
        )

    ax.axvline(
        overlap_threshold,
        color="k",
        linestyle="--",
        linewidth=1.2,
        zorder=5,
        label=f"overlap threshold = {overlap_threshold:.2f}",
    )
    ax.axhline(
        0.0,
        color="k",
        linestyle=":",
        linewidth=1.2,
        zorder=5,
        label="zero separation",
    )

    ax.annotate(
        "",
        xy=(0.92, 0.09),
        xytext=(0.64, 0.09),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.6,
            "color": "k",
        },
    )
    ax.text(
        0.78,
        0.13,
        "more overlap\nless safe",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    ax.annotate(
        "",
        xy=(0.08, 0.46),
        xytext=(0.08, 0.18),
        xycoords="axes fraction",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.6,
            "color": "k",
        },
    )
    ax.text(
        0.11,
        0.32,
        "larger source-lens\nseparation is safer",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=10,
        rotation=90,
    )

    ax.set_xlabel("maximum pair overlap")
    ax.set_ylabel("minimum pair separation")

    ax.legend(
        frameon=False,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        scatterpoints=1,
        labelspacing=1.2,
        handletextpad=0.8,
        columnspacing=1.6,
        handleheight=2.0,
        borderaxespad=0.8,
    )

    fig.tight_layout()

    return fig, ax


def plot_tomography_population_balance(rows):
    """Plot lens and source population balance across scenarios."""
    apply_dsf_plot_style()

    rows = valid_rows(rows)

    if not rows:
        return None, None

    rows = sorted(
        rows,
        key=lambda row: min(
            row["lens_fraction_balance"],
            row["source_fraction_balance"],
        ),
        reverse=True,
    )

    scenarios = [scenario_name_from_row(row) for row in rows]
    labels = [scenario_plot_label(scenario) for scenario in scenarios]
    colors_by_scenario = scenario_colors()

    y = np.arange(len(rows))

    lens_balance = np.asarray(
        [row["lens_fraction_balance"] for row in rows],
        dtype=float,
    )
    source_balance = np.asarray(
        [row["source_fraction_balance"] for row in rows],
        dtype=float,
    )

    colors = [
        transparent_facecolor(colors_by_scenario[scenario], alpha=ALPHA) for scenario in scenarios
    ]
    source_colors = [
        transparent_facecolor(colors_by_scenario[scenario], alpha=0.45) for scenario in scenarios
    ]
    hatches = [hatch_for_scenario(scenario) for scenario in scenarios]

    height = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(rows) + 2.0))

    lens_bars = ax.barh(
        y - height / 2.0,
        lens_balance,
        height,
        color=colors,
        edgecolor="k",
        linewidth=0.8,
    )
    apply_bar_hatches(lens_bars, hatches)

    source_bars = ax.barh(
        y + height / 2.0,
        source_balance,
        height,
        color=source_colors,
        edgecolor="k",
        linewidth=0.8,
    )
    apply_bar_hatches(source_bars, hatches)

    label_lens_source_bars(ax, lens_bars, source_bars)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel("min/max bin population fraction")
    ax.set_title("Tomography population balance")

    fig.tight_layout()

    return fig, ax


def plot_tomography_bin_widths(rows):
    """Plot mean 68 percent redshift widths across scenarios."""
    apply_dsf_plot_style()

    rows = valid_rows(rows)

    if not rows:
        return None, None

    rows = sorted(rows, key=lambda row: row["lens_mean_width_68"])

    scenarios = [scenario_name_from_row(row) for row in rows]
    labels = [scenario_plot_label(scenario) for scenario in scenarios]
    colors_by_scenario = scenario_colors()

    y = np.arange(len(rows))

    lens_width = np.asarray(
        [row["lens_mean_width_68"] for row in rows],
        dtype=float,
    )
    source_width = np.asarray(
        [row["source_mean_width_68"] for row in rows],
        dtype=float,
    )

    colors = [
        transparent_facecolor(colors_by_scenario[scenario], alpha=ALPHA) for scenario in scenarios
    ]
    source_colors = [
        transparent_facecolor(colors_by_scenario[scenario], alpha=0.45) for scenario in scenarios
    ]
    hatches = [hatch_for_scenario(scenario) for scenario in scenarios]

    height = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(rows) + 2.0))

    lens_bars = ax.barh(
        y - height / 2.0,
        lens_width,
        height,
        color=colors,
        edgecolor="k",
        linewidth=0.8,
    )
    apply_bar_hatches(lens_bars, hatches)

    source_bars = ax.barh(
        y + height / 2.0,
        source_width,
        height,
        color=source_colors,
        edgecolor="k",
        linewidth=0.8,
    )
    apply_bar_hatches(source_bars, hatches)

    label_lens_source_bars(ax, lens_bars, source_bars)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    xmax = 1.05 * max(
        np.nanmax(lens_width),
        np.nanmax(source_width),
    )
    ax.set_xlim(0.0, xmax)

    ax.set_xlabel("mean 68 percent redshift width")
    ax.set_title("Tomography bin width")

    fig.tight_layout()

    return fig, ax
