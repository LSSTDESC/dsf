"""Signal-to-noise plotting scripts for Delta Sigma forecast runs."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

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
from analysis.scripts.diagnostic_metrics import (
    cumulative_signal_to_noise,
    signal_to_noise_by_pair,
    signal_to_noise_by_radius,
)

__all__ = [
    "plot_signal_to_noise_by_pair",
    "plot_signal_to_noise_by_radius",
    "plot_cumulative_signal_to_noise",
    "plot_snr_by_radius_for_scenarios",
]


def _pair_label(pair):
    """Return a readable lens-source bin-pair label."""
    lens_bin, source_bin = pair
    return rf"$(L{int(lens_bin) + 1}-S{int(source_bin) + 1})$"


def _scenario_colors():
    """Return a fixed color lookup for tomography scenarios."""
    colors = binny_colors(len(SCENARIO_COLOR_ORDER))
    return scenario_color_map(colors)


def _style_for_scenario(scenario):
    """Return color, marker, and hatch for one scenario."""
    colors_by_scenario = _scenario_colors()

    if scenario is None:
        color = binny_colors(3)[0]
        marker = "o"
        hatch = None
        label = None
    else:
        scenario = str(scenario)
        color = colors_by_scenario[scenario]
        marker = marker_for_scenario(scenario)
        hatch = hatch_for_scenario(scenario)
        label = scenario_plot_label(scenario)

    return color, marker, hatch, label


def plot_signal_to_noise_by_pair(
    data_vector,
    covariance,
    bin_pairs,
    n_radius,
    *,
    scenario=None,
    figsize=None,
):
    """Plot signal-to-noise for each selected lens-source pair."""
    apply_dsf_plot_style()

    snr_by_pair, _ = signal_to_noise_by_pair(
        data_vector,
        covariance,
        bin_pairs,
        n_radius,
    )

    labels = [_pair_label(pair) for pair in snr_by_pair]
    values = np.asarray(list(snr_by_pair.values()), dtype=float)

    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    y = np.arange(len(values))

    if scenario is None:
        colors = binny_colors(len(values))
        facecolors = [transparent_facecolor(color, alpha=ALPHA) for color in colors]
        hatches = [None for _ in values]
    else:
        color, _, hatch, _ = _style_for_scenario(scenario)
        facecolors = [transparent_facecolor(color, alpha=ALPHA) for _ in values]
        hatches = [hatch for _ in values]

    bars = ax.barh(
        y,
        values,
        color=facecolors,
        edgecolor="k",
        linewidth=2.4,
    )

    for bar, hatch in zip(bars, hatches, strict=True):
        if hatch is not None:
            bar.set_hatch(hatch)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"${\rm SNR}$")
    ax.set_ylabel("Lens-Source pair")

    ax.invert_yaxis()

    fig.tight_layout()

    return fig, ax


def plot_signal_to_noise_by_radius(
    r,
    data_vector,
    covariance,
    bin_pairs,
    *,
    scenario=None,
    figsize=None,
):
    """Plot signal-to-noise by projected-radius bin across all pairs."""
    apply_dsf_plot_style()

    r = np.asarray(r, dtype=float).reshape(-1)

    n_pairs = len(bin_pairs)
    n_radius = len(r)

    snr_by_radius, _ = signal_to_noise_by_radius(
        data_vector,
        covariance,
        n_pairs,
        n_radius,
    )

    values = np.asarray(list(snr_by_radius.values()), dtype=float)

    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    color, marker, hatch, label = _style_for_scenario(scenario)

    ax.plot(
        r,
        values,
        color=color,
        linewidth=2.4,
        alpha=ALPHA,
        zorder=2,
        label=label,
    )

    # White/opaque mask under the marker so the line does not show through.
    ax.scatter(
        r,
        values,
        s=150,
        marker=marker,
        facecolor="white",
        edgecolor="white",
        linewidth=0.0,
        zorder=3,
    )

    # Actual marker on top.
    ax.scatter(
        r,
        values,
        s=70,
        marker=marker,
        facecolor=transparent_facecolor(color, alpha=ALPHA),
        edgecolor="k",
        linewidth=1.2,
        hatch=hatch,
        zorder=4,
    )

    n_ticks = len(r)
    tick_indices = np.linspace(0, len(r) - 1, n_ticks, dtype=int)
    tick_values = r[tick_indices]

    ax.set_xticks(tick_values)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))

    ax.set_xlabel(r"$R\,[h^{-1}{\rm Mpc}]$")
    ax.set_ylabel(r"${\rm SNR}$")

    if label is not None:
        ax.legend(frameon=False)

    fig.tight_layout()

    return fig, ax


def plot_cumulative_signal_to_noise(
    data_vector,
    covariance,
    *,
    bin_pairs=None,
    r=None,
    scenario=None,
    figsize=None,
):
    """Plot cumulative signal-to-noise along the stacked data vector."""
    apply_dsf_plot_style()

    cumulative_snr, _ = cumulative_signal_to_noise(data_vector, covariance)
    cumulative_snr = np.asarray(cumulative_snr, dtype=float)

    index = np.arange(1, len(cumulative_snr) + 1)

    if figsize is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=figsize)

    color, marker, hatch, label = _style_for_scenario(scenario)

    ax.plot(
        index,
        cumulative_snr,
        color=color,
        linewidth=2.4,
        alpha=ALPHA,
        zorder=2,
        label=label,
    )

    # White/opaque mask under the marker so the line does not show through.
    ax.scatter(
        index,
        cumulative_snr,
        s=70,
        marker=marker,
        facecolor="white",
        edgecolor="white",
        linewidth=0.0,
        zorder=3,
    )

    # Actual marker on top.
    ax.scatter(
        index,
        cumulative_snr,
        s=70,
        marker=marker,
        facecolor=transparent_facecolor(color, alpha=ALPHA),
        edgecolor="k",
        linewidth=1.2,
        hatch=hatch,
        zorder=4,
    )

    data_vector = np.asarray(data_vector, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    info_lines = [
        rf"Total SNR $= {cumulative_snr[-1]:.2f}$",
        rf"Data vector length $= {data_vector.size}$",
        rf"Covariance shape $= {covariance.shape[0]} \times {covariance.shape[1]}$",
    ]

    if bin_pairs is not None:
        n_pairs = len(bin_pairs)
        info_lines.append(rf"Lens-source pairs $= {n_pairs}$")

        if r is not None:
            n_r = len(np.asarray(r, dtype=float).reshape(-1))
            info_lines.append(rf"Radial points $= {n_r}$")
            info_lines.append(rf"Block data shape $= {n_pairs} \times {n_r}$")

    info_text = "\n".join(info_lines)

    ax.text(
        0.04,
        0.96,
        info_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.75",
            "alpha": 0.9,
        },
        zorder=10,
    )

    ax.set_xlabel("Data vector element")
    ax.set_ylabel(r"Cumulative ${\rm SNR}$")

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if label is not None:
        ax.legend(frameon=False)

    fig.tight_layout()

    return fig, ax


def plot_snr_by_radius_for_scenarios(
    rows,
    *,
    figsize=(10, 7),
):
    """Plot SNR by projected radius for multiple forecast scenarios."""
    apply_dsf_plot_style()

    good_rows = [
        row for row in rows if len(row["snr_by_radius"]) > 0 and np.all(np.isfinite(row["r"]))
    ]

    if len(good_rows) == 0:
        return None, None

    colors_by_scenario = _scenario_colors()

    fig, ax = plt.subplots(figsize=figsize)

    for row in good_rows:
        scenario = scenario_name_from_row(row)
        color = colors_by_scenario[scenario]

        r = np.asarray(row["r"], dtype=float)
        snr_by_radius = row["snr_by_radius"]

        if isinstance(snr_by_radius, dict):
            values = np.asarray(list(snr_by_radius.values()), dtype=float)
        else:
            values = np.asarray(snr_by_radius, dtype=float)

        ax.plot(
            r,
            values,
            color=color,
            linewidth=2.4,
            alpha=ALPHA,
            zorder=2,
        )

        ax.scatter(
            r,
            values,
            s=185,
            marker=marker_for_scenario(scenario),
            facecolor=transparent_facecolor(color, alpha=ALPHA),
            edgecolor="k",
            linewidth=0.8,
            hatch=hatch_for_scenario(scenario),
            zorder=20,
            label=scenario_plot_label(scenario),
        )

    r_ticks = np.asarray(good_rows[0]["r"], dtype=float)
    n_ticks = len(r_ticks)
    tick_indices = np.linspace(0, len(r_ticks) - 1, n_ticks, dtype=int)
    tick_values = r_ticks[tick_indices]

    ax.set_xticks(tick_values)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))

    ax.set_xlabel(r"$R\,[h^{-1}{\rm Mpc}]$")
    ax.set_ylabel(r"${\rm SNR}$")

    ax.legend(
        frameon=False,
        fontsize=11,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=5,
        scatterpoints=1,
        labelspacing=1.2,
        handletextpad=0.8,
        columnspacing=1.6,
        handleheight=2.0,
        borderaxespad=0.8,
    )

    fig.tight_layout()

    return fig, ax
