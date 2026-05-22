"""Redo Fisher and DALI contour plots from saved forecast arrays.

This script can be used in four main ways.

1. Redo contours for one completed run:

    PYTHONPATH=src:. python analysis/redo_forecast_contours_from_arrays.py \
        analysis/runs_output/dsf_desi_lrg_lsst_y1_hod_baryons_nla/baseline_lens3_source5_z0p3_0p9 \
        --config analysis/configs/dsf_desi_lrg_lsst_y1_hod_baryons_nla.yaml

2. Redo contours for every run directory inside a parent directory:

    PYTHONPATH=src:. python analysis/redo_forecast_contours_from_arrays.py \
        analysis/runs_output/dsf_desi_lrg_lsst_y1_hod_baryons_nla \
        --config analysis/configs/dsf_desi_lrg_lsst_y1_hod_baryons_nla.yaml \
        --all-runs

3. Redo contours for every run and also make combined overlay plots:

    PYTHONPATH=src:. python analysis/redo_forecast_contours_from_arrays.py \
        analysis/runs_output/dsf_desi_lrg_lsst_y1_hod_baryons_nla \
        --config analysis/configs/dsf_desi_lrg_lsst_y1_hod_baryons_nla.yaml \
        --all-runs \
        --combined

4. Only make combined overlay plots from already-saved Fisher/DALI samples:

    PYTHONPATH=src:. python analysis/redo_forecast_contours_from_arrays.py \
        analysis/runs_output/dsf_desi_lrg_lsst_y1_hod_baryons_nla \
        --config analysis/configs/dsf_desi_lrg_lsst_y1_hod_baryons_nla.yaml \
        --all-runs \
        --combined \
        --skip-individual

Useful options:

    --include-dali
        Also make DALI plots. This uses saved dali_samples.npy files when
        available, and only resamples if needed.

    --resample-dali
        Force resampling from saved dali.npy even if dali_samples.npy already
        exists.

    --combined
        Make combined overlay plots across all run directories.

    --skip-individual
        Do not regenerate per-run plots. Useful when you only want the
        combined plot.

    --max-runs N
        Limit the number of child run directories processed. Useful for tests.

Examples for one directory:

    PYTHONPATH=src:. python analysis/redo_forecast_contours_from_arrays.py \
        analysis/runs_output/dsf_desi_lrg_lsst_y1_hod_baryons_nla \
        --config analysis/configs/dsf_desi_lrg_lsst_y1_hod_baryons_nla.yaml \
        --all-runs \
        --combined \
        --include-dali

This will process:

    baseline_lens3_source5_z0p3_0p9
    fine_wide_z_lens5_source6_z0p1_1
    stress_wide_z_lens6_source7_z0p1_1

and save combined plots under:

    analysis/runs_output/dsf_desi_lrg_lsst_y1_hod_baryons_nla/plots/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from getdist import MCSamples

from analysis.plot_scripts.plot_contours import (
    fisher_to_gaussian,
    plot_dali_contours,
    plot_fisher_contours,
)
from analysis.scripts.config import load_config
from analysis.scripts.forecast import (
    add_gaussian_prior_to_fisher,
    make_forecast_kit,
    sample_dali,
)
from analysis.scripts.model_setup import (
    active_parameter_labels,
    active_parameter_names,
)


def load_array(path):
    """Load a NumPy array and unwrap scalar object arrays."""
    value = np.load(path, allow_pickle=True)

    if value.shape == ():
        return value.item()

    return value


def get_item(value, key):
    """Return a dictionary item or the value itself."""
    if isinstance(value, dict):
        return value[key]

    return value


def zero_model_from_covariance(covariance):
    """Create a zero model with the same data-vector size as the covariance."""
    n_data = covariance.shape[0]

    def model(theta):
        return np.zeros(n_data)

    return model


def config_with_ordered_dali_bounds(config, names, *, use_sampler_bounds=False):
    """Convert DALI prior bounds from name-keyed dicts to ordered lists."""
    config = dict(config)
    config["dali"] = dict(config.get("dali", {}))
    config["dali"]["priors"] = dict(config["dali"].get("priors", {}))

    bounds = config["dali"]["priors"].get("bounds")

    if isinstance(bounds, dict):
        ordered_bounds = [tuple(bounds[name]) for name in names]

        config["dali"]["priors"]["bounds"] = ordered_bounds

        if use_sampler_bounds:
            config["dali"]["sampler_bounds"] = ordered_bounds
        else:
            config["dali"]["sampler_bounds"] = None

    return config


def run_label_from_path(run_dir):
    """Return a readable label for one run directory."""
    return run_dir.name.replace("_", " ")


def has_required_arrays(run_dir):
    """Check whether a directory looks like a completed forecast run."""
    array_dir = run_dir / "arrays"

    required = [
        array_dir / "theta0.npy",
        array_dir / "covariance.npy",
        array_dir / "fisher.npy",
    ]

    return array_dir.is_dir() and all(path.exists() for path in required)


def find_run_dirs(path, *, all_runs=False, max_runs=None):
    """Find one or more completed run directories."""
    path = Path(path)

    if path.name == "arrays":
        path = path.parent

    if not all_runs:
        return [path]

    run_dirs = [
        child for child in sorted(path.iterdir()) if child.is_dir() and has_required_arrays(child)
    ]

    if max_runs is not None:
        run_dirs = run_dirs[:max_runs]

    return run_dirs


def make_saved_dali_samples(samples_path, names, labels, label):
    """Build a GetDist MCSamples object from saved samples."""
    samples = load_array(samples_path)

    return MCSamples(
        samples=samples,
        names=names,
        labels=labels,
        label=label,
    )


def make_or_load_dali_samples(
    *,
    array_dir,
    config,
    sampling_config,
    names,
    labels,
    covariance,
    dali,
    label,
    include_config_priors,
    samples_filename,
    resample_dali=False,
):
    """Load DALI samples when possible or sample them from saved DALI arrays."""
    samples_path = array_dir / samples_filename

    if samples_path.exists() and not resample_dali:
        return make_saved_dali_samples(
            samples_path,
            names=names,
            labels=labels,
            label=label,
        )

    kit = make_forecast_kit(
        model=zero_model_from_covariance(covariance),
        theta0=load_array(array_dir / "theta0.npy"),
        covariance=covariance,
    )

    samples = sample_dali(
        config=sampling_config,
        kit=kit,
        dali=dali,
        names=names,
        labels=labels,
        label=label,
        include_config_priors=include_config_priors,
    )

    np.save(samples_path, samples.samples)

    return samples


def process_one_run(
    run_dir,
    *,
    config,
    sampling_config,
    names,
    labels,
    include_dali=False,
    resample_dali=False,
    make_individual=True,
):
    """Redo contours for one completed forecast run."""
    run_dir = Path(run_dir)

    if run_dir.name == "arrays":
        run_dir = run_dir.parent

    array_dir = run_dir / "arrays"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    theta0 = load_array(array_dir / "theta0.npy")
    covariance = load_array(array_dir / "covariance.npy")

    fisher_result = load_array(array_dir / "fisher.npy")
    fisher = get_item(fisher_result, "fisher")

    fisher_with_priors = add_gaussian_prior_to_fisher(
        fisher=fisher,
        config=config,
        parameter_names=names,
    )

    np.save(array_dir / "fisher_with_priors.npy", fisher_with_priors)

    run_label = run_label_from_path(run_dir)

    fisher_contour = fisher_to_gaussian(
        fisher=fisher,
        theta0=theta0,
        names=names,
        labels=labels,
        label=f"{run_label}: Fisher",
    )

    fisher_prior_contour = fisher_to_gaussian(
        fisher=fisher_with_priors,
        theta0=theta0,
        names=names,
        labels=labels,
        label=f"{run_label}: Fisher + priors",
    )

    if make_individual:
        plot_fisher_contours(
            [fisher_contour],
            names,
            labels=["Fisher"],
            save_stem=plot_dir / "fisher_contours",
        )

        plot_fisher_contours(
            [fisher_contour, fisher_prior_contour],
            names,
            labels=["Fisher", "Fisher + priors"],
            save_stem=plot_dir / "fisher_contours_with_priors",
        )

        plot_fisher_contours(
            [fisher_prior_contour],
            names,
            labels=["Fisher + priors"],
            save_stem=plot_dir / "fisher_priors_only_contours",
        )

    result = {
        "run_dir": run_dir,
        "label": run_label,
        "fisher": fisher_contour,
        "fisher_with_priors": fisher_prior_contour,
        "dali": None,
        "dali_with_priors": None,
    }

    if not include_dali:
        print(f"[DSF] Saved Fisher plots to {plot_dir}")
        return result

    dali_path = array_dir / "dali.npy"

    if not dali_path.exists():
        print(f"[DSF] Skipping DALI for {run_dir.name}: missing {dali_path}")
        return result

    dali_result = load_array(dali_path)
    dali = get_item(dali_result, "dali")

    print(f"[DALI] Processing {run_dir.name}")
    print("[DALI] names:", names)
    print("[DALI] theta0:", theta0)
    print("[DALI] priors:", sampling_config.get("dali", {}).get("priors"))
    print("[DALI] sampler_bounds:", sampling_config.get("dali", {}).get("sampler_bounds"))

    dali_samples = make_or_load_dali_samples(
        array_dir=array_dir,
        config=config,
        sampling_config=sampling_config,
        names=names,
        labels=labels,
        covariance=covariance,
        dali=dali,
        label=f"{run_label}: DALI",
        include_config_priors=False,
        samples_filename="dali_samples.npy",
        resample_dali=resample_dali,
    )

    dali_prior_samples = make_or_load_dali_samples(
        array_dir=array_dir,
        config=config,
        sampling_config=sampling_config,
        names=names,
        labels=labels,
        covariance=covariance,
        dali=dali,
        label=f"{run_label}: DALI + priors",
        include_config_priors=True,
        samples_filename="dali_with_priors_samples.npy",
        resample_dali=resample_dali,
    )

    result["dali"] = dali_samples
    result["dali_with_priors"] = dali_prior_samples

    if make_individual:
        plot_dali_contours(
            [dali_samples],
            names,
            labels=["DALI"],
            save_stem=plot_dir / "dali_contours",
        )

        plot_dali_contours(
            [dali_samples, dali_prior_samples],
            names,
            labels=["DALI", "DALI + priors"],
            save_stem=plot_dir / "dali_contours_with_priors",
        )

        plot_dali_contours(
            [dali_prior_samples],
            names,
            labels=["DALI + priors"],
            save_stem=plot_dir / "dali_priors_only_contours",
        )

    print(f"[DSF] Saved plots to {plot_dir}")

    return result


def plot_combined_results(results, names, output_dir, *, include_dali=False):
    """Plot combined contours across all processed run directories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fisher_contours = [result["fisher"] for result in results if result["fisher"] is not None]

    fisher_prior_contours = [
        result["fisher_with_priors"]
        for result in results
        if result["fisher_with_priors"] is not None
    ]

    fisher_labels = [result["label"] for result in results if result["fisher"] is not None]

    fisher_prior_labels = [
        result["label"] for result in results if result["fisher_with_priors"] is not None
    ]

    if len(fisher_contours) > 0:
        plot_fisher_contours(
            fisher_contours,
            names,
            labels=fisher_labels,
            save_stem=output_dir / "combined_fisher_contours",
        )

    if len(fisher_prior_contours) > 0:
        plot_fisher_contours(
            fisher_prior_contours,
            names,
            labels=fisher_prior_labels,
            save_stem=output_dir / "combined_fisher_priors_contours",
        )

    if len(fisher_contours) > 0 and len(fisher_prior_contours) > 0:
        combined = []
        combined_labels = []

        for result in results:
            if result["fisher"] is not None:
                combined.append(result["fisher"])
                combined_labels.append(f"{result['label']}: Fisher")

            if result["fisher_with_priors"] is not None:
                combined.append(result["fisher_with_priors"])
                combined_labels.append(f"{result['label']}: Fisher + priors")

        plot_fisher_contours(
            combined,
            names,
            labels=combined_labels,
            save_stem=output_dir / "combined_fisher_all_contours",
        )

    if not include_dali:
        return

    dali_contours = [result["dali"] for result in results if result["dali"] is not None]

    dali_prior_contours = [
        result["dali_with_priors"] for result in results if result["dali_with_priors"] is not None
    ]

    dali_labels = [result["label"] for result in results if result["dali"] is not None]

    dali_prior_labels = [
        result["label"] for result in results if result["dali_with_priors"] is not None
    ]

    if len(dali_contours) > 0:
        plot_dali_contours(
            dali_contours,
            names,
            labels=dali_labels,
            save_stem=output_dir / "combined_dali_contours",
        )

    if len(dali_prior_contours) > 0:
        plot_dali_contours(
            dali_prior_contours,
            names,
            labels=dali_prior_labels,
            save_stem=output_dir / "combined_dali_priors_contours",
        )

    if len(dali_contours) > 0 and len(dali_prior_contours) > 0:
        combined = []
        combined_labels = []

        for result in results:
            if result["dali"] is not None:
                combined.append(result["dali"])
                combined_labels.append(f"{result['label']}: DALI")

            if result["dali_with_priors"] is not None:
                combined.append(result["dali_with_priors"])
                combined_labels.append(f"{result['label']}: DALI + priors")

        plot_dali_contours(
            combined,
            names,
            labels=combined_labels,
            save_stem=output_dir / "combined_dali_all_contours",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path")
    parser.add_argument("--config", required=True)

    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Treat run_path as a parent directory containing multiple completed run directories.",
    )

    parser.add_argument(
        "--combined",
        action="store_true",
        help="Make combined overlay contour plots across all processed runs.",
    )

    parser.add_argument(
        "--skip-individual",
        action="store_true",
        help="Skip per-run plots and only make combined plots.",
    )

    parser.add_argument(
        "--include-dali",
        action="store_true",
        help="Also make DALI plots. Uses saved DALI samples when available.",
    )

    parser.add_argument(
        "--resample-dali",
        action="store_true",
        help="Force DALI resampling even if saved DALI samples already exist.",
    )

    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Only process the first N run directories. Useful for testing.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    names = active_parameter_names(config)
    labels = active_parameter_labels(config)

    sampling_config = config_with_ordered_dali_bounds(
        config,
        names,
        use_sampler_bounds=False,
    )

    run_dirs = find_run_dirs(
        args.run_path,
        all_runs=args.all_runs,
        max_runs=args.max_runs,
    )

    if len(run_dirs) == 0:
        raise RuntimeError(f"No completed run directories found in {args.run_path}")

    print("[DSF] Found run directories:")
    for run_dir in run_dirs:
        print(f"  - {run_dir}")

    results = []

    for run_dir in run_dirs:
        result = process_one_run(
            run_dir,
            config=config,
            sampling_config=sampling_config,
            names=names,
            labels=labels,
            include_dali=args.include_dali,
            resample_dali=args.resample_dali,
            make_individual=not args.skip_individual,
        )
        results.append(result)

    if args.combined:
        if args.all_runs:
            combined_plot_dir = Path(args.run_path) / "plots"
        else:
            combined_plot_dir = Path(run_dirs[0]) / "plots"

        plot_combined_results(
            results,
            names,
            combined_plot_dir,
            include_dali=args.include_dali,
        )

        print(f"[DSF] Saved combined plots to {combined_plot_dir}")


if __name__ == "__main__":
    main()
