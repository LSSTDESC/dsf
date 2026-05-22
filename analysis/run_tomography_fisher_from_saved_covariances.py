"""Run Fisher and optional DALI forecasts from saved tomography covariances.

This script post-processes existing tomography SNR runs. It does not rebuild
the expensive covariance matrices. Instead, for each saved tomography scenario,
it loads:

    arrays/covariance.npy
    arrays/theta0.npy
    arrays/r.npy
    arrays/rp_bin_edges.npy
    arrays/bin_pairs.npy
    arrays/data_vector.npy

Then it rebuilds the corresponding Delta Sigma model function and runs Fisher
and optionally DALI using the saved covariance.

Run from the project root with:

    python analysis/run_tomography_fisher_from_saved_covariances.py
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from analysis.configs.tomography_sweep_config import (
    OUTPUT_DIR,
    SCENARIOS,
    case_label,
    lens_overrides,
    scenario_values,
    source_overrides,
)
from analysis.plot_scripts.plot_contours import (
    fisher_to_gaussian,
    plot_dali_contours,
    plot_fisher_contours,
)
from analysis.scripts.config import (
    load_config,
)
from analysis.scripts.forecast import (
    run_dali,
    run_fisher,
    sample_dali,
)
from analysis.scripts.io import make_run_output
from analysis.scripts.model_setup import (
    active_parameter_labels,
    active_parameter_names,
    make_fiducial_cosmology,
    make_theta_mapper,
    pk2d_model_func,
)
from dsf.delta_sigma_forecast_builder import DeltaSigmaForecastBuilder
from src.dsf.tomography.tomo_builder import TomographyBuilder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_CONFIG_PATH = PROJECT_ROOT / "analysis/configs/dsf_desi_lrg_lsst_y1_hod.yaml"

RUN_SCENARIOS = list(SCENARIOS)

RUN_FISHER = True
RUN_DALI = False
RUN_DALI_SAMPLING = False

OVERWRITE_EXISTING = True


def apply_tomography_scenario(config, scenario_name):
    """Apply one tomography sweep scenario to a forecast config."""
    _, _, lens_n_bins, source_n_bins, z_range = scenario_values(scenario_name)

    config.setdefault("tomography", {})
    config["tomography"].setdefault("lens", {})
    config["tomography"].setdefault("source", {})

    config["tomography"]["lens"]["overrides"] = lens_overrides(
        n_bins=lens_n_bins,
        z_range=z_range,
    )
    config["tomography"]["source"]["overrides"] = source_overrides(
        n_bins=source_n_bins,
    )

    label = case_label(
        scenario_name=scenario_name,
        lens_n_bins=lens_n_bins,
        source_n_bins=source_n_bins,
        z_range=z_range,
    )

    config["run_name"] = label

    return label


def load_saved_tomography_arrays(case_dir):
    """Load saved arrays from one tomography scenario directory."""
    array_dir = case_dir / "arrays"

    paths = {
        "theta0": array_dir / "theta0.npy",
        "data_vector": array_dir / "data_vector.npy",
        "covariance": array_dir / "covariance.npy",
        "r": array_dir / "r.npy",
        "rp_bin_edges": array_dir / "rp_bin_edges.npy",
        "bin_pairs": array_dir / "bin_pairs.npy",
    }

    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Cannot run forecasts for {case_dir.name}; missing files:\n{missing_text}"
        )

    arrays = {name: np.load(path, allow_pickle=True) for name, path in paths.items()}

    arrays["theta0"] = np.asarray(arrays["theta0"], dtype=float)
    arrays["data_vector"] = np.asarray(arrays["data_vector"], dtype=float)
    arrays["covariance"] = np.asarray(arrays["covariance"], dtype=float)
    arrays["r"] = np.asarray(arrays["r"], dtype=float)
    arrays["rp_bin_edges"] = np.asarray(arrays["rp_bin_edges"], dtype=float)
    arrays["bin_pairs"] = np.asarray(arrays["bin_pairs"], dtype=int)

    return arrays


def validate_saved_arrays(case_label_name, arrays):
    """Check that saved tomography arrays are internally consistent."""
    data_vector = arrays["data_vector"]
    covariance = arrays["covariance"]

    if covariance.ndim != 2:
        raise ValueError(f"{case_label_name}: covariance must be 2D, got {covariance.shape}.")

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{case_label_name}: covariance must be square, got {covariance.shape}.")

    expected_shape = (data_vector.size, data_vector.size)
    if covariance.shape != expected_shape:
        raise ValueError(
            f"{case_label_name}: covariance shape {covariance.shape} does not "
            f"match data vector size {data_vector.size}; expected "
            f"{expected_shape}."
        )

    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"{case_label_name}: covariance contains non-finite values.")

    if not np.all(np.isfinite(data_vector)):
        raise ValueError(f"{case_label_name}: data vector contains non-finite values.")

    if arrays["theta0"].ndim != 1:
        raise ValueError(
            f"{case_label_name}: theta0 must be one-dimensional, got {arrays['theta0'].shape}."
        )


def build_forecast_from_saved_arrays(config, arrays):
    """Build forecast inputs using saved data vector and covariance arrays."""
    cosmo = make_fiducial_cosmology(config)

    theta0 = arrays["theta0"]
    names = active_parameter_names(config)

    survey = config["survey"]
    tomography_config = config["tomography"]
    selection = config.get("bin_pair_selection", {})
    covariance_config = config.get("covariance", {})

    builder = DeltaSigmaForecastBuilder(
        cosmo=cosmo,
        pk2d_func=pk2d_model_func(config),
        theta0=theta0,
        theta_mapper=make_theta_mapper(config),
        parameter_names=names,
        r=arrays["r"],
        rp_bin_edges=arrays["rp_bin_edges"],
        area_deg2=survey["area_deg2"],
        sigma_e=survey.get("sigma_e", 0.26),
        lens_survey=survey.get("lens_survey", "lsst"),
        source_survey=survey.get("source_survey", "lsst"),
        lens_role=tomography_config["lens"].get("role", "lens"),
        source_role=tomography_config["source"].get("role", "source"),
        lens_sample=tomography_config["lens"].get("sample"),
        source_sample=tomography_config["source"].get("sample"),
        lens_year=tomography_config["lens"].get("year"),
        source_year=tomography_config["source"].get("year"),
        lens_overrides=tomography_config["lens"].get("overrides"),
        source_overrides=tomography_config["source"].get("overrides"),
        overlap_threshold=selection.get("overlap_threshold", 0.10),
        source_behind_lens=selection.get("source_behind_lens", True),
        center_method=selection.get("center_method", "mean"),
        covariance_kind=covariance_config.get("kind", "gm"),
        nonlinear=covariance_config.get("nonlinear", True),
        galaxy_bias=covariance_config.get("galaxy_bias"),
        galaxy_bias_prefactor=covariance_config.get("galaxy_bias_prefactor", 1.0),
        verbose=False,
    )

    tomography = TomographyBuilder(**builder.tomography_kwargs).prepare_bins()

    saved_bin_pairs = [
        (int(lens_bin), int(source_bin)) for lens_bin, source_bin in arrays["bin_pairs"]
    ]
    rebuilt_bin_pairs = [
        (int(lens_bin), int(source_bin)) for lens_bin, source_bin in tomography["bin_pairs"]
    ]

    if rebuilt_bin_pairs != saved_bin_pairs:
        raise ValueError("Rebuilt tomography bin pairs do not match saved bin_pairs.npy.")

    context = {
        "cosmo": builder.cosmo,
        "calculator": builder.calculator,
        "r": builder.r,
        "rp_bin_edges": builder.rp_bin_edges,
        "tomography": tomography,
        "lens_result": tomography["lens_result"],
        "source_result": tomography["source_result"],
        "bin_pairs": rebuilt_bin_pairs,
        "lens_bin_centers": tomography["lens_bin_centers"],
        "source_bin_centers": tomography["source_bin_centers"],
        "lens_shape_stats": tomography["lens_shape_stats"],
        "source_shape_stats": tomography["source_shape_stats"],
        "covariance_builder": None,
        "cov_pairs": rebuilt_bin_pairs,
        "covariance_kind": builder.covariance_kind,
        "observable": builder.observable,
    }

    def model(theta):
        return builder.model(theta, context=context)

    forecast = {
        "model": model,
        "theta0": arrays["theta0"],
        "parameter_names": names,
        "cov": arrays["covariance"],
        "data_vector": arrays["data_vector"],
        "fiducial_data_vector": arrays["data_vector"],
        "r": arrays["r"],
        "rp_bin_edges": arrays["rp_bin_edges"],
        "bin_pairs": rebuilt_bin_pairs,
        "cov_pairs": rebuilt_bin_pairs,
        "lens_result": tomography["lens_result"],
        "source_result": tomography["source_result"],
        "lens_population_stats": tomography["lens_population_stats"],
        "source_population_stats": tomography["source_population_stats"],
        "lens_bin_centers": tomography["lens_bin_centers"],
        "source_bin_centers": tomography["source_bin_centers"],
        "tomography": tomography,
        "covariance_builder": None,
        "calculator": builder.calculator,
        "context": context,
        "covariance_kind": builder.covariance_kind,
    }

    return forecast


def validate_forecast_against_saved_arrays(case_label_name, forecast, arrays):
    """Check that the rebuilt model setup matches the saved tomography arrays."""
    forecast_bin_pairs = np.asarray(forecast["bin_pairs"], dtype=int)
    saved_bin_pairs = np.asarray(arrays["bin_pairs"], dtype=int)

    if forecast_bin_pairs.shape != saved_bin_pairs.shape:
        raise ValueError(
            f"{case_label_name}: rebuilt bin-pair shape "
            f"{forecast_bin_pairs.shape} does not match saved bin-pair shape "
            f"{saved_bin_pairs.shape}."
        )

    if not np.array_equal(forecast_bin_pairs, saved_bin_pairs):
        raise ValueError(f"{case_label_name}: rebuilt bin pairs do not match saved bin pairs.")

    model_at_theta0 = np.asarray(forecast["model"](arrays["theta0"]), dtype=float)
    saved_data_vector = np.asarray(arrays["data_vector"], dtype=float)

    if model_at_theta0.shape != saved_data_vector.shape:
        raise ValueError(
            f"{case_label_name}: model output shape {model_at_theta0.shape} "
            f"does not match saved data vector shape {saved_data_vector.shape}."
        )

    if not np.allclose(model_at_theta0, saved_data_vector, rtol=1.0e-5, atol=1.0e-8):
        max_abs = np.max(np.abs(model_at_theta0 - saved_data_vector))
        print(
            f"[DSF] Warning: {case_label_name} rebuilt model(theta0) does not "
            f"exactly match saved data_vector.npy. Max abs difference: {max_abs:.6e}"
        )


def result_exists(case_dir, name):
    """Return whether a result array already exists for one case."""
    return (case_dir / "arrays" / f"{name}.npy").exists()


def save_forecast_result(run_output, name, result):
    """Save one forecast result array."""
    if result is None:
        return None

    return run_output.save_array(name, result)


def plot_fisher_results(
    *,
    run_output,
    theta0,
    names,
    labels,
    fisher,
    fisher_with_priors=None,
):
    """Plot Fisher contours separately and with priors."""
    fisher_gaussian = fisher_to_gaussian(
        fisher=fisher,
        theta0=theta0,
        names=names,
        labels=labels,
        label="Fisher",
    )

    plot_fisher_contours(
        [fisher_gaussian],
        names,
        labels=["Fisher"],
        save_stem=run_output.plot_dir / "fisher_contours",
    )

    if fisher_with_priors is None:
        return

    fisher_prior_gaussian = fisher_to_gaussian(
        fisher=fisher_with_priors,
        theta0=theta0,
        names=names,
        labels=labels,
        label="Fisher + priors",
    )

    plot_fisher_contours(
        [fisher_prior_gaussian],
        names,
        labels=["Fisher + priors"],
        save_stem=run_output.plot_dir / "fisher_with_priors_contours",
    )

    plot_fisher_contours(
        [fisher_gaussian, fisher_prior_gaussian],
        names,
        labels=["Fisher", "Fisher + priors"],
        save_stem=run_output.plot_dir / "fisher_prior_comparison_contours",
    )


def plot_dali_results(
    *,
    run_output,
    names,
    dali_samples=None,
    dali_samples_with_priors=None,
):
    """Plot DALI contours separately and with priors."""
    if dali_samples is not None:
        plot_dali_contours(
            [dali_samples],
            names,
            labels=["DALI"],
            save_stem=run_output.plot_dir / "dali_contours",
        )

    if dali_samples_with_priors is not None:
        plot_dali_contours(
            [dali_samples_with_priors],
            names,
            labels=["DALI + priors"],
            save_stem=run_output.plot_dir / "dali_with_priors_contours",
        )

    if dali_samples is not None and dali_samples_with_priors is not None:
        plot_dali_contours(
            [dali_samples, dali_samples_with_priors],
            names,
            labels=["DALI", "DALI + priors"],
            save_stem=run_output.plot_dir / "dali_prior_comparison_contours",
        )


def run_one_case(base_config, scenario_name):
    """Run Fisher and optional DALI for one saved tomography scenario."""
    config = deepcopy(base_config)
    label = apply_tomography_scenario(config, scenario_name)

    case_dir = OUTPUT_DIR / label
    array_dir = case_dir / "arrays"

    if not case_dir.exists():
        print(f"[DSF] Skipping {label}: directory does not exist.")
        return None

    if RUN_FISHER and result_exists(case_dir, "fisher") and not OVERWRITE_EXISTING:
        print(f"[DSF] Skipping {label}: fisher.npy already exists.")
        return None

    arrays = load_saved_tomography_arrays(case_dir)
    validate_saved_arrays(label, arrays)

    forecast = build_forecast_from_saved_arrays(
        config=config,
        arrays=arrays,
    )
    validate_forecast_against_saved_arrays(
        case_label_name=label,
        forecast=forecast,
        arrays=arrays,
    )

    names = active_parameter_names(config)
    parameter_labels = active_parameter_labels(config)

    run_output = make_run_output(
        run_name=label,
        base_dir=OUTPUT_DIR,
    )

    fisher_result = None
    dali_result = None
    dali_samples = None
    dali_samples_with_priors = None

    if RUN_FISHER:
        print(f"[DSF]   running Fisher for {label}")
        fisher_output = run_fisher(
            model=forecast["model"],
            theta0=forecast["theta0"],
            covariance=forecast["cov"],
            fisher_config=config.get("fisher", {}),
            parameter_names=names,
            config=config,
        )

        if fisher_output is not None:
            _, fisher_results = fisher_output

            fisher_result = fisher_results["fisher"]
            fisher_with_priors = fisher_results["fisher_with_priors"]

            save_forecast_result(run_output, "fisher", fisher_result)

            if fisher_with_priors is not None:
                save_forecast_result(
                    run_output,
                    "fisher_with_priors",
                    fisher_with_priors,
                )

            plot_fisher_results(
                run_output=run_output,
                theta0=forecast["theta0"],
                names=names,
                labels=parameter_labels,
                fisher=fisher_result,
                fisher_with_priors=fisher_with_priors,
            )

    if RUN_DALI:
        print(f"[DSF]   running DALI for {label}")
        dali_output = run_dali(
            model=forecast["model"],
            theta0=forecast["theta0"],
            covariance=forecast["cov"],
            dali_config=config.get("dali", {}),
            parameter_names=names,
            config=config,
        )

        if dali_output is not None:
            dali_kit, dali_results = dali_output

            dali_result = dali_results["dali"]
            dali_with_priors = dali_results["dali_with_priors"]

            save_forecast_result(run_output, "dali", dali_result)

            if dali_with_priors is not None:
                save_forecast_result(
                    run_output,
                    "dali_with_priors",
                    dali_with_priors,
                )

            if RUN_DALI_SAMPLING:
                dali_samples = sample_dali(
                    config=config,
                    kit=dali_kit,
                    dali=dali_result,
                    names=names,
                    labels=parameter_labels,
                    label="DALI",
                    include_config_priors=False,
                )

                if dali_with_priors is not None:
                    dali_samples_with_priors = sample_dali(
                        config=config,
                        kit=dali_kit,
                        dali=dali_with_priors,
                        names=names,
                        labels=parameter_labels,
                        label="DALI + priors",
                        include_config_priors=False,
                    )
                else:
                    dali_samples_with_priors = sample_dali(
                        config=config,
                        kit=dali_kit,
                        dali=dali_result,
                        names=names,
                        labels=parameter_labels,
                        label="DALI + priors",
                        include_config_priors=True,
                    )

                plot_dali_results(
                    run_output=run_output,
                    names=names,
                    dali_samples=dali_samples,
                    dali_samples_with_priors=dali_samples_with_priors,
                )

    summary = {
        "scenario": scenario_name,
        "label": label,
        "case_dir": case_dir,
        "array_dir": array_dir,
        "fisher_saved": fisher_result is not None,
        "dali_saved": dali_result is not None,
        "dali_samples_saved": (dali_samples is not None or dali_samples_with_priors is not None),
    }

    return summary


def save_summary(rows):
    """Save a small summary table for the tomography Fisher/DALI postprocess."""
    summary_dir = OUTPUT_DIR / "arrays"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = summary_dir / "tomography_forecast_postprocess_summary.tsv"

    header = [
        "scenario",
        "label",
        "fisher_saved",
        "dali_saved",
        "dali_samples_saved",
        "case_dir",
    ]

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")

        for row in rows:
            values = [
                row["scenario"],
                row["label"],
                str(row["fisher_saved"]),
                str(row["dali_saved"]),
                str(row["dali_samples_saved"]),
                str(row["case_dir"]),
            ]
            f.write("\t".join(values) + "\n")

    return summary_path


def main(path=BASE_CONFIG_PATH):
    """Run Fisher and optional DALI for saved tomography covariance runs."""
    base_config = load_config(path)

    rows = []

    print(f"[DSF] Reading saved tomography runs from {OUTPUT_DIR}")
    print(f"[DSF] Base config: {path}")

    for scenario_name in RUN_SCENARIOS:
        print(f"\n[DSF] Processing tomography scenario: {scenario_name}")

        row = run_one_case(
            base_config=base_config,
            scenario_name=scenario_name,
        )

        if row is not None:
            rows.append(row)

    summary_path = save_summary(rows)

    print("\n[DSF] Finished tomography Fisher/DALI postprocess")
    print(f"[DSF] Processed {len(rows)} scenarios")
    print(f"[DSF] Summary written to {summary_path}")

    return rows


if __name__ == "__main__":
    main()
