"""Benchmark covariance calculations with custom galaxy power spectra."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl

from dsf.covariance.cov_builder import DeltaSigmaCovarianceBuilder
from dsf.tomography.tomo_builder import TomographyBuilder


__all__ = ["main"]

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR / "figures" / "custom_power_spectra"

LENS_BIN = 0
SOURCE_BIN = 4
GALAXY_BIAS = 2.333


def build_cosmology():
    """Construct the cosmology used in the legacy covariance benchmark."""
    h = 0.69
    omega_b = 0.022 / h**2

    return ccl.Cosmology(
        Omega_c=0.292 - omega_b,
        Omega_b=omega_b,
        h=h,
        A_s=2.115e-9,
        n_s=0.965,
        transfer_function="boltzmann_camb",
        matter_power_spectrum="halofit",
    )


def build_tomography():
    """Construct the DESI lens and LSST source tomography."""
    tomography = TomographyBuilder(
        lens_survey="desi",
        source_survey="lsst",
        lens_sample="lrg",
        source_sample=None,
        lens_year=None,
        source_year="1",
        lens_role="lens",
        source_role="source",
        overlap_threshold=0.1,
        source_behind_lens=True,
        shared_overrides={"bins": {"count": 5}},
    )

    return tomography.prepare_bins()


def build_hankel_config():
    """Return the Hankel configuration used in the legacy benchmark."""
    return {
        "r_min": 0.6,
        "r_max": 110,
        "k_min": 1e-4,
        "k_max": 30.0,
        "orders": {"gm": 2, "gg": 2, "cross": 2},
        "n_zeros": 490000,
        "n_zeros_step": 1000,
        "prune_r": 0,
        "verbose": False,
        "max_iterations": 1000,
        "radial_weight_method": "gradient",
        "taper": True,
    }


def modified_power_spectra(cosmo, bias):
    """Construct controlled scale-dependent galaxy power spectra.

    The modification enhances galaxy clustering at high wavenumbers.
    These spectra are illustrative and are not physical HOD predictions.
    """
    pk_mm = cosmo.get_nonlin_power()

    a_arr, lk_arr, pk_arr = pk_mm.get_spline_arrays()
    k = np.exp(lk_arr)

    enhancement = 1.0 + 0.2 * (k / 0.5)**2 / (1.0 + (k / 0.5)**2)

    pk_gg = ccl.Pk2D(
        a_arr=a_arr,
        lk_arr=lk_arr,
        pk_arr=bias**2 * pk_arr * enhancement[None, :],
        is_logp=False,
    )

    pk_gm = ccl.Pk2D(
        a_arr=a_arr,
        lk_arr=lk_arr,
        pk_arr=bias * pk_arr * np.sqrt(enhancement)[None, :],
        is_logp=False,
    )

    return pk_gg, pk_gm


def compute_covariance(builder, label):
    """Compute and time the covariance for one tomographic bin pair."""
    start = time.perf_counter()
    covariance = builder.covariance_for_pair(LENS_BIN, SOURCE_BIN)
    elapsed = time.perf_counter() - start

    print(f"{label}: {elapsed:.3f} s")

    return covariance


def compare_covariances(reference, equivalent, modified):
    """Compare covariance blocks for default and custom spectra."""
    blocks = ("cov_gm_gm", "cov_gg_gg", "cov_gm_gg", "cov_joint")

    print("\nCovariance comparison")
    print("-" * 65)

    for block in blocks:
        baseline = reference[block]
        custom = equivalent[block]
        changed = modified[block]

        np.testing.assert_allclose(custom, baseline, rtol=1e-6, atol=0.0)

        difference = np.linalg.norm(changed - baseline)
        relative_change = difference / np.linalg.norm(baseline)

        print(f"\n{block}")
        print("  Equivalent custom spectra: PASS")
        print(f"  Relative Frobenius difference: {relative_change:.3e}")

        if block != "cov_gm_gg":
            baseline_errors = np.sqrt(np.diag(baseline))
            changed_errors = np.sqrt(np.diag(changed))

            fractional_change = changed_errors / baseline_errors - 1.0

            print(
                "  Maximum fractional change in errors: "
                f"{np.max(np.abs(fractional_change)):.3e}"
            )


def correlation_matrix(covariance):
    """Convert a covariance matrix to a correlation matrix."""
    errors = np.sqrt(np.diag(covariance))
    return covariance / np.outer(errors, errors)


def plot_diagonal_errors(reference, modified):
    """Plot the fractional change in diagonal uncertainties."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for block, label in (
        ("cov_gm_gm", "gm-gm"),
        ("cov_gg_gg", "gg-gg"),
    ):
        baseline = np.sqrt(np.diag(reference[block]))
        changed = np.sqrt(np.diag(modified[block]))

        fractional_change = changed / baseline - 1.0

        ax.plot(
            np.arange(len(fractional_change)),
            100.0 * fractional_change,
            marker="o",
            label=label,
        )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Radial bin index")
    ax.set_ylabel("Change in diagonal uncertainty [%]")
    ax.set_title("Effect of custom galaxy power spectra")
    ax.legend()
    fig.tight_layout()

    fig.savefig(OUTPUT_DIR / "diagonal_errors.png", dpi=200)
    plt.close(fig)


def plot_correlation_matrices(reference, modified):
    """Plot baseline and modified joint correlation matrices."""
    baseline = correlation_matrix(reference["cov_joint"])
    changed = correlation_matrix(modified["cov_joint"])

    difference = changed - baseline

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    panels = (
        (baseline, "Default"),
        (changed, "Modified custom"),
        (difference, "Difference"),
    )

    for ax, (matrix, title) in zip(axes, panels):
        if title == "Difference":
            vmax = np.max(np.abs(matrix))
            vmax = max(vmax, 1e-12)
            vmin = -vmax
        else:
            vmin, vmax = -1.0, 1.0

        im = ax.imshow(
            matrix,
            origin="lower",
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(title)
        ax.set_xlabel("Radial bin index")
        ax.set_ylabel("Radial bin index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_matrices.png", dpi=200)
    plt.close(fig)


def main():
    """Run the custom galaxy power-spectrum covariance benchmark."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cosmo = build_cosmology()
    tomography = build_tomography()

    builder_inputs = {
        "cosmo": cosmo,
        "tomography": tomography,
        "rp_bin_edges": np.loadtxt(DATA_DIR / "rp_bin_edges.dat"),
        "area_deg2": 5000.0,
        "sigma_e": 0.26,
        "k": np.geomspace(8e-5, 30.0, 5000),
        "hankel": build_hankel_config(),
    }

    pk_mm = cosmo.get_nonlin_power()

    default_builder = DeltaSigmaCovarianceBuilder(
        **builder_inputs,
        galaxy_bias=GALAXY_BIAS,
    )

    equivalent_builder = DeltaSigmaCovarianceBuilder(
        **builder_inputs,
        pk_gg=GALAXY_BIAS**2 * pk_mm,
        pk_gm=GALAXY_BIAS * pk_mm,
    )

    pk_gg_modified, pk_gm_modified = modified_power_spectra(
        cosmo, GALAXY_BIAS
    )

    modified_builder = DeltaSigmaCovarianceBuilder(
        **builder_inputs,
        pk_gg=pk_gg_modified,
        pk_gm=pk_gm_modified,
    )

    print("\nComputing covariance matrices")
    print("-" * 65)

    reference = compute_covariance(default_builder, "Default")
    equivalent = compute_covariance(equivalent_builder, "Equivalent custom")
    modified = compute_covariance(modified_builder, "Modified custom")

    compare_covariances(reference, equivalent, modified)

    plot_diagonal_errors(reference, modified)
    plot_correlation_matrices(reference, modified)

    print(f"\nFigures saved to: {OUTPUT_DIR}")
    print("All custom power-spectrum benchmark checks passed.")


if __name__ == "__main__":
    main()
