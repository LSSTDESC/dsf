"""Benchmark standalone Delta Sigma against the legacy CCL projection."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl

from dsf.data_vector.ccl_integration import DeltaSigma


__all__ = ["main"]

OUTPUT_DIR = Path(__file__).resolve().parent / "figures" / "delta_sigma"

SCALE_FACTORS = (0.5, 0.7, 1.0)
RADII = np.geomspace(0.6, 110.0, 50)

RTOL = 1e-3


class LegacyHaloProfile(ccl.halos.profiles.profile_base.HaloProfile):
    """Reproduce the original CCL-based Delta Sigma projection."""

    def __init__(self, *, pk2d: ccl.Pk2D) -> None:
        """Initialize the legacy profile with its original FFTLog settings."""
        self.pk2d = pk2d

        self.fourier_analytic = True
        self._fourier = self._fourier_analytic
        self.projected_analytic = False
        self.cumul2d_analytic = False

        super().__init__(
            mass_def="200m",
            concentration=ccl.halos.concentration.ConcentrationConstant(
                5.0, mass_def="200m"
            ),
        )

        self.update_precision_fftlog(
            padding_lo_fftlog=1e-6,
            padding_hi_fftlog=1e6,
            n_per_decade=5000,
        )

    def _fourier_analytic(self, cosmo, k, M, a):
        """Evaluate the Fourier-space profile using the legacy CCL interface."""
        if np.size(M) != 1:
            raise ValueError("M must be a single value.")

        k_arr = np.atleast_1d(np.asarray(k, dtype=float))
        values = np.asarray(self.pk2d(k_arr, a), dtype=float)

        return values.reshape((1, k_arr.size))


def build_cosmology():
    """Construct the cosmology used in the covariance benchmarks."""
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


def legacy_delta_sigma(profile, cosmo, r, a):
    """Evaluate Delta Sigma using the original CCL implementation."""
    sigma = profile.projected(cosmo, r, 1.0, a)
    mean_sigma = profile.cumul2d(cosmo, r, 1.0, a)

    return np.asarray(mean_sigma - sigma, dtype=float) / 1e12


def benchmark_projection(legacy, standalone, cosmo, r, a):
    """Compare numerical accuracy and execution time at one scale factor."""
    start = time.perf_counter()
    reference = legacy_delta_sigma(legacy, cosmo, r, a)
    legacy_time = time.perf_counter() - start

    start = time.perf_counter()
    result = standalone.delta_sigma(cosmo=cosmo, r=r, a=a) / 1e12
    standalone_time = time.perf_counter() - start

    np.testing.assert_allclose(result, reference, rtol=RTOL, atol=0.0)

    fractional_difference = (result - reference) / reference

    print(f"\na = {a:.2f}")
    print(f"  Legacy CCL:       {legacy_time:.3f} s")
    print(f"  Standalone FFTLog: {standalone_time:.3f} s")
    print(
        "  Maximum fractional difference: "
        f"{np.max(np.abs(fractional_difference)):.3e}"
    )

    return reference, result, fractional_difference


def plot_comparison(results):
    """Plot Delta Sigma predictions and their fractional differences."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    for a, (reference, result, fractional_difference) in results.items():
        axes[0].plot(RADII, reference, label=f"Legacy, a={a:.2f}")
        axes[0].plot(RADII, result, linestyle="--", label=f"Standalone, a={a:.2f}")

        axes[1].plot(RADII, fractional_difference, label=f"a={a:.2f}")

    axes[0].set_xscale("log")
    axes[0].set_yscale("symlog")
    axes[0].set_ylabel(r"$\Delta\Sigma\ [M_\odot/\mathrm{pc}^2]$")
    axes[0].legend(fontsize=8)

    axes[1].axhline(0.0, color="gray", linestyle="--")
    axes[1].axhline(RTOL, color="gray", linestyle=":")
    axes[1].axhline(-RTOL, color="gray", linestyle=":")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Comoving projected radius [Mpc]")
    axes[1].set_ylabel("Fractional difference")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "delta_sigma_comparison.png", dpi=200)
    plt.close(fig)


def main():
    """Run the legacy-versus-standalone Delta Sigma benchmark."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cosmo = build_cosmology()
    pk2d = cosmo.get_nonlin_power()

    legacy = LegacyHaloProfile(pk2d=pk2d)
    standalone = DeltaSigma(pk2d=pk2d)

    results = {}

    for a in SCALE_FACTORS:
        results[a] = benchmark_projection(
            legacy, standalone, cosmo, RADII, a
        )

    plot_comparison(results)

    print(f"\nFigures saved to: {OUTPUT_DIR}")
    print("All Delta Sigma benchmark checks passed.")


if __name__ == "__main__":
    main()
