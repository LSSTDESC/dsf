"""Benchmark numerical accuracy and performance of lens-averaged power spectra."""

from time import perf_counter

import numpy as np
import pyccl as ccl

from dsf.covariance.ingredients.power_spectrum import lens_averaged_matter_power
from dsf.tomography.tomo_builder import TomographyBuilder
from dsf.utils.converters import redshift_to_scale_factor
from dsf.utils.integrators import weighted_trapezoid_average


N_REPEATS = 5
RTOL = 1e-12
ATOL = 1e-12


def original_lens_averaged_matter_power(cosmo, k, z_lens, nz_lens):
    """Evaluate the original lens-averaged matter power spectrum."""
    h = cosmo["h"]

    pk_of_z = np.asarray(
        [
            ccl.nonlin_matter_power(
                cosmo,
                k * h,
                redshift_to_scale_factor(z_value),
            )
            * h**3
            for z_value in z_lens
        ]
    )

    return np.asarray(
        weighted_trapezoid_average(
            z_lens,
            pk_of_z,
            nz_lens,
            axis=0,
            normalize_weights=False,
        ),
        dtype=float,
    )


def benchmark(function, cosmo, k, z_lens, nz_lens):
    """Measure the median runtime of a power-spectrum implementation."""
    times = []
    result = None

    for _ in range(N_REPEATS):
        start = perf_counter()

        result = function(cosmo, k, z_lens, nz_lens)

        times.append(perf_counter() - start)

    return result, float(np.median(times))


def main():
    """Compare the original and optimized implementations for DESI LRG bins."""
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
        shared_overrides={
            "bins": {
                "count": 5,
            },
        },
    )

    tomo_inputs = tomography.prepare_bins()
    lens_result = tomo_inputs["lens_result"]

    z_lens = np.asarray(lens_result.z, dtype=float)
    k = np.geomspace(8e-5, 30.0, 5000)

    h = 0.69
    omega_b = 0.022 / h**2

    cosmo = ccl.Cosmology(
        Omega_c=0.292 - omega_b,
        Omega_b=omega_b,
        h=h,
        A_s=2.115e-9,
        n_s=0.965,
        transfer_function="boltzmann_camb",
        matter_power_spectrum="halofit",
    )

    print("\nLens-averaged matter power spectrum benchmark")
    print("============================================")
    print(f"Redshift points: {len(z_lens)}")
    print(f"Wavenumber points: {len(k)}")
    print(f"Repeats: {N_REPEATS}")
    print(f"Relative tolerance: {RTOL:.1e}")

    original_times = []
    optimized_times = []

    for lens_bin in sorted(lens_result.bins):
        nz_lens = np.asarray(lens_result.bins[lens_bin], dtype=float)

        n_active = np.count_nonzero(nz_lens)
        n_total = len(z_lens)

        original, original_time = benchmark(
            original_lens_averaged_matter_power,
            cosmo,
            k,
            z_lens,
            nz_lens,
        )

        optimized, optimized_time = benchmark(
            lens_averaged_matter_power,
            cosmo,
            k,
            z_lens,
            nz_lens,
        )

        np.testing.assert_allclose(
            optimized,
            original,
            rtol=RTOL,
            atol=ATOL,
        )

        abs_diff = np.max(np.abs(optimized - original))

        relative_diff = np.max(
            np.abs(optimized - original)
            / np.maximum(np.abs(original), 1e-30)
        )

        speedup = original_time / optimized_time

        original_times.append(original_time)
        optimized_times.append(optimized_time)

        print(f"\nLens bin {lens_bin}")
        print("--------------------")
        print(f"Active redshift points: {n_active}/{n_total}")
        print(f"Original runtime:       {original_time:.4f} s")
        print(f"Optimized runtime:      {optimized_time:.4f} s")
        print(f"Speedup:                {speedup:.2f}x")
        print(f"Maximum absolute diff:  {abs_diff:.6e}")
        print(f"Maximum relative diff:  {relative_diff:.6e}")
        print("Numerical regression:   PASS")

    total_original = sum(original_times)
    total_optimized = sum(optimized_times)

    print("\nOverall performance")
    print("===================")
    print(f"Original total:    {total_original:.4f} s")
    print(f"Optimized total:   {total_optimized:.4f} s")
    print(f"Overall speedup:   {total_original / total_optimized:.2f}x")
    print(
        f"Runtime reduction: "
        f"{(1.0 - total_optimized / total_original) * 100:.1f}%"
    )
    print("All numerical regressions: PASS")


if __name__ == "__main__":
    main()
