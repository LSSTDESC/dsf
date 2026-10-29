"""Benchmark the galaxy-matter lensing window."""

import time

import numpy as np
import pyccl as ccl

from dsf.covariance.ingredients.geometry import gm_lensing_window


def main():
    """Benchmark repeated galaxy-matter lensing-window calculations."""
    cosmo = ccl.CosmologyVanillaLCDM()

    z_lens = np.linspace(0.2, 0.8, 100)
    nz_lens = np.exp(-0.5 * ((z_lens - 0.5) / 0.1) ** 2)
    nz_lens /= np.trapezoid(nz_lens, z_lens)

    z_source = np.linspace(0.0, 3.0, 200)
    nz_source = np.exp(-0.5 * ((z_source - 1.2) / 0.3) ** 2)
    nz_source /= np.trapezoid(nz_source, z_source)

    pi = np.linspace(-500.0, 500.0, 101)

    def compute():
        return gm_lensing_window(
            cosmo,
            z_lens=z_lens,
            nz_lens=nz_lens,
            z_source=z_source,
            nz_source=nz_source,
            pi=pi,
        )

    # Warm up CCL and any internal caches.
    compute()

    times = []
    for _ in range(10):
        start = time.perf_counter()
        compute()
        times.append(time.perf_counter() - start)

    print(f"Median runtime: {np.median(times):.6f} s")
    print(f"Minimum runtime: {np.min(times):.6f} s")


if __name__ == "__main__":
    main()
