"""Generate baseline results for the galaxy-matter lensing window."""

import numpy as np
import pyccl as ccl

from dsf.covariance.ingredients.geometry import gm_lensing_window


def main():
    """Generate and save reference lensing-window calculations."""
    cosmo = ccl.CosmologyVanillaLCDM()

    z_lens = np.linspace(0.2, 0.8, 100)
    nz_lens = np.exp(-0.5 * ((z_lens - 0.5) / 0.1) ** 2)
    nz_lens /= np.trapezoid(nz_lens, z_lens)

    z_source = np.linspace(0.0, 3.0, 200)
    nz_source = np.exp(-0.5 * ((z_source - 1.2) / 0.3) ** 2)
    nz_source /= np.trapezoid(nz_source, z_source)

    pi = np.linspace(-500.0, 500.0, 101)

    pi_result, window = gm_lensing_window(
        cosmo,
        z_lens=z_lens,
        nz_lens=nz_lens,
        z_source=z_source,
        nz_source=nz_source,
        pi=pi,
    )

    np.savez(
        "gm_window_baseline.npz",
        pi=pi_result,
        window=window,
    )

    print("Baseline saved to gm_window_baseline.npz")
    print(f"Window minimum: {window.min():.12e}")
    print(f"Window maximum: {window.max():.12e}")
    print(f"Window at Pi=0: {window[len(window) // 2]:.12e}")


if __name__ == "__main__":
    main()