"""Numerical regression tests for the optimized galaxy-matter lensing window."""

from pathlib import Path

import numpy as np
import pyccl as ccl

from dsf.covariance.ingredients.geometry import gm_lensing_window


DATA_DIR = Path(__file__).resolve().parent


def test_gm_lensing_window_against_original():
    """Tests that the optimized window agrees with the original implementation."""
    reference = np.load(DATA_DIR / "gm_window_baseline.npz")

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
        n_z_interp=10000,
    )

    np.testing.assert_array_equal(pi_result, reference["pi"])
    np.testing.assert_allclose(
        window,
        reference["window"],
        rtol=1e-7,
        atol=1e-12,
    )