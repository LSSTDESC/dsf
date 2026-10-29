import argparse
import numpy as np
import pyccl as ccl

from dsf.covariance.ingredients.geometry import gm_lensing_window

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
args = parser.parse_args()

cosmo = ccl.CosmologyVanillaLCDM()

z_lens = np.linspace(0.2, 0.8, 100)
nz_lens = np.exp(-0.5 * ((z_lens - 0.5) / 0.1) ** 2)
nz_lens /= np.trapezoid(nz_lens, z_lens)

z_source = np.linspace(0.0, 3.0, 200)
nz_source = np.exp(-0.5 * ((z_source - 1.2) / 0.3) ** 2)
nz_source /= np.trapezoid(nz_source, z_source)

pi = np.linspace(-500.0, 500.0, 101)

resolutions = [1000, 5000, 10000, 20000, 50000]

results = {}

for n_z_interp in resolutions:
    _, window = gm_lensing_window(
        cosmo,
        z_lens=z_lens,
        nz_lens=nz_lens,
        z_source=z_source,
        nz_source=nz_source,
        pi=pi,
        n_z_interp=n_z_interp,
    )

    results[str(n_z_interp)] = window

np.savez(args.output, pi=pi, **results)

print(f"Saved {args.output}")
