# Covariance benchmarks

Benchmarks for the DSF DeltaSigma covariance calculation.

These scripts compare DSF covariance predictions against legacy or reference implementations.

## Compare to legacy code

### One lens / source bin pair:

We compare the covariance output from the legacy code from Dani Leonard et al. with the dsf covariance builder, using a single bin for DESI LRGs and the highest-z bin of sources in an LSST Year 1 set up with 5 equipopulated photometric bins. 

To run the benchmark, just run `python benchmark_covariance_scripy.py`. This will do a number of checks (see script for details), output the fractional difference in the diagonal elements between the dsf and legacy implementations, and tell you if the benchmark passed or not, and if not, why.

`generate_dNdz_for_legacy.py` generates the redshift distributions that are fed into the legacy code to produce the covariance.

`dNdz_source_LSSTY1Bin5.dat` contains (z, dNdz) for the sources

`dNdz_lens_DESI_LRG_1bin.dat` contains (z, dNdz) for the lenses

`rp_bin_edges.dat` contains the edges of the projected radial bins, in Mpc/h.

`rp_bin_centres.dat` contains the centres of the projected radial bins, in Mpc/h.

`cov_gmgm_LSSTY1Bin5_DESILRG_legacy.dat` contains the covariance of Delta Sigma gm from the legacy code, using projected radial bins defined by rp_bin_edges/centres, in units `Msun^2 h^2 / pc^4`.

`cov_gggg_LSSTY1Bin5_DESILRG_legacy.dat` contains the covariance of Delta Sigma gg from the legacy code, using projected radial bins defined by rp_bin_edges/centres, in units `Msun^2 h^2 / pc^4`.

`cov_gmgg_LSSTY1Bin5_DESILRG_legacy.dat` contains the cross-covariance of Delta Sigma gm x Delta Sigma gg from the legacy code, using projected radial bins defined by rp_bin_edges/centres, in units `Msun^2 h^2 / pc^4`.

`cov_joint_LSSTY1Bin5_DESILRG_legacy.dat` contains the joint covariance of Delta Sigma gm x gm, gm x gg, and gg x gg from the legacy code, using projected radial bins defined by rp_bin_edges/centres, in units `Msun^2 h^2 / pc^4`.

`vol_LSSTY1Bin5_DESILRG_legacy.dat` contains the volume associated with the lens sample in the legacy implementation, in units `Mpc/h`. Note this volume is different than dsf volume due to a change in approximation choice.

`shape_noise_LSSTY1Bin5_DESILRG_legacy.dat` contains the projected shape noise term in the legacy implementation, in units `h^2 / Mpc^2`.

`shot_noise_LSSTY1Bin5_DESILRG_legacy.dat` contains the projected shot noise term in the legacy implementation, in units `h^3 / Mpc^3`.
