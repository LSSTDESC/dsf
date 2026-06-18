# Covariance benchmarks

Benchmarks for the DSF DeltaSigma covariance calculation.

These scripts compare DSF covariance predictions against legacy or reference implementations.

## Compare to legacy code

### One lens / source bin pair

We compare the covariance output from the legacy code from Dani Leonard et al. with the dsf covariance builder, using a single bin for DESI LRGs and the highest-z bin of sources in an LSST Year 1 set up with 5 equipopulated photometric bins. 

``` generate_dNdz_for_legacy.py``` generates the redshift distributions that are fed into the legacy code to produce the covariance.

``` dNdz_source_LSSTY1Bin5.dat``` contains (z, dNdz) for the sources

```dNdz_lens_DESI_LRG_1bin.dat``` contains (z, dNdz) for the lenses

```rp_bin_edges.dat``` contains the edges of the projected radial bins, in Mpc/h.

```rp_bin_centres.dat``` contains the centres of the projected radial bins, in Mpc/h.

``` cov_gmgm_LSSTY1Bin5_DESILRG.dat``` contains the covariance of Delta Sigma from the legacy code, using projected radial bins defined by rp_bin_edges/centres, in units ```Msun^2 h^2 / pc^4```.
