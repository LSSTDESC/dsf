import matplotlib.pyplot as plt
import numpy as np

from dsf.covariance.cov_builder import DeltaSigmaCovarianceBuilder
from dsf.modelling import make_ccl_cosmology
from dsf.tomography.tomo_builder import TomographyBuilder
import pyccl as ccl


cosmo = make_ccl_cosmology(
    transfer_function="boltzmann_camb",
    matter_power_spectrum="halofit",
)

z = np.linspace(0.0, 5.0, 1000)

tomography = TomographyBuilder(
    lens_survey="desi",
    source_survey="lsst",
    lens_sample='lrg',
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

bin_pairs = tomo_inputs["bin_pairs"][:1]
# This set up gives the highest-z source bin with the single lens bin

# Load the bin edges from the legacy code (Mpc/h)
rp_bin_edges = np.loadtxt('./rp_bin_edges.dat')

# Make a CCL cosmology object with the parameters at which we have run the legacy code.
h=0.69
OmB = 0.022/h**2
params = {'OmB':OmB, 'h':h, 'n_s':0.965, 'A_s':2.115 * 10**(-9),'b':2.333, 'OmM': 0.292} 
cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], A_s=params['A_s'], n_s = params['n_s'])

# Load the grid in line-of-sight projection used in the legacy code for the gm x gg case (Mpc/h)
#pi_grid = np.loadtxt('./Pi_grid.dat')

# Specifics here are set to match what was provided for the legacy code.
covariance_builder = DeltaSigmaCovarianceBuilder(
    cosmo=cosmo,
    lens_result=tomo_inputs["lens_result"],
    source_result=tomo_inputs["source_result"],
    lens_population_stats=tomo_inputs["lens_population_stats"],
    source_population_stats=tomo_inputs["source_population_stats"],
    bin_pairs=bin_pairs,
    rp_bin_edges=rp_bin_edges,
    area_deg2=5000.0,
    sigma_e=0.26,
    galaxy_bias=params['b'],
    k=np.geomspace(10**(-4), 3.0, 5000),
    #pi = pi_grid,
    hankel_kwargs={
        "r_min": 0.6,
        "r_max": 110,
        "k_min": 10**(-4),
        "k_max": 30.0,
        "orders": (2,), 
        "n_zeros": 480000, # Starting here after some trial and error.
        "n_zeros_step": 1000,
        "prune_r": 0,
        "verbose": True,
        "max_iterations": 1000, 
    },
    taper=False,
)


dsf_cov_dict = covariance_builder.covariance_for_pair(lens_bin_index=0, 
                                                      source_bin_index=0)

cov_gm_gm_dsf = dsf_cov_dict['cov_gm_gm']
cov_gg_gg_dsf = dsf_cov_dict['cov_gg_gg']
cov_gm_gg_dsf = dsf_cov_dict['cov_gm_gg']
cov_joint_dsf = dsf_cov_dict['cov_joint']

# Save the covariance matrices from dsf
np.savetxt('./cov_gmgm_LSSTY1Bin5_DESILRG_dsf.dat', cov_gm_gm_dsf)
np.savetxt('./cov_gggg_LSSTY1Bin5_DESILRG_dsf.dat', cov_gg_gg_dsf)
np.savetxt('./cov_gmgg_LSSTY1Bin5_DESILRG_dsf.dat', cov_gm_gg_dsf)
np.savetxt('./cov_joint_LSSTY1Bin5_DESILRG_dsf.dat', cov_joint_dsf)

# Get the diagonal errors.
errors_gm_gm = covariance_builder.diagonal_error(cov_gm_gm_dsf)
errors_gg_gg = covariance_builder.diagonal_error(cov_gg_gg_dsf)
errors_gm_gg = covariance_builder.diagonal_error(cov_gm_gg_dsf)
errors_joint = covariance_builder.diagonal_error(cov_joint_dsf)

corr = covariance_builder.correlation_matrix(cov_gm_gm_dsf)

# Compare with legacy code:

# Load the legacy covariances:
cov_gm_gm_leg = np.loadtxt('./cov_gmgm_LSSTY1Bin5_DESILRG_legacy.dat')
cov_gg_gg_leg = np.loadtxt('./cov_gggg_LSSTY1Bin5_DESILRG_legacy.dat')
cov_gm_gg_leg = np.loadtxt('./cov_gmgg_LSSTY1Bin5_DESILRG_legacy.dat')
cov_joint_leg = np.loadtxt('./cov_joint_LSSTY1Bin5_DESILRG_legacy.dat')

print("covariance shape: dsf gmgm:", cov_gm_gm_dsf.shape, ", legacy gmgm:", cov_gm_gm_leg.shape)
print("covariance shape: dsf gggg:", cov_gg_gg_dsf.shape, ", legacy gggg:", cov_gg_gg_leg.shape)
print("covariance shape: dsf gmgg:", cov_gm_gg_dsf.shape, ", legacy gmgg:", cov_gm_gg_leg.shape)
print("covariance shape: dsf joint:", cov_joint_dsf.shape, ", legacy joint:", cov_joint_leg.shape)

"""fig, ax = plt.subplots(figsize=(5.4, 4.6))

image = ax.imshow(
    corr,
    vmin=-1.0,
    vmax=1.0,
    origin="lower",
)

ax.set_title("gm covariance correlation matrix", fontsize=15)
ax.set_xlabel("data-vector index", fontsize=14)
ax.set_ylabel("data-vector index", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=12)

cbar = fig.colorbar(image, ax=ax)
cbar.set_label("correlation", fontsize=13)
cbar.ax.tick_params(labelsize=11)

fig.subplots_adjust(left=0.16, right=0.92, bottom=0.14, top=0.90)

plt.savefig('./corr_test.pdf')"""