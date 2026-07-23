import matplotlib.pyplot as plt
import numpy as np

from dsf.covariance.cov_builder import DeltaSigmaCovarianceBuilder
from dsf.modelling import make_ccl_cosmology
from dsf.tomography.tomo_builder import TomographyBuilder
import pyccl as ccl
import pytest

# The fractional difference allowed between the legacy and dsf implementation.
diff_tol = 10**(-2) 

# Helper function:

def is_symmetric(a, tol = 10**(-14)):
    return np.all(np.abs(a - a.T) <= tol)

def is_pos_semi_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)


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

# This set up gives the highest-z source bin with the single lens bin
bin_pairs = tomo_inputs["bin_pairs"]

# Load the bin edges from the legacy code (Mpc/h)
rp_bin_edges = np.loadtxt('./rp_bin_edges.dat')

# Make a CCL cosmology object with the parameters at which we have run the legacy code.
h=0.69
OmB = 0.022/h**2
params = {'OmB':OmB, 'h':h, 'n_s':0.965, 'A_s':2.115 * 10**(-9),'b':2.333, 'OmM': 0.292} 
cosmo = ccl.Cosmology(Omega_c = params['OmM'] - params['OmB'], Omega_b = params['OmB'], h = params['h'], 
                      A_s=params['A_s'], n_s = params['n_s'], transfer_function="boltzmann_camb",
                      matter_power_spectrum="halofit")

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
    k=np.geomspace(10**(-4), 30.0, 5000),
    nonlinear=True,
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
    taper=True,
)

dsf_cov_dict = covariance_builder.covariance_for_pair(lens_bin_index=0, 
                                                      source_bin_index=4)


# Get the dsf covariance
cov_gm_gm_dsf = dsf_cov_dict['cov_gm_gm']
cov_gg_gg_dsf = dsf_cov_dict['cov_gg_gg']
cov_gm_gg_dsf = dsf_cov_dict['cov_gm_gg']
cov_joint_dsf = dsf_cov_dict['cov_joint']

# Get the diagonal errors.
errors_gm_gm_dsf = covariance_builder.diagonal_error(cov_gm_gm_dsf)
errors_gg_gg_dsf = covariance_builder.diagonal_error(cov_gg_gg_dsf)
errors_gm_gg_dsf = covariance_builder.diagonal_error(cov_gm_gg_dsf)
errors_joint_dsf = covariance_builder.diagonal_error(cov_joint_dsf)

# Start checks
Fail = False # Set this to True if anything is failed.

#### Do basic matrix sanity checks

# Check matrices do not contains NaNs or infs
def test_finite():
   assert np.all(np.isfinite(cov_gm_gm_dsf)) 
   assert np.all(np.isfinite(cov_gg_gg_dsf))
   assert np.all(np.isfinite(cov_gm_gg_dsf))
   assert np.all(np.isfinite(cov_joint_dsf))

# Make sure the covariances are symmetric
# Note we don't use diff_tol here because this
# is a basic numerical check.

def test_symmetric():
   assert is_symmetric(cov_gm_gm_dsf, tol=10**(-12))
   assert is_symmetric(cov_gg_gg_dsf, tol=10**(-12))
   assert is_symmetric(cov_gm_gg_dsf, tol=10**(-12))
   assert is_symmetric(cov_joint_dsf, tol=10**(-12))

# Make sure the covariances are positive semi-definite.

def test_pos_semi_def():
   assert is_pos_semi_def(cov_gm_gm_dsf)
   assert is_pos_semi_def(cov_gg_gg_dsf)
   assert is_pos_semi_def(cov_gm_gg_dsf)
   assert is_pos_semi_def(cov_joint_dsf)

### Now compare some ingredients:

shot_noise_dsf = dsf_cov_dict['ingredients']['shot_noise'] # (h / Mpc)^3
shape_noise_dsf = dsf_cov_dict['ingredients']['shape_noise'] # (h / Mpc)^2

shot_noise_leg = np.loadtxt('./shot_noise_LSSTY1Bin5_DESILRG_legacy.dat')
shape_noise_leg = np.loadtxt('./shape_noise_LSSTY1Bin5_DESILRG_legacy.dat')

def test_shot_shape_noise():
   assert np.abs((shot_noise_leg - shot_noise_dsf)/shot_noise_leg)<diff_tol
   assert np.abs((shape_noise_leg - shape_noise_dsf)/shape_noise_leg)<diff_tol

### Now directly compare the legacy and dsf covariance matrices

# We know that there is a slightly different approximatin used
# in calculating the volume associated with the lens sample
# in the legacy code vs DSF, so we need to correct for that.

vol_dsf = dsf_cov_dict['ingredients']['volume'] # (Mpc/h)^3
vol_leg = np.loadtxt('./vol_LSSTY1Bin5_DESILRG_legacy.dat') # (Mpc/h)^3

leg_vol_to_dsf =  vol_leg / vol_dsf # we will multiply legacy covariance by this

# Load legacy code covariances and correct for volume factor
cov_gm_gm_leg = leg_vol_to_dsf * np.loadtxt('./cov_gmgm_LSSTY1Bin5_DESILRG_legacy.dat')
cov_gg_gg_leg = leg_vol_to_dsf * np.loadtxt('./cov_gggg_LSSTY1Bin5_DESILRG_legacy.dat')
cov_gm_gg_leg = leg_vol_to_dsf * np.loadtxt('./cov_gmgg_LSSTY1Bin5_DESILRG_legacy.dat')
cov_joint_leg = leg_vol_to_dsf * np.loadtxt('./cov_joint_LSSTY1Bin5_DESILRG_legacy.dat')

# Get diagonal errors of the legacy code:
errors_gm_gm_leg = np.sqrt(np.diag(cov_gm_gm_leg))
errors_gg_gg_leg = np.sqrt(np.diag(cov_gg_gg_leg))
errors_gm_gg_leg = np.sqrt(np.diag(cov_gm_gg_leg))
errors_joint_leg = np.sqrt(np.diag(cov_joint_leg))

# Check shapes match
def test_array_shapes():
   assert cov_gm_gm_dsf.shape == cov_gm_gm_leg.shape
   assert cov_gm_gg_dsf.shape == cov_gm_gg_leg.shape
   assert cov_gg_gg_dsf.shape == cov_gg_gg_leg.shape
   assert cov_joint_dsf.shape == cov_joint_leg.shape

# Check diagonal errors
fracdiff_gmgm_error = np.abs((errors_gm_gm_dsf - errors_gm_gm_leg) / errors_gm_gm_leg)
fracdiff_gggg_error = np.abs((errors_gg_gg_dsf - errors_gg_gg_leg) / errors_gg_gg_leg)
fracdiff_gmgg_error = np.abs((errors_gm_gg_dsf - errors_gm_gg_leg) / errors_gm_gg_leg)
fracdiff_joint_error = np.abs((errors_joint_dsf - errors_joint_leg) / errors_joint_leg)

def test_diag_errors_agree():
   assert np.all(fracdiff_gmgm_error<diff_tol)
   assert np.all(fracdiff_gggg_error<diff_tol)
   assert np.all(fracdiff_gmgg_error<diff_tol)
   assert np.all(fracdiff_joint_error<diff_tol)
