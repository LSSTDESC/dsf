import numpy as np
import cov_hankel as ch
import pyccl as ccl
import utils as u
import specs as sp

""" This script gets the theoretical covariance matrix for 
a single lens and source bin covariance matrix for 
benchmarking the DP2 GGL Delta Sigma covariance analysis.
"""
__author__ = 'Dani leonard'

# Define the bin edges in Mpc/h
rp_bin_edges = np.logspace(0,2,20)

# Get the bin centres (not trivial because of log spacing)
rp_bin_c = u.rp_bins_mid(rp_bin_edges)

# Define a set of parameters

h=0.69
OmB = 0.022/h**2

# Using nonlinear bias parameters as fit in Kitanis & White 2022.
# They fit LPT parameters so we convert these to their Eulerian equivalents.
b1_LPT = 1.333
b2_LPT = 0.514
bs_LPT = 0 # They fix this to 0.

# Convert to Eulerian using the conversions in Chen, Vlah & White 2020 (these use the same convention as Kitanis & White 2022)
b1 = 1.0 + b1_LPT
b2 = b2_LPT + 8./21.*(b1_LPT)
bs = bs_LPT - 2./7*(b1_LPT)

params = {'mu_0': 0., 'sigma_0':0., 'OmB':OmB, 'h':h, 'n_s':0.965, 'A_s':2.115 * 10**(-9),'b':b1, 'OmM': 0.292, 'b_2':b2, 'b_s': bs} 

# Get and save some ingredients for comparison:
vol = sp.volume(params, 'LSSTY1_dp2-ggl', 'DESI_LRG_dp2-ggl') # (Mpc/h)^3
print('vol=', vol)
np.savetxt('./vol_LSSTY1Bin5_DESILRG_legacy.dat', [vol])
shape_noise = sp.shape_noise(params, 'LSSTY1_dp2-ggl', 'DESI_LRG_dp2-ggl', )
print('shape noise=', shape_noise)
np.savetxt('./shape_noise_LSSTY1Bin5_DESILRG_legacy.dat', [shape_noise])
shot_noise = sp.shot_noise('DESI_LRG_dp2-ggl')
print('shot_noise=', shot_noise)
np.savetxt('./shot_noise_LSSTY1Bin5_DESILRG_legacy.dat', [shot_noise])

(cov_gmgm, cov_gggg, cov_gmgg) = ch.get_DeltaSigma_covs(params, rp_bin_edges, 
                                                        rp_bin_c, 'DESI_LRG_dp2-ggl', 'LSSTY1_dp2-ggl',  'LSSTY1Bin5_DESILRG_legacy')

# Save the bin edges:
np.savetxt('./rp_bin_edges.dat', rp_bin_edges)

# Save the bin centres:
np.savetxt('./rp_bin_centres.dat', rp_bin_c)

#Save the gmgm covariance
#np.savetxt('./cov_gmgm_LSSTY1Bin5_DESILRG_legacy.dat', cov_gmgm)
cov_gmgm = np.loadtxt('./cov_gmgm_LSSTY1Bin5_DESILRG_legacy.dat')

#Save the gggg covariance
#np.savetxt('./cov_gggg_LSSTY1Bin5_DESILRG_legacy.dat', cov_gggg)
cov_gggg = np.loadtxt('./cov_gggg_LSSTY1Bin5_DESILRG_legacy.dat')

#Save the gmgg covariance
#np.savetxt('./cov_gmgg_LSSTY1Bin5_DESILRG_legacy.dat', cov_gmgg)
cov_gmgg = np.loadtxt('./cov_gmgg_LSSTY1Bin5_DESILRG_legacy.dat')

# Assemble the full joint covariance and save that too:

joint_cov = np.zeros((2*len(rp_bin_c), 2*len(rp_bin_c)))
joint_cov[0:len(rp_bin_c),0:len(rp_bin_c)] = cov_gmgm
joint_cov[len(rp_bin_c):2*len(rp_bin_c),len(rp_bin_c):2*len(rp_bin_c)] = cov_gggg
joint_cov[0:len(rp_bin_c),len(rp_bin_c):2*len(rp_bin_c)] = cov_gmgg
joint_cov[len(rp_bin_c):2*len(rp_bin_c), 0:len(rp_bin_c)] = cov_gmgg

np.savetxt('./cov_joint_LSSTY1Bin5_DESILRG_legacy.dat', joint_cov)