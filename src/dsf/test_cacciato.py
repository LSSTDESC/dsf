import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np

import pyccl as ccl
from dsf.data_vector.delta_sigma_builder import DeltaSigmaCalculator
from dsf.pk2d_cacciato_hod import *
from hod_cacciato import *
import matplotlib as mpl
mpl.rcParams["text.usetex"] = "True"


import sys
sys.path.insert(0, "/hpc/group/cosmology/nlc38/from_NERSC/dp2_proj/sham_proj/CCLX/my_work/cacciato_hod")
from cacciato_inputs import cosmo_pars, cacciato_med_pars, sampleinfo, mag_bins, magnitude_to_luminosity

log_L1 = np.log10(magnitude_to_luminosity(-20)) #log10( L1 / [Lsun/h^2] )
log_L2 = np.log10(magnitude_to_luminosity(-21)) #log10( L2 / [Lsun/h^2] )
hodparams = dict(log_L1=log_L1, log_L2=log_L2, hval=cosmo_pars['h'], **cacciato_med_pars)

print(cosmo_pars)
cosmo = ccl.Cosmology(**cosmo_pars)

r = np.geomspace(0.1, 10.0, 20)

z_lens = 0.1
a_lens = 1.0 / (1.0 + z_lens)

k_array = np.geomspace(1.0e-3, 30.0, 64)
a_array = np.linspace(0.3, 1.0, 16)


def pk2d_func(*, cosmo):
    return pk2d_cacciato_hod(
        cosmo,
        k_array=k_array,
        a_array=a_array,
        **hodparams
    )


calculator = DeltaSigmaCalculator(pk2d_func=pk2d_func)

delta_sigma = calculator.delta_sigma(
    r=r,
    a=a_lens,
    cosmo=cosmo,
)

print(np.c_[r,delta_sigma])

z = np.linspace(0.05, 0.15, 9)
n_z = np.exp(-0.5 * ((z - z_lens) / 0.05) ** 2)

cosmo = make_ccl_cosmology(
    transfer_function="eisenstein_hu",
    matter_power_spectrum="halofit",
)

delta_sigma_bin = calculator.delta_sigma_lens_bin(
    r=r,
    lens_dndz=(z, n_z),
    cosmo=cosmo,
    z_min=0.05,
    z_max=0.15,
)

ratio = delta_sigma_bin / delta_sigma

### Now plot the data point from Mandelbaum 2006
import pandas as pd
dfc = pd.read_csv("esd_data_points_mid.csv")
dfl = pd.read_csv("esd_data_points_lower.csv")
dfh = pd.read_csv("esd_data_points_upper.csv")

err = dfh["esd[h_Msun_per_pc2]"] - dfl["esd[h_Msun_per_pc2]"]

colors = cmr.take_cmap_colors(
    "viridis",
    3,
    cmap_range=(0.10, 0.90),
    return_fmt="hex",
)

fig, (ax, ax_res) = plt.subplots(
    2,
    1,
    figsize=(7.0, 5.2),
    sharex=True,
    gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.06},
)

ax.plot(
    r,
    delta_sigma,
    color=colors[0],
    marker="o",
    markersize=6,
    label=rf"single redshift, $z_l={z_lens:.2f}$",
)

print(cosmo_pars['h'])
print(np.c_[dfc["r[kpc/h]"]/1000, dfc["r[kpc/h]"]/1000/cosmo_pars['h']])
print(np.c_[dfc["esd[h_Msun_per_pc2]"], dfc["esd[h_Msun_per_pc2]"]*cosmo_pars['h']])

#ax.errorbar(
#    dfc["r[kpc/h]"]/1000/cosmo_pars['h'],
#    dfc["esd[h_Msun_per_pc2]"]*cosmo_pars['h'],
#    yerr=err*cosmo_pars['h'],
#    label="Mandelbaum+2006(Webplotdigitizer)"
#)

# ignoring the h-factor conversion between DSF and Mandelbaum dataset
# it might be possible that only delta_sigma predication is off by an h factor. But the radius units are already in Mpc.
ax.errorbar(
    dfc["r[kpc/h]"]/1000/cosmo_pars['h'],
    dfc["esd[h_Msun_per_pc2]"],
    yerr=err,
    label="Mandelbaum+2006\n(Webplotdigitizer)"
)

ax.scatter(
    r,
    delta_sigma_bin,
    color=colors[2],
    s=50,
    label="lens-bin averaged",
    zorder=3,
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_ylabel(r"$\Delta\Sigma(R)\ [M_\odot / {\rm pc}^2]$", fontsize=15)
ax.tick_params(axis="both", which="major", labelsize=13)
ax.tick_params(axis="both", which="minor", labelsize=11)
ax.legend(fontsize=13, frameon=False, title=r"$M_r-5\log h \in [-21,-20]$")

ax_res.axhline(1.0, color="lightgray", linewidth=1.4, linestyle="--")
ax_res.plot(
    r,
    ratio,
    color=colors[1],
)

ax_res.set_xscale("log")
ax_res.set_ylabel("ratio", fontsize=15)
ax_res.set_xlabel(r"$R\ [{\rm Mpc}]$", fontsize=15)
ax_res.tick_params(axis="both", which="major", labelsize=13)
ax_res.tick_params(axis="both", which="minor", labelsize=11)

fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.94)

plt.savefig("cacciato_esd.png", bbox_inches="tight", dpi=200)
