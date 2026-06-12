import sys

sys.path.insert(0, 
        "/hpc/group/cosmology/nlc38/from_NERSC/dp2_proj/sham_proj/CCLX/my_work/cacciato_hod"
)
import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from cacciato_inputs import cacciato_med_pars, cosmo_pars, magnitude_to_luminosity

from dsf.data_vector.delta_sigma_builder import DeltaSigmaCalculator
from dsf.pk2d_cacciato_hod import pk2d_cacciato_hod

mpl.rcParams["text.usetex"] = "True"

benchmark_indir = "data_vector/reference"
benchmark_figdir = "data_vector/figures"

if __name__=="__main__":

    magfaint, magbright = -20,-21
    log_L1 = np.log10(magnitude_to_luminosity(magfaint)) #log10( L1 / [Lsun/h^2] )
    log_L2 = np.log10(magnitude_to_luminosity(magbright)) #log10( L2 / [Lsun/h^2] )
    hodparams = dict(log_L1=log_L1, log_L2=log_L2, h=cosmo_pars['h'], **cacciato_med_pars)

    print(cosmo_pars)
    cosmo = ccl.Cosmology(**cosmo_pars)
    
    # prepare inputs for ESD calculation
    r = np.geomspace(0.1, 10.0, 20)

    z_lens = 0.1
    a_lens = 1.0 / (1.0 + z_lens)
    
    k_array = np.geomspace(1.0e-3, 30.0, 64)
    a_array = np.linspace(0.3, 1.0, 16)
    
    def pk2d_func(*, cosmo, hodpars=hodparams):
        return pk2d_cacciato_hod(
            cosmo,
            k_array=k_array,
            a_array=a_array,
            **hodpars
        )
    
    calculator = DeltaSigmaCalculator(pk2d_func=pk2d_func)
    delta_sigma = calculator.delta_sigma(
        r=r,
        a=a_lens,
        cosmo=cosmo,
    )
    
    print("predicted ESD signal:\nr, ESD")
    print(np.c_[r,delta_sigma])
    
    # Now prepare for ESD calculation of ESD by splitting the full lens
    # redshift range in multiple tomographic bins
    z = np.linspace(0.05, 0.15, 9)
    n_z = np.exp(-0.5 * ((z - z_lens) / 0.05) ** 2)
    
    ## Make this chose the same cosmology
    #cosmo = make_ccl_cosmology(
    #    transfer_function="eisenstein_hu",
    #    matter_power_spectrum="halofit",
    #)
    
    delta_sigma_bin = calculator.delta_sigma_lens_bin(
        r=r,
        lens_dndz=(z, n_z),
        cosmo=cosmo,
        z_min=0.05,
        z_max=0.15,
    )
    # compare the binned calculation with that at one redshift
    ratio = delta_sigma_bin / delta_sigma
    
    # -----------------------------------------------
    # Get the benchmarking data: Mandelbaum 2006
    import pandas as pd
    # note the data units for ESD:hMsun/pc^2, r:kpc/h
    ddf = pd.read_csv(f"{benchmark_indir}/L4_esd_data_points_mandelbaum2006_webplotdigitizer.csv", comment="#")
    err = ddf["esd_high"] - ddf["esd_low"] # data errorbar

    # theory prediction from Mandelbaum+2005
    df = pd.read_csv(f"{benchmark_indir}/rebin.lum.all.L4.lowfdev.csv", sep=" ", comment="#")
    # real data
    tdf = pd.read_csv(f"{benchmark_indir}/fitavgsig.hh.all.L4.lowfdev.csv", sep=" ", comment="#")
    # -----------------------------------------------
    
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
        r*cosmo_pars['h'],  # Mpc/h
        delta_sigma/cosmo_pars['h'], # hMsun/pc^2
        color=colors[0],
        marker="o",
        markersize=6,
        label=rf"single redshift, $z_l={z_lens:.2f}$",
    )
    
    ax.errorbar(
        ddf["r"]/1000, # Mpc/h
        ddf["esd"], # hMsun/pc^2
        yerr=err,
        label="Mandelbaum+2006\n(Webplotdigitizer)"
    )

    ax.errorbar(
        df.r/1000, # Mpc/h
        df.esd, # hMsun/pc^2
        yerr=df.err,
        label="L4 data\n(Mandelbaum+2006)"
    )

    ax.plot(
        tdf.r, # Mpc/h
        tdf.esd, # hMsun/pc^2
        label="fitavgsig L4\n(Mandelbaum+2006)"
    )
    
    ax.scatter(
        r*cosmo_pars['h'], # Mpc/h
        delta_sigma_bin/cosmo_pars['h'], # hMsun/pc^2
        color=colors[2],
        s=50,
        label="lens-bin averaged",
        zorder=3,
    )
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    ax.set_ylabel(r"$\Delta\Sigma(R)\ [hM_\odot / {\rm pc}^2]$", fontsize=15)
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
    ax_res.set_xlabel(r"$R\ [{\rm Mpc}/h]$", fontsize=15)
    ax_res.tick_params(axis="both", which="major", labelsize=13)
    ax_res.tick_params(axis="both", which="minor", labelsize=11)
    
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.94)
    
    outfname = f"{benchmark_figdir}/cacciato_esd.png"
    plt.savefig(outfname, bbox_inches="tight", dpi=200)
    print("saved figure at ", outfname)

