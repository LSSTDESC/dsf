import sys

import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from data_vector.reference.cacciato_inputs import (
    cacciato_med_pars,
    cosmo_pars,
    magnitude_to_luminosity,
)

from dsf.data_vector.delta_sigma_builder import DeltaSigmaCalculator
from dsf.pk2d_cacciato_hod import pk2d_cacciato_hod

mpl.rcParams["text.usetex"] = "True"

benchmark_indir = "data_vector/reference"
benchmark_figdir = "data_vector/figures"

if __name__=="__main__":

    fname = None
    if len(sys.argv) > 1:
        fname = f"{benchmark_figdir}/{sys.argv[1]}"
        plot_log = bool(int(sys.argv[2]))

    magfaint, magbright = -20,-21
    log_L1 = np.log10(magnitude_to_luminosity(magfaint)) #log10( L1 / [Lsun/h^2] )
    log_L2 = np.log10(magnitude_to_luminosity(magbright)) #log10( L2 / [Lsun/h^2] )
    hodparams = dict(log_L1=log_L1, log_L2=log_L2, h=cosmo_pars['h'], **cacciato_med_pars)

    print(cosmo_pars)
    cosmo = ccl.Cosmology(**cosmo_pars)
    
    # prepare inputs for ESD calculation
    # 0.04 - 2 Mpc/h in 12 bins
    r = np.geomspace(0.04/cosmo_pars['h'], 2.0/cosmo_pars['h'], 12)

    z_lens = 0.1
    a_lens = 1.0 / (1.0 + z_lens)
    
    k_array = np.geomspace(1.0e-3, 60.0, 90)
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
    ratio_y_label = r"$\Delta\Sigma_{\rm bin} / \Delta\Sigma$"
    
    # -----------------------------------------------
    # Get the benchmarking data: Mandelbaum 2006
    import pandas as pd
    # note the data units for ESD:hMsun/pc^2, r:kpc/h
    ddf = pd.read_csv(
            f"{benchmark_indir}/L4_esd_data_points_mandelbaum2006_webplotdigitizer.csv"
            , comment="#"
    )
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
    
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    fig, (ax, ax_res) = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0], "hspace": 0.08},
    )

    if plot_log:
        # webplotdigitizer
        dd_x = np.log10(ddf["r"]/1000)
        dd_y = np.log10(ddf["esd"])
        #dd_y_err = np.log10(err)
        #dd_y_err = np.log10(ddf["esd"]+err/2) - np.log10(ddf["esd"]-err/2)
        dd_y_err_lower = dd_y - np.log10(ddf.esd - err)
        dd_y_err_upper = np.log10(ddf.esd + err) - dd_y
        dd_y_err = np.array([dd_y_err_lower, dd_y_err_upper])

        # data
        d_x = np.log10(df["r"]/1000)
        d_y = np.log10(df.esd)
        #d_y_err = np.log10(df.err)
        #d_y_err = np.log10(df.esd+df.err/2) - np.log10(df.esd-df.err/2)
        d_y_err_lower = d_y - np.log10(df.esd - df.err)
        d_y_err_upper = np.log10(df.esd + df.err) - d_y
        d_y_err = np.array([d_y_err_lower, d_y_err_upper])
        # fit to data
        dfit_x = np.log10(tdf.r)
        dfit_y = np.log10(tdf.esd)
        #DSF model
        t_x = np.log10(r*cosmo_pars['h']) #Mpc/h
        t_y = np.log10(delta_sigma/cosmo_pars['h'])
        t_bin_x = np.log10(r*cosmo_pars['h']) #Mpc/h
        t_bin_y = np.log10(delta_sigma_bin/cosmo_pars['h'])

        xscale = "linear"
        yscale = "linear"
        xlabel = r"$\log_{10}\left[ R/ (h^{-1}{\rm Mpc}) \right]$"
        ylabel = r"$\log_{10}\left[ \Delta\Sigma(R)/\ (h{\rm M_\odot {pc}^{-2}}) \right]$"

    else:
        # webplotdigitizer
        dd_x = ddf["r"]/1000
        dd_y = ddf["esd"]
        dd_y_err = err
        # data
        d_x = df["r"]/1000
        d_y = df.esd
        d_y_err = df.err
        # fit to data
        dfit_x = tdf.r
        dfit_y = tdf.esd
        # theory
        t_x = r*cosmo_pars['h']
        t_y = delta_sigma/cosmo_pars['h']
        t_bin_x = r*cosmo_pars['h']
        t_bin_y = delta_sigma_bin/cosmo_pars['h']

        xscale = "log"
        yscale = "log"
        xlabel = r"$R/ (h^{-1}{\rm Mpc})$"
        ylabel = r"$\Delta\Sigma(R)\ [h / {\rm M_\odot {pc}^2}]$"
    
    ax.plot(
        t_x,  # Mpc/h
        t_y, # hMsun/pc^2
        color=colors[0],
        marker="o",
        markersize=6,
        label=rf"single redshift, $z_l={z_lens:.2f}$",
    )
    
    ax.scatter(
        t_bin_x, # Mpc/h
        t_bin_y, # hMsun/pc^2
        color=colors[2],
        s=50,
        label="lens-bin averaged",
        zorder=3,
    )

    # webplotdigitizer
    ax.errorbar(
        dd_x, # Mpc/h
        dd_y, # hMsun/pc^2
        yerr=dd_y_err,
        label="Mandelbaum+2006\n(Webplotdigitizer)"
    )

    # data
    ax.errorbar(
        d_x, # Mpc/h
        d_y, # hMsun/pc^2
        yerr=d_y_err,
        label="L4 data\n(Mandelbaum+2006)"
    )

    # fit to data
    ax.plot(
        dfit_x, # Mpc/h
        dfit_y, # hMsun/pc^2
        label="fitavgsig L4\n(Mandelbaum+2006)"
    )
    
    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.legend(fontsize=10, frameon=False, title=r"$M_r-5\log h \in [-21,-20]$")
    
    ax_res.axhline(1.0, color="lightgray", linewidth=1.4, linestyle="--")
    ax_res.plot(
        t_x,
        ratio,
        color=colors[1],
    )
    ax_res.set_yscale("linear")
    ax_res.set_ylabel(ratio_y_label, fontsize=15)
    ax_res.set_xscale(xscale)
    ax_res.set_xlabel(xlabel, fontsize=15)
    
    if plot_log:
        from matplotlib.ticker import MultipleLocator
        ax_res.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax_res.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.tick_params(axis='both', which='major', length=8, labelsize=13, direction="in")
    ax.tick_params(axis='both', which='minor', length=6, labelsize=11, direction="in")
    ax.tick_params(axis='y', which='both', direction="in", right=True)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    ax_res.tick_params(axis='x', which='both', direction="in", top=True)
    ax_res.tick_params(axis='both', which='major', length=8, labelsize=13, direction="in")
    ax_res.tick_params(axis='both', which='minor', length=6, labelsize=11, direction="in")
    
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.94)
    
    outfname = fname if fname is not None else f"{benchmark_figdir}/cacciato_esd.png"
    plt.savefig(outfname, bbox_inches="tight", dpi=200)
    print("saved figure at ", outfname)

