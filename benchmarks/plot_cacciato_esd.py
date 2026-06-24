# call signature: 
# ipython -i plot_cacciato_esd.py cacciato_esd.png 0 0 ==> cacciato_esd.png
# ipython -i plot_cacciato_esd.py cacciato_esd.png 0 1 ==> cacciato_esd_shifted_yval.png
# ipython -i plot_cacciato_esd.py cacciato_esd_log.png 1 0 ==> cacciato_esd_log.png
# ipython -i plot_cacciato_esd.py cacciato_esd_log.png 1 1 ==> cacciato_esd_log_shifted_yval.png

import sys
from pathlib import Path

import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cacciato_validation.ds_validation import main
from cacciato_validation.obs_data_for_cacciato import config, get_full_sample_esd

mpl.rcParams["text.usetex"] = "True"

benchmark_indir = config["dir"]
benchmark_figdir = "data_vector/figures/cacciato2013"

if __name__=="__main__":

    fname = None
    shift_h_factor = False
    if len(sys.argv) > 1:
        fname = Path(sys.argv[1])
        plot_log = bool(int(sys.argv[2]))
        shift_h_factor = bool(int(sys.argv[3]))

    # This string completely defines the Cacciato sample on interest
    lumbin = "L6f"

    fname = Path(fname.stem + f"_{lumbin}" + fname.suffix)
    if shift_h_factor:
        fname = fname.stem + f"_shifted_yval{fname.suffix}"

    fname = Path(f"{benchmark_figdir}/{fname}")
    print(fname)

    # get prediction from DSF and AUM
    # Note: dsfdict is fully contained in dsf_output
    aumdict, dsfdict, dsf_output = main(
            lumbin, config, 
            validate_hod=False, validate_esd=False, 
            verbose=True, return_dsf_vars=True, 
            figdir=benchmark_figdir
    )
    # locals().update(dsf_output)
    ratio = dsf_output["ratio"]
    magfaint = dsf_output["magfaint"]
    magbright = dsf_output["magbright"]
    delta_sigma = dsf_output["delta_sigma"]
    file_low = dsf_output["file_low"]
    file_high = dsf_output["file_high"]
    z_lens = dsf_output["z_lens"]
    cosmo_pars = dsf_output["cosmo_pars"]
    delta_sigma_bin = dsf_output["delta_sigma_bin"]
    r = dsf_output["r"]

    ratio_y_label = r"$\Delta\Sigma_{\rm bin} / \Delta\Sigma$"

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

    obs_data_vector = get_full_sample_esd(benchmark_indir, file_high, file_low)
    xval = obs_data_vector.r.values/1000
    yval = obs_data_vector.esd.values
    err = obs_data_vector.err.values
    if plot_log:
        # full sample data: Mandelbaum+2006
        f_x = np.log10(xval)
        rmask = (f_x > -1.5)
        f_x = f_x[rmask]
        f_y = np.log10(yval)[rmask]

        # --- extract the error bars ---
        fig_dummy, ax_dummy = plt.subplots()
        container = ax_dummy.errorbar(f_x, yval[rmask], yerr=err[rmask])
        ax_dummy.set_yscale('log')
        lines = container[2][0].get_segments()
        plt.close(fig_dummy)
        y_min = np.array([line[0, 1] for line in lines]) 
        y_max = np.array([line[1, 1] for line in lines])
        f_y_err_lower = f_y - np.log10(y_min)
        f_y_err_upper = np.log10(y_max) - f_y
        # Force any NaN/Negative values from the log zero floor to a safe visual constant
        f_y_err_lower = np.where(np.isnan(f_y_err_lower) | (f_y_err_lower < 0), 1.0, f_y_err_lower)
        f_y_err_upper = np.where(np.isnan(f_y_err_upper) | (f_y_err_upper < 0), 1.0, f_y_err_upper)
        f_y_err = np.array([f_y_err_lower, f_y_err_upper])

        #DSF model
        t_x = np.log10(r*cosmo_pars['h']) #Mpc/h
        t_y = np.log10(delta_sigma/cosmo_pars['h'])
        t_bin_x = np.log10(r*cosmo_pars['h']) #Mpc/h
        t_bin_y = np.log10(delta_sigma_bin/cosmo_pars['h'])

        #AUM model
        aum_x = np.log10(aumdict["r_Mpc_per_h"])
        aum_y = np.log10(aumdict["ds_hMsun_per_pc2_aum"])

        if shift_h_factor:
            # this is just a debugging step
            # assume units already in hMsun/pc^2
            t_y = np.log10(delta_sigma) 
            t_bin_y = np.log10(delta_sigma_bin)

        xscale = "linear"
        yscale = "linear"
        xlabel = r"$\log_{10}\left[ R/ (h^{-1}{\rm Mpc}) \right]$"
        ylabel = r"$\log_{10}\left[ \Delta\Sigma(R)/\ (h{\rm M_\odot {pc}^{-2}}) \right]$"
        ylim =  (-0.1, 2.5)
    else:
        # full sample data: Mandelbaum+2006
        f_x = xval
        rmask = np.log10(f_x)>-1.5
        f_x = f_x[rmask]
        f_y = yval[rmask]
        f_y_err = err[rmask]

        # theory
        t_x = r*cosmo_pars['h']
        t_y = delta_sigma/cosmo_pars['h']
        t_bin_x = r*cosmo_pars['h']
        t_bin_y = delta_sigma_bin/cosmo_pars['h']

        #AUM model
        aum_x = aumdict["r_Mpc_per_h"]
        aum_y = aumdict["ds_hMsun_per_pc2_aum"]

        if shift_h_factor:
            # this is just a debugging step
            # assume units already in hMsun/pc^2
            t_y = delta_sigma
            t_bin_y = delta_sigma_bin

        xscale = "log"
        yscale = "log"
        xlabel = r"$R/ (h^{-1}{\rm Mpc})$"
        ylabel = r"$\Delta\Sigma(R)\ [h{\rm M_\odot {pc}^{-2}}]$"

    # aum model
    ax.plot(
        aum_x,  # Mpc/h
        aum_y, # hMsun/pc^2
        color=colors[0],
        marker="o",
        markersize=6,
        label=r"AUM prediction",
    )
    
    # dsf model
    ax.plot(
        t_x,  # Mpc/h
        t_y, # hMsun/pc^2
        color=colors[1],
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

    # data
    ax.errorbar(
        f_x, # Mpc/h
        f_y, # hMsun/pc^2
        yerr=f_y_err,
        capsize=4,
        marker = "o",
        ls = "none",
        label=f"{lumbin} data\n(Mandelbaum+2006)"
    )

    ax.set_yscale(yscale)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.legend(
            fontsize=10, frameon=False, 
            title=rf"$M_r-5\log h \in [{magfaint:0.1f}, {magbright:0.1f}]$"
    )
    
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
        ax.set_ylim(ylim)
    ax.tick_params(axis='both', which='major', length=8, labelsize=13, direction="in")
    ax.tick_params(axis='both', which='minor', length=4, labelsize=11, direction="in")
    ax.tick_params(axis='y', which='both', direction="in", right=True)
    ax.grid(True, which="both", ls="--", alpha=0.3)

    ax_res.tick_params(axis='x', which='both', direction="in", top=True)
    ax_res.tick_params(axis='both', which='major', length=8, labelsize=13, direction="in")
    ax_res.tick_params(axis='both', which='minor', length=4, labelsize=11, direction="in")
    
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.15, top=0.94)
    
    outfname = fname if fname is not None else f"{benchmark_figdir}/cacciato_esd_{lumbin}.png"
    plt.savefig(outfname, bbox_inches="tight", dpi=200)
    print("saved figure at ", outfname)

