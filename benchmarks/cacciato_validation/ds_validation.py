from pathlib import Path

# The code representative of Cacciato work: AUM
import hod as h
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy import log10
from scipy.interpolate import InterpolatedUnivariateSpline as ius

from dsf.hod_cacciato import CacciatoHOD
from dsf.pk2d_cacciato_hod import CONCENTRATION, MASS_DEF

# prepare CLF class and dsf prediction to compare against AUM
from .obs_data_for_cacciato import config
from .predict_dsf_ds_for_cacciato_sample import predict_ds_from_dsf

mpl.rcParams["text.usetex"] = "True"

def getdblarr(r):
    temp=h.doubleArray(r.size)
    for i in range(r.size):
        temp[i]=r[i]
    return temp

def initialize_aum_hod(Om0=0.278, w0=-1, wa=0, Omk=0.0, hval=0.739, 
        Omb=0.041730, th=2.726, s8=0.763, nspec=0.978,\
        ximax=0.90309, cfac=1.0,\
        #irrelevant factors below
        logMmin=13.0, siglogM=0.5, logMsat=14.0, alpsat=1.0,\
        logMcut=13.5, csbycdm=1.0, fac=1.0
        ):
    p = h.cosmo()
    q = h.hodpars()
    p.Om0     = Om0
    p.w0      = w0
    p.wa      = wa
    p.Omk     = Omk
    p.hval    = hval
    p.Omb     = Omb
    p.th      = th
    p.s8      = s8
    p.nspec   = nspec
    p.ximax   = ximax
    p.cfac    = cfac
    q.Mmin    = logMmin
    q.siglogM = siglogM
    q.Msat    = logMsat
    q.alpsat  = alpsat
    q.Mcut    = logMcut
    q.csbycdm = csbycdm
    q.fac     = fac
    return h.hod(p, q)

def put_spline(xx, yy, ext=1):
    #if extrapolation attempted - 
    #ext=0 or ‘extrapolate’, return the extrapolated value.
    #ext=1 or ‘zeros’, return 0
    #ext=2 or ‘raise’, raise a ValueError
    #ext=3 of ‘const’, return the boundary value
    return ius(xx, yy, ext=ext)

def validate_HOD_interpolation(
        LMhalo, Ncen, Nsat, 
        aa, ius_Ncen, ius_Nsat, 
        ylim=None, lumbin="", figdir="./"
    ):
    """This function validates the interpolation bahaviour of AUM and scipy
    agaist the analytical value produced by the CacciatoHOD class."""

    test_mh_vals = np.linspace(9,15,50)
    print(f"interpolated log mass\n{LMhalo}\nmass to test the spline:\n{test_mh_vals}")

    # aum based HOD values at test halo masses
    aum_Ncen = np.array([aa.ncen(x) for x in test_mh_vals])
    aum_Nsat = np.array([aa.nsat(x) for x in test_mh_vals])

    fig,ax = plt.subplots()
    # plot from scipy interp
    ax.plot(
            10**LMhalo, Ncen, 
            ls="-", label="Ncen analytical"
            ) #Msun/h masses
    ax.plot(
            10**test_mh_vals, ius_Ncen(test_mh_vals), 
            ls="--", label="scipy Ncen interp"
            ) #Msun/h masses
    ax.plot(
            10**test_mh_vals, aum_Ncen, 
            "o", ms=5, mfc="None", 
            label="Ncen aum interp") #Msun/h masses

    ax.plot(10**LMhalo, Nsat, ls="-", label="Nsat analytical")
    ax.plot(10**test_mh_vals, ius_Nsat(test_mh_vals), ls="--", label="scipy Nsat interp")
    ax.plot(10**test_mh_vals, aum_Nsat, "o", ms=5, mfc="None", label="Nsat aum interp")

    print( "Central HOD difference:\n", aum_Ncen, np.max(aum_Ncen-ius_Ncen(test_mh_vals)) )
    print( "Satellite HOD difference:\n", aum_Nsat, np.max(aum_Nsat-ius_Ncen(test_mh_vals)) )

    plt.yscale("log")
    plt.xscale("log")
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    plt.ylabel(r"$\langle N \rangle$")
    plt.xlabel(r"$M_{\rm halo}/(h^{-1}{\rm M_\odot})$")
    if ylim is not None and isinstance(ylim, tuple):
        plt.ylim(*ylim)
    else:
        plt.ylim(1e-8, 1e3)
    plt.savefig(
            f"{figdir}/cacciato_{lumbin}_hod_scipy_aum_interpolation_check.png",
            bbox_inches="tight", dpi=240
    )

def aum_deltaSig_predict(aa, rp, z, renewz=False):
    esdbins = rp.size
    #prepare containers of rp and esd
    esdrp = getdblarr(rp)
    esd   = getdblarr(np.zeros(esdbins))
    aa.ESD(z,esdbins,esdrp,esd,esdbins+12,renewz)
    return getnparr(esd,esdbins)

def getnparr(r,n):
    temp=np.zeros(n)
    for i in range(n):
        temp[i]=r[i]
    return temp

def main(
        lumbin, config, 
        validate_hod=True, 
        validate_esd=True, 
        verbose=True, 
        return_dsf_vars=False, 
        figdir="./"
    ):
    dsf_prediciton = predict_ds_from_dsf(lumbin, config)
    cosmo_pars = dsf_prediciton["cosmo_pars"]
    hodpars = dsf_prediciton["hodpars"]
    dsf_ds = dsf_prediciton["delta_sigma"]
    magfaint = dsf_prediciton["magfaint"]
    magbright = dsf_prediciton["magbright"]

    # define the defaults
    Om0 = cosmo_pars["Omega_c"]
    hval = cosmo_pars["h"]
    Omb = cosmo_pars["Omega_b"]
    s8 = cosmo_pars["sigma8"]
    nspec = cosmo_pars["n_s"]
    # Halo masses in Msun/h
    LMhalo = np.linspace(9,15,100)
    # for Delta Sig calc
    z_lens = dsf_prediciton["z_lens"]
    r = dsf_prediciton["r"] *hval #Mpc/h

    # Initialize CacciatoHOD
    chod = CacciatoHOD(
            mass_def=MASS_DEF, #unused
            concentration=CONCENTRATION, #unused 
            **hodpars
    )

    # get HOD from CLF model
    # ----------------------
    # Artificially scaling the halo masses to nullify an additional h-factor
    # adjustment inside the cacciato_hod class.
    Ncen = chod._Nc(10**(LMhalo)/hval)
    ncidx = (Ncen>0)
    if (~ncidx).sum()>0: 
        print("Central HOD cleaning required before interpolation stage.")
    Ncen[~ncidx] = 0.0
    assert ncidx.sum()>3, "Need more than 3 data points to interpolate"
    Nsat = chod._Ns(10**(LMhalo)/hval)
    nsidx = (Nsat>0)
    if (~nsidx).sum()>0: 
        print("Satellite HOD cleaning required before interpolation stage.")
    Nsat[~nsidx] = 0.0
    assert nsidx.sum()>3, "Need more than 3 data points to interpolate"

    if verbose:
        print("Ncen after cleaning:\n", Ncen)
        print("Nsat after cleaning:\n", Nsat)

    # initialize AUM and pass the HOD
    aa = initialize_aum_hod(Om0=Om0, hval=hval, Omb=Omb, s8=s8, nspec=nspec)
    aa.hod_free()
    aa.init_Nc_spl(getdblarr(LMhalo[ncidx]), getdblarr(log10(Ncen[ncidx])), ncidx[ncidx].size)
    aa.init_Ns_spl(getdblarr(LMhalo[nsidx]), getdblarr(log10(Nsat[nsidx])), nsidx[nsidx].size)

    # initialize the scipy based spline for HOD validation
    ius_Ncen = put_spline(LMhalo[ncidx], Ncen[ncidx])
    ius_Nsat = put_spline(LMhalo[nsidx], Nsat[nsidx])
    if validate_HOD_interpolation:
        validate_HOD_interpolation(
                LMhalo, Ncen, Nsat, 
                aa, ius_Ncen, ius_Nsat, 
                lumbin=lumbin, figdir=figdir
        )

    # Next, work on ESD comparison
    aum_esd = aum_deltaSig_predict(aa, r, z_lens)

    if validate_esd:
        fig,ax = plt.subplots()
        # assuming dsf_ds is in Msun/pc^2
        #ax.plot(r, dsf_ds/hval, label="DSF prediction")
        # assuming dsf_ds is in hMsun/pc^2
        ax.plot(r, dsf_ds, label="DSF prediction")
        ax.plot(r, aum_esd, "-o", mfc="None", label="AUM prediction")
        plt.yscale("log")
        plt.xscale("log")
        plt.grid(True, ls='--', alpha=0.5)
        ax.legend(
                fontsize=10, 
                frameon=False, 
                title=rf"{lumbin}: $M_r-5\log h \in [{magfaint:0.1f}, {magbright:0.1f}]$"\
                        + "\n" + rf"$z_l={z_lens:.2f}$"
                )
        xlabel = r"$R \, \left[h^{-1}{\rm Mpc}\right]$"
        ylabel = r"$\Delta\Sigma(R)\ [h {\rm M_\odot {pc}^{-2}}]$"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.savefig(
                f"{benchmark_figdir}/cacciato_{lumbin}_esd_validation_aum_vs_dsf.png",
                bbox_inches="tight", dpi=240
        )

    # Here the unit of ds_dsf is assumed to be as returned (or as reported in
    # the DSF docs): Msun_per_pc2 This this is exactly what I'm trying to test.
    # So be aware that units of DSF is under scrutiny at this point.
    aumdict = {"r_Mpc_per_h": r, "ds_hMsun_per_pc2_aum": aum_esd}
    dsfdict = {"r_Mpc": dsf_prediciton["r"], "ds_dsf": dsf_ds}
    if return_dsf_vars:
        return aumdict, dsfdict, dsf_prediciton
    else:
        return aumdict, dsfdict

if __name__=="__main__":

    thisdir = Path(__file__).resolve().parent
    benchmark_figdir = thisdir.parent / "data_vector/figures/cacciato2013"

    # # ---- input is here -----
    # # Cacciato sample work on
    # lumbin = "L4"
    # # ------------------------
    # main(lumbin, config, validate_hod=True, validate_esd=True)

    for lumbin in ["L2", "L3", "L4", "L5f", "L5b", "L6f"]:
        print(f"working on {lumbin}...")
        main(lumbin, config, validate_hod=True, validate_esd=True, figdir=benchmark_figdir)
