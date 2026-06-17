import sys
from pathlib import Path
import numpy as np
from numpy import log10
from scipy.interpolate import InterpolatedUnivariateSpline as ius

# The code representative of Cacciato work: AUM
import cosmology as cc
import hod as h

from cacciato_inputs import (
    cacciato_med_pars,
    cosmo_pars,
    magnitude_to_luminosity,
)
from obs_data_for_cacciato import get_full_sample_esd, config

from dsf.data_vector.delta_sigma_builder import DeltaSigmaCalculator
from dsf.pk2d_cacciato_hod import pk2d_cacciato_hod
from dsf.hod_cacciato import CacciatoHOD

def getdblarr(r):
    temp=h.doubleArray(r.size)
    for i in range(r.size):
        temp[i]=r[i]
    return temp

def initializeHOD(Om0=Om0, w0=-1, wa=0, Omk=0.0, hval=hval, 
        Omb=Omb, th=2.726, s8=s8, nspec=nspec,\
        ximax=log10(8.0), cfac=1.0,\
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

if __name__=="__main__":

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams["text.usetex"] = "True"

    thisdir = Path(__file__).resolve.parent
    benchmark_figdir = thisdir / "data_vector/figures"

    # define the defaults
    Om0 = cosmo_pars.get("Omega_c", None)
    hval = cosmo_pars.get("h", None)
    Omb = cosmo_pars.get("Omega_b", None)
    s8 = cosmo_pars.get("sigma8", None)
    nspec = cosmo_pars.get("n_s", None)

    # Halo masses in Msun/h
    LMhalo = np.linspace(9,15,100)
    # Artificially scaling the halo masses to nullify an additional h-factor
    # adjustment inside the cacciato_hod class.
    LMhalo /= hval

    # Cacciato sample work on
    lumbin = "L4"
    magfaint = config[lumbin]["magnitude_bounds"]["faint_limit"]
    magbright= config[lumbin]["magnitude_bounds"]["bright_limit"]
    file_high = config[lumbin]["files"]["high_fdev"]
    file_low = config[lumbin]["files"]["low_fdev"]
    z_lens = config[lumbin]["mean_redshift"]
    a_lens = 1.0 / (1.0 + z_lens)

    log_L1 = np.log10(magnitude_to_luminosity(magfaint)) #log10( L1 / [Lsun/h^2] )
    log_L2 = np.log10(magnitude_to_luminosity(magbright)) #log10( L2 / [Lsun/h^2] )
    hodparams = dict(log_L1=log_L1, log_L2=log_L2, h=cosmo_pars['h'], **cacciato_med_pars)

    print(cosmo_pars)
    cosmo = ccl.Cosmology(**cosmo_pars)





    args = dict(LMcut=11.75, sigma=0.58, gamma=4.12, one_by_Q=1/100, pmax=0.33, 
                LM1=13.53, alpha=1.0, kappa=1.0
                )

    print(args)

    fig,ax = plt.subplots()
    mhC, Ncen, indexC, ius_Ncen = cenHOD_skew_normal(args, LMhalo, test_spline=True)
    mhS, Nsat, indexS, ius_Nsat = satHOD_PowerLawLMcut(args, LMhalo, test_spline=True)

    aa = initializeHOD()
    aa.hod_free()
    aa.init_Nc_spl(getdblarr(mhC[indexC]), getdblarr(log10(Ncen[indexC])), indexC[indexC].size)
    aa.init_Ns_spl(getdblarr(mhS[indexS]), getdblarr(log10(Nsat[indexS])), indexS[indexS].size)
    #print(mhC[indexC].min(), mhC[indexC].max())
    #print(mhS[indexS].min(), mhS[indexS].max())
    aum_Ncen = np.array([aa.ncen(x) for x in mhC])
    aum_Nsat = np.array([aa.nsat(x) for x in mhS])
    ax.plot(10**mhC, aum_Ncen, "o", ms=5, mfc="None", label="Ncen aum interp")
    ax.plot(10**mhS, aum_Nsat, "o", ms=5, mfc="None", label="Ncen aum interp")
    print( aum_Ncen, np.max(aum_Ncen-ius_Ncen(mhC)) )
    print( aum_Nsat, np.max(aum_Nsat-ius_Ncen(mhS)) )
    plt.yscale("log")
    plt.xscale("log")
    #plt.ylim(yrange)
    plt.grid(True, ls='--', alpha=0.5)
    plt.legend()
    plt.ylabel(r"$\langle N \rangle_{\rm HMQ}$")
    plt.xlabel(r"$M_{\rm halo}$")
    plt.savefig(f"hod_scipy_aum_interpolation_check.png", bbox_inches="tight", dpi=240)


