import numpy as np
from scipy.special import erfc, erf

#### To setup and test scipy interpolation
#Note InterpolatedUnivariateSpline and UnivariateSpline are different. First
#is better for interpolation while latter is used as 1-dimensional smoothing spline.
from scipy.interpolate import InterpolatedUnivariateSpline as ius

#### To setup and test AUM interpolation
import sys
from numpy import log10
sys.path.insert(0, "/mnt/home/student/cnavin/Softwares/aum_tink2/install/lib/python3.11/site-packages/aum-1.0rc0-py3.11-linux-x86_64.egg")
import cosmology as cc
import hod as h

def getdblarr(r):
    temp=h.doubleArray(r.size)
    for i in range(r.size):
        temp[i]=r[i]
    return temp

def initializeHOD(Om0=0.307, w0=-1, wa=0, Omk=0.0, hval=0.68, Omb=0.045, th=2.726, \
                  s8=0.811, nspec=0.961, ximax=log10(8.0), cfac=1.0, logMmin=13.0,\
                  siglogM=0.5, logMsat=14.0, alpsat=1.0, logMcut=13.5, csbycdm=1.0, fac=1.0):
    # initialize cosmology class object
    # the hod params are supplied just to initialize the hod object.
    # these params won't be used anyways since to get ncen and nsat, interpolation will be used.(TINK==2)
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


def cenHOD_skew_normal(args, LMhalo, test_spline=False):
    """central HOD of ELGs of HMQ type. Uses skewed normal distribution from https://en.wikipedia.org/wiki/Skew_normal_distribution."""

    #normal distribution
    z_score = (LMhalo-args['LMcut']) /args['sigma']
    #ignoring constant factors, it will cancel out in normalisation
    normdist = np.exp( - z_score**2 /2.0)
    cdfnorm = ( 1.0 + erf( args['gamma'] *z_score /2**0.5 ) )
    #----------------------------------------------------------------------------------------
    #or this?
    #cdfnorm = ( 1.0 + erf( (args['gamma']*LMhalo - args['LMcut']) /args['sigma'] /2**0.5 ) )
    #----------------------------------------------------------------------------------------
    #print(f"{normdist.size} {np.sum(normdist==0)}")
    #print("norm", normdist, "cdf", cdfnorm, sep="\n")

    skewnorm = 2.0 *normdist *cdfnorm
    max_val = np.max(skewnorm)
    print("max val", max_val)
    #normalize to pmax
    skewnorm = (args['pmax']-args['one_by_Q']) *skewnorm /max_val
    #smooth step function to limit high mass probability
    step_func = 0.5 *args['one_by_Q'] *( 1 + erf( (LMhalo-args['LMcut']) /0.01) ) #one_by_Q := 1/Q
    #Now add the step function to saturate the high mass end
    skewnorm = skewnorm + step_func

    #to avoid passing zero values to interpolation
    index = (skewnorm>0)
    skewnorm[~index] = 0.0
    if test_spline:
        Ncen_interp = put_spline(LMhalo[index], skewnorm[index])
        ax.plot(10**LMhalo, skewnorm, ls="-", label="Ncen formula")
        ax.plot(10**LMhalo, Ncen_interp(LMhalo), ls="--", label="Ncen interp")
        print("model and scipy interpo comparison", (skewnorm-Ncen_interp(LMhalo)).max())
        return LMhalo, skewnorm, index, Ncen_interp
    else:
        return skewnorm

def satHOD_PowerLawLMcut(args, LMhalo, test_spline=False):
    """ELG satellite HOD with kappa*Mcut cutoff"""
    Mdiff = 10**(LMhalo - args['LM1']) - args['kappa'] *10**(args['LMcut'] - args['LM1'])
    Nsat = np.zeros(Mdiff.size)

    #to avoid passing zero values to interpolation
    index = (Mdiff>0)
    Nsat[~index] = 0.0
    Nsat[index] = Mdiff[index]**args['alpha']

    if test_spline:
        Nsat_interp = put_spline(LMhalo[index], Nsat[index])
        ax.plot(10**LMhalo, Nsat, ls="-", label="Nsat formula")
        ax.plot(10**LMhalo, Nsat_interp(LMhalo), ls="--", label="Nsat interp")
        return LMhalo, Nsat, index, Nsat_interp
    else:
        return Nsat

def put_spline(xx, yy, ext=1):
    #if extrapolation attempted - 
    #ext=0 or ‘extrapolate’, return the extrapolated value.
    #ext=1 or ‘zeros’, return 0
    #ext=2 or ‘raise’, raise a ValueError
    #ext=3 of ‘const’, return the boundary value
    return ius(xx, yy, ext=ext)

if __name__=="__main__":
    import sys
    import matplotlib.pyplot as plt

    LMhalo = np.linspace(9,15,100)

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


