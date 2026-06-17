import numpy as np

# Cacciato+2013 cosmology parameters as required in CCL
cosmo_pars = dict(
    Omega_c = 0.278,
    h = 0.739,
    n_s = 0.978,
    sigma8 = 0.763,
)

hval = cosmo_pars["h"]
cosmo_pars['Omega_b'] = 2.279 / 100 / hval** 2  # baryon density

# raw median HOD parameters
cacciato_med_pars = dict(
    log_L0=9.95,  # units in paper are in Lsun/h^2
    log_M1=11.24,  # halo mass units are in Msun/h
    gamma_1=3.18,  # unitless
    gamma_2=0.245,  # unitless
    sigma_c=0.157,  # unitless
    alpha_s=-1.18,  # unitless
    b_0=-1.17,  # unitless 
    b_1=1.53,  # unitless -- needs grid rescaling for Msun/h -> Msun
    b_2=-0.217,  # unitless -- needs grid rescaling for Msun/h -> Msun
)

# to use units which do not have h factors, you will have to redefine M12
# function itself. A mere scaling of b parameters can't capture that.
#def get_h_dep_cacciato_pars(hval=None):
#    # remove h-independence to work with CCL
#    p = cacciato_med_pars
#    if hval is None:
#        hval = cosmo_pars["h"]
#    logh = np.log10(hval)
#    return dict(
#        log_L0=p["log_L0"] - 2 * logh,  # Lsun
#        log_M1=p["log_M1"] - logh,  # Msun
#        gamma_1=p["gamma_1"],  # unitless
#        gamma_2=p["gamma_2"],  # unitless
#        sigma_c=p["sigma_c"],  # unitless
#        alpha_s=p["alpha_s"],  # unitless
#        b_0=p["b_0"],  # unitless - no rescaling needed as per the HOD model
#        b_1=p["b_1"], # unitless - re-define M12 to use halo masses in Msun
#        b_2=p["b_2"] # unitless - re-define M12 to use halo masses in Msun 
#    )

# Note: Cacciato compiles data vectors from different studies, as a result,
# they used different samples (roughly similar) for clustering, lensing and LF
# measurements.  But they model chi-square contribution of each of the data
# vectors using the summary info of respective samples.  Only in the end they
# joinly maximize the likelihood!  The below listed summary stats are from the
# clustering sample (I. Zehavi 2011)
mag_edges = np.linspace(-23, -17, 7)
mag_bins = np.array(list(zip(mag_edges[:-1], mag_edges[1:], strict=True)))
sampleinfo = dict(
    mag_bins=mag_bins,
    zmins=np.array([0.011, 0.017, 0.027, 0.042, 0.066, 0.103]),
    zmaxs=np.array([0.026, 0.042, 0.064, 0.106, 0.159, 0.245]),
    zmeans=np.array([0.021, 0.032, 0.050, 0.082, 0.123, 0.187]),
)

# magnitude -> luminosity helper
def magnitude_to_luminosity(M, M_ref=4.76):
    """Convert magnitude to luminosity in Lsun/h^2"""
    return 10 ** (0.4 * (M_ref - M))

