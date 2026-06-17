import numpy as np
import pyccl as ccl
from cacciato_validation.cacciato_inputs import (
    cacciato_med_pars,
    cosmo_pars,
    magnitude_to_luminosity,
)
from dsf.data_vector.delta_sigma_builder import DeltaSigmaCalculator
from dsf.pk2d_cacciato_hod import pk2d_cacciato_hod

__all__ = ["predict_ds_from_dsf"]

def predict_ds_from_dsf(lumbin, config):
    output = {}

    magfaint = config[lumbin]["magnitude_bounds"]["faint_limit"]
    magbright= config[lumbin]["magnitude_bounds"]["bright_limit"]
    file_high = config[lumbin]["files"]["high_fdev"]
    file_low = config[lumbin]["files"]["low_fdev"]
    z_lens = config[lumbin]["mean_redshift"]
    a_lens = 1.0 / (1.0 + z_lens)

    log_L1 = np.log10(magnitude_to_luminosity(magfaint)) #log10( L1 / [Lsun/h^2] )
    log_L2 = np.log10(magnitude_to_luminosity(magbright)) #log10( L2 / [Lsun/h^2] )
    hodparams = dict(log_L1=log_L1, log_L2=log_L2, h=cosmo_pars['h'], **cacciato_med_pars)

    print(f"cosmology params passed to DSF:\n{cosmo_pars}")
    cosmo = ccl.Cosmology(**cosmo_pars)
    
    # prepare inputs for ESD calculation
    # equivalent numbers for 0.04 - 2 in Mpc/h in 12 bins
    r = np.geomspace(0.04/cosmo_pars['h'], 2.0/cosmo_pars['h'], 12) #Mpc
 
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
    zrange = z_lens-z_lens/2, z_lens+z_lens/2
    z = np.linspace(*zrange, 9)
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
        z_min=zrange[0],
        z_max=zrange[1],
    )
    # compare the binned calculation with that at one redshift
    ratio = delta_sigma_bin / delta_sigma

    output = dict(
            magfaint=magfaint, magbright=magbright, \
            file_low=file_low, file_high=file_high, \
            z_lens=z_lens, cosmo_pars=cosmo_pars, \
            delta_sigma=delta_sigma, delta_sigma_bin=delta_sigma_bin, \
            ratio=ratio, r=r,
    )

    return output
