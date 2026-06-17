import sys
from pathlib import Path

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
from data_vector.reference.obs_data_for_cacciato import obs_data_vector

from dsf.data_vector.delta_sigma_builder import DeltaSigmaCalculator
from dsf.pk2d_cacciato_hod import pk2d_cacciato_hod

mpl.rcParams["text.usetex"] = "True"

benchmark_indir = "data_vector/reference"
benchmark_figdir = "data_vector/figures"

if __name__=="__main__":

