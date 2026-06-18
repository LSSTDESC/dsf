import numpy as np
import matplotlib.pyplot as plt
from binny import NZTomography

''' Script to generate redshift distributions for comparison with legacy covariance implementation.

Borrows heavily from binny examples in docs. '''

# Sources, LSST Year 1

tomo_source = NZTomography()
results_source = tomo_source.build_survey_bins("lsst", role="source", year="1",include_tomo_metadata=True)

bin_dict_source = results_source.bins
z_source = results_source.z

keys_source = sorted(bin_dict_source.keys())

# For now just save one source bin to start with.
# Let's use the highest-z bin.
save_source = np.column_stack((z_source, bin_dict_source[4]))
np.savetxt('./dNdz_source_LSSTY1Bin5.dat', save_source)

# Lenses, DESI LRGs

tomo_lens = NZTomography()
results_lens = tomo_lens.build_survey_bins("desi", role="lens", sample='lrg', overrides={"bins": {"edges":  [0.4, 1.0]}},include_tomo_metadata=True,)

bin_dict_lens = results_lens.bins
z_lens = results_lens.z

keys_lens = sorted(bin_dict_lens.keys())

# Save:
save_lens = np.column_stack((z_lens, bin_dict_lens[0]))
np.savetxt('./dNdz_lens_DESILRG_1bin.dat', save_lens)