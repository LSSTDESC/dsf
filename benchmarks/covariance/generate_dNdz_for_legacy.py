"""Generate redshift distributions for legacy covariance comparisons."""

from pathlib import Path

import numpy as np

from binny import build_survey_tomography

OUTPUT = Path(__file__).resolve().parent

source = build_survey_tomography(
    "lsst",
    role="source",
    year="1",
)

lens = build_survey_tomography(
    "desi",
    role="lens",
    sample="lrg",
    overrides={"bins": {"edges": [0.4, 1.0]}},
)

np.savetxt(
    OUTPUT / "dNdz_source_LSSTY1Bin5.dat",
    np.column_stack((source.z, source.bins[4])),
)

np.savetxt(
    OUTPUT / "dNdz_lens_DESILRG_1bin.dat",
    np.column_stack((lens.z, lens.bins[0])),
)

print(f"Saved legacy covariance redshift distributions to {OUTPUT}")
