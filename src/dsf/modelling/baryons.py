"""Baryonic modifications to halo occupation power-spectrum models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyccl as ccl
from numpy.typing import NDArray

from dsf.modelling.hod import MASS_DEF, _pk2d_hod
from dsf.utils.validators import validate_positive_scalar

__all__ = ["pk2d_hod_baryonified"]


class _RescaledConcentration(ccl.halos.Concentration):
    """Apply multiplicative rescaling to a CCL concentration model."""

    name = "RescaledConcentration"

    def __init__(
        self,
        concentration: ccl.halos.Concentration,
        f_c: float,
    ) -> None:
        self.concentration = concentration
        self.f_c = float(f_c)
        super().__init__(mass_def=concentration.mass_def)

    def _check_mass_def_strict(self, mass_def):
        return self.concentration._check_mass_def_strict(mass_def)

    def _concentration(self, cosmo, M, a):
        return self.f_c * self.concentration(cosmo, M, a)


def pk2d_hod_baryonified(
    cosmo: ccl.Cosmology,
    *,
    k_array: NDArray[np.float64],
    a_array: NDArray[np.float64],
    f_c: float = 1.0,
    concentration: ccl.halos.Concentration | None = None,
    **hod_kwargs: Any,
) -> dict[str, ccl.Pk2D]:
    """Construct HOD power spectra with baryonic concentration rescaling.

    Args:
        cosmo: CCL cosmology.
        k_array: Wavenumber grid.
        a_array: Scale-factor grid.
        f_c: Multiplicative concentration rescaling.
        concentration: Optional CCL concentration model. Defaults to Duffy08.
        **hod_kwargs: HOD profile parameters.

    Returns:
        Density-weighted galaxy-matter, galaxy-matter, and galaxy-galaxy spectra.
    """
    validate_positive_scalar(f_c, "f_c")

    if concentration is None:
        concentration = ccl.halos.ConcentrationDuffy08(mass_def=MASS_DEF)

    if f_c != 1.0:
        concentration = _RescaledConcentration(concentration, f_c)

    return _pk2d_hod(
        cosmo,
        k_array=k_array,
        a_array=a_array,
        concentration=concentration,
        **hod_kwargs,
    )
