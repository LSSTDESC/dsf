"""Power-spectrum-backed density profiles for Delta Sigma.

This module builds generic CCL ``Pk2D`` objects that represent
density-weighted matter power spectra. These objects can be passed to
``HaloProfileGeneric`` so that CCL's projected-profile machinery can be reused
to compute Delta Sigma profiles.

The returned profiles are not halo-mass-dependent halo profiles. They are
Fourier-space density profiles constructed from

    rho_m(a) * P_mm(k, a),

where ``rho_m`` is the comoving mean matter density and ``P_mm`` is either the
linear or nonlinear matter power spectrum.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pyccl as ccl
from numpy.typing import NDArray

__all__ = [
    "density_weighted_power_spectrum",
    "linear_twohalo_density_profile",
    "nonlinear_twohalo_density_profile",
]


def density_weighted_power_spectrum(
    cosmo: ccl.Cosmology,
    power_spectrum: Callable[[ccl.Cosmology, NDArray[np.float64], float], NDArray[np.float64]],
) -> ccl.Pk2D:
    r"""Build a density-weighted matter power spectrum.

    This converts a CCL matter power spectrum function into a ``Pk2D`` object
    representing

    .. math::

        \rho_\mathrm{m}(a) P_\mathrm{mm}(k, a).

    The density weighting is needed because the generic Delta Sigma profile
    uses the supplied ``Pk2D`` as a Fourier-space density profile rather than
    as a dimensionless halo profile.

    Args:
        cosmo: CCL cosmology object.
        power_spectrum: Function with signature ``power_spectrum(cosmo, k, a)``
            returning the matter power spectrum evaluated at wavenumber ``k``
            and scale factor ``a``.

    Returns:
        CCL ``Pk2D`` object for the density-weighted matter power spectrum.
    """

    def density_weighted_power(k: NDArray[np.float64], a: float) -> NDArray[np.float64]:
        r"""Evaluate :math:`\rho_\mathrm{m}(a) P_\mathrm{mm}(k, a)`.

        Args:
            k: Wavenumber values in inverse Mpc.
            a: Scale factor.

        Returns:
            Density-weighted matter power spectrum evaluated at ``k`` and ``a``.
        """
        mean_density = float(cosmo.rho_x(a, "matter", is_comoving=True))
        pk = np.asarray(power_spectrum(cosmo, k, a), dtype=float)

        return mean_density * pk

    return ccl.Pk2D.from_function(
        density_weighted_power,
        is_logp=False,
        extrap_order_lok=1,
        extrap_order_hik=2,
    )


def linear_twohalo_density_profile(
    cosmo: ccl.Cosmology,
) -> ccl.Pk2D:
    r"""Return the density-weighted linear two-halo profile.

    This builds a ``Pk2D`` object proportional to

    .. math::

        \rho_\mathrm{m}(a) P_\mathrm{lin}(k, a),

    which can be used by ``HaloProfileGeneric`` as the Fourier-space profile
    for a linear two-halo Delta Sigma calculation.

    Args:
        cosmo: CCL cosmology object.

    Returns:
        Density-weighted linear matter power spectrum as a CCL ``Pk2D`` object.
    """
    return density_weighted_power_spectrum(cosmo, ccl.linear_matter_power)


def nonlinear_twohalo_density_profile(
    cosmo: ccl.Cosmology,
) -> ccl.Pk2D:
    r"""Return the density-weighted nonlinear two-halo profile.

    This builds a ``Pk2D`` object proportional to

    .. math::

        \rho_\mathrm{m}(a) P_\mathrm{nl}(k, a),

    which can be used by ``HaloProfileGeneric`` as the Fourier-space profile
    for a nonlinear two-halo Delta Sigma calculation.

    Args:
        cosmo: CCL cosmology object.

    Returns:
        Density-weighted nonlinear matter power spectrum as a CCL ``Pk2D``
        object.
    """
    return density_weighted_power_spectrum(cosmo, ccl.nonlin_matter_power)
