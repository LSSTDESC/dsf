r"""Excess surface density projection from CCL power spectra.

This module provides the standalone ``DeltaSigma`` class for calculating
excess surface density profiles from CCL ``Pk2D`` objects.

The calculation uses a second-order projected Hankel transform,

.. math::

    \Delta\Sigma(R, a) =
    \int_0^\infty \frac{k\,\mathrm{d}k}{2\pi}
    \tilde{\rho}(k, a) J_2(kR),

where the input ``Pk2D`` supplies the Fourier-space density profile.
The transform is evaluated using DSF's FFTLog implementation.
"""

from __future__ import annotations

import numpy as np
import pyccl as ccl
from numpy.typing import NDArray

from dsf.hankel.hankel_transform import HankelTransform
from dsf.utils.validators import (
    validate_positive_1d_array,
    validate_positive_strictly_increasing_1d_array,
    validate_scale_factor,
)

__all__ = ["DeltaSigma"]


class DeltaSigma:
    r"""Calculate excess surface density from a CCL ``Pk2D`` object.

    The class evaluates the Fourier-space profile on a logarithmic wavenumber
    grid and applies a second-order Hankel transform to obtain
    :math:`\Delta\Sigma(R, a)`.

    The input ``Pk2D`` must include the required density normalization.
    Projected radii are comoving and expressed in Mpc.
    """

    def __init__(
        self,
        *,
        pk2d: ccl.Pk2D,
        k: NDArray[np.float64] | None = None,
    ) -> None:
        """Initialize the excess surface density projection.

        Args:
            pk2d: CCL ``Pk2D`` object defining the Fourier-space density profile.
            k: Array of wavenumbers in 1/Mpc at which to evaluate the ``Pk2D``
                profile. If ``None``, defaults to 10000 logarithmically spaced
                points between 1e-6 and 1e6 1/Mpc.
        """
        if k is not None:
            k_arr = validate_positive_strictly_increasing_1d_array(k, "k", min_size=2)
        else:
            k_arr = np.geomspace(1.0e-6, 1.0e6, 10000)

        self.pk2d = pk2d
        self.k = k_arr
        self.hankel = HankelTransform(backend="fftlog")

    def delta_sigma(
        self,
        cosmo: ccl.Cosmology,
        r: NDArray[np.float64],
        a: float,
    ) -> NDArray[np.float64]:
        r"""Calculate :math:`\Delta\Sigma(R, a)` at projected radii.

        Args:
            cosmo: CCL cosmology object.
            r: Projected comoving radii in Mpc.
            a: Scale factor at which the profile is evaluated.

        Returns:
            Excess surface density in the projected-density units implied
            by the input ``Pk2D``.
        """
        r_arr = validate_positive_1d_array(r, "r")
        validate_scale_factor(a)

        pk = np.asarray(self.pk2d(self.k, a), dtype=float)

        _, delta_sigma = self.hankel.projected_correlation_interpolated(
            theta=r_arr,
            radial_input=self.k,
            spectrum=pk,
            order=2,
            grid_spacing="linear",
            taper=True,
        )

        return delta_sigma
