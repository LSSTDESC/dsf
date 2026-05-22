r"""Excess surface density profiles for galaxy-galaxy lensing.

This module evaluates the projected excess surface density,

.. math::

    \Delta\Sigma(R, z) = \bar{\Sigma}(<R, z) - \Sigma(R, z),

from power-spectrum-backed CCL halo profiles.  The main use case is the
galaxy-galaxy lensing observable around a lens sample, optionally averaged over
a lens-bin redshift distribution.

The returned profiles use proejcted comoving radii and are reported in
:math:`M_\odot / \mathrm{pc}^2`.  Any conversion between comoving and proper
surface-density conventions should be applied consistently at the likelihood or
data-vector level.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pyccl as ccl
from numpy.typing import NDArray

from src.dsf.utils.converters import redshift_to_scale_factor
from src.dsf.utils.validators import (
    redshift_window_mask,
    validate_finite_scalar,
    validate_positive_1d_array,
    validate_redshift_distribution,
    validate_redshift_distribution_support,
    validate_scale_factor,
)

from .ccl_integration import HaloProfileGeneric


class DeltaSigmaCalculator:
    r"""Calculate excess surface density profiles from a 3D matter profile.

    The calculator evaluates

    .. math::

        \Delta\Sigma(R, z) = \bar{\Sigma}(<R, z) - \Sigma(R, z),

    where :math:`\Sigma(R, z)` is the projected surface density and
    :math:`\bar{\Sigma}(<R, z)` is its mean within projected radius
    :math:`R`.

    The underlying 3D profile is supplied through a function that returns a CCL
    ``Pk2D`` object.  This makes the calculator agnostic to whether the profile
    comes from a linear matter power spectrum, nonlinear matter power spectrum,
    halo-model prediction, or another power-spectrum-backed model.
    """

    def __init__(
        self,
        pk2d_func: Callable[..., ccl.Pk2D],
    ) -> None:
        """Initialize the calculator with a power-spectrum profile model.

        Args:
            pk2d_func: Function returning the CCL ``Pk2D`` object used to define
                the 3D density profile. The function must accept ``cosmo`` as a
                keyword argument.
        """
        self.pk2d_func = pk2d_func
        self._ccl_profile_cache: HaloProfileGeneric | None = None

    def _generate_ccl_profile(
        self,
        cosmo: ccl.Cosmology,
        pk2d_kwargs: dict[str, Any] | None = None,
        *,
        overwrite_cache: bool = True,
    ) -> HaloProfileGeneric:
        r"""Build the CCL profile used to project the 3D density field.

        Args:
            cosmo: CCL cosmology object.
            pk2d_kwargs: Additional keyword arguments for the profile model.
            overwrite_cache: Whether to rebuild the stored CCL profile.

        Returns:
            CCL-compatible profile whose projection gives
            :math:`\Sigma(R, z)` and :math:`\bar{\Sigma}(<R, z)`.
        """
        kwargs = {} if pk2d_kwargs is None else pk2d_kwargs

        if overwrite_cache or self._ccl_profile_cache is None:
            pk2d = self.pk2d_func(cosmo=cosmo, **kwargs)
            self._ccl_profile_cache = HaloProfileGeneric(
                pk2d=pk2d,
                padding_lo_fftlog=1.0e-6,
                padding_hi_fftlog=1.0e6,
            )

        profile = self._ccl_profile_cache

        if profile is None:
            raise RuntimeError("CCL profile cache was not initialized.")

        return profile

    def _delta_sigma_from_profile(
        self,
        profile: HaloProfileGeneric,
        r: NDArray[np.float64],
        a: float,
        cosmo: ccl.Cosmology,
    ) -> NDArray[np.float64]:
        r"""Evaluate the excess surface density from a projected profile.

        Args:
            profile: CCL-compatible profile describing the 3D density field.
            r: Projected comoving radii in Mpc.
            a: Scale factor at which the lensing profile is evaluated.
            cosmo: CCL cosmology object.

        Returns:
            Excess surface density :math:`\Delta\Sigma(R)` in
            :math:`M_\odot / \mathrm{pc}^2`.
        """
        validate_scale_factor(a)

        sigma = profile.projected(cosmo, r, 1.0, a)
        mean_sigma = profile.cumul2d(cosmo, r, 1.0, a)

        delta_sigma_msun_mpc2 = np.asarray(mean_sigma - sigma, dtype=float)
        delta_sigma_msun_pc2 = delta_sigma_msun_mpc2 / 1.0e12

        if np.any(~np.isfinite(delta_sigma_msun_pc2)):
            bad = np.where(~np.isfinite(delta_sigma_msun_pc2))[0]
            first_bad = int(bad[0])

            raise FloatingPointError(
                "Non-finite DeltaSigma values at one scale factor. "
                f"First bad index: {first_bad}. "
                f"r = {r[first_bad]}, a = {a}."
            )

        return np.asarray(delta_sigma_msun_pc2, dtype=float)

    def delta_sigma(
        self,
        r: NDArray[np.float64],
        a: float,
        cosmo: ccl.Cosmology,
        pk2d_kwargs: dict[str, Any] | None = None,
    ) -> NDArray[np.float64]:
        r"""Calculate :math:`\Delta\Sigma(R)` at a single lens redshift.

        This is the excess projected surface density associated with the chosen
        3D density profile,

        .. math::

            \Delta\Sigma(R) = \bar{\Sigma}(<R) - \Sigma(R).

        Args:
            r: Projected comoving radii in Mpc.
            a: Scale factor of the lens redshift.
            cosmo: CCL cosmology object.
            pk2d_kwargs: Additional parameters for the 3D profile model.

        Returns:
            Excess surface density in :math:`M_\odot / \mathrm{pc}^2`.
        """
        r_arr = validate_positive_1d_array(r, "r")

        profile = self._generate_ccl_profile(
            cosmo=cosmo,
            pk2d_kwargs=pk2d_kwargs,
            overwrite_cache=True,
        )

        return self._delta_sigma_from_profile(
            profile=profile,
            r=r_arr,
            a=a,
            cosmo=cosmo,
        )

    def delta_sigma_lens_bin(
        self,
        r: NDArray[np.float64],
        lens_dndz: tuple[NDArray[np.float64], NDArray[np.float64]],
        cosmo: ccl.Cosmology,
        pk2d_kwargs: dict[str, Any] | None = None,
        *,
        z_min: float | None = None,
        z_max: float | None = None,
        trim_edge_points: int = 0,
    ) -> NDArray[np.float64]:
        r"""Calculate lens-bin-averaged :math:`\Delta\Sigma(R)`.

        For a lens sample with redshift distribution :math:`n_l(z)`, this
        returns the redshift-weighted excess surface density,

        .. math::

            \langle \Delta\Sigma(R) \rangle =
            \frac{\int dz\, n_l(z)\,\Delta\Sigma(R, z)}
                 {\int dz\, n_l(z)}.

        This is useful when the observable is assigned to a tomographic lens bin
        rather than to a single effective lens redshift.

        Args:
            r: Projected comoving radii in Mpc.
            lens_dndz: Tuple ``(z, n_z)`` describing the lens-bin redshift
                distribution.
            cosmo: CCL cosmology object.
            pk2d_kwargs: Additional parameters for the 3D profile model.
            z_min: Optional lower redshift limit of the lens bin.
            z_max: Optional upper redshift limit of the lens bin.
            trim_edge_points: Number of edge points removed from each side of
                the selected positive-support lens distribution.

        Returns:
            Lens-bin-averaged excess surface density in
            :math:`M_\odot / \mathrm{pc}^2`.
        """
        r_arr = validate_positive_1d_array(r, "r")

        z_arr, nz_arr = validate_redshift_distribution(
            lens_dndz[0],
            lens_dndz[1],
            name="lens_dndz",
        )

        window_mask = redshift_window_mask(
            z_arr,
            z_min=z_min,
            z_max=z_max,
        )

        z_use, nz_use, norm = validate_redshift_distribution_support(
            z_arr[window_mask],
            nz_arr[window_mask],
            name="lens_dndz",
            trim_edge_points=trim_edge_points,
        )

        nz_norm = nz_use / norm

        profile = self._generate_ccl_profile(
            cosmo=cosmo,
            pk2d_kwargs=pk2d_kwargs,
            overwrite_cache=True,
        )

        delta_sigma_arr = np.asarray(
            [
                self._delta_sigma_from_profile(
                    profile=profile,
                    r=r_arr,
                    a=float(redshift_to_scale_factor(z)),
                    cosmo=cosmo,
                )
                for z in z_use
            ],
            dtype=float,
        )

        if np.any(~np.isfinite(delta_sigma_arr)):
            bad = np.argwhere(~np.isfinite(delta_sigma_arr))
            first_z_index, first_r_index = bad[0]

            raise FloatingPointError(
                "Non-finite DeltaSigma values before redshift integration. "
                f"First bad index: {first_z_index}, {first_r_index}. "
                f"z = {z_use[first_z_index]}, "
                f"r index = {first_r_index}."
            )

        delta_sigma_mean = np.trapezoid(
            nz_norm[:, None] * delta_sigma_arr,
            z_use,
            axis=0,
        )

        if np.any(~np.isfinite(delta_sigma_mean)):
            bad = np.where(~np.isfinite(delta_sigma_mean))[0]
            first_bad = int(bad[0])

            raise FloatingPointError(
                "Non-finite DeltaSigma values after redshift integration. "
                f"First bad index: {first_bad}."
            )

        return np.asarray(delta_sigma_mean, dtype=float)


def stellar_point_mass_delta_sigma(
    r: NDArray[np.float64],
    log10_mstellar: float,
) -> NDArray[np.float64]:
    r"""Calculate the stellar point-mass contribution to :math:`\Delta\Sigma`.

    The stellar mass of the lens galaxy contributes a small-scale point-mass
    term,

    .. math::

        \Delta\Sigma_\star(R) = \frac{M_\star}{\pi R^2}.

    This contirbution is most relevant at small projected separations where the
    lens galaxy's stellar mass is non-negligible compared with the extended
    dark-matter contribution.

    Args:
        r: Projected comoving radii in Mpc.
        log10_mstellar: Stellar mass in units of
            :math:`\log_{10}(M_\odot)`.

    Returns:
        Stellar point-mass contribution in
        :math:`M_\odot / \mathrm{pc}^2`.
    """
    r_arr = validate_positive_1d_array(r, "r")
    validate_finite_scalar(log10_mstellar, "log10_mstellar")

    return np.asarray(
        (10.0**log10_mstellar) / (np.pi * r_arr**2) / 1.0e12,
        dtype=float,
    )
