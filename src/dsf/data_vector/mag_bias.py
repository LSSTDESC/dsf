"""Lens-magnification correction for Delta Sigma calculations.

This module computes the correction to the comoving excess surface density
profile caused by magnification of the lens sample. The correction is evaluated
using a nested integral over foreground redshift and multipole.

The public function ``delta_sigma_lens_mag_correction`` returns the quantity
that should be subtracted from the observed Delta Sigma signal when including
lens magnification.

The calculation currently approximates the source redshift as

    z_source = z_lens + delta_z_source,

where ``delta_z_source`` is controlled by the lens-magnification integration
parameters. But note that the choice of ``delta_z_source`` will cancel out upon
transformation from ``gamma_t`` to ``Delta Sigma`` space.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyccl as ccl
from numpy.typing import NDArray

from dsf.utils.converters import (
    hubble_over_c_cubed,
    redshift_to_scale_factor,
    scale_factor_to_redshift,
)
from dsf.utils.hankel_transform_1d import hankel_projected_order_2
from dsf.utils.integrators import trapezoid_integral
from dsf.utils.validators import (
    validate_finite_scalar,
    validate_integration_params,
    validate_positive_1d_array,
    validate_redshift_pair,
    validate_scale_factor,
)

_LENS_MAG_INTEG_PARAMS: dict[str, int | float] = {
    "n_ell": 5000,
    "ell_min": 1.0e-4,
    "ell_max": 1.0e6,
    "z_stepsize": 0.02,
    "z_min": 1.0e-5,
    "delta_z_source": 1.0,
    "use_hankel_offset": False,
}


def get_lens_mag_integ_params() -> dict[str, int | float]:
    """Return the lens-magnification integration parameters.

    Returns:
        Copy of the current lens-magnification integration parameter
        dictionary.
    """
    return _LENS_MAG_INTEG_PARAMS.copy()


def set_lens_mag_integ_params(**kwargs: Any) -> None:
    """Update the lens-magnification integration parameters.

    Args:
        **kwargs: Integration parameters to update. Supported keys are
            ``n_ell``, ``ell_min``, ``ell_max``, ``z_stepsize``, ``z_min``,
            ``delta_z_source``, and ``use_hankel_offset``.

    Raises:
        KeyError: If an unknown integration parameter is supplied.
        ValueError: If the updated integration parameters are invalid.
    """
    unknown_keys = set(kwargs) - set(_LENS_MAG_INTEG_PARAMS)

    if unknown_keys:
        raise KeyError(
            f"Unknown lens-magnification integration parameter(s): {sorted(unknown_keys)}"
        )

    updated_params = _LENS_MAG_INTEG_PARAMS.copy()
    updated_params.update(kwargs)

    validate_integration_params(updated_params)

    _LENS_MAG_INTEG_PARAMS.update(updated_params)


def _lens_mag_distance_kernel(
    cosmo: ccl.Cosmology,
    a_inner: NDArray[np.float64],
    a_lens: NDArray[np.float64],
    a_source: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute the distance kernel for the lens-magnification redshift integral.

    Args:
        cosmo: CCL cosmology object.
        a_inner: Scale-factor grid for the foreground redshift integral.
        a_lens: Lens scale factor broadcast to the shape of ``a_inner``.
        a_source: Source scale factor broadcast to the shape of ``a_inner``.

    Returns:
        Distance kernel evaluated on the foreground scale-factor grid.
    """
    distance_observer_to_lens = ccl.angular_diameter_distance(cosmo, a_lens)
    distance_observer_to_source = ccl.angular_diameter_distance(cosmo, a_source)
    distance_inner_to_lens = ccl.angular_diameter_distance(
        cosmo,
        a_inner,
        a_lens,
    )
    distance_inner_to_source = ccl.angular_diameter_distance(
        cosmo,
        a_inner,
        a_source,
    )

    kernel = (
        distance_inner_to_lens
        * distance_inner_to_source
        / (distance_observer_to_lens * distance_observer_to_source)
    )

    return kernel


def _inner_redshift_integrand(
    z_inner: NDArray[np.float64],
    ell: NDArray[np.float64],
    cosmo: ccl.Cosmology,
    z_lens: float,
    z_source: float,
) -> NDArray[np.float64]:
    """Evaluate the foreground-redshift integrand.

    The inner integral runs over redshift between the observer and the lens.
    The returned array has shape ``(n_z, n_ell)``.

    Args:
        z_inner: Foreground redshift grid for the inner integral.
        ell: Multipole grid for the outer integral.
        cosmo: CCL cosmology object.
        z_lens: Lens redshift.
        z_source: Source redshift.

    Returns:
        Foreground-redshift integrand evaluated on the redshift and multipole
        grids.

    Raises:
        ValueError: If the lens/source redshift pair or multipole grid is
            invalid.
    """
    validate_redshift_pair(z_lens, z_source)

    z_arr = np.asarray(z_inner, dtype=float)
    if z_arr.ndim != 1:
        raise ValueError("z_inner must be one-dimensional.")

    if z_arr.size < 2:
        raise ValueError("z_inner must contain at least two values.")

    if np.any(~np.isfinite(z_arr)):
        raise ValueError("z_inner must contain only finite values.")

    if np.any(z_arr < 0.0):
        raise ValueError("z_inner must be non-negative.")

    ell_arr = validate_positive_1d_array(ell, "ell")

    a_inner = redshift_to_scale_factor(z_arr)
    a_lens = np.broadcast_to(redshift_to_scale_factor(z_lens), np.shape(a_inner))
    a_source = np.broadcast_to(
        redshift_to_scale_factor(z_source),
        np.shape(a_inner),
    )

    distance_kernel = _lens_mag_distance_kernel(
        cosmo,
        a_inner,
        a_lens,
        a_source,
    )

    redshift_kernel = ((1.0 + z_arr) ** 2) / ccl.h_over_h0(cosmo, a_inner)
    prefactor = redshift_kernel * distance_kernel
    
    D_a = ccl.angular_diameter_distance(cosmo, a_inner)
    ell_plus = (ell_arr + 0.5).reshape((1, -1))
    lk_grid = np.log(a_inner.reshape((-1, 1)) * ell_plus / D_a.reshape((-1, 1)))

    pk2d_m = cosmo.get_nonlin_power()
    matter_power = np.array(
        [
            ccl.lib.pk2d_eval_multi(
                pk2d_m.psp, lk_grid[a_i], a_use, cosmo.cosmo, lk_grid[a_i].size, 0
            )[0]
            for a_i, a_use in enumerate(a_inner)
        ],
        dtype=float
    )

    return prefactor[:, None] * matter_power


def _lens_mag_lss_shear(
    cosmo: ccl.Cosmology,
    theta: NDArray[np.float64],
    z_lens: float,
    z_source: float,
) -> NDArray[np.float64]:
    """Compute the LSS tangential shear entering the lens-magnification term.

    Args:
        cosmo: CCL cosmology object.
        theta: Angular separation values in radians.
        z_lens: Lens redshift.
        z_source: Source redshift.

    Returns:
        Large-scale-structure tangential shear contribution evaluated at each
        angular separation.

    Raises:
        ValueError: If the redshift pair, angular grid, or integration
            parameters are invalid.
    """
    validate_redshift_pair(z_lens, z_source)

    theta_arr = validate_positive_1d_array(theta, "theta")

    ell_arr = np.geomspace(
        float(_LENS_MAG_INTEG_PARAMS["ell_min"]),
        float(_LENS_MAG_INTEG_PARAMS["ell_max"]),
        int(_LENS_MAG_INTEG_PARAMS["n_ell"]),
    )
    ell_arr = validate_positive_1d_array(ell_arr, "ell")
    
    z_arr = np.arange(
        float(_LENS_MAG_INTEG_PARAMS["z_min"]),
        z_lens,
        step=float(_LENS_MAG_INTEG_PARAMS["z_stepsize"]),
    )
    z_arr = validate_positive_1d_array(z_arr, "z")

    C_ell = trapezoid_integral(
        _inner_redshift_integrand(z_arr, ell_arr, cosmo, z_lens, z_source),
        z_arr,
        axis=0
    )
    gamma_t_spline = hankel_projected_order_2(C_ell, ell_arr, offset=_LENS_MAG_INTEG_PARAMS["use_hankel_offset"])

    prefactor = (
        9.0 * hubble_over_c_cubed(float(cosmo["h"])) * float(cosmo["Omega_m"]) ** 2 / 4
    )

    return prefactor * gamma_t_spline(theta_arr)


def delta_sigma_lens_mag_correction(
    r: NDArray[np.float64],
    a: float,
    cosmo: ccl.Cosmology,
    alpha_lens: float,
) -> NDArray[np.float64]:
    r"""Compute the lens-magnification correction to Delta Sigma.

    This returns the comoving correction to the excess surface density profile
    caused by magnification of the lens sample. The correction is evaluated at
    lens scale factor ``a`` and projected comoving radii ``r``.

    The returned correction has the same radial shape as ``r`` and is intended
    to be subtracted from the measured Delta Sigma signal.

    The source redshift is approximated as

    .. math::

        z_\mathrm{s} = z_\mathrm{l} + \Delta z_\mathrm{s},

    where ``delta_z_source`` is taken from the lens-magnification integration
    parameters.

    Args:
        r: Comoving projected radii in Mpc.
        a: Lens scale factor.
        cosmo: CCL cosmology object.
        alpha_lens: Lens-sample magnification-bias slope. The correction is
            proportional to ``alpha_lens - 1``.

    Returns:
        Comoving lens-magnification correction in
        :math:`M_\odot / \mathrm{pc}^2`.

    Raises:
        ValueError: If the radius array, scale factor, lens magnification-bias
            slope, or derived lens/source redshift pair is invalid.
    """
    validate_scale_factor(a)
    validate_finite_scalar(alpha_lens, "alpha_lens")

    r_arr = validate_positive_1d_array(r, "r")

    z_lens = scale_factor_to_redshift(a)
    z_source = z_lens + float(_LENS_MAG_INTEG_PARAMS["delta_z_source"])
    a_source = redshift_to_scale_factor(z_source).item()

    d_ang_lens = ccl.angular_diameter_distance(cosmo, a)

    # r is comoving, while D_A is physical. The transverse comoving distance is
    # D_M = D_A / a, so theta = r / D_M = r * a / D_A.
    theta = r_arr * a / d_ang_lens

    lss_shear = _lens_mag_lss_shear(
        cosmo,
        theta,
        z_lens,
        z_source,
    )

    correction = (
        2.0
        * a**2
        * ccl.sigma_critical(cosmo, a_lens=a, a_source=a_source)
        * (alpha_lens - 1.0)
        * lss_shear
        / 1.0e12
    )

    return correction
