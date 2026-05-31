"""Useful unit and cosmology conversions."""

from __future__ import annotations

from collections.abc import Callable

import astropy.units as u
import numpy as np
import pyccl as ccl

from .types import FloatArray, FloatLike
from .validators import validate_positive_scalar

__all__ = [
    "C_LIGHT_M_PER_S",
    "G_NEWTON_SI",
    "M_PER_MPC",
    "M_SUN_KG",
    "arcmin2_per_steradian",
    "comoving_delta_sigma_to_proper",
    "comoving_distance_h",
    "deg2_to_arcmin2",
    "hubble_constant_per_s_per_mpc",
    "hubble_over_c_cubed",
    "redshift_to_scale_factor",
    "rho_critical_comoving_msun_mpc3",
    "rho_critical_projected_msun_pc2_per_mpc",
    "resolve_h",
    "resolve_omega_m",
    "scale_factor_to_redshift",
    "sigma_crit_prefactor_msun_h_pc2",
    "speed_of_light_mpc_per_s",
]

M_PER_MPC = 3.0856776e22
M_SUN_KG = 1.989e30
G_NEWTON_SI = 6.67408e-11
C_LIGHT_M_PER_S = 2.99792458e8


def arcmin2_per_steradian() -> float:
    """Return the number of square arcminutes in one steradian.

    Returns:
        Number of square arcminutes in one steradian.
    """
    return float((180.0 * 60.0 / np.pi) ** 2)


def deg2_to_arcmin2(area_deg2: float) -> float:
    """Convert square degrees to square arcminutes.

    Args:
        area_deg2: Area in square degrees.

    Returns:
        Area in square arcminutes.
    """
    return float(area_deg2 * 60.0**2)


def scale_factor_to_redshift(a: FloatLike) -> FloatLike:
    """Convert scale factor to redshift.

    Args:
        a: Scale factor or array of scale factors.

    Returns:
        Redshift or array of redshifts.
    """
    return np.asarray(1.0 / np.asarray(a, dtype=float) - 1.0, dtype=float)


def redshift_to_scale_factor(z: FloatLike) -> FloatLike:
    """Convert redshift to scale factor.

    Args:
        z: Redshift or array of redshifts.

    Returns:
        Scale factor or array of scale factors.
    """
    return np.asarray(1.0 / (1.0 + np.asarray(z, dtype=float)), dtype=float)


def speed_of_light_mpc_per_s() -> float:
    """Return the speed of light in Mpc/s.

    Returns:
        Speed of light in Mpc/s.
    """
    return float((ccl.physical_constants["CLIGHT"] * u.m / u.s).to(u.Mpc / u.s).value)


def hubble_constant_per_s_per_mpc(h: float) -> u.Quantity:
    """Return the Hubble constant as an Astropy quantity.

    Args:
        h: Reduced Hubble parameter.

    Returns:
        Hubble constant in km/s/Mpc.
    """
    return h * 100.0 * u.km / u.s / u.Mpc


def hubble_over_c_cubed(h: float) -> float:
    """Return :math:`(H_0 / c)^3` in inverse cubic Mpc.

    Args:
        h: Reduced Hubble parameter.

    Returns:
        Value of :math:`(H_0 / c)^3` in :math:`Mpc^{-3}`.
    """
    h0 = hubble_constant_per_s_per_mpc(h)
    c_mpc_per_s = speed_of_light_mpc_per_s() * u.Mpc / u.s

    return float(((h0**3) / (c_mpc_per_s**3)).to(1 / u.Mpc**3).value)


def comoving_delta_sigma_to_proper(
    func: Callable[..., FloatArray],
) -> Callable[..., FloatArray]:
    """Convert a comoving Delta Sigma function to a proper Delta Sigma function.

    The wrapped function should accept comoving projected radius ``r`` and scale
    factor ``a``. The returned wrapper accepts proper projected radius ``r`` and
    returns proper :math:`\\Delta\\Sigma`.

    Args:
        func: Function computing comoving :math:`\\Delta\\Sigma`.

    Returns:
        Function computing proper :math:`\\Delta\\Sigma`.
    """

    def wrapper(
        r: FloatArray,
        a: float,
        *args: object,
        **kwargs: object,
    ) -> FloatArray:
        """Wrap comoving Delta Sigma function."""
        r = np.asarray(r, dtype=float)

        comoving_result = func(r / a, a, *args, **kwargs)

        return np.asarray(comoving_result / a**2, dtype=float)

    return wrapper


def rho_critical_comoving_msun_mpc3(cosmo: ccl.Cosmology, *, h: float | None = None) -> float:
    """Return critical density in comoving Msun / Mpc^3.

    Args:
        cosmo: CCL cosmology object.
        h: Optional dimensionless Hubble parameter. If omitted, it is read from
            ``cosmo``.

    Returns:
        Critical density in ``Msun / Mpc^3``.
    """
    h_value = float(cosmo["h"]) if h is None else float(h)

    return float(cosmo.rho_x(1.0, "critical", is_comoving=True) / h_value**2)


def rho_critical_projected_msun_pc2_per_mpc(
    cosmo: ccl.Cosmology,
    *,
    h: float | None = None,
) -> float:
    """Return critical density in projected DeltaSigma units.

    Args:
        cosmo: CCL cosmology object.
        h: Optional dimensionless Hubble parameter. If omitted, it is read from
            ``cosmo``.

    Returns:
        Critical density in ``Msun / pc^2 / Mpc``.
    """
    rho_crit = rho_critical_comoving_msun_mpc3(cosmo, h=h)

    return float(rho_crit / 1.0e12)


def comoving_distance_h(
    cosmo: ccl.Cosmology,
    z: FloatArray,
    *,
    h: float,
) -> float | FloatArray:
    """Return the comoving radial distance in Mpc/h.

    Args:
        cosmo: Cosmology used to evaluate the distance.
        z: Redshift value or array of redshift values.
        h: Dimensionless Hubble parameter.

    Returns:
        Comoving radial distance in Mpc/h.
    """
    z_arr = np.asarray(z, dtype=float)

    if z_arr.ndim == 0:
        a = float(redshift_to_scale_factor(float(z_arr)))
        return float(ccl.comoving_radial_distance(cosmo, a) * h)

    a = np.asarray(redshift_to_scale_factor(z_arr), dtype=float).reshape(-1)
    distance = np.asarray(ccl.comoving_radial_distance(cosmo, a), dtype=float) * h

    return distance.reshape(z_arr.shape)


def resolve_h(cosmo: ccl.Cosmology, h: float | None) -> float:
    """Return the supplied Hubble parameter or read it from the CCL cosmology.

    Args:
        cosmo: CCL cosmology object.
        h: Dimensionless Hubble parameter.

    Returns:
        Hubble parameter or read it from the CCL cosmology.
    """
    if h is None:
        h = float(cosmo["h"])

    validate_positive_scalar(h, "h")
    return h


def resolve_omega_m(
    cosmo: ccl.Cosmology,
    omega_m: float | None,
) -> float:
    """Return supplied Omega_m or read it from the CCL cosmology."""
    if omega_m is not None:
        omega_m_float = float(omega_m)
        validate_positive_scalar(omega_m_float, "omega_m")
        return omega_m_float

    try:
        omega_m_float = float(cosmo["Omega_m"])
    except KeyError:
        omega_m_float = float(cosmo["Omega_c"]) + float(cosmo["Omega_b"])

    validate_positive_scalar(omega_m_float, "omega_m")
    return omega_m_float


def sigma_crit_prefactor_msun_h_pc2() -> float:
    """Return the Sigma_crit prefactor for comoving distances in Mpc / h.

    This is the inverse of the prefactor used to compute comoving
    Sigma_crit^{-1} in units of pc^2 / (Msol h). It is intended for
    calculations where distances are supplied in Mpc / h and DeltaSigma-like
    quantities are expressed in Msol h / pc^2.
    """
    sigma_crit_inverse_prefactor = (
        4.0 * np.pi * G_NEWTON_SI * M_SUN_KG * (1.0e12 / C_LIGHT_M_PER_S**2) / M_PER_MPC
    )

    return 1.0 / sigma_crit_inverse_prefactor
