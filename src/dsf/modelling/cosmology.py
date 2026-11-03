"""Cosmology construction utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pyccl as ccl

__all__ = ["make_ccl_cosmology"]


def make_ccl_cosmology(
    cosmology: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> ccl.Cosmology:
    """Construct a CCL cosmology, supporting Omega_m and S8 parameterizations.

    Args:
        cosmology: Optional mapping of fiducial cosmological parameters.
        **kwargs: Cosmological parameters overriding the fiducial values.

    Returns:
        CCL cosmology object.

    Raises:
        ValueError: If incompatible parameterizations are supplied.
        KeyError: If parameters required for conversion are missing.

    Notes:
        Omega_m is interpreted as Omega_c + Omega_b, assuming massless
        neutrinos. Keyword overrides may replace the fiducial density or
        amplitude parameterization.
    """
    density_names = {"Omega_m", "Omega_c"}
    amplitude_names = {"S8", "sigma8", "A_s"}

    if cosmology is None and not (density_names | amplitude_names).intersection(kwargs):
        return ccl.CosmologyVanillaLCDM(**kwargs)

    values = dict(cosmology or {})

    if len(density_names.intersection(kwargs)) > 1:
        raise ValueError("Omega_m and Omega_c cannot be supplied together.")

    if len(amplitude_names.intersection(kwargs)) > 1:
        raise ValueError("S8, sigma8, and A_s cannot be supplied together.")

    if "Omega_m" in kwargs:
        values.pop("Omega_c", None)
    elif "Omega_c" in kwargs:
        values.pop("Omega_m", None)

    if "S8" in kwargs:
        values.pop("sigma8", None)
        values.pop("A_s", None)
    elif "sigma8" in kwargs:
        values.pop("S8", None)
        values.pop("A_s", None)
    elif "A_s" in kwargs:
        values.pop("S8", None)
        values.pop("sigma8", None)

    values.update(kwargs)

    if "Omega_m" in values and "Omega_c" in values:
        raise ValueError("Omega_m and Omega_c cannot be supplied together.")

    amplitudes = amplitude_names.intersection(values)

    if len(amplitudes) > 1:
        raise ValueError("S8, sigma8, and A_s cannot be supplied together.")

    if "S8" in values:
        omega_m = values.get("Omega_m")

        if omega_m is None:
            omega_m = values["Omega_c"] + values["Omega_b"]

        if omega_m <= 0:
            raise ValueError("Omega_m must be positive for S8 conversion.")

        values["sigma8"] = values.pop("S8") * (0.3 / omega_m) ** 0.5

    if "Omega_m" in values:
        values["Omega_c"] = values.pop("Omega_m") - values["Omega_b"]

    return ccl.Cosmology(**values)
