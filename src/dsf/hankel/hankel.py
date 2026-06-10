"""Public layer for all Hankel transform operations.

This module provides the ``HankelTransform`` class, is an interface for
the ``HankelTransformFFTLog``, ``HankelTransformMatrixDirect``, and
``HankelTransformMatrixZeros`` classes. It provides public methods to
compute projected and spherical correlation functions, as well as covariance
matricies, from these underlying Hankel transform implementations.
"""

from __future__ import annotations

import numpy as np

from dsf.hankel.hankel_transform_fftlog import HankelTransformFFTLog
from dsf.hankel.hankel_transform_matrix_direct import HankelTransformMatrixDirect
from dsf.hankel.hankel_transform_matrix_zeros import HankelTransformMatrixZeros
from dsf.utils.types import ArrayLike, FloatArray, SpectrumInput
from dsf.utils.validators import validate_interpolation_within_bounds


class HankelTransform:
    """Class for performing Hankel transforms using different algorithms."""

    def __init__(self, method="fftlog", **kwargs) -> None:
        """Initialize the HankelTransform class."""
        if method == "fftlog":
            self.backend = HankelTransformFFTLog(**kwargs)
        elif method == "matrix_zeros":
            self.backend = HankelTransformMatrixZeros(**kwargs)
        elif method == "matrix_direct":
            self.backend = HankelTransformMatrixDirect(**kwargs)
        else:
            raise ValueError(
                f"Unsupported method '{method}'. Use 'fftlog', 'matrix_zeros', or 'matrix_direct'."
            )

    def projected_correlation(
        self,
        ell: ArrayLike | None = None,
        c_ell: SpectrumInput | None = None,
        order: float | int = 0,
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute a projected radial statistic from one spectrum.

        Args:
            ell: ell grid for tabulated spectra.
            c_ell: Spectrum values or callable spectrum.
            order: Bessel order to use.
            taper: Whether to suppress low-k and high-k edge power.
            taper_kwargs: Optional settings for the spectrum taper.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Radial grid and projected radial statistic.
        """
        return self.backend.projected_correlation(
            ell=ell,
            c_ell=c_ell,
            order=order,
            **kwargs,
        )

    def spherical_correlation(
        self,
        k_pk: ArrayLike | None = None,
        pk: SpectrumInput | None = None,
        order: float | int = 0,
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute a spherical radial statistic from one spectrum.

        Args:
            k_pk: Wavenumber grid for tabulated spectra (in 1/Mpc).
            pk: Spectrum values or callable spectrum.
            order: Bessel order to use.
            taper: Whether to suppress low-k and high-k edge power.
            taper_kwargs: Optional settings for the spectrum taper.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Radial grid and spherical radial statistic.
        """
        return self.backend.spherical_correlation(
            k_pk=k_pk,
            pk=pk,
            order=order,
            **kwargs,
        )

    def projected_correlation_interpolated(
        self,
        theta: ArrayLike | None = None,
        ell: ArrayLike | None = None,
        c_ell: SpectrumInput | None = None,
        order: float | int = 0,
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute a projected radial statistic from one spectrum at given theta values.

        Args:
            theta: Theta values for interpolation (in radians).
            ell: ell grid for tabulated spectra.
            c_ell: Spectrum values or callable spectrum.
            order: Bessel order to use.
            taper: Whether to suppress low-k and high-k edge power.
            taper_kwargs: Optional settings for the spectrum taper.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Radial grid and projected radial statistic.
        """
        theta_grid, xi_grid = self.backend.projected_correlation(
            ell=ell, c_ell=c_ell, order=order, **kwargs
        )
        theta_eval_arr = validate_interpolation_within_bounds(theta, theta_grid, "theta")

        xi_out = np.interp(theta_eval_arr, theta_grid, xi_grid)
        return theta_eval_arr, np.asarray(xi_out, dtype=float)

    def spherical_correlation_interpolated(
        self,
        r: ArrayLike | None = None,
        k_pk: ArrayLike | None = None,
        pk: SpectrumInput | None = None,
        order: float | int = 0,
        taper: bool = False,
        taper_kwargs: dict | None = None,
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute a spherical radial statistic from one spectrum at given radial values.

        Args:
            r: radial values for interpolation (in Mpc).
            k_pk: Wavenumber grid for tabulated spectra (in 1/Mpc).
            pk: Spectrum values or callable spectrum.
            order: Bessel order to use.
            taper: Whether to suppress low-k and high-k edge power.
            taper_kwargs: Optional settings for the spectrum taper.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Radial grid and spherical radial statistic.
        """
        r_grid, xi_grid = self.backend.spherical_correlation(
            k_pk=k_pk,
            pk=pk,
            order=order,
            taper=taper,
            taper_kwargs=taper_kwargs,
            **kwargs,
        )
        r_eval_arr = validate_interpolation_within_bounds(r, r_grid, "r")

        xi_out = np.interp(r_eval_arr, r_grid, xi_grid)
        return r_eval_arr, np.asarray(xi_out, dtype=float)
