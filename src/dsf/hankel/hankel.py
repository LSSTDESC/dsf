"""Public layer for all Hankel transform operations.

This module provides the ``HankelTransform`` class, an interface for
the ``HankelTransformFFTLog``, ``HankelTransformMatrixDirect``, and
``HankelTransformMatrixZeros`` classes. It provides public methods to
compute projected and spherical correlation functions, as well as covariance
matrices, from these underlying Hankel transform implementations.

Each of the three available backends can be useful for different purposes.

- ``HankelTransformFFTLog``: Supports projected and spherical 1D correlation
  functions. Fastest option for most 1D applications.
- ``HankelTransformMatrixZeros``: Supports projected 1D and 2D Hankel
  transforms using zero-crossing methods. Requires fine-tuning of the
  zero-crossing points upon generation. Can exhibit ringing at small scales.
- ``HankelTransformMatrixDirect``: Supports projected 1D and 2D Hankel
  transforms using direct matrix operator methods. Requires fine-tuning
  of the k-sampling upon generation. Robust against ringing at small scales,
  but can exhibit minor ringing at large scales.

The mathematical conventions for the projected and spherical transforms
are as follows:

:math:`\\gamma_t(\\theta) = \\int \\frac{\\ell d\\ell}{2\\pi} C(\\ell )J_\\mu(\\ell \\theta)`.

:math:`\\xi(r) = \\int \\frac{k^2 dk}{2\\pi^2} P(k) j_\\mu(kr)`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from dsf.hankel.hankel_transform_fftlog import HankelTransformFFTLog
from dsf.hankel.hankel_transform_matrix_direct import HankelTransformMatrixDirect
from dsf.hankel.hankel_transform_matrix_zeros import HankelTransformMatrixZeros
from dsf.utils.interpolators import interpolate_linear, interpolate_loglog
from dsf.utils.types import ArrayLike, FloatArray, SpectrumInput
from dsf.utils.validators import (
    validate_interpolation_within_bounds,
    validate_positive_strictly_increasing_1d_array,
)

HankelBackend = Literal["fftlog", "matrix_zeros", "matrix_direct"]
_BACKENDS = {
    "fftlog": HankelTransformFFTLog,
    "matrix_zeros": HankelTransformMatrixZeros,
    "matrix_direct": HankelTransformMatrixDirect,
}


class HankelTransform:
    """Class for performing Hankel transforms using different algorithms."""

    def __init__(self, backend: HankelBackend = "fftlog", **kwargs) -> None:
        """Initialize the HankelTransform class."""
        try:
            backend_class = _BACKENDS[backend]
        except KeyError as e:
            valid = "', '".join(_BACKENDS)
            raise ValueError(
                f"Unsupported backend '{backend}'. Use one of: '{valid}'."
            ) from e

        self.backend_name = backend
        self.backend = backend_class(**kwargs)

    def projected_correlation(
        self,
        ell: ArrayLike | None = None,
        c_ell: SpectrumInput | None = None,
        order: float | int = 0,
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute a projected radial statistic from one spectrum.

        Args:
            ell: ell grid for tabulated spectra (unitless).
            c_ell: Spectrum values or callable spectrum.
            order: Bessel order to use.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Radial grid and projected radial statistic (units of radians).
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
            k_pk: Wavenumber grid for tabulated spectra.
            pk: Spectrum values or callable spectrum.
            order: Bessel order to use.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Radial grid and spherical radial statistic. The units of
            the radial grid are the inverse of the units of ``k``.
        """
        return self.backend.spherical_correlation(
            k_pk=k_pk,
            pk=pk,
            order=order,
            **kwargs,
        )

    def projected_correlation_interpolated(
        self,
        theta: FloatArray,
        ell: FloatArray,
        c_ell: SpectrumInput,
        order: float | int = 0,
        grid_spacing: str = "linear",
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute a projected radial statistic from one spectrum at given theta values.
            The statistic is interpolated from the internal Hankel grid to the requested
            theta values.

        Args:
            theta: Theta values for interpolation (in radians).
            ell: ell grid for tabulated spectra.
            c_ell: Spectrum values or callable spectrum.
            order: Bessel order to use.
            grid_spacing: Interpolate in "linear" or "log" space.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Radial grid and projected radial statistic.
        """
        if grid_spacing == "linear":
            interp_func = interpolate_linear
        elif grid_spacing == "log":
            interp_func = interpolate_loglog
        else:
            raise ValueError("grid_spacing must be 'linear' or 'log'.")

        theta_grid, xi_grid = self.backend.projected_correlation(
            ell=ell, c_ell=c_ell, order=order, **kwargs
        )

        xi_out = interp_func(
            theta,
            theta_grid,
            xi_grid,
            x_name="theta",
            xp_name="theta_grid",
            fp_name="xi_grid",
        )

        return theta, np.asarray(xi_out, dtype=float)

    def spherical_correlation_interpolated(
        self,
        r: FloatArray,
        k_pk: FloatArray,
        pk: SpectrumInput,
        order: float | int = 0,
        grid_spacing: str = "linear",
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute a spherical radial statistic from one spectrum at given radial values.
            The statistic is interpolated from the internal Hankel grid to the requested
            radial values.

        Args:
            r: radial values for interpolation (in Mpc).
            k_pk: Wavenumber grid for tabulated spectra (in 1/Mpc).
            pk: Spectrum values or callable spectrum.
            order: Bessel order to use.
            grid_spacing: Interpolate in "linear" or "log" space.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Radial grid and spherical radial statistic.
        """
        if grid_spacing == "linear":
            interp_func = interpolate_linear
        elif grid_spacing == "log":
            interp_func = interpolate_loglog
        else:
            raise ValueError("grid_spacing must be 'linear' or 'log'.")

        r_grid, xi_grid = self.backend.spherical_correlation(
            k_pk=k_pk,
            pk=pk,
            order=order,
            **kwargs,
        )

        xi_out = interp_func(
            r,
            r_grid,
            xi_grid,
            x_name="r",
            xp_name="r_grid",
            fp_name="xi_grid",
        )

        return r, np.asarray(xi_out, dtype=float)

    def projected_covariance(
        self,
        k_pk: ArrayLike | None = None,
        pk1: SpectrumInput | None = None,
        pk2: SpectrumInput | None = None,
        order: float | int = 0,
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute a projected covariance matrix from two spectra.

        Args:
            k_pk: Wavenumber grid for tabulated spectra.
            pk1: First spectrum values or callable spectrum.
            pk2: Second spectrum values or callable spectrum.
            order: Bessel order to use.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Radial grid and projected covariance matrix (units of radians).
        """
        return self.backend.projected_covariance(
            k_pk=k_pk,
            pk1=pk1,
            pk2=pk2,
            order=order,
            **kwargs,
        )

    def bin_radial_matrix(
        self,
        r: ArrayLike,
        matrix: FloatArray,
        r_bins: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Average a radial matrix or tensor into radial bins.

        Args:
            r: Radial grid associated with each axis of ``matrix``.
            matrix: Radial matrix or tensor to bin.
            r_bins: Radial bin edges.

        Returns:
            Radial bin centers and binned matrix or tensor.
        """
        return self.backend.bin_radial_matrix(r=r, matrix=matrix, r_bins=r_bins)

    def correlation_matrix(self, covariance: FloatArray) -> FloatArray:
        """Return the correlation matrix associated with a covariance matrix.

        Args:
            covariance: Covariance matrix.

        Returns:
            Dimensionless correlation matrix.
        """
        return self.backend.correlation_matrix(covariance)

    def diagonal_error(self, covariance: FloatArray) -> FloatArray:
        """Return one-sigma errors from a covariance matrix.

        Args:
            covariance: Covariance matrix.

        Returns:
            Square root of the covariance diagonal.
        """
        return self.backend.diagonal_error(covariance)
