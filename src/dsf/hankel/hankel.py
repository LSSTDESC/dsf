"""Public layer for all Hankel transform operations.

This module provides the ``HankelTransform`` class, an interface for
the ``HankelTransformFFTLog``, ``HankelTransformMatrixDirect``, and
``HankelTransformMatrixZeros`` classes. It provides public methods to
compute projected and spherical correlation functions, as well as covariance
matricies, from these underlying Hankel transform implementations.

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

    def __init__(self, backend="fftlog", **kwargs) -> None:
        """Initialize the HankelTransform class."""
        if backend == "fftlog":
            self.backend = HankelTransformFFTLog(**kwargs)
        elif backend == "matrix_zeros":
            self.backend = HankelTransformMatrixZeros(**kwargs)
        elif backend == "matrix_direct":
            self.backend = HankelTransformMatrixDirect(**kwargs)
        else:
            raise ValueError(
                f"Unsupported backend '{backend}'. Use 'fftlog', 'matrix_zeros', or 'matrix_direct'."
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

    def projected_covariance(
        self,
        k_pk: ArrayLike | None = None,
        pk1: SpectrumInput | None = None,
        pk2: SpectrumInput | None = None,
        order: float | int = 0,
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute a projected third-order radial tensor from three spectra.

        Args:
            k_pk: Wavenumber grid for tabulated spectra.
            pk1: First spectrum values or callable spectrum.
            pk2: Second spectrum values or callable spectrum.
            pk3: Third spectrum values or callable spectrum.
            order: Bessel order to use.
            taper: Whether to suppress low-k and high-k edge power.
            taper_kwargs: Optional settings for the spectrum taper.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Radial grid and projected third-order radial tensor.
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
