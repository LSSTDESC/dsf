"""Shared base interface for Hankel transform backends.

This module defines the common interface used by the FFTLog, direct matrix,
and zero-crossing matrix Hankel transform implementations. The base class
provides shared helpers for validating and tapering input spectra, while
backend-specific subclasses implement the projected, spherical, covariance,
and higher-order radial transforms.

Projected transforms are intended for flat-sky spectra on angular wavenumber
grids, while spherical transforms are intended for three-dimensional power
spectra on reciprocal radial wavenumber grids. Subclasses are expected to
document their exact normalization conventions and output-grid units.

Unsupported operations should raise ``NotImplementedError`` with a clear
message.
"""

from __future__ import annotations

from dsf.hankel.hankel_utils import apply_taper_spectrum
from dsf.utils.types import ArrayLike, FloatArray, SpectrumInput
from dsf.utils.validators import as_1d_float_array, validate_power_spectrum_inputs


class HankelTransformBase:
    """Base class for Hankel transforms."""

    def taper_spectrum(
        self,
        k: ArrayLike,
        pk: ArrayLike,
        **kwargs,
    ) -> FloatArray:
        """Return a smoothly tapered power spectrum.

        Args:
            k: Wavenumber grid.
            pk: Power-spectrum values evaluated on ``k``.
            **kwargs: Optional taper settings.

        Returns:
            Power spectrum with smooth low-k and high-k suppression.
        """
        k_arr = as_1d_float_array(k, "k")
        pk_arr = as_1d_float_array(pk, "pk")

        validate_power_spectrum_inputs(k_arr, pk_arr)

        return apply_taper_spectrum(k_arr, pk_arr, **kwargs)

    def power_grid(
        self,
        radial_grid: ArrayLike | None = None,
        power_spectrum: SpectrumInput | None = None,
        order: float | int = 0,
        taper: bool = False,
        taper_kwargs: dict | None = None,
        **kwargs,
    ) -> FloatArray:
        """Return a power spectrum evaluated on a Hankel grid.

        Args:
            radial_grid: Radial grid for tabulated spectra (k or ell).
            power_spectrum: Power-spectrum values or callable power spectrum.
            order: Bessel order used to select the backend k grid.
            taper: Whether to suppress low-k and high-k edge power.
            taper_kwargs: Optional settings for the spectrum taper.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Power spectrum evaluated on the internal radial grid.
        """
        if power_spectrum is None:
            raise ValueError("power_spectrum must be supplied.")

        if radial_grid is None:
            raise ValueError("radial_grid must be supplied.")

        return self._evaluate_spectrum(
            power_spectrum,
            order=order,
            radial_input=radial_grid,
            taper=taper,
            taper_kwargs=taper_kwargs,
            **kwargs,
        )

    def _evaluate_spectrum(
        self,
        spectrum: SpectrumInput,
        *,
        order: float | int = 0,
        radial_input: ArrayLike | None = None,
        **kwargs,
    ) -> FloatArray:
        """Evaluate a tabulated or callable spectrum on the backend grid."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support _evaluate_spectrum()."
        )

    def projected_correlation(
        self,
        ell: ArrayLike | None = None,
        c_ell: SpectrumInput | None = None,
        order: float | int = 0,
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute a projected radial statistic from one spectrum:

        :math:`\\gamma_t(\\theta) = \\int \\frac{\\ell d\\ell}{2\\pi} C(\\ell )J_\\mu(\\ell \\theta)`.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support projected_correlation()."
        )

    def spherical_correlation(
        self,
        k_pk: ArrayLike | None = None,
        pk: SpectrumInput | None = None,
        order: float | int = 0,
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute a spherical radial statistic from one spectrum:

        :math:`\\xi(r) = \\int \\frac{k^2 dk}{2\\pi^2} P(k) j_\\mu(kr)`.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support spherical_correlation()."
        )

    def projected_covariance(
        self,
        k_pk: ArrayLike | None = None,
        pk1: SpectrumInput | None = None,
        pk2: SpectrumInput | None = None,
        order: float | int = 0,
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Project two spectra into a radial covariance matrix."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support projected_covariance()."
        )

    def bin_radial_matrix(
        self,
        r: ArrayLike,
        matrix: FloatArray,
        r_bins: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Average a radial matrix or tensor into radial bins."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support bin_radial_matrix()."
        )

    def correlation_matrix(self, covariance: FloatArray) -> FloatArray:
        """Return the correlation matrix associated with a covariance matrix.

        Args:
            covariance: 2D square covariance matrix.

        Raises:
            NotImplementedError: If the backend does not support correlation matrices.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support correlation_matrix()."
        )

    def diagonal_error(self, covariance: FloatArray) -> FloatArray:
        """Return one-sigma errors from a covariance matrix.

        Args:
            covariance: 2D square covariance matrix.

        Raises:
            NotImplementedError: If the backend does not support diagonal errors.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support diagonal_error()."
        )

    def _validate_orders(self) -> None:
        """Validate the requested Bessel orders."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support _validate_orders()."
        )
