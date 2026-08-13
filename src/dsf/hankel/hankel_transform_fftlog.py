"""FFTLog-basedHankel transforms for 1D radial statistics.

This module provides the ``HankelTransformFFTLog`` class, which converts
Fourier-space spectra into radial-space quantities using the FFTLog
algorithm. It implements 1D projected and spherical Hankel transforms
as a ``HankelTransform`` backend. It also includes two public functions that
perform the FFTLog-based Hankel transforms directly, without requiring a
``HankelTransform`` class instance.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import fhtoffset, ifht

from dsf.hankel.hankel_transform_base import HankelTransformBase
from dsf.utils.types import ArrayLike, FloatArray, SpectrumInput
from dsf.utils.validators import (
    as_1d_float_array,
    validate_hankel_1d_grid_spacing,
    validate_power_spectrum_inputs,
)


class HankelTransformFFTLog(HankelTransformBase):
    """Hankel transform using the FFTLog algorithm.

    This class provides methods for performing 1D Hankel transforms
    using the scipy FFTLog algorithm.
    """

    def _check_order(self, order: float | int) -> None:
        """Validate that a requested Bessel order is available."""
        if order < 0:
            raise ValueError(f"Order {order} must be positive.")

    def _evaluate_spectrum(
        self,
        spectrum: SpectrumInput,
        order: float | int,
        k_input: ArrayLike | None = None,
        taper: bool = False,
        taper_kwargs: dict | None = None,
        **kwargs,
    ) -> FloatArray:
        """Evaluate a spectrum on the internal Hankel grid.

        Args:
            spectrum: Tabulated spectrum values or callable spectrum.
            order: Bessel order whose grid should be used.
            k_input: Wavenumber grid for tabulated spectra.
            taper: Whether to suppress low-k and high-k edge power.
            taper_kwargs: Optional settings for the spectrum taper.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Spectrum evaluated on the internal Hankel wavenumber grid.
        """
        if callable(spectrum):
            values = np.asarray(spectrum(k=k_input, **kwargs), dtype=float)
        else:
            values = as_1d_float_array(spectrum, "spectrum", min_size=2)
        if k_input is None:
            raise ValueError("k_input must be supplied.")

        if taper:
            taper_kwargs = {} if taper_kwargs is None else taper_kwargs
            values = self.taper_spectrum(
                k_input,
                values,
                **taper_kwargs,
            )

        if np.any(~np.isfinite(values)):
            raise ValueError("Evaluated spectrum must contain only finite values.")

        return values

    def projected_correlation(
        self,
        ell: ArrayLike | None = None,
        c_ell: SpectrumInput | None = None,
        order: float | int = 0,
        taper: bool = False,
        taper_kwargs: dict | None = None,
        use_offset: bool = False,
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
        if c_ell is None:
            raise ValueError("c_ell must be supplied.")

        c_ell_eval = self._evaluate_spectrum(
            c_ell,
            order=order,
            k_input=ell,
            taper=taper,
            taper_kwargs=taper_kwargs,
            **kwargs,
        )

        return hankel_projected(
            ell=ell, c_ell=c_ell_eval, order=order, use_offset=use_offset
        )

    def spherical_correlation(
        self,
        k_pk: ArrayLike | None = None,
        pk: SpectrumInput | None = None,
        order: float | int = 0,
        taper: bool = False,
        taper_kwargs: dict | None = None,
        use_offset: bool = False,
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute a spherical radial statistic from one spectrum.

        Args:
            k_pk: Wavenumber grid for tabulated spectra.
            pk: Spectrum values or callable spectrum.
            order: Bessel order to use.
            taper: Whether to suppress low-k and high-k edge power.
            taper_kwargs: Optional settings for the spectrum taper.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Radial grid and spherical radial statistic.
        """
        if pk is None:
            raise ValueError("pk must be supplied.")

        pk_eval = self._evaluate_spectrum(
            pk,
            order=order,
            k_input=k_pk,
            taper=taper,
            taper_kwargs=taper_kwargs,
            **kwargs,
        )

        return hankel_spherical(k=k_pk, pk=pk_eval, order=order, use_offset=use_offset)


def hankel_projected(
    ell: FloatArray,
    c_ell: FloatArray,
    order: float | int = 2,
    use_offset: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Convert projected GGL power spectrum to 2D correlation function using FFTLog:

    :math:`\\gamma_t(\\theta) = \\int \\frac{\\ell d\\ell}{2\\pi} C(\\ell )J_\\mu(\\ell \\theta)`.

    Args:
        ell: ell array (must be uniform in logspace).
        c_ell: Power spectrum to transform.
        order: Bessel order to use (default is 2 for tangential shear).
        use_offset: Optional flag to apply an offset to the logarithmic spacing
            of the output. Can reduce numerical ringing.

    Returns:
        Radial grid and projected radial statistic.
    """
    ell_arr = validate_hankel_1d_grid_spacing(ell, "ell")
    c_ell_arr = as_1d_float_array(c_ell, "c_ell", min_size=2)
    validate_power_spectrum_inputs(ell_arr, c_ell_arr, k_name="ell", pk_name="c_ell")

    theta = 1.0 / ell_arr[::-1]
    dln_ell = float(np.log(ell_arr[1] / ell_arr[0]))
    offset = fhtoffset(dln=dln_ell, mu=order) if use_offset else 0.0

    transformed_power = ifht(c_ell_arr * ell_arr, dln=dln_ell, mu=order, offset=offset)

    prefactor = 1.0 / (2.0 * np.pi * theta)
    xi = np.asarray(prefactor * transformed_power, dtype=float)

    return theta, xi


def hankel_spherical(
    k: FloatArray,
    pk: FloatArray,
    order: float | int = 0,
    use_offset: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Convert power spectrum to 3D correlation function using FFTLog:

    :math:`\\xi(r) = \\int \\frac{k^2 dk}{2\\pi^2} P(k) j_\\mu(kr)`.

    Args:
        k: Wavenumber array (must be uniform in logspace).
        pk: Power spectrum to transform.
        order: Bessel order to use (default is 0 for density).
        use_offset: Optional flag to apply an offset to the logarithmic spacing
            of the output. Can reduce numerical ringing.

    Returns:
        Radial grid and 3D correlation function.
    """
    if order != 0:
        raise NotImplementedError(
            "Only order 0 is currently implemented for spherical Hankel transforms."
        )
    k_arr = validate_hankel_1d_grid_spacing(k, "k")
    pk_arr = as_1d_float_array(pk, "pk", min_size=2)
    validate_power_spectrum_inputs(k_arr, pk_arr)

    r = 1.0 / k_arr[::-1]
    dln_k = float(np.log(k_arr[1] / k_arr[0]))
    offset = fhtoffset(dln=dln_k, mu=0.5) if use_offset else 0.0

    transformed_power = ifht(
        k_arr**1.5 * pk_arr,
        dln=dln_k,
        mu=0.5,
        offset=offset,
    )

    prefactor = 1.0 / (2.0 * np.pi * r) ** 1.5
    xi = np.asarray(prefactor * transformed_power, dtype=float)

    return r, xi
