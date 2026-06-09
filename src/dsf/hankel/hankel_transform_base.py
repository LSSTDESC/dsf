"""DOCSTRING"""

from __future__ import annotations

from dsf.hankel.hankel_utils import apply_taper_spectrum
from dsf.utils.types import ArrayLike, FloatArray, SpectrumInput
from dsf.utils.validators import as_1d_float_array, validate_power_spectrum_inputs


class HankelTransformBase:
    """Base class for Hankel transforms."""

    def __init__(self) -> None:
        """Initialize the HankelTransformBase class."""
        pass

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

    def pk_grid(
        self,
        k_pk: ArrayLike | None = None,
        pk: SpectrumInput | None = None,
        order: float | int = 0,
        taper: bool = False,
        taper_kwargs: dict | None = None,
        **kwargs,
    ) -> FloatArray:
        """Return a power spectrum evaluated on a Hankel grid.

        Args:
            k_pk: Wavenumber grid for tabulated spectra.
            pk: Power-spectrum values or callable power spectrum.
            order: Bessel order to use.
            taper: Whether to suppress low-k and high-k edge power.
            taper_kwargs: Optional settings for the spectrum taper.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Power spectrum evaluated on the internal wavenumber grid.
        """
        if pk is None:
            raise ValueError("pk must be supplied.")

        return self._evaluate_spectrum(
            pk,
            order=order,
            k_input=k_pk,
            taper=taper,
            taper_kwargs=taper_kwargs,
            **kwargs,
        )
