"""Hankel transforms for projected radial statistics.

This module provides the ``HankelTransform`` class, which converts
Fourier-space spectra into projected radial-space quantities. It is useful for
computing projected correlation functions, covariance matrices, and higher-order
radial tensors that appear in weak-lensing and Delta Sigma calculations.

The class owns the public validation layer and delegates low-level radial and
Bessel operations to ``hankel_utils``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.special import jv

from dsf.covariance.projection.hankel_utils import (
    apply_taper_spectrum,
    bessel_zeros,
    compute_bin_radial_matrix,
    compute_correlation_matrix,
    compute_diagonal_error,
)
from dsf.utils.types import ArrayLike, FloatArray, SpectrumInput
from dsf.utils.validators import (
    as_1d_float_array,
    validate_1d_pair,
    validate_positive_scalar,
    validate_power_spectrum_inputs,
    validate_strictly_increasing,
)


class HankelTransform:
    """Project Fourier-space spectra into radial-space statistics.

    The class builds Bessel-function grids for the requested orders and uses
    them to evaluate projected radial quantities. A single input spectrum gives
    a projected correlation-like statistic, two spectra give a projected
    covariance matrix, and three spectra give a projected third-order radial
    tensor.

    Args:
        r_min: Minimum radial scale to cover.
        r_max: Maximum radial scale to cover.
        k_min: Minimum wavenumber to cover.
        k_max: Maximum wavenumber to cover.
        orders: Bessel orders to precompute.
        n_zeros: Initial number of Bessel roots used for each grid.
        n_zeros_step: Number of extra Bessel roots added if the requested range
            is not covered.
        prune_r: Optional radial-grid pruning factor. Use ``None`` or ``0`` to
            keep the full grid.
        prune_log_space: Whether pruning should keep approximately logarithmic
            radial spacing.
        verbose: Whether to print grid-construction diagnostics.
        max_iterations: Maximum number of grid-construction attempts per order.
    """

    def __init__(
        self,
        r_min: float = 0.1,
        r_max: float = 100.0,
        k_min: float = 1.0e-4,
        k_max: float = 10.0,
        orders: Iterable[float | int] = (0, 2),
        n_zeros: int = 1000,
        n_zeros_step: int = 1000,
        prune_r: int | None = None,
        prune_log_space: bool = True,
        verbose: bool = False,
        max_iterations: int = 100,
    ) -> None:
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.orders = tuple(orders)
        self.n_zeros = int(n_zeros)
        self.n_zeros_step = int(n_zeros_step)
        self.prune_r = prune_r
        self.prune_log_space = bool(prune_log_space)
        self.verbose = bool(verbose)
        self.max_iterations = int(max_iterations)

        self._validate_init_inputs()
        self._init_cache()
        self._build_all_grids()

    def _init_cache(self) -> None:
        """Initialize storage for precomputed Hankel grids."""
        self.k: dict[float | int, FloatArray] = {}
        self.r: dict[float | int, FloatArray] = {}
        self.j: dict[float | int, FloatArray] = {}
        self.j_next_at_zeros: dict[float | int, FloatArray] = {}
        self.zeros: dict[float | int, FloatArray] = {}
        self.normalization: dict[float | int, float] = {}

    def _build_all_grids(self) -> None:
        """Build Hankel grids for all requested Bessel orders."""
        for order in self.orders:
            self._build_grid(order)

    def _validate_init_inputs(self) -> None:
        """Validate the radial and Fourier ranges used by the transform."""
        validate_positive_scalar(self.r_min, "r_min")
        validate_positive_scalar(self.r_max, "r_max")
        validate_positive_scalar(self.k_min, "k_min")
        validate_positive_scalar(self.k_max, "k_max")

        if self.r_max <= self.r_min:
            raise ValueError("r_max must be larger than r_min.")
        if self.k_max <= self.k_min:
            raise ValueError("k_max must be larger than k_min.")
        if self.n_zeros <= 0:
            raise ValueError("n_zeros must be positive.")
        if self.n_zeros_step <= 0:
            raise ValueError("n_zeros_step must be positive.")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")

        self._validate_orders()

    def _validate_orders(self) -> None:
        """Validate the requested Bessel orders."""
        if len(self.orders) == 0:
            raise ValueError("orders must contain at least one Bessel order.")

        for order in self.orders:
            if not np.isfinite(order):
                raise ValueError("orders must contain only finite values.")
            if order < 0:
                raise ValueError("orders must contain non-negative Bessel orders.")

    def _log(self, message: str) -> None:
        """Print grid diagnostics when verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _build_grid(self, order: float | int) -> None:
        """Build the radial and wavenumber grid for one Bessel order.

        Args:
            order: Bessel order used by the projected statistic.
        """
        n_zeros = self.n_zeros
        k_max = self.k_max

        for _ in range(self.max_iterations):
            zeros = bessel_zeros(order, n_zeros)
            k = zeros / zeros[-1] * k_max
            r = zeros / k_max

            if np.min(r) > self.r_min:
                k_max = zeros[0] / self.r_min
                self._log(f"order={order}: changed k_max to {k_max:g}")
                continue

            if np.max(r) < self.r_max:
                n_zeros += self.n_zeros_step
                self._log(f"order={order}: increasing n_zeros to {n_zeros} to cover r_max")
                continue

            if np.min(k) > self.k_min:
                n_zeros += self.n_zeros_step
                self._log(f"order={order}: increasing n_zeros to {n_zeros} to cover k_min")
                continue

            break
        else:
            raise RuntimeError(
                f"Failed to build Hankel grid for order={order} within "
                f"{self.max_iterations} iterations."
            )

        r_selected = self._select_radial_range(r)
        r_selected = self._prune_radial_grid(r_selected)
        r_selected = np.unique(r_selected)

        j_matrix = np.asarray(jv(order, np.outer(r_selected, k)), dtype=float)
        j_next = np.asarray(jv(order + 1, zeros), dtype=float)
        norm = (2.0 * k_max**2 / zeros[-1] ** 2) / (2.0 * np.pi)

        self.k[order] = k
        self.r[order] = r_selected
        self.j[order] = j_matrix
        self.j_next_at_zeros[order] = j_next
        self.zeros[order] = zeros
        self.normalization[order] = float(norm)

        self._log(
            f"order={order}: built grid with nr={r_selected.size}, "
            f"nk={k.size}, r=[{r_selected[0]:g}, {r_selected[-1]:g}], "
            f"k=[{k[0]:g}, {k[-1]:g}]"
        )

    def _select_radial_range(self, r: FloatArray) -> FloatArray:
        """Select radial grid values spanning the requested radial range.

        Args:
            r: Candidate radial grid.

        Returns:
            Radial grid covering ``r_min`` through ``r_max``.
        """
        lower = r[r <= self.r_min]
        upper = r[r >= self.r_max]

        if lower.size == 0 or upper.size == 0:
            raise RuntimeError("Radial grid does not cover requested range.")

        r_min_effective = lower[-1]
        r_max_effective = upper[0]
        mask = (r >= r_min_effective) & (r <= r_max_effective)

        return r[mask]

    def _prune_radial_grid(self, r: FloatArray) -> FloatArray:
        """Return a reduced radial grid when pruning is requested.

        Args:
            r: Radial grid before pruning.

        Returns:
            Original or pruned radial grid.
        """
        if self.prune_r in (None, 0):
            return r

        prune_r = int(self.prune_r)
        if prune_r <= 0:
            raise ValueError("prune_r must be positive, None, or 0.")

        n = r.size
        if n <= 2:
            return r

        if self.prune_log_space:
            n_keep = max(2, n // prune_r)
            indices = np.asarray(
                np.logspace(0.0, np.log10(n - 1), n_keep),
                dtype=int,
            )
            indices = np.unique(np.append([0], indices))
        else:
            indices = np.arange(0, n, step=prune_r, dtype=int)

        indices = np.unique(np.append(indices, [n - 1]))

        return r[indices]

    def available_orders(self) -> tuple[float | int, ...]:
        """Return the Bessel orders available for projection."""
        return tuple(self.k.keys())

    def _check_order(self, order: float | int) -> None:
        """Validate that a requested Bessel order is available."""
        if order not in self.k:
            raise ValueError(
                f"Order {order} was not precomputed. "
                f"Available orders are {self.available_orders()}."
            )

    def _evaluate_tabulated_spectrum(
        self,
        k_input: ArrayLike,
        spectrum: ArrayLike,
        order: float | int,
        taper: bool = False,
        taper_kwargs: dict | None = None,
    ) -> FloatArray:
        """Evaluate a tabulated spectrum on the internal Hankel grid.

        Args:
            k_input: Input wavenumber grid.
            spectrum: Spectrum values evaluated on ``k_input``.
            order: Bessel order whose grid should be used.
            taper: Whether to suppress low-k and high-k edge power.
            taper_kwargs: Optional settings for the spectrum taper.

        Returns:
            Spectrum evaluated on the internal Hankel wavenumber grid.
        """
        k_arr = as_1d_float_array(k_input, "k_input")
        spectrum_arr = as_1d_float_array(spectrum, "spectrum")

        validate_1d_pair(
            k_arr,
            spectrum_arr,
            x_name="k_input",
            y_name="spectrum",
        )
        validate_power_spectrum_inputs(
            k_arr,
            spectrum_arr,
            k_name="k_input",
            pk_name="spectrum",
        )

        if taper:
            taper_kwargs = {} if taper_kwargs is None else taper_kwargs
            spectrum_arr = apply_taper_spectrum(
                k_arr,
                spectrum_arr,
                **taper_kwargs,
            )

        return np.asarray(
            np.interp(
                self.k[order],
                k_arr,
                spectrum_arr,
                left=0.0,
                right=0.0,
            ),
            dtype=float,
        )

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
        self._check_order(order)
        target_k = self.k[order]

        if callable(spectrum):
            values = np.asarray(spectrum(k=target_k, **kwargs), dtype=float)
        else:
            if k_input is None:
                raise ValueError("k_input must be supplied for tabulated spectra.")

            values = self._evaluate_tabulated_spectrum(
                k_input=k_input,
                spectrum=spectrum,
                order=order,
                taper=taper,
                taper_kwargs=taper_kwargs,
            )

        if values.shape != target_k.shape:
            raise ValueError(
                "Evaluated spectrum must match the internal k-grid shape. "
                f"Got {values.shape}, expected {target_k.shape}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("Evaluated spectrum must contain only finite values.")

        return values

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

    def _project_spectra_to_radial(
        self,
        spectra: list[FloatArray],
        order: float | int,
    ) -> tuple[FloatArray, FloatArray]:
        """Project one or more spectra into radial-space statistics.

        Args:
            spectra: Spectra evaluated on the internal wavenumber grid.
            order: Bessel order to use.

        Returns:
            Radial grid and projected radial statistic.
        """
        self._check_order(order)

        product = np.ones_like(self.k[order])
        for spectrum in spectra:
            product *= spectrum

        weighted = product / self.j_next_at_zeros[order] ** 2
        j_matrix = self.j[order]
        norm = self.normalization[order]
        ndim = len(spectra)

        if ndim == 1:
            transformed = np.dot(j_matrix, weighted) * norm
        elif ndim == 2:
            transformed = np.dot(j_matrix, (j_matrix * weighted).T) * norm
        elif ndim == 3:
            transformed = (
                np.einsum(
                    "az,bz,cz,z->abc",
                    j_matrix,
                    j_matrix,
                    j_matrix,
                    weighted,
                )
                * norm
            )
        else:
            raise ValueError(f"Only 1, 2, or 3 spectra are supported. Got {ndim}.")

        return self.r[order], np.asarray(transformed, dtype=float)

    def projected_correlation(
        self,
        k_pk: ArrayLike | None = None,
        pk: SpectrumInput | None = None,
        order: float | int = 0,
        taper: bool = False,
        taper_kwargs: dict | None = None,
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute a projected radial statistic from one spectrum.

        Args:
            k_pk: Wavenumber grid for tabulated spectra.
            pk: Spectrum values or callable spectrum.
            order: Bessel order to use.
            taper: Whether to suppress low-k and high-k edge power.
            taper_kwargs: Optional settings for the spectrum taper.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Radial grid and projected radial statistic.
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

        return self._project_spectra_to_radial([pk_eval], order)

    def spherical_correlation(
        self,
        k_pk: ArrayLike | None = None,
        pk: SpectrumInput | None = None,
        order: float | int = 0,
        taper: bool = False,
        taper_kwargs: dict | None = None,
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute the k-weighted projected radial statistic.

        Args:
            k_pk: Wavenumber grid for tabulated spectra.
            pk: Spectrum values or callable spectrum.
            order: Bessel order to use.
            taper: Whether to suppress low-k and high-k edge power.
            taper_kwargs: Optional settings for the spectrum taper.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Radial grid and k-weighted projected radial statistic.
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

        return self._project_spectra_to_radial(
            [pk_eval * self.k[order]],
            order,
        )

    def projected_covariance(
        self,
        k_pk: ArrayLike | None = None,
        pk1: SpectrumInput | None = None,
        pk2: SpectrumInput | None = None,
        order: float | int = 0,
        taper: bool = False,
        taper_kwargs: dict | None = None,
        **kwargs,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute a projected covariance matrix from two spectra.

        Args:
            k_pk: Wavenumber grid for tabulated spectra.
            pk1: First spectrum values or callable spectrum.
            pk2: Second spectrum values or callable spectrum.
            order: Bessel order to use.
            taper: Whether to suppress low-k and high-k edge power.
            taper_kwargs: Optional settings for the spectrum taper.
            **kwargs: Extra arguments passed to callable spectra.

        Returns:
            Radial grid and projected covariance matrix.
        """
        if pk1 is None or pk2 is None:
            raise ValueError("pk1 and pk2 must both be supplied.")

        pk1_eval = self._evaluate_spectrum(
            pk1,
            order=order,
            k_input=k_pk,
            taper=taper,
            taper_kwargs=taper_kwargs,
            **kwargs,
        )
        pk2_eval = self._evaluate_spectrum(
            pk2,
            order=order,
            k_input=k_pk,
            taper=taper,
            taper_kwargs=taper_kwargs,
            **kwargs,
        )

        return self._project_spectra_to_radial([pk1_eval, pk2_eval], order)

    def projected_skewness(
        self,
        k_pk: ArrayLike | None = None,
        pk1: SpectrumInput | None = None,
        pk2: SpectrumInput | None = None,
        pk3: SpectrumInput | None = None,
        order: float | int = 0,
        taper: bool = False,
        taper_kwargs: dict | None = None,
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
        if pk1 is None or pk2 is None or pk3 is None:
            raise ValueError("pk1, pk2, and pk3 must all be supplied.")

        spectra = [
            self._evaluate_spectrum(
                pk,
                order=order,
                k_input=k_pk,
                taper=taper,
                taper_kwargs=taper_kwargs,
                **kwargs,
            )
            for pk in (pk1, pk2, pk3)
        ]

        return self._project_spectra_to_radial(spectra, order)

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
        r_arr = as_1d_float_array(r, "r")
        r_bins_arr = as_1d_float_array(r_bins, "r_bins")
        matrix_arr = np.asarray(matrix, dtype=float)

        validate_strictly_increasing(r_arr, "r")
        validate_strictly_increasing(r_bins_arr, "r_bins")

        if np.any(r_bins_arr <= 0.0):
            raise ValueError("r_bins must contain only positive values.")
        if matrix_arr.ndim == 0:
            raise ValueError("matrix must have at least one dimension.")
        if np.any(~np.isfinite(matrix_arr)):
            raise ValueError("matrix must contain only finite values.")

        expected_shape = tuple([r_arr.size] * matrix_arr.ndim)
        if matrix_arr.shape != expected_shape:
            raise ValueError(f"matrix shape must be {expected_shape}. Got {matrix_arr.shape}.")

        return compute_bin_radial_matrix(r_arr, matrix_arr, r_bins_arr)

    def correlation_matrix(self, covariance: FloatArray) -> FloatArray:
        """Return the correlation matrix associated with a covariance matrix.

        Args:
            covariance: Covariance matrix.

        Returns:
            Dimensionless correlation matrix.
        """
        covariance_arr = np.asarray(covariance, dtype=float)

        if covariance_arr.ndim != 2:
            raise ValueError("covariance must be two-dimensional.")
        if covariance_arr.shape[0] != covariance_arr.shape[1]:
            raise ValueError("covariance must be square.")
        if np.any(~np.isfinite(covariance_arr)):
            raise ValueError("covariance must contain only finite values.")

        return compute_correlation_matrix(covariance_arr)

    def diagonal_error(self, covariance: FloatArray) -> FloatArray:
        """Return one-sigma errors from a covariance matrix.

        Args:
            covariance: Covariance matrix.

        Returns:
            Square root of the covariance diagonal.
        """
        covariance_arr = np.asarray(covariance, dtype=float)

        if covariance_arr.ndim != 2:
            raise ValueError("covariance must be two-dimensional.")
        if covariance_arr.shape[0] != covariance_arr.shape[1]:
            raise ValueError("covariance must be square.")
        if np.any(~np.isfinite(covariance_arr)):
            raise ValueError("covariance must contain only finite values.")

        return compute_diagonal_error(covariance_arr)

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
