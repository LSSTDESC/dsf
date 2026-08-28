"""Direct matrix Hankel transforms for projected radial statistics.

This module provides the ``HankelTransformMatrixDirect`` class, which converts
Fourier-space spectra into projected radial-space quantities using precomputed
Hankel operator grids along a predefined radial grid. It is useful for computing
projected correlation functions, covariance matrices, and higher-order
radial tensors that appear in weak-lensing and Delta Sigma calculations.

This class inherits from ``HankelTransformMatrixZeros`` in order to reuse the
public matrix API. However, the grid construction and projection methods are
overridden.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.special import jv

from dsf.hankel.hankel_transform_matrix_zeros import HankelTransformMatrixZeros
from dsf.utils.types import FloatArray
from dsf.utils.validators import as_2d_float_array, validate_positive_scalar


class HankelTransformMatrixDirect(HankelTransformMatrixZeros):
    """Project Fourier-space spectra into radial-space statistics.

    The class builds Bessel-function grids for the requested orders and uses
    them to evaluate projected radial quantities. A single input spectrum gives
    a projected correlation-like statistic, two spectra give a projected
    covariance matrix, and three spectra give a projected third-order radial
    tensor.

    The input r and k ranges must have reciprocal units such that
    :math:`r \\times k` is dimensionless.

    Args:
        r_min: Minimum radial scale to cover.
        r_max: Maximum radial scale to cover.
        k_min: Minimum wavenumber to cover.
        k_max: Maximum wavenumber to cover.
        n_r: Number of radial grid points to use for the Hankel transform.
        n_k: Number of wavenumber grid points to use for Hankel transform.
        orders: Bessel orders to precompute.
    """

    def __init__(
        self,
        r_min: float = 0.1,
        r_max: float = 100.0,
        k_min: float = 1.0e-4,
        k_max: float = 10.0,
        n_r: int = 1000,
        n_k: int = 1000,
        orders: Iterable[float | int] = (0, 2),
    ) -> None:
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.n_r = int(n_r)
        self.n_k = int(n_k)
        self.orders = tuple(orders)

        self._validate_init_inputs()
        self._init_cache()
        self._build_all_grids()

    def _init_cache(self) -> None:
        """Initialize storage for precomputed Hankel grids."""
        self.k: dict[float | int, FloatArray] = {}
        self.r: dict[float | int, FloatArray] = {}
        self.j: dict[float | int, FloatArray] = {}
        self.j_t: dict[float | int, FloatArray] = {}
        self.weights: dict[float | int, FloatArray] = {}

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
        if self.n_r <= 0:
            raise ValueError("n_r must be positive.")
        if self.n_k <= 0:
            raise ValueError("n_k must be positive.")

        self._validate_orders()

    def _build_grid(self, order: float | int) -> None:
        """Build the radial and wavenumber grid for one Bessel order.

        Args:
            order: Bessel order used by the projected statistic.
        """

        k = np.geomspace(self.k_min, self.k_max, self.n_k)
        dlnk = np.gradient(np.log(k))
        weight = k**2 * dlnk / (2.0 * np.pi)

        r = np.geomspace(self.r_min, self.r_max, self.n_r)
        j = jv(order, np.outer(r, k))
        j_t = j.T

        j = as_2d_float_array(j, "j")
        j_t = as_2d_float_array(j_t, "j_t")

        self.k[order] = k
        self.r[order] = r
        self.j[order] = j
        self.j_t[order] = j_t
        self.weights[order] = weight

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

        Raises:
            ValueError: If the number of spectra is not 1, 2, or 3.
        """
        self._check_order(order)

        product = np.ones_like(self.k[order])
        for spectrum in spectra:
            product *= spectrum

        weight = self.weights[order]
        j = self.j[order]
        j_t = self.j_t[order]

        ndim = len(spectra)

        if ndim == 1:
            transformed = j @ (weight * product)
        elif ndim == 2:
            transformed = (j * weight * product) @ j_t
        elif ndim == 3:
            transformed = j @ (weight * product) @ j_t @ j_t
        else:
            raise ValueError(f"Only 1, 2, or 3 spectra are supported. Got {ndim}.")

        return self.r[order], np.asarray(transformed, dtype=float)
