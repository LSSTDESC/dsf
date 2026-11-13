r"""Forecast-ready Delta Sigma inputs.

This module provides the high-level builder used to prepare Delta Sigma forecast
inputs for Fisher or DALI analyses. It connects three pieces of the forecast:

- tomographic lens and source samples,
- the stacked Delta Sigma data vector,
- the covariance matrix matched to the selected lens-source bin pairs.

The builder is intentionally model-agnostic. It assumes that the data vector is
computed with ``DeltaSigmaCalculator``, but it does not assume a fixed physical
parameterization. Users provide a function that maps a parameter vector into the
cosmology and profile parameters used by the Delta Sigma calculation.

This makes the class suitable for simple amplitude-only forecasts, cosmology
forecasts, HOD or SHMR parameter forecasts, and later Fisher/DALI comparisons.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pyccl as ccl
from derivkit import ForecastKit

from dsf.covariance.cov_builder import DeltaSigmaCovarianceBuilder
from dsf.data_vector.delta_sigma_builder import DeltaSigmaCalculator
from dsf.tomography.tomo_builder import TomographyBuilder
from dsf.utils.types import ArrayLike, FloatArray, ScalarOrPerBin, ThetaMapper
from dsf.utils.validators import (
    validate_forecast_vector_and_covariance,
    validate_parameter_names,
)

__all__ = [
    "DeltaSigmaForecastBuilder",
]


class DeltaSigmaForecastBuilder:
    """Prepare Delta Sigma data vectors and covariance inputs for forecasts.

    This class is the user-facing forecast layer for Delta Sigma analyses. It
    builds the tomographic samples, selects valid lens-source bin pairs,
    constructs the matching covariance, and exposes a model function that can be
    passed directly to forecast tools.

    The current forecast data vector contains only the galaxy-matter
    Delta Sigma signal, so this builder currently supports only the matching
    ``gm`` covariance block. The covariance layer also contains helpers for
    ``gg`` and joint covariance blocks, which are intended for future forecast
    builders once the corresponding ``gg`` or joint data vectors are added.
    """

    def __init__(
        self,
        *,
        cosmo: ccl.Cosmology,
        pk2d_func: Callable[..., dict[str, ccl.Pk2D]],
        theta0: Sequence[float],
        r: ArrayLike,
        rp_bin_edges: ArrayLike,
        area_deg2: float,
        sigma_e: ScalarOrPerBin = 0.26,
        pk2d_kwargs: Mapping[str, Any] | None = None,
        theta_mapper: ThetaMapper | None = None,
        parameter_names: Sequence[str] | None = None,
        tomography_kwargs: Mapping[str, Any] | None = None,
        covariance_kwargs: Mapping[str, Any] | None = None,
        trim_edge_points: int = 0,
        verbose: bool = True,
    ) -> None:
        r"""Set the forecast configuration.

        Args:
            cosmo: Fiducial CCL cosmology. This is used as the default
                cosmology for the data vector and covariance unless a
                ``theta_mapper`` supplies a parameter-dependent cosmology.
            pk2d_func: Function that returns a dictionary of the ``Pk2D``
                objects used by the Delta Sigma and covariance models. Should
                have keys ``pk_deltasigma``, ``pk_gm``, and ``pk_gg``, which
                correspond to the Delta Sigma, galaxy-matter, and galaxy-galaxy
                power spectra, respectively.
            theta0: Fiducial parameter vector used as the expansion point for
                forecast calculations.
            r: Projected radii where the Delta Sigma data vector is evaluated.
            rp_bin_edges: Projected-radius bin edges used when constructing the
                covariance.
            area_deg2: Survey overlap area in square degrees.
            sigma_e: Source ellipticity dispersion used for the shape-noise
                contribution to the covariance.
            pk2d_kwargs: Optional fixed keyword arguments passed to the
                power-spectrum model.
            theta_mapper: Optional function mapping a parameter vector and
                forecast context into model inputs. If omitted, parameter names
                are mapped directly to power-spectrum keyword arguments.
                If supplied, the mapper may override ``cosmo`` and
                ``pk2d_kwargs``.
            parameter_names: Optional names for the entries of ``theta0``.
            tomography_kwargs: Optional keyword arguments passed to
                ``TomographyBuilder``, including ``lens`` and ``source``
                configurations. Defaults are defined by ``TomographyBuilder``.
            covariance_kwargs: Optional keyword arguments passed to
                ``DeltaSigmaCovarianceBuilder``.
            trim_edge_points: Number of edge points removed from each lens-bin
                redshift distribution before averaging Delta Sigma.
            verbose: Whether progress messages are printed while preparing the
                forecast.
        """
        self.cosmo = cosmo
        self.pk2d_func = pk2d_func
        self.pk2d_kwargs = dict(pk2d_kwargs or {})
        self.calculator = DeltaSigmaCalculator(
            lambda **kwargs: pk2d_func(**kwargs)["pk_deltasigma"]
        )
        self.theta0 = np.asarray(theta0, dtype=float)
        self.parameter_names = validate_parameter_names(parameter_names, self.theta0)
        self.theta_mapper = theta_mapper

        self.r = np.asarray(r, dtype=float)
        self.rp_bin_edges = np.asarray(rp_bin_edges, dtype=float)

        self.area_deg2 = float(area_deg2)
        self.sigma_e = sigma_e
        self.trim_edge_points = int(trim_edge_points)

        self.verbose = bool(verbose)

        self.tomography_kwargs = dict(tomography_kwargs or {})

        self.covariance_kwargs = dict(covariance_kwargs or {})

        self._forecast: dict[str, Any] | None = None

    def _log(self, message: str) -> None:
        """Print a progress message when verbose output is enabled."""
        if self.verbose:
            print(message)

    def prepare(self, covariance: FloatArray | None = None) -> dict[str, Any]:
        """Build and cache all forecast products.

        Args:
            covariance: Optional precomputed covariance matrix. If provided, the
                builder reuses this matrix instead of recomputing the covariance
                block. Tomography, context, and the fiducial data vector are still
                rebuilt so the model function remains matched to the selected
                lens-source bin pairs.

        Returns:
            Dictionary containing the model callable, fiducial parameters,
            covariance matrix, fiducial data vector, selected bin pairs,
            tomography products, covariance builder, Delta Sigma calculator, and
            shared forecast context.

        Notes:
            Calling this method explicitly recomputes the forecast products and
            refreshes the cached result. Use ``forecast()`` when you want the
            cached products if they already exist.
        """
        self._log("[1/5] Building lens and source tomography...")

        model_inputs = self._model_inputs(self.theta0, context={})
        fiducial_cosmo = model_inputs["cosmo"]

        pk2d_spectra = self.pk2d_func(
            cosmo=fiducial_cosmo,
            **model_inputs["pk2d_kwargs"],
        )

        tomography = TomographyBuilder(**self.tomography_kwargs).prepare_bins()

        self._log(
            "[1/5] Done: tomography built "
            f"with {len(tomography['bin_pairs'])} selected lens-source pairs."
        )
        self._log(f"Selected bin pairs: {tomography['bin_pairs']}")

        self._log("[2/5] Initializing Delta Sigma covariance builder...")

        cov_builder = DeltaSigmaCovarianceBuilder(
            cosmo=fiducial_cosmo,
            tomography=tomography,
            rp_bin_edges=self.rp_bin_edges,
            area_deg2=self.area_deg2,
            sigma_e=self.sigma_e,
            pk_gg=pk2d_spectra["pk_gg"],
            pk_gm=pk2d_spectra["pk_gm"],
            **self.covariance_kwargs,
        )

        self._log("[2/5] Done: covariance builder initialized.")

        data_pairs = [
            (int(lens_bin), int(source_bin)) for lens_bin, source_bin in tomography["bin_pairs"]
        ]

        if covariance is None:
            self._log("[3/5] Computing gm block-diagonal covariance...")

            cov_pairs, cov = cov_builder.gm_block_diagonal(bin_pairs=data_pairs)
            cov_pairs = list(cov_pairs)
            cov = np.asarray(cov, dtype=float)

            if list(cov_pairs) != data_pairs:
                raise RuntimeError(
                    "Covariance bin-pair ordering does not match the data-vector "
                    "bin-pair ordering. The data vector and covariance must use the "
                    "same selected lens-source pairs in the same order."
                )

            self._log(f"[3/5] Done: covariance computed with shape {np.shape(cov)}.")
        else:
            self._log("[3/5] Reusing supplied covariance matrix...")

            cov = np.asarray(covariance, dtype=float)
            cov_pairs = data_pairs

            if cov.ndim != 2:
                raise ValueError(
                    f"Supplied covariance must be a 2D matrix, but got shape {cov.shape}."
                )

            if cov.shape[0] != cov.shape[1]:
                raise ValueError(f"Supplied covariance must be square, but got shape {cov.shape}.")

            if not np.all(np.isfinite(cov)):
                raise ValueError("Supplied covariance contains non-finite values.")

            self._log(f"[3/5] Done: supplied covariance has shape {np.shape(cov)}.")

        self._log("[4/5] Preparing shared forecast context...")

        context = {
            "cosmo": fiducial_cosmo,
            "calculator": self.calculator,
            "r": self.r,
            "rp_bin_edges": self.rp_bin_edges,
            "tomography": tomography,
            "covariance_builder": cov_builder,
            "cov_pairs": list(cov_pairs),
            "covariance_kind": "gm",
            "observable": "delta_sigma",
        }

        self._log("[4/5] Done: forecast context prepared.")

        self._log(
            "[5/5] Computing fiducial Delta Sigma data vector in selected lens-source pair order..."
        )

        data_vector = self.model(self.theta0, context=context)
        data_vector, cov = validate_forecast_vector_and_covariance(data_vector, cov)

        self._log(f"[5/5] Done: fiducial data vector computed with length {len(data_vector)}.")

        forecast: dict[str, Any] = {
            **context,
            "lens_result": tomography["lens_result"],
            "source_result": tomography["source_result"],
            "bin_pairs": list(tomography["bin_pairs"]),
            "lens_bin_centers": tomography["lens_bin_centers"],
            "source_bin_centers": tomography["source_bin_centers"],
            "lens_shape_stats": tomography["lens_shape_stats"],
            "source_shape_stats": tomography["source_shape_stats"],
            "model": self.model,
            "theta0": self.theta0,
            "parameter_names": self.parameter_names,
            "cov": cov,
            "data_vector": data_vector,
            "fiducial_data_vector": data_vector,
            "lens_population_stats": tomography["lens_population_stats"],
            "source_population_stats": tomography["source_population_stats"],
            "context": context,
        }

        self._forecast = forecast

        self._log("Forecast preparation complete.")

        return forecast

    def forecast(self, covariance: FloatArray | None = None) -> dict[str, Any]:
        """Return cached forecast products, preparing them if needed."""
        if self._forecast is not None:
            if covariance is not None:
                raise RuntimeError(
                    "A forecast is already cached. Create a new "
                    "DeltaSigmaForecastBuilder instance before supplying a new "
                    "covariance matrix."
                )
            return self._forecast

        return self.prepare(covariance=covariance)

    def model(
        self,
        theta: ArrayLike,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> FloatArray:
        """Return the stacked Delta Sigma data vector for one parameter vector.

        Args:
            theta: Parameter vector at which the Delta Sigma model is evaluated.
            context: Optional forecast context. If omitted, the cached forecast
                context is used.

        Returns:
            One-dimensional stacked data vector containing the Delta Sigma blocks
            for all selected lens-source bin pairs.

        Notes:
            The optional ``theta_mapper`` controls how ``theta`` changes the
            physical model. For example, it can update the cosmology, HOD
            parameters, SHMR parameters, or any keyword arguments accepted by
            the supplied ``pk2d_func``.
        """
        theta_arr = np.asarray(theta, dtype=float)

        if context is None:
            context = self.context()

        model_inputs = self._model_inputs(theta_arr, context)

        cosmo = model_inputs["cosmo"]
        pk2d_kwargs = model_inputs["pk2d_kwargs"]

        return self.calculator.stacked_delta_sigma(
            r=self.r / float(cosmo["h"]),
            tomography=context["tomography"],
            cosmo=cosmo,
            pk2d_kwargs=pk2d_kwargs,
            trim_edge_points=self.trim_edge_points,
        )

    def context(self) -> dict[str, Any]:
        """Return the shared forecast context.

        Returns:
            Dictionary containing the tomography products, covariance builder,
            calculator, radius grids, and fiducial cosmology.

        Notes:
            The returned dictionary is a shallow copy of the cached context so
            that model calls can safely use it without directly exposing the
            cached context object.
        """
        return dict(self.forecast()["context"])

    def forecast_inputs(
        self,
    ) -> tuple[Callable[[ArrayLike], FloatArray], FloatArray, FloatArray]:
        """Return the model function, fiducial parameters, and covariance.

        Returns:
            Tuple containing the model callable, fiducial parameter vector, and
            covariance matrix.

        Notes:
            This is the minimal interface needed by many Fisher-style forecast
            tools. It uses cached forecast products when available, so repeated
            calls do not rebuild tomography or covariance.
        """
        forecast = self.forecast()
        return forecast["model"], forecast["theta0"], forecast["cov"]

    def forecastkit(self) -> ForecastKit:
        """Return a DerivKit ForecastKit initialized with the forecast inputs."""
        model, theta0, cov = self.forecast_inputs()
        return ForecastKit(function=model, theta0=theta0, cov=cov)

    def fisher(self) -> FloatArray:
        """Compute the Fisher matrix using DerivKit defaults."""
        return self.forecastkit().fisher()

    def dali(self) -> dict[str, FloatArray]:
        """Compute the DALI tensors using DerivKit defaults."""
        return self.forecastkit().dali()

    def _model_inputs(
        self,
        theta: FloatArray,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return model inputs for one forecast parameter vector.

        Args:
            theta: Parameter vector for the current model evaluation.
            context: Shared forecast context.

        Returns:
            Dictionary containing the cosmology and power-spectrum keyword
            arguments used to evaluate the Delta Sigma model.

        Notes:
            Fixed model keyword arguments are combined with parameter-dependent
            values. If no theta_mapper is supplied, parameter names are mapped
            directly to model keyword arguments. Otherwise, the mapper controls
            the parameter-dependent inputs.
        """
        pk2d_kwargs = dict(self.pk2d_kwargs)

        if self.theta_mapper is None:
            pk2d_kwargs.update(zip(self.parameter_names, map(float, theta), strict=True))
            return {"cosmo": self.cosmo, "pk2d_kwargs": pk2d_kwargs}

        mapped = dict(self.theta_mapper(theta, dict(context)))
        mapped.setdefault("cosmo", self.cosmo)

        varying_kwargs = mapped.get("pk2d_kwargs", {})
        if not isinstance(varying_kwargs, Mapping):
            raise TypeError("theta_mapper output 'pk2d_kwargs' must be a mapping.")

        pk2d_kwargs.update(varying_kwargs)
        mapped["pk2d_kwargs"] = pk2d_kwargs

        return mapped
