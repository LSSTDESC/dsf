r"""Forecast-ready DeltaSigma inputs.

This module provides the high-level builder used to prepare DeltaSigma forecast
inputs for Fisher or DALI analyses. It connects three pieces of the forecast:

- tomographic lens and source samples,
- the stacked DeltaSigma data vector,
- the covariance matrix matched to the selected lens-source bin pairs.

The builder is intentionally model-agnostic. It assumes that the data vector is
computed with ``DeltaSigmaCalculator``, but it does not assume a fixed physical
parameterization. Users provide a function that maps a parameter vector into the
cosmology and profile parameters used by the DeltaSigma calculation.

This makes the class suitable for simple amplitude-only forecasts, cosmology
forecasts, HOD or SHMR parameter forecasts, and later Fisher/DALI comparisons.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pyccl as ccl

from src.dsf.covariance.cov_builder import DeltaSigmaCovarianceBuilder
from src.dsf.data_vector.delta_sigma_builder import DeltaSigmaCalculator
from src.dsf.tomography.tomo_builder import TomographyBuilder
from src.dsf.utils.types import ArrayLike, FloatArray, ScalarOrPerBin, ThetaMapper
from src.dsf.utils.validators import (
    validate_forecast_vector_and_covariance,
    validate_parameter_names,
)

__all__ = [
    "DeltaSigmaForecastBuilder",
]


class DeltaSigmaForecastBuilder:
    """Prepare DeltaSigma data vectors and covariance inputs for forecasts.

    This class is the user-facing forecast layer for DeltaSigma analyses. It
    builds the tomographic samples, selects valid lens-source bin pairs,
    constructs the matching covariance, and exposes a model function that can be
    passed directly to forecast tools.

    The current forecast data vector contains only the galaxy-matter
    DeltaSigma signal, so this builder currently supports only the matching
    ``gm`` covariance block. The covariance layer also contains heleprs for
    ``gg`` and joint covariance blocks, which are intended for future forecast
    builders once the corresponding ``gg`` or joint data vectors are added.
    """

    def __init__(
        self,
        *,
        cosmo: ccl.Cosmology,
        pk2d_func: Callable[..., ccl.Pk2D],
        theta0: Sequence[float],
        r: ArrayLike,
        rp_bin_edges: ArrayLike,
        area_deg2: float,
        sigma_e: ScalarOrPerBin = 0.26,
        theta_mapper: ThetaMapper | None = None,
        parameter_names: Sequence[str] | None = None,
        lens_survey: str = "desi",
        source_survey: str = "lsst",
        lens_sample: str | None = "lrg",
        source_sample: str | None = None,
        lens_year: str | None = None,
        source_year: str | None = "1",
        lens_role: str = "lens",
        source_role: str = "source",
        overlap_threshold: float = 0.10,
        source_behind_lens: bool = True,
        center_method: str = "mean",
        decimal_places: int = 4,
        shared_overrides: Mapping[str, Any] | None = None,
        lens_overrides: Mapping[str, Any] | None = None,
        source_overrides: Mapping[str, Any] | None = None,
        observable: str = "delta_sigma",
        sigma_crit_prefactor: float | None = None,
        galaxy_bias: ScalarOrPerBin | None = None,
        galaxy_bias_prefactor: float = 1.0,
        galaxy_bias_round_decimals: int | None = 4,
        k: FloatArray | None = None,
        h: float | None = None,
        omega_m: float | None = None,
        rho_crit: float | None = None,
        nonlinear: bool = True,
        pi: FloatArray | None = None,
        gm_window: FloatArray | None = None,
        hankel_kwargs: Mapping[str, Any] | None = None,
        covariance_orders: Mapping[str, int] | None = None,
        covariance_kind: str = "gm",
        taper: bool = True,
        taper_kwargs: dict[str, Any] | None = None,
        trim_edge_points: int = 0,
        verbose: bool = True,
    ) -> None:
        r"""Set the forecast configuration.

        Args:
            cosmo: Fiducial CCL cosmology. This is used as the default
                cosmology for the data vector and covariance unless a
                ``theta_mapper`` supplies a parameter-dependent cosmology.
            pk2d_func: Function that returns the ``Pk2D`` object used by the
                DeltaSigma model.
            theta0: Fiducial parameter vector used as the expansion point for
                forecast calculations.
            r: Projected radii where the DeltaSigma data vector is evaluated.
            rp_bin_edges: Projected-radius bin edges used when constructing the
                covariance.
            area_deg2: Survey overlap area in square degrees.
            sigma_e: Source ellipticity dispersion used for the shape-noise
                contribution to the covariance.
            theta_mapper: Optional function mapping a parameter vector and
                forecast context into model inputs. It may return ``cosmo`` and
                ``pk2d_kwargs``.
            parameter_names: Optional names for the entries of ``theta0``.
            lens_survey: Survey preset used for the foreground lens sample.
            source_survey: Survey preset used for the background source sample.
            lens_sample: Optional sample label for the lens survey.
            source_sample: Optional sample label for the source survey.
            lens_year: Optional lens survey year or scenario label.
            source_year: Optional source survey year or scenario label.
            lens_role: Tomography role for the foreground sample.
            source_role: Tomography role for the background sample.
            overlap_threshold: Maximum allowed redshift overlap between a lens
                bin and source bin.
            source_behind_lens: Whether selected source bins must sit behind
                selected lens bins.
            center_method: Bin-center definition used when applying ordering
                cuts to lens-source bin pairs.
            decimal_places: Number of decimal places used in Binny summaries.
            shared_overrides: Binny tomography-spec overrides applied to both samples.
            lens_overrides: Binny tomography-spec overrides applied only to the lens
                sample.
            source_overrides: Binny tomography-spec overrides applied only to the
                source sample.
            observable: Forecast observable to evaluate. Currently only
                ``"delta_sigma"`` is supported.
            sigma_crit_prefactor: Optional unit-conversion factor applied to the
                effective critical surface density used in the covariance.
                If omitted, the covariance builder uses its internal default.
            galaxy_bias: Galaxy bias value or per-lens-bin values.
            galaxy_bias_prefactor: Normalization for the default galaxy-bias
                model.
            galaxy_bias_round_decimals: Optional rounding applied to the default
                galaxy-bias values.
            k: Wavenumber grid in ``h / Mpc`` used by the covariance.
            h: Dimensionless Hubble parameter used by covariance ingredients
                when needed explicitly.
            omega_m: Matter density fraction used by covariance ingredients
                when needed explicitly.
            rho_crit: Critical-density normalization used by covariance
                ingredients when needed explicitly.
            nonlinear: Whether covariance matter spectra should use nonlinear
                power.
            pi: Optional line-of-sight grid in ``Mpc / h``.
            gm_window: Optional galaxy-matter line-of-sight window.
            hankel_kwargs: Optional settings for covariance Hankel projections.
            covariance_orders: Bessel orders used by covariance projections.
            covariance_kind: Covariance product to use. Only ``"gm"`` is currently
                supported by this builder because the forecast data vector is
                DeltaSigma-only. The ``"gg"`` and ``"joint"`` covariance paths are
                reserved for future builders with matching data vectors.
            taper: Whether covariance spectra are tapered before projection.
            taper_kwargs: Optional taper settings.
            trim_edge_points: Number of edge points removed from each lens-bin
                redshift distribution before averaging DeltaSigma.
            verbose: Whether progress messages are printed while preparing the
                forecast.
        """
        if covariance_kind != "gm":
            raise NotImplementedError(
                "Only covariance_kind='gm' is currently supported because "
                "the forecast data vector is DeltaSigma-only. Add a joint data-vector "
                "builder before using 'gg' or 'joint' covariance."
            )

        self.cosmo = cosmo
        self.pk2d_func = pk2d_func
        self.calculator = DeltaSigmaCalculator(pk2d_func)

        self.theta0 = np.asarray(theta0, dtype=float)
        self.parameter_names = validate_parameter_names(parameter_names, self.theta0)
        self.theta_mapper = theta_mapper

        self.r = np.asarray(r, dtype=float)
        self.rp_bin_edges = np.asarray(rp_bin_edges, dtype=float)

        self.area_deg2 = float(area_deg2)
        self.sigma_e = sigma_e
        self.trim_edge_points = int(trim_edge_points)

        allowed_observables = {"delta_sigma"}

        if observable not in allowed_observables:
            raise NotImplementedError(
                f"observable={observable!r} is not supported yet. "
                "Currently this builder supports only 'delta_sigma'. "
                "Use a pair-dependent data-vector builder for gamma_t."
            )

        self.observable = observable

        self.covariance_kind = covariance_kind
        self.verbose = bool(verbose)

        self.tomography_kwargs: dict[str, Any] = {
            "lens_survey": lens_survey,
            "source_survey": source_survey,
            "lens_sample": lens_sample,
            "source_sample": source_sample,
            "lens_year": lens_year,
            "source_year": source_year,
            "lens_role": lens_role,
            "source_role": source_role,
            "overlap_threshold": overlap_threshold,
            "source_behind_lens": source_behind_lens,
            "center_method": center_method,
            "decimal_places": decimal_places,
            "shared_overrides": shared_overrides,
            "lens_overrides": lens_overrides,
            "source_overrides": source_overrides,
        }

        self.covariance_kwargs: dict[str, Any] = {
            "sigma_crit_prefactor": sigma_crit_prefactor,
            "galaxy_bias": galaxy_bias,
            "galaxy_bias_prefactor": galaxy_bias_prefactor,
            "galaxy_bias_round_decimals": galaxy_bias_round_decimals,
            "k": k,
            "h": h,
            "omega_m": omega_m,
            "rho_crit": rho_crit,
            "nonlinear": nonlinear,
            "pi": pi,
            "gm_window": gm_window,
            "hankel_kwargs": hankel_kwargs,
            "covariance_orders": covariance_orders,
            "taper": taper,
            "taper_kwargs": taper_kwargs,
        }

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
            tomography products, covariance builder, DeltaSigma calculator, and
            shared forecast context.

        Notes:
            Calling this method explicitly recomputes the forecast products and
            refreshes the cached result. Use ``forecast()`` when you want the
            cached products if they already exist.
        """
        self._log("[1/5] Building lens and source tomography...")

        tomography = TomographyBuilder(**self.tomography_kwargs).prepare_bins()

        self._log(
            "[1/5] Done: tomography built "
            f"with {len(tomography['bin_pairs'])} selected lens-source pairs."
        )
        self._log(f"Selected bin pairs: {tomography['bin_pairs']}")

        self._log("[2/5] Initializing DeltaSigma covariance builder...")

        cov_builder = DeltaSigmaCovarianceBuilder(
            cosmo=self.cosmo,
            lens_result=tomography["lens_result"],
            source_result=tomography["source_result"],
            lens_population_stats=tomography["lens_population_stats"],
            source_population_stats=tomography["source_population_stats"],
            bin_pairs=tomography["bin_pairs"],
            rp_bin_edges=self.rp_bin_edges,
            area_deg2=self.area_deg2,
            sigma_e=self.sigma_e,
            **self.covariance_kwargs,
        )

        self._log("[2/5] Done: covariance builder initialized.")

        data_pairs = [
            (int(lens_bin), int(source_bin)) for lens_bin, source_bin in tomography["bin_pairs"]
        ]

        if covariance is None:
            self._log(f"[3/5] Computing {self.covariance_kind} block-diagonal covariance...")

            cov_pairs, cov = self._covariance(
                cov_builder=cov_builder,
                bin_pairs=tomography["bin_pairs"],
            )

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

        context = self._context(
            tomography=tomography,
            cov_builder=cov_builder,
            cov_pairs=cov_pairs,
        )

        self._log("[4/5] Done: forecast context prepared.")

        self._log(
            "[5/5] Computing fiducial DeltaSigma data vector in selected lens-source pair order..."
        )

        data_vector = self.model(self.theta0, context=context)
        data_vector, cov = validate_forecast_vector_and_covariance(data_vector, cov)

        self._log(f"[5/5] Done: fiducial data vector computed with length {len(data_vector)}.")

        forecast: dict[str, Any] = {
            "model": self.model,
            "theta0": self.theta0,
            "parameter_names": self.parameter_names,
            "cov": cov,
            "data_vector": data_vector,
            "fiducial_data_vector": data_vector,
            "r": self.r,
            "rp_bin_edges": self.rp_bin_edges,
            "bin_pairs": list(tomography["bin_pairs"]),
            "cov_pairs": list(cov_pairs),
            "lens_result": tomography["lens_result"],
            "source_result": tomography["source_result"],
            "lens_population_stats": tomography["lens_population_stats"],
            "source_population_stats": tomography["source_population_stats"],
            "lens_bin_centers": tomography["lens_bin_centers"],
            "source_bin_centers": tomography["source_bin_centers"],
            "tomography": tomography,
            "covariance_builder": cov_builder,
            "calculator": self.calculator,
            "context": context,
            "covariance_kind": self.covariance_kind,
        }

        self._forecast = forecast

        self._log("Forecast preparation complete.")

        return forecast

    def forecast(self, covariance: FloatArray | None = None) -> dict[str, Any]:
        """Return cached forecast products, preparing them if needed.

        Args:
            covariance: Optional precomputed covariance matrix. If provided and no
                forecast has been cached yet, the builder uses this matrix instead
                of recomputing the covariance.

        Returns:
            Dictionary of forecast products suitable for direct use in Fisher,
            DALI, plotting, or diagnostic workflows.

        Notes:
            This is the safe accessor to use in downstream code. Unlike
            ``prepare()``, it does not rebuild tomography and covariance if the
            forecast has already been prepared.

            If a cached forecast already exists, passing a new covariance is not
            allowed because it would silently mismatch the cached context.
        """
        if self._forecast is not None:
            if covariance is not None:
                raise RuntimeError(
                    "A forecast is already cached. Create a new "
                    "DeltaSigmaForecastBuilder instance before supplying a new "
                    "covariance matrix."
                )

            return self._forecast

        self.prepare(covariance=covariance)

        if self._forecast is None:
            raise RuntimeError("DeltaSigma forecast products were not prepared.")

        return self._forecast

    def model(
        self,
        theta: ArrayLike,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> FloatArray:
        """Return the stacked DeltaSigma data vector for one parameter vector.

        Args:
            theta: Parameter vector at which the DeltaSigma model is evaluated.
            context: Optional forecast context. If omitted, the cached forecast
                context is used.

        Returns:
            One-dimensional stacked data vector containing the DeltaSigma blocks
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
        raw_pk2d_kwargs = model_inputs.get("pk2d_kwargs", {})

        if not isinstance(raw_pk2d_kwargs, Mapping):
            raise TypeError("theta_mapper output 'pk2d_kwargs' must be a mapping.")

        pk2d_kwargs: dict[str, Any] = dict(raw_pk2d_kwargs)

        return self._data_vector(
            cosmo=cosmo,
            pk2d_kwargs=pk2d_kwargs,
            tomography=context["tomography"],
        )

    def context(self) -> dict[str, Any]:
        """Return the shared forecast context.

        Returns:
            Dictionary containing the tomography products, bin centers, selected
            bin pairs, covariance builder, calculator, radius grids, and
            fiducial cosmology.

        Notes:
            The returned dictionary is a shallow copy of the cached context so
            that model calls can safely use it without directly exposing the
            cached context object.
        """
        forecast = self.forecast()
        cached_context = forecast["context"]

        if not isinstance(cached_context, Mapping):
            raise TypeError("Cached forecast context must be a mapping.")

        context: dict[str, Any] = dict(cached_context)
        return context

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

        model = forecast["model"]
        theta0 = forecast["theta0"]
        cov = forecast["cov"]

        return model, theta0, cov

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
            Dictionary containing the cosmology and profile keyword arguments
            used to evaluate the DeltaSigma model.

        Notes:
            If no ``theta_mapper`` is supplied, the model uses the fiducial
            cosmology and no additional ``pk2d_func`` keyword arguments. If a
            mapper is supplied, missing ``cosmo`` or ``pk2d_kwargs`` entries are
            filled with sensible defaults.
        """
        if self.theta_mapper is None:
            return {
                "cosmo": self.cosmo,
                "pk2d_kwargs": {},
            }

        context_dict: dict[str, Any] = dict(context)
        mapped: dict[str, Any] = dict(self.theta_mapper(theta, context_dict))

        if "cosmo" not in mapped:
            mapped["cosmo"] = self.cosmo

        if "pk2d_kwargs" not in mapped:
            mapped["pk2d_kwargs"] = {}

        return mapped

    def _data_vector(
        self,
        *,
        cosmo: ccl.Cosmology,
        pk2d_kwargs: Mapping[str, Any],
        tomography: Mapping[str, Any],
    ) -> FloatArray:
        """Return the stacked forecast data vector for selected bin pairs.

        Args:
            cosmo: Cosmology used for the current model evaluation.
            pk2d_kwargs: Keyword arguments passed to the profile model.
            tomography: Tomography products containing lens bins, source bins,
                and selected lens-source bin pairs.

        Returns:
            One-dimensional stacked data vector.

        Notes:
            For the current ``delta_sigma`` observable, the mean signal depends
            on the lens bin only. The selected source bin still matters for the
            pair selection and covariance ordering, so the returned vector keeps
            the exact Binny lens-source pair order.

            A future ``gamma_t`` observable should be evaluated per full
            lens-source pair because the source redshift distribution enters
            through the lensing efficiency or effective critical surface density.
        """
        if self.observable == "delta_sigma":
            return self._delta_sigma_data_vector(
                cosmo=cosmo,
                pk2d_kwargs=pk2d_kwargs,
                tomography=tomography,
            )

        raise NotImplementedError(
            f"observable={self.observable!r} is not supported by this builder."
        )

    def _delta_sigma_data_vector(
        self,
        *,
        cosmo: ccl.Cosmology,
        pk2d_kwargs: Mapping[str, Any],
        tomography: Mapping[str, Any],
    ) -> FloatArray:
        """Return the stacked lens-bin DeltaSigma vector in pair order.

        Args:
            cosmo: Cosmology used for the current model evaluation.
            pk2d_kwargs: Keyword arguments passed to the profile model.
            tomography: Tomography products containing lens bins and selected
                lens-source bin pairs.

        Returns:
            One-dimensional stacked DeltaSigma vector ordered by the selected
            lens-source bin pairs.

        Notes:
            DeltaSigma is computed once per unique lens bin because the mean
            DeltaSigma profile is lens-bin dependent. The resulting blocks are
            then repeated according to the exact Binny-selected pair order so
            that the data vector remains matched to the covariance.
        """
        lens_result = tomography["lens_result"]
        bin_pairs = tomography["bin_pairs"]

        lens_edges = np.asarray(
            lens_result.tomo_meta["bins"]["bin_edges"],
            dtype=float,
        )

        lens_z = np.asarray(lens_result.z, dtype=float)
        unique_lens_bins = sorted({int(lens_bin) for lens_bin, _ in bin_pairs})

        block_by_lens_bin = {}

        # Keep the user-facing forecast and covariance radius grid in comoving
        # Mpc / h. Convert to comoving Mpc here because the DeltaSigma
        # data-vector calculation passes radii to the CCL-backed profile
        # evaluation, which expects Mpc.
        h = float(cosmo["h"])
        r_mpc = self.r / h

        for lens_bin in unique_lens_bins:
            block_by_lens_bin[lens_bin] = np.asarray(
                self.calculator.delta_sigma_lens_bin(
                    r_mpc,
                    lens_dndz=(
                        lens_z,
                        np.asarray(lens_result.bins[lens_bin], dtype=float),
                    ),
                    cosmo=cosmo,
                    pk2d_kwargs=dict(pk2d_kwargs),
                    z_min=float(lens_edges[lens_bin]),
                    z_max=float(lens_edges[lens_bin + 1]),
                    trim_edge_points=self.trim_edge_points,
                ),
                dtype=float,
            )

        blocks = [block_by_lens_bin[int(lens_bin)] for lens_bin, _ in bin_pairs]

        data_vector = np.concatenate(blocks)

        return np.asarray(data_vector, dtype=float)

    def _covariance(
        self,
        *,
        cov_builder: DeltaSigmaCovarianceBuilder,
        bin_pairs: Sequence[tuple[int, int]],
    ) -> tuple[list[tuple[int, int]], FloatArray]:
        """Return the covariance matrix requested by the forecast.

        Args:
            cov_builder: Covariance builder matched to the selected tomography.
            bin_pairs: Lens-source bin pairs used by the DeltaSigma data vector.

        Returns:
            Bin-pair ordering and covariance matrix for the selected covariance
            product.

        Notes:
            The default is ``gm`` because the current forecast data vector is the
            lensing-style DeltaSigma signal. The ``joint`` option should only be
            used when the data vector is also expanded to include the matching
            galaxy-galaxy observable.
        """
        pair_list: list[tuple[int, int]] = [
            (int(lens_bin), int(source_bin)) for lens_bin, source_bin in bin_pairs
        ]

        cov_pairs, cov = cov_builder.gm_block_diagonal(bin_pairs=pair_list)

        return list(cov_pairs), np.asarray(cov, dtype=float)

    def _context(
        self,
        *,
        tomography: Mapping[str, Any],
        cov_builder: DeltaSigmaCovarianceBuilder,
        cov_pairs: Sequence[tuple[int, int]],
    ) -> dict[str, Any]:
        """Return shared objects needed by model and forecast calls.

        Args:
            tomography: Tomography products for the lens and source samples.
            cov_builder: Covariance builder matched to the same tomography.
            cov_pairs: Bin-pair ordering used by the covariance.

        Returns:
            Dictionary collecting the forecast state that should remain fixed
            during model evaluations.

        Notes:
            The context is designed to keep expensive or shared objects in one
            place. Forecast tools can repeatedly evaluate ``model(theta)`` while
            reusing the same tomography, bin-pair selection, radius grids, and
            covariance setup.
        """
        return {
            "cosmo": self.cosmo,
            "calculator": self.calculator,
            "r": self.r,
            "rp_bin_edges": self.rp_bin_edges,
            "tomography": tomography,
            "lens_result": tomography["lens_result"],
            "source_result": tomography["source_result"],
            "bin_pairs": list(tomography["bin_pairs"]),
            "lens_bin_centers": tomography["lens_bin_centers"],
            "source_bin_centers": tomography["source_bin_centers"],
            "lens_shape_stats": tomography["lens_shape_stats"],
            "source_shape_stats": tomography["source_shape_stats"],
            "covariance_builder": cov_builder,
            "cov_pairs": list(cov_pairs),
            "covariance_kind": self.covariance_kind,
            "observable": self.observable,
        }
