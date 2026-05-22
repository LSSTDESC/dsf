"""DeltaSigma covariance builder.

This module assembles covariance matrices for projected DeltaSigma forecast
data vectors. Binny provides the tomographic redshift distributions, bin
metadata, densities, and selected lens-source pairs. This builder combines
those tomography products with covariance ingredients such as matter power
spectra, survey volume, shape noise, shot noise, galaxy bias, and critical
surface density factors.

The main class is designed for forecasts where the observable may be the
galaxy-matter DeltaSigma signal alone, the projected clustering-like signal
alone, or a joint data vector containing both. The block-diagonal scripts give
simple first-pass covariances in which different tomographic lens-source pairs
are treated as independent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pyccl as ccl

from src.dsf.covariance.ingredients.cov_blocks import (
    delta_sigma_gg_covariance,
    delta_sigma_gm_covariance,
    delta_sigma_gm_gg_cross_covariance,
    joint_delta_sigma_covariance,
)
from src.dsf.covariance.ingredients.galaxy_bias import linear_galaxy_bias
from src.dsf.covariance.ingredients.geometry import (
    delta_pi_gg_from_edges,
    delta_pi_gm_factors,
    effective_comoving_distance,
    lens_number_density_3d_from_angular_density,
    survey_volume_from_edges,
)
from src.dsf.covariance.ingredients.noise import projected_shape_noise, shot_noise
from src.dsf.covariance.ingredients.power_spectrum import lens_averaged_matter_power
from src.dsf.covariance.ingredients.sigma_crit import effective_sigma_crit_squared
from src.dsf.covariance.projection.hankel_transform import HankelTransform
from src.dsf.utils.converters import (
    resolve_h,
    resolve_omega_m,
    rho_critical_projected_msun_pc2_per_mpc,
    sigma_crit_prefactor_msun_h_pc2,
)
from src.dsf.utils.types import ArrayLike, BinPairs, FloatArray, ScalarOrPerBin

__all__ = [
    "DeltaSigmaCovarianceBuilder",
]


class DeltaSigmaCovarianceBuilder:
    """Build projected DeltaSigma covariance matrices from tomography outputs.

    The builder collects the survey, cosmology, tomography, nuisance, and
    projection inputs needed to evaluate covariance blocks for tomographic
    lens-source bin pairs.

    It supports three useful covariance views:

    - ``gm x gm`` for the galaxy-matter DeltaSigma signal.
    - ``gg x gg`` for a projected clustering-like contribution.
    - a joint ``gm + gg`` covariance including the cross block.

    The class assumes that the input tomography objects already contain the
    redshift grids, binned distributions, bin edges, bin summaries, and sample
    metadata needed for the forecast.
    """

    def __init__(
        self,
        *,
        cosmo: ccl.Cosmology,
        lens_result: Any,
        source_result: Any,
        lens_population_stats: Mapping[str, Any],
        source_population_stats: Mapping[str, Any],
        bin_pairs: BinPairs,
        rp_bin_edges: ArrayLike,
        area_deg2: float,
        sigma_e: ScalarOrPerBin,
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
        hankel: HankelTransform | None = None,
        hankel_kwargs: Mapping[str, Any] | None = None,
        covariance_orders: Mapping[str, int] | None = None,
        taper: bool = True,
        taper_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the covariance builder.

        Args:
            cosmo: CCL cosmology used for distances, matter power spectra,
                growth-dependent quantities, and density normalizations.
            lens_result: Binny tomography result for the lens sample.
            source_result: Binny tomography result for the source sample.
            lens_population_stats: Lens sample population summary.
            source_population_stats: Source sample population summary.
            bin_pairs: Selected ``(lens_bin, source_bin)`` pairs.
            rp_bin_edges: Projected-radius bin edges in ``Mpc / h``.
            area_deg2: Survey area in square degrees.
            sigma_e: Source ellipticity dispersion.
            sigma_crit_prefactor: Optional unit-conversion prefactor.
            galaxy_bias: Galaxy bias for the lens sample.
            galaxy_bias_prefactor: Normalization used by the default linear
                galaxy-bias model.
            galaxy_bias_round_decimals: Optional decimal rounding applied to the
                default linear galaxy-bias values.
            k: Wavenumber grid in ``h / Mpc``.
            h: Dimensionless Hubble parameter.
            omega_m: Matter density fraction.
            rho_crit: Projected critical-density normalization.
            nonlinear: Whether to use nonlinear matter power.
            pi: Optional line-of-sight grid in ``Mpc / h``.
            gm_window: Optional galaxy-matter line-of-sight window.
            hankel: Optional pre-built Hankel transform object.
            hankel_kwargs: Optional settings for the Hankel transform.
            covariance_orders: Optional Bessel orders.
            taper: Whether covariance spectra should be tapered.
            taper_kwargs: Optional taper settings.
        """
        self.cosmo = cosmo
        self.lens_result = lens_result
        self.source_result = source_result
        self.bin_pairs: list[tuple[int, int]] = [
            (int(lens_bin), int(source_bin)) for lens_bin, source_bin in bin_pairs
        ]

        self.lens_meta = self.lens_result.tomo_meta
        self.source_meta = self.source_result.tomo_meta

        self.lens_edges = np.asarray(
            self.lens_result.tomo_meta["bins"]["bin_edges"],
            dtype=float,
        )
        self.source_edges = np.asarray(
            self.source_result.tomo_meta["bins"]["bin_edges"],
            dtype=float,
        )

        self.lens_centers = np.asarray(
            [
                self.lens_result.tomo_meta["bins"]["truez_summary"][i]["z_mean"]
                for i in self.lens_result.tomo_meta["bins"]["indices"]
            ],
            dtype=float,
        )
        self.source_centers = np.asarray(
            [
                self.source_result.tomo_meta["bins"]["truez_summary"][i]["z_mean"]
                for i in self.source_result.tomo_meta["bins"]["indices"]
            ],
            dtype=float,
        )

        self.z_lens = np.asarray(self.lens_result.z, dtype=float)
        self.z_source = np.asarray(self.source_result.z, dtype=float)

        self.lens_population_stats = dict(lens_population_stats)
        self.source_population_stats = dict(source_population_stats)

        self.rp_bin_edges = np.asarray(rp_bin_edges, dtype=float)
        self.area_deg2 = float(area_deg2)

        self.sigma_e = sigma_e
        self.sigma_crit_prefactor = (
            sigma_crit_prefactor_msun_h_pc2()
            if sigma_crit_prefactor is None
            else float(sigma_crit_prefactor)
        )

        self.h = resolve_h(cosmo, h)
        self.omega_m = resolve_omega_m(cosmo, omega_m)
        self.rho_crit = (
            rho_critical_projected_msun_pc2_per_mpc(self.cosmo, h=self.h)
            if rho_crit is None
            else float(rho_crit)
        )

        self.galaxy_bias = (
            galaxy_bias
            if galaxy_bias is not None
            else linear_galaxy_bias(
                self.cosmo,
                self.lens_centers,
                bias_prefactor=galaxy_bias_prefactor,
                round_decimals=galaxy_bias_round_decimals,
            )
        )

        if k is None:
            self.k: FloatArray = np.logspace(
                np.log10(1.0e-4),
                np.log10(30.0),
                5000,
                dtype=np.float64,
            )
        else:
            self.k = np.asarray(k, dtype=np.float64)

        self.nonlinear = bool(nonlinear)
        self.pi = None if pi is None else np.asarray(pi, dtype=float)
        self.gm_window = None if gm_window is None else np.asarray(gm_window, dtype=float)

        self.taper = bool(taper)
        self.taper_kwargs = taper_kwargs

        orders = {"gm": 2, "gg": 2, "cross": 2}
        if covariance_orders is not None:
            orders.update(covariance_orders)

        self.order_gm = int(orders["gm"])
        self.order_gg = int(orders["gg"])
        self.order_cross = int(orders["cross"])

        self._lens_ingredient_cache: dict[int, dict[str, Any]] = {}
        self._source_ingredient_cache: dict[int, dict[str, Any]] = {}

        self.hankel = hankel if hankel is not None else self.hankel_transform(hankel_kwargs)

    def covariance_for_pair(
        self,
        lens_bin_index: int,
        source_bin_index: int,
    ) -> dict[str, Any]:
        """Return all DeltaSigma covariance blocks for one bin pair.

        Args:
            lens_bin_index: Index of the lens tomographic bin.
            source_bin_index: Index of the source tomographic bin.

        Returns:
            Dictionary containing the projected radii, individual covariance
            blocks, joint covariance matrix, and physical ingredients for the
            requested lens-source pair.
        """
        ingredients = self.pair_ingredients(lens_bin_index, source_bin_index)

        gm_output = self.gm_covariance_for_pair(
            lens_bin_index,
            source_bin_index,
            ingredients=ingredients,
        )
        gg_output = self.gg_covariance_for_pair(
            lens_bin_index,
            source_bin_index,
            ingredients=ingredients,
        )
        cross_output = self.cross_covariance_for_pair(
            lens_bin_index,
            source_bin_index,
            ingredients=ingredients,
        )

        cov_joint = joint_delta_sigma_covariance(
            gm_output["cov_gm_gm"],
            gg_output["cov_gg_gg"],
            cross_output["cov_gm_gg"],
        )

        return {
            "lens_bin": int(lens_bin_index),
            "source_bin": int(source_bin_index),
            "r": gm_output["r"],
            "r_gm": gm_output["r_gm"],
            "r_gg": gg_output["r_gg"],
            "r_cross": cross_output["r_cross"],
            "cov_gm_gm": gm_output["cov_gm_gm"],
            "cov_gg_gg": gg_output["cov_gg_gg"],
            "cov_gm_gg": cross_output["cov_gm_gg"],
            "cov_joint": cov_joint,
            "ingredients": ingredients,
        }

    def gm_covariance_for_pair(
        self,
        lens_bin_index: int,
        source_bin_index: int,
        ingredients: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the ``gm x gm`` DeltaSigma covariance for one bin pair.

        Args:
            lens_bin_index: Index of the lens tomographic bin.
            source_bin_index: Index of the source tomographic bin.
            ingredients: Optional precomputed covariance ingredients.

        Returns:
            Dictionary containing the ``gm x gm`` covariance block and metadata.
        """
        if ingredients is None:
            ingredients = self.pair_ingredients(lens_bin_index, source_bin_index)

        r_gm, cov_gm_gm = delta_sigma_gm_covariance(
            self.hankel,
            self.k,
            ingredients["pk"],
            galaxy_bias=ingredients["galaxy_bias"],
            omega_m=self.omega_m,
            rho_crit=self.rho_crit,
            delta_pi_gm_squared_window=ingredients["delta_pi_gm_squared_window"],
            sigma_crit_squared_average=ingredients["sigma_crit_squared_average"],
            shape_noise=ingredients["shape_noise"],
            shot_noise=ingredients["shot_noise"],
            volume=ingredients["volume"],
            rp_bin_edges=self.rp_bin_edges,
            order=self.order_gm,
            taper=self.taper,
            taper_kwargs=self.taper_kwargs,
        )

        return {
            "lens_bin": int(lens_bin_index),
            "source_bin": int(source_bin_index),
            "r": r_gm,
            "r_gm": r_gm,
            "cov_gm_gm": cov_gm_gm,
            "ingredients": ingredients,
        }

    def gg_covariance_for_pair(
        self,
        lens_bin_index: int,
        source_bin_index: int,
        ingredients: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the ``gg x gg`` projected clustering covariance for one bin pair.

        Args:
            lens_bin_index: Index of the lens tomographic bin.
            source_bin_index: Index of the source tomographic bin.
            ingredients: Optional precomputed covariance ingredients.

        Returns:
            Dictionary containing the ``gg x gg`` covariance block and metadata.
        """
        if ingredients is None:
            ingredients = self.pair_ingredients(lens_bin_index, source_bin_index)

        r_gg, cov_gg_gg = delta_sigma_gg_covariance(
            self.hankel,
            self.k,
            ingredients["pk"],
            galaxy_bias=ingredients["galaxy_bias"],
            rho_crit=self.rho_crit,
            delta_pi_gg=ingredients["delta_pi_gg"],
            shot_noise=ingredients["shot_noise"],
            volume=ingredients["volume"],
            rp_bin_edges=self.rp_bin_edges,
            order=self.order_gg,
            taper=self.taper,
            taper_kwargs=self.taper_kwargs,
        )

        return {
            "lens_bin": int(lens_bin_index),
            "source_bin": int(source_bin_index),
            "r": r_gg,
            "r_gg": r_gg,
            "cov_gg_gg": cov_gg_gg,
            "ingredients": ingredients,
        }

    def cross_covariance_for_pair(
        self,
        lens_bin_index: int,
        source_bin_index: int,
        ingredients: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the ``gm x gg`` cross-covariance for one bin pair.

        Args:
            lens_bin_index: Index of the lens tomographic bin.
            source_bin_index: Index of the source tomographic bin.
            ingredients: Optional precomputed covariance ingredients.

        Returns:
            Dictionary containing the cross-covariance block and metadata.
        """
        if ingredients is None:
            ingredients = self.pair_ingredients(lens_bin_index, source_bin_index)

        r_cross, cov_gm_gg = delta_sigma_gm_gg_cross_covariance(
            self.hankel,
            self.k,
            ingredients["pk"],
            galaxy_bias=ingredients["galaxy_bias"],
            omega_m=self.omega_m,
            rho_crit=self.rho_crit,
            delta_pi_gm_gg=ingredients["delta_pi_gm_gg"],
            shot_noise=ingredients["shot_noise"],
            volume=ingredients["volume"],
            rp_bin_edges=self.rp_bin_edges,
            order=self.order_cross,
            taper=self.taper,
            taper_kwargs=self.taper_kwargs,
        )

        return {
            "lens_bin": int(lens_bin_index),
            "source_bin": int(source_bin_index),
            "r": r_cross,
            "r_cross": r_cross,
            "cov_gm_gg": cov_gm_gg,
            "ingredients": ingredients,
        }

    def covariance_for_pairs(
        self,
        bin_pairs: BinPairs | None = None,
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """Return all covariance outputs for multiple lens-source bin pairs.

        Args:
            bin_pairs: Optional set of ``(lens_bin, source_bin)`` pairs. If
                omitted, the pairs stored on the builder are used.

        Returns:
            Dictionary keyed by ``(lens_bin, source_bin)``. Each value is the
            full covariance output returned by :meth:`covariance_for_pair`.
        """
        pairs = self.selected_pairs(bin_pairs)

        return {pair: self.covariance_for_pair(pair[0], pair[1]) for pair in pairs}

    def gm_covariance_for_pairs(
        self,
        bin_pairs: BinPairs | None = None,
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """Return ``gm x gm`` covariance outputs for multiple bin pairs.

        Args:
            bin_pairs: Optional set of ``(lens_bin, source_bin)`` pairs. If
                omitted, the pairs stored on the builder are used.

        Returns:
            Dictionary keyed by ``(lens_bin, source_bin)``.
        """
        pairs = self.selected_pairs(bin_pairs)

        return {pair: self.gm_covariance_for_pair(pair[0], pair[1]) for pair in pairs}

    def gg_covariance_for_pairs(
        self,
        bin_pairs: BinPairs | None = None,
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """Return ``gg x gg`` covariance outputs for multiple bin pairs.

        Args:
            bin_pairs: Optional set of ``(lens_bin, source_bin)`` pairs. If
                omitted, the pairs stored on the builder are used.

        Returns:
            Dictionary keyed by ``(lens_bin, source_bin)``.
        """
        pairs = self.selected_pairs(bin_pairs)

        return {pair: self.gg_covariance_for_pair(pair[0], pair[1]) for pair in pairs}

    def cross_covariance_for_pairs(
        self,
        bin_pairs: BinPairs | None = None,
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """Return ``gm x gg`` cross-covariance outputs for multiple bin pairs.

        Args:
            bin_pairs: Optional set of ``(lens_bin, source_bin)`` pairs. If
                omitted, the pairs stored on the builder are used.

        Returns:
            Dictionary keyed by ``(lens_bin, source_bin)``.
        """
        pairs = self.selected_pairs(bin_pairs)

        return {pair: self.cross_covariance_for_pair(pair[0], pair[1]) for pair in pairs}

    def block_diagonal_from_outputs(
        self,
        outputs: dict[tuple[int, int], dict[str, Any]],
        output_key: str,
    ) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Return a block-diagonal covariance from already-computed outputs.

        Args:
            outputs: Covariance outputs keyed by bin pair.
            output_key: Name of the covariance block stored in each pair output.

        Returns:
            Pair list and block-diagonal covariance matrix.
        """
        pairs = list(outputs)
        blocks = [np.asarray(outputs[pair][output_key], dtype=float) for pair in pairs]

        size = int(sum(block.shape[0] for block in blocks))
        covariance = np.zeros((size, size), dtype=float)

        start = 0
        for block in blocks:
            stop = start + block.shape[0]
            covariance[start:stop, start:stop] = block
            start = stop

        return pairs, covariance

    def joint_block_diagonal(
        self,
        bin_pairs: BinPairs | None = None,
    ) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Return a block-diagonal joint covariance matrix.

        Args:
            bin_pairs: Optional set of ``(lens_bin, source_bin)`` pairs. If
                omitted, the builder's stored bin pairs are used.

        Returns:
            Pair list and block-diagonal covariance matrix.
        """
        outputs = self.covariance_for_pairs(bin_pairs)

        return self.block_diagonal_from_outputs(outputs, "cov_joint")

    def gm_block_diagonal(
        self,
        bin_pairs: BinPairs | None = None,
    ) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Return a block-diagonal covariance for the DeltaSigma signal.

        Args:
            bin_pairs: Optional set of ``(lens_bin, source_bin)`` pairs. If
                omitted, the builder's stored bin pairs are used.

        Returns:
            Pair list and block-diagonal ``gm x gm`` covariance matrix.
        """
        outputs = self.gm_covariance_for_pairs(bin_pairs)

        return self.block_diagonal_from_outputs(outputs, "cov_gm_gm")

    def gg_block_diagonal(
        self,
        bin_pairs: BinPairs | None = None,
    ) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Return a block-diagonal covariance for the projected gg signal.

        Args:
            bin_pairs: Optional set of ``(lens_bin, source_bin)`` pairs. If
                omitted, the builder's stored bin pairs are used.

        Returns:
            Pair list and block-diagonal ``gg x gg`` covariance matrix.
        """
        outputs = self.gg_covariance_for_pairs(bin_pairs)

        return self.block_diagonal_from_outputs(outputs, "cov_gg_gg")

    def cross_block_diagonal(
        self,
        bin_pairs: BinPairs | None = None,
    ) -> tuple[list[tuple[int, int]], np.ndarray]:
        """Return a block-diagonal covariance for the gm-gg cross block.

        Args:
            bin_pairs: Optional set of ``(lens_bin, source_bin)`` pairs. If
                omitted, the builder's stored bin pairs are used.

        Returns:
            Pair list and block-diagonal ``gm x gg`` covariance matrix.
        """
        outputs = self.cross_covariance_for_pairs(bin_pairs)

        return self.block_diagonal_from_outputs(outputs, "cov_gm_gg")

    def lens_ingredients(self, lens_bin_index: int) -> dict[str, Any]:
        """Return cached covariance ingredients that only depend on the lens bin.

        Args:
            lens_bin_index: Index of the lens tomographic bin.

        Returns:
            Dictionary containing lens-only covariance ingredients.
        """
        lens_bin_index = int(lens_bin_index)

        if lens_bin_index in self._lens_ingredient_cache:
            return self._lens_ingredient_cache[lens_bin_index]

        z_min = float(self.lens_edges[lens_bin_index])
        z_max = float(self.lens_edges[lens_bin_index + 1])
        z_center = float(self.lens_centers[lens_bin_index])

        nz_lens = np.asarray(self.lens_result.bins[lens_bin_index], dtype=float)

        galaxy_bias = self.bin_value(self.galaxy_bias, lens_bin_index)

        lens_density = self.lens_result.spec.get("sample_properties", {}).get(
            "number_density",
            {},
        )

        if "n_gal_comoving_h3_mpc3" in lens_density:
            n_lens_arcmin2 = None
            n_lens_3d = float(lens_density["n_gal_comoving_h3_mpc3"])
        else:
            n_lens_arcmin2 = float(self.lens_population_stats["density_per_bin"][lens_bin_index])
            n_lens_3d = lens_number_density_3d_from_angular_density(
                self.cosmo,
                n_lens_arcmin2=n_lens_arcmin2,
                z_min=z_min,
                z_max=z_max,
                area_deg2=self.area_deg2,
                h=self.h,
            )

        pk = lens_averaged_matter_power(
            self.cosmo,
            self.k,
            self.z_lens,
            nz_lens,
            h=self.h,
            nonlinear=self.nonlinear,
        )

        volume = survey_volume_from_edges(
            self.cosmo,
            z_min=z_min,
            z_max=z_max,
            area_deg2=self.area_deg2,
            h=self.h,
        )

        lens_shot_noise = shot_noise(n_lens_3d)

        chi_eff = effective_comoving_distance(
            self.cosmo,
            self.z_lens,
            nz_lens,
            h=self.h,
        )

        delta_pi_gg = delta_pi_gg_from_edges(
            self.cosmo,
            z_min=z_min,
            z_max=z_max,
            h=self.h,
        )

        ingredients = {
            "pk": pk,
            "volume": volume,
            "galaxy_bias": galaxy_bias,
            "n_lens_arcmin2": n_lens_arcmin2,
            "n_lens_3d": n_lens_3d,
            "shot_noise": lens_shot_noise,
            "chi_eff": chi_eff,
            "delta_pi_gg": delta_pi_gg,
            "z_min": z_min,
            "z_max": z_max,
            "z_center": z_center,
            "nz_lens": nz_lens,
        }

        self._lens_ingredient_cache[lens_bin_index] = ingredients
        return ingredients

    def source_ingredients(self, source_bin_index: int) -> dict[str, Any]:
        """Return cached covariance ingredients that only depend on the source bin.

        Args:
            source_bin_index: Index of the source tomographic bin.

        Returns:
            Dictionary containing source-only covariance ingredients.
        """
        source_bin_index = int(source_bin_index)

        if source_bin_index in self._source_ingredient_cache:
            return self._source_ingredient_cache[source_bin_index]

        nz_source = np.asarray(self.source_result.bins[source_bin_index], dtype=float)
        sigma_e = self.bin_value(self.sigma_e, source_bin_index)
        n_eff_source_arcmin2 = float(
            self.source_population_stats["density_per_bin"][source_bin_index]
        )

        ingredients = {
            "nz_source": nz_source,
            "sigma_e": sigma_e,
            "n_eff_source_arcmin2": n_eff_source_arcmin2,
        }

        self._source_ingredient_cache[source_bin_index] = ingredients
        return ingredients

    def pair_ingredients(
        self,
        lens_bin_index: int,
        source_bin_index: int,
    ) -> dict[str, Any]:
        """Return the covariance ingredients for one lens-source bin pair.

        Args:
            lens_bin_index: Index of the lens tomographic bin.
            source_bin_index: Index of the source tomographic bin.

        Returns:
            Dictionary containing the lens-averaged matter power spectrum,
            survey volume, lens and source noise terms, galaxy bias, redshift
            limits, and line-of-sight projection factors for the requested pair.
        """
        lens = self.lens_ingredients(lens_bin_index)
        source = self.source_ingredients(source_bin_index)

        source_shape_noise = projected_shape_noise(
            sigma_e=source["sigma_e"],
            n_eff_arcmin2=source["n_eff_source_arcmin2"],
            chi_eff=lens["chi_eff"],
        )

        delta_pi_gm_squared_window, delta_pi_gm_gg = delta_pi_gm_factors(
            self.cosmo,
            z_lens=self.z_lens,
            nz_lens=lens["nz_lens"],
            z_source=self.z_source,
            nz_source=source["nz_source"],
            z_min=lens["z_min"],
            z_max=lens["z_max"],
            z_center=lens["z_center"],
            h=self.h,
            sigma_crit_prefactor=self.sigma_crit_prefactor,
            pi=self.pi,
            gm_window=self.gm_window,
        )

        sigma_crit_squared_average = effective_sigma_crit_squared(
            self.cosmo,
            z_lens=self.z_lens,
            nz_lens=lens["nz_lens"],
            z_source=self.z_source,
            nz_source=source["nz_source"],
            h=self.h,
            sigma_crit_prefactor=self.sigma_crit_prefactor,
        )

        return {
            "pk": lens["pk"],
            "volume": lens["volume"],
            "galaxy_bias": lens["galaxy_bias"],
            "sigma_e": source["sigma_e"],
            "n_lens_arcmin2": lens["n_lens_arcmin2"],
            "n_lens_3d": lens["n_lens_3d"],
            "shot_noise": lens["shot_noise"],
            "n_eff_source_arcmin2": source["n_eff_source_arcmin2"],
            "shape_noise": source_shape_noise,
            "chi_eff": lens["chi_eff"],
            "delta_pi_gg": lens["delta_pi_gg"],
            "delta_pi_gm_squared_window": delta_pi_gm_squared_window,
            "delta_pi_gm_gg": delta_pi_gm_gg,
            "sigma_crit_squared_average": sigma_crit_squared_average,
            "z_min": lens["z_min"],
            "z_max": lens["z_max"],
            "z_center": lens["z_center"],
        }

    def hankel_transform(
        self,
        hankel_kwargs: Mapping[str, Any] | None = None,
    ) -> HankelTransform:
        """Return the Hankel transform used for radial covariance projections.

        Args:
            hankel_kwargs: Optional settings for the Hankel transform.

        Returns:
            Hankel transform object configured for the covariance Bessel orders.
        """
        kwargs = {} if hankel_kwargs is None else dict(hankel_kwargs)

        kwargs.setdefault("r_min", 0.6)
        kwargs.setdefault("r_max", 110.0)
        kwargs.setdefault("k_min", float(self.k[0]))
        kwargs.setdefault("k_max", float(self.k[-1]))
        kwargs.setdefault(
            "orders",
            tuple(sorted({self.order_gm, self.order_gg, self.order_cross})),
        )
        kwargs.setdefault("n_zeros", 28000)
        kwargs.setdefault("n_zeros_step", 1000)
        kwargs.setdefault("prune_r", None)
        kwargs.setdefault("verbose", False)
        kwargs.setdefault("max_iterations", 1000)

        return HankelTransform(**kwargs)

    def selected_pairs(
        self,
        bin_pairs: BinPairs | None = None,
    ) -> list[tuple[int, int]]:
        """Return requested bin pairs in a consistent integer format.

        Args:
            bin_pairs: Optional set of ``(lens_bin, source_bin)`` pairs. If
                omitted, the builder's stored bin pairs are used.

        Returns:
            List of integer ``(lens_bin, source_bin)`` pairs.
        """
        if bin_pairs is None:
            return list(self.bin_pairs)

        return [(int(lens_bin), int(source_bin)) for lens_bin, source_bin in bin_pairs]

    @staticmethod
    def bin_value(value: ScalarOrPerBin, bin_index: int) -> float:
        """Return the value associated with one tomographic bin.

        Args:
            value: Scalar value shared by all bins, or an array/dictionary with
                one value per tomographic bin.
            bin_index: Tomographic bin index to select.

        Returns:
            Scalar value for the requested bin.
        """
        if isinstance(value, Mapping):
            return float(value[bin_index])

        values = np.asarray(value, dtype=float)

        if values.ndim == 0:
            return float(values)

        return float(values[bin_index])

    def correlation_matrix(self, covariance: FloatArray) -> FloatArray:
        """Return the correlation matrix for a covariance matrix.

        Args:
            covariance: Covariance matrix.

        Returns:
            Correlation matrix with the same shape as ``covariance``.
        """
        return self.hankel.correlation_matrix(covariance)

    def diagonal_error(self, covariance: FloatArray) -> FloatArray:
        """Return one-sigma errors from a covariance matrix.

        Args:
            covariance: Covariance matrix.

        Returns:
            One-dimensional array containing the square root of the covariance
            diagonal.
        """
        return self.hankel.diagonal_error(covariance)
