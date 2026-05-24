"""Unit tests for ``dsf.covariance.ingredients.cov_blocks``."""

from types import SimpleNamespace

import numpy as np
import pytest

from dsf.covariance.ingredients.cov_blocks import (
    _build_taper_kwargs,
    _covariance_on_requested_radius_grid,
    delta_sigma_gg_covariance,
    delta_sigma_gm_covariance,
    delta_sigma_gm_gg_cross_covariance,
    joint_delta_sigma_covariance,
)


@pytest.fixture
def power_spectrum_inputs():
    """Returns a small valid power-spectrum grid."""
    k = np.asarray([0.1, 1.0, 20.0], dtype=float)
    pk = np.asarray([2.0, 3.0, 5.0], dtype=float)

    return k, pk


@pytest.fixture
def radius_grid():
    """Returns a small projected-radius grid."""
    return np.asarray([1.0, 2.0, 3.0], dtype=float)


@pytest.fixture
def hankel(radius_grid):
    """Returns a mutable Hankel-like object for monkeypatching."""
    hankel = SimpleNamespace()
    hankel.calls = []
    hankel.bin_calls = []
    hankel.r = radius_grid

    return hankel


def patch_projected_covariance(monkeypatch, hankel, *, radii=None):
    """Patches a deterministic projected covariance method onto a Hankel object."""
    if radii is None:
        radii = hankel.r

    def projected_covariance(
        *,
        k_pk,
        pk1,
        pk2,
        order,
        taper,
        taper_kwargs,
    ):
        hankel.calls.append(
            {
                "k_pk": np.asarray(k_pk, dtype=float),
                "pk1": np.asarray(pk1, dtype=float),
                "pk2": np.asarray(pk2, dtype=float),
                "order": order,
                "taper": taper,
                "taper_kwargs": taper_kwargs,
            }
        )

        return np.asarray(radii, dtype=float), np.outer(pk1, pk2)

    monkeypatch.setattr(
        hankel,
        "projected_covariance",
        projected_covariance,
        raising=False,
    )


def patch_binned_radial_matrix(monkeypatch, hankel):
    """Patches a deterministic radial binning method onto a Hankel object."""

    def bin_radial_matrix(r, cov, rp_bin_edges):
        hankel.bin_calls.append(
            {
                "r": np.asarray(r, dtype=float),
                "cov": np.asarray(cov, dtype=float),
                "rp_bin_edges": np.asarray(rp_bin_edges, dtype=float),
            }
        )

        return np.asarray([1.5, 2.5]), np.asarray(
            [
                [cov[0, 0], cov[0, 1]],
                [cov[1, 0], cov[1, 1]],
            ],
            dtype=float,
        )

    monkeypatch.setattr(
        hankel,
        "bin_radial_matrix",
        bin_radial_matrix,
        raising=False,
    )


def test_build_taper_kwargs_uses_default_edges(power_spectrum_inputs):
    """Tests that default taper edges are inferred from the input k grid."""
    k, _ = power_spectrum_inputs

    taper_kwargs = _build_taper_kwargs(k, None)

    assert taper_kwargs == {
        "large_k_lower": 10.0,
        "large_k_upper": 20.0,
        "low_k_lower": 0.1,
        "low_k_upper": 0.12,
    }


def test_build_taper_kwargs_returns_user_kwargs_unchanged(power_spectrum_inputs):
    """Tests that user-supplied taper settings are returned unchanged."""
    k, _ = power_spectrum_inputs
    user_kwargs = {
        "low_k_lower": 0.2,
        "low_k_upper": 0.4,
        "large_k_lower": 8.0,
        "large_k_upper": 30.0,
    }

    taper_kwargs = _build_taper_kwargs(k, user_kwargs)

    assert taper_kwargs is user_kwargs


@pytest.mark.parametrize(
    ("rp_bin_edges", "expected_r", "expected_cov", "expected_bin_calls"),
    [
        (
            None,
            np.asarray([1.0, 2.0, 3.0]),
            np.eye(3),
            0,
        ),
        (
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([1.5, 2.5]),
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            1,
        ),
    ],
)
def test_covariance_on_requested_radius_grid(
    monkeypatch,
    hankel,
    rp_bin_edges,
    expected_r,
    expected_cov,
    expected_bin_calls,
):
    """Tests that covariance is returned on the requested radial grid."""
    patch_binned_radial_matrix(monkeypatch, hankel)

    r_out, cov_out = _covariance_on_requested_radius_grid(
        hankel,
        np.asarray([1.0, 2.0, 3.0]),
        np.eye(3),
        rp_bin_edges=rp_bin_edges,
    )

    np.testing.assert_allclose(r_out, expected_r)
    np.testing.assert_allclose(cov_out, expected_cov)
    assert len(hankel.bin_calls) == expected_bin_calls


def test_covariance_on_requested_radius_grid_rejects_wrong_covariance_shape(hankel):
    """Tests that radius-grid covariance validation rejects shape mismatches."""
    r = np.asarray([1.0, 2.0, 3.0])
    cov = np.eye(2)

    with pytest.raises(ValueError, match="cov must have shape"):
        _covariance_on_requested_radius_grid(
            hankel,
            r,
            cov,
            rp_bin_edges=None,
        )


def test_delta_sigma_gm_covariance_matches_expected_block(
    monkeypatch,
    hankel,
    power_spectrum_inputs,
):
    """Tests that the gm covariance block matches the expected scaling."""
    patch_projected_covariance(monkeypatch, hankel)
    k, pk = power_spectrum_inputs

    galaxy_bias = 1.5
    omega_m = 0.3
    rho_crit = 2.0
    delta_pi_gm_squared_window = 4.0
    sigma_crit_squared_average = 6.0
    shape_noise = 0.1
    shot_noise = 0.2
    volume = 10.0

    r, cov = delta_sigma_gm_covariance(
        hankel,
        k,
        pk,
        galaxy_bias=galaxy_bias,
        omega_m=omega_m,
        rho_crit=rho_crit,
        delta_pi_gm_squared_window=delta_pi_gm_squared_window,
        sigma_crit_squared_average=sigma_crit_squared_average,
        shape_noise=shape_noise,
        shot_noise=shot_noise,
        volume=volume,
        taper=True,
        gkgk_taper=False,
    )

    p_g = pk * galaxy_bias**2
    p_kappa = pk * (rho_crit * omega_m) ** 2 * delta_pi_gm_squared_window
    p_gk = pk * galaxy_bias * rho_crit * omega_m
    shape_delta_sigma_noise = shape_noise * sigma_crit_squared_average

    expected_ggkk = np.outer(p_g + shot_noise, p_kappa + shape_delta_sigma_noise)
    expected_gkgk = np.outer(p_gk, p_gk)
    expected_cov = (expected_ggkk + expected_gkgk * delta_pi_gm_squared_window) / volume

    np.testing.assert_allclose(r, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cov, expected_cov)

    assert len(hankel.calls) == 2
    assert hankel.calls[0]["taper"] is True
    assert hankel.calls[1]["taper"] is False
    assert hankel.calls[0]["order"] == 2
    assert hankel.calls[1]["order"] == 2


def test_delta_sigma_gm_covariance_rejects_mismatched_radial_grids(
    monkeypatch,
    hankel,
    power_spectrum_inputs,
):
    """Tests that gm covariance rejects inconsistent Hankel radial grids."""
    k, pk = power_spectrum_inputs

    def projected_covariance(
        *,
        k_pk,
        pk1,
        pk2,
        order,
        taper,
        taper_kwargs,
    ):
        hankel.calls.append({})

        if len(hankel.calls) == 1:
            return np.asarray([1.0, 2.0, 3.0]), np.outer(pk1, pk2)

        return np.asarray([1.0, 2.1, 3.0]), np.outer(pk1, pk2)

    monkeypatch.setattr(
        hankel,
        "projected_covariance",
        projected_covariance,
        raising=False,
    )

    with pytest.raises(ValueError, match="same radial grid"):
        delta_sigma_gm_covariance(
            hankel,
            k,
            pk,
            galaxy_bias=1.5,
            omega_m=0.3,
            rho_crit=2.0,
            delta_pi_gm_squared_window=4.0,
            sigma_crit_squared_average=6.0,
            shape_noise=0.1,
            shot_noise=0.2,
            volume=10.0,
        )


def test_delta_sigma_gg_covariance_matches_expected_block(
    monkeypatch,
    hankel,
    power_spectrum_inputs,
):
    """Tests that the gg covariance block matches the expected scaling."""
    patch_projected_covariance(monkeypatch, hankel)
    k, pk = power_spectrum_inputs

    galaxy_bias = 1.5
    rho_crit = 2.0
    delta_pi_gg = 5.0
    shot_noise = 0.2
    volume = 10.0

    r, cov = delta_sigma_gg_covariance(
        hankel,
        k,
        pk,
        galaxy_bias=galaxy_bias,
        rho_crit=rho_crit,
        delta_pi_gg=delta_pi_gg,
        shot_noise=shot_noise,
        volume=volume,
    )

    p_g = pk * galaxy_bias**2
    expected_cov = (
        np.outer(p_g + shot_noise, p_g + shot_noise) * 2.0 * delta_pi_gg * rho_crit**2 / volume
    )

    np.testing.assert_allclose(r, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cov, expected_cov)

    assert len(hankel.calls) == 1
    np.testing.assert_allclose(hankel.calls[0]["pk1"], p_g + shot_noise)
    np.testing.assert_allclose(hankel.calls[0]["pk2"], p_g + shot_noise)


def test_delta_sigma_gm_gg_cross_covariance_matches_expected_block(
    monkeypatch,
    hankel,
    power_spectrum_inputs,
):
    """Tests that the gm-gg cross-covariance block matches the expected scaling."""
    patch_projected_covariance(monkeypatch, hankel)
    k, pk = power_spectrum_inputs

    galaxy_bias = 1.5
    omega_m = 0.3
    rho_crit = 2.0
    delta_pi_gm_gg = 7.0
    shot_noise = 0.2
    volume = 10.0

    r, cov = delta_sigma_gm_gg_cross_covariance(
        hankel,
        k,
        pk,
        galaxy_bias=galaxy_bias,
        omega_m=omega_m,
        rho_crit=rho_crit,
        delta_pi_gm_gg=delta_pi_gm_gg,
        shot_noise=shot_noise,
        volume=volume,
    )

    p_g = pk * galaxy_bias**2
    p_gk = pk * galaxy_bias * rho_crit * omega_m

    expected_cov = np.outer(p_gk, p_g + shot_noise) * 2.0 * delta_pi_gm_gg * rho_crit / volume

    np.testing.assert_allclose(r, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(cov, expected_cov)

    assert len(hankel.calls) == 1
    np.testing.assert_allclose(hankel.calls[0]["pk1"], p_gk)
    np.testing.assert_allclose(hankel.calls[0]["pk2"], p_g + shot_noise)


@pytest.mark.parametrize(
    "covariance_function,kwargs",
    [
        (
            delta_sigma_gg_covariance,
            {
                "galaxy_bias": 1.5,
                "rho_crit": 2.0,
                "delta_pi_gg": 5.0,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
        ),
        (
            delta_sigma_gm_gg_cross_covariance,
            {
                "galaxy_bias": 1.5,
                "omega_m": 0.3,
                "rho_crit": 2.0,
                "delta_pi_gm_gg": 7.0,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
        ),
    ],
)
def test_covariance_functions_bin_to_requested_radius_edges(
    monkeypatch,
    hankel,
    power_spectrum_inputs,
    covariance_function,
    kwargs,
):
    """Tests that covariance functions bin to requested projected-radius edges."""
    patch_projected_covariance(monkeypatch, hankel)
    patch_binned_radial_matrix(monkeypatch, hankel)
    k, pk = power_spectrum_inputs

    r, cov = covariance_function(
        hankel,
        k,
        pk,
        **kwargs,
        rp_bin_edges=[1.0, 2.0, 3.0],
    )

    assert len(hankel.bin_calls) == 1
    np.testing.assert_allclose(r, [1.5, 2.5])
    assert cov.shape == (2, 2)


def test_joint_delta_sigma_covariance_assembles_blocks():
    """Tests that the joint covariance matrix is assembled from input blocks."""
    cov_gm_gm = np.asarray(
        [
            [1.0, 0.1],
            [0.1, 2.0],
        ]
    )
    cov_gg_gg = np.asarray(
        [
            [3.0, 0.2],
            [0.2, 4.0],
        ]
    )
    cov_gm_gg = np.asarray(
        [
            [0.5, 0.6],
            [0.7, 0.8],
        ]
    )

    cov = joint_delta_sigma_covariance(
        cov_gm_gm,
        cov_gg_gg,
        cov_gm_gg,
    )

    expected = np.asarray(
        [
            [1.0, 0.1, 0.5, 0.6],
            [0.1, 2.0, 0.7, 0.8],
            [0.5, 0.7, 3.0, 0.2],
            [0.6, 0.8, 0.2, 4.0],
        ]
    )

    np.testing.assert_allclose(cov, expected)


@pytest.mark.parametrize(
    ("covariance_function", "base_kwargs", "bad_key"),
    [
        (
            delta_sigma_gm_covariance,
            {
                "galaxy_bias": 1.5,
                "omega_m": 0.3,
                "rho_crit": 2.0,
                "delta_pi_gm_squared_window": 4.0,
                "sigma_crit_squared_average": 6.0,
                "shape_noise": 0.1,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
            "galaxy_bias",
        ),
        (
            delta_sigma_gm_covariance,
            {
                "galaxy_bias": 1.5,
                "omega_m": 0.3,
                "rho_crit": 2.0,
                "delta_pi_gm_squared_window": 4.0,
                "sigma_crit_squared_average": 6.0,
                "shape_noise": 0.1,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
            "omega_m",
        ),
        (
            delta_sigma_gm_covariance,
            {
                "galaxy_bias": 1.5,
                "omega_m": 0.3,
                "rho_crit": 2.0,
                "delta_pi_gm_squared_window": 4.0,
                "sigma_crit_squared_average": 6.0,
                "shape_noise": 0.1,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
            "rho_crit",
        ),
        (
            delta_sigma_gm_covariance,
            {
                "galaxy_bias": 1.5,
                "omega_m": 0.3,
                "rho_crit": 2.0,
                "delta_pi_gm_squared_window": 4.0,
                "sigma_crit_squared_average": 6.0,
                "shape_noise": 0.1,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
            "delta_pi_gm_squared_window",
        ),
        (
            delta_sigma_gm_covariance,
            {
                "galaxy_bias": 1.5,
                "omega_m": 0.3,
                "rho_crit": 2.0,
                "delta_pi_gm_squared_window": 4.0,
                "sigma_crit_squared_average": 6.0,
                "shape_noise": 0.1,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
            "sigma_crit_squared_average",
        ),
        (
            delta_sigma_gm_covariance,
            {
                "galaxy_bias": 1.5,
                "omega_m": 0.3,
                "rho_crit": 2.0,
                "delta_pi_gm_squared_window": 4.0,
                "sigma_crit_squared_average": 6.0,
                "shape_noise": 0.1,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
            "shape_noise",
        ),
        (
            delta_sigma_gm_covariance,
            {
                "galaxy_bias": 1.5,
                "omega_m": 0.3,
                "rho_crit": 2.0,
                "delta_pi_gm_squared_window": 4.0,
                "sigma_crit_squared_average": 6.0,
                "shape_noise": 0.1,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
            "shot_noise",
        ),
        (
            delta_sigma_gm_covariance,
            {
                "galaxy_bias": 1.5,
                "omega_m": 0.3,
                "rho_crit": 2.0,
                "delta_pi_gm_squared_window": 4.0,
                "sigma_crit_squared_average": 6.0,
                "shape_noise": 0.1,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
            "volume",
        ),
        (
            delta_sigma_gg_covariance,
            {
                "galaxy_bias": 1.5,
                "rho_crit": 2.0,
                "delta_pi_gg": 5.0,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
            "delta_pi_gg",
        ),
        (
            delta_sigma_gm_gg_cross_covariance,
            {
                "galaxy_bias": 1.5,
                "omega_m": 0.3,
                "rho_crit": 2.0,
                "delta_pi_gm_gg": 7.0,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
            "delta_pi_gm_gg",
        ),
    ],
)
def test_covariance_functions_reject_non_positive_scalars(
    monkeypatch,
    hankel,
    power_spectrum_inputs,
    covariance_function,
    base_kwargs,
    bad_key,
):
    """Tests that covariance functions reject non-positive scalar inputs."""
    patch_projected_covariance(monkeypatch, hankel)
    k, pk = power_spectrum_inputs
    kwargs = dict(base_kwargs)
    kwargs[bad_key] = 0.0

    with pytest.raises(ValueError):
        covariance_function(
            hankel,
            k,
            pk,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("covariance_function", "kwargs"),
    [
        (
            delta_sigma_gm_covariance,
            {
                "galaxy_bias": 1.5,
                "omega_m": 0.3,
                "rho_crit": 2.0,
                "delta_pi_gm_squared_window": 4.0,
                "sigma_crit_squared_average": 6.0,
                "shape_noise": 0.1,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
        ),
        (
            delta_sigma_gg_covariance,
            {
                "galaxy_bias": 1.5,
                "rho_crit": 2.0,
                "delta_pi_gg": 5.0,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
        ),
        (
            delta_sigma_gm_gg_cross_covariance,
            {
                "galaxy_bias": 1.5,
                "omega_m": 0.3,
                "rho_crit": 2.0,
                "delta_pi_gm_gg": 7.0,
                "shot_noise": 0.2,
                "volume": 10.0,
            },
        ),
    ],
)
def test_covariance_functions_reject_invalid_power_spectrum_shape(
    monkeypatch,
    hankel,
    covariance_function,
    kwargs,
):
    """Tests that covariance functions reject mismatched k and pk shapes."""
    patch_projected_covariance(monkeypatch, hankel)
    k = np.asarray([0.1, 1.0, 20.0])
    pk = np.asarray([2.0, 3.0])

    with pytest.raises(ValueError):
        covariance_function(
            hankel,
            k,
            pk,
            **kwargs,
        )
