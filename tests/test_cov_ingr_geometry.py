"""Unit tests for ``dsf.covariance.ingredients.geometry``."""

import numpy as np
import pytest

from dsf.covariance.ingredients.geometry import (
    delta_pi_from_window,
    delta_pi_gg_from_edges,
    delta_pi_gm_factors,
    delta_pi_gm_from_window,
    delta_pi_gm_gg_from_window,
    effective_comoving_distance,
    gm_lensing_window,
    lens_number_density_3d_from_angular_density,
    pi_limits_from_lens_edges,
    survey_volume_from_edges,
)

MODULE = "dsf.covariance.ingredients.geometry"


def patch_linear_comoving_distance(monkeypatch, *, factor=1000.0):
    """Patches comoving distance to a linear function of redshift."""
    monkeypatch.setattr(
        f"{MODULE}.comoving_distance_h",
        lambda cosmo, z: factor * np.asarray(z, dtype=float),
    )


@pytest.fixture
def cosmo():
    """Returns a dummy cosmology object."""
    return object()


@pytest.fixture
def redshift_distribution():
    """Returns a small normalized redshift distribution."""
    z = np.asarray([0.0, 0.5, 1.0], dtype=float)
    nz = np.asarray([0.5, 1.0, 0.5], dtype=float)

    return z, nz


def test_effective_comoving_distance_integrates_distance_weighted_nz(
    monkeypatch,
    cosmo,
    redshift_distribution,
):
    """Tests that effective distance integrates chi weighted by n(z)."""
    patch_linear_comoving_distance(monkeypatch)

    z, nz = redshift_distribution

    distance = effective_comoving_distance(cosmo, z, nz)

    expected = np.trapezoid(1000.0 * z * nz, z)

    assert distance == pytest.approx(expected)


def test_delta_pi_gg_from_edges_returns_comoving_width(monkeypatch, cosmo):
    """Tests that gg line-of-sight width is the distance difference."""
    patch_linear_comoving_distance(monkeypatch)

    delta_pi = delta_pi_gg_from_edges(
        cosmo,
        z_min=0.2,
        z_max=0.5,
    )

    assert delta_pi == pytest.approx(300.0)


@pytest.mark.parametrize(
    ("squared", "expected"),
    [
        (False, np.trapezoid([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])),
        (True, np.trapezoid([1.0, 4.0, 9.0], [0.0, 1.0, 2.0])),
    ],
)
def test_delta_pi_from_window_integrates_window_or_squared_window(
    squared,
    expected,
):
    """Tests that delta-pi integrates either window or window squared."""
    pi = np.asarray([0.0, 1.0, 2.0], dtype=float)
    window = np.asarray([1.0, 2.0, 3.0], dtype=float)

    delta_pi = delta_pi_from_window(pi, window, squared=squared)

    assert delta_pi == pytest.approx(expected)


def test_delta_pi_gm_from_window_integrates_squared_window():
    """Tests that gm line-of-sight factor integrates squared window."""
    pi = np.asarray([0.0, 1.0, 2.0], dtype=float)
    window = np.asarray([1.0, 2.0, 3.0], dtype=float)

    delta_pi = delta_pi_gm_from_window(pi, window)

    expected = np.trapezoid(window**2, pi)

    assert delta_pi == pytest.approx(expected)


def test_delta_pi_gm_gg_from_window_integrates_selected_pi_range():
    """Tests that gm-gg line-of-sight factor integrates only selected pi values."""
    pi = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    window = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    delta_pi = delta_pi_gm_gg_from_window(
        pi,
        window,
        pi_min=-1.0,
        pi_max=1.0,
    )

    expected = np.trapezoid([2.0, 3.0, 4.0], [-1.0, 0.0, 1.0])

    assert delta_pi == pytest.approx(expected)


@pytest.mark.parametrize(
    ("pi_min", "pi_max"),
    [
        (1.0, 1.0),
        (2.0, 1.0),
    ],
)
def test_delta_pi_gm_gg_from_window_rejects_invalid_pi_limits(pi_min, pi_max):
    """Tests that gm-gg line-of-sight factor rejects invalid pi limits."""
    pi = np.asarray([0.0, 1.0, 2.0], dtype=float)
    window = np.asarray([1.0, 2.0, 3.0], dtype=float)

    with pytest.raises(ValueError, match="pi_max must be greater than pi_min"):
        delta_pi_gm_gg_from_window(
            pi,
            window,
            pi_min=pi_min,
            pi_max=pi_max,
        )


def test_delta_pi_gm_gg_from_window_rejects_too_few_selected_points():
    """Tests that gm-gg line-of-sight factor requires at least two pi points."""
    pi = np.asarray([0.0, 1.0, 2.0], dtype=float)
    window = np.asarray([1.0, 2.0, 3.0], dtype=float)

    with pytest.raises(ValueError, match="at least two points"):
        delta_pi_gm_gg_from_window(
            pi,
            window,
            pi_min=0.5,
            pi_max=1.5,
        )


def test_survey_volume_from_edges_returns_shell_volume(monkeypatch, cosmo):
    """Tests that survey volume is computed from shell distance limits and area."""
    patch_linear_comoving_distance(monkeypatch)

    volume = survey_volume_from_edges(
        cosmo,
        z_min=0.1,
        z_max=0.4,
        area_deg2=100.0,
    )

    area_sr = 100.0 * (np.pi / 180.0) ** 2
    expected = area_sr * (400.0**3 - 100.0**3) / 3.0

    assert volume == pytest.approx(expected)


def test_survey_volume_from_edges_rejects_non_positive_area(monkeypatch, cosmo):
    """Tests that survey volume rejects non-positive survey area."""
    with pytest.raises(ValueError):
        survey_volume_from_edges(
            cosmo,
            z_min=0.1,
            z_max=0.4,
            area_deg2=0.0,
        )


def test_lens_number_density_3d_from_angular_density_uses_volume(
    monkeypatch,
    cosmo,
):
    """Tests that lens density divides total angular counts by survey volume."""

    monkeypatch.setattr(
        f"{MODULE}.survey_volume_from_edges",
        lambda cosmo, z_min, z_max, area_deg2: 2.0e6,
    )

    density = lens_number_density_3d_from_angular_density(
        cosmo,
        n_lens_arcmin2=10.0,
        z_min=0.1,
        z_max=0.4,
        area_deg2=2.0,
    )

    expected = 10.0 * 2.0 * 60.0**2 / 2.0e6

    assert density == pytest.approx(expected)


def test_lens_number_density_3d_from_angular_density_rejects_non_positive_density(
    cosmo,
):
    """Tests that lens number density rejects non-positive angular density."""
    with pytest.raises(ValueError):
        lens_number_density_3d_from_angular_density(
            cosmo,
            n_lens_arcmin2=0.0,
            z_min=0.1,
            z_max=0.4,
            area_deg2=2.0,
        )


def test_pi_limits_from_lens_edges_returns_limits_relative_to_center(
    monkeypatch,
    cosmo,
):
    """Tests that pi limits are measured relative to the center distance."""
    patch_linear_comoving_distance(monkeypatch)

    pi_min, pi_max = pi_limits_from_lens_edges(
        cosmo,
        z_min=0.1,
        z_max=0.5,
        z_center=0.3,
    )

    assert pi_min == pytest.approx(-200.0)
    assert pi_max == pytest.approx(200.0)


def test_pi_limits_from_lens_edges_rejects_center_outside_edges(
    monkeypatch,
    cosmo,
):
    """Tests that pi limits reject centers outside the lens-bin edges."""
    with pytest.raises(ValueError, match="z_center must lie"):
        pi_limits_from_lens_edges(
            cosmo,
            z_min=0.1,
            z_max=0.5,
            z_center=0.6,
        )


def test_gm_lensing_window_uses_supplied_pi_grid(monkeypatch, cosmo):
    """Tests that gm lensing window returns values on the supplied pi grid."""
    patch_linear_comoving_distance(monkeypatch)

    monkeypatch.setattr(
        f"{MODULE}.effective_squared_sigma_crit",
        lambda *args, **kwargs: 4.0,
    )

    monkeypatch.setattr(
        f"{MODULE}.sigma_crit_inverse_comoving",
        lambda *args, **kwargs: np.ones(3),
    )

    z_lens = np.asarray([0.1, 0.2, 0.3], dtype=float)
    nz_lens = np.asarray([0.0, 1.0, 0.0], dtype=float)
    z_source = np.asarray([0.5, 0.6, 0.7], dtype=float)
    nz_source = np.asarray([0.0, 1.0, 0.0], dtype=float)
    pi = np.asarray([-10.0, 0.0, 10.0], dtype=float)

    pi_out, window = gm_lensing_window(
        cosmo,
        z_lens=z_lens,
        nz_lens=nz_lens,
        z_source=z_source,
        nz_source=nz_source,
        pi=pi,
        n_z_interp=20,
    )

    np.testing.assert_allclose(pi_out, pi)
    np.testing.assert_allclose(window, np.full_like(pi, 0.02))


def test_gm_lensing_window_builds_default_symmetric_pi_grid(monkeypatch, cosmo):
    """Tests that gm lensing window builds a symmetric default pi grid."""
    patch_linear_comoving_distance(monkeypatch)

    monkeypatch.setattr(
        f"{MODULE}.effective_squared_sigma_crit",
        lambda *args, **kwargs: 4.0,
    )

    monkeypatch.setattr(
        f"{MODULE}.sigma_crit_inverse_comoving",
        lambda *args, **kwargs: np.ones(3),
    )

    z_lens = np.asarray([0.1, 0.2, 0.3], dtype=float)
    nz_lens = np.asarray([0.0, 1.0, 0.0], dtype=float)
    z_source = np.asarray([0.5, 0.6, 0.7], dtype=float)
    nz_source = np.asarray([0.0, 1.0, 0.0], dtype=float)

    pi_out, window = gm_lensing_window(
        cosmo,
        z_lens=z_lens,
        nz_lens=nz_lens,
        z_source=z_source,
        nz_source=nz_source,
        pi_min=1.0,
        pi_max=10.0,
        n_pi=3,
        n_z_interp=20,
    )

    assert pi_out.size == 6
    np.testing.assert_allclose(pi_out[:3], -pi_out[:2:-1])
    np.testing.assert_allclose(window, np.full_like(pi_out, 0.02))


@pytest.mark.parametrize(
    "bad_pi",
    [
        np.asarray([[0.0, 1.0], [2.0, 3.0]]),
        np.asarray([0.0]),
    ],
)
def test_gm_lensing_window_rejects_invalid_supplied_pi_grid(
    monkeypatch,
    cosmo,
    bad_pi,
):
    """Tests that gm lensing window rejects invalid supplied pi grids."""
    with pytest.raises(ValueError):
        gm_lensing_window(
            cosmo,
            z_lens=[0.1, 0.2, 0.3],
            nz_lens=[0.0, 1.0, 0.0],
            z_source=[0.5, 0.6, 0.7],
            nz_source=[0.0, 1.0, 0.0],
            pi=bad_pi,
        )


def test_gm_lensing_window_rejects_invalid_default_pi_limits(monkeypatch, cosmo):
    """Tests that gm lensing window rejects invalid default pi limits."""
    with pytest.raises(ValueError, match="pi_max must be greater than pi_min"):
        gm_lensing_window(
            cosmo,
            z_lens=[0.1, 0.2, 0.3],
            nz_lens=[0.0, 1.0, 0.0],
            z_source=[0.5, 0.6, 0.7],
            nz_source=[0.0, 1.0, 0.0],
            pi_min=10.0,
            pi_max=1.0,
        )


def test_delta_pi_gm_factors_uses_supplied_window(monkeypatch, cosmo):
    """Tests that gm factors use supplied pi and window arrays."""
    monkeypatch.setattr(
        f"{MODULE}.pi_limits_from_lens_edges",
        lambda cosmo, z_min, z_max, z_center: (-1.0, 1.0),
    )

    pi = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
    window = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)

    delta_pi_gm, delta_pi_gm_gg = delta_pi_gm_factors(
        cosmo,
        z_lens=[0.1, 0.2, 0.3],
        nz_lens=[0.0, 1.0, 0.0],
        z_source=[0.5, 0.6, 0.7],
        nz_source=[0.0, 1.0, 0.0],
        z_min=0.1,
        z_max=0.3,
        z_center=0.2,
        pi=pi,
        gm_window=window,
    )

    expected_gm = np.trapezoid(window**2, pi)
    expected_gm_gg = np.trapezoid([2.0, 3.0, 4.0], [-1.0, 0.0, 1.0])

    assert delta_pi_gm == pytest.approx(expected_gm)
    assert delta_pi_gm_gg == pytest.approx(expected_gm_gg)


@pytest.mark.parametrize(
    ("pi", "gm_window"),
    [
        (None, np.asarray([1.0, 2.0, 3.0])),
        (np.asarray([0.0, 1.0, 2.0]), None),
    ],
)
def test_delta_pi_gm_factors_rejects_partially_supplied_window(
    cosmo,
    pi,
    gm_window,
):
    """Tests that gm factors require pi and window to be supplied together."""
    with pytest.raises(ValueError, match="both be supplied or both be None"):
        delta_pi_gm_factors(
            cosmo,
            z_lens=[0.1, 0.2, 0.3],
            nz_lens=[0.0, 1.0, 0.0],
            z_source=[0.5, 0.6, 0.7],
            nz_source=[0.0, 1.0, 0.0],
            z_min=0.1,
            z_max=0.3,
            z_center=0.2,
            pi=pi,
            gm_window=gm_window,
        )


def test_delta_pi_gm_factors_builds_window_when_not_supplied(monkeypatch, cosmo):
    """Tests that gm factors build the lensing window when none is supplied."""
    monkeypatch.setattr(
        f"{MODULE}.gm_lensing_window",
        lambda *args, **kwargs: (
            np.asarray([-1.0, 0.0, 1.0], dtype=float),
            np.asarray([2.0, 3.0, 4.0], dtype=float),
        ),
    )

    monkeypatch.setattr(
        f"{MODULE}.pi_limits_from_lens_edges",
        lambda cosmo, z_min, z_max, z_center: (-1.0, 1.0),
    )

    delta_pi_gm, delta_pi_gm_gg = delta_pi_gm_factors(
        cosmo,
        z_lens=[0.1, 0.2, 0.3],
        nz_lens=[0.0, 1.0, 0.0],
        z_source=[0.5, 0.6, 0.7],
        nz_source=[0.0, 1.0, 0.0],
        z_min=0.1,
        z_max=0.3,
        z_center=0.2,
    )

    expected_gm = np.trapezoid([4.0, 9.0, 16.0], [-1.0, 0.0, 1.0])
    expected_gm_gg = np.trapezoid([2.0, 3.0, 4.0], [-1.0, 0.0, 1.0])

    assert delta_pi_gm == pytest.approx(expected_gm)
    assert delta_pi_gm_gg == pytest.approx(expected_gm_gg)
