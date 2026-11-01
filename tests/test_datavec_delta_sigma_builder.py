"""Unit tests for ``dsf.data_vector.delta_sigma_builder``."""

import numpy as np
import pytest

from dsf.data_vector import delta_sigma_builder
from dsf.data_vector.delta_sigma_builder import (
    DeltaSigmaCalculator,
    stellar_point_mass_delta_sigma,
)


class DummyProfile:
    """Small stand-in for the standalone Delta Sigma projection."""

    def __init__(self, pk2d, **kwargs):
        """Store the projection inputs."""
        self.pk2d = pk2d
        self.kwargs = kwargs
        self.calls = []

    def delta_sigma(self, cosmo, r, a):
        """Return a deterministic excess surface density."""
        self.calls.append((cosmo, np.asarray(r, dtype=float), a))
        return 1.0e12 * a * np.full_like(np.asarray(r, dtype=float), 2.0)


class NonFiniteProfile:
    """Small projection backend that returns non-finite values."""

    def delta_sigma(self, cosmo, r, a):
        """Return one non-finite excess surface density value."""
        values = np.ones_like(np.asarray(r, dtype=float))
        values[1] = np.nan
        return values


class DummyPk2DFunction:
    """Small callable stand-in for a Pk2D factory."""

    def __init__(self):
        """Initialize an empty call log."""
        self.calls = []

    def __call__(self, **kwargs):
        """Record inputs and return a dummy Pk2D object."""
        self.calls.append(kwargs)
        return {"pk2d_kwargs": kwargs}


@pytest.fixture
def patch_delta_sigma(monkeypatch):
    """Patch the standalone Delta Sigma projection with a lightweight fake."""
    monkeypatch.setattr(delta_sigma_builder, "DeltaSigma", DummyProfile)


def test_delta_sigma_calculator_stores_pk2d_function():
    """Tests that the calculator stores the supplied Pk2D factory."""
    pk2d_func = DummyPk2DFunction()

    calculator = DeltaSigmaCalculator(pk2d_func=pk2d_func)

    assert calculator.pk2d_func is pk2d_func
    assert calculator._delta_sigma_cache is None


def test_generate_delta_sigma_builds_projection_and_stores_cache(patch_delta_sigma):
    """Tests that projection generation builds and stores a Delta Sigma object."""
    pk2d_func = DummyPk2DFunction()
    calculator = DeltaSigmaCalculator(pk2d_func=pk2d_func)

    profile = calculator._generate_delta_sigma(
        cosmo="cosmo",
        pk2d_kwargs={"model": "test"},
        overwrite_cache=True,
    )

    assert profile is calculator._delta_sigma_cache
    assert profile.pk2d == {
        "pk2d_kwargs": {"cosmo": "cosmo", "model": "test"}
    }
    assert pk2d_func.calls == [{"cosmo": "cosmo", "model": "test"}]


def test_generate_delta_sigma_reuses_cache_when_requested(patch_delta_sigma):
    """Tests that projection generation reuses the cache when requested."""
    pk2d_func = DummyPk2DFunction()
    calculator = DeltaSigmaCalculator(pk2d_func=pk2d_func)

    first = calculator._generate_delta_sigma(
        cosmo="cosmo",
        pk2d_kwargs={"model": "first"},
        overwrite_cache=True,
    )

    second = calculator._generate_delta_sigma(
        cosmo="cosmo",
        pk2d_kwargs={"model": "second"},
        overwrite_cache=False,
    )

    assert second is first
    assert pk2d_func.calls == [{"cosmo": "cosmo", "model": "first"}]


def test_generate_delta_sigma_overwrites_cache_when_requested(patch_delta_sigma):
    """Tests that projection generation rebuilds the cache when requested."""
    pk2d_func = DummyPk2DFunction()
    calculator = DeltaSigmaCalculator(pk2d_func=pk2d_func)

    first = calculator._generate_delta_sigma(
        cosmo="cosmo",
        pk2d_kwargs={"model": "first"},
        overwrite_cache=True,
    )

    second = calculator._generate_delta_sigma(
        cosmo="cosmo",
        pk2d_kwargs={"model": "second"},
        overwrite_cache=True,
    )

    assert second is not first
    assert second is calculator._delta_sigma_cache
    assert pk2d_func.calls == [
        {"cosmo": "cosmo", "model": "first"},
        {"cosmo": "cosmo", "model": "second"},
    ]


def test_delta_sigma_from_profile_converts_to_pc_units():
    """Tests that excess surface density is converted from Mpc to pc units."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())
    profile = DummyProfile(pk2d=None)

    r = np.array([1.0, 2.0, 4.0])
    result = calculator._delta_sigma_from_profile(
        profile=profile,
        r=r,
        a=0.5,
        cosmo="cosmo",
    )

    expected = np.full_like(r, 0.5 * 2.0)

    np.testing.assert_allclose(result, expected)


def test_delta_sigma_from_profile_forwards_projection_arguments():
    """Tests that cosmology, radii, and scale factor reach the projection."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())
    profile = DummyProfile(pk2d=None)

    r = np.array([1.0, 2.0])
    calculator._delta_sigma_from_profile(
        profile=profile,
        r=r,
        a=0.75,
        cosmo="cosmo",
    )

    called_cosmo, called_r, called_a = profile.calls[-1]

    assert called_cosmo == "cosmo"
    np.testing.assert_allclose(called_r, r)
    assert called_a == 0.75


@pytest.mark.parametrize("a", [-0.1, 0.0, 1.1])
def test_delta_sigma_from_profile_rejects_invalid_scale_factor(a):
    """Tests that invalid scale factors are rejected."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())

    with pytest.raises(ValueError):
        calculator._delta_sigma_from_profile(
            profile=DummyProfile(pk2d=None),
            r=np.array([1.0, 2.0]),
            a=a,
            cosmo="cosmo",
        )


def test_delta_sigma_from_profile_rejects_non_finite_values():
    """Tests that non-finite Delta Sigma values raise an error."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())

    with pytest.raises(
        FloatingPointError, match="Non-finite Delta Sigma values"
    ):
        calculator._delta_sigma_from_profile(
            profile=NonFiniteProfile(),
            r=np.array([1.0, 2.0, 3.0]),
            a=0.5,
            cosmo="cosmo",
        )


def test_delta_sigma_validates_radius_and_uses_generated_profile(patch_delta_sigma):
    """Tests that single-redshift Delta Sigma validates radii and uses a profile."""
    pk2d_func = DummyPk2DFunction()
    calculator = DeltaSigmaCalculator(pk2d_func=pk2d_func)

    result = calculator.delta_sigma(
        r=np.array([1.0, 2.0, 3.0]),
        a=0.5,
        cosmo="cosmo",
        pk2d_kwargs={"tag": "abc"},
    )

    np.testing.assert_allclose(result, np.ones(3))
    assert pk2d_func.calls == [{"cosmo": "cosmo", "tag": "abc"}]


@pytest.mark.parametrize(
    "r",
    [
        np.array([0.0, 1.0]),
        np.array([-1.0, 1.0]),
        np.array([[1.0, 2.0]]),
    ],
)
def test_delta_sigma_rejects_invalid_radius_arrays(patch_delta_sigma, r):
    """Tests that single-redshift Delta Sigma rejects invalid radial arrays."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())

    with pytest.raises(ValueError):
        calculator.delta_sigma(
            r=r,
            a=0.5,
            cosmo="cosmo",
        )


def test_delta_sigma_lens_bin_averages_over_redshift_distribution(
    patch_delta_sigma,
):
    """Tests that lens-bin Delta Sigma averages over the redshift distribution."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())

    r = np.array([1.0, 2.0])
    z = np.array([0.0, 1.0, 2.0])
    nz = np.array([1.0, 1.0, 1.0])

    result = calculator.delta_sigma_lens_bin(
        r=r,
        lens_dndz=(z, nz),
        cosmo="cosmo",
    )

    a = 1.0 / (1.0 + z)
    expected = np.full(r.size, np.trapezoid(a * 2.0, z) / np.trapezoid(nz, z))

    np.testing.assert_allclose(result, expected)


def test_delta_sigma_lens_bin_applies_redshift_window(patch_delta_sigma):
    """Tests that lens-bin Delta Sigma applies redshift window limits."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())

    r = np.array([1.0, 2.0])
    z = np.array([0.0, 0.5, 1.0, 1.5])
    nz = np.ones_like(z)

    result = calculator.delta_sigma_lens_bin(
        r=r,
        lens_dndz=(z, nz),
        cosmo="cosmo",
        z_min=0.5,
        z_max=1.0,
    )

    z_use = np.array([0.5, 1.0])
    a_use = 1.0 / (1.0 + z_use)
    nz_use = np.ones_like(z_use)

    expected = np.full(
        r.size,
        np.trapezoid(nz_use * a_use * 2.0, z_use)
        / np.trapezoid(nz_use, z_use),
    )

    np.testing.assert_allclose(result, expected)


def test_delta_sigma_lens_bin_trims_edge_points(patch_delta_sigma):
    """Tests that lens-bin Delta Sigma trims selected support edge points."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())

    r = np.array([1.0, 2.0])
    z = np.array([0.0, 0.5, 1.0, 1.5])
    nz = np.ones_like(z)

    result = calculator.delta_sigma_lens_bin(
        r=r,
        lens_dndz=(z, nz),
        cosmo="cosmo",
        trim_edge_points=1,
    )

    z_use = np.array([0.5, 1.0])
    a_use = 1.0 / (1.0 + z_use)
    nz_use = np.ones_like(z_use)

    expected = np.full(
        r.size,
        np.trapezoid(nz_use * a_use * 2.0, z_use)
        / np.trapezoid(nz_use, z_use),
    )

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("trim_edge_points", [-1, 0.5, 1.2])
def test_delta_sigma_lens_bin_rejects_invalid_trim_edge_points(
    patch_delta_sigma,
    trim_edge_points,
):
    """Tests that invalid edge-trimming values are rejected."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())

    with pytest.raises(ValueError, match="trim_edge_points"):
        calculator.delta_sigma_lens_bin(
            r=np.array([1.0, 2.0]),
            lens_dndz=(np.array([0.1, 0.2, 0.3]), np.ones(3)),
            cosmo="cosmo",
            trim_edge_points=trim_edge_points,
        )


def test_delta_sigma_lens_bin_rejects_trim_that_removes_all_support(
    patch_delta_sigma,
):
    """Tests that trimming away all lens support raises an error."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())

    with pytest.raises(ValueError, match="removes all lens_dndz support"):
        calculator.delta_sigma_lens_bin(
            r=np.array([1.0, 2.0]),
            lens_dndz=(np.array([0.1, 0.2]), np.ones(2)),
            cosmo="cosmo",
            trim_edge_points=1,
        )


def test_delta_sigma_lens_bin_rejects_too_few_support_points(
    patch_delta_sigma,
):
    """Tests that at least two selected redshift support points are required."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())

    with pytest.raises(ValueError, match="at least two redshift values"):
        calculator.delta_sigma_lens_bin(
            r=np.array([1.0, 2.0]),
            lens_dndz=(np.array([0.1, 0.2, 0.3]), np.array([0.0, 1.0, 0.0])),
            cosmo="cosmo",
        )


@pytest.mark.parametrize(
    ("z_min", "z_max"),
    [
        (-0.1, 1.0),
        (0.1, -1.0),
        (1.0, 0.5),
    ],
)
def test_delta_sigma_lens_bin_rejects_invalid_redshift_window(
    patch_delta_sigma,
    z_min,
    z_max,
):
    """Tests that invalid redshift window inputs are rejected."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())

    with pytest.raises(ValueError):
        calculator.delta_sigma_lens_bin(
            r=np.array([1.0, 2.0]),
            lens_dndz=(np.array([0.1, 0.2, 0.3]), np.ones(3)),
            cosmo="cosmo",
            z_min=z_min,
            z_max=z_max,
        )


def test_delta_sigma_lens_bin_raises_before_integration_for_non_finite_values(
    monkeypatch,
    patch_delta_sigma,
):
    """Tests that non-finite redshift-dependent Delta Sigma values are rejected."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())

    def bad_delta_sigma_from_profile(**kwargs):
        """Return one non-finite value before redshift integration."""
        return np.array([1.0, np.nan])

    monkeypatch.setattr(
        calculator,
        "_delta_sigma_from_profile",
        bad_delta_sigma_from_profile,
    )

    with pytest.raises(
        FloatingPointError, match="before redshift integration"
    ):
        calculator.delta_sigma_lens_bin(
            r=np.array([1.0, 2.0]),
            lens_dndz=(np.array([0.1, 0.2, 0.3]), np.ones(3)),
            cosmo="cosmo",
        )


def test_delta_sigma_lens_bin_raises_after_integration_for_non_finite_values(
    monkeypatch,
    patch_delta_sigma,
):
    """Tests that non-finite integrated Delta Sigma values are rejected."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())

    def finite_then_infinite_delta_sigma_from_profile(**kwargs):
        """Return infinite values that survive to the integration check."""
        return np.array([1.0, np.inf])

    monkeypatch.setattr(
        calculator,
        "_delta_sigma_from_profile",
        finite_then_infinite_delta_sigma_from_profile,
    )

    with pytest.raises(
        FloatingPointError, match="before redshift integration"
    ):
        calculator.delta_sigma_lens_bin(
            r=np.array([1.0, 2.0]),
            lens_dndz=(np.array([0.1, 0.2, 0.3]), np.ones(3)),
            cosmo="cosmo",
        )


def test_stellar_point_mass_delta_sigma_matches_expected_profile():
    """Tests that the stellar point-mass term follows M over pi R squared."""
    r = np.array([1.0, 2.0, 4.0])
    log10_mstellar = 12.0

    result = stellar_point_mass_delta_sigma(
        r=r,
        log10_mstellar=log10_mstellar,
    )

    expected = (10.0**log10_mstellar) / (np.pi * r**2) / 1.0e12

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "r",
    [
        np.array([0.0, 1.0]),
        np.array([-1.0, 1.0]),
        np.array([[1.0, 2.0]]),
    ],
)
def test_stellar_point_mass_delta_sigma_rejects_invalid_radii(r):
    """Tests that the stellar point-mass term rejects invalid radii."""
    with pytest.raises(ValueError):
        stellar_point_mass_delta_sigma(
            r=r,
            log10_mstellar=11.0,
        )


@pytest.mark.parametrize("log10_mstellar", [np.nan, np.inf, -np.inf])
def test_stellar_point_mass_delta_sigma_rejects_non_finite_mass(
    log10_mstellar,
):
    """Tests that the stellar point-mass term rejects non-finite mass values."""
    with pytest.raises(ValueError):
        stellar_point_mass_delta_sigma(
            r=np.array([1.0, 2.0]),
            log10_mstellar=log10_mstellar,
        )


def test_delta_sigma_rebuilds_projection_between_calls(patch_delta_sigma):
    """Tests that successive evaluations rebuild the power-spectrum projection."""
    pk2d_func = DummyPk2DFunction()
    calculator = DeltaSigmaCalculator(pk2d_func=pk2d_func)

    first = calculator.delta_sigma(
        r=np.array([1.0, 2.0]),
        a=0.5,
        cosmo="cosmo",
        pk2d_kwargs={"model": "first"},
    )
    first_profile = calculator._delta_sigma_cache

    second = calculator.delta_sigma(
        r=np.array([1.0, 2.0]),
        a=0.5,
        cosmo="cosmo",
        pk2d_kwargs={"model": "second"},
    )
    second_profile = calculator._delta_sigma_cache

    np.testing.assert_allclose(first, second)
    assert first_profile is not second_profile
    assert pk2d_func.calls == [
        {"cosmo": "cosmo", "model": "first"},
        {"cosmo": "cosmo", "model": "second"},
    ]


def test_delta_sigma_lens_bin_builds_projection_once(patch_delta_sigma):
    """Tests that lens-bin integration reuses one projection across redshifts."""
    pk2d_func = DummyPk2DFunction()
    calculator = DeltaSigmaCalculator(pk2d_func=pk2d_func)

    calculator.delta_sigma_lens_bin(
        r=np.array([1.0, 2.0]),
        lens_dndz=(np.array([0.0, 0.5, 1.0]), np.ones(3)),
        cosmo="cosmo",
    )

    assert len(pk2d_func.calls) == 1
    assert len(calculator._delta_sigma_cache.calls) == 3


def test_delta_sigma_lens_bin_uses_normalized_nonuniform_distribution(
    patch_delta_sigma,
):
    """Tests redshift weighting with nonuniform sampling and lens weights."""
    calculator = DeltaSigmaCalculator(pk2d_func=DummyPk2DFunction())

    r = np.array([1.0, 2.0])
    z = np.array([0.0, 0.2, 0.7, 1.5])
    nz = np.array([1.0, 3.0, 2.0, 1.0])

    result = calculator.delta_sigma_lens_bin(
        r=r,
        lens_dndz=(z, nz),
        cosmo="cosmo",
    )

    a = 1.0 / (1.0 + z)
    expected = np.trapezoid(nz * 2.0 * a, z) / np.trapezoid(nz, z)

    np.testing.assert_allclose(result, np.full(r.size, expected))


def test_delta_sigma_lens_bin_forwards_pk2d_kwargs(patch_delta_sigma):
    """Tests that lens-bin projection forwards power-spectrum parameters."""
    pk2d_func = DummyPk2DFunction()
    calculator = DeltaSigmaCalculator(pk2d_func=pk2d_func)

    calculator.delta_sigma_lens_bin(
        r=np.array([1.0, 2.0]),
        lens_dndz=(np.array([0.0, 0.5, 1.0]), np.ones(3)),
        cosmo="cosmo",
        pk2d_kwargs={"model": "nonlinear", "amplitude": 2.0},
    )

    assert pk2d_func.calls == [
        {"cosmo": "cosmo", "model": "nonlinear", "amplitude": 2.0}
    ]
