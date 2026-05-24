"""Tests for ``dsf.utils.validators.py``."""

import numpy as np
import pytest

from dsf.utils.validators import (
    as_1d_float_array,
    as_2d_float_array,
    is_non_negative_integer,
    normalize_axis,
    occupied_redshift_range_from_nz,
    redshift_window_mask,
    validate_1d_pair,
    validate_finite_scalar,
    validate_forecast_vector_and_covariance,
    validate_integration_axis,
    validate_integration_params,
    validate_joint_covariance_blocks,
    validate_nonnegative_1d_array,
    validate_nonnegative_scalar,
    validate_parameter_names,
    validate_positive_1d_array,
    validate_positive_scalar,
    validate_positive_strictly_increasing_1d_array,
    validate_power_spectrum_inputs,
    validate_redshift_distribution,
    validate_redshift_distribution_support,
    validate_redshift_edges,
    validate_redshift_pair,
    validate_scale_factor,
    validate_strictly_increasing,
)


def test_validate_finite_scalar_accepts_finite_values():
    """Tests that finite scalar values are accepted."""
    validate_finite_scalar(1.0, "value")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_validate_finite_scalar_rejects_non_finite_values(value):
    """Tests that non-finite scalar values are rejected."""
    with pytest.raises(ValueError, match="value must be finite"):
        validate_finite_scalar(value, "value")


def test_validate_positive_scalar_accepts_positive_values():
    """Tests that positive scalar values are accepted."""
    validate_positive_scalar(1.0, "value")


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_validate_positive_scalar_rejects_non_positive_values(value):
    """Tests that non-positive scalar values are rejected."""
    with pytest.raises(ValueError, match="value must be positive"):
        validate_positive_scalar(value, "value")


def test_validate_nonnegative_scalar_accepts_zero():
    """Tests that zero is accepted as non-negative."""
    validate_nonnegative_scalar(0.0, "value")


def test_validate_nonnegative_scalar_rejects_negative_values():
    """Tests that negative scalar values are rejected."""
    with pytest.raises(ValueError, match="value must be non-negative"):
        validate_nonnegative_scalar(-1.0, "value")


def test_validate_scale_factor_accepts_unit_interval_values():
    """Tests that scale factors in the physical interval are accepted."""
    validate_scale_factor(0.5)


@pytest.mark.parametrize("a", [0.0, -0.1, 1.1])
def test_validate_scale_factor_rejects_values_outside_unit_interval(a):
    """Tests that invalid scale factors are rejected."""
    with pytest.raises(ValueError, match="0 < a <= 1"):
        validate_scale_factor(a)


def test_validate_redshift_pair_accepts_source_behind_lens():
    """Tests that source redshifts behind lens redshifts are accepted."""
    validate_redshift_pair(0.3, 0.8)


def test_validate_redshift_pair_rejects_source_not_behind_lens():
    """Tests that source redshifts must be greater than lens redshifts."""
    with pytest.raises(ValueError, match="z_source must be greater than z_lens"):
        validate_redshift_pair(0.5, 0.5)


def test_as_1d_float_array_returns_float_array():
    """Tests that one-dimensional inputs are returned as float arrays."""
    actual = as_1d_float_array([1, 2, 3], "values")

    np.testing.assert_allclose(actual, np.array([1.0, 2.0, 3.0]))


def test_as_1d_float_array_rejects_multidimensional_inputs():
    """Tests that multidimensional inputs are rejected."""
    with pytest.raises(ValueError, match="values must be one-dimensional"):
        as_1d_float_array([[1.0, 2.0]], "values")


def test_as_1d_float_array_rejects_short_inputs():
    """Tests that arrays shorter than the minimum size are rejected."""
    with pytest.raises(ValueError, match="values must contain at least 2 values"):
        as_1d_float_array([1.0], "values", min_size=2)


def test_as_1d_float_array_rejects_non_finite_values():
    """Tests that non-finite array values are rejected."""
    with pytest.raises(ValueError, match="values must contain only finite values"):
        as_1d_float_array([1.0, np.nan], "values")


def test_as_2d_float_array_returns_float_array():
    """Tests that two-dimensional inputs are returned as float arrays."""
    actual = as_2d_float_array([[1, 2], [3, 4]], "matrix")

    np.testing.assert_allclose(actual, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_as_2d_float_array_rejects_1d_inputs():
    """Tests that non-two-dimensional inputs are rejected."""
    with pytest.raises(ValueError, match="matrix must be two-dimensional"):
        as_2d_float_array([1.0, 2.0], "matrix")


def test_validate_positive_1d_array_accepts_positive_values():
    """Tests that positive one-dimensional arrays are accepted."""
    actual = validate_positive_1d_array([1.0, 2.0], "values")

    np.testing.assert_allclose(actual, np.array([1.0, 2.0]))


def test_validate_positive_1d_array_rejects_non_positive_values():
    """Tests that non-positive array values are rejected."""
    with pytest.raises(ValueError, match="values must contain only positive values"):
        validate_positive_1d_array([1.0, 0.0], "values")


def test_validate_nonnegative_1d_array_accepts_zero_values():
    """Tests that non-negative one-dimensional arrays are accepted."""
    actual = validate_nonnegative_1d_array([0.0, 1.0], "values")

    np.testing.assert_allclose(actual, np.array([0.0, 1.0]))


def test_validate_nonnegative_1d_array_rejects_negative_values():
    """Tests that negative array values are rejected."""
    with pytest.raises(ValueError, match="values must be non-negative"):
        validate_nonnegative_1d_array([0.0, -1.0], "values")


def test_validate_strictly_increasing_accepts_ordered_values():
    """Tests that strictly increasing values are accepted."""
    actual = validate_strictly_increasing([0.0, 1.0, 2.0], "x")

    np.testing.assert_allclose(actual, np.array([0.0, 1.0, 2.0]))


def test_validate_strictly_increasing_rejects_repeated_values():
    """Tests that repeated values are rejected."""
    with pytest.raises(ValueError, match="x must be strictly increasing"):
        validate_strictly_increasing([0.0, 1.0, 1.0], "x")


def test_validate_positive_strictly_increasing_1d_array_accepts_valid_grid():
    """Tests that positive strictly increasing grids are accepted."""
    actual = validate_positive_strictly_increasing_1d_array([0.1, 1.0, 2.0], "k")

    np.testing.assert_allclose(actual, np.array([0.1, 1.0, 2.0]))


def test_validate_positive_strictly_increasing_1d_array_rejects_zero():
    """Tests that non-positive grid values are rejected."""
    with pytest.raises(ValueError, match="k must contain only positive values"):
        validate_positive_strictly_increasing_1d_array([0.0, 1.0], "k")


def test_validate_1d_pair_accepts_matching_arrays():
    """Tests that matching one-dimensional pairs are accepted."""
    validate_1d_pair([0.0, 1.0], [2.0, 3.0])


def test_validate_1d_pair_rejects_mismatched_shapes():
    """Tests that one-dimensional pairs must have matching shapes."""
    with pytest.raises(ValueError, match="x and y must have matching shapes"):
        validate_1d_pair([0.0, 1.0], [1.0])


def test_validate_joint_covariance_blocks_accepts_compatible_blocks():
    """Tests that compatible joint covariance blocks are accepted."""
    gm, gg, cross = validate_joint_covariance_blocks(
        np.eye(2),
        np.eye(3),
        np.ones((2, 3)),
    )

    assert gm.shape == (2, 2)
    assert gg.shape == (3, 3)
    assert cross.shape == (2, 3)


def test_validate_joint_covariance_blocks_rejects_non_square_auto_block():
    """Tests that auto-covariance blocks must be square."""
    with pytest.raises(ValueError, match="cov_gm_gm must be square"):
        validate_joint_covariance_blocks(np.ones((2, 3)), np.eye(2), np.ones((2, 2)))


def test_validate_joint_covariance_blocks_rejects_bad_cross_shape():
    """Tests that cross-covariance blocks must have compatible shape."""
    with pytest.raises(ValueError, match="cov_gm_gg has incompatible shape"):
        validate_joint_covariance_blocks(np.eye(2), np.eye(3), np.ones((3, 2)))


def test_normalize_axis_converts_negative_axis():
    """Tests that negative axes are converted to non-negative axes."""
    assert normalize_axis(-1, 3) == 2


def test_normalize_axis_rejects_out_of_bounds_axis():
    """Tests that axes outside the array dimensions are rejected."""
    with pytest.raises(ValueError, match="axis 3 is out of bounds"):
        normalize_axis(3, 3)


def test_validate_integration_axis_accepts_matching_axis_length():
    """Tests that integration coordinates matching an axis are accepted."""
    validate_integration_axis(np.ones((2, 3)), np.array([0.0, 1.0, 2.0]), axis=1)


def test_validate_integration_axis_rejects_scalar_values():
    """Tests that scalar integration values are rejected."""
    with pytest.raises(ValueError, match="values must have at least one dimension"):
        validate_integration_axis(1.0, np.array([0.0, 1.0]))


def test_validate_power_spectrum_inputs_accepts_matching_arrays():
    """Tests that matching power-spectrum arrays are accepted."""
    validate_power_spectrum_inputs([0.1, 1.0], [2.0, 3.0])


def test_validate_power_spectrum_inputs_rejects_mismatched_arrays():
    """Tests that power-spectrum arrays must have matching shapes."""
    with pytest.raises(ValueError, match="k and pk must have matching shapes"):
        validate_power_spectrum_inputs([0.1, 1.0], [2.0, 3.0, 4.0])


def test_validate_redshift_edges_accepts_ordered_edges():
    """Tests that ordered redshift edges are accepted."""
    validate_redshift_edges(0.2, 0.8)


def test_validate_redshift_edges_rejects_reversed_edges():
    """Tests that upper redshift edges must exceed lower edges."""
    with pytest.raises(ValueError, match="z_max must be greater than z_min"):
        validate_redshift_edges(0.8, 0.2)


def test_validate_redshift_distribution_accepts_valid_distribution():
    """Tests that valid redshift distributions are accepted."""
    z, nz = validate_redshift_distribution([0.0, 0.5, 1.0], [0.0, 1.0, 0.0])

    np.testing.assert_allclose(z, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(nz, np.array([0.0, 1.0, 0.0]))


def test_validate_redshift_distribution_rejects_negative_redshift():
    """Tests that negative redshift values are rejected."""
    with pytest.raises(ValueError, match="z must be non-negative"):
        validate_redshift_distribution([-0.1, 0.5], [1.0, 1.0])


def test_validate_integration_params_accepts_valid_settings():
    """Tests that valid integration settings are accepted."""
    validate_integration_params(
        {
            "n_ell": 10,
            "ell_min": 10.0,
            "ell_max": 100.0,
            "z_stepsize": 0.01,
            "z_min": 0.0,
            "delta_z_source": 0.1,
        }
    )


def test_validate_integration_params_rejects_invalid_ell_range():
    """Tests that ell_max must be greater than ell_min."""
    with pytest.raises(ValueError, match="ell_max must be greater than ell_min"):
        validate_integration_params(
            {
                "n_ell": 10,
                "ell_min": 100.0,
                "ell_max": 10.0,
                "z_stepsize": 0.01,
                "z_min": 0.0,
                "delta_z_source": 0.1,
            }
        )


def test_is_non_negative_integer_accepts_integer_values():
    """Tests that non-negative integer values are detected."""
    assert is_non_negative_integer(2.0)


def test_is_non_negative_integer_rejects_fractional_values():
    """Tests that fractional values are rejected."""
    assert not is_non_negative_integer(2.5)


def test_occupied_redshift_range_from_nz_returns_thresholded_range():
    """Tests that occupied redshift ranges follow the threshold."""
    actual = occupied_redshift_range_from_nz(
        [0.0, 0.5, 1.0, 1.5],
        [0.0, 0.2, 0.3, 0.0],
        threshold=0.1,
    )

    assert actual == pytest.approx((0.5, 1.0))


def test_occupied_redshift_range_from_nz_rejects_insufficient_support():
    """Tests that occupied redshift ranges need at least two cells."""
    with pytest.raises(ValueError, match="nz must be non-zero in at least two redshift cells"):
        occupied_redshift_range_from_nz([0.0, 0.5, 1.0], [0.0, 1.0, 0.0])


def test_validate_forecast_vector_and_covariance_accepts_matching_inputs():
    """Tests that forecast vectors and covariance matrices are matched."""
    data_vector, covariance = validate_forecast_vector_and_covariance(
        [1.0, 2.0],
        np.eye(2),
    )

    np.testing.assert_allclose(data_vector, np.array([1.0, 2.0]))
    np.testing.assert_allclose(covariance, np.eye(2))


def test_validate_forecast_vector_and_covariance_rejects_bad_covariance_shape():
    """Tests that covariance shape must match the data-vector length."""
    with pytest.raises(ValueError, match="covariance shape does not match data_vector"):
        validate_forecast_vector_and_covariance([1.0, 2.0], np.eye(3))


def test_validate_parameter_names_accepts_matching_names():
    """Tests that parameter names matching theta0 are accepted."""
    assert validate_parameter_names(["a", "b"], [1.0, 2.0]) == ("a", "b")


def test_validate_parameter_names_allows_missing_names():
    """Tests that missing parameter names return an empty tuple."""
    assert validate_parameter_names(None, [1.0, 2.0]) == ()


def test_validate_parameter_names_rejects_non_string_names():
    """Tests that parameter names must be strings."""
    with pytest.raises(TypeError, match="parameter_names must contain only strings"):
        validate_parameter_names(["a", 1], [1.0, 2.0])


def test_redshift_window_mask_selects_closed_window():
    """Tests that redshift masks include both window edges."""
    actual = redshift_window_mask(np.array([0.0, 0.5, 1.0]), z_min=0.5, z_max=1.0)

    np.testing.assert_array_equal(actual, np.array([False, True, True]))


def test_validate_redshift_distribution_support_trims_zero_edges():
    """Tests that redshift support removes leading and trailing zeros."""
    z, nz, norm = validate_redshift_distribution_support(
        [0.0, 0.5, 1.0, 1.5],
        [0.0, 1.0, 1.0, 0.0],
    )

    np.testing.assert_allclose(z, np.array([0.5, 1.0]))
    np.testing.assert_allclose(nz, np.array([1.0, 1.0]))
    assert norm == pytest.approx(0.5)


def test_validate_redshift_distribution_support_trims_edge_points():
    """Tests that support edge points can be trimmed."""
    z, nz, norm = validate_redshift_distribution_support(
        [0.0, 0.5, 1.0, 1.5],
        [1.0, 1.0, 1.0, 1.0],
        trim_edge_points=1,
    )

    np.testing.assert_allclose(z, np.array([0.5, 1.0]))
    np.testing.assert_allclose(nz, np.array([1.0, 1.0]))
    assert norm == pytest.approx(0.5)


def test_validate_redshift_distribution_support_rejects_zero_support():
    """Tests that empty positive redshift support is rejected."""
    with pytest.raises(ValueError, match="normalization must be finite and positive"):
        validate_redshift_distribution_support([0.0, 0.5], [0.0, 0.0])
