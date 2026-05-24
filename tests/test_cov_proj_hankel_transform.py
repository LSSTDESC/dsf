"""Unit tests for ``dsf.covariance.projection.hankel_transform``."""

import numpy as np
import pytest

from dsf.covariance.projection.hankel_transform import HankelTransform

HANKEL_MODULE = "dsf.covariance.projection.hankel_transform"


def make_fake_transform():
    """Return a HankelTransform instance with small deterministic internal grids."""
    transform = HankelTransform.__new__(HankelTransform)

    transform.k = {0: np.array([1.0, 2.0, 4.0])}
    transform.r = {0: np.array([10.0, 20.0])}
    transform.j = {
        0: np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
    }
    transform.j_next_at_zeros = {0: np.array([1.0, 2.0, 4.0])}
    transform.zeros = {0: np.array([1.0, 2.0, 4.0])}
    transform.normalization = {0: 0.5}

    return transform


def test_available_orders_returns_precomputed_orders():
    """Test that available_orders reports the stored Bessel orders."""
    transform = make_fake_transform()

    assert transform.available_orders() == (0,)


def test_check_order_accepts_available_order():
    """Test that _check_order accepts precomputed Bessel orders."""
    transform = make_fake_transform()

    transform._check_order(0)


def test_check_order_rejects_missing_order():
    """Test that _check_order rejects Bessel orders that were not precomputed."""
    transform = make_fake_transform()

    with pytest.raises(ValueError, match="Order 2 was not precomputed"):
        transform._check_order(2)


def test_evaluate_tabulated_spectrum_interpolates_to_internal_k_grid():
    """Test that tabulated spectra are interpolated onto the internal k grid."""
    transform = make_fake_transform()
    k_input = np.array([1.0, 2.0, 4.0])
    spectrum = np.array([10.0, 20.0, 40.0])

    result = transform._evaluate_tabulated_spectrum(
        k_input=k_input,
        spectrum=spectrum,
        order=0,
    )

    expected = np.array([10.0, 20.0, 40.0])

    np.testing.assert_allclose(result, expected)


def test_evaluate_tabulated_spectrum_sets_out_of_range_values_to_zero():
    """Test that interpolation returns zero outside the tabulated k range."""
    transform = make_fake_transform()
    transform.k = {0: np.array([0.5, 1.5, 3.0, 8.0])}
    k_input = np.array([1.0, 2.0, 4.0])
    spectrum = np.array([10.0, 20.0, 40.0])

    result = transform._evaluate_tabulated_spectrum(
        k_input=k_input,
        spectrum=spectrum,
        order=0,
    )

    expected = np.array([0.0, 15.0, 30.0, 0.0])

    np.testing.assert_allclose(result, expected)


def test_evaluate_tabulated_spectrum_applies_taper_when_requested(monkeypatch):
    """Test that tabulated spectra are tapered before interpolation if requested."""
    transform = make_fake_transform()
    k_input = np.array([1.0, 2.0, 4.0])
    spectrum = np.array([10.0, 20.0, 40.0])

    def fake_apply_taper_spectrum(k, pk, **kwargs):
        return pk * 2.0

    monkeypatch.setattr(
        f"{HANKEL_MODULE}.apply_taper_spectrum",
        fake_apply_taper_spectrum,
    )

    result = transform._evaluate_tabulated_spectrum(
        k_input=k_input,
        spectrum=spectrum,
        order=0,
        taper=True,
    )

    expected = spectrum * 2.0

    np.testing.assert_allclose(result, expected)


def test_evaluate_spectrum_calls_callable_on_internal_k_grid():
    """Test that callable spectra are evaluated on the internal k grid."""
    transform = make_fake_transform()

    def spectrum(k, amplitude):
        return amplitude * k

    result = transform._evaluate_spectrum(
        spectrum,
        order=0,
        amplitude=3.0,
    )

    expected = 3.0 * transform.k[0]

    np.testing.assert_allclose(result, expected)


def test_evaluate_spectrum_requires_k_input_for_tabulated_spectra():
    """Test that tabulated spectra require an input k grid."""
    transform = make_fake_transform()

    with pytest.raises(ValueError, match="k_input must be supplied"):
        transform._evaluate_spectrum(
            np.array([1.0, 2.0, 3.0]),
            order=0,
        )


def test_evaluate_spectrum_rejects_wrong_callable_shape():
    """Test that callable spectra must match the internal k-grid shape."""
    transform = make_fake_transform()

    def bad_spectrum(k):
        return np.ones(k.size + 1)

    with pytest.raises(ValueError, match="Evaluated spectrum must match"):
        transform._evaluate_spectrum(
            bad_spectrum,
            order=0,
        )


def test_evaluate_spectrum_rejects_nonfinite_callable_values():
    """Test that callable spectra must return finite values."""
    transform = make_fake_transform()

    def bad_spectrum(k):
        return np.array([1.0, np.nan, 3.0])

    with pytest.raises(ValueError, match="finite values"):
        transform._evaluate_spectrum(
            bad_spectrum,
            order=0,
        )


def test_pk_grid_requires_power_spectrum():
    """Test that pk_grid requires a supplied power spectrum."""
    transform = make_fake_transform()

    with pytest.raises(ValueError, match="pk must be supplied"):
        transform.pk_grid(order=0)


def test_pk_grid_returns_spectrum_on_internal_grid():
    """Test that pk_grid returns the spectrum evaluated on the Hankel k grid."""
    transform = make_fake_transform()

    result = transform.pk_grid(
        k_pk=np.array([1.0, 2.0, 4.0]),
        pk=np.array([10.0, 20.0, 40.0]),
        order=0,
    )

    expected = np.array([10.0, 20.0, 40.0])

    np.testing.assert_allclose(result, expected)


def test_project_spectra_to_radial_projects_one_spectrum():
    """Test that one spectrum is projected into a radial statistic."""
    transform = make_fake_transform()
    spectrum = np.array([2.0, 4.0, 8.0])

    r, result = transform._project_spectra_to_radial([spectrum], order=0)

    weighted = spectrum / transform.j_next_at_zeros[0] ** 2
    expected = np.dot(transform.j[0], weighted) * transform.normalization[0]

    np.testing.assert_allclose(r, transform.r[0])
    np.testing.assert_allclose(result, expected)


def test_project_spectra_to_radial_projects_two_spectra_to_matrix():
    """Test that two spectra are projected into a radial covariance matrix."""
    transform = make_fake_transform()
    spectrum_1 = np.array([2.0, 4.0, 8.0])
    spectrum_2 = np.array([1.0, 3.0, 5.0])

    r, result = transform._project_spectra_to_radial(
        [spectrum_1, spectrum_2],
        order=0,
    )

    product = spectrum_1 * spectrum_2
    weighted = product / transform.j_next_at_zeros[0] ** 2
    j_matrix = transform.j[0]
    expected = np.dot(j_matrix, (j_matrix * weighted).T) * transform.normalization[0]

    np.testing.assert_allclose(r, transform.r[0])
    np.testing.assert_allclose(result, expected)
    assert result.shape == (transform.r[0].size, transform.r[0].size)


def test_project_spectra_to_radial_projects_three_spectra_to_tensor():
    """Test that three spectra are projected into a third-order radial tensor."""
    transform = make_fake_transform()
    spectrum_1 = np.array([2.0, 4.0, 8.0])
    spectrum_2 = np.array([1.0, 3.0, 5.0])
    spectrum_3 = np.array([2.0, 1.0, 4.0])

    r, result = transform._project_spectra_to_radial(
        [spectrum_1, spectrum_2, spectrum_3],
        order=0,
    )

    product = spectrum_1 * spectrum_2 * spectrum_3
    weighted = product / transform.j_next_at_zeros[0] ** 2
    j_matrix = transform.j[0]
    expected = (
        np.einsum("az,bz,cz,z->abc", j_matrix, j_matrix, j_matrix, weighted)
        * transform.normalization[0]
    )

    np.testing.assert_allclose(r, transform.r[0])
    np.testing.assert_allclose(result, expected)
    assert result.shape == (
        transform.r[0].size,
        transform.r[0].size,
        transform.r[0].size,
    )


def test_project_spectra_to_radial_rejects_more_than_three_spectra():
    """Test that radial projection rejects unsupported spectrum products."""
    transform = make_fake_transform()
    spectra = [np.ones_like(transform.k[0]) for _ in range(4)]

    with pytest.raises(ValueError, match="Only 1, 2, or 3 spectra are supported"):
        transform._project_spectra_to_radial(spectra, order=0)


def test_projected_correlation_projects_one_input_spectrum():
    """Test that projected_correlation evaluates and projects one spectrum."""
    transform = make_fake_transform()
    pk = np.array([2.0, 4.0, 8.0])

    r, result = transform.projected_correlation(
        k_pk=np.array([1.0, 2.0, 4.0]),
        pk=pk,
        order=0,
    )

    weighted = pk / transform.j_next_at_zeros[0] ** 2
    expected = np.dot(transform.j[0], weighted) * transform.normalization[0]

    np.testing.assert_allclose(r, transform.r[0])
    np.testing.assert_allclose(result, expected)


def test_spherical_correlation_projects_k_weighted_spectrum():
    """Test that spherical_correlation projects the k-weighted spectrum."""
    transform = make_fake_transform()
    pk = np.array([2.0, 4.0, 8.0])

    r, result = transform.spherical_correlation(
        k_pk=np.array([1.0, 2.0, 4.0]),
        pk=pk,
        order=0,
    )

    weighted_spectrum = pk * transform.k[0]
    weighted = weighted_spectrum / transform.j_next_at_zeros[0] ** 2
    expected = np.dot(transform.j[0], weighted) * transform.normalization[0]

    np.testing.assert_allclose(r, transform.r[0])
    np.testing.assert_allclose(result, expected)


def test_projected_covariance_projects_two_input_spectra():
    """Test that projected_covariance evaluates and projects two spectra."""
    transform = make_fake_transform()
    pk1 = np.array([2.0, 4.0, 8.0])
    pk2 = np.array([1.0, 3.0, 5.0])

    r, result = transform.projected_covariance(
        k_pk=np.array([1.0, 2.0, 4.0]),
        pk1=pk1,
        pk2=pk2,
        order=0,
    )

    product = pk1 * pk2
    weighted = product / transform.j_next_at_zeros[0] ** 2
    expected = np.dot(transform.j[0], (transform.j[0] * weighted).T) * transform.normalization[0]

    np.testing.assert_allclose(r, transform.r[0])
    np.testing.assert_allclose(result, expected)


def test_projected_skewness_projects_three_input_spectra():
    """Test that projected_skewness evaluates and projects three spectra."""
    transform = make_fake_transform()
    pk1 = np.array([2.0, 4.0, 8.0])
    pk2 = np.array([1.0, 3.0, 5.0])
    pk3 = np.array([2.0, 1.0, 4.0])

    r, result = transform.projected_skewness(
        k_pk=np.array([1.0, 2.0, 4.0]),
        pk1=pk1,
        pk2=pk2,
        pk3=pk3,
        order=0,
    )

    product = pk1 * pk2 * pk3
    weighted = product / transform.j_next_at_zeros[0] ** 2
    expected = (
        np.einsum(
            "az,bz,cz,z->abc",
            transform.j[0],
            transform.j[0],
            transform.j[0],
            weighted,
        )
        * transform.normalization[0]
    )

    np.testing.assert_allclose(r, transform.r[0])
    np.testing.assert_allclose(result, expected)


def test_projected_correlation_requires_power_spectrum():
    """Test that projected_correlation requires a supplied power spectrum."""
    transform = make_fake_transform()

    with pytest.raises(ValueError, match="pk must be supplied"):
        transform.projected_correlation(order=0)


def test_spherical_correlation_requires_power_spectrum():
    """Test that spherical_correlation requires a supplied power spectrum."""
    transform = make_fake_transform()

    with pytest.raises(ValueError, match="pk must be supplied"):
        transform.spherical_correlation(order=0)


def test_projected_covariance_requires_two_power_spectra():
    """Test that projected_covariance requires both input spectra."""
    transform = make_fake_transform()

    with pytest.raises(ValueError, match="pk1 and pk2 must both be supplied"):
        transform.projected_covariance(pk1=np.ones(3), order=0)


def test_projected_skewness_requires_three_power_spectra():
    """Test that projected_skewness requires all three input spectra."""
    transform = make_fake_transform()

    with pytest.raises(ValueError, match="pk1, pk2, and pk3 must all be supplied"):
        transform.projected_skewness(pk1=np.ones(3), pk2=np.ones(3), order=0)


def test_bin_radial_matrix_delegates_valid_inputs(monkeypatch):
    """Test that bin_radial_matrix delegates valid radial binning inputs."""
    transform = make_fake_transform()
    r = np.array([1.0, 2.0, 3.0])
    matrix = np.arange(9.0).reshape(3, 3)
    r_bins = np.array([1.0, 2.0, 3.0])
    expected_centers = np.array([1.5, 2.5])
    expected_matrix = np.eye(2)

    def fake_compute_bin_radial_matrix(r_arg, matrix_arg, r_bins_arg):
        np.testing.assert_allclose(r_arg, r)
        np.testing.assert_allclose(matrix_arg, matrix)
        np.testing.assert_allclose(r_bins_arg, r_bins)
        return expected_centers, expected_matrix

    monkeypatch.setattr(
        f"{HANKEL_MODULE}.compute_bin_radial_matrix",
        fake_compute_bin_radial_matrix,
    )

    centers, binned = transform.bin_radial_matrix(r, matrix, r_bins)

    np.testing.assert_allclose(centers, expected_centers)
    np.testing.assert_allclose(binned, expected_matrix)


def test_bin_radial_matrix_rejects_wrong_matrix_shape():
    """Test that bin_radial_matrix rejects matrices inconsistent with r."""
    transform = make_fake_transform()
    r = np.array([1.0, 2.0, 3.0])
    matrix = np.ones((3, 2))
    r_bins = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="matrix shape must be"):
        transform.bin_radial_matrix(r, matrix, r_bins)


def test_bin_radial_matrix_rejects_nonpositive_bin_edges():
    """Test that bin_radial_matrix rejects non-positive radial bin edges."""
    transform = make_fake_transform()
    r = np.array([1.0, 2.0, 3.0])
    matrix = np.ones((3, 3))
    r_bins = np.array([0.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="r_bins must contain only positive values"):
        transform.bin_radial_matrix(r, matrix, r_bins)


def test_correlation_matrix_delegates_valid_covariance(monkeypatch):
    """Test that correlation_matrix delegates valid covariance matrices."""
    transform = make_fake_transform()
    covariance = np.array([[4.0, 2.0], [2.0, 9.0]])
    expected = np.array([[1.0, 1.0 / 3.0], [1.0 / 3.0, 1.0]])

    def fake_compute_correlation_matrix(covariance_arg):
        np.testing.assert_allclose(covariance_arg, covariance)
        return expected

    monkeypatch.setattr(
        f"{HANKEL_MODULE}.compute_correlation_matrix",
        fake_compute_correlation_matrix,
    )

    result = transform.correlation_matrix(covariance)

    np.testing.assert_allclose(result, expected)


def test_correlation_matrix_rejects_non_square_covariance():
    """Test that correlation_matrix rejects non-square covariance matrices."""
    transform = make_fake_transform()

    with pytest.raises(ValueError, match="covariance must be square"):
        transform.correlation_matrix(np.ones((2, 3)))


def test_diagonal_error_delegates_valid_covariance(monkeypatch):
    """Test that diagonal_error delegates valid covariance matrices."""
    transform = make_fake_transform()
    covariance = np.array([[4.0, 2.0], [2.0, 9.0]])
    expected = np.array([2.0, 3.0])

    def fake_compute_diagonal_error(covariance_arg):
        np.testing.assert_allclose(covariance_arg, covariance)
        return expected

    monkeypatch.setattr(
        f"{HANKEL_MODULE}.compute_diagonal_error",
        fake_compute_diagonal_error,
    )

    result = transform.diagonal_error(covariance)

    np.testing.assert_allclose(result, expected)


def test_diagonal_error_rejects_non_square_covariance():
    """Test that diagonal_error rejects non-square covariance matrices."""
    transform = make_fake_transform()

    with pytest.raises(ValueError, match="covariance must be square"):
        transform.diagonal_error(np.ones((2, 3)))


def test_taper_spectrum_delegates_valid_power_spectrum(monkeypatch):
    """Test that taper_spectrum validates and delegates tapering."""
    transform = make_fake_transform()
    k = np.array([1.0, 2.0, 4.0])
    pk = np.array([10.0, 20.0, 40.0])
    expected = pk * 0.5

    def fake_apply_taper_spectrum(k_arg, pk_arg, **kwargs):
        np.testing.assert_allclose(k_arg, k)
        np.testing.assert_allclose(pk_arg, pk)
        assert kwargs["alpha"] == 0.25
        return expected

    monkeypatch.setattr(
        f"{HANKEL_MODULE}.apply_taper_spectrum",
        fake_apply_taper_spectrum,
    )

    result = transform.taper_spectrum(k, pk, alpha=0.25)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"r_min": 0.0},
        {"r_max": 0.0},
        {"k_min": 0.0},
        {"k_max": 0.0},
        {"r_min": 2.0, "r_max": 1.0},
        {"k_min": 2.0, "k_max": 1.0},
        {"n_zeros": 0},
        {"n_zeros_step": 0},
        {"max_iterations": 0},
        {"orders": ()},
        {"orders": (-1,)},
        {"orders": (np.nan,)},
    ],
)
def test_init_rejects_invalid_inputs(monkeypatch, kwargs):
    """Test that HankelTransform initialization rejects invalid configuration."""
    monkeypatch.setattr(HankelTransform, "_build_all_grids", lambda self: None)

    with pytest.raises(ValueError):
        HankelTransform(**kwargs)


def test_prune_radial_grid_returns_original_grid_when_pruning_disabled(monkeypatch):
    """Test that radial pruning can be disabled."""
    monkeypatch.setattr(HankelTransform, "_build_all_grids", lambda self: None)
    transform = HankelTransform(prune_r=None)
    r = np.array([1.0, 2.0, 3.0])

    result = transform._prune_radial_grid(r)

    np.testing.assert_allclose(result, r)


def test_prune_radial_grid_keeps_endpoints_for_linear_pruning(monkeypatch):
    """Test that linearly pruned radial grids keep both endpoints."""
    monkeypatch.setattr(HankelTransform, "_build_all_grids", lambda self: None)
    transform = HankelTransform(prune_r=2, prune_log_space=False)
    r = np.arange(1.0, 8.0)

    result = transform._prune_radial_grid(r)

    assert result[0] == r[0]
    assert result[-1] == r[-1]


def test_select_radial_range_selects_grid_covering_requested_range(monkeypatch):
    """Test that radial range selection includes bracketing grid points."""
    monkeypatch.setattr(HankelTransform, "_build_all_grids", lambda self: None)
    transform = HankelTransform(r_min=2.5, r_max=6.5)
    r = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 9.0])

    result = transform._select_radial_range(r)

    expected = np.array([2.0, 3.0, 5.0, 7.0])

    np.testing.assert_allclose(result, expected)
