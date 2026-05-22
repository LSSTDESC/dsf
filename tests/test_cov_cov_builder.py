"""Unit tests for ``src.dsf.covariance.cov_builder``."""

import numpy as np

from src.dsf.covariance.cov_builder import DeltaSigmaCovarianceBuilder


def test_selected_pairs_returns_stored_pairs_as_integer_tuples():
    """Tests that stored bin pairs are returned as integer tuples."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 2), (1, 3)]

    assert builder.selected_pairs() == [(0, 2), (1, 3)]


def test_selected_pairs_casts_requested_pairs_to_integer_tuples():
    """Tests that requested bin pairs are cast to integer tuples."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 1)]

    actual = builder.selected_pairs([(0.0, 2.0), ("1", "3")])

    assert actual == [(0, 2), (1, 3)]


def test_bin_value_returns_scalar_value_for_all_bins():
    """Tests that scalar bin values are returned unchanged."""
    assert DeltaSigmaCovarianceBuilder.bin_value(1.5, 3) == 1.5


def test_bin_value_selects_sequence_value_by_bin_index():
    """Tests that sequence bin values are selected by index."""
    assert DeltaSigmaCovarianceBuilder.bin_value([1.0, 2.0, 3.0], 1) == 2.0


def test_bin_value_selects_mapping_value_by_bin_index():
    """Tests that mapping bin values are selected by index."""
    assert DeltaSigmaCovarianceBuilder.bin_value({0: 1.0, 2: 3.0}, 2) == 3.0


def test_block_diagonal_from_outputs_preserves_pair_order():
    """Tests that block-diagonal assembly preserves output pair order."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    outputs = {
        (0, 2): {"cov": np.array([[1.0, 0.1], [0.1, 2.0]])},
        (1, 3): {"cov": np.array([[3.0]])},
    }

    pairs, covariance = builder.block_diagonal_from_outputs(outputs, "cov")

    assert pairs == [(0, 2), (1, 3)]
    np.testing.assert_allclose(
        covariance,
        np.array(
            [
                [1.0, 0.1, 0.0],
                [0.1, 2.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        ),
    )


def test_correlation_matrix_delegates_to_hankel():
    """Tests that correlation matrices are delegated to the Hankel object."""

    class DummyHankel:
        def correlation_matrix(self, covariance):
            return covariance + 1.0

    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.hankel = DummyHankel()

    actual = builder.correlation_matrix(np.eye(2))

    np.testing.assert_allclose(actual, np.eye(2) + 1.0)


def test_diagonal_error_delegates_to_hankel():
    """Tests that diagonal errors are delegated to the Hankel object."""

    class DummyHankel:
        def diagonal_error(self, covariance):
            return np.sqrt(np.diag(covariance))

    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.hankel = DummyHankel()

    actual = builder.diagonal_error(np.diag([4.0, 9.0]))

    np.testing.assert_allclose(actual, np.array([2.0, 3.0]))


def test_gm_block_diagonal_uses_gm_outputs():
    """Tests that gm block diagonal uses gm covariance outputs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)

    def gm_covariance_for_pairs(bin_pairs=None):
        return {
            (0, 2): {"cov_gm_gm": np.eye(2)},
            (1, 3): {"cov_gm_gm": 2.0 * np.eye(1)},
        }

    builder.gm_covariance_for_pairs = gm_covariance_for_pairs

    pairs, covariance = builder.gm_block_diagonal()

    assert pairs == [(0, 2), (1, 3)]
    np.testing.assert_allclose(covariance, np.diag([1.0, 1.0, 2.0]))


def test_gg_block_diagonal_uses_gg_outputs():
    """Tests that gg block diagonal uses gg covariance outputs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)

    def gg_covariance_for_pairs(bin_pairs=None):
        return {
            (0, 2): {"cov_gg_gg": np.eye(1)},
            (1, 3): {"cov_gg_gg": 3.0 * np.eye(2)},
        }

    builder.gg_covariance_for_pairs = gg_covariance_for_pairs

    pairs, covariance = builder.gg_block_diagonal()

    assert pairs == [(0, 2), (1, 3)]
    np.testing.assert_allclose(covariance, np.diag([1.0, 3.0, 3.0]))


def test_cross_block_diagonal_uses_cross_outputs():
    """Tests that cross block diagonal uses gm-gg covariance outputs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)

    def cross_covariance_for_pairs(bin_pairs=None):
        return {
            (0, 2): {"cov_gm_gg": np.array([[1.0]])},
            (1, 3): {"cov_gm_gg": np.array([[2.0, 0.1], [0.1, 3.0]])},
        }

    builder.cross_covariance_for_pairs = cross_covariance_for_pairs

    pairs, covariance = builder.cross_block_diagonal()

    assert pairs == [(0, 2), (1, 3)]
    np.testing.assert_allclose(
        covariance,
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 2.0, 0.1],
                [0.0, 0.1, 3.0],
            ]
        ),
    )


def test_joint_block_diagonal_uses_joint_outputs():
    """Tests that joint block diagonal uses joint covariance outputs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)

    def covariance_for_pairs(bin_pairs=None):
        return {
            (0, 2): {"cov_joint": np.eye(2)},
            (1, 3): {"cov_joint": 4.0 * np.eye(1)},
        }

    builder.covariance_for_pairs = covariance_for_pairs

    pairs, covariance = builder.joint_block_diagonal()

    assert pairs == [(0, 2), (1, 3)]
    np.testing.assert_allclose(covariance, np.diag([1.0, 1.0, 4.0]))


def test_gm_covariance_for_pairs_uses_selected_pairs():
    """Tests that gm covariance is evaluated for selected pairs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 2), (1, 3)]

    def gm_covariance_for_pair(lens_bin, source_bin):
        return {"pair": (lens_bin, source_bin)}

    builder.gm_covariance_for_pair = gm_covariance_for_pair

    actual = builder.gm_covariance_for_pairs()

    assert actual == {
        (0, 2): {"pair": (0, 2)},
        (1, 3): {"pair": (1, 3)},
    }


def test_gg_covariance_for_pairs_uses_selected_pairs():
    """Tests that gg covariance is evaluated for selected pairs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 2), (1, 3)]

    def gg_covariance_for_pair(lens_bin, source_bin):
        return {"pair": (lens_bin, source_bin)}

    builder.gg_covariance_for_pair = gg_covariance_for_pair

    actual = builder.gg_covariance_for_pairs()

    assert actual == {
        (0, 2): {"pair": (0, 2)},
        (1, 3): {"pair": (1, 3)},
    }


def test_cross_covariance_for_pairs_uses_selected_pairs():
    """Tests that cross covariance is evaluated for selected pairs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 2), (1, 3)]

    def cross_covariance_for_pair(lens_bin, source_bin):
        return {"pair": (lens_bin, source_bin)}

    builder.cross_covariance_for_pair = cross_covariance_for_pair

    actual = builder.cross_covariance_for_pairs()

    assert actual == {
        (0, 2): {"pair": (0, 2)},
        (1, 3): {"pair": (1, 3)},
    }


def test_covariance_for_pairs_uses_selected_pairs():
    """Tests that full covariance is evaluated for selected pairs."""
    builder = object.__new__(DeltaSigmaCovarianceBuilder)
    builder.bin_pairs = [(0, 2), (1, 3)]

    def covariance_for_pair(lens_bin, source_bin):
        return {"pair": (lens_bin, source_bin)}

    builder.covariance_for_pair = covariance_for_pair

    actual = builder.covariance_for_pairs()

    assert actual == {
        (0, 2): {"pair": (0, 2)},
        (1, 3): {"pair": (1, 3)},
    }
