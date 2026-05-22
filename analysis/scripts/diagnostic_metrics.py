"""Diagnostic metrics for forecast data vectors and covariances."""

import numpy as np


def signal_to_noise(data_vector, covariance):
    """Return the total covariance-weighted signal-to-noise.

    This computes ``sqrt(d^T C^-1 d)`` for the full data vector. It measures
    how strongly the forecast signal is detected relative to the covariance,
    using all data-vector elements together.
    """
    data_vector = np.asarray(data_vector, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    snr_squared = data_vector @ np.linalg.solve(covariance, data_vector)
    snr = np.sqrt(snr_squared)

    return snr, snr_squared


def signal_to_noise_by_pair(data_vector, covariance, bin_pairs, n_radius):
    """Return signal-to-noise values for each lens-source bin pair.

    The stacked data vector is split into one block per selected lens-source
    pair, with ``n_radius`` radial points per block. Each pair is evaluated with
    its corresponding covariance sub-block, so the result shows which tomographic
    pair contributes most strongly to the total signal.
    """
    data_vector = np.asarray(data_vector, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    snr_by_pair = {}
    snr_squared_by_pair = {}

    for pair_index, pair in enumerate(bin_pairs):
        start = pair_index * n_radius
        stop = (pair_index + 1) * n_radius

        pair_data_vector = data_vector[start:stop]
        pair_covariance = covariance[start:stop, start:stop]

        pair_snr_squared = pair_data_vector @ np.linalg.solve(
            pair_covariance,
            pair_data_vector,
        )
        pair_snr = np.sqrt(pair_snr_squared)

        pair_key = tuple(pair)
        snr_by_pair[pair_key] = pair_snr
        snr_squared_by_pair[pair_key] = pair_snr_squared

    return snr_by_pair, snr_squared_by_pair


def signal_to_noise_by_radius(data_vector, covariance, n_pairs, n_radius):
    """Return signal-to-noise values for each radial bin across all pairs.

    For each radial index, this gathers the corresponding radial point from
    every lens-source pair and evaluates its covariance-weighted SNR. The result
    shows which projected-radius bins carry the strongest signal after combining
    all selected tomographic pairs.
    """
    data_vector = np.asarray(data_vector, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    snr_by_radius = {}
    snr_squared_by_radius = {}

    for radius_index in range(n_radius):
        indices = np.arange(radius_index, n_pairs * n_radius, n_radius)

        radius_data_vector = data_vector[indices]
        radius_covariance = covariance[np.ix_(indices, indices)]

        snr_squared = radius_data_vector @ np.linalg.solve(
            radius_covariance,
            radius_data_vector,
        )
        snr = np.sqrt(snr_squared)

        snr_by_radius[radius_index] = snr
        snr_squared_by_radius[radius_index] = snr_squared

    return snr_by_radius, snr_squared_by_radius


def cumulative_signal_to_noise(data_vector, covariance):
    """Return cumulative signal-to-noise as data-vector elements are added.

    The first value uses only the first element of the data vector, the second
    value uses the first two elements, and so on. This is useful for checking
    where the signal accumulates along the chosen data-vector ordering.
    """
    data_vector = np.asarray(data_vector, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    cumulative_snr = []
    cumulative_snr_squared = []

    for stop in range(1, len(data_vector) + 1):
        sub_vector = data_vector[:stop]
        sub_covariance = covariance[:stop, :stop]

        snr_squared = sub_vector @ np.linalg.solve(sub_covariance, sub_vector)
        cumulative_snr_squared.append(snr_squared)
        cumulative_snr.append(np.sqrt(snr_squared))

    return np.asarray(cumulative_snr), np.asarray(cumulative_snr_squared)


def chi2_difference(data_vector_a, data_vector_b, covariance):
    """Return the covariance-weighted difference between two data vectors.

    This computes ``Delta d^T C^-1 Delta d``, where ``Delta d`` is the
    difference between two model data vectors. It measures how distinguishable
    the two predictions are under the supplied covariance.
    """
    data_vector_a = np.asarray(data_vector_a, dtype=float)
    data_vector_b = np.asarray(data_vector_b, dtype=float)
    covariance = np.asarray(covariance, dtype=float)

    delta = data_vector_a - data_vector_b

    return delta @ np.linalg.solve(covariance, delta)


def reduced_chi2_difference(data_vector_a, data_vector_b, covariance):
    """Return the average covariance-weighted difference per data-vector element.

    This divides the chi-squared difference by the data-vector length. It is a
    compact diagnostic for comparing model differences across data vectors with
    different sizes, but it is not a full goodness-of-fit statistic by itself.
    """
    chi2 = chi2_difference(data_vector_a, data_vector_b, covariance)

    return chi2 / len(data_vector_a)


def correlation_matrix(covariance):
    """Return the correlation matrix corresponding to a covariance matrix.

    Each covariance entry is normalized by the product of the relevant standard
    deviations. The result has unit diagonal and shows the dimensionless
    correlation structure between data-vector elements.
    """
    covariance = np.asarray(covariance, dtype=float)

    sigma = np.sqrt(np.diag(covariance))

    return covariance / np.outer(sigma, sigma)


def standard_deviation(covariance):
    """Return the standard deviation of each data-vector element.

    This extracts ``sqrt(C_ii)`` from the covariance diagonal. These values are
    the absolute forecast uncertainties associated with individual data-vector
    elements before any normalization by the signal.
    """
    covariance = np.asarray(covariance, dtype=float)

    return np.sqrt(np.diag(covariance))


def fractional_uncertainty(data_vector, covariance):
    """Return the fractional uncertainty for each data-vector element.

    This divides the standard deviation by the absolute value of the signal.
    It is useful for checking which data-vector elements are relatively well
    constrained and which are noise-dominated.
    """
    data_vector = np.asarray(data_vector, dtype=float)

    return standard_deviation(covariance) / np.abs(data_vector)
