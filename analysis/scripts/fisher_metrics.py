"""Diagnostic metrics for Fisher matrices and parameter constraints."""

import numpy as np


def fisher_with_priors(fisher_matrix, prior_sigmas):
    """Return a Fisher matrix with independent Gaussian priors added.

    Each finite prior width contributes ``1 / sigma_prior^2`` to the matching
    diagonal Fisher entry. Parameters with ``None`` or infinite prior widths are
    left unchanged, corresponding to no external prior information.

    Args:
        fisher_matrix: Likelihood Fisher matrix.
        prior_sigmas: Prior standard deviations for each varied parameter.
            Use ``None`` or ``np.inf`` for parameters without priors.

    Returns:
        Fisher matrix including prior information.
    """
    fisher_matrix = np.asarray(fisher_matrix, dtype=float)

    prior_sigmas = [np.inf if sigma is None else sigma for sigma in prior_sigmas]
    prior_sigmas = np.asarray(prior_sigmas, dtype=float)

    prior_fisher = np.zeros_like(fisher_matrix)

    finite_priors = np.isfinite(prior_sigmas)
    prior_fisher[finite_priors, finite_priors] = 1.0 / prior_sigmas[finite_priors] ** 2

    return fisher_matrix + prior_fisher


def parameter_covariance(fisher_matrix):
    """Return the parameter covariance matrix implied by a Fisher matrix.

    This inverts the Fisher matrix. The result is the forecast parameter
    covariance after marginalizing over the other varied parameters.
    """
    fisher_matrix = np.asarray(fisher_matrix, dtype=float)

    return np.linalg.inv(fisher_matrix)


def parameter_errors(fisher_matrix):
    """Return marginalized 1-sigma parameter errors from a Fisher matrix.

    The errors are the square roots of the diagonal entries of the parameter
    covariance matrix. They represent the forecast uncertainty on each parameter
    after marginalizing over the rest.
    """
    covariance = parameter_covariance(fisher_matrix)

    return np.sqrt(np.diag(covariance))


def parameter_correlation_matrix(fisher_matrix):
    """Return the parameter correlation matrix implied by a Fisher matrix.

    The Fisher matrix is first inverted to obtain the marginalized parameter
    covariance. That covariance is then normalized to show dimensionless
    parameter degeneracies and correlations.
    """
    covariance = parameter_covariance(fisher_matrix)
    sigma = np.sqrt(np.diag(covariance))

    return covariance / np.outer(sigma, sigma)


def fisher_figure_of_merit(fisher_matrix):
    """Return the full-parameter Fisher figure of merit.

    The value is computed as ``sqrt(det(F))``. Larger values correspond to a
    smaller forecast parameter volume and therefore tighter joint constraints.
    """
    fisher_matrix = np.asarray(fisher_matrix, dtype=float)

    return np.sqrt(np.linalg.det(fisher_matrix))


def fisher_figure_of_merit_2d(fisher_matrix, parameter_indices):
    """Return a two-parameter marginalized Fisher figure of merit.

    The full Fisher matrix is inverted first, so the selected two-parameter
    covariance includes marginalization over all other parameters. The returned
    value is larger when the marginalized 2D constraint ellipse is smaller.

    Args:
        fisher_matrix: Full Fisher matrix.
        parameter_indices: Two parameter indices defining the parameter plane.

    Returns:
        Inverse square root of the determinant of the marginalized 2D
        parameter covariance.
    """
    covariance = parameter_covariance(fisher_matrix)
    indices = np.asarray(parameter_indices, dtype=int)

    covariance_2d = covariance[np.ix_(indices, indices)]

    return 1.0 / np.sqrt(np.linalg.det(covariance_2d))


def fisher_summary(fisher_matrix, parameter_names=None):
    """Return a compact summary of marginalized Fisher constraints.

    This collects the most common Fisher diagnostics in one dictionary:
    parameter covariance, marginalized errors, parameter correlations, and the
    full Fisher figure of merit. Optional names are used to label the parameter
    errors in a readable mapping.

    Args:
        fisher_matrix: Fisher matrix.
        parameter_names: Optional parameter names. If omitted, integer
            parameter indices are used.

    Returns:
        Dictionary with parameter covariance, errors, correlation matrix,
        and full Fisher figure of merit.
    """
    fisher_matrix = np.asarray(fisher_matrix, dtype=float)

    if parameter_names is None:
        parameter_names = list(range(fisher_matrix.shape[0]))

    covariance = parameter_covariance(fisher_matrix)
    errors = np.sqrt(np.diag(covariance))
    sigma = np.sqrt(np.diag(covariance))
    correlations = covariance / np.outer(sigma, sigma)
    fom = fisher_figure_of_merit(fisher_matrix)

    errors_by_parameter = {name: error for name, error in zip(parameter_names, errors, strict=True)}

    return {
        "covariance": covariance,
        "errors": errors,
        "errors_by_parameter": errors_by_parameter,
        "correlation_matrix": correlations,
        "figure_of_merit": fom,
    }


def fisher_summary_with_priors(fisher_matrix, prior_sigmas, parameter_names=None):
    """Return a Fisher summary after adding independent Gaussian priors.

    This first augments the likelihood Fisher matrix with diagonal prior
    information and then returns the same marginalized diagnostics as
    ``fisher_summary``. It is useful for comparing constraints with and without
    external prior assumptions.
    """
    fisher_total = fisher_with_priors(fisher_matrix, prior_sigmas)

    return fisher_summary(fisher_total, parameter_names=parameter_names)
