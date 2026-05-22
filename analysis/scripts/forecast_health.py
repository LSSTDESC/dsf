"""Numerical health checks for forecast data vectors, covariances, and Fisher matrices."""

from __future__ import annotations

import numpy as np


def has_finite_values(array):
    """Return whether all values in an array are finite."""
    array = np.asarray(array, dtype=float)

    return bool(np.all(np.isfinite(array)))


def count_nonfinite_values(array):
    """Return the number of non-finite values in an array."""
    array = np.asarray(array, dtype=float)

    return int(np.size(array) - np.count_nonzero(np.isfinite(array)))


def matrix_is_symmetric(matrix, rtol=1e-10, atol=1e-12):
    """Return whether a matrix is symmetric within numerical tolerance."""
    matrix = np.asarray(matrix, dtype=float)

    return bool(np.allclose(matrix, matrix.T, rtol=rtol, atol=atol))


def matrix_eigenvalues(matrix):
    """Return the eigenvalues of a symmetric matrix."""
    matrix = np.asarray(matrix, dtype=float)

    return np.linalg.eigvalsh(matrix)


def matrix_min_eigenvalue(matrix):
    """Return the minimum eigenvalue of a symmetric matrix."""
    eigenvalues = matrix_eigenvalues(matrix)

    return float(np.min(eigenvalues))


def matrix_condition_number(matrix):
    """Return the condition number of a matrix."""
    matrix = np.asarray(matrix, dtype=float)

    return float(np.linalg.cond(matrix))


def matrix_is_positive_definite(matrix):
    """Return whether a symmetric matrix is positive definite."""
    matrix = np.asarray(matrix, dtype=float)

    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False

    return True


def covariance_health(covariance):
    """Return numerical health diagnostics for a covariance matrix."""
    covariance = np.asarray(covariance, dtype=float)

    return {
        "shape": covariance.shape,
        "finite": has_finite_values(covariance),
        "n_nonfinite": count_nonfinite_values(covariance),
        "symmetric": matrix_is_symmetric(covariance),
        "positive_definite": matrix_is_positive_definite(covariance),
        "min_eigenvalue": matrix_min_eigenvalue(covariance),
        "condition_number": matrix_condition_number(covariance),
    }


def fisher_health(fisher_matrix):
    """Return numerical health diagnostics for a Fisher matrix."""
    fisher_matrix = np.asarray(fisher_matrix, dtype=float)

    return {
        "shape": fisher_matrix.shape,
        "finite": has_finite_values(fisher_matrix),
        "n_nonfinite": count_nonfinite_values(fisher_matrix),
        "symmetric": matrix_is_symmetric(fisher_matrix),
        "positive_definite": matrix_is_positive_definite(fisher_matrix),
        "min_eigenvalue": matrix_min_eigenvalue(fisher_matrix),
        "condition_number": matrix_condition_number(fisher_matrix),
    }


def data_vector_health(data_vector):
    """Return numerical health diagnostics for a forecast data vector."""
    data_vector = np.asarray(data_vector, dtype=float)

    finite = np.isfinite(data_vector)

    if np.any(finite):
        minimum = float(np.min(data_vector[finite]))
        maximum = float(np.max(data_vector[finite]))
        absolute_maximum = float(np.max(np.abs(data_vector[finite])))
    else:
        minimum = np.nan
        maximum = np.nan
        absolute_maximum = np.nan

    return {
        "size": int(data_vector.size),
        "finite": has_finite_values(data_vector),
        "n_nonfinite": count_nonfinite_values(data_vector),
        "minimum": minimum,
        "maximum": maximum,
        "absolute_maximum": absolute_maximum,
        "n_zero": int(np.count_nonzero(data_vector == 0.0)),
    }


def relative_parameter_errors(errors, theta0):
    """Return parameter errors divided by absolute fiducial values.

    Parameters with zero fiducial value are assigned ``np.inf`` unless their
    error is also zero.
    """
    errors = np.asarray(errors, dtype=float)
    theta0 = np.asarray(theta0, dtype=float)

    relative_errors = np.full_like(errors, np.inf, dtype=float)

    nonzero = np.abs(theta0) > 0.0
    relative_errors[nonzero] = errors[nonzero] / np.abs(theta0[nonzero])

    zero_and_zero_error = (~nonzero) & (errors == 0.0)
    relative_errors[zero_and_zero_error] = 0.0

    return relative_errors


def unstable_parameter_flags(errors, theta0, threshold=1.0):
    """Return boolean flags for parameters with large relative errors.

    A parameter is flagged when ``sigma / abs(fiducial)`` is larger than the
    chosen threshold. This is a simple forecast diagnostic rather than a formal
    failure criterion.
    """
    relative_errors = relative_parameter_errors(errors, theta0)

    return relative_errors > threshold


def parameter_health(errors, theta0, parameter_names=None, threshold=1.0):
    """Return compact health diagnostics for forecast parameter errors."""
    errors = np.asarray(errors, dtype=float)
    theta0 = np.asarray(theta0, dtype=float)

    if parameter_names is None:
        parameter_names = list(range(errors.size))

    relative_errors = relative_parameter_errors(errors, theta0)
    unstable_flags = unstable_parameter_flags(errors, theta0, threshold=threshold)

    return {
        "finite_errors": has_finite_values(errors),
        "n_nonfinite_errors": count_nonfinite_values(errors),
        "relative_errors": relative_errors,
        "unstable_flags": unstable_flags,
        "unstable_parameters": [
            name for name, flagged in zip(parameter_names, unstable_flags, strict=True) if flagged
        ],
    }


def health_flag_summary(health):
    """Return a compact list of warning labels from a health dictionary."""
    flags = []

    if not health.get("finite", True):
        flags.append("nonfinite_values")

    if health.get("n_nonfinite", 0) > 0:
        flags.append("has_nan_or_inf")

    if health.get("symmetric") is False:
        flags.append("not_symmetric")

    if health.get("positive_definite") is False:
        flags.append("not_positive_definite")

    condition_number = health.get("condition_number")

    if condition_number is not None and np.isfinite(condition_number):
        if condition_number > 1.0e12:
            flags.append("ill_conditioned")

    min_eigenvalue = health.get("min_eigenvalue")

    if min_eigenvalue is not None and np.isfinite(min_eigenvalue):
        if min_eigenvalue <= 0.0:
            flags.append("nonpositive_eigenvalue")

    return flags


def forecast_health_summary(
    *,
    data_vector=None,
    covariance=None,
    fisher_matrix=None,
    parameter_errors=None,
    theta0=None,
    parameter_names=None,
    relative_error_threshold=1.0,
):
    """Return a combined health summary for one forecast run."""
    summary = {}

    if data_vector is not None:
        summary["data_vector"] = data_vector_health(data_vector)

    if covariance is not None:
        summary["covariance"] = covariance_health(covariance)
        summary["covariance_flags"] = health_flag_summary(summary["covariance"])

    if fisher_matrix is not None:
        summary["fisher"] = fisher_health(fisher_matrix)
        summary["fisher_flags"] = health_flag_summary(summary["fisher"])

    if parameter_errors is not None and theta0 is not None:
        summary["parameters"] = parameter_health(
            parameter_errors,
            theta0,
            parameter_names=parameter_names,
            threshold=relative_error_threshold,
        )

    return summary


def print_health_summary(name, summary):
    """Print a readable forecast health summary."""
    print()
    print(f"{name} health summary")

    if "data_vector" in summary:
        health = summary["data_vector"]
        print(
            "data vector: "
            f"size={health['size']}, "
            f"finite={health['finite']}, "
            f"nonfinite={health['n_nonfinite']}, "
            f"zeros={health['n_zero']}"
        )

    if "covariance" in summary:
        health = summary["covariance"]
        flags = summary.get("covariance_flags", [])
        print(
            "covariance: "
            f"shape={health['shape']}, "
            f"finite={health['finite']}, "
            f"symmetric={health['symmetric']}, "
            f"positive_definite={health['positive_definite']}, "
            f"min_eig={health['min_eigenvalue']:.3e}, "
            f"cond={health['condition_number']:.3e}, "
            f"flags={flags}"
        )

    if "fisher" in summary:
        health = summary["fisher"]
        flags = summary.get("fisher_flags", [])
        print(
            "fisher: "
            f"shape={health['shape']}, "
            f"finite={health['finite']}, "
            f"symmetric={health['symmetric']}, "
            f"positive_definite={health['positive_definite']}, "
            f"min_eig={health['min_eigenvalue']:.3e}, "
            f"cond={health['condition_number']:.3e}, "
            f"flags={flags}"
        )

    if "parameters" in summary:
        health = summary["parameters"]
        print(
            "parameters: "
            f"finite_errors={health['finite_errors']}, "
            f"nonfinite_errors={health['n_nonfinite_errors']}, "
            f"unstable={health['unstable_parameters']}"
        )
