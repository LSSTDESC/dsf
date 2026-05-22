"""DerivKit forecast helpers for DSF analyses."""

import numpy as np
from derivkit import ForecastKit


def make_forecast_kit(model, theta0, covariance):
    """Return a DerivKit ForecastKit object."""
    return ForecastKit(
        function=model,
        theta0=theta0,
        cov=covariance,
    )


def selected_parameter_configs(config, parameter_names=None):
    """Return varied parameter configs, optionally restricted by name."""
    varied = [p for p in config["parameters"] if p.get("vary", True)]

    if parameter_names is None:
        return varied

    by_name = {p["name"]: p for p in varied}
    missing = [name for name in parameter_names if name not in by_name]

    if missing:
        raise ValueError(
            f"Requested parameter names are not varied parameters in the config: {missing}"
        )

    return [by_name[name] for name in parameter_names]


def gaussian_prior_from_config(config, parameter_names=None):
    """Return Gaussian prior mean and covariance from parameter prior_sigma values."""
    params = selected_parameter_configs(config, parameter_names)

    means = []
    sigmas = []

    for param in params:
        sigma = param.get("prior_sigma")

        if sigma is None:
            return None

        sigma = float(sigma)

        if sigma <= 0.0:
            raise ValueError(f"Parameter {param['name']} has invalid prior_sigma={sigma}.")

        means.append(float(param["fiducial"]))
        sigmas.append(sigma)

    mean = np.asarray(means, dtype=float)
    covariance = np.diag(np.asarray(sigmas, dtype=float) ** 2)

    return mean, covariance


def gaussian_prior_precision(config, parameter_names=None):
    """Return Gaussian prior precision from parameter prior_sigma values."""
    prior = gaussian_prior_from_config(config, parameter_names)

    if prior is None:
        return None

    _, covariance = prior

    return np.linalg.inv(covariance)


def add_gaussian_prior_to_fisher(fisher, config, parameter_names=None):
    """Return Fisher matrix with Gaussian prior precision added."""
    prior_precision = gaussian_prior_precision(config, parameter_names)

    if prior_precision is None:
        return None

    fisher = np.asarray(fisher, dtype=float)

    if fisher.shape != prior_precision.shape:
        raise ValueError(
            "Prior precision shape does not match Fisher shape. "
            f"Got {prior_precision.shape} and {fisher.shape}."
        )

    return fisher + prior_precision


def dali_prior_terms(config, parameter_names=None):
    """Return DerivKit Gaussian prior terms from parameter prior_sigma values."""
    prior = gaussian_prior_from_config(config, parameter_names)

    if prior is None:
        return None

    mean, covariance = prior

    return [
        (
            "gaussian",
            {
                "mean": mean,
                "cov": covariance,
            },
        )
    ]


def dali_prior_bounds(config):
    """Return DALI sampling bounds from the config."""
    bounds = config.get("dali", {}).get("priors", {}).get("bounds")

    if bounds is None:
        return None

    return [tuple(bound) for bound in bounds]


def run_fisher(model, theta0, covariance, fisher_config, parameter_names=None, config=None):
    """Run Fisher forecast with and without Gaussian priors."""
    if not fisher_config.get("run", True):
        print("[Fisher] Skipping Fisher forecast.")
        return None

    print("[Fisher] Running Fisher forecast...")

    kit = make_forecast_kit(model, theta0, covariance)

    fisher = kit.fisher(
        method=fisher_config.get("method", "finite"),
        stepsize=float(fisher_config.get("stepsize", 1.0e-2)),
        num_points=int(fisher_config.get("num_points", 5)),
        extrapolation=fisher_config.get("extrapolation", "ridders"),
        levels=int(fisher_config.get("levels", 4)),
    )

    fisher_with_priors = None

    if config is not None:
        fisher_with_priors = add_gaussian_prior_to_fisher(
            fisher,
            config,
            parameter_names,
        )

    print("[Fisher] Done.")
    print(f"[Fisher] Fisher shape: {np.shape(fisher)}")

    return kit, {
        "fisher": fisher,
        "fisher_with_priors": fisher_with_priors,
    }


def run_dali(model, theta0, covariance, dali_config):
    """Run DALI forecast."""
    if not dali_config.get("run", False):
        print("[DALI] Skipping DALI forecast.")
        return None

    print("[DALI] Running DALI forecast...")

    kit = make_forecast_kit(model, theta0, covariance)

    dali = kit.dali(
        forecast_order=int(dali_config.get("forecast_order", 2)),
        method=dali_config.get("method", "finite"),
        stepsize=float(dali_config.get("stepsize", 1.0e-2)),
        num_points=int(dali_config.get("num_points", 5)),
        extrapolation=dali_config.get("extrapolation", "ridders"),
        levels=int(dali_config.get("levels", 4)),
    )

    print("[DALI] Done.")

    return kit, {"dali": dali}


def sample_dali(
    config,
    kit,
    dali,
    names,
    labels,
    *,
    label="DALI",
    include_config_priors=False,
):
    """Sample a DALI posterior as a GetDist sample object."""
    sampling = config.get("dali", {}).get("sampling", {})

    if not sampling.get("run", False):
        print("[DALI] Skipping DALI sampling.")
        return None

    prior_terms = None
    prior_bounds = None

    if include_config_priors:
        prior_terms = dali_prior_terms(config, names)
        prior_bounds = dali_prior_bounds(config)

    print("[DALI] Sampling DALI posterior...")

    samples = kit.getdist_dali_emcee(
        dali=dali,
        names=names,
        labels=labels,
        label=label,
        prior_bounds=prior_bounds,
        prior_terms=prior_terms,
        n_steps=int(sampling.get("n_steps", 10000)),
        burn=int(sampling.get("burn", 2000)),
        thin=int(sampling.get("thin", 2)),
        n_walkers=sampling.get("n_walkers"),
        init_scale=float(sampling.get("init_scale", 1.0e-2)),
        seed=sampling.get("seed"),
    )

    print("[DALI] Sampling done.")
    print(f"[DALI] Sample rows: {samples.numrows}")

    return samples
