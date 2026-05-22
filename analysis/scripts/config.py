"""Configuration scripts for DSF analysis runs."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from dsf.utils.converters import redshift_to_scale_factor


def load_config(path):
    """Load one YAML analysis configuration."""
    with Path(path).open("r") as stream:
        return yaml.safe_load(stream)


def parameter_entries(config):
    """Return all parameter entries from the run configuration."""
    return list(config["parameters"])


def fiducial_parameter_entries(config):
    """Return all parameters that define the fiducial model."""
    return parameter_entries(config)


def varied_parameter_entries(config):
    """Return parameters varied in the forecast."""
    return [entry for entry in parameter_entries(config) if bool(entry.get("vary", True))]


def parameter_names(config):
    """Return varied parameter names in theta-vector order."""
    return [entry["name"] for entry in varied_parameter_entries(config)]


def parameter_labels(config):
    """Return varied parameter labels in theta-vector order."""
    return [entry.get("label", entry["name"]) for entry in varied_parameter_entries(config)]


def theta0(config):
    """Return fiducial theta values for varied parameters."""
    return np.array(
        [float(entry["fiducial"]) for entry in varied_parameter_entries(config)],
        dtype=float,
    )


def sigma_prior(config):
    """Return Gaussian prior widths for varied parameters."""
    return np.array(
        [float(entry["prior_sigma"]) for entry in varied_parameter_entries(config)],
        dtype=float,
    )


def fiducial_values_by_target(config):
    """Return fiducial values keyed by model target."""
    return {
        entry["target"]: float(entry["fiducial"]) for entry in fiducial_parameter_entries(config)
    }


def varied_targets(config):
    """Return varied model targets in theta-vector order."""
    return [entry["target"] for entry in varied_parameter_entries(config)]


def set_nested_value(config, target, value):
    """Set one dotted target path in a copied run configuration."""
    keys = target.split(".")
    block = config

    for key in keys[:-1]:
        block = block[key]

    block[keys[-1]] = value


def config_with_parameter_values(config, values_by_target):
    """Return a copied config with supplied parameter values applied."""
    updated_config = deepcopy(config)

    for target, value in values_by_target.items():
        set_nested_value(updated_config, target, value)

    return updated_config


def fiducial_config(config):
    """Return a copied config with all fiducial parameter values applied."""
    return config_with_parameter_values(
        config,
        fiducial_values_by_target(config),
    )


def geomspace_from_config(config_block):
    """Return a logarithmically spaced array from a config block."""
    return np.geomspace(
        float(config_block["min"]),
        float(config_block["max"]),
        int(config_block["n"]),
    )


def linspace_from_config(config_block):
    """Return a linearly spaced array from a config block."""
    return np.linspace(
        float(config_block["min"]),
        float(config_block["max"]),
        int(config_block["n"]),
    )


def halo_k_array(config):
    """Return the halo-model wavenumber grid."""
    return geomspace_from_config(config["halo_model"]["k"])


def halo_redshift_array(config):
    """Return the halo-model redshift grid."""
    return linspace_from_config(config["halo_model"]["redshift"])


def halo_scale_factor_array(config):
    """Return the halo-model scale-factor grid in increasing order."""
    redshift = halo_redshift_array(config)
    scale_factor = redshift_to_scale_factor(redshift)

    return np.sort(scale_factor)


def covariance_config(config):
    """Return covariance settings from the run configuration."""
    return dict(config.get("covariance", {}))


def hankel_kwargs(config):
    """Return Hankel-transform settings from the covariance config."""
    return covariance_config(config).get("hankel_kwargs")


def covariance_kind(config):
    """Return the covariance kind requested by the run configuration."""
    return covariance_config(config).get("kind", "gm")


def covariance_nonlinear(config):
    """Return whether the covariance should use nonlinear matter power."""
    return bool(covariance_config(config).get("nonlinear", True))


def covariance_galaxy_bias(config):
    """Return the galaxy-bias value requested by the covariance config."""
    return covariance_config(config).get("galaxy_bias")


def covariance_galaxy_bias_prefactor(config):
    """Return the default galaxy-bias prefactor for covariance calculations."""
    return float(covariance_config(config).get("galaxy_bias_prefactor", 1.0))
