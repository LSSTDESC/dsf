"""Model setup scripts for DSF analysis scripts."""

from analysis.scripts.config import (
    fiducial_values_by_target,
    halo_k_array,
    halo_scale_factor_array,
    varied_targets,
)
from src.dsf.modelling import (
    make_ccl_cosmology,
    pk2d_hod,
    pk2d_hod_baryonified,
    pk2d_hod_baryonified_with_nla,
    pk2d_hod_with_nla,
    pk2d_nla,
)

PK2D_MODELS = {
    "hod": pk2d_hod,
    "hod_baryonified": pk2d_hod_baryonified,
    "hod_with_nla": pk2d_hod_with_nla,
    "hod_baryonified_with_nla": pk2d_hod_baryonified_with_nla,
    "nla": pk2d_nla,
}


PK2D_MODEL_GROUPS = {
    "hod": ("hod", "pk2d"),
    "hod_baryonified": ("hod", "baryons", "pk2d"),
    "hod_with_nla": ("hod", "ia", "pk2d"),
    "hod_baryonified_with_nla": ("hod", "ia", "baryons", "pk2d"),
    "nla": ("ia", "pk2d"),
}


PK2D_CONFIG_BLOCKS = {
    "hod": "hod",
    "ia": "ia",
    "baryons": "baryons",
}


def pk2d_model_name(config):
    """Return the selected pk2d model name."""
    return config.get("pk2d", {}).get("model", "hod")


def pk2d_model_func(config):
    """Return the selected pk2d model function."""
    model_name = pk2d_model_name(config)

    if model_name not in PK2D_MODELS:
        allowed = ", ".join(sorted(PK2D_MODELS))
        raise ValueError(f"Unsupported pk2d model {model_name!r}. Allowed models are: {allowed}.")

    return PK2D_MODELS[model_name]


def pk2d_model_groups(config):
    """Return config target groups used by the selected pk2d model."""
    model_name = pk2d_model_name(config)

    if model_name not in PK2D_MODEL_GROUPS:
        allowed = ", ".join(sorted(PK2D_MODEL_GROUPS))
        raise ValueError(f"Unsupported pk2d model {model_name!r}. Allowed models are: {allowed}.")

    return PK2D_MODEL_GROUPS[model_name]


def split_targets(values_by_target):
    """Split dotted YAML targets into cosmology and pk2d keyword values."""
    cosmology_values = {}
    pk2d_values = {}

    for target, value in values_by_target.items():
        group, name = target.split(".", 1)

        if group == "cosmology":
            cosmology_values[name] = value
            continue

        if group in ("hod", "ia", "baryons", "pk2d"):
            pk2d_values[target] = value
            continue

        raise ValueError(
            f"Unsupported target {target!r}. "
            "Use targets like cosmology.Omega_m, hod.log10M1_0, "
            "ia.A_IA, baryons.f_c, or pk2d.a_bias."
        )

    return cosmology_values, pk2d_values


def clean_none_values(values):
    """Return a copy with explicit None values removed."""
    return {key: value for key, value in values.items() if value is not None}


def base_pk2d_kwargs(config):
    """Return baseline pk2d keyword arguments from the run config."""
    groups = pk2d_model_groups(config)

    kwargs = {}

    for group in groups:
        block_name = PK2D_CONFIG_BLOCKS.get(group)

        if block_name is not None:
            kwargs.update(config.get(block_name, {}))

    kwargs.update(config.get("pk2d", {}).get("kwargs", {}))

    return clean_none_values(kwargs)


def filter_pk2d_values_by_model(pk2d_values, config):
    """Return varied pk2d values used by the selected pk2d model."""
    groups = set(pk2d_model_groups(config))

    filtered = {}

    for target, value in pk2d_values.items():
        group, name = target.split(".", 1)

        if group in groups:
            filtered[name] = value

    return filtered


def make_cosmo(config, cosmology_values=None):
    """Return a CCL cosmology from config values."""
    values = dict(config["cosmology"])

    if cosmology_values is not None:
        values.update(cosmology_values)

    return make_ccl_cosmology(values)


def make_fiducial_cosmology(config):
    """Return the fiducial CCL cosmology."""
    values_by_target = fiducial_values_by_target(config)
    cosmology_values, _ = split_targets(values_by_target)

    return make_cosmo(config, cosmology_values)


def make_theta_mapper(config):
    """Return a theta-to-builder-kwargs mapper."""
    fiducial = fiducial_values_by_target(config)
    targets = varied_targets(config)

    k_array = halo_k_array(config)
    a_array = halo_scale_factor_array(config)

    base_kwargs = base_pk2d_kwargs(config)

    def theta_mapper(theta, context):
        values_by_target = dict(fiducial)

        for target, value in zip(targets, theta, strict=False):
            values_by_target[target] = float(value)

        cosmology_values, pk2d_values = split_targets(values_by_target)

        pk2d_kwargs = dict(base_kwargs)
        pk2d_kwargs.update(filter_pk2d_values_by_model(pk2d_values, config))
        pk2d_kwargs = clean_none_values(pk2d_kwargs)

        return {
            "cosmo": make_cosmo(config, cosmology_values),
            "pk2d_func": pk2d_model_func(config),
            "pk2d_kwargs": {
                "k_array": k_array,
                "a_array": a_array,
                **pk2d_kwargs,
            },
        }

    return theta_mapper


def parameter_is_used_by_model(parameter, config):
    """Return whether a varied parameter is used by the selected pk2d model."""
    target = parameter["target"]
    group, _ = target.split(".", 1)

    if group == "cosmology":
        return True

    return group in set(pk2d_model_groups(config))


def active_parameter_configs(config):
    """Return varied parameter configs used by the selected model."""
    return [
        parameter
        for parameter in config["parameters"]
        if parameter.get("vary", True) and parameter_is_used_by_model(parameter, config)
    ]


def active_theta0(config):
    """Return fiducial values for active varied parameters."""
    return [float(parameter["fiducial"]) for parameter in active_parameter_configs(config)]


def active_parameter_names(config):
    """Return names of active varied parameters."""
    return [parameter["name"] for parameter in active_parameter_configs(config)]


def active_parameter_labels(config):
    """Return labels of active varied parameters."""
    return [
        parameter.get("label", parameter["name"]) for parameter in active_parameter_configs(config)
    ]
