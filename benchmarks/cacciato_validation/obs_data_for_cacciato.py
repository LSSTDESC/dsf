from pathlib import Path

import yaml
import numpy as np
import pandas as pd

__all__ = [ "get_full_sample_esd", "config" ] 


def load_luminosity_config(yaml_path):
    """Loads the YAML configuration file."""
    try:
        with open(yaml_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: The file {yaml_path} was not found.")
        return None


def get_bin_info(config, bin_name):
    """Extracts and prints metadata and filenames for a given luminosity bin."""
    if not config:
        return

    if bin_name not in config:
        print(
            f"Bin '{bin_name}' not found. Available bins: {list(config.keys())}"
        )
        return

    bin_data = config[bin_name]

    print(f"=== Information for Bin {bin_name} ===")
    print(f"Absolute Magnitude Range : {bin_data.get('magnitude_bounds')}")
    print(f"Mean Redshift            : {bin_data.get('mean_redshift')}")
    print(f"Total Object Count       : {bin_data.get('total_object_count')}")

    print("\nAssociated Files:")
    print(f"  - High fdev file: {bin_data['files']['high_fdev']}")
    print(f"  - Low fdev file : {bin_data['files']['low_fdev']}")
    print("=" * 40)


def get_full_sample_esd(indir, file_high, file_low):
    """Compute the Delta Sigma data vector for the full sample using the
    samples split in red and blue galaxies. This script weights each galaxy
    type by it's inverse variance weights before computing the final data
    vector."""
    df_high = pd.read_csv(f"{indir}/{file_high}", comment="#", sep=" ")
    df_low = pd.read_csv(f"{indir}/{file_low}", comment="#",  sep=" ")

    # Add a marker if you need to keep track, but we will combine them
    df_combined = pd.concat([df_high, df_low], ignore_index=True)

    # Adding a small epsilon to prevent division by zero just in case
    df_combined["weight"] = 1.0 / (df_combined["err"] ** 2 + 1e-20)

    # Compute the inverse-variance weighted average grouped by radius 'r'
    # Formula: Combined_ESD = Sum(esd * w) / Sum(w)
    df_combined["weighted_esd"] = df_combined["esd"] * df_combined["weight"]

    grouped = df_combined.groupby("r").sum().reset_index()

    final_r = grouped["r"].values
    final_esd = (grouped["weighted_esd"] / grouped["weight"]).values

    # Calculate the combined error propagate for inverse variance: 1 / sqrt(Sum(w))
    final_err = (1.0 / np.sqrt(grouped["weight"])).values

    # Create a clean summary dataframe
    data_vector = pd.DataFrame({"r": final_r, "esd": final_esd, "err": final_err})

    return data_vector


thisdir = Path(__file__).resolve().parent
config = load_luminosity_config(thisdir / "sample_info.yaml")

if __name__ == "__main__":

    indir = thisdir.parent / config["dir"]
    lumbin = "L6f"

    if config:
        get_bin_info(config, lumbin)

    config = config[lumbin]
    faint_lim = config["magnitude_bounds"]["faint_limit"]
    bright_lim = config["magnitude_bounds"]["bright_limit"]
    file_high = config["files"]["high_fdev"]
    file_low = config["files"]["low_fdev"]

    try:
        result_df = get_full_sample_esd(indir, file_high, file_low)
    
        print(result_df.to_string(index=False))
    
        r_vector = result_df["r"].to_numpy()
        esd_vector = result_df["esd"].to_numpy()

        print(result_df.dtypes)
    
    except FileNotFoundError as e:
        print(
            f"Error: Could not find the files. Please ensure they are in the\
            same directory. \nDetails: {e}"
        )
