from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [ "obs_data_vector" ] 

def get_full_sample_esd(file_high, file_low):
    """Compute the Delta Sigma data vector for the full sample using the
    samples split in red and blue galaxies. This script weights each galaxy
    type by it's inverse variance weights before computing the final data
    vector."""
    script_dir = Path(__file__).resolve().parent
    df_high = pd.read_csv(script_dir/file_high, comment="#", sep=" ")
    df_low = pd.read_csv(script_dir/file_low, comment="#",  sep=" ")

    # Add a marker if you need to keep track, but we will combine them
    df_combined = pd.concat([df_high, df_low], ignore_index=True)

    # Adding a small epsilon to prevent division by zero just in case
    df_combined["weight"] = 1.0 / (df_combined["err"] ** 2 + 1e-20)

    # 3. Compute the inverse-variance weighted average grouped by radius 'r'
    # Formula: Combined_ESD = Sum(esd * w) / Sum(w)
    df_combined["weighted_esd"] = df_combined["esd"] * df_combined["weight"]

    grouped = df_combined.groupby("r").sum().reset_index()

    final_r = grouped["r"].values
    final_esd = (grouped["weighted_esd"] / grouped["weight"]).values

    # 4. Calculate the combined error propagate for inverse variance: 1 / sqrt(Sum(w))
    final_err = (1.0 / np.sqrt(grouped["weight"])).values

    # Create a clean summary dataframe
    data_vector = pd.DataFrame({"r": final_r, "esd": final_esd, "err": final_err})

    return data_vector


file_high = "./rebin.lum.all.L4.highfdev.csv"
file_low = "./rebin.lum.all.L4.lowfdev.csv"

obs_data_vector = get_full_sample_esd(file_high, file_low)

if __name__ == "__main__":
    try:
        result_df = get_full_sample_esd(file_high, file_low)
    
        print(result_df.to_string(index=False))
    
        r_vector = result_df["r"].to_numpy()
        esd_vector = result_df["esd"].to_numpy()
    
    except FileNotFoundError as e:
        print(
            f"Error: Could not find the files. Please ensure they are in the\
            same directory. \nDetails: {e}"
        )
