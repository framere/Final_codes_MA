import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from matplotlib.ticker import LogLocator


molecules = ["H2", "formaldehyde", "uracil"]
folder = "Figures"
os.makedirs(folder, exist_ok=True) # Ensure the output directory exists if 

for mol in molecules:
    df = pd.read_csv(f"diagonal_analysis_detailed_{mol}.txt", sep=r"\s+", header=None, engine="c")
    indices = df.index
    sampled_indices = indices[::1000]
    sampled_values = df.iloc[sampled_indices, 0]  # take only the first column

    # Remove non-finite or non-positive entries
    sampled_values = sampled_values[np.isfinite(sampled_values)]
    sampled_values = sampled_values[sampled_values > 0]

    # print the mean and the median of the sampled values
    mean_val = sampled_values.mean()
    median_val = sampled_values.median()
    print(f"{mol}: mean = {mean_val}, median = {median_val}")

    if len(sampled_values) == 0:
        print(f"Skipping {mol}: no valid positive values.")
        continue

    vmin, vmax = sampled_values.min(), sampled_values.max()
    if vmin <= 0 or vmax <= 0:
        print(f"Skipping {mol}: contains non-positive values (vmin={vmin}, vmax={vmax})")
        continue

    # Define logarithmic bins safely
    n_bins = int(np.log10(vmax / vmin) * 20)  # ~20 bins per decade, adjustable
    n_bins = max(n_bins, 10)  # ensure at least 10 bins

    bins = np.logspace(np.log10(vmin), np.log10(vmax), num=n_bins)

    plt.figure(figsize=(8, 6))
    sns.histplot(sampled_values, bins=bins, color='blue')
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_val:.2f}')
    plt.legend()
    plt.xscale('log')
    plt.xlabel(r"$a_{ii}$ / $a_{ij}$")
    plt.ylabel("Frequency")
    
    # Only put major ticks (and thus grid lines) at 10^0, 10^1, 10^2, ...
    plt.gca().xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))

    # Remove minor ticks if you don't want them
    plt.gca().xaxis.set_minor_locator(LogLocator(base=10.0, subs=[]))

    # Draw grid only at major ticks
    plt.grid(True, which='major', ls='--', lw=0.8, alpha=0.6)

    plt.title(f"Histogram of Diagonal Dominance Analysis (Sparse_matrix_{mol})")
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{folder}/histo_diag_dom_{mol}_sampled.pdf")
    plt.close()

    print(f"Saved histogram for Sparse_matrix_{mol}")