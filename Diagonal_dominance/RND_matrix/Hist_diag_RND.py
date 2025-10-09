import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


numbers = {}
folder = "Figures"

for i in range(1, 15):
    df = pd.read_csv(f"diagonal_analysis_detailed_{i}.txt", sep=r"\s+", header=None, engine="c")
    indices = df.index
    sampled_indices = indices[::1000]
    sampled_values = df.iloc[sampled_indices, 0]  # take only the first column

    # Remove non-finite or non-positive entries
    sampled_values = sampled_values[np.isfinite(sampled_values)]
    sampled_values = sampled_values[sampled_values > 0]

    if len(sampled_values) == 0:
        print(f"Skipping {i}: no valid positive values.")
        continue

    vmin, vmax = sampled_values.min(), sampled_values.max()
    if vmin <= 0 or vmax <= 0:
        print(f"Skipping {i}: contains non-positive values (vmin={vmin}, vmax={vmax})")
        continue

    # Define logarithmic bins safely
    n_bins = int(np.log10(vmax / vmin) * 20)  # ~20 bins per decade, adjustable
    n_bins = max(n_bins, 10)  # ensure at least 10 bins

    bins = np.logspace(np.log10(vmin), np.log10(vmax), num=n_bins)

    plt.figure(figsize=(8, 6))
    sns.histplot(sampled_values, bins=bins, color='blue')
    plt.xscale('log')
    plt.xlabel(r"$a_{ii}$ / $a_{ij}$")
    plt.ylabel("Frequency")
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.title(f"Histogram of Diagonal Dominance Analysis (Sparse_matrix_{i})")
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{folder}/histo_diag_dom_{i}_sampled.pdf")
    plt.close()

    print(f"Saved histogram for Sparse_matrix_{i}")