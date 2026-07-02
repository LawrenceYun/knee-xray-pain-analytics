#!/usr/bin/env python3
"""
sane_global_plot.py — create bar chart and LaTeX table summarizing S/N frequencies
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the brute-force results (S and N subsets)
data_file = Path("outputs/sane_samples_5.json")
print(f"Loading data from {data_file}")
raw_data = json.load(open(data_file))

# Build a list of records with S and N lists
records = []
for entry in raw_data.values():
    S = entry.get("S") or []  # sufficient subset
    N = entry.get("N") or []  # necessary subset
    records.append({"S": S, "N": N})
df = pd.DataFrame(records)

# Determine all features and compute frequencies
features = sorted({feat for lst in df["S"].tolist() + df["N"].tolist() for feat in lst})
freq_S = {feat: df["S"].apply(lambda lst: feat in lst).mean() for feat in features}
freq_N = {feat: df["N"].apply(lambda lst: feat in lst).mean() for feat in features}
# Sort features by how often they appear in S
order = sorted(features, key=lambda feat: freq_S[feat], reverse=True)

# Plot grouped bar chart of P(S) and P(N)
plt.figure(figsize=(7, 4.5))
x = np.arange(len(order))
vals_S = [freq_S[f] * 100 for f in order]
vals_N = [freq_N[f] * 100 for f in order]
plt.bar(x - 0.15, vals_S, width=0.3, label="Sufficient")
plt.bar(x + 0.15, vals_N, width=0.3, label="Necessary")
plt.xticks(x, order, rotation=60, ha="right")
plt.ylabel("Percentage of samples (%)")
plt.legend()
plt.tight_layout()

# Save figure
fig_dir = Path("reports/figures")
fig_dir.mkdir(parents=True, exist_ok=True)
fig_path = fig_dir / "fig_sane_bar.png"
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"Saved bar chart to {fig_path}")

# Generate LaTeX table with probabilities
table_dir = Path("reports/tables")
table_dir.mkdir(parents=True, exist_ok=True)
rows = []
for feat in order:
    pS = freq_S[feat] * 100
    pN = freq_N[feat] * 100
    rows.append(f"{feat} & {pS:.1f}\\% & {pN:.1f}\\\\")

latex_lines = [
    "\\begin{tabular}{lcc}\\toprule",
    "Feature & $P(S)$ & $P(N)$\\\\midrule"
] + rows + ["\\bottomrule\\end{tabular}"]

table_path = table_dir / "sane_full.tex"
table_path.write_text("\n".join(latex_lines))
print(f"Wrote LaTeX table to {table_path}")
