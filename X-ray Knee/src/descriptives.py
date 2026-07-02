#!/usr/bin/env python3
"""
descriptives.py — summarize pain scores by race group

Generates:
  - reports/tables/pain_by_race.tex  # summary table: count, mean, median, std dev
  - reports/figures/pain_dist_by_race.png  # density curves of pain scores by race
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- 1) Load data and set up output folders ---
data_file = Path("data/master_v00_right.csv")
output_table_dir = Path("reports/tables")
output_fig_dir = Path("reports/figures")
output_table_dir.mkdir(parents=True, exist_ok=True)
output_fig_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_file)

# --- 2) Define and order race categories ---
race_order = [
    "White",
    "Black or African American",
    "Asian",
    "Other Non-White",
    "Unknown or not reported"
]
df["race"] = pd.Categorical(df["race"], categories=race_order, ordered=True)

# --- 3) Create summary statistics table ---
summary = (
    df.groupby("race")["pain_score"]
      .agg(count="count", mean="mean", median="median", stddev="std")
      .round(2)
      .rename(columns={
          "count": "$n$",
          "mean": "Mean",
          "median": "Median",
          "stddev": "StdDev"
      })
)

latex_str = summary.to_latex(column_format="lcccc", escape=False)
table_path = output_table_dir / "pain_by_race.tex"
table_path.write_text(latex_str)
print(f"Wrote summary table to {table_path}")

# --- 4) Plot density curves by race ---
plt.figure(figsize=(6, 3.5))
for race in race_order:
    sns.kdeplot(
        data=df[df["race"] == race],
        x="pain_score",
        label=race,
        linewidth=1.5
    )

plt.xlabel("Pain score (0–100)")
plt.ylabel("Density")
plt.legend(title="Race", fontsize=8)
plt.tight_layout()

fig_path = output_fig_dir / "pain_dist_by_race.png"
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"Saved density plot to {fig_path}")
