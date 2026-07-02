#!/usr/bin/env python3
"""
shap_analysis.py — generate TreeSHAP plots and tables

Outputs:
  reports/figures/fig_shap_beeswarm.png   # top 20 features beeswarm plot
  outputs/top20.json                      # list of top 20 feature names
  reports/tables/tbl_shap_top.tex         # LaTeX table of top 15 features by mean |SHAP|
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# paths
DATA_FILE = Path("data/master_v00_right.csv")
MODEL_FILE = Path("models/model_full_right.pkl")
FIG_FILE = Path("reports/figures/fig_shap_beeswarm.png")
TABLE_FILE = Path("reports/tables/tbl_shap_top.tex")
TOP20_FILE = Path("outputs/top20.json")

# ensure directories exist
FIG_FILE.parent.mkdir(parents=True, exist_ok=True)
TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
TOP20_FILE.parent.mkdir(exist_ok=True)

# load data and model
df = pd.read_csv(DATA_FILE)
model = pickle.load(open(MODEL_FILE, "rb"))

# prepare feature matrix
struct_cols = [c for c in df.columns if c.startswith("xr")]
demo_cols = [c for c in ["sex", "ageyears", "race", "BMI", "site", "side"] if c in df.columns]
X = pd.concat([df[struct_cols], df[demo_cols]], axis=1)
for col in X.select_dtypes(include="object"):
    X[col] = X[col].astype("category")
X = X.fillna(X.median(numeric_only=True))

# sample for speed
sample = X.sample(n=1000, random_state=42)

# compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(sample)

# determine top 20 features by mean absolute SHAP value
mean_abs = shap_values.abs.mean(axis=0).values
indices = np.argsort(mean_abs)[::-1][:20]
top20 = X.columns[indices].tolist()

# save top20 list
json.dump(top20, open(TOP20_FILE, "w"))
print(f"Saved top-20 features list to {TOP20_FILE}")

# create beeswarm plot
plt.figure(figsize=(6.3, 9))
shap.plots.beeswarm(shap_values[:, indices], max_display=len(top20), show=False)
plt.tight_layout()
plt.savefig(FIG_FILE, dpi=300)
plt.close()
print(f"Saved beeswarm plot to {FIG_FILE}")

# generate LaTeX table for top 15 features
indices15 = indices[:15]
features15 = X.columns[indices15]
values15 = mean_abs[indices15].round(3)

lines = [
    "\\begin{tabular}{lcr}\\toprule",
    "Rank & Feature & $|\mathrm{SHAP}|$\\\\midrule"
]
for i, (feat, val) in enumerate(zip(features15, values15), start=1):
    lines.append(f"{i} & {feat} & {val:.3f}\\\\")
lines.append("\\bottomrule\\end{tabular}")
TABLE_FILE.write_text("\n".join(lines))
print(f"Wrote LaTeX table to {TABLE_FILE}")
