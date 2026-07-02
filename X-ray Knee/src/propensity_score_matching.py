#!/usr/bin/env python3
"""
propensity_score_matching.py — estimate propensity scores and perform 1:1 nearest-neighbor matching

Steps:
 1. Impute missing values in structural features
 2. Train a logistic regression model for propensity scores
 3. Match treated and control samples using nearest neighbors
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

# Load the data
data_file = Path("data/master_v00_right.csv")
df = pd.read_csv(data_file)

# Select structural features (columns starting with 'xr')
struct_cols = [col for col in df.columns if col.startswith("xr")]

# Step 1: Impute missing values with column medians
imputer = SimpleImputer(strategy="median")
X_struct = pd.DataFrame(
    imputer.fit_transform(df[struct_cols]),
    columns=struct_cols
)

# Step 2: Train logistic regression for propensity scoring
treated_flag = df["race"] == "Black or African American"
log_model = LogisticRegression(max_iter=1000, solver="lbfgs")
log_model.fit(X_struct, treated_flag)
# Assign propensity scores to the DataFrame
df["pscore"] = log_model.predict_proba(X_struct)[:, 1]

# Step 3: Perform 1:1 nearest-neighbor matching
#   - treated group: Black or African American
#   - control group: White

treated = df[treated_flag].copy()
control = df[~treated_flag].copy()

# Fit nearest neighbors on control propensity scores
nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["pscore"]])

# Find the best control match for each treated sample
dists, idxs = nn.kneighbors(treated[["pscore"]])
matched_idx = control.iloc[idxs.flatten()].index

# Combine treated and matched control samples
matched = pd.concat([
    treated.reset_index(drop=True),
    control.loc[matched_idx].reset_index(drop=True)
], ignore_index=True)

# Report the number of matched pairs
print(f"Number of matched pairs: {len(matched) // 2}")
