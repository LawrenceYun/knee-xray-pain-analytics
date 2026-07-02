#!/usr/bin/env python3
"""
residual_analysis.py — impute missing values and compute regression residuals

Steps:
 1. Fill missing structural feature values with medians
 2. Fit a linear regression model to predict pain_score
 3. Calculate residuals and report average by race
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from pathlib import Path

# Load the dataset
data_path = Path("data/master_v00_right.csv")
df = pd.read_csv(data_path)

# Select structural features (columns starting with 'xr')
struct_cols = [col for col in df.columns if col.startswith("xr")]

# Step 1: Impute missing values using the median
target_df = pd.DataFrame(
    SimpleImputer(strategy="median")
    .fit_transform(df[struct_cols]),
    columns=struct_cols
)

# Prepare target variable
y = df["pain_score"]

# Step 2: Fit a linear regression model
model = LinearRegression()
model.fit(target_df, y)

# Step 3: Compute residuals and add to DataFrame
residuals = y - model.predict(target_df)
df["residuals"] = residuals

# Print average residuals per race
avg_resid_by_race = df.groupby("race")["residuals"].mean()
print("Average residuals by race:")
print(avg_resid_by_race)
