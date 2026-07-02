#!/usr/bin/env python3
"""
sane_bruteforce.py — simple brute-force search for minimal sufficient and necessary subsets

Usage:
  python src/sane_bruteforce.py \
    --model models/model_full_right.pkl \
    --data data/master_v00_right.csv \
    --eps 1e-3 --kmax 6 --nsamp 200
"""
import argparse
import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from tqdm import tqdm

# parse command-line options
p = argparse.ArgumentParser()
p.add_argument(
    "--model", default="models/model_full_right.pkl",
    help="path to trained model"
)
p.add_argument(
    "--data", default="data/master_v00_right.csv",
    help="path to input CSV data"
)
p.add_argument(
    "--eps", type=float, default=1e-3,
    help="tolerance for prediction difference"
)
p.add_argument(
    "--kmax", type=int, default=6,
    help="maximum feature subset size to test"
)
p.add_argument(
    "--nsamp", type=int, default=200,
    help="number of random samples to analyze"
)
args = p.parse_known_args()[0]

# load model and dataset
model = pickle.load(open(args.model, "rb"))
df = pd.read_csv(args.data)

# prepare features: structural + demographic
struct = [c for c in df.columns if c.startswith("xr")]
demo = [c for c in ["sex", "ageyears", "race", "site", "side"] if c in df.columns]
X = df[struct + demo].copy()

# convert object columns to category, fill missing
for col in X.select_dtypes(include="object").columns:
    X[col] = X[col].astype("category")
X = X.fillna(X.median(numeric_only=True))

def predict(row_df):
    """
    Predict for a single-row DataFrame, returning a scalar.
    """
    for col in row_df.select_dtypes(include="object").columns:
        row_df[col] = row_df[col].astype("category")
    return float(model.predict(row_df, validate_features=False)[0])

def brute(i, kmax, eps):
    """
    Find minimal sufficient set S and necessary set N for instance i.
    """
    row = X.iloc[[i]].copy()
    original = predict(row)
    features = row.columns

    # find S: smallest subset keeping prediction within eps
    Smin = None
    for size in range(1, kmax + 1):
        for combo in itertools.combinations(features, size):
            test = row.copy()
            for f in features:
                if f not in combo:
                    if isinstance(test[f].dtype, CategoricalDtype):
                        test[f] = test[f].cat.categories[0]
                    else:
                        test[f] = 0
            if abs(predict(test) - original) < eps:
                Smin = list(combo)
                break
        if Smin:
            break

    # find N: smallest subset whose removal alters prediction more than eps
    Nmin = None
    for size in range(1, kmax + 1):
        for combo in itertools.combinations(features, size):
            test = row.copy()
            for f in combo:
                if isinstance(test[f].dtype, CategoricalDtype):
                    test[f] = test[f].cat.categories[0]
                else:
                    test[f] = 0
            if abs(predict(test) - original) > eps:
                Nmin = list(combo)
                break
        if Nmin:
            break

    return {"y_hat": original, "S": Smin, "N": Nmin}

# select random sample indices
rng = np.random.RandomState(0)
idxs = rng.choice(X.index, size=min(args.nsamp, len(X)), replace=False)
print(f"Processing {len(idxs)} samples (kmax={args.kmax}, eps={args.eps})")

# run brute-force analysis
tresults = {int(i): brute(i, args.kmax, args.eps) for i in tqdm(idxs, desc="Brute force")}

# save output
out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)
json.dump(tresults, open("outputs/sane_samples_7.json", "w"))
print("Saved results to outputs/sane_samples_7.json")
