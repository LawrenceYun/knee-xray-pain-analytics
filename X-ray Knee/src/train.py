#!/usr/bin/env python3
"""
train.py — train XGBoost regressors

Usage:
  python src/train.py --data data/master_v00_right.csv --mode struct
  python src/train.py --data data/master_v00_right.csv --mode full
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="data/master_v00_right.csv",
        help="path to input CSV"
    )
    parser.add_argument(
        "--mode", choices=["struct", "full"], default="struct",
        help="'struct' uses only structure features; 'full' adds demographics"
    )
    args = parser.parse_known_args()[0]

    data_file = Path(args.data)
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    table_dir = Path("reports/tables")
    table_dir.mkdir(parents=True, exist_ok=True)

    # load data
    df = pd.read_csv(data_file)

    # select feature columns
    struct_cols = [c for c in df.columns if c.startswith("xr")]
    demo_cols = [c for c in ["sex", "ageyears", "race", "BMI", "site", "side"]
                 if c in df.columns]
    X = df[struct_cols].copy()
    if args.mode == "full":
        X = pd.concat([X, df[demo_cols]], axis=1)

    # convert object columns to category and fill missing values
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")
    X = X.fillna(X.median(numeric_only=True))

    y = df["pain_score"]

    # split into train and test sets
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # initialize and train XGBoost regressor
    model = XGBRegressor(
        tree_method="hist",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        enable_categorical=True
    )
    print(f"Training mode = {args.mode}")
    model.fit(X_tr, y_tr)

    # evaluate and display R^2 score
    r2 = r2_score(y_te, model.predict(X_te))
    print(f"Mode {args.mode} R^2 score: {r2:.3f}")

    # save trained model
    model_file = model_dir / f"model_{args.mode}.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {model_file}")

    # update R2 rows JSON
    row_file = table_dir / "r2_rows.json"
    if row_file.exists():
        rows = json.load(row_file.open())
    else:
        rows = []
    rows = [r for r in rows if r.get("mode") != args.mode]
    rows.append({"mode": args.mode, "r2": r2})
    row_file.write_text(json.dumps(rows, indent=2))

    # write LaTeX table when both modes are available
    modes_done = {r.get("mode") for r in rows}
    if modes_done == {"struct", "full"}:
        sorted_rows = sorted(rows, key=lambda x: x.get("mode"))
        delta = sorted_rows[1]["r2"] - sorted_rows[0]["r2"]
        tex_lines = [
            "\\begin{tabular}{lcc}\\toprule",
            "\\multicolumn{3}{c}{Right knee only}\\midrule",
            "Model & $R^2$ & $\\Delta R^2$\\midrule",
            f"Structure & {sorted_rows[0]['r2']:.3f} & --\\",
            f"Struct+Demo & {sorted_rows[1]['r2']:.3f} & {delta:+.3f}\\",
            "\\bottomrule\\end{tabular}"
        ]
        tex_file = table_dir / "r2.tex"
        tex_file.write_text("\n".join(tex_lines))
        print(f"Wrote LaTeX table to {tex_file}")

if __name__ == "__main__":
    main()
