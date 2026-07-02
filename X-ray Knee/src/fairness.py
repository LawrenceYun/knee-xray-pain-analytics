#!/usr/bin/env python3
"""
fairness.py — 评估 Equalized Odds & Multi-accuracy（右膝默认）
---------------------------------------------------------------
用法示例（Colab / 终端）：
  python src/fairness.py                                   # <== 默认右膝
  python src/fairness.py --data data/master_v00_right.csv \
                         --model models/model_full_right.pkl
"""
import argparse, pickle, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix

# -------- CLI --------
p = argparse.ArgumentParser()
p.add_argument("--data",  default="data/master_v00_right.csv")
p.add_argument("--model", default="models/model_full_right.pkl")
args = p.parse_known_args()[0]          # ← 关键：忽略 -f 注入

print(f"[INFO]  Data  = {args.data}")
print(f"[INFO]  Model = {args.model}")

# -------- 载入 --------
df     = pd.read_csv(args.data)
model  = pickle.load(open(args.model,"rb"))

struct = [c for c in df if c.startswith("xr")]
demo   = [c for c in ["sex","ageyears","race","site","side"] if c in df.columns]
X      = pd.concat([df[struct], df[demo]], axis=1)
for c in X.select_dtypes("object"): X[c]=X[c].astype("category")
X = X.fillna(X.median(numeric_only=True))

y_true = (df["pain_score"] >= 50).astype(int).values
y_hat  = (model.predict(X) >= 50).astype(int)

df_eval = pd.DataFrame({
    "race": df["race"],
    "y":    y_true,
    "yhat": y_hat
})

# -------- Equalized Odds --------
def eo_gap(mask):
    y, yhat = df_eval.loc[mask, ["y","yhat"]].values.T
    if y.sum()==0 or (len(y)-y.sum())==0:        # 只有 0 或 1 → 无法定义
        return None, None
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    tpr = tp / (tp+fn)
    fpr = fp / (fp+tn)
    return tpr, fpr

mask_B = df_eval["race"]=="Black"
mask_W = df_eval["race"]=="White"
tpr_B, fpr_B = eo_gap(mask_B)
tpr_W, fpr_W = eo_gap(mask_W)

delta_tpr = None if (tpr_B is None or tpr_W is None) else abs(tpr_B-tpr_W)
delta_fpr = None if (fpr_B is None or fpr_W is None) else abs(fpr_B-fpr_W)

# -------- Multi-accuracy --------
# 分 10 桶（按预测概率）；计算每桶残差
proba = model.predict(X) / 100      # XGBoost 0–100 → 0–1
df_ma = pd.DataFrame({"prob":proba, "y":y_true})
df_ma["bucket"] = pd.qcut(df_ma["prob"], 10, duplicates="drop")
bucket_gap = (df_ma.groupby("bucket")
                    .apply(lambda g: abs(g["y"].mean() - g["prob"].mean()))
                    .max()) * 100     # 转百分比

# -------- 输出 TeX --------
TABLE_DIR = Path("reports/tables"); TABLE_DIR.mkdir(parents=True, exist_ok=True)

def fmt(x):
    return "--" if x is None else f"{x:.3f}"

tex = (
    "\\begin{tabular}{lcc}\\toprule\n"
    "\\multicolumn{3}{c}{\\textbf{Fairness (right)}}\\\\\\midrule\n"
    "& $\\Delta$TPR & $\\Delta$FPR \\\\\\midrule\n"
    f"Equalized~Odds & {fmt(delta_tpr)} & {fmt(delta_fpr)}\\\\\\midrule\n"
    f"\\multicolumn{{3}}{{c}}{{Max multiaccuracy residual = {bucket_gap:.2f}\\,pp}}\\\\\n"
    "\\bottomrule\\end{tabular}\n"
)

(TABLE_DIR/"fairness.tex").write_text(tex)
print("📄 wrote  reports/tables/fairness.tex")
