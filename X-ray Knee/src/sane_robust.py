#!/usr/bin/env python3

import json, collections, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path


files = {
    4: "outputs/sane_samples.json",
    5: "outputs/sane_samples_5.json",
    6: "outputs/sane_samples_6.json",
    7: "outputs/sane_samples_7.json",
}

records = []
for k, fp in files.items():
    data = json.load(open(fp))
    n   = len(data)
    cntS = collections.Counter()
    cntN = collections.Counter()
    for v in data.values():
        if v["S"]: cntS.update(set(v["S"]))
        if v["N"]: cntN.update(set(v["N"]))
    for feat in set(cntS) | set(cntN):
        records.append({
            "k": k,
            "feature": feat,
            "pct_S": 100*cntS[feat]/n,
            "pct_N": 100*cntN[feat]/n,
        })

df = pd.DataFrame(records)


top_feats = (df.groupby("feature")["pct_S"].max()
               .sort_values(ascending=False)
               .head(10).index)
fig, ax = plt.subplots(figsize=(8,4))
width = 0.18
ks = sorted(files)
for i,k in enumerate(ks):
    sub = df[(df.k==k)&(df.feature.isin(top_feats))]
    ax.bar([x+i*width for x in range(len(top_feats))], sub["pct_S"],
           width, label=f"k={k}")
ax.set_xticks([x+width*1.5 for x in range(len(top_feats))])
ax.set_xticklabels(top_feats, rotation=45, ha="right")
ax.set_ylabel("% knees where feature ∈ S")
ax.set_title("Sufficient-set frequency vs. k")
ax.legend()
Path("reports/figures").mkdir(parents=True, exist_ok=True)
plt.tight_layout(); plt.savefig("reports/figures/fig_sane_robust.png", dpi=300)
print(" fig_sane_robust.png saved")


tbl = (df[df.feature.isin(top_feats)]
         .pivot(index="feature", columns="k", values="pct_S")
         .round(1).fillna(0))
latex = tbl.to_latex(column_format="l"+"c"*len(ks),
                     caption="Frequency (\\%) that each feature appears in a Sufficient set under different $k_{\\max}$.",
                     label="tab:sane-robust")
Path("reports/tables").mkdir(parents=True, exist_ok=True)
Path("reports/tables/sane_robust.tex").write_text(latex)
print("sane_robust.tex written")
