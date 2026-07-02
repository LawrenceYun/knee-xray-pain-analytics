#!/usr/bin/env python3
"""
prepare_data.py — Build V00 master CSVs with side-specific pain labels

Steps:
1. Load semi-quantitative X-ray (semi), KOOS/WOMAC pain (pain), and demographics (demo)
2. Drop overlapping keys from demo to avoid suffix conflicts
3. Merge semi + pain on (src_subject_id, interview_date, visit)
4. Merge demo on src_subject_id
5. Normalize and filter baseline visit == "V00"
6. Convert side to numeric and compute side-specific pain_score
7. Optional: compute BMI
8. Deduplicate by averaging per (src_subject_id, interview_date, visit, side)
9. Save master_v00_both.csv & master_v00_right.csv
"""
import pandas as pd
from pathlib import Path

DATA = Path("data")
OUT_B = DATA / "master_v00_both.csv"
OUT_R = DATA / "master_v00_right.csv"

# 1. Load raw CSVs
semi = pd.read_csv(DATA/"oai_kxrsemiquant01.csv", low_memory=False)
pain = pd.read_csv(DATA/"oai_koos_womac01.csv",  low_memory=False)
demo = pd.read_csv(DATA/"oai_enrollee01.csv",    low_memory=False)

# 2. Sanity-check keys
req_semi = {"src_subject_id","interview_date","visit","side"}
req_pain = {"src_subject_id","interview_date","visit"}
assert req_semi.issubset(semi.columns), f"Semi missing: {req_semi - set(semi.columns)}"
assert req_pain.issubset(pain.columns), f"Pain missing: {req_pain - set(pain.columns)}"

# 2b. Drop overlapping key cols from demo
demo = demo.drop(columns=["interview_date","visit","side"], errors="ignore")

# 3. Merge semi + pain on true keys
keys = ["src_subject_id","interview_date","visit"]
df = semi.merge(pain, on=keys, how="inner", suffixes=("_semi","_pain"))

# 4. Merge demographics
df = df.merge(demo, on="src_subject_id", how="left")

# ---------- git rid of _x / _y / _semi / _pain  --------------------------------
def unify_duplicate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """
    对形如  foo_x / foo_y  或  foo_semi / foo_pain  的列做统一：
    - 如果两列内容完全一致，则保留一列并改名成 base；
    - 如果不一致：优先级 _semi > _x > _y > _pain，然后改名成 base；
    - 只留唯一版本，避免同一个 feature 出现两列。
    """
    priority = {"_semi": 0, "_x": 1, "_y": 2, "_pain": 3}
    keep = {}          # base -> (col, prio)

    for col in list(frame.columns):
        suf = next((s for s in priority if col.endswith(s)), "")
        base = col[:-len(suf)] if suf else col
        prio = priority.get(suf, 99)

        if base not in keep:
            keep[base] = (col, prio)
        else:
            # 若已有同名列，比较保留优先级
            old_col, old_prio = keep[base]
            if prio < old_prio:
                keep[base] = (col, prio)

    # 构造最终 DataFrame：只保留优选列并改名为 base
    out = frame[[v[0] for v in keep.values()]].copy()
    out.columns = list(keep.keys())
    return out


df = unify_duplicate_columns(df)




# 5. Normalize and filter visit == "V00"
df["visit"] = df["visit"].astype(str).str.strip().str.upper()
df = df[df["visit"]=="V00"].copy()

# 6. Convert side to numeric & compute pain_score
df["side"] = pd.to_numeric(df["side"], errors="coerce")


def build_old_pain(df):
    # 所有 pain_* 项（排除 _rkfr/_lkfr 两个 fill-in）
    items = [c for c in df.columns
             if c.startswith("pain_") and not c.endswith(("rkfr","lkfr"))]

    # 对每列单独做 to_numeric，再拼回同一个 DataFrame
    v = df[items].apply(pd.to_numeric, errors="coerce")

    n = v.notna().sum(axis=1)           # 每行有效题目数
    ok = n >= 5
    score = 100 - 100 * v.sum(axis=1) / (4 * n)
    return score.where(ok)              # ≥5 题才保留


df["pain_score"] = build_old_pain(df)

df = df.dropna(subset=["pain_score"])
df["pain_bin"] = (df["pain_score"]>=50).astype(int)

# 7. Optional BMI
if {"wtkg","htm"}.issubset(df.columns):
    df["BMI"] = df["wtkg"] / (df["htm"]**2)

# 8. Deduplicate per group_keys
group_keys = keys + ["side"]
num = df.groupby(group_keys, as_index=False).mean(numeric_only=True)
cat = df.groupby(group_keys, as_index=False).first()
df  = num.merge(cat, on=group_keys, how="left")

df = unify_duplicate_columns(df)


# 9. Save outputs
df.to_csv(OUT_B, index=False)
df[df["side"]==2].to_csv(OUT_R, index=False)
print(f"✅ Wrote {OUT_R.name} ({len(df[df['side']==2])} rows) and {OUT_B.name} ({len(df)} rows)")

