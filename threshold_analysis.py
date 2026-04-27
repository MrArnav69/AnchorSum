import pandas as pd
import numpy as np
import os

base_dir = "/Users/mrarnav69/Documents/AnchorSum"

def load_and_standardize(path, val_col):
    df = pd.read_csv(path)
    # Standardize ID column
    if 'example_id' in df.columns:
        df = df.rename(columns={'example_id': 'id'})
    elif 'id' not in df.columns:
        df['id'] = df.index
    return df[['id', val_col]]

# Load T1 Data
summac_t1 = load_and_standardize(
    os.path.join(base_dir, "Results", "Component Ablation ", "summac_final_results", "summac_detailed_full.csv"),
    "summac_score"
).rename(columns={"summac_score": "summac_t1"})

bart_t1 = load_and_standardize(
    os.path.join(base_dir, "Results", "Component Ablation ", "bartscore_simple_results", "bartscore_detailed_full.csv"),
    "bartscore_sum2doc"
).rename(columns={"bartscore_sum2doc": "bart_t1"})

alignscore_t1 = load_and_standardize(
    os.path.join(base_dir, "Results", "Component Ablation ", "alignscore_results", "alignscore_detailed_full.csv"),
    "alignscore"
).rename(columns={"alignscore": "align_t1"})

# Load T2 Data
summac_t2 = load_and_standardize(
    os.path.join(base_dir, "Results", "Revision Depth", "summac_full_revisions_2_results", "summac_detailed_full_revisions_2.csv"),
    "summac_score"
).rename(columns={"summac_score": "summac_t2"})

bart_t2 = load_and_standardize(
    os.path.join(base_dir, "Results", "Revision Depth", "bartscore_full_revisions_2_results", "bartscore_detailed_full_revisions_2.csv"),
    "bartscore_sum2doc"
).rename(columns={"bartscore_sum2doc": "bart_t2"})

alignscore_t2 = load_and_standardize(
    os.path.join(base_dir, "Results", "Revision Depth", "alignscore_full_revisions_2_results", "alignscore_detailed_full_revisions_2.csv"),
    "alignscore"
).rename(columns={"alignscore": "align_t2"})

# Merge all into one dataframe safely on 'id'
df = summac_t1.merge(summac_t2, on="id", how="inner")
df = df.merge(bart_t1, on="id", how="inner")
df = df.merge(bart_t2, on="id", how="inner")
df = df.merge(alignscore_t1, on="id", how="inner")
df = df.merge(alignscore_t2, on="id", how="inner")

# Compute deltas
df["delta_summac"] = df["summac_t2"] - df["summac_t1"]
df["delta_bart"] = df["bart_t2"] - df["bart_t1"]
df["delta_align"] = df["align_t2"] - df["align_t1"]

n = len(df)

# Condition (i): ΔSummaCConv > 0.1
cond1 = (df["delta_summac"] > 0.1)
print(f"Condition (i) ΔSummaCConv > 0.1:")
print(f"  Instances satisfying: {cond1.sum()}/{n} ({100*cond1.mean():.1f}%)")
print(f"  Mean ΔSummaCConv: {df['delta_summac'].mean():.4f}")
print(f"  Median ΔSummaCConv: {df['delta_summac'].median():.4f}")
print(f"  Min: {df['delta_summac'].min():.4f}, Max: {df['delta_summac'].max():.4f}")
print()

# Condition (ii): |ΔAlignScore| < 0.01
cond2 = (df["delta_align"].abs() < 0.01)
print(f"Condition (ii) |ΔAlignScore| < 0.01:")
print(f"  Instances satisfying: {cond2.sum()}/{n} ({100*cond2.mean():.1f}%)")
print(f"  Mean ΔAlignScore: {df['delta_align'].mean():.6f}")
print(f"  Median ΔAlignScore: {df['delta_align'].median():.6f}")
print(f"  Min: {df['delta_align'].min():.4f}, Max: {df['delta_align'].max():.4f}")
print()

# Condition (iii): ΔBARTScore s→d < -1.0
cond3 = (df["delta_bart"] < -1.0)
print(f"Condition (iii) ΔBARTScore s→d < -1.0:")
print(f"  Instances satisfying: {cond3.sum()}/{n} ({100*cond3.mean():.1f}%)")
print(f"  Mean ΔBARTScore s→d: {df['delta_bart'].mean():.4f}")
print(f"  Median ΔBARTScore s→d: {df['delta_bart'].median():.4f}")
print(f"  Min: {df['delta_bart'].min():.4f}, Max: {df['delta_bart'].max():.4f}")
print()

# All three conditions simultaneously
all_three = cond1 & cond2 & cond3
print(f"All three conditions simultaneously:")
print(f"  Instances satisfying: {all_three.sum()}/{n} ({100*all_three.mean():.1f}%)")
print()

# Percentile analysis for threshold justification
print(f"ΔSummaCConv percentiles:")
for p in [10, 25, 50, 75, 90]:
    print(f"  {p}th: {np.percentile(df['delta_summac'].dropna(), p):.4f}")
print()

print(f"ΔBARTScore s→d percentiles:")
for p in [10, 25, 50, 75, 90]:
    print(f"  {p}th: {np.percentile(df['delta_bart'].dropna(), p):.4f}")
print()

print(f"ΔAlignScore percentiles:")
for p in [10, 25, 50, 75, 90]:
    print(f"  {p}th: {np.percentile(df['delta_align'].dropna(), p):.4f}")
