import pandas as pd
import json
import os

# Base paths
base_dir = "/Users/mrarnav69/Documents/AnchorSum"

summac_t1_path = os.path.join(base_dir, "Results/Component Ablation /summac_final_results/summac_detailed_full.csv")
summac_t2_path = os.path.join(base_dir, "Results/Revision Depth/summac_full_revisions_2_results/summac_detailed_full_revisions_2.csv")

bart_t1_path = os.path.join(base_dir, "Results/Component Ablation /bartscore_simple_results/bartscore_detailed_full.csv")
bart_t2_path = os.path.join(base_dir, "Results/Revision Depth/bartscore_full_revisions_2_results/bartscore_detailed_full_revisions_2.csv")

align_t1_path = os.path.join(base_dir, "Results/Component Ablation /alignscore_results/alignscore_detailed_full.csv")
align_t2_path = os.path.join(base_dir, "Results/Revision Depth/alignscore_full_revisions_2_results/alignscore_detailed_full_revisions_2.csv")

json_t1_path = os.path.join(base_dir, "data/ablations/full/ablation_full_final_500.json")
json_t2_path = os.path.join(base_dir, "data/ablations/full_revisions_2/ablation_full_revisions_2_final_500.json")

# Load CSVs
summac_t1 = pd.read_csv(summac_t1_path)
summac_t2 = pd.read_csv(summac_t2_path)

bart_t1 = pd.read_csv(bart_t1_path)
bart_t2 = pd.read_csv(bart_t2_path)

align_t1 = pd.read_csv(align_t1_path)
align_t2 = pd.read_csv(align_t2_path)

# Ensure 'example_id' exists or use index
def ensure_id(df):
    if 'example_id' not in df.columns:
        df['example_id'] = df.index
    return df

summac_t1 = ensure_id(summac_t1)
summac_t2 = ensure_id(summac_t2)
bart_t1 = ensure_id(bart_t1)
bart_t2 = ensure_id(bart_t2)
align_t1 = ensure_id(align_t1)
align_t2 = ensure_id(align_t2)

# Load JSONs
with open(json_t1_path, 'r') as f:
    json_t1 = json.load(f)
with open(json_t2_path, 'r') as f:
    json_t2 = json.load(f)

len_t1 = {item['example_id']: len(item['final_summary']) for item in json_t1}
len_t2 = {item['example_id']: len(item['final_summary']) for item in json_t2}

# Merge all into one dataframe
df = summac_t1[['example_id', 'summac_score']].rename(columns={'summac_score': 'summac_t1'})
df = df.merge(summac_t2[['example_id', 'summac_score']].rename(columns={'summac_score': 'summac_t2'}), on='example_id')

df = df.merge(bart_t1[['example_id', 'bartscore_sum2doc']].rename(columns={'bartscore_sum2doc': 'bart_t1'}), on='example_id')
df = df.merge(bart_t2[['example_id', 'bartscore_sum2doc']].rename(columns={'bartscore_sum2doc': 'bart_t2'}), on='example_id')

df = df.merge(align_t1[['example_id', 'alignscore']].rename(columns={'alignscore': 'align_t1'}), on='example_id')
df = df.merge(align_t2[['example_id', 'alignscore']].rename(columns={'alignscore': 'align_t2'}), on='example_id')

df['len_t1'] = df['example_id'].map(len_t1)
df['len_t2'] = df['example_id'].map(len_t2)

df['delta_summac'] = df['summac_t2'] - df['summac_t1']
df['delta_bart'] = df['bart_t2'] - df['bart_t1']
df['delta_align'] = df['align_t2'] - df['align_t1']
df['delta_len_pct'] = abs(df['len_t2'] - df['len_t1']) / df['len_t1']

filtered = df[
    (df['delta_summac'] > 0.25) &
    (df['delta_bart'] < -2.0) &
    (df['delta_len_pct'] < 0.05) &
    (df['delta_align'] <= 0)
]

print("Found instances:")
for index, row in filtered.iterrows():
    print(f"Example ID: {int(row['example_id'])}")
    print(f"  SummaC: {row['summac_t1']:.3f} -> {row['summac_t2']:.3f} (Δ {row['delta_summac']:.3f})")
    print(f"  BART s->d: {row['bart_t1']:.3f} -> {row['bart_t2']:.3f} (Δ {row['delta_bart']:.3f})")
    print(f"  AlignScore: {row['align_t1']:.3f} -> {row['align_t2']:.3f} (Δ {row['delta_align']:.3f})")
    print(f"  Length: {row['len_t1']:.0f} -> {row['len_t2']:.0f} (Δ {row['delta_len_pct']*100:.1f}%)")
    print()
