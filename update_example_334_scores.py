import pandas as pd
import json
import os

base_dir = "/Users/mrarnav69/Documents/AnchorSum"
example_id_target = 334

# Paths
rouge_bert_t1_path = os.path.join(base_dir, "Results/Component Ablation /rouge_bert_simple_results/rouge_bert_detailed_full.csv")
unieval_t1_path = os.path.join(base_dir, "Results/Component Ablation /unieval_fluency_results/unieval_fluency_detailed_full.csv")

rouge_bert_t2_path = os.path.join(base_dir, "Results/Revision Depth/rouge_bert_full_revisions_2_results/rouge_bert_detailed_full_revisions_2.csv")
unieval_t2_path = os.path.join(base_dir, "Results/Revision Depth/unieval_fluency_full_revisions_2_results/unieval_fluency_detailed_full_revisions_2.csv")

# We will also try to load bertscore_xlarge just in case it exists for T2, but fallback to rouge_bert's bertscore column
bertscore_xlarge_t1_path = os.path.join(base_dir, "Results/Component Ablation /bertscore_xlarge_results/bertscore_detailed_full.csv")

def get_row(csv_path, example_id):
    if not os.path.exists(csv_path): return None
    df = pd.read_csv(csv_path)
    if 'example_id' in df.columns:
        row = df[df['example_id'] == example_id]
    elif 'id' in df.columns:
        row = df[df['id'] == example_id]
    else:
        df['id'] = df.index
        row = df[df['id'] == example_id]
        
    if not row.empty:
        return row.iloc[0].to_dict()
    return None

t1_rb = get_row(rouge_bert_t1_path, example_id_target) or {}
t1_unieval = get_row(unieval_t1_path, example_id_target) or {}
t1_bert_xl = get_row(bertscore_xlarge_t1_path, example_id_target) or {}

t2_rb = get_row(rouge_bert_t2_path, example_id_target) or {}
t2_unieval = get_row(unieval_t2_path, example_id_target) or {}

json_path = os.path.join(base_dir, "example_334_analysis.json")
with open(json_path, 'r') as f:
    data = json.load(f)

# Extract scores
def safe_float(val):
    if val is None: return None
    try: return float(val)
    except: return None

# T1
data["T1_Revision_1"]["scores"]["ROUGE-1"] = safe_float(t1_rb.get('rouge1'))
data["T1_Revision_1"]["scores"]["ROUGE-2"] = safe_float(t1_rb.get('rouge2'))
data["T1_Revision_1"]["scores"]["ROUGE-L"] = safe_float(t1_rb.get('rougel'))
data["T1_Revision_1"]["scores"]["BERTScore"] = safe_float(t1_bert_xl.get('bertscore', t1_rb.get('bertscore')))
data["T1_Revision_1"]["scores"]["UniEval_Fluency"] = safe_float(t1_unieval.get('fluency'))

# T2
data["T2_Revision_2"]["scores"]["ROUGE-1"] = safe_float(t2_rb.get('rouge1'))
data["T2_Revision_2"]["scores"]["ROUGE-2"] = safe_float(t2_rb.get('rouge2'))
data["T2_Revision_2"]["scores"]["ROUGE-L"] = safe_float(t2_rb.get('rougel'))
data["T2_Revision_2"]["scores"]["BERTScore"] = safe_float(t2_rb.get('bertscore'))
data["T2_Revision_2"]["scores"]["UniEval_Fluency"] = safe_float(t2_unieval.get('fluency'))

# Deltas
def calc_delta(k):
    t1_val = data["T1_Revision_1"]["scores"].get(k)
    t2_val = data["T2_Revision_2"]["scores"].get(k)
    if t1_val is not None and t2_val is not None:
        return t2_val - t1_val
    return None

data["Differences"]["Delta_ROUGE-1"] = calc_delta("ROUGE-1")
data["Differences"]["Delta_ROUGE-2"] = calc_delta("ROUGE-2")
data["Differences"]["Delta_ROUGE-L"] = calc_delta("ROUGE-L")
data["Differences"]["Delta_BERTScore"] = calc_delta("BERTScore")
data["Differences"]["Delta_UniEval_Fluency"] = calc_delta("UniEval_Fluency")

with open(json_path, 'w') as f:
    json.dump(data, f, indent=4)

print(f"Updated {json_path} with ROUGE, BERTScore, and UniEval Fluency.")
