import pandas as pd
import json
import os

base_dir = "/Users/mrarnav69/Documents/AnchorSum"
example_id_target = 334

# Paths
summac_t1_path = os.path.join(base_dir, "Results/Component Ablation /summac_final_results/summac_detailed_full.csv")
summac_t2_path = os.path.join(base_dir, "Results/Revision Depth/summac_full_revisions_2_results/summac_detailed_full_revisions_2.csv")
bart_t1_path = os.path.join(base_dir, "Results/Component Ablation /bartscore_simple_results/bartscore_detailed_full.csv")
bart_t2_path = os.path.join(base_dir, "Results/Revision Depth/bartscore_full_revisions_2_results/bartscore_detailed_full_revisions_2.csv")
align_t1_path = os.path.join(base_dir, "Results/Component Ablation /alignscore_results/alignscore_detailed_full.csv")
align_t2_path = os.path.join(base_dir, "Results/Revision Depth/alignscore_full_revisions_2_results/alignscore_detailed_full_revisions_2.csv")

json_t1_path = os.path.join(base_dir, "data/ablations/full/ablation_full_final_500.json")
json_t2_path = os.path.join(base_dir, "data/ablations/full_revisions_2/ablation_full_revisions_2_final_500.json")

def get_score(csv_path, col_name, example_id):
    df = pd.read_csv(csv_path)
    if 'example_id' not in df.columns:
        df['example_id'] = df.index
    row = df[df['example_id'] == example_id]
    if not row.empty:
        return float(row.iloc[0][col_name])
    return None

def get_json_item(json_path, example_id):
    with open(json_path, 'r') as f:
        data = json.load(f)
    for item in data:
        if item.get('example_id') == example_id:
            return item
    return None

# Get JSON data
t1_data = get_json_item(json_t1_path, example_id_target)
t2_data = get_json_item(json_t2_path, example_id_target)

# Get Scores
summac_t1 = get_score(summac_t1_path, 'summac_score', example_id_target)
summac_t2 = get_score(summac_t2_path, 'summac_score', example_id_target)
bart_t1 = get_score(bart_t1_path, 'bartscore_sum2doc', example_id_target)
bart_t2 = get_score(bart_t2_path, 'bartscore_sum2doc', example_id_target)
align_t1 = get_score(align_t1_path, 'alignscore', example_id_target)
align_t2 = get_score(align_t2_path, 'alignscore', example_id_target)

output = {
    "example_id": example_id_target,
    "reference_documents": t1_data.get("reference", "") if t1_data else "",
    "anchors": t1_data.get("anchors", []) if t1_data else [],
    "T1_Revision_1": {
        "final_summary": t1_data.get("final_summary", "") if t1_data else "",
        "history": t1_data.get("history", []) if t1_data else [],
        "scores": {
            "SummaCConv": summac_t1,
            "BARTScore_s2d": bart_t1,
            "AlignScore": align_t1
        },
        "length": len(t1_data.get("final_summary", "")) if t1_data else 0
    },
    "T2_Revision_2": {
        "final_summary": t2_data.get("final_summary", "") if t2_data else "",
        "history": t2_data.get("history", []) if t2_data else [],
        "scores": {
            "SummaCConv": summac_t2,
            "BARTScore_s2d": bart_t2,
            "AlignScore": align_t2
        },
        "length": len(t2_data.get("final_summary", "")) if t2_data else 0
    },
    "Differences": {
        "Delta_SummaCConv": summac_t2 - summac_t1 if summac_t1 is not None and summac_t2 is not None else None,
        "Delta_BARTScore_s2d": bart_t2 - bart_t1 if bart_t1 is not None and bart_t2 is not None else None,
        "Delta_AlignScore": align_t2 - align_t1 if align_t1 is not None and align_t2 is not None else None,
        "Delta_Length": len(t2_data.get("final_summary", "")) - len(t1_data.get("final_summary", "")) if t1_data and t2_data else None
    }
}

out_file = os.path.join(base_dir, "example_334_analysis.json")
with open(out_file, "w") as f:
    json.dump(output, f, indent=4)

print(f"Extracted data saved to {out_file}")
