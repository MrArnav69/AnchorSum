import json
import numpy as np
from rouge_score import rouge_scorer
import argparse
import sys
import os

def calculate_rouge(results_path):
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    r1_scores = []
    r2_scores = []
    rl_scores = []
    total_flags = 0
    samples_with_flags = 0

    for item in data:
        ref = item.get('reference', '')
        hyp = item.get('final_summary', item.get('anchorsum', ''))
        
        if not ref or not hyp:
            continue
            
        scores = scorer.score(ref, hyp)
        r1_scores.append(scores['rouge1'].fmeasure)
        r2_scores.append(scores['rouge2'].fmeasure)
        rl_scores.append(scores['rougeL'].fmeasure)

        history = item.get('history', [])
        sample_flags = 0
        for step in history:
            flags = step.get('flags', [])
            sample_flags += len(flags)
        
        total_flags += sample_flags
        if sample_flags > 0:
            samples_with_flags += 1

    print("\n" + "="*40)
    print(f"ROUGE Results for: {os.path.basename(results_path)}")
    print("="*40)
    print(f"Total Samples: {len(r1_scores)}")
    print(f"ROUGE-1: {np.mean(r1_scores)*100:.2f}")
    print(f"ROUGE-2: {np.mean(r2_scores)*100:.2f}")
    print(f"ROUGE-L: {np.mean(rl_scores)*100:.2f}")
    print("-" * 20)
    print(f"Avg Flags Per Sample: {total_flags/len(r1_scores):.2f}")
    print(f"Samples Needing Revision: {samples_with_flags} ({samples_with_flags/len(r1_scores)*100:.1f}%)")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    args = parser.parse_args()
    calculate_rouge(args.file)
