#!/usr/bin/env python3
"""
Quick BARTScore Test - Just first 10 samples
"""

import json
import pandas as pd
import torch
import os
import sys
from datasets import load_dataset

# Import BARTScore
sys.path.append(os.path.join(os.path.dirname(__file__), 'BARTScore'))
from bart_score import BARTScorer

def main():
    print("🧪 Quick BARTScore Test - First 10 samples")
    
    # Initialize BARTScore
    scorer = BARTScorer(device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn')
    
    # Load small sample
    with open('data/ablations/base/ablation_base_final_500.json', 'r') as f:
        data = json.load(f)
    
    # Take first 3 samples
    sample_data = data[:3]
    summaries = [item['final_summary'] for item in sample_data]
    references = [item['reference'] for item in sample_data]
    
    print(f"Testing {len(summaries)} samples...")
    
    # Calculate BARTScore
    sum2doc_scores = scorer.score(srcs=summaries, tgts=references, batch_size=3)
    doc2sum_scores = scorer.score(srcs=references, tgts=summaries, batch_size=3)
    
    # Create results
    results = []
    for i, (item, sum2doc, doc2sum) in enumerate(zip(sample_data, sum2doc_scores, doc2sum_scores)):
        results.append({
            'id': item.get('example_id', i),
            'bartscore_sum2doc': float(sum2doc),
            'bartscore_doc2sum': float(doc2sum)
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('test_bartscore_results.csv', index=False)
    
    print("✅ Results saved to test_bartscore_results.csv")
    print(df)

if __name__ == "__main__":
    main()
