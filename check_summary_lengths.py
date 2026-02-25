#!/usr/bin/env python3
"""
Check number of summaries exceeding 1024 tokens using BART tokenizer
For all 4 baseline ablation files
"""

import json
import os
from transformers import BartTokenizer
import sys

def load_ablation_data(file_path):
    """Load ablation data from JSON file"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def check_summary_lengths(data, config_name, tokenizer):
    """Check summary lengths using BART tokenizer"""
    print(f"\n🔍 Analyzing {config_name}")
    print("=" * 50)
    
    total_samples = len(data)
    exceeding_1024 = 0
    exceeding_2048 = 0
    exceeding_4096 = 0
    
    lengths = []
    
    for i, item in enumerate(data):
        summary = item.get('final_summary', '')
        example_id = item.get('example_id', i)
        
        # Tokenize with BART tokenizer
        tokens = tokenizer.encode(summary, truncation=False)
        token_count = len(tokens)
        lengths.append(token_count)
        
        # Check thresholds
        if token_count > 1024:
            exceeding_1024 += 1
        if token_count > 2048:
            exceeding_2048 += 1
        if token_count > 4096:
            exceeding_4096 += 1
        
        # Show first few examples that exceed 1024
        if token_count > 1024 and exceeding_1024 <= 5:
            print(f"📝 Sample {example_id}: {token_count} tokens")
            print(f"   Summary: {summary[:200]}...")
    
    # Calculate statistics
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    max_length = max(lengths) if lengths else 0
    min_length = min(lengths) if lengths else 0
    
    print(f"\n📊 Statistics for {config_name}:")
    print(f"   Total samples: {total_samples}")
    print(f"   Average tokens: {avg_length:.1f}")
    print(f"   Min tokens: {min_length}")
    print(f"   Max tokens: {max_length}")
    print(f"   > 1024 tokens: {exceeding_1024} ({exceeding_1024/total_samples*100:.1f}%)")
    print(f"   > 2048 tokens: {exceeding_2048} ({exceeding_2048/total_samples*100:.1f}%)")
    print(f"   > 4096 tokens: {exceeding_4096} ({exceeding_4096/total_samples*100:.1f}%)")
    
    return {
        'config_name': config_name,
        'total_samples': total_samples,
        'exceeding_1024': exceeding_1024,
        'exceeding_2048': exceeding_2048,
        'exceeding_4096': exceeding_4096,
        'avg_length': avg_length,
        'max_length': max_length,
        'min_length': min_length
    }

def main():
    """Check summary lengths for all 4 baseline ablation files"""
    
    print("🔍 Checking Summary Token Lengths (BART Tokenizer)")
    print("=" * 60)
    
    # Initialize BART tokenizer
    print("📦 Loading BART tokenizer...")
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    
    # Define the 4 baseline files
    baseline_files = [
        ('base', 'data/ablations/base/ablation_base_final_500.json'),
        ('no_nli', 'data/ablations/no_nli/ablation_no_nli_final_500.json'),
        ('no_entity', 'data/ablations/no_entity/ablation_no_entity_final_500.json'),
        ('full', 'data/ablations/full/ablation_full_final_500.json')
    ]
    
    results = []
    
    for config_name, file_path in baseline_files:
        data = load_ablation_data(file_path)
        if data:
            result = check_summary_lengths(data, config_name, tokenizer)
            results.append(result)
    
    # Summary table
    print("\n" + "=" * 60)
    print("📋 SUMMARY TABLE")
    print("=" * 60)
    print(f"{'Config':<12} {'Total':<8} {'>1024':<8} {'>2048':<8} {'>4096':<8} {'Avg':<8} {'Max':<8}")
    print("-" * 70)
    
    for result in results:
        print(f"{result['config_name']:<12} {result['total_samples']:<8} {result['exceeding_1024']:<8} "
              f"{result['exceeding_2048']:<8} {result['exceeding_4096']:<8} "
              f"{result['avg_length']:<8.0f} {result['max_length']:<8}")
    
    print("\n" + "=" * 60)
    print("🎯 Key Findings:")
    
    total_exceeding_1024 = sum(r['exceeding_1024'] for r in results)
    total_samples = sum(r['total_samples'] for r in results)
    
    print(f"   Total samples across all baselines: {total_samples}")
    print(f"   Total summaries > 1024 tokens: {total_exceeding_1024}")
    print(f"   Overall percentage > 1024 tokens: {total_exceeding_1024/total_samples*100:.1f}%")

if __name__ == "__main__":
    main()
