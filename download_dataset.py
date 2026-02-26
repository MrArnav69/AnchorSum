#!/usr/bin/env python3
"""
Download and save Multi-News dataset samples locally
Saves to data/multi_news_500_samples.json with proper indexing
"""

import json
import os
from datasets import load_dataset
import torch
import random

SEED = 42
SAMPLE_SIZE = 500

def download_and_save_dataset():
    """Download dataset and save locally with proper indexing"""
    
    print("📥 Downloading Multi-News dataset...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    # Set random seeds for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Sample dataset
    sampled_dataset = dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))
    
    print(f"✅ Loaded {len(sampled_dataset)} samples")
    
    # Convert to list with proper indexing
    samples = []
    for i, example in enumerate(sampled_dataset):
        samples.append({
            'example_id': i,
            'document': example['document'],
            'summary': example['summary']
        })
    
    # Save to JSON
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'multi_news_500_samples.json')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved to {output_file}")
    print(f"📊 Total samples: {len(samples)}")
    print(f"📊 Example ID range: 0 to {len(samples)-1}")
    
    # Verify first and last samples
    print(f"\n✅ First sample ID: {samples[0]['example_id']}")
    print(f"✅ Last sample ID: {samples[-1]['example_id']}")
    
    return output_file

if __name__ == "__main__":
    print("=" * 60)
    print("📥 DATASET DOWNLOADER")
    print("=" * 60)
    download_and_save_dataset()
    print("\n🎉 Dataset saved successfully!")
