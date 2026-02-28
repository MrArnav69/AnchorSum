#!/usr/bin/env python3
"""
Continue full_revisions_2 experiment from checkpoint 350 to 500 samples
Uses the modified ablation_base_runner.py
"""

import sys
import os
import json

# Add the ablations directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ablation_base_runner import run_experiment, SEED, MODEL_NAME, NLI_MODEL, ENTITY_MODEL, HF_TOKEN, logger

# Additional imports for continuation
import torch
import random
from datasets import load_dataset
from src.pipeline import AnchorSumPipeline

def continue_full_revisions_2_experiment():
    """Continue full_revisions_2 from checkpoint 350 to 500 samples"""
    
    config_name = 'full_revisions_2'
    max_revisions = 2
    sample_size = 500
    
    output_dir = f"data/ablations/{config_name}"
    checkpoint_path = os.path.join(output_dir, "checkpoint_350_samples.json")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Run the original experiment first or check the path.")
        return
    
    print("🚀 Continuing full_revisions_2 Experiment")
    print("=" * 60)
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    # Load existing results
    with open(checkpoint_path, 'r') as f:
        results = json.load(f)
    
    processed_ids = set(item['example_id'] for item in results)
    max_id = max(processed_ids) if processed_ids else -1
    
    print(f"✅ Loaded {len(results)} samples (example_ids 0-{max_id})")
    print(f"📊 Missing example_ids: {[i for i in range(max_id+1) if i not in processed_ids]}")
    print(f"🎯 Continuing from example_id 350 to 499")
    print()
    
    # Load dataset with same seed
    from dotenv import load_dotenv
    load_dotenv()
    
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    random.seed(SEED)
    torch.manual_seed(SEED)
    sampled_dataset = dataset.shuffle(seed=SEED).select(range(sample_size))
    
    # Initialize pipeline
    pipeline = AnchorSumPipeline(
        model_name=MODEL_NAME,
        nli_model_name=NLI_MODEL,
        entity_model_name=ENTITY_MODEL,
        token=os.getenv("HF_TOKEN"),
        max_revisions=max_revisions
    )
    
    # Continue from example_id 350
    for i in range(350, sample_size):
        if i % 10 == 0:
            print(f"Processing example_id {i}/{sample_size-1}")
        
        example = sampled_dataset[i]
        document = example['document']
        reference_summary = example['summary']
        
        try:
            result = pipeline.process(document, reference_summary)
            result['config_name'] = config_name
            result['example_id'] = i
            results.append(result)
            
            # Save checkpoint at 400, 450, 500
            if (i + 1) in [400, 450, 500]:
                checkpoint_file = os.path.join(output_dir, f"checkpoint_{i+1}_samples.json")
                with open(checkpoint_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"💾 Saved checkpoint: {checkpoint_file} ({len(results)} samples)")
                
        except Exception as e:
            print(f"❌ Error processing example_id {i}: {str(e)}")
            continue
    
    # Save final results
    final_path = os.path.join(output_dir, f"ablation_{config_name}_final_500.json")
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print("=" * 60)
    print("🎉 Experiment completed!")
    print(f"📁 Total samples: {len(results)}")
    print(f"📁 Final file: {final_path}")
    print(f"📁 Checkpoints: 400, 450, 500 samples")

def main():
    """Main entry point - only continues full_revisions_2"""
    continue_full_revisions_2_experiment()

if __name__ == "__main__":
    main()
