#!/usr/bin/env python3
"""
Continue Experiment from Checkpoint 350
Generates checkpoints 400, 450, 500 and final file for full_revisions_2
"""

import os
import json
import logging
import torch
import random
from dotenv import load_dotenv
from datasets import load_dataset
import sys
from pathlib import Path

# Add the src directory to path properly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_dir)

from pipeline import AnchorSumPipeline

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
ENTITY_MODEL = "en_core_web_trf"
HF_TOKEN = os.getenv("HF_TOKEN")
SEED = 42
SAMPLE_SIZE = 500
MAX_REVISIONS = 2  # Same as full_revisions_2

def continue_from_checkpoint_350():
    """Continue experiment from checkpoint 350 to completion"""
    output_dir = f"data/ablations/full_revisions_2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load checkpoint 350
    checkpoint_350_path = os.path.join(output_dir, "checkpoint_350_samples.json")
    logger.info(f"Loading checkpoint from {checkpoint_350_path}")
    
    if not os.path.exists(checkpoint_350_path):
        logger.error(f"Checkpoint file not found: {checkpoint_350_path}")
        return
    
    with open(checkpoint_350_path, 'r') as f:
        results = json.load(f)
    
    # Get the set of already processed example_ids
    processed_ids = set(item['example_id'] for item in results)
    max_processed_id = max(processed_ids) if processed_ids else -1
    
    logger.info(f"Loaded {len(results)} samples from checkpoint 350")
    logger.info(f"Processed example_ids range: 0 to {max_processed_id}")
    logger.info(f"Missing example_ids: {[i for i in range(max_processed_id+1) if i not in processed_ids]}")
    
    # Load original dataset
    logger.info("Loading original dataset...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    # Set random seeds for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Get first 500 samples (same shuffle as original)
    first_500_samples = dataset.shuffle(seed=SEED).select(range(500))
    logger.info(f"Loaded {len(first_500_samples)} samples from original dataset")
    
    # Initialize pipeline
    logger.info(f"Initializing pipeline with max_revisions={MAX_REVISIONS}")
    pipeline = AnchorSumPipeline(
        model_name=MODEL_NAME,
        nli_model_name=NLI_MODEL,
        entity_model_name=ENTITY_MODEL,
        token=HF_TOKEN,
        max_revisions=MAX_REVISIONS
    )
    
    # Continue from example_id 350 onwards
    start_id = 350
    logger.info(f"Continuing from example_id {start_id} to 499")
    
    for i in range(start_id, 500):
        if i % 10 == 0:
            logger.info(f"Processing example_id {i}/499")
        
        sample = first_500_samples[i]
        
        # Handle both dict and string formats
        if isinstance(sample, dict):
            original_doc = sample['document']
            original_ref = sample['summary']
        elif isinstance(sample, str):
            logger.warning(f"Sample {i} is string, not dict. Skipping.")
            continue
        else:
            logger.error(f"Unknown sample format at index {i}: {type(sample)}")
            continue
        
        try:
            # Process with pipeline
            result = pipeline.process(original_doc, original_ref)
            result['config_name'] = 'full_revisions_2'
            result['example_id'] = i
            result['max_revisions_used'] = MAX_REVISIONS
            results.append(result)
            
            # Save checkpoint every 50 samples (at 400, 450, 500)
            if (i + 1) in [400, 450, 500]:
                checkpoint_file = os.path.join(output_dir, f"checkpoint_{i+1}_samples.json")
                with open(checkpoint_file, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"✅ Saved checkpoint to {checkpoint_file} ({len(results)} samples, last example_id: {i})")
                
        except Exception as e:
            logger.error(f"❌ Error processing sample {i}: {str(e)}")
            # Continue with next sample
            continue
    
    # Save final results
    final_file = os.path.join(output_dir, "ablation_full_revisions_2_final_500.json")
    with open(final_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"✅ Saved final results to {final_file} ({len(results)} total samples)")
    
    logger.info("🎉 Experiment continuation completed!")
    return results

if __name__ == "__main__":
    print("🚀 Continuing Experiment from Checkpoint 350")
    print("=" * 60)
    print("📋 Configuration:")
    print(f"   Output dir: data/ablations/full_revisions_2")
    print(f"   Starting from: example_id 350")
    print(f"   Target: example_id 499")
    print(f"   Max revisions: {MAX_REVISIONS}")
    print(f"   Note: Will create checkpoints at 400, 450, 500 samples")
    print()
    
    continue_from_checkpoint_350()
