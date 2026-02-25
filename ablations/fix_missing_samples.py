#!/usr/bin/env python3
"""
Fix missing samples (IDs 47 and 330) in all ablation files
Uses the same logic as ablation_base_runner.py
"""

import os
import json
import logging
import torch
import random
from dotenv import load_dotenv
from datasets import load_dataset
import sys

# Use the same import logic as ablation_base_runner.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import AnchorSumPipeline

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants matching ablation_base_runner.py
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
ENTITY_MODEL = "en_core_web_trf"
HF_TOKEN = os.getenv("HF_TOKEN")
SEED = 42

def load_existing_data(file_path):
    """Load existing ablation data"""
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_missing_ids(data):
    """Get missing IDs from 0-499"""
    existing_ids = {item.get('example_id', i) for i, item in enumerate(data)}
    missing_ids = [i for i in range(500) if i not in existing_ids]
    return missing_ids

def generate_missing_samples(config_name, ablation_flags, missing_ids):
    """Generate missing samples for a specific configuration"""
    logger.info(f"Generating {len(missing_ids)} missing samples for {config_name}: {missing_ids}")
    
    # Load original dataset with same logic as ablation runner
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    random.seed(SEED)
    torch.manual_seed(SEED)
    sampled_dataset = dataset.shuffle(seed=SEED).select(range(500))
    
    try:
        # Initialize pipeline with ablation flags
        pipeline = AnchorSumPipeline(
            model_name=MODEL_NAME,
            nli_model_name=NLI_MODEL,
            entity_model_name=ENTITY_MODEL,
            token=HF_TOKEN,
            **ablation_flags
        )
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        logger.error("Please ensure spacy model is installed: python -m spacy download en_core_web_trf")
        return []
    
    # Generate missing samples
    results = []
    for sample_id in missing_ids:
        logger.info(f"Processing sample {sample_id}")
        
        # Get the specific sample
        sample = sampled_dataset[sample_id]
        document = sample['document']
        reference_summary = sample['summary']
        
        try:
            result = pipeline.process(document, reference_summary)
            result['config_name'] = config_name
            result['example_id'] = sample_id
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_id}: {str(e)}")
            continue
    
    return results

def fix_ablation_file(config_name, ablation_flags):
    """Fix missing samples in a specific ablation file"""
    file_path = f"data/ablations/{config_name}/ablation_{config_name}_final_500.json"
    
    logger.info(f"\n🔧 Fixing {config_name}...")
    
    # Load existing data
    existing_data = load_existing_data(file_path)
    logger.info(f"Loaded {len(existing_data)} existing samples")
    
    # Find missing IDs
    missing_ids = get_missing_ids(existing_data)
    
    if not missing_ids:
        logger.info("✅ No missing samples found")
        return
    
    logger.info(f"❌ Missing IDs: {missing_ids}")
    
    # Generate missing samples
    new_samples = generate_missing_samples(config_name, ablation_flags, missing_ids)
    
    if not new_samples:
        logger.error("❌ Failed to generate any new samples")
        return
    
    # Merge existing and new data
    all_samples = existing_data + new_samples
    
    # Sort by example_id to maintain order
    all_samples.sort(key=lambda x: x.get('example_id', 0))
    
    # Save fixed data
    with open(file_path, 'w') as f:
        json.dump(all_samples, f, indent=2)
    
    logger.info(f"✅ Fixed {config_name}: {len(new_samples)} new samples added, total: {len(all_samples)}")

def main():
    """Fix missing samples in all ablation files"""
    
    # Define the 4 ablation configurations (same as run_all_sequential.py)
    experiments = [
        ("base", {'nli': False, 'entity': False, 'revision': False}),
        ("no_nli", {'nli': False, 'entity': True, 'revision': True}),
        ("no_entity", {'nli': True, 'entity': False, 'revision': True}),
        ("full", {'nli': True, 'entity': True, 'revision': True}),
    ]
    
    print("🔧 Fixing Missing Samples in All Ablation Files")
    print("=" * 60)
    
    for config_name, ablation_flags in experiments:
        fix_ablation_file(config_name, ablation_flags)
    
    print("\n" + "=" * 60)
    print("🎉 All missing samples have been fixed!")
    print("Files updated:")
    for config_name, _ in experiments:
        print(f"   - data/ablations/{config_name}/ablation_{config_name}_final_500.json")

if __name__ == "__main__":
    main()
