import os
import json
import logging
import torch
import random
from dotenv import load_dotenv
from datasets import load_dataset
import sys

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
SAMPLE_SIZE = 200  # First 200 samples
MAX_REVISIONS = 3  # Run up to 3 revisions

def run_extended_revisions_experiment(config_name, max_revisions):
    """Run experiment with extended revisions on first 200 samples"""
    output_dir = f"data/ablations/{config_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original dataset with same logic as ablation runner (first 500 samples)
    logger.info("Loading original dataset...")
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    # Set random seeds for reproducibility (same as ablation runner)
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Get first 500 samples (same as original ablation runner)
    first_500_samples = dataset.shuffle(seed=SEED).select(range(500))
    logger.info(f"Loaded {len(first_500_samples)} samples from original dataset")
    
    # Debug: Check the format of first sample
    logger.info(f"Dataset features: {dataset.features}")
    logger.info(f"First sample type: {type(first_500_samples[0])}")
    if hasattr(first_500_samples[0], 'keys'):
        logger.info(f"First sample keys: {list(first_500_samples[0].keys())}")
    
    # Use only first 200 from those 500
    first_200_samples = first_500_samples[:200]
    logger.info(f"Processing first {len(first_200_samples)} samples from 500")
    
    # Initialize pipeline
    pipeline = AnchorSumPipeline(
        model_name=MODEL_NAME,
        nli_model_name=NLI_MODEL,
        entity_model_name=ENTITY_MODEL,
        token=HF_TOKEN,
        max_revisions=max_revisions
    )
    
    results = []
    for i, sample in enumerate(first_200_samples):
        if i % 10 == 0:
            logger.info(f"Processing sample {i+1}/{len(first_200_samples)}")
        
        # Handle both dict and string formats
        if isinstance(sample, dict):
            original_doc = sample['document']
            original_ref = sample['summary']
        elif isinstance(sample, str):
            # Some datasets return strings, need to parse
            # This is a fallback - you may need to adjust based on actual dataset format
            logger.warning(f"Sample {i} is string, not dict. Skipping.")
            continue
        else:
            logger.error(f"Unknown sample format at index {i}: {type(sample)}")
            continue
        
        try:
            # Process with extended revisions
            result = pipeline.process(original_doc, original_ref)
            result['config_name'] = config_name
            result['example_id'] = i
            result['max_revisions_used'] = max_revisions
            results.append(result)
            
            # Save checkpoint every 50 samples
            if (i + 1) % 50 == 0:
                checkpoint_file = os.path.join(output_dir, f"checkpoint_{i+1}_samples.json")
                with open(checkpoint_file, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved checkpoint to {checkpoint_file}")
                
        except Exception as e:
            logger.error(f"Error processing sample {i}: {str(e)}")
            continue
    
    # Save final results
    final_path = os.path.join(output_dir, f"ablation_{config_name}_final_{len(results)}.json")
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Experiment {config_name} completed. Results saved to {final_path}")
    return results

if __name__ == "__main__":
    # Run experiments with max revisions 2 and 3
    for max_rev in [2, 3]:
        config_name = f"extended_revisions_{max_rev}"
        logger.info(f"Starting experiment with max_revisions={max_rev}")
        run_extended_revisions_experiment(config_name, max_rev)
        logger.info(f"Completed experiment with max_revisions={max_rev}")
