import os
import json
import logging
import torch
import random
from dotenv import load_dotenv
from datasets import load_dataset
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipeline import AnchorSumPipeline

# Load environment variables from .env file
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
ENTITY_MODEL = "en_core_web_trf"
HF_TOKEN = os.getenv("HF_TOKEN")
SEED = 42
SAMPLE_SIZE = 500
MAX_REVISIONS = 1  # Support continuous correction loops

def run_experiment(config_name, ablation_flags=None):
    """
    Runs a specific ablation experiment.
    ablation_flags: a dict to disable components, e.g., {'nli': False, 'entity': False}
    """
    # Create experiment-specific output directory
    output_dir = f"data/ablations/{config_name}"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {output_dir}")
    
    # Load and Sample Dataset (Fixed Seed)
    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
    
    # Set random seeds for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Sample dataset
    sampled_dataset = dataset.shuffle(seed=SEED).select(range(SAMPLE_SIZE))
    
    # Initialize pipeline with ablation flags
    pipeline = AnchorSumPipeline(
        model_name=MODEL_NAME,
        nli_model_name=NLI_MODEL,
        entity_model_name=ENTITY_MODEL,
        token=HF_TOKEN,
        **(ablation_flags or {})
    )
    
    results = []
    for i, example in enumerate(sampled_dataset):
        if i % 10 == 0:
            logger.info(f"Processing example {i+1}/{SAMPLE_SIZE}")
        
        document = example['document']
        reference_summary = example['summary']
        
        try:
            result = pipeline.process(document, reference_summary)
            result['config_name'] = config_name
            result['example_id'] = i
            results.append(result)
            # Intermediate Save (Every 50 samples)
            if (i + 1) % 50 == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_{i+1}_samples.json")
                with open(checkpoint_path, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"💾 Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error processing example {i}: {str(e)}")
            continue
    
    # Save Final results (Consolidated 1000 sample file)
    final_output_path = os.path.join(output_dir, f"ablation_{config_name}_final_500.json")
    with open(final_output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Experiment {config_name} completed. Final results saved to {final_output_path}")
    
    return results

if __name__ == "__main__":
    # Run different ablation experiments
    pass
