import os
import json
import logging
import torch
import random
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipeline import anchorsumpipeline

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model_name = "meta-llama/Llama-3.1-8B-Instruct"
nli_model = "cross-encoder/nli-deberta-v3-large"
entity_model = "en_core_web_trf"
hf_token = os.getenv("HF_TOKEN")
seed = 42
sample_size = 500
max_revisions = 1


def run_experiment(config_name, ablation_flags=None, max_revisions=1, sample_size=500):
    output_dir = str(Path(__file__).resolve().parent.parent / 'data' / 'ablations' / config_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {output_dir}")

    dataset = load_dataset("Awesome075/multi_news_parquet", split="test")

    random.seed(seed)
    torch.manual_seed(seed)

    sampled_dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    pipeline = anchorsumpipeline(
        model_name=model_name,
        nli_model_name=nli_model,
        entity_model_name=entity_model,
        token=hf_token,
        max_revisions=max_revisions,
        **(ablation_flags or {})
    )

    results = []
    for i, example in enumerate(sampled_dataset):
        if i % 10 == 0:
            logger.info(f"Processing example {i+1}/{sample_size}")

        document = example['document']
        reference_summary = example['summary']

        try:
            result = pipeline.process(document, reference_summary)
            result['config_name'] = config_name
            result['example_id'] = i
            results.append(result)
            if (i + 1) % 50 == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_{i+1}_samples.json")
                with open(checkpoint_path, "w") as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error processing example {i}: {str(e)}")
            continue

    final_output_path = os.path.join(output_dir, f"ablation_{config_name}_final_500.json")
    with open(final_output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Experiment {config_name} completed. Final results saved to {final_output_path}")

    return results


if __name__ == "__main__":
    pass
