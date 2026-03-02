#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
import torch
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm
from datasets import load_dataset
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'BARTScore'))
try:
    from bart_score import BARTScorer
except ImportError:
    print("BARTScore not found. Ensure 'BARTScore' folder exists.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

root_dir = Path(__file__).resolve().parent.parent.parent


class SimpleBARTScoreEvaluator:
    def __init__(self, device='cuda:0', batch_size=8):
        self.device = device
        self.batch_size = batch_size

        self.seed = 42
        self.sample_size = 500

        logger.info("Initializing BARTScore...")
        self.scorer = BARTScorer(
            device=device,
            max_length=1024,
            checkpoint='facebook/bart-large-cnn'
        )
        logger.info("BARTScore initialized successfully")

    def load_original_dataset(self):
        logger.info("Loading original dataset...")
        dataset = load_dataset("Awesome075/multi_news_parquet", split="test")

        import random
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        sampled_dataset = dataset.shuffle(seed=self.seed).select(range(self.sample_size))

        logger.info(f"Loaded {len(sampled_dataset)} original documents")
        return sampled_dataset

    def load_ablation_data(self, file_path):
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples")
        return data

    def calculate_scores(self, data: List[Dict], original_documents) -> pd.DataFrame:
        results = []

        summaries = []
        reference_summaries = []
        config_names = []
        example_ids = []

        for item in data:
            example_id = item.get('example_id', -1)
            if example_id >= 0 and example_id < len(original_documents):
                summaries.append(item['final_summary'])
                reference_summaries.append(item['reference'])
                config_names.append(item.get('config_name', 'unknown'))
                example_ids.append(example_id)
            else:
                logger.warning(f"Skipping sample {example_id} - no matching document")

        logger.info(f"Evaluating {len(summaries)} matched samples (skipped {len(data) - len(summaries)} unmatched)")

        max_length = 1024
        truncated_summaries = []
        for summary in summaries:
            if len(summary) > max_length * 4:
                truncated_summaries.append(summary[:max_length * 4])
            else:
                truncated_summaries.append(summary)

        documents = [doc['document'] for doc in original_documents]

        logger.info("Calculating BARTScore scores...")
        try:
            sum2doc_scores = []
            doc2sum_scores = []

            for i in tqdm(range(0, len(summaries), self.batch_size), desc="BARTScore Evaluation"):
                end_idx = min(i + self.batch_size, len(summaries))
                batch_summaries = truncated_summaries[i:end_idx]
                batch_documents = documents[i:end_idx]

                try:
                    batch_sum2doc = self.scorer.score(
                        srcs=batch_summaries,
                        tgts=batch_documents,
                        batch_size=len(batch_summaries)
                    )
                    batch_doc2sum = self.scorer.score(
                        srcs=batch_documents,
                        tgts=batch_summaries,
                        batch_size=len(batch_summaries)
                    )

                    sum2doc_scores.extend(batch_sum2doc)
                    doc2sum_scores.extend(batch_doc2sum)

                except Exception as e:
                    logger.error(f"Batch {i}-{end_idx} failed: {e}")
                    sum2doc_scores.extend([-5.0] * len(batch_summaries))
                    doc2sum_scores.extend([-5.0] * len(batch_summaries))

            logger.info("BARTScore calculated successfully")

        except Exception as e:
            logger.error(f"Error calculating BARTScore: {e}")
            sum2doc_scores = [-5.0] * len(summaries)
            doc2sum_scores = [-5.0] * len(summaries)

        for i, (example_id, config_name, sum2doc, doc2sum) in enumerate(
            zip(example_ids, config_names, sum2doc_scores, doc2sum_scores)
        ):
            results.append({
                'id': example_id,
                'bartscore_sum2doc': float(sum2doc),
                'bartscore_doc2sum': float(doc2sum),
                'truncated': len(summaries[i]) > max_length * 4
            })

        return pd.DataFrame(results)

    def calculate_summary_stats(self, df: pd.DataFrame, config_name: str) -> Dict[str, Any]:
        stats = {'config_name': config_name}

        metrics = ['bartscore_sum2doc', 'bartscore_doc2sum']

        for metric in metrics:
            metric_data = df[metric]
            stats[f'{metric}_mean'] = float(metric_data.mean())
            stats[f'{metric}_median'] = float(metric_data.median())
            stats[f'{metric}_std'] = float(metric_data.std())
            stats[f'{metric}_min'] = float(metric_data.min())
            stats[f'{metric}_max'] = float(metric_data.max())

        return stats

    def evaluate_file(self, file_path: str, output_dir: str):
        config_name = Path(file_path).stem.replace('ablation_', '').replace('_final_500', '')
        logger.info(f"Evaluating {config_name} configuration")

        data = self.load_ablation_data(file_path)
        original_documents = self.load_original_dataset()

        results_df = self.calculate_scores(data, original_documents)

        stats = self.calculate_summary_stats(results_df, config_name)

        detailed_path = os.path.join(output_dir, f'bartscore_detailed_{config_name}.csv')
        summary_path = os.path.join(output_dir, f'bartscore_summary_{config_name}.csv')

        os.makedirs(output_dir, exist_ok=True)

        results_df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"Detailed results saved to {detailed_path}")

        summary_df = pd.DataFrame([stats])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        logger.info(f"Summary statistics saved to {summary_path}")

        if os.path.exists(detailed_path) and os.path.exists(summary_path):
            logger.info("Files created successfully")
        else:
            logger.error("Failed to create output files")

        return results_df, stats


def main():
    base_dir = root_dir / 'data' / 'ablations'
    ablation_files = [
        str(base_dir / 'base'      / 'ablation_base_final_500.json'),
        str(base_dir / 'no_nli'    / 'ablation_no_nli_final_500.json'),
        str(base_dir / 'no_entity' / 'ablation_no_entity_final_500.json'),
        str(base_dir / 'full'      / 'ablation_full_final_500.json'),
    ]

    output_dir = str(root_dir / 'Results' / 'Component Ablation ' / 'bartscore_simple_results')
    os.makedirs(output_dir, exist_ok=True)

    evaluator = SimpleBARTScoreEvaluator(device='cuda:0', batch_size=8)

    all_results = []
    for file_path in ablation_files:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            continue

        try:
            results_df, stats = evaluator.evaluate_file(file_path, output_dir)
            all_results.append(stats)
            logger.info(f"Completed {file_path}")
        except Exception as e:
            logger.error(f"Failed to evaluate {file_path}: {str(e)}")

    if all_results:
        combined_summary = pd.DataFrame(all_results)
        combined_summary_path = os.path.join(output_dir, 'combined_summary.csv')
        combined_summary.to_csv(combined_summary_path, index=False, encoding='utf-8')
        logger.info(f"Combined summary saved to {combined_summary_path}")

        print("\nCOMBINED SUMMARY:")
        print("-" * 40)
        for _, row in combined_summary.iterrows():
            config = row['config_name']
            print(f"\n{config.upper()}:")
            print(f"   BARTScore sum2doc: {row['bartscore_sum2doc_mean']:.3f} (+/-{row['bartscore_sum2doc_std']:.3f})")
            print(f"   BARTScore doc2sum: {row['bartscore_doc2sum_mean']:.3f} (+/-{row['bartscore_doc2sum_std']:.3f})")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
