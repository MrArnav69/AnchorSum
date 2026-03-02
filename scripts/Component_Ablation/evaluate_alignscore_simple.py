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

sys.path.append(str(Path(__file__).parent / "AlignScore"))
try:
    from alignscore import AlignScore
    has_alignscore = True
except ImportError as e:
    print(f"AlignScore import failed: {e}")
    has_alignscore = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

root_dir = Path(__file__).resolve().parent.parent.parent


class SimpleAlignScoreEvaluator:
    def __init__(self, device='cuda:0', batch_size=16):
        self.device = device
        self.batch_size = batch_size
        self.scorer = None

        self.seed = 42
        self.sample_size = 500

        if has_alignscore:
            try:
                ckpt_path = str(root_dir / 'AlignScore' / 'AlignScore-large.ckpt')
                if os.path.exists(ckpt_path):
                    logger.info("Loading AlignScore-large model...")
                    self.scorer = AlignScore(
                        model='roberta-large',
                        batch_size=batch_size,
                        device=device,
                        ckpt_path=ckpt_path,
                        evaluation_mode='nli_sp'
                    )
                    logger.info("AlignScore loaded")
                else:
                    logger.warning(f"Checkpoint not found: {ckpt_path}")
            except Exception as e:
                logger.error(f"AlignScore init failed: {e}")

    def load_original_dataset(self):
        logger.info("Loading original dataset from local file...")

        dataset_path = str(root_dir / 'data' / 'multi_news_500_samples.json')
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            return None

        with open(dataset_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        logger.info(f"Loaded {len(samples)} original documents from local file")
        return samples

    def load_ablation_data(self, file_path):
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples")
        return data

    def calculate_scores(self, data: List[Dict], original_documents) -> pd.DataFrame:
        results = []

        summaries = []
        config_names = []
        example_ids = []
        matched_documents = []

        for item in data:
            example_id = item.get('example_id', -1)
            if example_id >= 0 and example_id < len(original_documents):
                summaries.append(item['final_summary'])
                config_names.append(item.get('config_name', 'unknown'))
                example_ids.append(example_id)
                matched_documents.append(original_documents[example_id]['document'])
            else:
                logger.warning(f"Skipping sample {example_id} - no matching document")

        logger.info(f"Evaluating {len(summaries)} matched samples")

        if self.scorer is not None:
            logger.info("Using AlignScore for evaluation...")
            scores = []

            for i in tqdm(range(0, len(summaries), self.batch_size), desc="AlignScore Evaluation"):
                batch_summaries = summaries[i:i+self.batch_size]
                batch_documents = matched_documents[i:i+self.batch_size]

                try:
                    batch_scores = self.scorer.score(
                        contexts=batch_documents,
                        claims=batch_summaries
                    )
                    scores.extend(batch_scores)

                except Exception as e:
                    logger.error(f"Batch {i} failed: {e}")
                    scores.extend([0.0] * len(batch_summaries))

            method = 'AlignScore-large'
            logger.info("AlignScore evaluation completed")

        else:
            logger.warning("Using fallback scoring")
            scores = [0.0] * len(summaries)
            method = 'fallback'

        for i, (example_id, score) in enumerate(zip(example_ids, scores)):
            results.append({
                'id': example_id,
                'alignscore': float(score),
                'method': method
            })

        return pd.DataFrame(results)

    def calculate_summary_stats(self, df: pd.DataFrame, config_name: str) -> Dict[str, Any]:
        stats = {'config_name': config_name}

        metric_data = df['alignscore']
        stats['alignscore_mean'] = float(metric_data.mean())
        stats['alignscore_median'] = float(metric_data.median())
        stats['alignscore_std'] = float(metric_data.std())
        stats['alignscore_min'] = float(metric_data.min())
        stats['alignscore_max'] = float(metric_data.max())
        stats['method'] = df['method'].iloc[0] if len(df) > 0 else 'unknown'

        return stats

    def evaluate_file(self, file_path: str, output_dir: str):
        config_name = Path(file_path).stem.replace('ablation_', '').replace('_final_500', '')
        logger.info(f"Evaluating: {config_name}")

        data = self.load_ablation_data(file_path)
        original_documents = self.load_original_dataset()

        results_df = self.calculate_scores(data, original_documents)

        stats = self.calculate_summary_stats(results_df, config_name)

        detailed_path = os.path.join(output_dir, f'alignscore_detailed_{config_name}.csv')
        summary_path = os.path.join(output_dir, f'alignscore_summary_{config_name}.csv')

        os.makedirs(output_dir, exist_ok=True)

        results_df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"Detailed results: {detailed_path}")

        summary_df = pd.DataFrame([stats])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        logger.info(f"Summary statistics: {summary_path}")

        return results_df, stats


def main():
    base_dir = root_dir / 'data' / 'ablations'
    ablation_files = [
        str(base_dir / 'base'      / 'ablation_base_final_500.json'),
        str(base_dir / 'no_nli'    / 'ablation_no_nli_final_500.json'),
        str(base_dir / 'no_entity' / 'ablation_no_entity_final_500.json'),
        str(base_dir / 'full'      / 'ablation_full_final_500.json'),
    ]

    output_dir = str(root_dir / 'Results' / 'Component Ablation ' / 'alignscore_results')
    os.makedirs(output_dir, exist_ok=True)

    if has_alignscore:
        print("AlignScore available")
    else:
        print("AlignScore not available - install from AlignScore/ directory")

    evaluator = SimpleAlignScoreEvaluator(device='cuda:0', batch_size=16)

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

        print("\nALIGNSCORE SUMMARY:")
        print("-" * 40)
        for _, row in combined_summary.iterrows():
            config = row['config_name']
            method = row.get('method', 'unknown')
            print(f"\n{config.upper()} (Method: {method}):")
            print(f"   AlignScore Mean:   {row['alignscore_mean']:.3f}")
            print(f"   AlignScore Median: {row['alignscore_median']:.3f}")
            print(f"   AlignScore Std:    {row['alignscore_std']:.3f}")
            print(f"   Range: [{row['alignscore_min']:.3f}, {row['alignscore_max']:.3f}]")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
