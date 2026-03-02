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

sys.path.append(str(Path(__file__).parent / "summac"))
try:
    from summac.model_summac import SummaCConv
    has_summac = True
except ImportError as e:
    print(f"SummaC import failed: {e}")
    has_summac = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

root_dir = Path(__file__).resolve().parent.parent.parent


class SimpleSummaCEvaluator:
    def __init__(self, device='cuda:0', batch_size=4):
        self.device = device
        self.batch_size = batch_size
        self.model = None

        self.seed = 42
        self.sample_size = 500

        if has_summac:
            try:
                logger.info("Initializing SummaCConv with vitc...")
                self.model = SummaCConv(
                    models=["vitc"],
                    bins='percentile',
                    granularity="sentence",
                    nli_labels="e",
                    device=device,
                    imager_load_cache=True,
                    agg="mean"
                )
                logger.info("SummaCConv initialized")
            except Exception as e:
                logger.error(f"SummaCConv init failed: {e}")
                self.model = None

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
        matched_documents = []

        for item in data:
            example_id = item.get('example_id', -1)
            if example_id >= 0 and example_id < len(original_documents):
                summaries.append(item['final_summary'])
                reference_summaries.append(item['reference'])
                config_names.append(item.get('config_name', 'unknown'))
                example_ids.append(example_id)
                matched_documents.append(original_documents[example_id]['document'])
            else:
                logger.warning(f"Skipping sample {example_id} - no matching document")

        logger.info(f"Evaluating {len(summaries)} matched samples")

        if self.model is not None:
            logger.info("Using SummaCConv for evaluation...")
            scores = []

            for i in tqdm(range(0, len(summaries), self.batch_size), desc="SummaC Evaluation"):
                end_idx = min(i + self.batch_size, len(summaries))
                batch_summaries = summaries[i:end_idx]
                batch_documents = matched_documents[i:end_idx]

                try:
                    batch_result = self.model.score(batch_summaries, batch_documents)

                    if isinstance(batch_result, dict) and 'scores' in batch_result:
                        batch_scores = batch_result['scores']
                    elif isinstance(batch_result, (list, np.ndarray)):
                        batch_scores = list(batch_result)
                    else:
                        batch_scores = [0.0] * len(batch_summaries)

                    scores.extend(batch_scores)

                except Exception as e:
                    logger.error(f"Batch {i}-{end_idx} failed: {e}")
                    batch_scores = self._fallback_score(batch_summaries, batch_documents)
                    scores.extend(batch_scores)

            method = 'SummaCConv_vitc'
            logger.info("SummaCConv evaluation completed")

        else:
            logger.warning("Using fallback consistency scoring")
            scores = self._fallback_score(summaries, matched_documents)
            method = 'fallback'

        for i, (example_id, score) in enumerate(zip(example_ids, scores)):
            results.append({
                'id': example_id,
                'summac_score': float(score),
                'method': method
            })

        return pd.DataFrame(results)

    def _fallback_score(self, summaries: List[str], documents: List[str]) -> List[float]:
        scores = []
        for summary, document in zip(summaries, documents):
            summary_words = set(summary.lower().split())
            document_words = set(document.lower().split())

            if len(summary_words) == 0:
                scores.append(0.0)
            else:
                overlap = len(summary_words & document_words)
                consistency = overlap / len(summary_words)
                scores.append(consistency)

        return scores

    def calculate_summary_stats(self, df: pd.DataFrame, config_name: str) -> Dict[str, Any]:
        stats = {'config_name': config_name}

        metric_data = df['summac_score']
        stats['summac_mean'] = float(metric_data.mean())
        stats['summac_median'] = float(metric_data.median())
        stats['summac_std'] = float(metric_data.std())
        stats['summac_min'] = float(metric_data.min())
        stats['summac_max'] = float(metric_data.max())
        stats['method'] = df['method'].iloc[0] if len(df) > 0 else 'unknown'

        return stats

    def evaluate_file(self, file_path: str, output_dir: str):
        config_name = Path(file_path).stem.replace('ablation_', '').replace('_final_500', '')
        logger.info(f"Evaluating: {config_name}")

        data = self.load_ablation_data(file_path)
        original_documents = self.load_original_dataset()

        results_df = self.calculate_scores(data, original_documents)

        stats = self.calculate_summary_stats(results_df, config_name)

        detailed_path = os.path.join(output_dir, f'summac_detailed_{config_name}.csv')
        summary_path = os.path.join(output_dir, f'summac_summary_{config_name}.csv')

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

    output_dir = str(root_dir / 'Results' / 'Component Ablation ' / 'summac_final_results')
    os.makedirs(output_dir, exist_ok=True)

    if has_summac:
        print("SummaCConv available - using research-grade method")
    else:
        print("SummaCConv not available - using fallback method")

    evaluator = SimpleSummaCEvaluator(device='cuda:0', batch_size=4)

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

        print("\nFINAL SUMMARY:")
        print("-" * 40)
        for _, row in combined_summary.iterrows():
            config = row['config_name']
            method = row.get('method', 'unknown')
            print(f"\n{config.upper()} (Method: {method}):")
            print(f"   SummaC Mean:   {row['summac_mean']:.3f}")
            print(f"   SummaC Median: {row['summac_median']:.3f}")
            print(f"   SummaC Std:    {row['summac_std']:.3f}")
            print(f"   Range: [{row['summac_min']:.3f}, {row['summac_max']:.3f}]")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
