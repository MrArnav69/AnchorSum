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

try:
    from bertscore import score as bert_score
    has_bertscore = True
except ImportError:
    try:
        from bert_score import score as bert_score
        has_bertscore = True
    except ImportError:
        print("BERTScore not available - install with: pip install bertscore")
        has_bertscore = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

root_dir = Path(__file__).resolve().parent.parent.parent


class SimpleBERTScoreEvaluator:
    def __init__(self, model_type='microsoft/deberta-xlarge-mnli', device='cuda:0', batch_size=8):
        self.device = device
        self.batch_size = batch_size
        self.model_type = model_type

        self.seed = 42
        self.sample_size = 500

        logger.info(f"BERTScore model: {model_type}")

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
        matched_references = []

        for item in data:
            example_id = item.get('example_id', -1)
            if example_id >= 0 and example_id < len(original_documents):
                summaries.append(item['final_summary'])
                config_names.append(item.get('config_name', 'unknown'))
                example_ids.append(example_id)
                matched_references.append(original_documents[example_id]['summary'])
            else:
                logger.warning(f"Skipping sample {example_id} - no matching document")

        logger.info(f"Evaluating {len(summaries)} matched samples")

        if has_bertscore and len(summaries) > 0:
            logger.info(f"Using BERTScore with {self.model_type}...")

            try:
                P, R, F1 = bert_score(
                    summaries,
                    matched_references,
                    model_type=self.model_type,
                    device=self.device,
                    batch_size=self.batch_size,
                    lang='en',
                    verbose=True
                )

                precision_scores = P.cpu().numpy().tolist()
                recall_scores = R.cpu().numpy().tolist()
                f1_scores = F1.cpu().numpy().tolist()

                method = f'BERTScore-{self.model_type.split("/")[-1]}'
                logger.info("BERTScore evaluation completed")

            except Exception as e:
                logger.error(f"BERTScore calculation failed: {e}")
                precision_scores = [0.0] * len(summaries)
                recall_scores = [0.0] * len(summaries)
                f1_scores = [0.0] * len(summaries)
                method = 'fallback'
        else:
            logger.warning("Using fallback scoring")
            precision_scores = [0.0] * len(summaries)
            recall_scores = [0.0] * len(summaries)
            f1_scores = [0.0] * len(summaries)
            method = 'fallback'

        for i, (example_id, p, r, f1) in enumerate(zip(example_ids, precision_scores, recall_scores, f1_scores)):
            results.append({
                'id': example_id,
                'bertscore_precision': float(p),
                'bertscore_recall': float(r),
                'bertscore_f1': float(f1),
                'method': method
            })

        return pd.DataFrame(results)

    def calculate_summary_stats(self, df: pd.DataFrame, config_name: str) -> Dict[str, Any]:
        stats = {'config_name': config_name}

        for metric in ['bertscore_precision', 'bertscore_recall', 'bertscore_f1']:
            metric_data = df[metric]
            stats[f'{metric}_mean'] = float(metric_data.mean())
            stats[f'{metric}_median'] = float(metric_data.median())
            stats[f'{metric}_std'] = float(metric_data.std())
            stats[f'{metric}_min'] = float(metric_data.min())
            stats[f'{metric}_max'] = float(metric_data.max())

        stats['method'] = df['method'].iloc[0] if len(df) > 0 else 'unknown'

        return stats

    def evaluate_file(self, file_path: str, output_dir: str):
        config_name = Path(file_path).stem.replace('ablation_', '').replace('_final_500', '')
        logger.info(f"Evaluating: {config_name}")

        data = self.load_ablation_data(file_path)
        original_documents = self.load_original_dataset()

        results_df = self.calculate_scores(data, original_documents)

        stats = self.calculate_summary_stats(results_df, config_name)

        detailed_path = os.path.join(output_dir, f'bertscore_detailed_{config_name}.csv')
        summary_path = os.path.join(output_dir, f'bertscore_summary_{config_name}.csv')

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

    model_type = 'microsoft/deberta-xlarge-mnli'

    output_dir = str(root_dir / 'Results' / 'Component Ablation ' / 'bertscore_xlarge_results')
    os.makedirs(output_dir, exist_ok=True)

    if has_bertscore:
        print("BERTScore available")
    else:
        print("BERTScore not available - install with: pip install bertscore")

    evaluator = SimpleBERTScoreEvaluator(
        model_type=model_type,
        device='cuda:0',
        batch_size=4
    )

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

        print("\nBERTSCORE SUMMARY (DeBERTa-XLarge):")
        print("-" * 40)
        for _, row in combined_summary.iterrows():
            config = row['config_name']
            method = row.get('method', 'unknown')
            print(f"\n{config.upper()} (Method: {method}):")
            print(f"   Precision: {row['bertscore_precision_mean']:.4f} (+/-{row['bertscore_precision_std']:.4f})")
            print(f"   Recall:    {row['bertscore_recall_mean']:.4f} (+/-{row['bertscore_recall_std']:.4f})")
            print(f"   F1:        {row['bertscore_f1_mean']:.4f} (+/-{row['bertscore_f1_std']:.4f})")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
