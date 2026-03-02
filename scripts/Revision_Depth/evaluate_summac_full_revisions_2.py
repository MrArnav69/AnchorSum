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
                config_names.append(item.get('config_name', 'full_revisions_2'))
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
        config_name = 'full_revisions_2'
        logger.info(f"Evaluating: {config_name}")
        
        data = self.load_ablation_data(file_path)
        original_documents = self.load_original_dataset()
        
        results_df = self.calculate_scores(data, original_documents)
        
        stats = self.calculate_summary_stats(results_df, config_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
        detailed_path = os.path.join(output_dir, f'summac_detailed_{config_name}.csv')
        results_df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"Detailed results: {detailed_path}")
        
        summary_path = os.path.join(output_dir, f'summac_summary_{config_name}.csv')
        summary_df = pd.DataFrame([stats])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        logger.info(f"Summary statistics: {summary_path}")
        
        return results_df, stats


def update_combined_summary(output_dir: str, stats: Dict[str, Any]):
    combined_path = os.path.join(output_dir, 'combined_summary.csv')
    
    new_row = {
        'config_name': stats['config_name'],
        'summac_mean': stats['summac_mean'],
        'summac_median': stats['summac_median'],
        'summac_std': stats['summac_std'],
        'summac_min': stats['summac_min'],
        'summac_max': stats['summac_max'],
        'method': stats['method']
    }
    
    if os.path.exists(combined_path):
        existing_df = pd.read_csv(combined_path)
        existing_df = existing_df[existing_df['config_name'] != 'full_revisions_2']
        new_df = pd.DataFrame([new_row])
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = pd.DataFrame([new_row])
    
    combined_df.to_csv(combined_path, index=False, encoding='utf-8')
    logger.info(f"Combined summary updated: {combined_path}")
    return combined_df


def main():
    ablation_file = str(root_dir / 'data' / 'ablations' / 'full_revisions_2' / 'ablation_full_revisions_2_final_500.json')
    output_dir = str(root_dir / 'Results' / 'Revision Depth' / 'summac_full_revisions_2_results')
    
    if not os.path.exists(ablation_file):
        logger.error(f"File not found: {ablation_file}")
        return
    
    evaluator = SimpleSummaCEvaluator(device='cuda:0', batch_size=4)
    
    try:
        results_df, stats = evaluator.evaluate_file(ablation_file, output_dir)
        update_combined_summary(output_dir, stats)
        
        print("\nEVALUATION SUMMARY - Full Revisions 2")
        print("=" * 60)
        print(f"SummaC Mean: {stats['summac_mean']:.4f}")
        print(f"Method: {stats['method']}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")


if __name__ == "__main__":
    main()
