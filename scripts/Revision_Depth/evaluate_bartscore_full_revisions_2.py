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

sys.path.append(os.path.join(os.path.dirname(__file__), 'BARTScore'))
try:
    from bart_score import BARTScorer
    has_bartscore = True
except ImportError:
    print("BARTScore not found. Ensure 'BARTScore' folder exists.")
    has_bartscore = False
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

root_dir = Path(__file__).resolve().parent.parent.parent


class BARTScoreFullRevisions2:
    def __init__(self, device='cuda:0', batch_size=8):
        self.device = device
        self.batch_size = batch_size
        self.scorer = None
        
        if has_bartscore:
            logger.info("Initializing BARTScore...")
            self.scorer = BARTScorer(
                device=device, 
                max_length=1024, 
                checkpoint='facebook/bart-large-cnn'
            )
            logger.info("BARTScore initialized")
    
    def load_ablation_data(self, file_path: str) -> List[Dict]:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def calculate_scores(self, data: List[Dict]) -> pd.DataFrame:
        results = []
        
        summaries = []
        documents = []
        example_ids = []
        
        logger.info("Extracting summaries and documents...")
        for item in data:
            example_id = item.get('example_id', -1)
            summary = item.get('final_summary', '')
            document = item.get('document', '')
            
            summaries.append(summary)
            documents.append(document)
            example_ids.append(example_id)
        
        logger.info(f"Evaluating {len(summaries)} samples")
        
        max_length = 1024
        truncated_summaries = []
        for summary in summaries:
            if len(summary) > max_length * 4:
                truncated_summaries.append(summary[:max_length * 4])
            else:
                truncated_summaries.append(summary)
        
        truncated_documents = []
        for doc in documents:
            if len(doc) > max_length * 4:
                truncated_documents.append(doc[:max_length * 4])
            else:
                truncated_documents.append(doc)
        
        logger.info("Calculating BARTScore scores...")
        try:
            sum2doc_scores = []
            doc2sum_scores = []
            
            for i in tqdm(range(0, len(summaries), self.batch_size), desc="BARTScore"):
                end_idx = min(i + self.batch_size, len(summaries))
                batch_summaries = truncated_summaries[i:end_idx]
                batch_documents = truncated_documents[i:end_idx]
                
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
        
        for i, (example_id, sum2doc, doc2sum) in enumerate(
            zip(example_ids, sum2doc_scores, doc2sum_scores)
        ):
            results.append({
                'id': example_id,
                'bartscore_sum2doc': float(sum2doc),
                'bartscore_doc2sum': float(doc2sum)
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
        config_name = 'full_revisions_2'
        logger.info(f"Evaluating: {config_name}")
        
        data = self.load_ablation_data(file_path)
        
        results_df = self.calculate_scores(data)
        
        stats = self.calculate_summary_stats(results_df, config_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
        detailed_path = os.path.join(output_dir, f'bartscore_detailed_{config_name}.csv')
        results_df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"Detailed results: {detailed_path}")
        
        summary_path = os.path.join(output_dir, f'bartscore_summary_{config_name}.csv')
        summary_df = pd.DataFrame([stats])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        logger.info(f"Summary statistics: {summary_path}")
        
        return results_df, stats


def update_combined_summary(output_dir: str, stats: Dict[str, Any]):
    combined_path = os.path.join(output_dir, 'combined_summary.csv')
    
    new_row = {
        'config_name': stats['config_name'],
        'bartscore_sum2doc_mean': stats['bartscore_sum2doc_mean'],
        'bartscore_sum2doc_median': stats['bartscore_sum2doc_median'],
        'bartscore_sum2doc_std': stats['bartscore_sum2doc_std'],
        'bartscore_sum2doc_min': stats['bartscore_sum2doc_min'],
        'bartscore_sum2doc_max': stats['bartscore_sum2doc_max'],
        'bartscore_doc2sum_mean': stats['bartscore_doc2sum_mean'],
        'bartscore_doc2sum_median': stats['bartscore_doc2sum_median'],
        'bartscore_doc2sum_std': stats['bartscore_doc2sum_std'],
        'bartscore_doc2sum_min': stats['bartscore_doc2sum_min'],
        'bartscore_doc2sum_max': stats['bartscore_doc2sum_max']
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
    file_path = str(root_dir / 'data' / 'ablations' / 'full_revisions_2' / 'ablation_full_revisions_2_final_500.json')
    output_dir = str(root_dir / 'Results' / 'Revision Depth' / 'bartscore_full_revisions_2_results')
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    
    evaluator = BARTScoreFullRevisions2(device='cuda:0', batch_size=8)
    
    try:
        results_df, stats = evaluator.evaluate_file(file_path, output_dir)
        update_combined_summary(output_dir, stats)
        
        print("\nEVALUATION SUMMARY - Full Revisions 2")
        print("=" * 60)
        print(f"BARTScore sum2doc: {stats['bartscore_sum2doc_mean']:.4f}")
        print(f"BARTScore doc2sum: {stats['bartscore_doc2sum_mean']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
