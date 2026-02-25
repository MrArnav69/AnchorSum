#!/usr/bin/env python3
"""
BARTScore Evaluation Script
Calculates both sum2doc and doc2sum BARTScore for ablation studies
Optimized for A40 GPU
"""

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

# Add BARTScore to path
sys.path.append(str(Path(__file__).parent / "BARTScore"))
from bart_score import BARTScorer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BARTScoreEvaluator:
    def __init__(self, device='cuda:0', batch_size=16):
        """
        Initialize BARTScore evaluator
        """
        self.device = device
        self.batch_size = batch_size
        
        # Constants matching ablation runner
        self.SEED = 42
        self.SAMPLE_SIZE = 500
        
        # Initialize BARTScore models
        logger.info("Loading BARTScore models...")
        
        # For sum2doc (summary -> document)
        self.scorer_sum2doc = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
        
        # For doc2sum (document -> summary) 
        self.scorer_doc2sum = BARTScorer(device=device, checkpoint='facebook/bart-large-cnn')
        
        logger.info("BARTScore models loaded successfully")
    
    def load_original_dataset(self):
        """Load the original dataset with same logic as ablation runner"""
        logger.info("Loading original dataset...")
        dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
        
        # Set random seeds for reproducibility (same as ablation runner)
        import random
        random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        
        # Sample dataset (same as ablation runner)
        sampled_dataset = dataset.shuffle(seed=self.SEED).select(range(self.SAMPLE_SIZE))
        logger.info(f"Loaded {len(sampled_dataset)} original documents")
        
        return sampled_dataset
    
    def load_data(self, file_path: str) -> List[Dict]:
        """Load ablation data from JSON file"""
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def calculate_scores(self, data: List[Dict], original_documents) -> pd.DataFrame:
        """Calculate BARTScore for all samples"""
        results = []
        
        summaries = [item['final_summary'] for item in data]
        reference_summaries = [item['reference'] for item in data]
        documents = [doc['document'] for doc in original_documents]
        config_names = [item.get('config_name', 'unknown') for item in data]
        example_ids = [item.get('example_id', i) for i, item in enumerate(data)]
        
        logger.info("Calculating BARTScore scores...")
        
        # Truncate summaries if they exceed model's max length
        max_length = 1024  # BART's typical max length
        truncated_summaries = []
        for summary in summaries:
            if len(summary) > max_length * 4:  # Rough estimate (4 chars per token)
                truncated_summaries.append(summary[:max_length * 4])
            else:
                truncated_summaries.append(summary)
        
        # Calculate sum2doc scores (summary -> document)
        logger.info("Calculating sum2doc scores...")
        sum2doc_scores = self.scorer.score(
            srcs=truncated_summaries,
            tgts=documents,
            batch_size=self.batch_size
        )
        
        # Calculate doc2sum scores (document -> summary)
        logger.info("Calculating doc2sum scores...")
        doc2sum_scores = self.scorer.score(
            srcs=documents,
            tgts=truncated_summaries,
            batch_size=self.batch_size
        )
        
        # Create detailed results
        for i, (summary, document, reference_summary, config_name, example_id, sum2doc, doc2sum) in enumerate(
            zip(summaries, documents, reference_summaries, config_names, example_ids, sum2doc_scores, doc2sum_scores)
        ):
            results.append({
                'example_id': example_id,
                'config_name': config_name,
                'summary': summary,
                'document': document,
                'reference_summary': reference_summary,
                'bartscore_sum2doc': float(sum2doc),
                'bartscore_doc2sum': float(doc2sum),
                'truncated': len(summary) > max_length * 4
            })
        
        return pd.DataFrame(results)
    
    def calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics"""
        stats = {}
        
        for metric in ['bartscore_sum2doc', 'bartscore_doc2sum', 'bartscore_avg']:
            metric_data = df[metric]
            stats[f'{metric}_mean'] = float(metric_data.mean())
            stats[f'{metric}_median'] = float(metric_data.median())
            stats[f'{metric}_std'] = float(metric_data.std())
            stats[f'{metric}_min'] = float(metric_data.min())
            stats[f'{metric}_max'] = float(metric_data.max())
            stats[f'{metric}_q25'] = float(metric_data.quantile(0.25))
            stats[f'{metric}_q75'] = float(metric_data.quantile(0.75))
        
        return stats
    
    def save_results(self, df: pd.DataFrame, stats: Dict[str, Any], output_dir: str, config_name: str):
        """Save detailed and summary results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        detailed_path = os.path.join(output_dir, f'bartscore_detailed_{config_name}.csv')
        df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"Detailed results saved to {detailed_path}")
        
        # Save summary statistics
        summary_df = pd.DataFrame([stats])
        summary_path = os.path.join(output_dir, f'bartscore_summary_{config_name}.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        logger.info(f"Summary statistics saved to {summary_path}")
    
    def evaluate_file(self, file_path: str, output_dir: str):
        """Evaluate a single ablation file"""
        config_name = os.path.basename(os.path.dirname(file_path))
        logger.info(f"Evaluating {config_name} configuration")
        
        # Load ablation data
        data = self.load_data(file_path)
        
        # Load original documents
        original_documents = self.load_original_dataset()
        
        # Calculate scores
        results_df = self.calculate_scores(data, original_documents)
        
        # Calculate summary statistics
        stats = self.calculate_summary_stats(results_df)
        
        # Save results
        self.save_results(results_df, stats, output_dir, config_name)
        
        return results_df, stats

def main():
    # Configuration
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_size = 16  # Optimized for A40 GPU
    
    # File paths
    base_dir = '/workspace/AnchorSum/data/ablations'
    output_dir = '/workspace/AnchorSum/evaluation_results/bartscore'
    
    # Ablation files to evaluate
    ablation_files = [
        'base/ablation_base_final_500.json',
        'no_nli/ablation_no_nli_final_500.json', 
        'no_entity/ablation_no_entity_final_500.json',
        'full/ablation_full_final_500.json'  # This file doesn't exist yet, but will be created
    ]
    
    # Initialize evaluator
    evaluator = BARTScoreEvaluator(device=device, batch_size=batch_size)
    
    # Evaluate each ablation file
    all_results = {}
    all_stats = []
    
    for file_path in ablation_files:
        full_path = os.path.join(base_dir, file_path)
        
        if os.path.exists(full_path):
            try:
                results_df, stats = evaluator.evaluate_file(full_path, output_dir)
                config_name = os.path.basename(os.path.dirname(full_path))
                all_results[config_name] = results_df
                stats['config_name'] = config_name
                all_stats.append(stats)
                
            except Exception as e:
                logger.error(f"Error evaluating {file_path}: {e}")
                continue
        else:
            logger.warning(f"File not found: {full_path}")
    
    # Save combined summary
    if all_stats:
        combined_summary_df = pd.DataFrame(all_stats)
        combined_summary_path = os.path.join(output_dir, 'bartscore_combined_summary.csv')
        combined_summary_df.to_csv(combined_summary_path, index=False, encoding='utf-8')
        logger.info(f"Combined summary saved to {combined_summary_path}")
    
    logger.info("BARTScore evaluation completed!")

if __name__ == "__main__":
    main()
