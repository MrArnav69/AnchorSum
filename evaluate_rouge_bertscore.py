#!/usr/bin/env python3
"""
ROUGE and BERTScore Evaluation Script
Calculates ROUGE-1, ROUGE-2, ROUGE-L and BERTScore (deberta-xlarge-mnli) for ablation studies
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

# Import evaluation libraries
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ROUGEAndBERTScoreEvaluator:
    def __init__(self, device='cuda:0', batch_size=32):
        """
        Initialize ROUGE and BERTScore evaluator
        """
        self.device = device
        self.batch_size = batch_size
        
        # Constants matching ablation runner
        self.SEED = 42
        self.SAMPLE_SIZE = 500
        
        # Initialize ROUGE scorer
        logger.info("Initializing ROUGE scorer...")
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # BERTScore model
        self.bert_model = "microsoft/deberta-xlarge-mnli"
        logger.info(f"BERTScore will use model: {self.bert_model}")
        
        logger.info("ROUGE and BERTScore evaluators initialized successfully")
    
    def load_original_dataset(self):
        """Load original dataset with same logic as ablation runner"""
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
        """Calculate ROUGE and BERTScore for all samples"""
        results = []
        
        summaries = [item['final_summary'] for item in data]
        reference_summaries = [item['reference'] for item in data]
        documents = [doc['document'] for doc in original_documents]
        config_names = [item.get('config_name', 'unknown') for item in data]
        example_ids = [item.get('example_id', i) for i, item in enumerate(data)]
        
        logger.info("Calculating ROUGE scores...")
        rouge_scores = []
        
        # Calculate ROUGE scores (summary vs reference summary)
        for i, (summary, reference_summary) in enumerate(tqdm(zip(summaries, reference_summaries), desc="ROUGE Evaluation", total=len(summaries))):
            try:
                scores = self.rouge_scorer.score(reference_summary, summary)
                rouge_scores.append({
                    'rouge1_precision': scores['rouge1'].precision,
                    'rouge1_recall': scores['rouge1'].recall,
                    'rouge1_fmeasure': scores['rouge1'].fmeasure,
                    'rouge2_precision': scores['rouge2'].precision,
                    'rouge2_recall': scores['rouge2'].recall,
                    'rouge2_fmeasure': scores['rouge2'].fmeasure,
                    'rougeL_precision': scores['rougeL'].precision,
                    'rougeL_recall': scores['rougeL'].recall,
                    'rougeL_fmeasure': scores['rougeL'].fmeasure
                })
            except Exception as e:
                logger.error(f"Error calculating ROUGE for sample {i}: {e}")
                rouge_scores.append({
                    'rouge1_precision': 0.0, 'rouge1_recall': 0.0, 'rouge1_fmeasure': 0.0,
                    'rouge2_precision': 0.0, 'rouge2_recall': 0.0, 'rouge2_fmeasure': 0.0,
                    'rougeL_precision': 0.0, 'rougeL_recall': 0.0, 'rougeL_fmeasure': 0.0
                })
        
        logger.info("Calculating BERTScore...")
        
        # Calculate BERTScore in batches (summary vs reference summary)
        ):
            results.append({
                'id': example_id,
                'rouge1': rouge_score['rouge1'].fmeasure,
                'rouge2': rouge_score['rouge2'].fmeasure,
                'rougel': rouge_score['rougeL'].fmeasure,
                'bertscore': float(bert_f)
            })
        
        return pd.DataFrame(results)
    
    def calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics"""
        stats = {}
        
        # All metrics to calculate statistics for
        metrics = [
            'rouge1_precision', 'rouge1_recall', 'rouge1_fmeasure',
            'rouge2_precision', 'rouge2_recall', 'rouge2_fmeasure',
            'rougeL_precision', 'rougeL_recall', 'rougeL_fmeasure',
            'bertscore_precision', 'bertscore_recall', 'bertscore_f1'
        ]
        
        for metric in metrics:
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
        detailed_path = os.path.join(output_dir, f'rouge_bertscore_detailed_{config_name}.csv')
        df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"Detailed results saved to {detailed_path}")
        
        # Save summary statistics
        summary_df = pd.DataFrame([stats])
        summary_path = os.path.join(output_dir, f'rouge_bertscore_summary_{config_name}.csv')
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
    batch_size = 32  # Optimized for A40 GPU
    
    # File paths
    base_dir = '/workspace/AnchorSum/data/ablations'
    output_dir = '/workspace/AnchorSum/evaluation_results/rouge_bertscore'
    
    # Ablation files to evaluate
    ablation_files = [
        'base/ablation_base_final_500.json',
        'no_nli/ablation_no_nli_final_500.json', 
        'no_entity/ablation_no_entity_final_500.json',
        'full/ablation_full_final_500.json'  # This file doesn't exist yet, but will be created
    ]
    
    # Initialize evaluator
    evaluator = ROUGEAndBERTScoreEvaluator(device=device, batch_size=batch_size)
    
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
        combined_summary_path = os.path.join(output_dir, 'rouge_bertscore_combined_summary.csv')
        combined_summary_df.to_csv(combined_summary_path, index=False, encoding='utf-8')
        logger.info(f"Combined summary saved to {combined_summary_path}")
    
    logger.info("ROUGE and BERTScore evaluation completed!")

if __name__ == "__main__":
    main()
