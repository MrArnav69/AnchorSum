#!/usr/bin/env python3
"""
AlignScore Evaluation Script
Calculates AlignScore for ablation studies using the best model (AlignScore-large)
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

# Add AlignScore to path
sys.path.append(str(Path(__file__).parent / "AlignScore"))
from alignscore import AlignScore

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlignScoreEvaluator:
    def __init__(self, device='cuda:0', batch_size=32):
        """
        Initialize AlignScore evaluator with the best model (AlignScore-large)
        """
        self.device = device
        self.batch_size = batch_size
        
        # Constants matching ablation runner
        self.SEED = 42
        self.SAMPLE_SIZE = 500
        
        # Path to AlignScore-large checkpoint
        ckpt_path = '/workspace/AnchorSum/AlignScore/AlignScore-large.ckpt'
        
        # Initialize AlignScore with large model
        logger.info("Loading AlignScore-large model...")
        self.scorer = AlignScore(
            model='roberta-large',
            batch_size=batch_size,
            device=device,
            ckpt_path=ckpt_path,
            evaluation_mode='nli_sp'  # Default and best setting
        )
        logger.info("AlignScore-large model loaded successfully")
    
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
        """Calculate AlignScore for all samples"""
        results = []
        
        summaries = [item['final_summary'] for item in data]
        reference_summaries = [item['reference'] for item in data]
        documents = [doc['document'] for doc in original_documents]
        config_names = [item.get('config_name', 'unknown') for item in data]
        example_ids = [item.get('example_id', i) for i, item in enumerate(data)]
        
        logger.info("Calculating AlignScore...")
        
        # Process in batches to avoid memory issues
        align_scores = []
        
        for i in tqdm(range(0, len(summaries), self.batch_size), desc="AlignScore Evaluation"):
            batch_summaries = summaries[i:i+self.batch_size]
            batch_documents = documents[i:i+self.batch_size]
            
            try:
                # Calculate AlignScore (summary vs original document)
                scores = self.scorer.score(
                    contexts=batch_documents,
                    claims=batch_summaries
                )
                align_scores.extend(scores)
                
            except Exception as e:
                logger.error(f"Error in AlignScore batch {i}: {e}")
                align_scores.extend([0.0] * len(batch_summaries))
        
        # Create detailed results
        for i, (summary, document, reference_summary, config_name, example_id, score) in enumerate(
            zip(summaries, documents, reference_summaries, config_names, example_ids, align_scores)
        ):
            results.append({
                'example_id': example_id,
                'config_name': config_name,
                'summary': summary,
                'document': document,
                'reference_summary': reference_summary,
                'alignscore': float(score)
            })
        
        return pd.DataFrame(results)
    
    def calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics"""
        score_data = df['alignscore']
        
        stats = {
            'alignscore_mean': float(score_data.mean()),
            'alignscore_median': float(score_data.median()),
            'alignscore_std': float(score_data.std()),
            'alignscore_min': float(score_data.min()),
            'alignscore_max': float(score_data.max()),
            'alignscore_q25': float(score_data.quantile(0.25)),
            'alignscore_q75': float(score_data.quantile(0.75))
        }
        
        return stats
    
    def save_results(self, df: pd.DataFrame, stats: Dict[str, Any], output_dir: str, config_name: str):
        """Save detailed and summary results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        detailed_path = os.path.join(output_dir, f'alignscore_detailed_{config_name}.csv')
        df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"Detailed results saved to {detailed_path}")
        
        # Save summary statistics
        summary_df = pd.DataFrame([stats])
        summary_path = os.path.join(output_dir, f'alignscore_summary_{config_name}.csv')
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
    output_dir = '/workspace/AnchorSum/evaluation_results/alignscore'
    
    # Ablation files to evaluate
    ablation_files = [
        'base/ablation_base_final_500.json',
        'no_nli/ablation_no_nli_final_500.json', 
        'no_entity/ablation_no_entity_final_500.json',
        'full/ablation_full_final_500.json'  # This file doesn't exist yet, but will be created
    ]
    
    # Initialize evaluator
    evaluator = AlignScoreEvaluator(device=device, batch_size=batch_size)
    
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
        combined_summary_path = os.path.join(output_dir, 'alignscore_combined_summary.csv')
        combined_summary_df.to_csv(combined_summary_path, index=False, encoding='utf-8')
        logger.info(f"Combined summary saved to {combined_summary_path}")
    
    logger.info("AlignScore evaluation completed!")

if __name__ == "__main__":
    main()
