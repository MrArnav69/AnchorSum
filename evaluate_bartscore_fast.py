#!/usr/bin/env python3
"""
Fast BARTScore Evaluation - Optimized for speed
Clean format: id, bartscore_sum2doc, bartscore_doc2sum
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

# Import BARTScore
sys.path.append(os.path.join(os.path.dirname(__file__), 'BARTScore'))
from bart_score import BARTScorer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastBARTScoreEvaluator:
    def __init__(self, device='cuda:0', batch_size=4):
        """Initialize BARTScore evaluator with smaller batch for speed"""
        self.device = device
        self.batch_size = batch_size  # Smaller batch for memory
        
        # Constants matching ablation runner
        self.SEED = 42
        self.SAMPLE_SIZE = 500
        
        # Initialize BARTScore
        logger.info("Initializing BARTScore...")
        self.scorer = BARTScorer(
            device=device, 
            max_length=1024, 
            checkpoint='facebook/bart-large-cnn'
        )
        logger.info("BARTScore initialized successfully")
    
    def load_original_dataset(self):
        """Load original dataset with same logic as ablation runner"""
        logger.info("Loading original dataset...")
        dataset = load_dataset("Awesome075/multi_news_parquet", split="test")
        
        # Set random seeds for reproducibility
        import random
        random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        
        # Sample dataset with same logic as ablation runner
        sampled_dataset = dataset.shuffle(seed=self.SEED).select(range(self.SAMPLE_SIZE))
        
        logger.info(f"Loaded {len(sampled_dataset)} original documents")
        return sampled_dataset
    
    def load_ablation_data(self, file_path):
        """Load ablation data from JSON file"""
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def calculate_scores(self, data: List[Dict], original_documents) -> pd.DataFrame:
        """Calculate BARTScore for all samples with proper ID matching"""
        results = []
        
        # Extract data with proper ID matching
        summaries = []
        reference_summaries = []
        config_names = []
        example_ids = []
        
        for item in data:
            example_id = item.get('example_id', -1)
            if example_id >= 0 and example_id < len(original_documents):
                # Match summary ID with document ID
                original_doc = original_documents[example_id]
                summaries.append(item['final_summary'])
                reference_summaries.append(item['reference'])
                config_names.append(item.get('config_name', 'unknown'))
                example_ids.append(example_id)
            else:
                logger.warning(f"Skipping sample {example_id} - no matching document")
        
        logger.info(f"Evaluating {len(summaries)} matched samples (skipped {len(data) - len(summaries)} unmatched)")
        
        # Simple truncation
        max_length = 1024
        truncated_summaries = [s[:max_length*4] if len(s) > max_length*4 else s for s in summaries]
        documents = [doc['document'] for doc in original_documents]
        
        logger.info("Calculating BARTScore scores...")
        try:
            # Calculate both scores in one go
            sum2doc_scores = self.scorer.score(
                srcs=truncated_summaries,
                tgts=documents,
                batch_size=self.batch_size
            )
            
            doc2sum_scores = self.scorer.score(
                srcs=documents,
                tgts=truncated_summaries,
                batch_size=self.batch_size
            )
            
            logger.info(f"✅ BARTScore calculated successfully")
            
        except Exception as e:
            logger.error(f"Error calculating BARTScore: {e}")
            sum2doc_scores = [0.0] * len(summaries)
            doc2sum_scores = [0.0] * len(summaries)
        
        # Create simplified detailed results
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
        """Calculate summary statistics"""
        stats = {'config_name': config_name}
        
        # BARTScore metrics
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
        """Evaluate a single ablation file"""
        config_name = Path(file_path).stem.replace('ablation_', '').replace('_final_500', '')
        logger.info(f"Evaluating {config_name} configuration")
        
        # Load data
        data = self.load_ablation_data(file_path)
        original_documents = self.load_original_dataset()
        
        # Calculate scores
        results_df = self.calculate_scores(data, original_documents)
        
        # Calculate summary statistics
        stats = self.calculate_summary_stats(results_df, config_name)
        
        # Save results
        detailed_path = os.path.join(output_dir, f'bartscore_detailed_{config_name}.csv')
        summary_path = os.path.join(output_dir, f'bartscore_summary_{config_name}.csv')
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results (simplified format)
        results_df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"Detailed results saved to {detailed_path}")
        
        # Save summary statistics
        summary_df = pd.DataFrame([stats])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        logger.info(f"Summary statistics saved to {summary_path}")
        
        # Verify files were created
        if os.path.exists(detailed_path) and os.path.exists(summary_path):
            logger.info(f"✅ Files created successfully")
        else:
            logger.error(f"❌ Failed to create output files")
        
        return results_df, stats

def main():
    """Main evaluation function"""
    
    # Define ablation files to evaluate
    ablation_files = [
        'data/ablations/base/ablation_base_final_500.json',
        'data/ablations/no_nli/ablation_no_nli_final_500.json',
        'data/ablations/no_entity/ablation_no_entity_final_500.json',
        'data/ablations/full/ablation_full_final_500.json'
    ]
    
    print("🚀 FAST BARTScore EVALUATION")
    print("=" * 60)
    print("📋 Format: id, bartscore_sum2doc, bartscore_doc2sum")
    print("📁 Files to evaluate:")
    for file_path in ablation_files:
        print(f"   • {file_path}")
    print()
    print("⚡ Optimized: Smaller batch size, better error handling")
    print()
    
    # Initialize evaluator
    evaluator = FastBARTScoreEvaluator(device='cuda:0', batch_size=4)
    
    # Create output directory
    output_dir = 'bartscore_fast_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate all files
    all_results = []
    for file_path in ablation_files:
        if not os.path.exists(file_path):
            logger.error(f"❌ File not found: {file_path}")
            continue
        
        try:
            results_df, stats = evaluator.evaluate_file(file_path, output_dir)
            all_results.append(stats)
            logger.info(f"✅ Completed {file_path}")
        except Exception as e:
            logger.error(f"❌ Failed to evaluate {file_path}: {str(e)}")
    
    # Create combined summary
    if all_results:
        combined_summary = pd.DataFrame(all_results)
        combined_summary_path = os.path.join(output_dir, 'combined_summary.csv')
        combined_summary.to_csv(combined_summary_path, index=False, encoding='utf-8')
        logger.info(f"📊 Combined summary saved to {combined_summary_path}")
        
        print("\n📊 COMBINED SUMMARY:")
        print("-" * 40)
        for _, row in combined_summary.iterrows():
            config = row['config_name']
            print(f"\n📋 {config.upper()}:")
            print(f"   BARTScore sum2doc: {row['bartscore_sum2doc_mean']:.3f} (±{row['bartscore_sum2doc_std']:.3f})")
            print(f"   BARTScore doc2sum: {row['bartscore_doc2sum_mean']:.3f} (±{row['bartscore_doc2sum_std']:.3f})")
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print("🎉 Fast BARTScore evaluation completed!")

if __name__ == "__main__":
    main()
