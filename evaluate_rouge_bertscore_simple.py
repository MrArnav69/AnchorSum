#!/usr/bin/env python3
"""
Simplified ROUGE and BERTScore Evaluation Script
Clean format: id, rouge1, rouge2, rougel, bertscore
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

class SimpleROUGEBERTScoreEvaluator:
    def __init__(self, device='cuda:0', batch_size=16):
        """Initialize simplified ROUGE and BERTScore evaluator"""
        self.device = device
        self.batch_size = batch_size
        self.bert_model = "microsoft/deberta-xlarge-mnli"
        
        # Constants matching ablation runner
        self.SEED = 42
        self.SAMPLE_SIZE = 500
        
        # Initialize ROUGE scorer
        logger.info("Initializing ROUGE scorer...")
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        logger.info(f"BERTScore will use model: {self.bert_model}")
        logger.info("ROUGE and BERTScore evaluators initialized successfully")
    
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
        """Calculate ROUGE and BERTScore for all samples with proper ID matching"""
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
        
        logger.info("Calculating ROUGE scores...")
        rouge_scores = []
        for summary, reference in tqdm(zip(summaries, reference_summaries), 
                                   total=len(summaries), desc="ROUGE Evaluation"):
            try:
                scores = self.rouge_scorer.score(reference, summary)
                rouge_scores.append(scores)
            except Exception as e:
                logger.error(f"Error calculating ROUGE: {e}")
                rouge_scores.append({
                    'rouge1': type('Score', (), {'fmeasure': 0.0})(),
                    'rouge2': type('Score', (), {'fmeasure': 0.0})(),
                    'rougeL': type('Score', (), {'fmeasure': 0.0})()
                })
        
        logger.info("Calculating BERTScore...")
        try:
            P, R, F1 = bert_score(
                cands=summaries,
                refs=reference_summaries,
                model_type=self.bert_model,
                batch_size=self.batch_size,
                lang="en",
                rescale_with_baseline=True,
                device=self.device
            )
        except Exception as e:
            logger.error(f"Error calculating BERTScore: {e}")
            P, R, F1 = [0.0] * len(summaries), [0.0] * len(summaries), [0.0] * len(summaries)
        
        # Create simplified detailed results
        for i, (example_id, config_name, rouge_score, bert_f) in enumerate(
            zip(example_ids, config_names, rouge_scores, F1)
        ):
            results.append({
                'id': example_id,
                'rouge1': float(rouge_score['rouge1'].fmeasure),
                'rouge2': float(rouge_score['rouge2'].fmeasure),
                'rougel': float(rouge_score['rougeL'].fmeasure),
                'bertscore': float(bert_f)
            })
        
        return pd.DataFrame(results)
    
    def calculate_summary_stats(self, df: pd.DataFrame, config_name: str) -> Dict[str, Any]:
        """Calculate summary statistics for simplified format"""
        stats = {'config_name': config_name}
        
        # Simple metrics only
        metrics = ['rouge1', 'rouge2', 'rougel', 'bertscore']
        
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
        detailed_path = os.path.join(output_dir, f'rouge_bert_detailed_{config_name}.csv')
        summary_path = os.path.join(output_dir, f'rouge_bert_summary_{config_name}.csv')
        
        # Save detailed results (simplified format)
        results_df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"Detailed results saved to {detailed_path}")
        
        # Save summary statistics
        summary_df = pd.DataFrame([stats])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        logger.info(f"Summary statistics saved to {summary_path}")
        
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
    
    print("🔬 SIMPLIFIED ROUGE & BERTScore EVALUATION")
    print("=" * 60)
    print("📋 Format: id, rouge1, rouge2, rougel, bertscore")
    print("📁 Files to evaluate:")
    for file_path in ablation_files:
        print(f"   • {file_path}")
    print()
    
    # Initialize evaluator
    evaluator = SimpleROUGEBERTScoreEvaluator(device='cuda:0', batch_size=16)
    
    # Create output directory
    output_dir = 'rouge_bert_simple_results'
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
            print(f"   ROUGE-1: {row['rouge1_mean']:.3f} (±{row['rouge1_std']:.3f})")
            print(f"   ROUGE-2: {row['rouge2_mean']:.3f} (±{row['rouge2_std']:.3f})")
            print(f"   ROUGE-L: {row['rougel_mean']:.3f} (±{row['rougel_std']:.3f})")
            print(f"   BERTScore: {row['bertscore_mean']:.3f} (±{row['bertscore_std']:.3f})")
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print("🎉 Evaluation completed!")

if __name__ == "__main__":
    main()
