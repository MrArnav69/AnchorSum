#!/usr/bin/env python3
"""
Evaluate ROUGE and BERTScore for full_revisions_2 results
Saves detailed and summary CSVs in rouge_bert_simple_results format
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

# ROUGE imports
from rouge_score import rouge_scorer

# BERTScore imports
try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FullRevisions2Evaluator:
    def __init__(self, device='cuda:0', batch_size=8):
        self.device = device
        self.batch_size = batch_size
        self.model_type = 'microsoft/deberta-xlarge-mnli'
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        logger.info(f"ROUGE scorer initialized")
        logger.info(f"BERTScore model: {self.model_type}")
    
    def load_ablation_data(self, file_path: str) -> List[Dict]:
        """Load ablation data from JSON file"""
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def calculate_rouge(self, hypothesis: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores for a single pair"""
        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougel': scores['rougeL'].fmeasure
        }
    
    def calculate_scores(self, data: List[Dict]) -> pd.DataFrame:
        """Calculate ROUGE and BERTScore for all samples"""
        results = []
        
        summaries = []
        references = []
        example_ids = []
        
        logger.info("Extracting summaries and references...")
        for item in data:
            example_id = item.get('example_id', -1)
            summary = item.get('final_summary', '')
            reference = item.get('reference', '')
            
            summaries.append(summary)
            references.append(reference)
            example_ids.append(example_id)
        
        # Calculate ROUGE scores
        logger.info("Calculating ROUGE scores...")
        rouge_results = []
        for summary, reference in tqdm(zip(summaries, references), total=len(summaries), desc="ROUGE"):
            rouge_scores = self.calculate_rouge(summary, reference)
            rouge_results.append(rouge_scores)
        
        # Calculate BERTScore
        bert_precision = [0.0] * len(summaries)
        bert_recall = [0.0] * len(summaries)
        bert_f1 = [0.0] * len(summaries)
        method = 'fallback'
        
        if HAS_BERTSCORE:
            logger.info(f"Calculating BERTScore with {self.model_type}...")
            try:
                P, R, F1 = bert_score(
                    summaries,
                    references,
                    model_type=self.model_type,
                    device=self.device,
                    batch_size=self.batch_size,
                    lang='en',
                    verbose=True
                )
                bert_precision = P.cpu().numpy().tolist()
                bert_recall = R.cpu().numpy().tolist()
                bert_f1 = F1.cpu().numpy().tolist()
                method = f'BERTScore-{self.model_type.split("/")[-1]}'
                logger.info("✅ BERTScore calculation completed")
            except Exception as e:
                logger.error(f"BERTScore calculation failed: {e}")
        else:
            logger.warning("⚠️ BERTScore not available - install with: pip install bert-score")
        
        # Combine results
        for i, example_id in enumerate(example_ids):
            results.append({
                'id': example_id,
                'rouge1': rouge_results[i]['rouge1'],
                'rouge2': rouge_results[i]['rouge2'],
                'rougel': rouge_results[i]['rougel'],
                'bertscore_precision': bert_precision[i],
                'bertscore_recall': bert_recall[i],
                'bertscore_f1': bert_f1[i],
                'method': method
            })
        
        return pd.DataFrame(results)
    
    def calculate_summary_stats(self, df: pd.DataFrame, config_name: str) -> Dict[str, Any]:
        """Calculate summary statistics"""
        stats = {'config_name': config_name}
        
        # ROUGE metrics
        for metric in ['rouge1', 'rouge2', 'rougel']:
            metric_data = df[metric]
            stats[f'{metric}_mean'] = float(metric_data.mean())
            stats[f'{metric}_median'] = float(metric_data.median())
            stats[f'{metric}_std'] = float(metric_data.std())
            stats[f'{metric}_min'] = float(metric_data.min())
            stats[f'{metric}_max'] = float(metric_data.max())
        
        # BERTScore metrics (precision, recall, f1)
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
        """Evaluate a single ablation file"""
        config_name = 'full_revisions_2'
        logger.info(f"🔬 Evaluating: {config_name}")
        
        # Load data
        data = self.load_ablation_data(file_path)
        
        # Calculate scores
        results_df = self.calculate_scores(data)
        
        # Calculate summary statistics
        stats = self.calculate_summary_stats(results_df, config_name)
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        detailed_path = os.path.join(output_dir, f'rouge_bert_detailed_{config_name}.csv')
        results_df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"✅ Detailed results: {detailed_path}")
        
        # Save summary statistics
        summary_path = os.path.join(output_dir, f'rouge_bert_summary_{config_name}.csv')
        summary_df = pd.DataFrame([stats])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        logger.info(f"✅ Summary statistics: {summary_path}")
        
        return results_df, stats


def update_combined_summary(output_dir: str, stats: Dict[str, Any]):
    """Update or create combined summary CSV"""
    combined_path = os.path.join(output_dir, 'combined_summary.csv')
    
    # Create new row for full_revisions_2
    new_row = {
        'config_name': stats['config_name'],
        'rouge1_mean': stats['rouge1_mean'],
        'rouge1_median': stats['rouge1_median'],
        'rouge1_std': stats['rouge1_std'],
        'rouge1_min': stats['rouge1_min'],
        'rouge1_max': stats['rouge1_max'],
        'rouge2_mean': stats['rouge2_mean'],
        'rouge2_median': stats['rouge2_median'],
        'rouge2_std': stats['rouge2_std'],
        'rouge2_min': stats['rouge2_min'],
        'rouge2_max': stats['rouge2_max'],
        'rougel_mean': stats['rougel_mean'],
        'rougel_median': stats['rougel_median'],
        'rougel_std': stats['rougel_std'],
        'rougel_min': stats['rougel_min'],
        'rougel_max': stats['rougel_max'],
        'bertscore_precision_mean': stats['bertscore_precision_mean'],
        'bertscore_precision_median': stats['bertscore_precision_median'],
        'bertscore_precision_std': stats['bertscore_precision_std'],
        'bertscore_precision_min': stats['bertscore_precision_min'],
        'bertscore_precision_max': stats['bertscore_precision_max'],
        'bertscore_recall_mean': stats['bertscore_recall_mean'],
        'bertscore_recall_median': stats['bertscore_recall_median'],
        'bertscore_recall_std': stats['bertscore_recall_std'],
        'bertscore_recall_min': stats['bertscore_recall_min'],
        'bertscore_recall_max': stats['bertscore_recall_max'],
        'bertscore_f1_mean': stats['bertscore_f1_mean'],
        'bertscore_f1_median': stats['bertscore_f1_median'],
        'bertscore_f1_std': stats['bertscore_f1_std'],
        'bertscore_f1_min': stats['bertscore_f1_min'],
        'bertscore_f1_max': stats['bertscore_f1_max'],
        'method': stats['method']
    }
    
    if os.path.exists(combined_path):
        # Load existing and append
        existing_df = pd.read_csv(combined_path)
        # Remove full_revisions_2 if it exists
        existing_df = existing_df[existing_df['config_name'] != 'full_revisions_2']
        new_df = pd.DataFrame([new_row])
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = pd.DataFrame([new_row])
    
    combined_df.to_csv(combined_path, index=False, encoding='utf-8')
    logger.info(f"✅ Combined summary updated: {combined_path}")
    return combined_df


def main():
    """Main evaluation function"""
    
    file_path = 'data/ablations/full_revisions_2/ablation_full_revisions_2_final_500.json'
    output_dir = 'rouge_bert_full_revisions_2_results'
    
    print("🔬 ROUGE & BERTScore EVALUATION - Full Revisions 2")
    print("=" * 60)
    print(f"📁 Input file: {file_path}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Check dependencies
    if not HAS_BERTSCORE:
        print("⚠️ BERTScore not available - install with: pip install bert-score")
    else:
        print("✅ BERTScore available")
    print()
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = FullRevisions2Evaluator(device='cuda:0', batch_size=4)
    
    # Run evaluation
    try:
        results_df, stats = evaluator.evaluate_file(file_path, output_dir)
        
        # Update combined summary
        combined_df = update_combined_summary(output_dir, stats)
        
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY - Full Revisions 2")
        print("=" * 60)
        print(f"\n🎯 Performance Metrics:")
        print(f"   ROUGE-1: {stats['rouge1_mean']:.4f} (±{stats['rouge1_std']:.4f})")
        print(f"   ROUGE-2: {stats['rouge2_mean']:.4f} (±{stats['rouge2_std']:.4f})")
        print(f"   ROUGE-L: {stats['rougel_mean']:.4f} (±{stats['rougel_std']:.4f})")
        print(f"   BERTScore F1: {stats['bertscore_f1_mean']:.4f} (±{stats['bertscore_f1_std']:.4f})")
        print(f"   BERTScore Precision: {stats['bertscore_precision_mean']:.4f} (±{stats['bertscore_precision_std']:.4f})")
        print(f"   BERTScore Recall: {stats['bertscore_recall_mean']:.4f} (±{stats['bertscore_recall_std']:.4f})")
        
        print(f"\n📁 Results saved to:")
        print(f"   - {output_dir}/rouge_bert_detailed_full_revisions_2.csv")
        print(f"   - {output_dir}/rouge_bert_summary_full_revisions_2.csv")
        print(f"   - {output_dir}/combined_summary.csv")
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
