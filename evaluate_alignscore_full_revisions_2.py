#!/usr/bin/env python3
"""
AlignScore Evaluation for full_revisions_2 results
Saves detailed and summary CSVs
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

# Add AlignScore to path
sys.path.append(str(Path(__file__).parent / "AlignScore"))
try:
    from alignscore import AlignScore
    HAS_ALIGNSCORE = True
except ImportError as e:
    print(f"⚠️ AlignScore import failed: {e}")
    HAS_ALIGNSCORE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AlignScoreFullRevisions2:
    def __init__(self, device='cuda:0', batch_size=16):
        self.device = device
        self.batch_size = batch_size
        self.scorer = None
        
        # Initialize AlignScore if available
        if HAS_ALIGNSCORE:
            try:
                ckpt_path = '/workspace/AnchorSum/AlignScore/AlignScore-large.ckpt'
                if os.path.exists(ckpt_path):
                    logger.info("Loading AlignScore-large model...")
                    self.scorer = AlignScore(
                        model='roberta-large',
                        batch_size=batch_size,
                        device=device,
                        ckpt_path=ckpt_path,
                        evaluation_mode='nli_sp'
                    )
                    logger.info("✅ AlignScore loaded")
                else:
                    logger.warning(f"⚠️ Checkpoint not found: {ckpt_path}")
            except Exception as e:
                logger.error(f"❌ AlignScore init failed: {e}")
    
    def load_original_dataset(self):
        """Load original dataset from local JSON file"""
        logger.info("Loading original dataset from local file...")
        
        dataset_path = 'data/multi_news_500_samples.json'
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found: {dataset_path}")
            logger.error("Run download_dataset.py first to save the dataset locally")
            return None
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        logger.info(f"Loaded {len(samples)} original documents from local file")
        return samples
    
    def load_ablation_data(self, file_path: str) -> List[Dict]:
        """Load ablation data from JSON file"""
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def calculate_scores(self, data: List[Dict], original_documents) -> pd.DataFrame:
        """Calculate AlignScore with proper ID matching"""
        results = []
        
        # Extract data with proper ID matching
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
        
        if self.scorer is not None:
            logger.info("Using AlignScore for evaluation...")
            scores = []
            
            for i in tqdm(range(0, len(summaries), self.batch_size), desc="AlignScore"):
                batch_summaries = summaries[i:i+self.batch_size]
                batch_documents = matched_documents[i:i+self.batch_size]
                
                try:
                    # Calculate AlignScore (context=document, claim=summary)
                    batch_scores = self.scorer.score(
                        contexts=batch_documents,
                        claims=batch_summaries
                    )
                    scores.extend(batch_scores)
                    
                except Exception as e:
                    logger.error(f"Batch {i} failed: {e}")
                    scores.extend([0.0] * len(batch_summaries))
            
            method = 'AlignScore-large'
            logger.info(f"✅ AlignScore evaluation completed")
            
        else:
            logger.warning("⚠️ Using fallback scoring")
            scores = [0.0] * len(summaries)
            method = 'fallback'
        
        # Create results dataframe
        for i, (example_id, score) in enumerate(zip(example_ids, scores)):
            results.append({
                'id': example_id,
                'alignscore': float(score),
                'method': method
            })
        
        return pd.DataFrame(results)
    
    def calculate_summary_stats(self, df: pd.DataFrame, config_name: str) -> Dict[str, Any]:
        """Calculate summary statistics"""
        stats = {'config_name': config_name}
        
        metric_data = df['alignscore']
        stats['alignscore_mean'] = float(metric_data.mean())
        stats['alignscore_median'] = float(metric_data.median())
        stats['alignscore_std'] = float(metric_data.std())
        stats['alignscore_min'] = float(metric_data.min())
        stats['alignscore_max'] = float(metric_data.max())
        stats['method'] = df['method'].iloc[0] if len(df) > 0 else 'unknown'
        
        return stats
    
    def evaluate_file(self, file_path: str, output_dir: str):
        """Evaluate a single ablation file"""
        config_name = 'full_revisions_2'
        logger.info(f"🔬 Evaluating: {config_name}")
        
        # Load data
        data = self.load_ablation_data(file_path)
        original_documents = self.load_original_dataset()
        
        # Calculate scores
        results_df = self.calculate_scores(data, original_documents)
        
        # Calculate summary statistics
        stats = self.calculate_summary_stats(results_df, config_name)
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        detailed_path = os.path.join(output_dir, f'alignscore_detailed_{config_name}.csv')
        results_df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"✅ Detailed results: {detailed_path}")
        
        # Save summary statistics
        summary_path = os.path.join(output_dir, f'alignscore_summary_{config_name}.csv')
        summary_df = pd.DataFrame([stats])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        logger.info(f"✅ Summary statistics: {summary_path}")
        
        return results_df, stats


def update_combined_summary(output_dir: str, stats: Dict[str, Any]):
    """Update or create combined summary CSV"""
    combined_path = os.path.join(output_dir, 'combined_summary.csv')
    
    # Create new row
    new_row = {
        'config_name': stats['config_name'],
        'alignscore_mean': stats['alignscore_mean'],
        'alignscore_median': stats['alignscore_median'],
        'alignscore_std': stats['alignscore_std'],
        'alignscore_min': stats['alignscore_min'],
        'alignscore_max': stats['alignscore_max'],
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
    logger.info(f"✅ Combined summary updated: {combined_path}")
    return combined_df


def main():
    """Main evaluation function"""
    
    file_path = 'data/ablations/full_revisions_2/ablation_full_revisions_2_final_500.json'
    output_dir = 'alignscore_full_revisions_2_results'
    
    print("🔬 ALIGNSCORE EVALUATION - Full Revisions 2")
    print("=" * 60)
    print(f"📁 Input file: {file_path}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Check if AlignScore is available
    if HAS_ALIGNSCORE:
        print("✅ AlignScore available")
    else:
        print("⚠️ AlignScore not available - check AlignScore/ directory")
    print()
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"❌ File not found: {file_path}")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = AlignScoreFullRevisions2(device='cuda:0', batch_size=16)
    
    # Run evaluation
    try:
        results_df, stats = evaluator.evaluate_file(file_path, output_dir)
        
        # Update combined summary
        combined_df = update_combined_summary(output_dir, stats)
        
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY - Full Revisions 2")
        print("=" * 60)
        print(f"\n🎯 Performance Metrics:")
        print(f"   AlignScore Mean: {stats['alignscore_mean']:.4f}")
        print(f"   AlignScore Median: {stats['alignscore_median']:.4f}")
        print(f"   AlignScore Std: {stats['alignscore_std']:.4f}")
        print(f"   Range: [{stats['alignscore_min']:.4f}, {stats['alignscore_max']:.4f}]")
        print(f"   Method: {stats['method']}")
        
        print(f"\n📁 Results saved to:")
        print(f"   - {output_dir}/alignscore_detailed_full_revisions_2.csv")
        print(f"   - {output_dir}/alignscore_summary_full_revisions_2.csv")
        print(f"   - {output_dir}/combined_summary.csv")
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
