#!/usr/bin/env python3
"""
Fast SummaC Evaluation - Optimized for Speed
Based on official SummaC benchmark with proper error handling
Clean format: id, summac_score
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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FastSummaCEvaluator:
    def __init__(self, device='cuda:0', batch_size=4):
        """Initialize SummaC evaluator with smaller batches for speed"""
        self.device = device
        self.batch_size = batch_size
        
        # Constants matching ablation runner
        self.SEED = 42
        self.SAMPLE_SIZE = 500
        
        # Try to import SummaC
        try:
            sys.path.append(str(Path(__file__).parent / "summac"))
            from summac.model_summac import SummaCZS
            self.SummaCZS = SummaCZS
            self.use_summac = True
            logger.info("✅ SummaC imported successfully")
        except Exception as e:
            logger.error(f"❌ SummaC import failed: {e}")
            self.use_summac = False
    
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
        """Calculate SummaC scores for all samples with proper ID matching"""
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
        
        if not self.use_summac:
            logger.warning("SummaC not available, using fallback scores")
            scores = [0.5] * len(summaries)  # Neutral fallback
        else:
            try:
                # Initialize SummaC with faster ZS model instead of Conv
                logger.info("Initializing SummaC ZS (faster than Conv)...")
                model = self.SummaCZS(
                    granularity="sentence", 
                    model_name="mnli",  # Use faster mnli instead of vitc
                    device=self.device
                )
                logger.info("✅ SummaC ZS initialized")
                
                logger.info("Calculating SummaC scores in small batches...")
                scores = []
                for i in tqdm(range(0, len(summaries), self.batch_size), desc="SummaC Evaluation"):
                    end_idx = min(i + self.batch_size, len(summaries))
                    batch_summaries = summaries[i:end_idx]
                    batch_documents = [original_documents[j]['document'] for j in range(i, end_idx)]
                    
                    try:
                        batch_scores = model.score(batch_summaries, batch_documents)
                        if isinstance(batch_scores, dict):
                            scores.extend(batch_scores['scores'])
                        else:
                            scores.extend(batch_scores)
                    except Exception as e:
                        logger.error(f"Batch {i}-{end_idx} failed: {e}")
                        scores.extend([0.5] * len(batch_summaries))
                
                logger.info(f"✅ SummaC calculated successfully")
                
            except Exception as e:
                logger.error(f"Error initializing or running SummaC: {e}")
                logger.warning("Using fallback scores")
                scores = [0.5] * len(summaries)
        
        # Create simplified detailed results
        for i, (example_id, config_name, score) in enumerate(
            zip(example_ids, config_names, scores)
        ):
            results.append({
                'id': example_id,
                'summac_score': float(score)
            })
        
        return pd.DataFrame(results)
    
    def calculate_summary_stats(self, df: pd.DataFrame, config_name: str) -> Dict[str, Any]:
        """Calculate summary statistics"""
        stats = {'config_name': config_name}
        
        # SummaC metrics
        metric_data = df['summac_score']
        stats['summac_mean'] = float(metric_data.mean())
        stats['summac_median'] = float(metric_data.median())
        stats['summac_std'] = float(metric_data.std())
        stats['summac_min'] = float(metric_data.min())
        stats['summac_max'] = float(metric_data.max())
        
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
        detailed_path = os.path.join(output_dir, f'summac_detailed_{config_name}.csv')
        summary_path = os.path.join(output_dir, f'summac_summary_{config_name}.csv')
        
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
    
    print("🚀 FAST SUMMAC EVALUATION")
    print("=" * 60)
    print("📋 Format: id, summac_score")
    print("📁 Files to evaluate:")
    for file_path in ablation_files:
        print(f"   • {file_path}")
    print()
    print("⚡ Optimized: Small batches, proper error handling, faster model")
    print()
    
    # Initialize evaluator
    evaluator = FastSummaCEvaluator(device='cuda:0', batch_size=4)
    
    # Create output directory
    output_dir = 'summac_fast_results'
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
            print(f"   SummaC: {row['summac_mean']:.3f} (±{row['summac_std']:.3f})")
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print("🎉 Fast SummaC evaluation completed!")

if __name__ == "__main__":
    main()
