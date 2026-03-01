#!/usr/bin/env python3
"""
SummaC Evaluation Script for full_revisions_2 dataset
Evaluates factual consistency between summaries and source documents
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

# Add SummaC to path - use local version
sys.path.append(str(Path(__file__).parent / "summac"))
try:
    from summac.model_summac import SummaCConv
    HAS_SUMMAC = True
except ImportError as e:
    print(f"⚠️ SummaC import failed: {e}")
    HAS_SUMMAC = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleSummaCEvaluator:
    def __init__(self, device='cuda:0', batch_size=4):
        """Initialize SummaC evaluator"""
        self.device = device
        self.batch_size = batch_size
        self.model = None
        
        # Constants matching ablation runner
        self.SEED = 42
        self.SAMPLE_SIZE = 500
        
        # Initialize SummaC if available
        if HAS_SUMMAC:
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
                logger.info("✅ SummaCConv initialized")
            except Exception as e:
                logger.error(f"❌ SummaCConv init failed: {e}")
                self.model = None
    
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
    
    def load_ablation_data(self, file_path):
        """Load ablation data from JSON file"""
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def calculate_scores(self, data: List[Dict], original_documents) -> pd.DataFrame:
        """Calculate SummaC scores with proper ID matching"""
        results = []
        
        # Extract data with proper ID matching
        summaries = []
        reference_summaries = []
        config_names = []
        example_ids = []
        matched_documents = []
        
        for item in data:
            example_id = item.get('example_id', -1)
            if example_id >= 0 and example_id < len(original_documents):
                summaries.append(item['final_summary'])
                # Try to get reference summary, fallback to empty string if not available
                reference_summaries.append(item.get('reference', ''))
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
                    # Fallback for this batch
                    batch_scores = self._fallback_score(batch_summaries, batch_documents)
                    scores.extend(batch_scores)
            
            method = 'SummaCConv_vitc'
            logger.info(f"✅ SummaCConv evaluation completed")
            
        else:
            logger.warning("⚠️ Using fallback consistency scoring")
            scores = self._fallback_score(summaries, matched_documents)
            method = 'fallback'
        
        # Create results dataframe
        for i, (example_id, score) in enumerate(zip(example_ids, scores)):
            results.append({
                'id': example_id,
                'summac_score': float(score),
                'method': method
            })
        
        return pd.DataFrame(results)
    
    def _fallback_score(self, summaries: List[str], documents: List[str]) -> List[float]:
        """Fallback consistency scoring using word overlap"""
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
        """Calculate summary statistics"""
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
        """Evaluate the full_revisions_2 ablation file"""
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
        detailed_path = os.path.join(output_dir, f'summac_detailed_{config_name}.csv')
        results_df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"✅ Detailed results: {detailed_path}")
        
        # Save summary statistics
        summary_path = os.path.join(output_dir, f'summac_summary_{config_name}.csv')
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
    logger.info(f"✅ Combined summary updated: {combined_path}")
    return combined_df

def main():
    """Main evaluation function"""
    
    # Define the specific file to evaluate
    ablation_file = 'data/ablations/full_revisions_2/ablation_full_revisions_2_final_500.json'
    
    print("🔬 SUMMAC EVALUATION FOR FULL_REVISIONS_2")
    print("=" * 60)
    print("📋 Format: id, summac_score, method")
    print("📋 Model: SummaCConv-vitc (factual consistency)")
    print("📋 Dataset: full_revisions_2")
    print()
    
    # Check if SummaC is available
    if HAS_SUMMAC:
        print("✅ SummaCConv available - using research-grade method")
    else:
        print("⚠️ SummaCConv not available - using fallback method")
    print()
    
    # Check if file exists
    if not os.path.exists(ablation_file):
        logger.error(f"❌ File not found: {ablation_file}")
        return
    
    print(f"📁 File to evaluate: {ablation_file}")
    print()
    
    # Initialize evaluator
    evaluator = SimpleSummaCEvaluator(device='cuda:0', batch_size=4)
    
    # Create output directory
    output_dir = 'summac_full_revisions_2_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate the file
    try:
        results_df, stats = evaluator.evaluate_file(ablation_file, output_dir)
        
        # Update combined summary
        combined_df = update_combined_summary(output_dir, stats)
        
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY - Full Revisions 2")
        print("=" * 60)
        print(f"\n🎯 Performance Metrics:")
        print(f"   SummaC Mean: {stats['summac_mean']:.4f}")
        print(f"   SummaC Median: {stats['summac_median']:.4f}")
        print(f"   SummaC Std: {stats['summac_std']:.4f}")
        print(f"   Range: [{stats['summac_min']:.4f}, {stats['summac_max']:.4f}]")
        print(f"   Method: {stats['method']}")
        
        print(f"\n📁 Results saved to:")
        print(f"   - {output_dir}/summac_detailed_full_revisions_2.csv")
        print(f"   - {output_dir}/summac_summary_full_revisions_2.csv")
        print(f"   - {output_dir}/combined_summary.csv")
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
