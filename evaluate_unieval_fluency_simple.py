#!/usr/bin/env python3
"""
Simplified UniEval Fluency Evaluation Script
Clean format: id, fluency
Calculates fluency scores for summaries
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

# Add UniEval to path
unieval_path = str(Path(__file__).parent / "UniEval")
if unieval_path not in sys.path:
    sys.path.insert(0, unieval_path)

try:
    import UniEval.utils as unieval_utils
    from UniEval.metric.evaluator import get_evaluator
    convert_to_json = unieval_utils.convert_to_json
    HAS_UNIEVAL = True
except ImportError as e:
    print(f"⚠️ UniEval import failed: {e}")
    HAS_UNIEVAL = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleUniEvalFluencyEvaluator:
    def __init__(self, device='cuda:0', batch_size=16):
        """Initialize UniEval Fluency evaluator"""
        self.device = device
        self.batch_size = batch_size
        self.evaluator = None
        
        # Constants matching ablation runner
        self.SEED = 42
        self.SAMPLE_SIZE = 500
        
        # Initialize UniEval if available
        if HAS_UNIEVAL:
            try:
                logger.info("Loading UniEval model for fluency evaluation...")
                self.evaluator = get_evaluator('summarization')
                logger.info("✅ UniEval loaded")
            except Exception as e:
                logger.error(f"❌ UniEval init failed: {e}")
    
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
        """Calculate fluency scores with proper ID matching"""
        results = []
        
        # Extract data with proper ID matching
        summaries = []
        config_names = []
        example_ids = []
        matched_documents = []
        matched_references = []
        
        for item in data:
            example_id = item.get('example_id', -1)
            if example_id >= 0 and example_id < len(original_documents):
                summaries.append(item['final_summary'])
                config_names.append(item.get('config_name', 'unknown'))
                example_ids.append(example_id)
                matched_documents.append(original_documents[example_id]['document'])
                matched_references.append(original_documents[example_id]['summary'])
            else:
                logger.warning(f"Skipping sample {example_id} - no matching document")
        
        logger.info(f"Evaluating {len(summaries)} matched samples")
        
        # Truncate summaries if they exceed model limits
        max_length = 1024
        truncated_summaries = []
        for summary in summaries:
            if len(summary) > max_length * 4:
                truncated_summaries.append(summary[:max_length * 4])
            else:
                truncated_summaries.append(summary)
        
        if self.evaluator is not None:
            logger.info("Using UniEval for fluency evaluation...")
            fluency_scores = []
            
            for i in tqdm(range(0, len(truncated_summaries), self.batch_size), desc="UniEval Fluency"):
                batch_summaries = truncated_summaries[i:i+self.batch_size]
                batch_documents = matched_documents[i:i+self.batch_size]
                batch_references = matched_references[i:i+self.batch_size]
                
                try:
                    # Prepare data for UniEval
                    batch_data = convert_to_json(
                        output_list=batch_summaries,
                        src_list=batch_documents,
                        ref_list=batch_references
                    )
                    
                    # Get fluency scores only
                    eval_scores = self.evaluator.evaluate(
                        batch_data,
                        dims=['fluency'],
                        overall=False,
                        print_result=False
                    )
                    
                    # Extract fluency scores
                    batch_fluency = [score['fluency'] for score in eval_scores]
                    fluency_scores.extend(batch_fluency)
                    
                except Exception as e:
                    logger.error(f"Batch {i} failed: {e}")
                    fluency_scores.extend([0.0] * len(batch_summaries))
            
            method = 'UniEval-fluency'
            logger.info(f"✅ UniEval fluency evaluation completed")
            
        else:
            logger.warning("⚠️ Using fallback scoring")
            fluency_scores = [0.0] * len(summaries)
            method = 'fallback'
        
        # Create results dataframe
        for i, (example_id, score) in enumerate(zip(example_ids, fluency_scores)):
            results.append({
                'id': example_id,
                'fluency': float(score),
                'method': method
            })
        
        return pd.DataFrame(results)
    
    def calculate_summary_stats(self, df: pd.DataFrame, config_name: str) -> Dict[str, Any]:
        """Calculate summary statistics"""
        stats = {'config_name': config_name}
        
        metric_data = df['fluency']
        stats['fluency_mean'] = float(metric_data.mean())
        stats['fluency_median'] = float(metric_data.median())
        stats['fluency_std'] = float(metric_data.std())
        stats['fluency_min'] = float(metric_data.min())
        stats['fluency_max'] = float(metric_data.max())
        stats['method'] = df['method'].iloc[0] if len(df) > 0 else 'unknown'
        
        return stats
    
    def evaluate_file(self, file_path: str, output_dir: str):
        """Evaluate a single ablation file"""
        config_name = Path(file_path).stem.replace('ablation_', '').replace('_final_500', '')
        logger.info(f"🔬 Evaluating: {config_name}")
        
        # Load data
        data = self.load_ablation_data(file_path)
        original_documents = self.load_original_dataset()
        
        # Calculate scores
        results_df = self.calculate_scores(data, original_documents)
        
        # Calculate summary statistics
        stats = self.calculate_summary_stats(results_df, config_name)
        
        # Save results
        detailed_path = os.path.join(output_dir, f'unieval_fluency_detailed_{config_name}.csv')
        summary_path = os.path.join(output_dir, f'unieval_fluency_summary_{config_name}.csv')
        
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        results_df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"✅ Detailed results: {detailed_path}")
        
        # Save summary statistics
        summary_df = pd.DataFrame([stats])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        logger.info(f"✅ Summary statistics: {summary_path}")
        
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
    
    print("🔬 UNIEVAL FLUENCY EVALUATION")
    print("=" * 60)
    print("📋 Format: id, fluency, method")
    print("📋 Model: UniEval (fluency dimension)")
    print()
    
    # Check if UniEval is available
    if HAS_UNIEVAL:
        print("✅ UniEval available")
    else:
        print("⚠️ UniEval not available - check UniEval/ directory")
    print()
    
    print("📁 Files to evaluate:")
    for file_path in ablation_files:
        print(f"   • {file_path}")
    print()
    
    # Initialize evaluator
    evaluator = SimpleUniEvalFluencyEvaluator(device='cuda:0', batch_size=16)
    
    # Create output directory
    output_dir = 'unieval_fluency_results'
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
        
        print("\n📊 UNIEVAL FLUENCY SUMMARY:")
        print("-" * 40)
        for _, row in combined_summary.iterrows():
            config = row['config_name']
            method = row.get('method', 'unknown')
            print(f"\n📋 {config.upper()} (Method: {method}):")
            print(f"   Fluency Mean: {row['fluency_mean']:.3f}")
            print(f"   Fluency Median: {row['fluency_median']:.3f}")
            print(f"   Fluency Std: {row['fluency_std']:.3f}")
            print(f"   Range: [{row['fluency_min']:.3f}, {row['fluency_max']:.3f}]")
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print("🎉 UniEval Fluency evaluation completed!")

if __name__ == "__main__":
    main()
