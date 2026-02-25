#!/usr/bin/env python3
"""
UniEval Fluency Evaluation Script
Calculates only fluency scores for ablation studies
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

# Add UniEval to path and use absolute imports
unieval_path = str(Path(__file__).parent / "UniEval")
if unieval_path not in sys.path:
    sys.path.insert(0, unieval_path)

# Import UniEval modules with absolute paths
import UniEval.utils as unieval_utils
from UniEval.metric.evaluator import get_evaluator

# Use the convert_to_json function from UniEval utils
convert_to_json = unieval_utils.convert_to_json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UniEvalFluencyEvaluator:
    def __init__(self, device='cuda:0', batch_size=16):
        """
        Initialize UniEval Fluency evaluator
        """
        self.device = device
        self.batch_size = batch_size
        
        # Constants matching ablation runner
        self.SEED = 42
        self.SAMPLE_SIZE = 500
        
        # Initialize UniEval evaluator for summarization task
        logger.info("Loading UniEval model for fluency evaluation...")
        self.evaluator = get_evaluator('summarization')
        logger.info("UniEval model loaded successfully")
    
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
    
    def calculate_fluency_scores(self, data: List[Dict], original_documents) -> pd.DataFrame:
        """Calculate fluency scores for all samples"""
        results = []
        
        summaries = [item['final_summary'] for item in data]
        reference_summaries = [item['reference'] for item in data]
        documents = [doc['document'] for doc in original_documents]
        config_names = [item.get('config_name', 'unknown') for item in data]
        example_ids = [item.get('example_id', i) for i, item in enumerate(data)]
        
        logger.info("Calculating fluency scores...")
        
        # Truncate summaries if they exceed typical model limits
        max_length = 1024  # Typical max length for most models
        truncated_summaries = []
        for summary in summaries:
            if len(summary) > max_length * 4:  # Rough estimate (4 chars per token)
                truncated_summaries.append(summary[:max_length * 4])
            else:
                truncated_summaries.append(summary)
        
        # Process in batches to avoid memory issues
        fluency_scores = []
        
        for i in tqdm(range(0, len(truncated_summaries), self.batch_size), desc="Fluency Evaluation"):
            batch_summaries = truncated_summaries[i:i+self.batch_size]
            batch_documents = documents[i:i+self.batch_size]
            batch_references = reference_summaries[i:i+self.batch_size]
            
            try:
                # Prepare data for UniEval
                batch_data = convert_to_json(
                    output_list=batch_summaries,
                    src_list=batch_documents,
                    ref_list=batch_references
                )
                
                # Get only fluency scores
                eval_scores = self.evaluator.evaluate(
                    batch_data, 
                    dims=['fluency'],  # Only evaluate fluency
                    overall=False,
                    print_result=False
                )
                
                # Extract fluency scores
                batch_fluency = [score['fluency'] for score in eval_scores]
                fluency_scores.extend(batch_fluency)
                
            except Exception as e:
                logger.error(f"Error in fluency batch {i}: {e}")
                fluency_scores.extend([0.0] * len(batch_summaries))
        
        # Create detailed results
        for i, (summary, document, reference_summary, config_name, example_id, fluency) in enumerate(
            zip(summaries, documents, reference_summaries, config_names, example_ids, fluency_scores)
        ):
            results.append({
                'example_id': example_id,
                'config_name': config_name,
                'summary': summary,
                'document': document,
                'reference_summary': reference_summary,
                'unieval_fluency': float(fluency),
                'truncated': len(summary) > max_length * 4
            })
        
        return pd.DataFrame(results)
    
    def calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics"""
        fluency_data = df['unieval_fluency']
        
        stats = {
            'unieval_fluency_mean': float(fluency_data.mean()),
            'unieval_fluency_median': float(fluency_data.median()),
            'unieval_fluency_std': float(fluency_data.std()),
            'unieval_fluency_min': float(fluency_data.min()),
            'unieval_fluency_max': float(fluency_data.max()),
            'unieval_fluency_q25': float(fluency_data.quantile(0.25)),
            'unieval_fluency_q75': float(fluency_data.quantile(0.75))
        }
        
        return stats
    
    def save_results(self, df: pd.DataFrame, stats: Dict[str, Any], output_dir: str, config_name: str):
        """Save detailed and summary results to CSV files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        detailed_path = os.path.join(output_dir, f'unieval_fluency_detailed_{config_name}.csv')
        df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"Detailed results saved to {detailed_path}")
        
        # Save summary statistics
        summary_df = pd.DataFrame([stats])
        summary_path = os.path.join(output_dir, f'unieval_fluency_summary_{config_name}.csv')
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
        results_df = self.calculate_fluency_scores(data, original_documents)
        
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
    output_dir = '/workspace/AnchorSum/evaluation_results/unieval_fluency'
    
    # Ablation files to evaluate
    ablation_files = [
        'base/ablation_base_final_500.json',
        'no_nli/ablation_no_nli_final_500.json', 
        'no_entity/ablation_no_entity_final_500.json',
        'full/ablation_full_final_500.json'  # This file doesn't exist yet, but will be created
    ]
    
    # Initialize evaluator
    evaluator = UniEvalFluencyEvaluator(device=device, batch_size=batch_size)
    
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
        combined_summary_path = os.path.join(output_dir, 'unieval_fluency_combined_summary.csv')
        combined_summary_df.to_csv(combined_summary_path, index=False, encoding='utf-8')
        logger.info(f"Combined summary saved to {combined_summary_path}")
    
    logger.info("UniEval Fluency evaluation completed!")

if __name__ == "__main__":
    main()
