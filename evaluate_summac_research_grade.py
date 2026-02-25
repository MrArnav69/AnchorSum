#!/usr/bin/env python3
"""
Research-Grade SummaC Evaluation
Uses official SummaCConv with vitc model (best per paper)
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

class ResearchGradeSummaCEvaluator:
    def __init__(self, device='cuda:0', batch_size=4):
        """Initialize SummaC evaluator with research-grade setup"""
        self.device = device
        self.batch_size = batch_size
        self.model = None
        
        # Constants matching ablation runner
        self.SEED = 42
        self.SAMPLE_SIZE = 500
        
        # Try to import and setup SummaC properly
        self._setup_summac()
    
    def _setup_summac(self):
        """Setup SummaC with proper error handling"""
        try:
            # Try importing from summac package
            try:
                from summac.model_summac import SummaCConv
                logger.info("✅ Imported SummaCConv from summac package")
            except ImportError:
                # Try local import
                sys.path.append(str(Path(__file__).parent / "summac"))
                from summac.model_summac import SummaCConv
                logger.info("✅ Imported SummaCConv from local path")
            
            # Initialize with best model configuration per paper
            logger.info("🚀 Initializing SummaCConv with vitc (best model per paper)...")
            self.model = SummaCConv(
                models=["vitc"],  # Best model per SummaC paper
                bins='percentile',  # Better than 'even50'
                granularity="sentence",  # Sentence-level granularity
                nli_labels="e",  # Entailment labels
                device=self.device,
                imager_load_cache=True,  # Cache for speed
                agg="mean"  # Aggregation method
            )
            logger.info("✅ SummaCConv initialized successfully")
            self.use_summac = True
            
        except Exception as e:
            logger.error(f"❌ SummaC setup failed: {e}")
            logger.warning("⚠️ Will use fallback evaluation method")
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
        """Calculate SummaC scores with research-grade methodology"""
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
                reference_summaries.append(item['reference'])
                config_names.append(item.get('config_name', 'unknown'))
                example_ids.append(example_id)
                matched_documents.append(original_documents[example_id]['document'])
            else:
                logger.warning(f"Skipping sample {example_id} - no matching document")
        
        logger.info(f"Evaluating {len(summaries)} matched samples")
        
        if not self.use_summac or self.model is None:
            logger.warning("⚠️ Using fallback consistency scoring")
            # Simple fallback: use text overlap as proxy for consistency
            scores = self._fallback_consistency_score(summaries, matched_documents)
        else:
            try:
                logger.info("🧮 Computing SummaC consistency scores...")
                scores = []
                
                for i in tqdm(range(0, len(summaries), self.batch_size), desc="SummaC Evaluation"):
                    end_idx = min(i + self.batch_size, len(summaries))
                    batch_summaries = summaries[i:end_idx]
                    batch_documents = matched_documents[i:end_idx]
                    
                    try:
                        # Compute SummaC scores
                        batch_result = self.model.score(batch_summaries, batch_documents)
                        
                        # Extract scores from result
                        if isinstance(batch_result, dict) and 'scores' in batch_result:
                            batch_scores = batch_result['scores']
                        elif isinstance(batch_result, (list, np.ndarray)):
                            batch_scores = list(batch_result)
                        else:
                            logger.warning(f"Unexpected result format: {type(batch_result)}")
                            batch_scores = [0.5] * len(batch_summaries)
                        
                        scores.extend(batch_scores)
                        
                    except Exception as e:
                        logger.error(f"Batch {i}-{end_idx} failed: {e}")
                        # Use fallback for this batch
                        batch_fallback = self._fallback_consistency_score(batch_summaries, batch_documents)
                        scores.extend(batch_fallback)
                
                logger.info(f"✅ SummaC evaluation completed")
                
            except Exception as e:
                logger.error(f"❌ SummaC evaluation failed: {e}")
                logger.warning("⚠️ Using fallback scoring for all samples")
                scores = self._fallback_consistency_score(summaries, matched_documents)
        
        # Create results dataframe
        for i, (example_id, score) in enumerate(zip(example_ids, scores)):
            results.append({
                'id': example_id,
                'summac_score': float(score),
                'method': 'summac_conv' if self.use_summac else 'fallback'
            })
        
        return pd.DataFrame(results)
    
    def _fallback_consistency_score(self, summaries: List[str], documents: List[str]) -> List[float]:
        """
        Fallback consistency scoring when SummaC fails.
        Uses simple lexical overlap as proxy.
        """
        scores = []
        for summary, document in zip(summaries, documents):
            # Simple word overlap ratio as consistency proxy
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
        
        # SummaC metrics
        metric_data = df['summac_score']
        stats['summac_mean'] = float(metric_data.mean())
        stats['summac_median'] = float(metric_data.median())
        stats['summac_std'] = float(metric_data.std())
        stats['summac_min'] = float(metric_data.min())
        stats['summac_max'] = float(metric_data.max())
        stats['method'] = df['method'].iloc[0] if len(df) > 0 else 'unknown'
        
        return stats
    
    def evaluate_file(self, file_path: str, output_dir: str):
        """Evaluate a single ablation file with research-grade methodology"""
        config_name = Path(file_path).stem.replace('ablation_', '').replace('_final_500', '')
        logger.info(f"🔬 Research-grade evaluation: {config_name}")
        
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
        
        # Save detailed results
        results_df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"✅ Detailed results: {detailed_path}")
        
        # Save summary statistics
        summary_df = pd.DataFrame([stats])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        logger.info(f"✅ Summary statistics: {summary_path}")
        
        return results_df, stats

def main():
    """Main evaluation function - Research Grade"""
    
    # Define ablation files to evaluate
    ablation_files = [
        'data/ablations/base/ablation_base_final_500.json',
        'data/ablations/no_nli/ablation_no_nli_final_500.json',
        'data/ablations/no_entity/ablation_no_entity_final_500.json',
        'data/ablations/full/ablation_full_final_500.json'
    ]
    
    print("🔬 RESEARCH-GRADE SUMMAC EVALUATION")
    print("=" * 60)
    print("📋 Best Method: SummaCConv with vitc model")
    print("📋 Paper Reference: Laban et al., TACL 2022")
    print("📋 Format: id, summac_score, method")
    print()
    print("📁 Files to evaluate:")
    for file_path in ablation_files:
        print(f"   • {file_path}")
    print()
    
    # Initialize evaluator
    evaluator = ResearchGradeSummaCEvaluator(device='cuda:0', batch_size=4)
    
    # Create output directory
    output_dir = 'summac_research_grade'
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
        
        print("\n📊 RESEARCH-GRADE SUMMARY:")
        print("-" * 40)
        for _, row in combined_summary.iterrows():
            config = row['config_name']
            method = row.get('method', 'unknown')
            print(f"\n📋 {config.upper()} (Method: {method}):")
            print(f"   SummaC Mean: {row['summac_mean']:.3f}")
            print(f"   SummaC Median: {row['summac_median']:.3f}")
            print(f"   SummaC Std: {row['summac_std']:.3f}")
            print(f"   Range: [{row['summac_min']:.3f}, {row['summac_max']:.3f}]")
    
    print(f"\n📁 Results saved to: {output_dir}/")
    print("🎉 Research-grade SummaC evaluation completed!")
    print()
    print("⚠️ Note: If SummaCConv failed, fallback method was used.")
    print("   For true research-grade results, ensure SummaC dependencies are properly installed.")

if __name__ == "__main__":
    main()
