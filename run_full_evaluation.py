#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for All Ablation Files
Runs BARTScore, UniEval Fluency, AlignScore, SummaC, and ROUGE/BERTScore
Professional research-grade evaluation with proper setup
"""

import os
import sys
import logging
import time
from datetime import datetime
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all evaluation scripts (excluding AlignScore for now)
from evaluate_bartscore import BARTScoreEvaluator
from evaluate_unieval_fluency import UniEvalFluencyEvaluator
# from evaluate_alignscore import AlignScoreEvaluator  # Temporarily disabled
from evaluate_summac import SummaCEvaluator
from evaluate_rouge_bertscore import ROUGEAndBERTScoreEvaluator

# Setup professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [EVALUATION] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FullEvaluationSuite:
    """Professional evaluation suite for all metrics"""
    
    def __init__(self, device='cuda:0', batch_size=16):
        self.device = device
        self.batch_size = batch_size
        self.output_dir = 'evaluation_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize all evaluators
        logger.info("🔧 Initializing evaluation suite...")
        self.initialize_evaluators()
        
    def initialize_evaluators(self):
        """Initialize all evaluation models"""
        try:
            logger.info("📦 Loading BARTScore evaluator...")
            self.bartscore_evaluator = BARTScoreEvaluator(
                device=self.device, 
                batch_size=min(8, self.batch_size)  # Smaller batch for BARTScore
            )
            
            logger.info("📦 Loading UniEval Fluency evaluator...")
            self.unieval_evaluator = UniEvalFluencyEvaluator(
                device=self.device, 
                batch_size=self.batch_size
            )
            
            # Temporarily skip AlignScore
            logger.info("⏭️ Skipping AlignScore evaluator (will run later)")
            self.alignscore_evaluator = None
            
            logger.info("📦 Loading SummaC evaluator...")
            self.summac_evaluator = SummaCEvaluator(
                device=self.device, 
                batch_size=self.batch_size
            )
            
            logger.info("📦 Loading ROUGE/BERTScore evaluator...")
            self.rouge_bert_evaluator = ROUGEAndBERTScoreEvaluator(
                device=self.device, 
                batch_size=self.batch_size
            )
            
            logger.info("✅ All evaluators initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize evaluators: {str(e)}")
            raise
    
    def evaluate_single_file(self, file_path, config_name):
        """Evaluate a single ablation file with all metrics"""
        logger.info(f"🔍 Evaluating {config_name} configuration...")
        start_time = time.time()
        
        results = {}
        
        try:
            # 1. BARTScore Evaluation
            logger.info("   📊 Running BARTScore evaluation...")
            bartscore_results, bartscore_stats = self.bartscore_evaluator.evaluate_file(
                file_path, self.output_dir
            )
            results['bartscore'] = {
                'detailed': bartscore_results,
                'summary': bartscore_stats
            }
            
        except Exception as e:
            logger.error(f"   ❌ BARTScore failed: {str(e)}")
            results['bartscore'] = None
        
        try:
            # 2. UniEval Fluency Evaluation
            logger.info("   📊 Running UniEval Fluency evaluation...")
            unieval_results, unieval_stats = self.unieval_evaluator.evaluate_file(
                file_path, self.output_dir
            )
            results['unieval_fluency'] = {
                'detailed': unieval_results,
                'summary': unieval_stats
            }
            
        except Exception as e:
            logger.error(f"   ❌ UniEval Fluency failed: {str(e)}")
            results['unieval_fluency'] = None
        
        try:
            # 3. AlignScore Evaluation (skipped for now)
            logger.info("   ⏭️ Skipping AlignScore evaluation...")
            results['alignscore'] = None
            
        except Exception as e:
            logger.error(f"   ❌ AlignScore failed: {str(e)}")
            results['alignscore'] = None
        
        try:
            # 4. SummaC Evaluation
            logger.info("   📊 Running SummaC evaluation...")
            summac_results, summac_stats = self.summac_evaluator.evaluate_file(
                file_path, self.output_dir
            )
            results['summac'] = {
                'detailed': summac_results,
                'summary': summac_stats
            }
            
        except Exception as e:
            logger.error(f"   ❌ SummaC failed: {str(e)}")
            results['summac'] = None
        
        try:
            # 5. ROUGE/BERTScore Evaluation
            logger.info("   📊 Running ROUGE/BERTScore evaluation...")
            rouge_bert_results, rouge_bert_stats = self.rouge_bert_evaluator.evaluate_file(
                file_path, self.output_dir
            )
            results['rouge_bertscore'] = {
                'detailed': rouge_bert_results,
                'summary': rouge_bert_stats
            }
            
        except Exception as e:
            logger.error(f"   ❌ ROUGE/BERTScore failed: {str(e)}")
            results['rouge_bertscore'] = None
        
        # Calculate evaluation time
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        logger.info(f"   ✅ {config_name} evaluation completed in {evaluation_time:.1f}s")
        
        return results, evaluation_time
    
    def create_summary_report(self, all_results):
        """Create comprehensive summary report"""
        logger.info("📋 Creating summary report...")
        
        summary_data = []
        
        for config_name, results in all_results.items():
            row = {'config_name': config_name}
            
            # Add summary statistics for each metric
            if results.get('bartscore'):
                stats = results['bartscore']['summary']
                row.update({
                    'bartscore_sum2doc_mean': stats.get('bartscore_sum2doc_mean', 0),
                    'bartscore_doc2sum_mean': stats.get('bartscore_doc2sum_mean', 0),
                    'bartscore_sum2doc_median': stats.get('bartscore_sum2doc_median', 0),
                    'bartscore_doc2sum_median': stats.get('bartscore_doc2sum_median', 0)
                })
            
            if results.get('unieval_fluency'):
                stats = results['unieval_fluency']['summary']
                row.update({
                    'unieval_fluency_mean': stats.get('unieval_fluency_mean', 0),
                    'unieval_fluency_median': stats.get('unieval_fluency_median', 0)
                })
            
            if results.get('alignscore'):
                stats = results['alignscore']['summary']
                row.update({
                    'alignscore_mean': stats.get('alignscore_mean', 0),
                    'alignscore_median': stats.get('alignscore_median', 0)
                })
            
            if results.get('summac'):
                stats = results['summac']['summary']
                row.update({
                    'summac_mean': stats.get('summac_mean', 0),
                    'summac_median': stats.get('summac_median', 0)
                })
            
            if results.get('rouge_bertscore'):
                stats = results['rouge_bertscore']['summary']
                row.update({
                    'rouge1_mean': stats.get('rouge1_mean', 0),
                    'rouge2_mean': stats.get('rouge2_mean', 0),
                    'rougel_mean': stats.get('rougel_mean', 0),
                    'bertscore_mean': stats.get('bertscore_mean', 0)
                })
            
            summary_data.append(row)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save comprehensive summary
        summary_path = os.path.join(self.output_dir, 'comprehensive_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        logger.info(f"📊 Comprehensive summary saved to {summary_path}")
        
        return summary_df
    
    def run_full_evaluation(self, ablation_files):
        """Run full evaluation suite on all ablation files"""
        logger.info("🚀 Starting Comprehensive Evaluation Suite")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🔧 Device: {self.device}")
        logger.info(f"📦 Batch size: {self.batch_size}")
        
        all_results = {}
        total_start_time = time.time()
        
        for file_path, config_name in ablation_files:
            if not os.path.exists(file_path):
                logger.error(f"❌ File not found: {file_path}")
                continue
            
            results, eval_time = self.evaluate_single_file(file_path, config_name)
            all_results[config_name] = results
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        # Create summary report
        summary_df = self.create_summary_report(all_results)
        
        # Final report
        logger.info("🎉 Full evaluation completed!")
        logger.info(f"⏱️  Total time: {total_time:.1f}s")
        logger.info(f"📊 Summary report: {os.path.join(self.output_dir, 'comprehensive_summary.csv')}")
        
        return all_results, summary_df

def main():
    """Main evaluation function"""
    
    # Define ablation files to evaluate
    ablation_files = [
        ('data/ablations/base/ablation_base_final_500.json', 'base'),
        ('data/ablations/no_nli/ablation_no_nli_final_500.json', 'no_nli'),
        ('data/ablations/no_entity/ablation_no_entity_final_500.json', 'no_entity'),
        ('data/ablations/full/ablation_full_final_500.json', 'full')
    ]
    
    print("🔬 COMPREHENSIVE EVALUATION SUITE")
    print("=" * 60)
    print("📋 Metrics to evaluate:")
    print("   • BARTScore (sum2doc & doc2sum)")
    print("   • UniEval Fluency")
    print("   • SummaC")
    print("   • ROUGE-1, ROUGE-2, ROUGE-L")
    print("   • BERTScore (deberta-xlarge-mnli)")
    print("   ⏭️ AlignScore (skipped for now)")
    print()
    print("📁 Files to evaluate:")
    for file_path, config_name in ablation_files:
        print(f"   • {config_name}: {file_path}")
    print()
    
    # TIME ESTIMATE
    print("⏱️ TIME ESTIMATE:")
    print("-" * 30)
    print("• BARTScore: ~15-20 minutes per file")
    print("• UniEval Fluency: ~10-15 minutes per file") 
    print("• SummaC: ~8-12 minutes per file")
    print("• ROUGE/BERTScore: ~20-25 minutes per file")
    print("• Total estimated time: ~3-4 hours for all 4 files")
    print()
    
    # Initialize evaluation suite
    try:
        suite = FullEvaluationSuite(device='cuda:0', batch_size=16)
        
        # Run full evaluation
        all_results, summary_df = suite.run_full_evaluation(ablation_files)
        
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        # Display key metrics
        print("\n🎯 Key Performance Metrics:")
        print("-" * 40)
        
        for _, row in summary_df.iterrows():
            config = row['config_name']
            print(f"\n📋 {config.upper()} Configuration:")
            
            if 'bartscore_sum2doc_mean' in row and pd.notna(row['bartscore_sum2doc_mean']):
                print(f"   BARTScore sum2doc: {row['bartscore_sum2doc_mean']:.3f}")
                print(f"   BARTScore doc2sum: {row['bartscore_doc2sum_mean']:.3f}")
            
            if 'unieval_fluency_mean' in row and pd.notna(row['unieval_fluency_mean']):
                print(f"   UniEval Fluency: {row['unieval_fluency_mean']:.3f}")
            
            if 'summac_mean' in row and pd.notna(row['summac_mean']):
                print(f"   SummaC: {row['summac_mean']:.3f}")
            
            if 'rouge1_mean' in row and pd.notna(row['rouge1_mean']):
                print(f"   ROUGE-1: {row['rouge1_mean']:.3f}")
                print(f"   ROUGE-2: {row['rouge2_mean']:.3f}")
                print(f"   ROUGE-L: {row['rougel_mean']:.3f}")
                print(f"   BERTScore: {row['bertscore_mean']:.3f}")
        
        print(f"\n📁 Detailed results saved to: evaluation_results/")
        print("🎉 Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
