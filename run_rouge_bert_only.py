#!/usr/bin/env python3
"""
ROUGE and BERTScore Only Evaluation
Runs ROUGE-1, ROUGE-2, ROUGE-L and BERTScore (deberta-xlarge-mnli) on all ablation files
"""

import os
import sys
import logging
import time
from datetime import datetime
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import ROUGE/BERTScore evaluator
from evaluate_rouge_bertscore import ROUGEAndBERTScoreEvaluator

# Setup professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [ROUGE_BERT] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rouge_bert_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ROUGEBERTOnlyEvaluator:
    """ROUGE and BERTScore only evaluation suite"""
    
    def __init__(self, device='cuda:0', batch_size=16):
        self.device = device
        self.batch_size = batch_size
        self.output_dir = 'rouge_bert_results'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize evaluator
        logger.info("🔧 Initializing ROUGE/BERTScore evaluator...")
        self.evaluator = ROUGEAndBERTScoreEvaluator(
            device=self.device, 
            batch_size=self.batch_size
        )
        logger.info("✅ ROUGE/BERTScore evaluator initialized successfully")
        
    def evaluate_single_file(self, file_path, config_name):
        """Evaluate a single ablation file"""
        logger.info(f"🔍 Evaluating {config_name} configuration...")
        start_time = time.time()
        
        try:
            # Run evaluation
            logger.info("   📊 Running ROUGE/BERTScore evaluation...")
            results_df, stats = self.evaluator.evaluate_file(file_path, self.output_dir)
            
            end_time = time.time()
            evaluation_time = end_time - start_time
            
            logger.info(f"   ✅ {config_name} evaluation completed in {evaluation_time:.1f}s")
            
            return results_df, stats, evaluation_time
            
        except Exception as e:
            logger.error(f"   ❌ ROUGE/BERTScore evaluation failed: {str(e)}")
            return None, None, 0
    
    def run_evaluation(self, ablation_files):
        """Run ROUGE/BERTScore evaluation on all ablation files"""
        logger.info("🚀 Starting ROUGE/BERTScore Evaluation Suite")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🔧 Device: {self.device}")
        logger.info(f"📦 Batch size: {self.batch_size}")
        
        all_results = {}
        total_start_time = time.time()
        
        for file_path, config_name in ablation_files:
            if not os.path.exists(file_path):
                logger.error(f"❌ File not found: {file_path}")
                continue
            
            results_df, stats, eval_time = self.evaluate_single_file(file_path, config_name)
            all_results[config_name] = {
                'detailed': results_df,
                'summary': stats
            }
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        # Create summary report
        self.create_summary_report(all_results)
        
        logger.info(f"🎉 Full evaluation completed in {total_time:.1f}s!")
        logger.info(f"📊 Summary report: {os.path.join(self.output_dir, 'rouge_bert_summary.csv')}")
        
        return all_results
    
    def create_summary_report(self, all_results):
        """Create comprehensive summary report"""
        logger.info("📋 Creating ROUGE/BERTScore summary report...")
        
        summary_data = []
        
        for config_name, results in all_results.items():
            if results and results['summary']:
                stats = results['summary']
                row = {'config_name': config_name}
                
                # Add ROUGE metrics
                row.update({
                    'rouge1_mean': stats.get('rouge1_mean', 0),
                    'rouge1_median': stats.get('rouge1_median', 0),
                    'rouge1_std': stats.get('rouge1_std', 0),
                    'rouge2_mean': stats.get('rouge2_mean', 0),
                    'rouge2_median': stats.get('rouge2_median', 0),
                    'rouge2_std': stats.get('rouge2_std', 0),
                    'rougel_mean': stats.get('rougel_mean', 0),
                    'rougel_median': stats.get('rougel_median', 0),
                    'rougel_std': stats.get('rougel_std', 0),
                    'bertscore_mean': stats.get('bertscore_mean', 0),
                    'bertscore_median': stats.get('bertscore_median', 0),
                    'bertscore_std': stats.get('bertscore_std', 0)
                })
                
                summary_data.append(row)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save comprehensive summary
        summary_path = os.path.join(self.output_dir, 'rouge_bert_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        logger.info(f"📊 ROUGE/BERTScore summary saved to {summary_path}")
        
        return summary_df

def main():
    """Main evaluation function"""
    
    # Define ablation files to evaluate
    ablation_files = [
        ('data/ablations/base/ablation_base_final_500.json', 'base'),
        ('data/ablations/no_nli/ablation_no_nli_final_500.json', 'no_nli'),
        ('data/ablations/no_entity/ablation_no_entity_final_500.json', 'no_entity'),
        ('data/ablations/full/ablation_full_final_500.json', 'full')
    ]
    
    print("🔬 ROUGE & BERTScore EVALUATION SUITE")
    print("=" * 60)
    print("📋 Metrics to evaluate:")
    print("   • ROUGE-1, ROUGE-2, ROUGE-L")
    print("   • BERTScore (deberta-xlarge-mnli)")
    print()
    print("📁 Files to evaluate:")
    for file_path, config_name in ablation_files:
        print(f"   • {config_name}: {file_path}")
    print()
    
    # TIME ESTIMATE
    print("⏱️ TIME ESTIMATE:")
    print("-" * 30)
    print("• ROUGE evaluation: ~8-12 minutes per file")
    print("• BERTScore evaluation: ~12-18 minutes per file") 
    print("• Combined: ~20-30 minutes per file")
    print("• Total estimated time: ~1.5-2 hours for all 4 files")
    print()
    
    # Initialize evaluation suite
    try:
        suite = ROUGEBERTOnlyEvaluator(device='cuda:0', batch_size=16)
        
        # Run evaluation
        all_results = suite.run_evaluation(ablation_files)
        
        print("\n" + "=" * 60)
        print("📊 ROUGE/BERTScore EVALUATION SUMMARY")
        print("=" * 60)
        
        # Display key metrics
        print("\n🎯 Key Performance Metrics:")
        print("-" * 40)
        
        # Load summary for display
        summary_df = pd.read_csv(os.path.join('rouge_bert_results', 'rouge_bert_summary.csv'))
        
        for _, row in summary_df.iterrows():
            config = row['config_name']
            print(f"\n📋 {config.upper()} Configuration:")
            print(f"   ROUGE-1: {row['rouge1_mean']:.3f} (±{row['rouge1_std']:.3f})")
            print(f"   ROUGE-2: {row['rouge2_mean']:.3f} (±{row['rouge2_std']:.3f})")
            print(f"   ROUGE-L: {row['rougel_mean']:.3f} (±{row['rougel_std']:.3f})")
            print(f"   BERTScore: {row['bertscore_mean']:.3f} (±{row['bertscore_std']:.3f})")
        
        print(f"\n📁 Detailed results saved to: rouge_bert_results/")
        print("🎉 ROUGE/BERTScore evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
