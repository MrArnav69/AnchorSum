#!/usr/bin/env python3

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

unieval_path = str(Path(__file__).parent / "UniEval")
if unieval_path not in sys.path:
    sys.path.insert(0, unieval_path)

try:
    import UniEval.utils as unieval_utils
    from UniEval.metric.evaluator import get_evaluator
    convert_to_json = unieval_utils.convert_to_json
    has_unieval = True
except ImportError as e:
    print(f"UniEval import failed: {e}")
    has_unieval = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

root_dir = Path(__file__).resolve().parent.parent.parent


class UniEvalFluencyFullRevisions2:
    def __init__(self, device='cuda:0', batch_size=16):
        self.device = device
        self.batch_size = batch_size
        self.evaluator = None
        
        if has_unieval:
            try:
                logger.info("Loading UniEval model for fluency evaluation...")
                self.evaluator = get_evaluator('summarization')
                logger.info("UniEval loaded")
            except Exception as e:
                logger.error(f"UniEval init failed: {e}")
    
    def load_ablation_data(self, file_path: str) -> List[Dict]:
        logger.info(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} samples")
        return data
    
    def calculate_scores(self, data: List[Dict]) -> pd.DataFrame:
        results = []
        
        summaries = []
        documents = []
        references = []
        example_ids = []
        
        logger.info("Extracting summaries, documents, and references...")
        for item in data:
            example_id = item.get('example_id', -1)
            summary = item.get('final_summary', '')
            document = item.get('document', '')
            reference = item.get('reference', '')
            
            summaries.append(summary)
            documents.append(document)
            references.append(reference)
            example_ids.append(example_id)
        
        logger.info(f"Evaluating {len(summaries)} samples")
        
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
                batch_documents = documents[i:i+self.batch_size]
                batch_references = references[i:i+self.batch_size]
                
                try:
                    batch_data = convert_to_json(
                        output_list=batch_summaries,
                        src_list=batch_documents,
                        ref_list=batch_references
                    )
                    
                    eval_scores = self.evaluator.evaluate(
                        batch_data,
                        dims=['fluency'],
                        overall=False,
                        print_result=False
                    )
                    
                    batch_fluency = [score['fluency'] for score in eval_scores]
                    fluency_scores.extend(batch_fluency)
                    
                except Exception as e:
                    logger.error(f"Batch {i} failed: {e}")
                    fluency_scores.extend([0.0] * len(batch_summaries))
            
            method = 'UniEval-fluency'
            logger.info("UniEval fluency evaluation completed")
            
        else:
            logger.warning("Using fallback scoring")
            fluency_scores = [0.0] * len(summaries)
            method = 'fallback'
        
        for i, (example_id, score) in enumerate(zip(example_ids, fluency_scores)):
            results.append({
                'id': example_id,
                'fluency': float(score),
                'method': method
            })
        
        return pd.DataFrame(results)
    
    def calculate_summary_stats(self, df: pd.DataFrame, config_name: str) -> Dict[str, Any]:
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
        config_name = 'full_revisions_2'
        logger.info(f"Evaluating: {config_name}")
        
        data = self.load_ablation_data(file_path)
        
        results_df = self.calculate_scores(data)
        
        stats = self.calculate_summary_stats(results_df, config_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
        detailed_path = os.path.join(output_dir, f'unieval_fluency_detailed_{config_name}.csv')
        results_df.to_csv(detailed_path, index=False, encoding='utf-8')
        logger.info(f"Detailed results: {detailed_path}")
        
        summary_path = os.path.join(output_dir, f'unieval_fluency_summary_{config_name}.csv')
        summary_df = pd.DataFrame([stats])
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        logger.info(f"Summary statistics: {summary_path}")
        
        return results_df, stats


def update_combined_summary(output_dir: str, stats: Dict[str, Any]):
    combined_path = os.path.join(output_dir, 'combined_summary.csv')
    
    new_row = {
        'config_name': stats['config_name'],
        'fluency_mean': stats['fluency_mean'],
        'fluency_median': stats['fluency_median'],
        'fluency_std': stats['fluency_std'],
        'fluency_min': stats['fluency_min'],
        'fluency_max': stats['fluency_max'],
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
    logger.info(f"Combined summary updated: {combined_path}")
    return combined_df


def main():
    file_path = str(root_dir / 'data' / 'ablations' / 'full_revisions_2' / 'ablation_full_revisions_2_final_500.json')
    output_dir = str(root_dir / 'Results' / 'Revision Depth' / 'unieval_fluency_full_revisions_2_results')
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    
    evaluator = UniEvalFluencyFullRevisions2(device='cuda:0', batch_size=16)
    
    try:
        results_df, stats = evaluator.evaluate_file(file_path, output_dir)
        update_combined_summary(output_dir, stats)
        
        print("\nEVALUATION SUMMARY - Full Revisions 2")
        print("=" * 60)
        print(f"Fluency Mean: {stats['fluency_mean']:.4f}")
        print(f"Method: {stats['method']}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
