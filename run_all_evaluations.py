#!/usr/bin/env python3
"""
Master Evaluation Script
Runs all evaluation scripts for ablation studies
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_evaluation_script(script_name, description):
    """Run a single evaluation script"""
    logger.info(f"Starting {description}...")
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, check=True)
        logger.info(f"{description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed with error: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running {description}: {e}")
        return False

def main():
    """Run all evaluation scripts"""
    logger.info("Starting comprehensive evaluation of ablation studies...")
    
    # Define evaluation scripts
    evaluation_scripts = [
        ("evaluate_bartscore.py", "BARTScore Evaluation"),
        ("evaluate_unieval_fluency.py", "UniEval Fluency Evaluation"),
        ("evaluate_alignscore.py", "AlignScore Evaluation"),
        ("evaluate_summac.py", "SummaC Evaluation"),
        ("evaluate_rouge_bertscore.py", "ROUGE and BERTScore Evaluation")
    ]
    
    # Check if scripts exist
    missing_scripts = []
    for script, _ in evaluation_scripts:
        if not os.path.exists(script):
            missing_scripts.append(script)
    
    if missing_scripts:
        logger.error(f"Missing evaluation scripts: {missing_scripts}")
        return False
    
    # Run each evaluation
    results = {}
    for script, description in evaluation_scripts:
        success = run_evaluation_script(script, description)
        results[description] = success
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    
    for description, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{description}: {status}")
    
    # Overall result
    all_success = all(results.values())
    if all_success:
        logger.info("\n🎉 All evaluations completed successfully!")
        logger.info("Results are available in /workspace/AnchorSum/evaluation_results/")
    else:
        logger.error(f"\n❌ {sum(1 for s in results.values() if not s)} evaluation(s) failed")
    
    return all_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
