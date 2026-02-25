#!/usr/bin/env python3
"""
Run full experiments with max revisions 2 and 3 on first 200 samples
Uses the modified ablation_base_runner.py
"""

import sys
import os

# Add the ablations directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ablation_base_runner import run_experiment

def main():
    """Run full experiments with max revisions 2 and 3"""
    
    # Experiment configurations
    experiments = [
        {
            'config_name': 'full_revisions_2',
            'max_revisions': 2,
            'ablation_flags': {}  # No ablations, full model with max_revisions=2
        },
        {
            'config_name': 'full_revisions_3', 
            'max_revisions': 3,
            'ablation_flags': {}  # No ablations, full model with max_revisions=3
        }
    ]
    
    print("🚀 Starting Full Experiments")
    print("=" * 50)
    
    for experiment in experiments:
        print(f"\n🔧 Running experiment: {experiment['config_name']}")
        print(f"   Max revisions: {experiment['max_revisions']}")
        print(f"   Ablation flags: {experiment['ablation_flags']}")
        
        try:
            # Run the experiment using the modified base runner
            run_experiment(
                config_name=experiment['config_name'],
                ablation_flags=experiment['ablation_flags'],
                max_revisions=experiment['max_revisions'],
                sample_size=200  # Override to 200 samples
            )
            print(f"✅ Completed: {experiment['config_name']}")
            
        except Exception as e:
            print(f"❌ Failed: {experiment['config_name']} - {str(e)}")
            continue
    
    print("\n" + "=" * 50)
    print("🎉 All experiments completed!")
    print("Results saved to:")
    print("   - data/ablations/full_revisions_2/ablation_full_revisions_2_final_200.json")
    print("   - data/ablations/full_revisions_3/ablation_full_revisions_3_final_200.json")

if __name__ == "__main__":
    main()
