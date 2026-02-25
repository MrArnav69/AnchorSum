#!/usr/bin/env python3
"""
Run experiment with max revisions 3 on first 300 samples
Uses the modified ablation_base_runner.py
"""

import sys
import os

# Add the ablations directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ablation_base_runner import run_experiment

def main():
    """Run experiment with max revisions 3 on first 300 samples"""
    
    print("🚀 Starting Revision 3 Experiment (300 samples)")
    print("=" * 50)
    
    experiment_config = {
        'config_name': 'full_revisions_3_300',
        'max_revisions': 3,
        'sample_size': 300,
        'ablation_flags': {}  # No ablations, full model with max_revisions=3
    }
    
    print(f"\n🔧 Running experiment: {experiment_config['config_name']}")
    print(f"   Max revisions: {experiment_config['max_revisions']}")
    print(f"   Sample size: {experiment_config['sample_size']}")
    print(f"   Ablation flags: {experiment_config['ablation_flags']}")
    
    try:
        # Run the experiment using the modified base runner
        run_experiment(
            config_name=experiment_config['config_name'],
            ablation_flags=experiment_config['ablation_flags'],
            max_revisions=experiment_config['max_revisions'],
            sample_size=experiment_config['sample_size']
        )
        print(f"✅ Completed: {experiment_config['config_name']}")
        
    except Exception as e:
        print(f"❌ Failed: {experiment_config['config_name']} - {str(e)}")
        return
    
    print("\n" + "=" * 50)
    print("🎉 Experiment completed!")
    print("Results saved to:")
    print(f"   - data/ablations/{experiment_config['config_name']}/ablation_{experiment_config['config_name']}_final_300.json")
    print("Checkpoints will be saved at: 50, 100, 150, 200, 250, 300 samples")

if __name__ == "__main__":
    main()
