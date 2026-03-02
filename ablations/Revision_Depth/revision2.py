#!/usr/bin/env python3

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from ablation_base_runner import run_experiment


def main():
    experiments = [
        {
            'config_name': 'full_revisions_2',
            'max_revisions': 2,
            'ablation_flags': {}
        }
    ]

    print("Starting Full Experiments")
    print("=" * 50)

    for experiment in experiments:
        print(f"\nRunning experiment: {experiment['config_name']}")
        print(f"   Max revisions: {experiment['max_revisions']}")
        print(f"   Ablation flags: {experiment['ablation_flags']}")

        try:
            run_experiment(
                config_name=experiment['config_name'],
                ablation_flags=experiment['ablation_flags'],
                max_revisions=experiment['max_revisions'],
                sample_size=500
            )
            print(f"Completed: {experiment['config_name']}")

        except Exception as e:
            print(f"Failed: {experiment['config_name']} - {str(e)}")
            continue

    print("\n" + "=" * 50)
    print("All experiments completed!")
    print("Results saved to:")
    print("   - data/ablations/full_revisions_2/ablation_full_revisions_2_final_500.json")


if __name__ == "__main__":
    main()
