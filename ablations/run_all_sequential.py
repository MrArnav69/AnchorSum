import os
import logging
from ablation_base_runner import run_experiment

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MASTER] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Define the 4 ablation baselines
    experiments = [
        ("base", {'nli': False, 'entity': False, 'revision': False}),     # 6.1h  ← done first
        ("no_nli", {'nli': False, 'entity': True, 'revision': True}),      # 6.7h  ← done second
        ("no_entity", {'nli': True, 'entity': False, 'revision': True}),   # 13.6h ← done third
        ("full", {'nli': True, 'entity': True, 'revision': True}),         # 14.2h ← done last
    ]

    logger.info("🎬 NVIDIA A40 Master Sequential Study Initiated (500 Samples/Baseline) 🎬")
    logger.info(f"Total Study Scale: 2000 Samples across {len(experiments)} baselines.")

    for i, (name, flags) in enumerate(experiments):
        logger.info(f"🚀 Starting Experiment {i+1}/{len(experiments)}: {name}...")
        
        try:
            run_experiment(name, flags)
            logger.info(f"✅ Experiment {name} completed successfully.")
        except Exception as e:
            logger.error(f"❌ FATAL ERROR in experiment {name}: {str(e)}")
            logger.info("Continuing to next experiment in queue...")

    logger.info("🏁 ALL RESEARCH STREAMS COMPLETED 🏁")
    logger.info("Data saved in data/ablations/")
    logger.info("Use scripts/evaluate_rouge.py to analyze the results.")

if __name__ == "__main__":
    main()
