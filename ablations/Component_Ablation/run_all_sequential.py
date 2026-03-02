import os
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from ablation_base_runner import run_experiment

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MASTER] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    experiments = [
        ("base",      {'nli': False, 'entity': False, 'revision': False}),
        ("no_nli",    {'nli': False, 'entity': True,  'revision': True}),
        ("no_entity", {'nli': True,  'entity': False, 'revision': True}),
        ("full",      {'nli': True,  'entity': True,  'revision': True}),
    ]

    logger.info(f"Sequential ablation study started. {len(experiments)} experiments, 500 samples each.")

    for i, (name, flags) in enumerate(experiments):
        logger.info(f"Starting experiment {i+1}/{len(experiments)}: {name}")

        try:
            run_experiment(name, flags)
            logger.info(f"Experiment {name} completed successfully.")
        except Exception as e:
            logger.error(f"Fatal error in experiment {name}: {str(e)}")
            logger.info("Continuing to next experiment.")

    logger.info("All experiments completed. Data saved in data/ablations/")


if __name__ == "__main__":
    main()
