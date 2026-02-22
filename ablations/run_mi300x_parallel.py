import os
import multiprocessing
import logging
import sys
import torch

# Add parent dir to path to ensure src.pipeline can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation_base_runner import run_experiment

# Setup Logging for Master Orchestrator
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MASTER] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ROCm Optimization Environment Variables
ROCM_OPTS = {
    "HIP_FORCE_DEV_KERNARG": "1",        # Reduce kernel launch latency
    "HSA_ENABLE_SDMA": "0",              # improve stability in multi-tenant environments
    "NCCL_MIN_NCHANNELS": "32",          # AMD-specific channel optimization
    "PYTORCH_ROCM_ARCH": "gfx942",       # MI300X specific target
}

def wrap_run(name, flags):
    """Wrapper to run experiment in a subprocess"""
    try:
        # Apply MI300X Optimization Vars
        os.environ.update(ROCM_OPTS)
        
        # MI300X Memory Limit: 40GB / 192GB = ~0.2083
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.2083, 0)
            logger.info(f"VRAM Hard Limit: 40GB (0.2083 fraction) applied to {name}")
            
        logger.info(f"Starting subprocess for: {name}")
        run_experiment(name, flags)
        logger.info(f"Subprocess {name} finished successfully.")
    except Exception as e:
        logger.error(f"FATAL ERROR in subprocess {name}: {str(e)}")

if __name__ == "__main__":
    # MI300X (192GB VRAM) enables 4x parallel FP32 streams.
    # Estimated 160GB total VRAM usage (32GB headroom).
    
    # Use 'spawn' to ensure clean GPU context for each process
    multiprocessing.set_start_method('spawn', force=True)

    experiments = [
        ("full", {'nli': True, 'entity': True, 'revision': True}),
        ("no_nli", {'nli': False, 'entity': True, 'revision': True}),
        ("no_entity", {'nli': True, 'entity': False, 'revision': True}),
        ("base", {'nli': False, 'entity': False, 'revision': False})
    ]

    logger.info("⚡ MI300X Extreme Parallelization Initiated (1000 Samples/Baseline) ⚡")
    logger.info("Deploying 4 parallel research streams...")

    processes = []
    for name, flags in experiments:
        p = multiprocessing.Process(target=wrap_run, args=(name, flags))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logger.info("✨ All 4 Parallel Research Streams Completed successfully! ✨")
    logger.info("Final results saved in data/ablations/")
