import sys
import json
import logging
import time
from pathlib import Path

# Add src to path so we can import our modules
src_path = Path(__file__).resolve().parent.parent / "src"
sys.path.append(str(src_path))

from pipeline import AnchorSumPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_smoke_test():
    data_path = Path(__file__).resolve().parent.parent / "data" / "anchorsum_5_test.json"
    output_path = Path(__file__).resolve().parent.parent / "data" / "anchorsum_5_test_results.json"
    
    logger.info(f"Loading data from {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)
        
    # We will only run on the first 2 samples for the smoke test
    test_data = data[:2]
    
    # Initialize Pipeline using a small model for local testing
    # Using Qwen2.5-1.5B to support the large context window
    pipeline = AnchorSumPipeline(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        nli_model_name="cross-encoder/nli-deberta-v3-base",
        device="auto", # Should pick up MPS or CPU on Mac
        load_in_4bit=False, # Local Mac MPS probably doesn't support bitsandbytes nicely
        max_revisions=1
    )
    
    results = []
    
    for i, item in enumerate(test_data):
        logger.info(f"--- Processing Sample {i+1}/2 (ID: {item['sample_id']}) ---")
        start_time = time.time()
        
        source_doc = item["document"]
        
        # Run the full summarize Draft-Audit-Revise loop
        pipeline_output = pipeline.summarize(source_doc)
        
        elapsed = time.time() - start_time
        logger.info(f"Sample {item['sample_id']} finished in {elapsed:.2f} seconds.")
        
        # Save results
        new_item = {
            "sample_id": item["sample_id"],
            "document": item["document"],
            "reference_summary": item["reference_summary"],
            "anchorsum_final": pipeline_output["final_summary"],
            "anchors_extracted": pipeline_output["anchors"],
            "total_revisions": pipeline_output["revisions"],
            "history": pipeline_output["history"]
        }
        results.append(new_item)
        
        # Save intermediate in case of crash
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
    logger.info(f"Smoke test complete. Results saved to {output_path}")

if __name__ == "__main__":
    run_smoke_test()
