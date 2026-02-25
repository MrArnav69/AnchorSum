#!/usr/bin/env python3
"""
Debug SummaC Evaluation
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "summac"))
from summac.model_summac import SummaCConv

def main():
    print("🧪 SummaC Debug Test")
    
    # Initialize SummaC
    try:
        model = SummaCConv(
            models=["vitc"], 
            bins='even50', 
            granularity="sentence", 
            nli_labels="e", 
            device='cuda:0'
        )
        print("✅ SummaC initialized successfully")
        
        # Test with simple data
        summary = "This is a test summary."
        document = "This is a test document with multiple sentences. The summary should capture the key information accurately."
        
        try:
            score = model.score([summary], [document])
            print(f"✅ Score calculated: {score}")
            print(f"Score type: {type(score)}")
            print(f"Score value: {float(score) if score is not None else 'None'}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"❌ Failed to initialize SummaC: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
