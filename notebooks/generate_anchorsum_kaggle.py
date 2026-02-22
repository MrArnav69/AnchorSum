import os
import sys
import json
import time
import torch
import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

# --- AUTHENTICATION ---
# Hardcode your Hugging Face token here if not using Kaggle Secrets
HF_TOKEN = os.getenv("HF_TOKEN") 

if HF_TOKEN and HF_TOKEN != "your_huggingface_token_here":
    from huggingface_hub import login
    login(token=HF_TOKEN)

# Install/Update Dependencies for Kaggle Environment
os.system("pip install -q transformers accelerate bitsandbytes datasets sentence-transformers spacy nltk")
os.system("python -m spacy download en_core_web_sm")

import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import spacy

# Download NLTK data
nltk.download('punkt', quiet=True)

# --- CONFIGURATION ---
class Config:
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
    ENTITY_MODEL = "en_core_web_sm"
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7
    TOP_P = 0.9
    DO_SAMPLE = True
    BATCH_SIZE = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- ANCHORSUM PIPELINE ---
class AnchorSumPipeline:
    def __init__(self):
        self.setup_models()
        
    def setup_models(self):
        """Initialize all required models"""
        logging.info("Loading models...")
        
        # Load main model
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.MODEL_NAME, 
            token=HF_TOKEN,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            token=HF_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load NLI model
        self.nli_model = SentenceTransformer(Config.NLI_MODEL)
        
        # Load NER model
        self.ner_model = spacy.load(Config.ENTITY_MODEL)
        
        # Create generation pipeline
        self.generation_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
            do_sample=Config.DO_SAMPLE,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        logging.info("All models loaded successfully!")
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text"""
        doc = self.ner_model(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART']:
                entities.append(ent.text)
        return list(set(entities))
    
    def compute_nli_scores(self, premise: str, hypothesis: str) -> float:
        """Compute NLI entailment score"""
        embeddings = self.nli_model.encode([premise, hypothesis])
        similarity = torch.cosine_similarity(
            torch.tensor(embeddings[0:1]), 
            torch.tensor(embeddings[1:2])
        ).item()
        return similarity
    
    def generate_summary(self, document: str, entities: List[str]) -> str:
        """Generate summary using LLM"""
        entity_context = ""
        if entities:
            entity_context = f"Important entities to include: {', '.join(entities[:5])}. "
        
        prompt = f"""Summarize the following document in 3-4 sentences. {entity_context}

Document:
{document}

Summary:"""
        
        try:
            result = self.generation_pipeline(prompt, num_return_sequences=1, max_length=2048)
            summary = result[0]['generated_text'].split('Summary:')[-1].strip()
            return summary
        except Exception as e:
            logging.error(f"Generation error: {e}")
            return "Error generating summary."
    
    def process_document(self, document: str) -> Dict[str, Any]:
        """Process a single document through the AnchorSum pipeline"""
        start_time = time.time()
        
        # Extract entities
        entities = self.extract_entities(document)
        
        # Generate initial summary
        summary = self.generate_summary(document, entities)
        
        # Compute NLI score
        nli_score = self.compute_nli_scores(document, summary)
        
        processing_time = time.time() - start_time
        
        return {
            'summary': summary,
            'entities': entities,
            'nli_score': nli_score,
            'processing_time': processing_time
        }

# --- KAGGLE DATASET PROCESSING ---
def process_kaggle_dataset(input_path: str, output_path: str, sample_size: Optional[int] = None):
    """Process Kaggle dataset and generate summaries"""
    
    # Initialize pipeline
    pipeline = AnchorSumPipeline()
    
    # Load dataset
    logging.info(f"Loading dataset from {input_path}")
    
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.json'):
        df = pd.read_json(input_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")
    
    # Sample data if specified
    if sample_size:
        df = df.head(sample_size)
        logging.info(f"Processing {sample_size} samples")
    
    results = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            logging.info(f"Processing document {idx+1}/{len(df)}")
        
        document = row.get('document') or row.get('text') or row.get('content')
        if not document:
            logging.warning(f"No document content found in row {idx}")
            continue
        
        try:
            result = pipeline.process_document(document)
            result['document_id'] = idx
            result['original_length'] = len(document)
            result['summary_length'] = len(result['summary'])
            results.append(result)
            
        except Exception as e:
            logging.error(f"Error processing document {idx}: {e}")
            continue
    
    # Save results
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")
    
    # Print statistics
    avg_nli_score = output_df['nli_score'].mean()
    avg_processing_time = output_df['processing_time'].mean()
    
    print(f"\n=== Processing Statistics ===")
    print(f"Documents processed: {len(results)}")
    print(f"Average NLI score: {avg_nli_score:.4f}")
    print(f"Average processing time: {avg_processing_time:.2f} seconds")
    print(f"Results saved to: {output_path}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Configuration for Kaggle environment
    INPUT_PATH = "/kaggle/input/your-dataset/data.csv"  # Update this path
    OUTPUT_PATH = "/kaggle/working/anchorsum_results.csv"
    SAMPLE_SIZE = 100  # Process 100 samples for demonstration
    
    # Check if running in Kaggle
    if os.path.exists("/kaggle/input"):
        logging.info("Running in Kaggle environment")
        process_kaggle_dataset(INPUT_PATH, OUTPUT_PATH, SAMPLE_SIZE)
    else:
        logging.info("Running in local environment")
        # For local testing, you can specify your own paths
        local_input = "test_data.csv"  # Update this path
        local_output = "local_results.csv"
        if os.path.exists(local_input):
            process_kaggle_dataset(local_input, local_output, SAMPLE_SIZE)
        else:
            print("Please provide a valid input file path or run in Kaggle environment")
