import torch
import logging
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize

# Ensure nltk punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class NLIVerifier:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", device: str = "auto", token: str = None):
        """
        Initializes the Cross-Encoder for NLI-based factual verification.
        """
        self.token = token
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing NLIVerifier with {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.token)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, token=self.token).to(self.device)
        self.model.eval()

        # Label mapping for nli-deberta-v3-base: 0: Contradiction, 1: Entailment, 2: Neutral
        # We want to flag sentences that are Contradiction or Neutral (hallucinated)
        self.label_mapping = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}

    def verify_draft(self, source_text: str, draft_summary: str) -> Tuple[List[str], List[str]]:
        """
        Verifies every sentence in the draft against the source text.
        Returns:
            passed_sentences: List of sentences that are entailed by the source.
            flagged_sentences: List of sentences that are contradictions or neutral/hallucinated.
        """
        draft_sentences = sent_tokenize(draft_summary)
        
        passed = []
        flagged = []
        
        for sentence in draft_sentences:
            if not sentence.strip():
                continue
                
            features = self.tokenizer(
                [source_text], 
                [sentence], 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
                max_length=512 # Limiting source context for speed, ideally we'd chunk source or use longformer NLI
            ).to(self.device)
            
            with torch.no_grad():
                scores = self.model(**features).logits
                label_mapping = [self.label_mapping[i] for i in scores.argmax(dim=1).tolist()]
                
            prediction = label_mapping[0]
            
            if prediction == "ENTAILMENT":
                passed.append(sentence)
            else:
                flag_reason = f"Sentence: '{sentence}' -> Fails NLI check ({prediction})."
                flagged.append(flag_reason)
                
        return passed, flagged