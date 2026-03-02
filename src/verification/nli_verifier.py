import torch
import logging
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class nliverifier:
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", device: str = "auto", token: str = None):
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

        self.label_mapping = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}

    def verify_draft(self, source_text: str, draft_summary: str) -> Tuple[List[str], List[str]]:
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
                max_length=512
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