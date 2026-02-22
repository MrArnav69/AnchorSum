import logging
import spacy
from typing import List, Tuple, Set

logger = logging.getLogger(__name__)

class EntityGuard:
    def __init__(self, model_name: str = "en_core_web_sm", top_n: int = 15):
        """
        Initializes the Entity Guard for Anchor Extraction and Coverage/Hallucination verification.
        """
        self.top_n = top_n
        try:
            logger.info(f"Loading spacy model: {model_name}...")
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.info(f"Downloading spacy {model_name} model...")
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)
            
        # Entity labels we care about for factual consistency
        self.target_labels = {"PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY", "PERCENT"}

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extracts relevant named entities from text, lowercased for matching.
        """
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in self.target_labels:
                # Basic cleaning to remove stop words from entities if any
                clean_ent = ent.text.strip().lower()
                if len(clean_ent) > 2:
                    entities.append(clean_ent)
        return entities

    def extract_anchors(self, source_text: str) -> List[str]:
        """
        Extracts the most frequent and important entities from the source text 
        to serve as narrative "Anchors".
        """
        entities = self._extract_entities(source_text)
        
        # Count frequencies
        freq_map = {}
        for e in entities:
            freq_map[e] = freq_map.get(e, 0) + 1
            
        # Sort by frequency
        sorted_ents = sorted(freq_map.items(), key=lambda item: item[1], reverse=True)
        
        # Take the top N unique anchors
        anchors = [e[0].title() for e in sorted_ents[:self.top_n]]
        return anchors

    def verify_draft(self, source_anchors: List[str], source_text: str, draft_summary: str) -> List[str]:
        """
        Verifies the draft summary against the source anchors and source text.
        Returns:
            flags: A list of strings describing missing anchors or hallucinated entities.
        """
        flags = []
        draft_entities_raw = self._extract_entities(draft_summary)
        draft_entities_set = set(draft_entities_raw)
        
        source_entities_raw = self._extract_entities(source_text)
        source_entities_set = set(source_entities_raw)
        
        # 1. Coverage Check: Did the draft miss critical anchors?
        draft_text_lower = draft_summary.lower()
        missing_anchors = []
        for anchor in source_anchors:
            if anchor.lower() not in draft_text_lower:
                missing_anchors.append(anchor)
                
        if missing_anchors:
            flags.append(f"Missing Key Anchors: Your draft failed to mention: {', '.join(missing_anchors)}.")
            
        # 2. Hallucination Check: Did the draft invent new proper nouns?
        hallucinated_ents = []
        for draft_ent in draft_entities_set:
            # If the entity is completely novel and not in the source text
            if draft_ent not in source_entities_set and draft_ent not in source_text.lower():
                hallucinated_ents.append(draft_ent.title())
                
        if hallucinated_ents:
            flags.append(f"Hallucinated Entities Detected: The following entities were found in your draft but NOT in the source documents: {', '.join(hallucinated_ents)}. Remove them.")
            
        return flags
