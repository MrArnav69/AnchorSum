import logging
import spacy
from typing import List, Tuple, Set

logger = logging.getLogger(__name__)

class entityguard:
    def __init__(self, model_name: str = "en_core_web_sm", top_n: int = 15):
        self.top_n = top_n
        try:
            logger.info(f"Loading spacy model: {model_name}...")
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.info(f"Downloading spacy {model_name} model...")
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)
            
        self.target_labels = {"PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY", "PERCENT"}

    def _extract_entities(self, text: str) -> List[str]:
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in self.target_labels:
                clean_ent = ent.text.strip().lower()
                if len(clean_ent) > 2:
                    entities.append(clean_ent)
        return entities

    def extract_anchors(self, source_text: str) -> List[str]:
        entities = self._extract_entities(source_text)
        
        freq_map = {}
        for e in entities:
            freq_map[e] = freq_map.get(e, 0) + 1
            
        sorted_ents = sorted(freq_map.items(), key=lambda item: item[1], reverse=True)
        
        anchors = [e[0].title() for e in sorted_ents[:self.top_n]]
        return anchors

    def verify_draft(self, source_anchors: List[str], source_text: str, draft_summary: str) -> List[str]:
        flags = []
        draft_entities_raw = self._extract_entities(draft_summary)
        draft_entities_set = set(draft_entities_raw)
        
        source_entities_raw = self._extract_entities(source_text)
        source_entities_set = set(source_entities_raw)
        
        draft_text_lower = draft_summary.lower()
        missing_anchors = []
        for anchor in source_anchors:
            if anchor.lower() not in draft_text_lower:
                missing_anchors.append(anchor)
                
        if missing_anchors:
            flags.append(f"Missing Key Anchors: Your draft failed to mention: {', '.join(missing_anchors)}.")
            
        hallucinated_ents = []
        for draft_ent in draft_entities_set:
            if draft_ent not in source_entities_set and draft_ent not in source_text.lower():
                hallucinated_ents.append(draft_ent.title())
                
        if hallucinated_ents:
            flags.append(f"Hallucinated Entities Detected: The following entities were found in your draft but NOT in the source documents: {', '.join(hallucinated_ents)}. Remove them.")
            
        return flags