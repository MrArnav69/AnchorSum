import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from .llm_summarizer import singlepassmds
from .verification.nli_verifier import nliverifier
from .verification.entity_guard import entityguard

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class anchorsumpipeline:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        nli_model_name: str = "cross-encoder/nli-deberta-v3-large",
        entity_model_name: str = "en_core_web_trf",
        token: Optional[str] = None,
        max_revisions: int = 1,
        nli: bool = True,
        entity: bool = True,
        revision: bool = True
    ):
        self.max_revisions = max_revisions
        self.use_nli = nli
        self.use_entity = entity
        self.use_revision = revision

        logger.info("Initializing AnchorSum Pipeline Components...")
        
        self.summarizer = singlepassmds(model_name=model_name, token=token)
        
        self.nli_verifier = None
        if self.use_nli:
            self.nli_verifier = nliverifier(model_name=nli_model_name, token=token)
            
        self.entity_guard = None
        if self.use_entity:
            self.entity_guard = entityguard(model_name=entity_model_name)
            
        logger.info("Pipeline initialized successfully.")

    def process(self, document: str, reference_summary: Optional[str] = None) -> Dict[str, Any]:
        anchors = []
        if self.use_entity and self.entity_guard:
            anchors = self.entity_guard.extract_anchors(document)
            
        initial_draft = self.summarizer.generate_draft(document, anchors=anchors)
        
        current_summary = initial_draft
        history = [{"revision": 0, "summary": initial_draft, "flags": []}]
        
        if self.use_revision:
            for i in range(self.max_revisions):
                flags = []
                
                if self.use_nli and self.nli_verifier:
                    nli_flags = self.nli_verifier.verify(document, current_summary)
                    flags.extend(nli_flags)
                    
                if self.use_entity and self.entity_guard:
                    entity_flags = self.entity_guard.verify_coverage(document, current_summary)
                    flags.extend(entity_flags)
                
                if not flags:
                    logger.info(f"No errors found in iteration {i+1}. Stopping revision.")
                    break
                    
                logger.info(f"Revision {i+1}: Found {len(flags)} issues. Revise...")
                revised_summary = self.summarizer.revise_draft(document, current_summary, flags)
                current_summary = revised_summary
                history.append({"revision": i + 1, "summary": current_summary, "flags": flags})
        
        return {
            "initial_draft": initial_draft,
            "final_summary": current_summary,
            "reference": reference_summary,
            "history": history,
            "num_revisions": len(history) - 1
        }