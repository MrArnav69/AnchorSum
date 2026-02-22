import logging
from typing import List, Dict, Any, Tuple
from .llm_summarizer import SinglePassMDS
from .verification.nli_verifier import NLIVerifier
from .verification.entity_guard import EntityGuard

logger = logging.getLogger(__name__)

class AnchorSumPipeline:
    def __init__(
        self, 
        model_name: str = "HuggingFaceTB/SmolLM-1.7B-Instruct",
        nli_model_name: str = "cross-encoder/nli-deberta-v3-base",
        entity_model_name: str = "en_core_web_sm",
        device: str = "auto",
        load_in_4bit: bool = False,
        max_revisions: int = 1,
        token: str = None,
        nli: bool = True,
        entity: bool = True,
        revision: bool = True
    ):
        """
        Initializes the full AnchorSum Draft-Audit-Revise pipeline.
        """
        logger.info("Initializing AnchorSum Pipeline...")
        self.max_revisions = max_revisions if revision else 0
        self.nli_enabled = nli
        self.entity_enabled = entity
        
        self.llm = SinglePassMDS(model_name=model_name, device=device, load_in_4bit=load_in_4bit, token=token)
        self.nli_verifier = NLIVerifier(model_name=nli_model_name, device=device) if nli else None
        self.entity_guard = EntityGuard(model_name=entity_model_name, top_n=15) if entity else None
        
        logger.info("Pipeline Ready.")

    def summarize(self, documents: str) -> Dict[str, Any]:
        """
        Executes the full pipeline on a given set of concatenated source documents.
        """
        # Step 1: Pre-generation Anchor Extraction
        anchors = []
        if self.entity_enabled:
            logger.info("Extracting Anchors from source text...")
            anchors = self.entity_guard.extract_anchors(documents)
            logger.info(f"Extracted {len(anchors)} primary anchors.")
        
        # Step 2: Draft Generation
        logger.info("Generating Draft 1...")
        current_draft = self.llm.generate_draft(documents, anchors=anchors)
        
        revision_history = [{"draft": current_draft, "flags": []}]
        
        # Step 3 & 4: The Audit & Revise Loop
        for iteration in range(self.max_revisions):
            logger.info(f"--- Audit/Revise Loop {iteration + 1} ---")
            
            all_flags = []
            
            # 3A. NLI Audit
            if self.nli_enabled:
                logger.info("Running NLI Hallucination Verification...")
                passed_sents, nli_flags = self.nli_verifier.verify_draft(documents, current_draft)
                all_flags.extend(nli_flags)
            
            # 3B. Entity Coverage & Hallucination Audit
            if self.entity_enabled:
                logger.info("Running Entity Coverage & Guard Verification...")
                entity_flags = self.entity_guard.verify_draft(anchors, documents, current_draft)
                all_flags.extend(entity_flags)
            
            if not all_flags:
                logger.info("Audit Passed cleanly. No revisions needed.")
                break
                
            logger.info(f"Audit failed with {len(all_flags)} flags. Passing to LLM Editor...")
            for f in all_flags:
                logger.debug(f)
                
            # 4. Revise Draft
            logger.info("LLM is revising the draft based on audit flags...")
            revised_draft = self.llm.revise_draft(documents, current_draft, all_flags)
            
            # Save history
            revision_history[-1]["flags"] = all_flags
            current_draft = revised_draft
            revision_history.append({"draft": current_draft, "flags": []})
            
        logger.info("Summarization pipeline complete.")
        
        return {
            "final_summary": current_draft,
            "anchors": anchors,
            "revisions": len(revision_history) - 1,
            "history": revision_history
        }

    def process(self, document: str, reference_summary: str = None) -> Dict[str, Any]:
        """
        Process a document through the pipeline and return results.
        """
        result = self.summarize(document)
        
        # Add reference summary if provided
        if reference_summary:
            result["reference_summary"] = reference_summary
            
        return result
