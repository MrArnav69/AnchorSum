import torch
import logging
from typing import List, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SinglePassMDS:
    def __init__(
        self, 
        model_name: str = "",
        device: str = "auto",
        load_in_4bit: bool = False,
        token: Optional[str] = None
    ):
        self.model_name = model_name
        self.token = token
        
        # Enforce Full FP32 Precision as per user request for maximum scientific fidelity
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        logger.info(f"Full Precision (FP32) enforced for {model_name} on {self.device}")

        logger.info(f"Initializing SinglePassMDS with model: {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=self.token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization_config = None
        if load_in_4bit and self.device == 'cuda':
            logger.info("Loading model in 4-bit precision via BitsAndBytes...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            quantization_config=quantization_config,
            device_map="auto" if load_in_4bit else self.device,
            token=self.token
            # Strict standard attention for raw scientific fidelity
        )
        self.model.eval()
        logger.info("Model loaded successfully.")

    def _format_initial_prompt(self, documents: str, anchors: Optional[List[str]] = None) -> str:
        """
        Formats the prompt for the initial Draft 1 generation.
        """
        system_msg = (
            "You are an expert investigative journalist. Your task is to write a highly detailed, "
            "comprehensive, narrative summary synthesizing the provided source documents. "
            "You must remain absolutely factually faithful to the sources. Do not hallucinate any "
            "names, numbers, or events not explicitly present."
        )
        
        user_msg = "Here are the source documents:\n\n" + documents + "\n\n"
        if anchors and len(anchors) > 0:
            user_msg += "You MUST include the following key facts and entities in your narrative:\n"
            for a in anchors:
                user_msg += f"- {a}\n"
            user_msg += "\n"
        
        user_msg += "Write the comprehensive journalistic summary now. Output ONLY the summary text. Do not include any introductory or concluding remarks like 'Here is the summary' or 'I have synthesized the documents'. Start immediately with the narrative summary:"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"

    def _format_revision_prompt(self, documents: str, draft: str, flags: List[str]) -> str:
        """
        Formats the prompt for the Revise phase, providing the LLM with the previous draft and specific errors.
        """
        system_msg = (
            "You are an expert editor. You have written a draft summary based on source documents, "
            "but an audit has flagged factual errors and omissions. You must revise the draft to fix these issues."
        )
        
        user_msg = "Source Documents:\n" + documents + "\n\n"
        user_msg += "Your Current Draft:\n" + draft + "\n\n"
        
        user_msg += "Audit Flags (Errors to Fix):\n"
        for i, flag in enumerate(flags):
            user_msg += f"{i+1}. {flag}\n"
            
        user_msg += "\nPlease provide the complete, revised summary that fixes all these issues while maintaining a flowing narrative. Output ONLY the revised summary text. Do not include any explanations, labels, or confirmation text. Start immediately with the revised text:"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return f"System: {system_msg}\n\nUser: {user_msg}\n\nAssistant:"

    def _generate(self, prompt: str, max_new_tokens: int = 1024) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False, # We want deterministic, highly factual synthesis
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]
        summary = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return summary.strip()

    def generate_draft(self, documents: str, anchors: Optional[List[str]] = None, max_new_tokens: int = 1024) -> str:
        """
        Generates the first Draft summary.
        """
        prompt = self._format_initial_prompt(documents, anchors)
        return self._generate(prompt, max_new_tokens=max_new_tokens)

    def revise_draft(self, documents: str, draft: str, flags: List[str], max_new_tokens: int = 1024) -> str:
        """
        Revises the draft based on audit flags.
        """
        prompt = self._format_revision_prompt(documents, draft, flags)
        return self._generate(prompt, max_new_tokens=max_new_tokens)