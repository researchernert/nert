# nert/utils/llm_client.py
"""LLM client wrapper for multiple providers."""

import os
from pathlib import Path
from typing import Optional
from openai import OpenAI  # Updated import
import google.generativeai as genai
from dotenv import load_dotenv
import logging

# Load .env file
load_dotenv()

logger = logging.getLogger(__name__)

class LLMClient:
    """Unified interface for LLM providers."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.setup_client()
    
    def setup_client(self):
        """Setup appropriate LLM client."""
        if "gpt" in self.model:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            self.openai_client = OpenAI(api_key=api_key)
        elif "gemini" in self.model:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found")
            genai.configure(api_key=api_key)


    def call(self, prompt: str, temperature: float = 0.1, max_tokens: int = 2000) -> str:
        """
        Make LLM API call with proper encoding handling.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            
        Returns:
            The LLM's response as a string
            
        Raises:
            Exception: If the LLM call fails after retries
        """

        if "SOCIETAL ALIGNMENT" in prompt and "ORGANIZATIONAL ALIGNMENT" in prompt:
            max_tokens = min(max_tokens, 500)
            logger.debug(f"Safety check detected, reducing max_tokens to {max_tokens}")
        
        # Unicode character replacements
        char_replacements = {
            '\u2113': 'l',  
            '\u2208': 'in',  
            '\u2200': 'forall',  
            '\u2203': 'exists',  
            '\u2227': 'and',  
            '\u2228': 'or',   
            '\u2192': '->',   
            '\u00AC': 'not', 
            '\u22A2': '|-',   
            '\u03BD': 'v',    
            '\u209B': 's',   
            '\u2092': 'o',    
            '\u1D62': 'i',    
            '\u2081': '1',    
            '\u2082': '2',    
            '\u2083': '3',    
            '\u2084': '4',    
            '\u2085': '5',    
            '\u03BD\u2081': 'v1',  
            '\u03BD\u2082': 'v2',  
            '\u2115': 'N',    
            '\u2124': 'Z',    
            '\u211D': 'R',    
            '\u2102': 'C',    
            '\u2026': '...',  
            # More math symbols from the prompt
            '∈': 'in',
            '∀': 'forall',
            '∃': 'exists',
            '∧': 'and',
            '∨': 'or',
            '→': '->',
            '¬': 'not',
            '⊢': '|-',
            'ν': 'v',
            'ℓ': 'l',
            '₁': '1',
            '₂': '2',
            '₃': '3',
            '₄': '4',
            '₅': '5',
            '…': '...'
        }

        clean_prompt = prompt
        for unicode_char, replacement in char_replacements.items():
            clean_prompt = clean_prompt.replace(unicode_char, replacement)

        try:
            clean_prompt = clean_prompt.encode('utf-8', 'replace').decode('utf-8')
            clean_prompt = clean_prompt.encode('ascii', 'ignore').decode('ascii')
        except:
            import re
            clean_prompt = re.sub(r'[^\x00-\x7F]+', ' ', clean_prompt)

        try:
            if "gpt" in self.model:
                
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": clean_prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if response.choices and response.choices[0].message:
                    return response.choices[0].message.content
                else:
                    raise Exception("Empty response from OpenAI API")
                    
            elif "gemini" in self.model:
                
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(
                    clean_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature
                    )
                )
                
                if response.text:
                    return response.text
                else:
                    raise Exception("Empty response from Gemini API")
            else:
                raise ValueError(f"Unsupported model: {self.model}")
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise Exception(f"LLM API call failed: {str(e)}. Please check your API key and connection.")