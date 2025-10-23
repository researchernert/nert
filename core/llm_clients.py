import os
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def _call_with_retry(self, api_call_func, retryable_exceptions: tuple, max_retries: int = 3):
        for attempt in range(max_retries):
            try:
                return api_call_func()
            except Exception as e:
                is_retryable = isinstance(e, retryable_exceptions)

                if is_retryable and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(
                        f"{self.provider_name} API retry {attempt + 1}/{max_retries} "
                        f"after {wait_time}s: {type(e).__name__}: {e}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    raise

    @abstractmethod
    def call(self, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

class OpenAIClient(BaseLLMClient):

    def __init__(self, model_name: str = "gpt-4o", api_key: str = None):
        api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key is required")
        super().__init__(model_name, api_key)

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, timeout=30.0)
        except ImportError:
            raise ImportError("OpenAI library not installed. Run: pip install openai")

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    def call(self, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        temperature = kwargs.pop('temperature', 0.7)

        try:
            from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
            retryable_exceptions = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
        except ImportError:
            retryable_exceptions = (Exception,)

        def api_call():
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens or 4000,
                    temperature=temperature,
                    **kwargs
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI API call failed: {e}")
                raise Exception(f"OpenAI API error: {str(e)}")

        return self._call_with_retry(api_call, retryable_exceptions)

class GeminiClient(BaseLLMClient):

    def __init__(self, model_name: str = "gemini-pro", api_key: str = None):
        api_key = api_key or os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Google API key is required")
        super().__init__(model_name, api_key)

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model_name)
        except ImportError:
            raise ImportError("Google GenerativeAI library not installed. Run: pip install google-generativeai")

    @property
    def provider_name(self) -> str:
        return "Google"

    def call(self, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        temperature = kwargs.pop('temperature', 0.7)
        top_p = kwargs.pop('top_p', 0.9)
        top_k = kwargs.pop('top_k', 40)

        generation_config = {
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
        }

        if max_tokens:
            generation_config['max_output_tokens'] = max_tokens

        try:
            from google.api_core.exceptions import (
                ResourceExhausted, DeadlineExceeded, ServiceUnavailable, InternalServerError
            )
            retryable_exceptions = (ResourceExhausted, DeadlineExceeded, ServiceUnavailable, InternalServerError)
        except ImportError:
            retryable_exceptions = (Exception,)

        def api_call():
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config=generation_config
                )

                if response.candidates:
                    candidate = response.candidates[0]
                    finish_reason = candidate.finish_reason

                    if finish_reason == 2:  
                        logger.warning(f"Gemini blocked response due to safety filters")
                        return "GEMINI_SAFETY_BLOCK: The request was blocked by Gemini's safety filters."

                    if finish_reason == 3: 
                        logger.warning(f"Gemini blocked response due to recitation concerns")
                        return "GEMINI_RECITATION_BLOCK: The request may violate copyright policies."

                return response.text
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}")
                raise Exception(f"Gemini API error: {str(e)}")

        return self._call_with_retry(api_call, retryable_exceptions)

class AnthropicClient(BaseLLMClient):

    def __init__(self, model_name: str = "claude-3-opus-20240229", api_key: str = None):
        api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key is required")
        super().__init__(model_name, api_key)

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic library not installed. Run: pip install anthropic")

    @property
    def provider_name(self) -> str:
        return "Anthropic"

    def call(self, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        temperature = kwargs.pop('temperature', 0.7)

        try:
            from anthropic import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
            retryable_exceptions = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
        except ImportError:
            retryable_exceptions = (Exception,)

        def api_call():
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens or 4000,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                    **kwargs
                )
                return response.content[0].text
            except Exception as e:
                logger.error(f"Anthropic API call failed: {e}")
                raise Exception(f"Anthropic API error: {str(e)}")

        return self._call_with_retry(api_call, retryable_exceptions)

class LLMClientFactory:

    MODEL_REGISTRY = {

        'gpt-4o': {
            'provider': 'openai',
            'model_name': 'gpt-4o',
            'display_name': 'GPT-4O',
            'description': 'Latest OpenAI model with vision capabilities'
        },
        'gpt-4-turbo': {
            'provider': 'openai',
            'model_name': 'gpt-4-turbo',
            'display_name': 'GPT-4 Turbo',
            'description': 'Fast and capable OpenAI model'
        },
        'gpt-3.5-turbo': {
            'provider': 'openai',
            'model_name': 'gpt-3.5-turbo',
            'display_name': 'GPT-3.5 Turbo',
            'description': 'Fast and cost-effective OpenAI model'
        },

        'gemini-2.0-flash': {
            'provider': 'google',
            'model_name': 'gemini-2.0-flash',
            'display_name': 'Gemini 2.0 Flash',
            'description': 'Google\'s fast and efficient model'
        },
        'gemini-2.5-pro': {
            'provider': 'google',
            'model_name': 'gemini-2.5-pro',
            'display_name': 'Gemini 2.5 Pro',
            'description': 'Google\'s advanced model with enhanced capabilities'
        },
        'gemini-1.5-pro-002': {
            'provider': 'google',
            'model_name': 'gemini-1.5-pro-002',
            'display_name': 'Gemini 1.5 Pro 002 (offline Sept 24-\'25)',
            'description': 'Google\'s advanced model'
        },

        'claude-3-opus': {
            'provider': 'anthropic',
            'model_name': 'claude-3-opus-20240229',
            'display_name': 'Claude 3 Opus',
            'description': 'Anthropic\'s most capable model'
        },
        'claude-3-sonnet': {
            'provider': 'anthropic',
            'model_name': 'claude-3-sonnet-20240229',
            'display_name': 'Claude 3 Sonnet',
            'description': 'Balanced performance and speed'
        }
    }

    @classmethod
    def create_client(cls, model_id: str, api_key: str = None) -> BaseLLMClient:
        if model_id not in cls.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_id}. Available models: {list(cls.MODEL_REGISTRY.keys())}")

        model_info = cls.MODEL_REGISTRY[model_id]
        provider = model_info['provider']
        model_name = model_info['model_name']

        try:
            if provider == 'openai':
                return OpenAIClient(model_name, api_key)
            elif provider == 'google':
                return GeminiClient(model_name, api_key)
            elif provider == 'anthropic':
                return AnthropicClient(model_name, api_key)
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to create {provider} client: {e}")
            raise

    @classmethod
    def get_available_models(cls) -> Dict[str, Dict[str, Any]]:
        return cls.MODEL_REGISTRY.copy()

    @classmethod
    def check_api_keys(cls) -> Dict[str, bool]:
        return {
            'openai': bool(os.environ.get('OPENAI_API_KEY')),
            'google': bool(os.environ.get('GOOGLE_API_KEY')),
            'anthropic': bool(os.environ.get('ANTHROPIC_API_KEY'))
        }