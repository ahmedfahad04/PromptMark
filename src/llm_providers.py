"""
LLM Provider Strategy Pattern
=============================

Abstraction layer for multiple LLM providers (Claude, Gemini, etc.)
Enables easy switching between providers without modifying experiment code.

Usage:
    provider = LLMProviderFactory.create("claude")
    response = provider.generate(prompt, max_tokens=2048, temperature=0.1)
    
    # Or switch to Gemini:
    provider = LLMProviderFactory.create("gemini")
    response = provider.generate(prompt, max_tokens=2048, temperature=0.1)
"""

from abc import ABC, abstractmethod
from typing import Dict, Union, Optional
import os
import json
import re
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMResponse:
    """Unified LLM response structure."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    
    def to_dict(self) -> Dict:
        """Convert response to dictionary."""
        return {
            "text": self.text,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize provider with API key."""
        self.api_key = api_key
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt/query
            system_prompt: System instruction (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 - 1.0)
            
        Returns:
            LLMResponse with text and token counts
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate that connection to LLM is working."""
        pass


class ClaudeProvider(LLMProvider):
    """AWS Bedrock Claude implementation."""
    
    def __init__(self, api_key: Optional[str] = None, region: str = "us-east-1"):
        """
        Initialize Claude provider.
        
        Args:
            api_key: Unused for boto3 (uses AWS credentials from environment)
            region: AWS region for Bedrock service
        """
        super().__init__(api_key)
        self.region = region
        # Try explicit credentials first, then fall back to boto3's default chain
        # (which handles ~/.aws/credentials, IAM roles, environment variables, etc.)
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self._client = None
        self._model_id = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize boto3 Bedrock client."""
        try:
            import boto3

            # If credentials are in environment, use them explicitly
            # Otherwise boto3 will use its default credential chain
            if self.aws_access_key and self.aws_secret_key:
                self._client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.region,
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                )
            else:
                # Use boto3's default credential chain
                # (checks ~/.aws/credentials, environment, IAM roles, etc.)
                self._client = boto3.client(
                    "bedrock-runtime",
                    region_name=self.region,
                )
            
            self._model_id = os.getenv(
                "DEFAULT_MODEL",
                "us.anthropic.claude-sonnet-4-20250514-v1:0"
            )
        except ImportError:
            raise ImportError("boto3 is required for Claude provider. Install with: pip install boto3")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Bedrock client: {e}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Generate response from Claude via Bedrock."""
        if not self._client:
            raise RuntimeError("Bedrock client not initialized")
        
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            
            if system_prompt:
                request_body["system"] = system_prompt
            
            response = self._client.invoke_model(
                modelId=self._model_id,
                body=json.dumps(request_body),
                contentType="application/json",
            )
            
            response_body = json.loads(response["body"].read())
            text = ""
            if "content" in response_body and len(response_body["content"]) > 0:
                text = response_body["content"][0]["text"]
            
            input_tokens = response_body.get("usage", {}).get("input_tokens", 0)
            output_tokens = response_body.get("usage", {}).get("output_tokens", 0)
            
            return LLMResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        
        except Exception as e:
            error_str = str(e)
            if "UnrecognizedClientException" in error_str or "invalid" in error_str.lower():
                raise RuntimeError(
                    f"Claude authentication failed: {e}\n"
                    "Fix: Set AWS credentials via environment variables:\n"
                    "  export AWS_ACCESS_KEY_ID='your-key'\n"
                    "  export AWS_SECRET_ACCESS_KEY='your-secret'\n"
                    "Or use .env file: Create .env with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
                )
            else:
                raise RuntimeError(f"Claude generation failed: {e}")
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Claude typically uses ~4 chars per token on average
        return len(text) // 4
    
    def validate_connection(self) -> bool:
        """Validate Bedrock connection."""
        try:
            if not self._client:
                return False
            # Try to list available models
            self._client.list_foundation_models()
            return True
        except Exception:
            return False


class GeminiProvider(LLMProvider):
    """Google Gemini API implementation."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Google Gemini API key (from environment if not provided)
        """
        super().__init__(api_key)
        self._client = None
        self._model = "gemini-2.5-flash"
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Google Gemini client."""
        try:
            from google import genai
            from google.genai import types
            self.genai = genai
            self.types = types
            
            # Get API key from parameter or environment
            api_key = self.api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not provided and not in environment")
            
            self._client = genai.Client(api_key=api_key)
        except ImportError:
            raise ImportError(
                "google-genai is required for Gemini provider. "
                "Install with: pip install google-genai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {e}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        """Generate response from Gemini API."""
        if not self._client:
            raise RuntimeError("Gemini client not initialized")
        
        try:
            contents = [
                self.types.Content(
                    role="user",
                    parts=[self.types.Part.from_text(text=prompt)],
                )
            ]
            
            config = self.types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            if system_prompt:
                config.system_instruction = system_prompt
            
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
            
            text = response.text.strip() if response.text else ""
            
            # Use actual token counts from response metadata
            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            
            return LLMResponse(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini API token counter."""
        try:
            # Use Gemini's token counting API
            response = self._client.models.count_tokens(
                model=self._model,
                contents=self.types.Content(
                    role="user",
                    parts=[self.types.Part.from_text(text=text)],
                ),
            )
            return response.total_tokens
        except Exception:
            # Fallback to estimation
            return self._estimate_tokens(text)
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count (Gemini models: ~4-5 chars per token avg)."""
        return max(1, len(text) // 4)
    
    def validate_connection(self) -> bool:
        """Validate Gemini connection."""
        try:
            if not self._client:
                return False
            # Try a minimal token count request
            self._client.models.count_tokens(
                model=self._model,
                contents=self.types.Content(
                    role="user",
                    parts=[self.types.Part.from_text(text="test")],
                ),
            )
            return True
        except Exception:
            return False


class GroqProvider(LLMProvider):
    """Groq API implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mixtral-8x7b-32768"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self._model = model
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not provided")
        try:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError("Install: pip install groq")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        return LLMResponse(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
    
    def count_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def validate_connection(self) -> bool:
        try:
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10,
            )
            return True
        except Exception:
            return False


class CerebrasProvider(LLMProvider):
    """Cerebras API implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b"):
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        self._model = model
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY not provided")
        try:
            from cerebras.cloud.sdk import Cerebras
            self._client = Cerebras(api_key=self.api_key)
        except ImportError:
            raise ImportError("Install: pip install cerebras-cloud-sdk")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        
        return LLMResponse(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
    
    def count_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def validate_connection(self) -> bool:
        try:
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "test"}],
                max_completion_tokens=10,
            )
            return True
        except Exception:
            return False


class AISuiteProvider(LLMProvider):
    """AISuite unified API implementation."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "groq:mixtral-8x7b-32768"):
        self.api_key = api_key
        self._model = model
        try:
            import aisuite as ai
            self._client = ai.Client()
        except ImportError:
            raise ImportError("Install: pip install aisuite")
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        
        return LLMResponse(
            text=response.choices[0].message.content,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
    
    def count_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def validate_connection(self) -> bool:
        try:
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "test"}],
            )
            return True
        except Exception:
            return False


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""
    
    _providers = {
        "claude": ClaudeProvider,
        "gemini": GeminiProvider,
        "groq": GroqProvider,
        "cerebras": CerebrasProvider,
        "aisuite": AISuiteProvider,
    }
    
    @classmethod
    def create(
        cls,
        provider_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> LLMProvider:
        """
        Create an LLM provider instance.
        
        Args:
            provider_name: Name of provider ("claude", "gemini")
            api_key: Optional API key (provider-specific)
            **kwargs: Additional provider-specific arguments
            
        Returns:
            LLMProvider instance
            
        Raises:
            ValueError: If provider name is unknown
        """
        provider_class = cls._providers.get(provider_name.lower())
        
        if not provider_class:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown provider '{provider_name}'. Available: {available}"
            )
        
        try:
            return provider_class(api_key=api_key, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create {provider_name} provider: {e}"
            )
    
    @classmethod
    def register(cls, name: str, provider_class: type):
        """
        Register a new provider.
        
        Args:
            name: Name to register provider under
            provider_class: Provider class (must inherit from LLMProvider)
        """
        if not issubclass(provider_class, LLMProvider):
            raise TypeError(f"{provider_class} must inherit from LLMProvider")
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available provider names."""
        return list(cls._providers.keys())


# Convenience function for backward compatibility
def get_llm_provider(
    provider_name: str = "claude",
    api_key: Optional[str] = None,
) -> LLMProvider:
    """
    Get an LLM provider instance.
    
    Args:
        provider_name: Name of provider ("claude" or "gemini")
        api_key: Optional API key
        
    Returns:
        Initialized LLMProvider instance
    """
    return LLMProviderFactory.create(provider_name, api_key=api_key)


if __name__ == "__main__":
    # Example usage and testing
    print("LLM Provider Strategy Pattern")
    print("=" * 50)
    
    # Show available providers
    available = LLMProviderFactory.get_available_providers()
    print(f"Available providers: {', '.join(available)}")
    
    # Example: Create Claude provider
    try:
        provider = LLMProviderFactory.create("claude")
        print(f"✓ Claude provider initialized")
        response = provider.generate('Write one line about ramadan')
        print(f"  Response: {response.text}")
        print(f"  Connection valid: {provider.validate_connection()}")
    except Exception as e:
        print(f"✗ Claude provider error: {e}")
    
    # Example: Create Gemini provider
    try:
        provider = LLMProviderFactory.create("gemini")
        print(f"✓ Gemini provider initialized")
        response = provider.generate('Write one line about ramadan')
        print(f"  Response: {response.text}")    
        print(f"  Connection valid: {provider.validate_connection()}")
    except Exception as e:
        print(f"✗ Gemini provider error: {e}")
