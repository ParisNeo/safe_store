# safe_store/processing/tokenizers.py
from typing import Dict, Any, List
from abc import ABC, abstractmethod
from safe_store.core.exceptions import ConfigurationError

class TokenizerWrapper(ABC):
    """An abstract base class for a standardized tokenizer interface."""
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        pass

class TikTokenWrapper(TokenizerWrapper):
    """A wrapper for tiktoken's Encoding object."""
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        # tiktoken's decode does not take extra arguments
        return self.tokenizer.decode(tokens)

class HuggingFaceTokenizerWrapper(TokenizerWrapper):
    """A wrapper for Hugging Face's tokenizer objects."""
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int]) -> str:
        # Hugging Face tokenizers use skip_special_tokens
        return self.tokenizer.decode(tokens, skip_special_tokens=True)


def get_tokenizer(config: Dict[str, Any]) -> TokenizerWrapper:
    """
    Loads and returns a wrapped tokenizer based on the provided configuration.
    """
    if not isinstance(config, dict) or "name" not in config:
        raise ValueError("Custom tokenizer configuration must be a dictionary with a 'name' key.")

    tokenizer_name = config["name"]

    if tokenizer_name == "tiktoken":
        try:
            import tiktoken
        except ImportError:
            raise ConfigurationError("The 'tiktoken' library is required. Please run: pip install tiktoken")
        
        model = config.get("model")
        if not model:
            raise ValueError("The 'tiktoken' tokenizer requires a 'model' key (e.g., 'cl100k_base').")
        
        try:
            tokenizer_instance = tiktoken.get_encoding(model)
            return TikTokenWrapper(tokenizer_instance)
        except Exception as e:
            raise ConfigurationError(f"Failed to load tiktoken encoding '{model}': {e}") from e

    else:
        raise ValueError(f"Unknown custom tokenizer name: '{tokenizer_name}'")