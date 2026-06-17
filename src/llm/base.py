"""Abstract base class for language model backends."""

from abc import ABC, abstractmethod


class LLM(ABC):
    """Abstract base for language model implementations.

    Implementations wrap specific LLM providers (OpenAI, Google, etc.)
    and expose a unified interface for response generation.
    """

    @abstractmethod
    def get_llm(self):
        """Retrieve the underlying language model instance.

        Returns:
            Initialized language model ready for inference.
        """
        pass
