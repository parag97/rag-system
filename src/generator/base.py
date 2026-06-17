"""Abstract base classes for prompt and response generation."""

from abc import ABC, abstractmethod
from langchain_core.prompts import ChatPromptTemplate

from src.context.model import Context


class promptGenerator(ABC):
    """Abstract base for prompt generators.

    Prompt generators transform context and queries into formatted
    prompts suitable for LLM input.
    """

    @abstractmethod
    def generate_prompt(
        self,
        context: Context,
        query: str,
    ) -> ChatPromptTemplate:
        """Generate a prompt template for the LLM.

        Args:
            context: Assembled context relevant to the query.
            query: User's original natural-language question.

        Returns:
            ChatPromptTemplate ready for LLM invocation.
        """
        pass


class responseGenerator(ABC):
    """Abstract base for response generators.

    Response generators orchestrate prompt creation, LLM invocation,
    and output parsing to produce final answers.
    """

    @abstractmethod
    def generate_response(
        self,
        query: str,
        context: Context,
    ) -> str:
        """Generate an LLM response for a query with context.

        Args:
            query: User's original natural-language question.
            context: Assembled context relevant to the query.

        Returns:
            Generated response text from the LLM.
        """
        pass
