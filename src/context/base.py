"""Abstract base classes for context pipeline components."""

from abc import ABC, abstractmethod

from src.core.search import SearchResult
from src.context.model import Context
from src.vectordb.base import VectorStore


class ContextChunkAssembler(ABC):
    """Abstract base for assembling retrieved chunks into a context string.

    Implementations format and combine search results into a unified
    prompt-friendly context suitable for LLM input.
    """

    @abstractmethod
    def assemble_chunks(
        self,
        retrieved_chunks: list[SearchResult],
    ) -> Context:
        """Assemble retrieved chunks into unified context.

        Args:
            retrieved_chunks: List of search results to combine.

        Returns:
            Context object containing formatted text and chunk references.
        """
        pass


class ContextExpander(ABC):
    """Abstract base for expanding retrieved chunks with additional context.

    Expanders take initially retrieved chunks and add related chunks
    (e.g., neighbors in the same document) to provide richer context.
    """

    @abstractmethod
    def expand(
        self,
        chunks: list[SearchResult],
    ) -> list[SearchResult]:
        """Expand retrieved chunks with additional context.

        Args:
            chunks: Initial retrieved search results.

        Returns:
            Expanded list of chunks (original plus additional context).
        """
        pass


class ContextBuilder(ABC):
    """Abstract base for building final context from retrieved chunks.

    Builders orchestrate expansion and assembly into a complete context.
    """

    @abstractmethod
    def build(
        self,
        chunks: list[SearchResult],
    ) -> Context:
        """Build context from retrieved chunks.

        Args:
            chunks: Retrieved search results.

        Returns:
            Assembled context ready for LLM prompt.
        """
        pass
