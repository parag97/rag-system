"""Abstract interface for vector database backends."""

from abc import ABC, abstractmethod

from src.core.search import SearchResult
from src.vectordb.models import VectorPoint


class VectorStore(ABC):
    """Protocol for storing and searching dense document embeddings."""

    @abstractmethod
    def create_collection(self, dimension: int) -> None:
        """Create the backing collection if it does not already exist.

        Args:
            dimension: Length of embedding vectors stored in the collection.
        """
        pass

    @abstractmethod
    def upsert(self, points: list[VectorPoint]) -> None:
        """Insert or update vector points and their metadata payloads.

        Args:
            points: Points to write, each with an ID, vector, and chunk payload.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        """Find the ``top_k`` chunks most similar to ``query_vector``.

        Args:
            query_vector: Query embedding with length matching the collection.
            top_k: Maximum number of results to return.

        Returns:
            Ranked search hits with scores and chunk metadata.
        """
        pass
