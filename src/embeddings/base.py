"""Abstract interfaces for embedding text into dense and sparse vectors.

Implementers should provide efficient batch document encoding and a
single-query API used by the ingestion and retrieval pipelines.
"""

from abc import ABC, abstractmethod
from src.embeddings.model import SparseEmbedding


class SparseEmbeddingService(ABC):
    """Protocol for encoding documents and queries as sparse vectors.

    Concrete implementations should support batching for document
    embeddings and a single-query API for retrieval-time encoding.
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[SparseEmbedding]:
        """Encode document text for sparse indexing."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> SparseEmbedding:
        """Encode one query for sparse retrieval."""
        pass


class DenseEmbeddingService(ABC):
    """Protocol for models that encode text as floating-point vectors.

    The `dimension` property allows callers to provision vector stores
    with the correct vector size before inserting points.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of dimensions in vectors produced by this service."""
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode document text for indexing.

        Args:
            texts: Input strings to embed, in order.

        Returns:
            One vector per input string, each of length :attr:`dimension`.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Encode a user query for retrieval."""
        pass
