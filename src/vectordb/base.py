"""Abstract interface and types for vector database backends.

Concrete adapters (Qdrant, Faiss, etc.) implement `VectorStore` to provide
a consistent upsert/search API. The interface focuses on minimal operations
required by ingestion and retrieval: collection creation, point insertion,
and nearest-neighbor search.
"""

from abc import ABC, abstractmethod

from src.core.search import SearchResult
from src.embeddings.model import SparseEmbedding
from src.vectordb.models import VectorPoint


class VectorStore(ABC):
    """Protocol for storing and searching document embeddings.

    Implementations handle vector storage, efficient upserts, and return
    domain SearchResult objects for queries.
    """

    @abstractmethod
    def create_collection(self, dimension: int) -> None:
        """Create or validate the target collection.

        Args:
            dimension: Length of embedding vectors in the collection.

        Raises:
            ValueError: If collection exists with mismatched schema.
        """
        pass

    @abstractmethod
    def upsert(self, points: list[VectorPoint]) -> None:
        """Insert or update vector points with metadata.

        Args:
            points: Points to write, each with ID, vectors, and payload.
        """
        pass

    @abstractmethod
    def search(
        self,
        dense_query_vector: list[float],
        sparse_query_vector: SparseEmbedding,
    ) -> list[SearchResult]:
        """Return chunks most similar to query vectors.

        Args:
            dense_query_vector: Query as dense embedding vector.
            sparse_query_vector: Query as sparse embedding.

        Returns:
            Ranked search results ordered by relevance.
        """
        pass

    @abstractmethod
    def get_chunks_by_range(
        self,
        document_id: str,
        start_chunk_index: int,
        end_chunk_index: int,
    ) -> list[SearchResult]:
        """Retrieve chunks for a document within chunk index range.

        Args:
            document_id: Source document identifier.
            start_chunk_index: Starting chunk index (inclusive).
            end_chunk_index: Ending chunk index (exclusive).

        Returns:
            SearchResults for chunks in the specified range.
        """
        pass

    @abstractmethod
    def delete_document(self, source_file: str) -> None:
        """Delete all chunks belonging to a document.

        Args:
            source_file: Document identifier to delete.
        """
        pass

