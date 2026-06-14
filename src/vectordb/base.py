"""Abstract interface and types for vector database backends.

Concrete adapters (Qdrant, Faiss, etc.) should implement `VectorStore` to
provide a consistent upsert/search API used by the rest of the application.
The interface focuses on the minimal operations required by the ingestion
and retrieval pipelines: creating collections, writing points, and
performing nearest-neighbour lookups.
"""

from abc import ABC, abstractmethod

from src.core.search import SearchResult
from src.embeddings.model import SparseEmbedding
from src.vectordb.models import VectorPoint


class VectorStore(ABC):
    """Protocol for storing and searching document embeddings.

    Implementations are expected to handle vector creation, efficient
    upserts, and returning domain `SearchResult` objects for queries.
    """

    @abstractmethod
    def create_collection(self, dimension: int) -> None:
        """Create the backing collection if it does not already exist.

        Args:
            dimension: Length of embedding vectors stored in the collection.
        """
        # Concrete implementations should provision schema, vector params,
        # or index structures required by the backend.
        pass

    @abstractmethod
    def upsert(self, points: list[VectorPoint]) -> None:
        """Insert or update vector points and their metadata payloads.

        Args:
            points: Points to write, each with an ID, vector, and chunk payload.
        """
        # Implementations should batch writes for efficiency where possible.
        pass

    @abstractmethod
    def search(self, dense_query_vector: list[float], sparse_query_vector: SparseEmbedding) -> list[SearchResult]:
        """Return the ``top_k`` chunks most similar to the query vectors.

        Args:
            dense_query_vector: Query embedding with length matching the collection.
            sparse_query_vector: Sparse query embedding.
            top_k: Maximum number of results to return.

        Returns:
            Ranked search hits with scores and chunk metadata.
        """
        # Backends should map storage-specific results into `SearchResult`.
        pass

    @abstractmethod
    def get_chunks_by_range(
        self,document_id: str,
        start_chunk_index: int,
        end_chunk_index: int
        ) -> list[SearchResult]:
        
        pass
