"""High-level retrieval API combining embedding and vector search.

This module exposes `Retriever`, a small orchestration layer that
encodes a user query using dense and sparse embedding services and
delegates nearest-neighbour lookup to a `VectorStore` implementation.
"""

from src.core.search import SearchResult

from src.embeddings.base import DenseEmbeddingService, SparseEmbeddingService


from src.vectordb.base import VectorStore


class Retriever:
    """End-to-end semantic search over an indexed document corpus.

    The class intentionally keeps responsibilities minimal: embed the
    query and call the vector store. Re-ranking or result aggregation
    can be implemented by callers if needed.
    """

    def __init__(
        self,
        dense_embedding_service: DenseEmbeddingService,
        sparse_embedding_service: SparseEmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        """Wire together embedding and storage dependencies.

        Args:
            dense_embedding_service: Model used to create dense query vectors.
            sparse_embedding_service: Model used to create sparse query vectors.
            vector_store: Backend that searches pre-indexed chunk vectors.
        """
        self.dense_embedding_service = dense_embedding_service
        self.sparse_embedding_service = sparse_embedding_service
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Find chunks most relevant to a natural-language query.

        Validates inputs, builds dense and sparse query embeddings, then
        calls the vector store to obtain ranked hits.
        """
        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Build both sparse and dense query representations.
        sparse_query = self.sparse_embedding_service.embed_query(query)
        dense_query = self.dense_embedding_service.embed_query(query)

        # Delegate the nearest-neighbour search to the vector store.
        search_result = self.vector_store.search(
            dense_query_vector=dense_query,
            sparse_query_vector=sparse_query,
            top_k=top_k,
        )

        return search_result
