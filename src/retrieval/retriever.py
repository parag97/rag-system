"""High-level retrieval API combining embedding and vector search."""

from src.core.search import SearchResult

from src.embeddings.base import DenseEmbeddingService, SparseEmbeddingService


from src.vectordb.base import VectorStore


class Retriever:
    """End-to-end semantic search over an indexed document corpus.

    Embeds the user query with an :class:`~src.embeddings.base.EmbeddingService`
    and delegates nearest-neighbor lookup to a
    :class:`~src.vectordb.base.VectorStore`.
    """

    def __init__(
        self,
        dense_embedding_service: DenseEmbeddingService,
        sparse_embeddig_service: SparseEmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        """Wire together embedding and storage dependencies.

        Args:
            embedding_service: Model used to embed the query string.
            vector_store: Backend that searches pre-indexed chunk vectors.
        """
        self.dense_embedding_service = dense_embedding_service
        self.sparse_embeddig_service = sparse_embeddig_service
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Find chunks most relevant to a natural-language query.

        Args:
            query: User question or search phrase.
            top_k: Maximum number of chunks to return.

        Returns:
            Ranked :class:`~src.core.search.SearchResult` objects from the
            vector store.
        """
        if not query.strip():
            raise ValueError("query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        sparse_query = self.sparse_embeddig_service.embed_query(query)
        dense_query = self.dense_embedding_service.embed_query(query)

        search_result = self.vector_store.search(
            dense_query_vector = dense_query,
            sparse_query_vector = sparse_query, 
            top_k=top_k,
        )
        
        
        return search_result
