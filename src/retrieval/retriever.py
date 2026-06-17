"""High-level retrieval orchestration combining embeddings and vector search."""

from src.core.search import SearchResult
from src.embeddings.base import DenseEmbeddingService, SparseEmbeddingService
from src.vectordb.base import VectorStore
from src.reranking.base import ReRanker


class Retriever:
    """End-to-end semantic search over an indexed document corpus.

    This orchestration layer coordinates:
    1. Query embedding (dense + sparse)
    2. Vector store nearest-neighbor search
    3. Result re-ranking for quality improvement

    By separating these concerns, callers can easily extend the retrieval
    pipeline without modifying the core embedding or storage layers.

    Attributes:
        dense_embedding_service: Dense vector encoder for queries.
        sparse_embedding_service: Sparse vector encoder for queries.
        vector_store: Backend for vector search and storage.
        re_ranker: Optional re-ranking module for result refinement.
    """

    def __init__(
        self,
        dense_embedding_service: DenseEmbeddingService,
        sparse_embedding_service: SparseEmbeddingService,
        vector_store: VectorStore,
        re_ranker: ReRanker,
    ):
        """Wire together embedding and storage dependencies.

        Args:
            dense_embedding_service: Model for creating dense query vectors.
            sparse_embedding_service: Model for creating sparse query vectors.
            vector_store: Backend for searching pre-indexed vectors.
            re_ranker: Post-processing module to refine top results.
        """
        self.dense_embedding_service = dense_embedding_service
        self.sparse_embedding_service = sparse_embedding_service
        self.vector_store = vector_store
        self.re_ranker = re_ranker

    def retrieve(self, query: str) -> list[SearchResult]:
        """Find chunks most relevant to a natural-language query.

        Executes the full retrieval pipeline:
        1. Validates query is non-empty
        2. Encodes query into dense and sparse vectors
        3. Performs vector search using reciprocal rank fusion
        4. Re-ranks results for improved relevance
        5. Returns top results

        Args:
            query: Natural-language search query.

        Returns:
            List of SearchResult objects ordered by relevance.

        Raises:
            ValueError: If query is empty or validation fails.
        """
        # Validate query
        if not query.strip():
            raise ValueError("Query must not be empty.")

        # Encode query into both dense and sparse vector representations
        sparse_query = self.sparse_embedding_service.embed_query(query)
        dense_query = self.dense_embedding_service.embed_query(query)

        # Search vector store using both representations
        search_results = self.vector_store.search(
            dense_query_vector=dense_query,
            sparse_query_vector=sparse_query,
        )

        # Re-rank results using cross-encoder for final quality pass
        final_results = self.re_ranker.re_rank(search_results, query)

        return final_results

