"""Qdrant vector database implementation."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FieldCondition,
    FilterSelector,
    MatchValue,
    PointStruct,
    Range,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from pydantic import ValidationError

from src.core.chunk import Chunk
from src.core.search import SearchResult
from src.embeddings.model import SparseEmbedding
from src.vectordb.base import VectorStore
from src.vectordb.models import VectorPoint


class QdrantVectorStore(VectorStore):
    """Qdrant-based vector store for hybrid (dense + sparse) similarity search.

    This implementation stores document chunks in Qdrant using both dense
    and sparse vector representations. Search uses reciprocal rank fusion
    to combine results from both indexes, providing robust retrieval.

    Attributes:
        collection_name: Target Qdrant collection name.
        client: Connected Qdrant client instance.
        top_k: Default number of results to return per search.
    """

    def __init__(
        self,
        host: str,
        port: int,
        top_k: int,
        collection_name: str,
    ) -> None:
        """Initialize Qdrant vector store connection.

        Args:
            host: Qdrant server hostname.
            port: Qdrant server port.
            top_k: Default number of top results to return.
            collection_name: Name of the collection for all operations.
        """
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.top_k = top_k

    def create_collection(self, dimension: int) -> None:
        """Create or validate the target collection.

        If collection doesn't exist, creates it with cosine distance metrics
        for both dense and sparse vectors. If it exists, validates that
        vector dimensions match expectations.

        Args:
            dimension: Expected dense embedding dimensionality.

        Raises:
            ValueError: If collection exists with mismatched dimension.
        """
        # Skip if collection already exists
        if self.client.collection_exists(self.collection_name):
            # Validate existing collection configuration
            existing_collection = self.client.get_collection(
                self.collection_name
            )
            vectors_config = existing_collection.config.params.vectors

            # Extract dense vector dimension from config
            dimension_in_store = None
            if isinstance(vectors_config, dict):
                if "dense" in vectors_config:
                    dimension_in_store = vectors_config["dense"].size
            elif hasattr(vectors_config, "size"):
                dimension_in_store = vectors_config.size

            # Validate dimension matches
            if (
                dimension_in_store is not None
                and dimension_in_store != dimension
            ):
                raise ValueError(
                    f"Collection '{self.collection_name}' exists with "
                    f"dimension={dimension_in_store}, expected {dimension}"
                )
            return

        # Create new collection with both dense and sparse vector configs
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            },
        )

    def upsert(self, points: list[VectorPoint]) -> None:
        """Write or replace vector points in the collection.

        Converts internal VectorPoint objects to Qdrant PointStruct format,
        preserving both dense and sparse vectors plus chunk metadata payloads.

        Args:
            points: Vector points with embeddings and chunk metadata.
        """
        if not points:
            return

        # Convert internal VectorPoint to Qdrant PointStruct format
        qdrant_points = [
            PointStruct(
                id=point.id,
                vector={
                    "dense": point.vectorEmbedding,
                    "sparse": {
                        "indices": point.SparseEmbedding.indices,
                        "values": point.SparseEmbedding.values,
                    },
                },
                payload=point.payload.model_dump(),
            )
            for point in points
        ]

        # Upsert points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=qdrant_points,
        )

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_points: list,
        sparse_points: list,
        top_k: int,
        rrf_k: int = 60,
    ) -> list:
        """Combine dense and sparse search rankings using reciprocal rank fusion.

        RRF assigns scores based on position in each ranking rather than
        raw similarity scores, making it robust to different scale between
        dense and sparse representations.

        Formula: RRF(d) = 1 / (k + rank(d))

        Args:
            dense_points: Points from dense vector search.
            sparse_points: Points from sparse vector search.
            top_k: Number of top results to return after fusion.
            rrf_k: RRF constant (default 60 is commonly used).

        Returns:
            Top-k fused results ordered by combined score.
        """
        # Initialize fusion scores tracking
        fused_scores: dict[str, dict] = {}

        def apply_rrf_scores(points: list) -> None:
            """Apply RRF formula and accumulate scores for each point."""
            for rank, point in enumerate(points, start=1):
                # RRF score for this ranking position
                rrf_score = 1.0 / (rrf_k + rank)

                # Get or create entry for this point
                if point.id in fused_scores:
                    fused_scores[point.id]["score"] += rrf_score
                else:
                    fused_scores[point.id] = {
                        "point": point,
                        "score": rrf_score,
                    }

        # Apply RRF scores from both dense and sparse rankings
        apply_rrf_scores(dense_points)
        apply_rrf_scores(sparse_points)

        # Sort by combined score and return top-k
        sorted_results = sorted(
            fused_scores.values(),
            key=lambda x: x["score"],
            reverse=True,
        )
        return [item["point"] for item in sorted_results[:top_k]]

    def search(
        self,
        dense_query_vector: list[float],
        sparse_query_vector: SparseEmbedding,
    ) -> list[SearchResult]:
        """Search for chunks matching query using hybrid (dense + sparse) search.

        Performs two parallel searches (dense and sparse) and fuses results
        using reciprocal rank fusion for robust ranking.

        Args:
            dense_query_vector: Query as dense embedding vector.
            sparse_query_vector: Query as sparse embedding.

        Returns:
            List of SearchResult objects ordered by relevance.

        Raises:
            ValueError: If vectors are empty or top_k invalid.
        """
        # Validate inputs
        if not dense_query_vector:
            raise ValueError("Dense query vector must not be empty.")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Query dense index
        dense_search_response = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_query_vector,
            using="dense",
            limit=self.top_k,
        )

        # Query sparse index
        sparse_search_response = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(
                indices=sparse_query_vector.indices,
                values=sparse_query_vector.values,
            ),
            using="sparse",
            limit=self.top_k,
        )

        # Fuse rankings from both searches
        fused_points = self._reciprocal_rank_fusion(
            dense_search_response.points,
            sparse_search_response.points,
            top_k=self.top_k,
        )

        # Convert Qdrant points to SearchResult objects
        results: list[SearchResult] = []
        for point in fused_points:
            # Validate and map payload to Chunk domain model
            try:
                chunk = Chunk.model_validate(point.payload)
            except ValidationError as exc:
                raise ValueError(
                    f"Qdrant point '{point.id}' has invalid chunk payload"
                ) from exc

            # Convert to SearchResult
            results.append(
                SearchResult.from_chunk_match(chunk, point.score)
            )

        return results

    def get_chunks_by_range(
        self,
        document_id: str,
        start_chunk_index: int,
        end_chunk_index: int,
    ) -> list[SearchResult]:
        """Retrieve chunks for a document within a chunk index range.

        Used by context expanders to fetch neighboring chunks of retrieved
        results. Queries Qdrant with a range filter on chunk_index.

        Args:
            document_id: Source document identifier.
            start_chunk_index: Starting chunk index (inclusive).
            end_chunk_index: Ending chunk index (exclusive).

        Returns:
            List of SearchResult objects in the specified range.
        """
        # Build filter for document + chunk index range
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="source_file",
                    match=MatchValue(value=document_id),
                ),
                FieldCondition(
                    key="chunk_index",
                    range=Range(
                        gte=start_chunk_index,
                        lt=end_chunk_index,
                    ),
                ),
            ]
        )

        # Scroll through matching points
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=filter_condition,
            limit=self.top_k,
        )

        # Convert records to SearchResult objects
        results: list[SearchResult] = []
        for record in records:
            # Validate and map payload to Chunk domain model
            try:
                chunk = Chunk.model_validate(record.payload)
            except ValidationError as exc:
                raise ValueError(
                    f"Qdrant record '{record.id}' has invalid chunk payload"
                ) from exc

            # Convert to SearchResult with neutral score
            results.append(
                SearchResult.from_chunk_match(chunk, score=1.0)
            )

        return results

    def delete_document(self, source_file: str) -> None:
        """Delete all chunks belonging to a document.

        Args:
            source_file: Document identifier to delete.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_file",
                            match=MatchValue(value=source_file),
                        )
                    ]
                )
            ),
        )
