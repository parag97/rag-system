"""Qdrant-backed implementation of :class:`~src.vectordb.base.VectorStore`."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, SparseVectorParams, SparseVector
from pydantic import ValidationError

from src.core.chunk import Chunk
from src.core.search import SearchResult
from src.embeddings.model import SparseEmbedding

from src.vectordb.base import VectorStore
from src.vectordb.models import VectorPoint


class QdrantVectorStore(VectorStore):
    """Store and query document embeddings in a Qdrant collection.

    Collections use cosine distance. Chunk metadata from each
    :class:`~src.vectordb.models.VectorPoint` is serialized into the Qdrant
    point payload and mapped back to :class:`~src.core.search.SearchResult` on
    search.
    """

    def __init__(self, host: str, port: int, collection_name: str) -> None:
        """Connect to a Qdrant instance.

        Args:
            host: Qdrant server hostname.
            port: Qdrant server port.
            collection_name: Collection used for all operations on this store.
        """
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, dimension: int) -> None:
        """Create the configured collection with cosine vector similarity.

        Args:
            dimension: Embedding vector length for stored points.
        """
        if not self.client.collection_exists(self.collection_name):
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
            return

        # Collection already exists. Validate the existing configuration schema.
        existing_info = self.client.get_collection(self.collection_name)
        vectors_config = existing_info.config.params.vectors

        size = None
        if isinstance(vectors_config, dict):
            if "dense" in vectors_config:
                size = vectors_config["dense"].size
        elif hasattr(vectors_config, "size"):
            size = vectors_config.size  # type: ignore

        if size is not None and size != dimension:
            raise ValueError(
                f"Collection '{self.collection_name}' already exists but has unexpected size={size} (expected size={dimension})."
            )


    def upsert(self, points: list[VectorPoint]) -> None:
        """Write or replace points in the collection.

        Args:
            points: Vector points with chunk payloads to persist.
        """
        if not points:
            return

        # Convert our internal VectorPoint objects into Qdrant PointStructs.
        # Keep dense and sparse vectors in the expected nested mapping.
        qdrant_point = [
            PointStruct(
                id=point.id,
                vector={
                    "dense": point.vectorEmbedding,
                    "sparse": {
                        "indices": point.SparseEmbedding.indices,
                        "values": point.SparseEmbedding.values,
                    },
                }, # type: ignore
                payload=point.payload.model_dump(),
            )
            for point in points
        ]

        self.client.upsert(collection_name=self.collection_name, points=qdrant_point)

    @staticmethod
    def _reciprocal_rank_fusion(
        dense_points: list,
        sparse_points: list,
        top_k: int,
        rrf_k: int = 60,
    ) -> list:
        """Fuse dense and sparse rankings using Reciprocal Rank Fusion.

        Points are ranked by their position in each list rather than by raw
        similarity score, which is useful for combining dense and sparse
        retrieval signals.
        """
        fused_scores: dict[str, dict] = {}

        def update_rank(points: list) -> None:
            for rank, point in enumerate(points, start=1):
                score = 1.0 / (rrf_k + rank)
                entry = fused_scores.get(point.id)
                if entry is None:
                    fused_scores[point.id] = {
                        "point": point,
                        "score": score,
                    }
                else:
                    entry["score"] += score

        update_rank(dense_points)
        print(f"Fused scores after dense update: {fused_scores}")
        update_rank(sparse_points)
        print()
        print(f"Fused scores after sparse update: {fused_scores}")

        fused = sorted(
            fused_scores.values(),
            key=lambda item: item["score"],
            reverse=True,
        )
        return [item["point"] for item in fused[:top_k]]

    def search(
        self,
        dense_query_vector: list[float],
        sparse_query_vector: SparseEmbedding,
        top_k: int,
    ) -> list[SearchResult]:
        """Return the top matching chunks for a query embedding.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of nearest neighbors to return.

        Returns:
            Hits ordered by similarity score, highest first.
        """
        if not dense_query_vector:
            raise ValueError("query_vector must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Query dense and sparse indexes separately and combine the raw
        # point lists. Downstream we validate payloads before mapping.
        resp_dense = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_query_vector,
            using="dense",
            limit=top_k,
        )

        resp_sparse = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(
                indices=sparse_query_vector.indices,
                values=sparse_query_vector.values,
            ),
            using="sparse",
            limit=top_k,
        )

        dense_resp_points = resp_dense.points
        sparse_resp_points = resp_sparse.points

        fused_resp_points = self._reciprocal_rank_fusion(
            dense_resp_points,
            sparse_resp_points,
            top_k=top_k,
        )

        result: list[SearchResult] = []

        for each_point in fused_resp_points:
            # Validate payload structure before mapping to domain model.
            try:
                chunk = Chunk.model_validate(each_point.payload)
            except ValidationError as exc:
                raise ValueError(
                    f"Qdrant point '{each_point.id}' has an invalid chunk payload."
                ) from exc
            result.append(SearchResult.from_chunk_match(chunk, each_point.score))

        return result
