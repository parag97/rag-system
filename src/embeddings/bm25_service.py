"""BM25 sparse embedding service via FastEmbed."""

from fastembed import SparseTextEmbedding
from fastembed import SparseEmbedding as FastEmbedSparseEmbedding

from src.embeddings.model import SparseEmbedding
from src.embeddings.base import SparseEmbeddingService


class BM25EmbeddingService(SparseEmbeddingService):
    """Sparse vector embeddings using BM25 via FastEmbed.

    BM25 is a probabilistic ranking function that provides sparse,
    interpretable vectors useful for hybrid search when combined with
    dense embeddings. FastEmbed provides an efficient implementation.

    The service converts FastEmbed's numpy-based SparseEmbedding objects
    into the project's own SparseEmbedding Pydantic model for consistency.

    Attributes:
        model: Loaded FastEmbed SparseTextEmbedding instance.
    """

    def __init__(self, model_name: str) -> None:
        """Load a BM25 sparse text embedding model.

        Args:
            model_name: FastEmbed model ID (e.g., 'Qdrant/bm25').

        Raises:
            EnvironmentError: If model cannot be downloaded or loaded.
        """
        # Load the FastEmbed sparse text embedding model
        self.model = SparseTextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[SparseEmbedding]:
        """Encode multiple documents into sparse BM25 vectors.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of SparseEmbedding objects (one per input text).
        """
        return [
            self._convert(embedding)
            for embedding in self.model.embed(texts)
        ]

    def embed_query(self, query: str) -> SparseEmbedding:
        """Encode a single query into a sparse BM25 vector.

        Args:
            query: Query string to embed.

        Returns:
            SparseEmbedding object for the query.
        """
        # FastEmbed query_embed returns an iterator
        embedding = next(iter(self.model.query_embed(query)))
        return self._convert(embedding)

    @staticmethod
    def _convert(embedding: FastEmbedSparseEmbedding) -> SparseEmbedding:
        """Convert FastEmbed sparse embedding to project model.

        Converts numpy arrays to Python lists for JSON serializability
        and Pydantic validation.

        Args:
            embedding: FastEmbed SparseEmbedding with numpy arrays.

        Returns:
            Project's SparseEmbedding with Python lists.
        """
        return SparseEmbedding(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )

