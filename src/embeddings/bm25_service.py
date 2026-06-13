"""FastEmbed-backed BM25 sparse encoder.

This adapter wraps a `fastembed` sparse encoder and converts its output
into the project's `SparseEmbedding` model. The conversion step ensures
compatibility and enforces list/serializable types for persistence.
"""

from fastembed import SparseTextEmbedding
from fastembed import SparseEmbedding as FastEmbedSparseEmbedding

from src.embeddings.model import SparseEmbedding
from src.embeddings.base import SparseEmbeddingService


class BM25EmbeddingService(SparseEmbeddingService):
    """Encode documents and queries into BM25 sparse vectors.

    The service returns `SparseEmbedding` instances for documents and a
    single `SparseEmbedding` for queries. The underlying `fastembed`
    model yields numpy arrays which we convert to Python lists.
    """

    def __init__(self, model_name: str) -> None:
        # Load the FastEmbed model once at construction.
        self.model = SparseTextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[SparseEmbedding]:
        """Encode document texts for sparse indexing."""
        return [self._convert(embedding) for embedding in self.model.embed(texts)]

    def embed_query(self, query: str) -> SparseEmbedding:
        """Encode one query for sparse retrieval."""
        embedding = next(iter(self.model.query_embed(query)))
        return self._convert(embedding)

    @staticmethod
    def _convert(embedding: FastEmbedSparseEmbedding) -> SparseEmbedding:
        # Convert numpy arrays to lists to satisfy Pydantic models and
        # make the embeddings JSON-serializable if needed.
        return SparseEmbedding(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )
