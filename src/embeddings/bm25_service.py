"""FastEmbed-backed BM25 sparse encoder."""

from fastembed import SparseTextEmbedding
from fastembed import SparseEmbedding as FastEmbedSparseEmbedding

from src.embeddings.model import SparseEmbedding
from src.embeddings.base import SparseEmbeddingService


class BM25EmbeddingService(SparseEmbeddingService):
    """Encode documents and queries into BM25 sparse vectors."""

    def __init__(self, model_name: str) -> None:
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
        return SparseEmbedding(
            indices=embedding.indices.tolist(),
            values=embedding.values.tolist(),
        )
