"""Sentence-transformers implementation of dense embedding service.

This adapter loads a Hugging Face `sentence-transformers` model and
exposes the minimal `DenseEmbeddingService` API used by the rest of
the application. The model is loaded once and reused for all calls.
"""

from sentence_transformers import SentenceTransformer

from src.embeddings.base import DenseEmbeddingService


class SentenceTransformerEmbeddingService(DenseEmbeddingService):
    """Embed text using a Hugging Face ``sentence-transformers`` model.

    The model is loaded once at construction and reused for document and query
    embedding.
    """

    def __init__(self, model_name: str) -> None:
        """Load a sentence-transformers model from Hugging Face or disk.

        Args:
            model_name: Model ID (e.g. ``sentence-transformers/all-MiniLM-L6-v2``)
                or path to a local checkpoint.
        """
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Embedding size for vectors returned by this service."""
        dimensions = self.model.get_embedding_dimension()
        return dimensions # type: ignore

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode document texts into dense vectors.

        Args:
            texts: Strings to embed.

        Returns:
            List of Python float lists, one per input string.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.tolist()
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Encode one retrieval query into a dense vector."""
        embedding = self.model.encode([query], convert_to_numpy=True)[0]
        embedding = embedding.tolist()
        return embedding
