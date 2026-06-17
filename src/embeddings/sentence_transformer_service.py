"""Sentence-Transformers dense embedding service."""

from sentence_transformers import SentenceTransformer

from src.embeddings.base import DenseEmbeddingService


class SentenceTransformerEmbeddingService(DenseEmbeddingService):
    """Dense vector embeddings using Hugging Face Sentence-Transformers.

    This adapter loads a Sentence-Transformers model (e.g., 'all-MiniLM-L6-v2')
    from Hugging Face or disk and exposes the minimal DenseEmbeddingService API.

    The model is loaded once at initialization and reused for all embedding calls.

    Attributes:
        model: Loaded SentenceTransformer model instance.
    """

    def __init__(self, model_name: str) -> None:
        """Load a Sentence-Transformers model.

        Args:
            model_name: Model ID or path (e.g., 'sentence-transformers/all-MiniLM-L6-v2').

        Raises:
            EnvironmentError: If model cannot be downloaded or loaded.
        """
        self.model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Embedding vector dimensionality for this model.

        Returns:
            Integer dimension of output vectors.
        """
        return self.model.get_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple document texts into dense vectors.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Encode a single query string into a dense vector.

        Args:
            query: Natural-language query to embed.

        Returns:
            Single embedding vector as a Python float list.
        """
        embedding = self.model.encode([query], convert_to_numpy=True)[0]
        return embedding.tolist()

