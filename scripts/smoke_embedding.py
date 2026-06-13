"""Manual smoke test script for the configured embedding model.

Loads the configured sentence-transformers model and tests embedding
of sample texts and a query.
"""

from src.core.config import load_config
from src.embeddings.sentence_transformer_service import (
    SentenceTransformerEmbeddingService,
)


def main() -> None:
    """Load config and test embedding dimensions and outputs."""
    config = load_config()
    embedder = SentenceTransformerEmbeddingService(config.embedding.vector_model_name)
    print(embedder.dimension)
    print(embedder.embed_documents(["Hello, world!"]))
    print(embedder.embed_query("Hello, world!"))


if __name__ == "__main__":
    main()
