"""Manual smoke script for the configured embedding model."""

from src.core.config import load_config
from src.embeddings.sentence_transformer_service import (
    SentenceTransformerEmbeddingService,
)


def main() -> None:
    config = load_config()
    embedder = SentenceTransformerEmbeddingService(config.embedding.model_name)
    print(embedder.dimension)
    print(embedder.embed_documents(["Hello, world!"]))
    print(embedder.embed_query("Hello, world!"))


if __name__ == "__main__":
    main()
