"""Manual smoke script for creating the configured Qdrant collection."""

from src.core.config import load_config

from src.embeddings.sentence_transformer_service import SentenceTransformerEmbeddingService

from src.vectordb.qdrant_store import QdrantVectorStore


def main() -> None:
    config = load_config()
    embedder = SentenceTransformerEmbeddingService(config.embedding.vector_model_name)
    store = QdrantVectorStore(
        host=config.qdrant.host,
        port=config.qdrant.port,
        collection_name=config.qdrant.collection_name,
    )
    store.create_collection(dimension=embedder.dimension)
    print("Collection created")


if __name__ == "__main__":
    main()
