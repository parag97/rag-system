from src.core.config import load_config
from src.embeddings.sentence_transformer_service import (
    SentenceTransformerEmbeddingService,
)
from src.vectordb.qdrant_store import (
    QdrantVectorStore,
)

config = load_config()

embedder = SentenceTransformerEmbeddingService(
    model_name=config.embedding.model_name,
)

store = QdrantVectorStore(
    host=config.qdrant.host,
    port=config.qdrant.port,
    collection_name=config.qdrant.collection_name,
)

store.create_collection(
    dimension=embedder.dimension,
)

print("Collection created")