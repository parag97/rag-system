from src.core.config import load_config

from src.embeddings.sentence_transformer_service import SentenceTransformerEmbeddingService

config = load_config()

embedder = SentenceTransformerEmbeddingService(
    model_name=config.embedding.model_name
)

print(embedder.dimension)
print(embedder.embed(["Hello, world!"]))