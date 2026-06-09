from src.core.config import load_config
from src.embeddings.sentence_transformer_service import (
    SentenceTransformerEmbeddingService,
)
from src.retrieval.retriever import Retriever
from src.vectordb.qdrant_store import (
    QdrantVectorStore,
)

config = load_config()

embedder = SentenceTransformerEmbeddingService(
    model_name=config.embedding.model_name,
)

vector_store = QdrantVectorStore(
    host=config.qdrant.host,
    port=config.qdrant.port,
    collection_name=config.qdrant.collection_name,
)

retriever = Retriever(
    embedding_service=embedder,
    vector_store=vector_store,
)

results = retriever.retrieve(
    query="What is the maternity leave policy?",
    top_k=3,
)

for result in results:

    print("=" * 80)

    print(f"Score: {result.score:.4f}")

    print(f"Chunk ID: {result.chunk_id}")

    print(f"Source: {result.source_file}")

    print()

    print(result.text)

    print()