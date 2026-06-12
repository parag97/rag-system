"""Manual smoke script: run an end-to-end retrieval query via Retriever."""

from src.core.config import load_config
from src.embeddings.sentence_transformer_service import SentenceTransformerEmbeddingService
from src.embeddings.bm25_service import BM25EmbeddingService
from src.retrieval.retriever import Retriever
from src.vectordb.qdrant_store import QdrantVectorStore

config = load_config()

embedder = SentenceTransformerEmbeddingService(
    model_name=config.embedding.vector_model_name,
)
sparse = BM25EmbeddingService(
    model_name=config.embedding.sparse_model_name
)

vector_store = QdrantVectorStore(
    host=config.qdrant.host,
    port=config.qdrant.port,
    collection_name=config.qdrant.collection_name,
)

retriever = Retriever(
    dense_embedding_service=embedder,
    sparse_embeddig_service=sparse,
    vector_store=vector_store,
)
query = "what is the leave policy?"
results = retriever.retrieve(
    query,
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
