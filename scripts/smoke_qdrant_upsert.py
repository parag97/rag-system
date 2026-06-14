"""Manual smoke test script for embedding and upserting one chunk.

Tests that a single chunk can be embedded with both dense and sparse
embeddings and successfully upserted to Qdrant.
"""

from src.core.chunk import Chunk
from src.core.config import load_config
from src.embeddings.sentence_transformer_service import SentenceTransformerEmbeddingService
from src.embeddings.bm25_service import BM25EmbeddingService
from src.vectordb.models import VectorPoint
from src.vectordb.qdrant_store import QdrantVectorStore


def main() -> None:
    """Load config, create embeddings, and upsert a test chunk."""
    config = load_config()
    VectorEmbedder = SentenceTransformerEmbeddingService(config.embedding.vector_model_name)
    SparseEmbedder = BM25EmbeddingService(config.embedding.sparse_model_name)
    store = QdrantVectorStore(
        host=config.qdrant.host,
        port=config.qdrant.port,
        collection_name=config.qdrant.collection_name,
        top_k=config.qdrant.top_k,
    )

    chunk = Chunk(
        document_id="employee_handbook",
        chunk_index=1,
        text="Employees are entitled to 26 weeks of paid maternity leave.",
        source_file="employee_handbook.pdf",
        start_page=1,
        end_page=1,
        section_title="Maternity Leave",
    )
    point = VectorPoint(
        id=chunk.vector_id,
        vectorEmbedding=VectorEmbedder.embed_documents([chunk.text])[0],
        SparseEmbedding=SparseEmbedder.embed_documents([chunk.text])[0],
        payload=chunk,
    )
    print(point.id)
    store.upsert([point])


if __name__ == "__main__":
    main()
