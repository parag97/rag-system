"""Manual smoke script for embedding and upserting one chunk."""

from src.core.chunk import Chunk
from src.core.config import load_config
from src.embeddings.sentence_transformer_service import SentenceTransformerEmbeddingService
from src.embeddings.bm25_service import BM25EmbeddingService
from src.vectordb.models import VectorPoint
from src.vectordb.qdrant_store import QdrantVectorStore


def main() -> None:
    config = load_config()
    VectorEmbedder = SentenceTransformerEmbeddingService(config.embedding.vector_model_name)
    SparceEmbedder = BM25EmbeddingService(config.embedding.sparce_model_name)
    store = QdrantVectorStore(
        host=config.qdrant.host,
        port=config.qdrant.port,
        collection_name=config.qdrant.collection_name,
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
        vectorEmbedding = VectorEmbedder.embed_documents([chunk.text])[0],
        SparseEmbedding = SparceEmbedder.embed_documents([chunk.text])[0],
        payload=chunk,
    )
    print(point.id)
    store.upsert([point])


if __name__ == "__main__":
    main()
