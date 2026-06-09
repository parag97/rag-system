from src.core.config import load_config
from src.embeddings.sentence_transformer_service import (
    SentenceTransformerEmbeddingService,
)
from uuid import uuid4
from src.vectordb.qdrant_store import (
    QdrantVectorStore,
)
from src.core.chunk import Chunk
from src.vectordb.models import VectorPoint

config = load_config()

embedder = SentenceTransformerEmbeddingService(
    model_name=config.embedding.model_name,
)

store = QdrantVectorStore(
    host=config.qdrant.host,
    port=config.qdrant.port,
    collection_name=config.qdrant.collection_name,
)

chunk = Chunk(
    document_id="employee_handbook",
    chunk_index=1,
    text="Employees are entitled to 26 weeks of paid maternity leave.",
    source_file="employee_handbook.pdf"

)   

start_page=1,
end_page=1,
section_title="Maternity Leave"


vector_point = VectorPoint(
    id=chunk.vector_id,
    vector=embedder.embed([chunk.text])[0],
    payload=chunk,
    collection_name=config.qdrant.collection_name
)
print(vector_point.id)

store.upsert([vector_point])