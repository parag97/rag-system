# test_search.py

from gc import collect
from uuid import uuid4

from src.core.chunk import Chunk
from src.core.config import load_config
from src.embeddings.sentence_transformer_service import (
    SentenceTransformerEmbeddingService,
)
from src.vectordb.models import VectorPoint
from src.vectordb.qdrant_store import QdrantVectorStore


config = load_config()

embedder = SentenceTransformerEmbeddingService(
    model_name=config.embedding.model_name
)

store = QdrantVectorStore(
    host=config.qdrant.host,
    port=config.qdrant.port,
    collection_name=config.qdrant.collection_name,
)

chunk_1 = Chunk(
    document_id="employee_handbook",
    chunk_index=1,
    text="Employees are entitled to 26 weeks of paid maternity leave.",
    source_file="employee_handbook.pdf",
)

chunk_2 = Chunk(
    document_id="employee_handbook",
    chunk_index=2,
    text="Employees receive health insurance benefits.",
    source_file="employee_handbook.pdf",
)

chunk_3 = Chunk(
    document_id="employee_handbook",
    chunk_index=3,
    text="Employees may work remotely two days per week.",
    source_file="employee_handbook.pdf",
)

chunks = [
    chunk_1,
    chunk_2,
    chunk_3,
]

vectors = embedder.embed([chunk.text for chunk in chunks])

points = []

for chunk, vector in zip(chunks, vectors):

    points.append(
        VectorPoint(
            id = chunk.vector_id,
            vector = vector,
            payload = chunk,
            collection_name=config.qdrant.collection_name
        )
    )

store.upsert(points)

query_vector = embedder.embed(
    ["What is the maternity leave policy?"]
)[0]

results = store.search(
    query_vector=query_vector,
    top_k=3,
)

for result in results:
    print("=" * 80)
    print(f"Score: {result.score:.4f}")
    print(result.text)