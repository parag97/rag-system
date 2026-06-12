"""Unit tests for service boundaries and deterministic identities."""

from pathlib import Path

import pytest

from src.chunking.recursive_chunker import RecursiveChunker
from src.core.chunk import Chunk
from src.core.config import ChunkingConfig
from src.core.search import SearchResult
from src.embeddings.base import EmbeddingService
from src.ingestion.document import Document, Page
from src.ingestion.document_ingestion_service import DocumentIngestionService
from src.ingestion.loaders.base import DocumentLoader
from src.retrieval.retriever import Retriever
from src.vectordb.base import VectorStore
from src.vectordb.models import VectorPoint


class FakeLoader(DocumentLoader):
    def load(self, source_path: str | Path) -> Document:
        return Document(
            document_id="document-1",
            source_file=str(source_path),
            pages=[Page(page_number=1, text="alpha beta gamma")],
        )


class FakeEmbedder(EmbeddingService):
    document_calls = 0
    query_calls = 0

    @property
    def dimension(self) -> int:
        return 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        self.query_calls += 1
        return [0.0, 1.0]


class FakeVectorStore(VectorStore):
    def __init__(self) -> None:
        self.points: list[VectorPoint] = []

    def create_collection(self, dimension: int) -> None:
        pass

    def upsert(self, points: list[VectorPoint]) -> None:
        self.points = points

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        return []


def test_chunk_vector_id_is_deterministic():
    first = Chunk(
        document_id="doc",
        chunk_index=0,
        text="text",
        source_file="source.pdf",
    )
    second = first.model_copy()

    assert first.vector_id == second.vector_id


def test_ingestion_uses_document_embedding_boundary():
    embedder = FakeEmbedder()
    store = FakeVectorStore()
    service = DocumentIngestionService(
        FakeLoader(),
        RecursiveChunker(chunk_size=100, chunk_overlap=0),
        embedder,
        store,
    )

    first = service.ingest("source.pdf")
    first_id = store.points[0].id
    second = service.ingest("source.pdf")

    assert embedder.document_calls == 2
    assert embedder.query_calls == 0
    assert first.document_id == second.document_id
    assert first_id == store.points[0].id


def test_retriever_uses_query_embedding_boundary():
    embedder = FakeEmbedder()
    Retriever(embedder, FakeVectorStore()).retrieve("question")

    assert embedder.query_calls == 1
    assert embedder.document_calls == 0


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (100, -1), (100, 100)],
)
def test_chunking_config_rejects_invalid_values(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
