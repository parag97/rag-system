"""Unit tests for service boundaries and deterministic identities.

Validates that chunks produce stable vector IDs and that service
boundaries (document vs query embeddings) are respected.
"""

from pathlib import Path

import pytest

from src.chunking.recursive_chunker import RecursiveChunker
from src.core.chunk import Chunk
from src.core.config import ChunkingConfig
from src.core.search import SearchResult
from src.embeddings.base import DenseEmbeddingService, SparseEmbeddingService
from src.embeddings.model import SparseEmbedding
from src.ingestion.document import Document, Page
from src.ingestion.document_ingestion_service import DocumentIngestionService
from src.ingestion.loaders.base import DocumentLoader
from src.retrieval.retriever import Retriever
from src.vectordb.base import VectorStore
from src.vectordb.models import VectorPoint


class FakeLoader(DocumentLoader):
    """Mock document loader for testing."""

    def load(self, source_path: str | Path) -> Document:
        """Return a test document."""
        return Document(
            document_id="document-1",
            source_file=str(source_path),
            pages=[Page(page_number=1, text="alpha beta gamma")],
        )


class FakeEmbedder(DenseEmbeddingService):
    """Mock embedding service for testing."""

    document_calls = 0
    query_calls = 0

    @property
    def dimension(self) -> int:
        """Return fixed embedding dimension."""
        return 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return fixed document embeddings."""
        self.document_calls += 1
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> list[float]:
        """Return fixed query embedding."""
        self.query_calls += 1
        return [0.0, 1.0]


class FakeSparseEmbedder(SparseEmbeddingService):
    """Mock sparse embedding service for testing."""

    document_calls = 0
    query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[SparseEmbedding]:
        """Return fixed sparse document embeddings."""
        self.document_calls += 1
        return [SparseEmbedding(indices=[0, 1], values=[1.0, 0.5]) for _ in texts]

    def embed_query(self, query: str) -> SparseEmbedding:
        """Return fixed sparse query embedding."""
        self.query_calls += 1
        return SparseEmbedding(indices=[1, 2], values=[0.8, 0.2])


class FakeVectorStore(VectorStore):
    """Mock vector store for testing."""

    def __init__(self) -> None:
        self.points: list[VectorPoint] = []

    def create_collection(self, dimension: int) -> None:
        """Create collection (no-op)."""
        pass

    def upsert(self, points: list[VectorPoint]) -> None:
        """Store points in memory."""
        self.points = points

    def search(
        self,
        dense_query_vector: list[float],
        sparse_query_vector: SparseEmbedding,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Return empty results."""
        return []
    
    def get_chunks_by_range(self, document_id: str, start_chunk_index: int, end_chunk_index: int) -> list[SearchResult]:
        """Return an empty list for range queries in the fake store."""
        return []


class FakeReRanker:
    """Simple re-ranker implementation for tests that returns inputs unchanged."""

    def re_rank(self, search_results: list[SearchResult], query: str) -> list[SearchResult]:
        return search_results


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
    sparse_embedder = FakeSparseEmbedder()
    store = FakeVectorStore()
    service = DocumentIngestionService(
        FakeLoader(),
        RecursiveChunker(chunk_size=100, chunk_overlap=0),
        embedder,
        sparse_embedder,
        store,
    )

    first = service.ingest("source.pdf")
    first_id = store.points[0].id
    second = service.ingest("source.pdf")

    assert embedder.document_calls == 2
    assert embedder.query_calls == 0
    assert sparse_embedder.document_calls == 2
    assert sparse_embedder.query_calls == 0
    assert first.document_id == second.document_id
    assert first_id == store.points[0].id


def test_retriever_uses_query_embedding_boundary():
    embedder = FakeEmbedder()
    sparse_embedder = FakeSparseEmbedder()
    Retriever(embedder, sparse_embedder, FakeVectorStore(), FakeReRanker()).retrieve("question")

    assert embedder.query_calls == 1
    assert embedder.document_calls == 0
    assert sparse_embedder.query_calls == 1
    assert sparse_embedder.document_calls == 0


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (100, -1), (100, 100)],
)
def test_chunking_config_rejects_invalid_values(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
