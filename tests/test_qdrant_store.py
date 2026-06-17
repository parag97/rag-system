"""Unit tests for the Qdrant adapter boundary.

Validates that the `QdrantVectorStore` correctly handles schema validation,
payload conversion, and search result mapping using a fake Qdrant client.
"""

from types import SimpleNamespace

import pytest
from qdrant_client.models import Distance, VectorParams

from src.core.chunk import Chunk
from src.embeddings.model import SparseEmbedding
from src.vectordb.qdrant_store import QdrantVectorStore


class FakeClient:
    """Mock Qdrant client for unit testing the adapter."""

    def __init__(self) -> None:
        self.exists = True
        self.vector_params = VectorParams(size=2, distance=Distance.COSINE)
        self.created = False
        self.points = []

    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        return self.exists

    def create_collection(self, **kwargs) -> None:
        """Create a collection."""
        self.created = True

    def get_collection(self, collection_name: str):
        """Retrieve collection metadata."""
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=self.vector_params)
            )
        )

    def query_points(self, **kwargs):
        """Simulate a point query response."""
        return SimpleNamespace(points=self.points)


def make_store(client: FakeClient) -> QdrantVectorStore:
    """Create a test store with a mocked Qdrant client."""
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.collection_name = "documents"
    store.client = client
    # tests expect a default top_k to exist on the store
    store.top_k = 10
    return store


def test_create_collection_validates_existing_schema():
    client = FakeClient()
    client.vector_params = VectorParams(size=3, distance=Distance.COSINE)

    with pytest.raises(ValueError, match="expected 2"):
        make_store(client).create_collection(dimension=2)


def test_search_validates_payload_before_mapping_result():
    client = FakeClient()
    client.points = [
        SimpleNamespace(id="bad-point", score=0.5, payload={"text": "missing fields"})
    ]

    with pytest.raises(ValueError, match="bad-point"):
        make_store(client).search([1.0, 0.0], SparseEmbedding(indices=[0], values=[0.0]))


def test_search_maps_valid_chunk_payload():
    client = FakeClient()
    chunk = Chunk(
        document_id="doc",
        chunk_index=0,
        text="hello",
        source_file="source.pdf",
    )
    client.points = [
        SimpleNamespace(id=chunk.vector_id, score=0.8, payload=chunk.model_dump())
    ]

    result = make_store(client).search([1.0, 0.0], SparseEmbedding(indices=[0], values=[0.0]))

    assert result[0].chunk_id == chunk.chunk_id


def test_search_fuses_dense_and_sparse_results_with_rrf():
    class FusionClient(FakeClient):
        def __init__(self) -> None:
            super().__init__()
            self.dense_points = []
            self.sparse_points = []

        def query_points(self, **kwargs):
            if kwargs.get("using") == "dense":
                return SimpleNamespace(points=self.dense_points)
            return SimpleNamespace(points=self.sparse_points)

    client = FusionClient()
    chunk_a = Chunk(
        document_id="doc",
        chunk_index=0,
        text="alpha",
        source_file="source.pdf",
    )
    chunk_b = Chunk(
        document_id="doc",
        chunk_index=1,
        text="beta",
        source_file="source.pdf",
    )
    chunk_c = Chunk(
        document_id="doc",
        chunk_index=2,
        text="gamma",
        source_file="source.pdf",
    )

    client.dense_points = [
        SimpleNamespace(id=chunk_a.vector_id, score=0.9, payload=chunk_a.model_dump()),
        SimpleNamespace(id=chunk_b.vector_id, score=0.8, payload=chunk_b.model_dump()),
    ]
    client.sparse_points = [
        SimpleNamespace(id=chunk_b.vector_id, score=0.7, payload=chunk_b.model_dump()),
        SimpleNamespace(id=chunk_c.vector_id, score=0.6, payload=chunk_c.model_dump()),
    ]

    result = make_store(client).search([1.0, 0.0], SparseEmbedding(indices=[0], values=[0.0]))

    assert [hit.chunk_id for hit in result] == [chunk_b.chunk_id, chunk_a.chunk_id, chunk_c.chunk_id]
