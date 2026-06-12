"""Unit tests for the Qdrant adapter boundary."""

from types import SimpleNamespace

import pytest
from qdrant_client.models import Distance, VectorParams

from src.core.chunk import Chunk
from src.vectordb.qdrant_store import QdrantVectorStore


class FakeClient:
    def __init__(self) -> None:
        self.exists = True
        self.vector_params = VectorParams(size=2, distance=Distance.COSINE)
        self.created = False
        self.points = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def create_collection(self, **kwargs) -> None:
        self.created = True

    def get_collection(self, collection_name: str):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(vectors=self.vector_params)
            )
        )

    def query_points(self, **kwargs):
        return SimpleNamespace(points=self.points)


def make_store(client: FakeClient) -> QdrantVectorStore:
    store = QdrantVectorStore.__new__(QdrantVectorStore)
    store.collection_name = "documents"
    store.client = client
    return store


def test_create_collection_validates_existing_schema():
    client = FakeClient()
    client.vector_params = VectorParams(size=3, distance=Distance.COSINE)

    with pytest.raises(ValueError, match="expected size=2"):
        make_store(client).create_collection(dimension=2)


def test_search_validates_payload_before_mapping_result():
    client = FakeClient()
    client.points = [
        SimpleNamespace(id="bad-point", score=0.5, payload={"text": "missing fields"})
    ]

    with pytest.raises(ValueError, match="bad-point"):
        make_store(client).search([1.0, 0.0], top_k=1)


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

    result = make_store(client).search([1.0, 0.0], top_k=1)

    assert result[0].chunk_id == chunk.chunk_id
