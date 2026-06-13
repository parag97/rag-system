"""Tests for the FastEmbed-backed BM25 sparse encoder.

Validates that the `BM25EmbeddingService` correctly converts FastEmbed
sparse embeddings into our internal `SparseEmbedding` model.
"""

import numpy as np
import pytest
from fastembed import SparseEmbedding as FastEmbedSparseEmbedding

from src.embeddings.bm25_service import BM25EmbeddingService
from src.embeddings.model import SparseEmbedding


class FakeFastEmbedModel:
    """Mock FastEmbed model for testing."""

    def embed(self, texts: list[str]):
        """Yield fake sparse embeddings for each input text."""
        for index, _ in enumerate(texts):
            yield FastEmbedSparseEmbedding(
                indices=np.array([index, index + 1]),
                values=np.array([1.0, 0.5]),
            )

    def query_embed(self, query: str):
        """Yield a fake sparse embedding for a query."""
        yield FastEmbedSparseEmbedding(
            indices=np.array([7, 9]),
            values=np.array([0.8, 0.2]),
        )


def make_encoder() -> BM25EmbeddingService:
    """Create a test encoder with a mocked FastEmbed model."""
    encoder = BM25EmbeddingService.__new__(BM25EmbeddingService)
    encoder.model = FakeFastEmbedModel()
    return encoder


def test_embed_documents_converts_fastembed_sparse_vectors():
    embeddings = make_encoder().embed_documents(["first", "second"])

    assert embeddings == [
        SparseEmbedding(indices=[0, 1], values=[1.0, 0.5]),
        SparseEmbedding(indices=[1, 2], values=[1.0, 0.5]),
    ]


def test_embed_query_uses_query_embedding_api():
    embedding = make_encoder().embed_query("question")

    assert embedding == SparseEmbedding(indices=[7, 9], values=[0.8, 0.2])


def test_sparse_embedding_requires_matching_indices_and_values():
    with pytest.raises(ValueError, match="equal length"):
        SparseEmbedding(indices=[1, 2], values=[0.5])
