"""Data models describing vector store payloads.

This module defines `VectorPoint`, the schema used when upserting
vectors and their associated chunk payloads into the vector database.
The model mirrors the shape expected by the Qdrant adapter in this
project and intentionally keeps field names compatible with the
rest of the codebase.
"""

from pydantic import BaseModel, Field

from src.core.chunk import Chunk
from src.embeddings.model import SparseEmbedding


class VectorPoint(BaseModel):
    """A vector and validated chunk payload to persist.

    Attributes:
        id: Unique identifier used as the point key in the vector store.
        vectorEmbedding: Dense embedding vector for nearest-neighbour search.
        SparseEmbedding: Sparse embedding used for BM25-like retrieval.
        payload: The original `Chunk` object serialized into the point payload.
    """

    id: str = Field(min_length=1)
    vectorEmbedding: list[float] = Field(min_length=1)
    SparseEmbedding: SparseEmbedding
    payload: Chunk
