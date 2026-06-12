"""Data models for vector store points."""

from pydantic import BaseModel, Field

from src.core.chunk import Chunk
from src.embeddings.model import SparseEmbedding


class VectorPoint(BaseModel):
    """A vector and validated chunk payload to persist."""

    id: str = Field(min_length=1)
    vectorEmbedding: list[float] = Field(min_length=1)
    SparseEmbedding: SparseEmbedding 
    payload: Chunk
