"""Interfaces and models for sparse text embeddings."""

from pydantic import BaseModel, model_validator


class SparseEmbedding(BaseModel):
    """Sparse vector represented by matching term indices and weights."""

    indices: list[int]
    values: list[float]

    @model_validator(mode="after")
    def validate_lengths(self) -> "SparseEmbedding":
        if len(self.indices) != len(self.values):
            raise ValueError("Sparse embedding indices and values must have equal length.")
        return self


