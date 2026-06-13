"""Interfaces and models for sparse text embeddings."""

from pydantic import BaseModel, model_validator


class SparseEmbedding(BaseModel):
    """Sparse vector represented by matching term indices and weights.

    This model enforces that indices and values are parallel lists of
    equal length, making the representation unambiguous.

    Attributes:
        indices: Integer indices into the vocabulary or feature space.
        values: Floating-point weights corresponding to each index.
    """

    indices: list[int]
    values: list[float]

    @model_validator(mode="after")
    def validate_lengths(self) -> "SparseEmbedding":
        # Ensure the sparse representation is well-formed.
        if len(self.indices) != len(self.values):
            raise ValueError("Sparse embedding indices and values must have equal length.")
        return self


