"""Result and metrics models for document ingestion."""

from pydantic import BaseModel, Field


class StageMetrics(BaseModel):
    """Performance metrics for a single ingestion stage."""

    duration_seconds: float = Field(ge=0)
    unit_count: int = Field(ge=0)
    items_per_second: float = Field(ge=0)

    @classmethod
    def from_duration(cls, duration_seconds: float, unit_count: int) -> "StageMetrics":
        """Build consistent metrics from a measured duration and count."""
        return cls(
            duration_seconds=duration_seconds,
            unit_count=unit_count,
            items_per_second=(
                unit_count / duration_seconds if duration_seconds > 0 else 0.0
            ),
        )


class IngestionMetrics(BaseModel):
    """Performance metrics for the ingestion pipeline."""

    load: StageMetrics
    chunking: StageMetrics
    embedding: StageMetrics
    upsert: StageMetrics
    total_duration_seconds: float = Field(ge=0)


class IngestionResult(BaseModel):
    """Summary of a document ingestion run."""

    document_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    embedding_count: int = Field(ge=0)
    vector_count: int = Field(ge=0)
    metrics: IngestionMetrics
