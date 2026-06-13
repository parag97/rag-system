"""Result and metrics models for document ingestion."""

from pydantic import BaseModel, Field


class StageMetrics(BaseModel):
    """Performance metrics for a single ingestion stage.

    Tracks duration, unit count, and throughput for one pipeline step
    to help identify bottlenecks.

    Attributes:
        duration_seconds: Wall-clock time elapsed.
        unit_count: Number of items processed.
        items_per_second: Throughput metric (may be 0 if duration is 0).
    """

    duration_seconds: float = Field(ge=0)
    unit_count: int = Field(ge=0)
    items_per_second: float = Field(ge=0)

    @classmethod
    def from_duration(cls, duration_seconds: float, unit_count: int) -> "StageMetrics":
        """Build consistent metrics from a measured duration and count.

        Computes throughput safely, avoiding division by zero.
        """
        return cls(
            duration_seconds=duration_seconds,
            unit_count=unit_count,
            items_per_second=(
                unit_count / duration_seconds if duration_seconds > 0 else 0.0
            ),
        )


class IngestionMetrics(BaseModel):
    """Performance metrics for the ingestion pipeline.

    Attributes:
        load: Metrics for the document loading stage.
        chunking: Metrics for the chunking stage.
        embedding: Metrics for the embedding stage.
        upsert: Metrics for the vector store upsert.
        total_duration_seconds: End-to-end wall-clock time.
    """

    load: StageMetrics
    chunking: StageMetrics
    embedding: StageMetrics
    upsert: StageMetrics
    total_duration_seconds: float = Field(ge=0)


class IngestionResult(BaseModel):
    """Summary of a document ingestion run.

    Captures high-level counts and metrics for a single source document
    processed through the ingestion pipeline. Used for logging and monitoring.

    Attributes:
        document_id: Stable identifier of the ingested document.
        source_file: Path or name of the source file.
        page_count: Number of pages extracted.
        chunk_count: Total chunks produced.
        embedding_count: Embeddings created.
        vector_count: Vectors persisted to the store.
        metrics: Detailed `IngestionMetrics` breakdown.
    """

    document_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    page_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    embedding_count: int = Field(ge=0)
    vector_count: int = Field(ge=0)
    metrics: IngestionMetrics
