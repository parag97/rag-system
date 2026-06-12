from pathlib import Path

from fastembed.sparse import sparse_embedding_base
import pytest

from src.chunking.recursive_chunker import RecursiveChunker
from src.core.config import load_config
from src.embeddings import bm25_service
from src.embeddings.sentence_transformer_service import (
    SentenceTransformerEmbeddingService,
)
from src.embeddings.bm25_service import BM25EmbeddingService
from src.ingestion.document_ingestion_service import (
    DocumentIngestionService,
)
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.vectordb.qdrant_store import QdrantVectorStore


PDF_PATH = Path("data/raw/amazon_annual_report.pdf")
pytestmark = pytest.mark.integration


def create_service() -> DocumentIngestionService:
    config = load_config()

    vector_store = QdrantVectorStore(config.qdrant.host, config.qdrant.port, config.qdrant.collection_name)
    VectorEmbedder = SentenceTransformerEmbeddingService(config.embedding.vector_model_name)
    SparceEmbedder = BM25EmbeddingService(config.embedding.sparce_model_name)
    chunker=RecursiveChunker(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
        )
    document_loader=PDFLoader()


    # Ensure the collection exists
    vector_store.create_collection(
        dimension=VectorEmbedder.dimension
    )

    return DocumentIngestionService(
        document_loader,
        chunker,
        VectorEmbedder,
        SparceEmbedder,
        vector_store=vector_store,
    )


def test_ingest_returns_result():
    service = create_service()

    result = service.ingest(PDF_PATH)

    assert result is not None


def test_document_metadata():
    service = create_service()

    result = service.ingest(PDF_PATH)

    assert result.document_id.startswith("amazon_annual_report-")
    assert result.source_file.endswith("amazon_annual_report.pdf")


def test_counts_are_positive():
    service = create_service()

    result = service.ingest(PDF_PATH)

    assert result.page_count > 0
    assert result.chunk_count > 0
    assert result.embedding_count > 0
    assert result.vector_count > 0


def test_embedding_and_vector_counts_match():
    service = create_service()

    result = service.ingest(PDF_PATH)

    assert result.chunk_count == result.embedding_count
    assert result.chunk_count == result.vector_count


def test_stage_timings_are_non_negative():
    service = create_service()

    result = service.ingest(PDF_PATH)

    assert result.metrics.load.duration_seconds >= 0
    assert result.metrics.chunking.duration_seconds >= 0
    assert result.metrics.embedding.duration_seconds >= 0
    assert result.metrics.upsert.duration_seconds >= 0
    assert result.metrics.total_duration_seconds >= 0


def test_stage_item_counts_match():
    service = create_service()

    result = service.ingest(PDF_PATH)

    assert result.metrics.load.unit_count == result.page_count
    assert result.metrics.chunking.unit_count == result.chunk_count
    assert result.metrics.embedding.unit_count == result.embedding_count
    assert result.metrics.upsert.unit_count == result.vector_count


def test_stage_throughput_is_non_negative():
    service = create_service()

    result = service.ingest(PDF_PATH)

    assert result.metrics.load.items_per_second >= 0
    assert result.metrics.chunking.items_per_second >= 0
    assert result.metrics.embedding.items_per_second >= 0
    assert result.metrics.upsert.items_per_second >= 0


def test_total_time_is_at_least_sum_of_major_stages():
    service = create_service()

    result = service.ingest(PDF_PATH)

    measured = (
        result.metrics.load.duration_seconds
        + result.metrics.chunking.duration_seconds
        + result.metrics.embedding.duration_seconds
        + result.metrics.upsert.duration_seconds
    )

    # Small scheduling differences are expected.
    assert result.metrics.total_duration_seconds >= measured


def test_ingestion_is_repeatable():
    service = create_service()

    first = service.ingest(PDF_PATH)
    second = service.ingest(PDF_PATH)

    assert first.page_count == second.page_count
    assert first.chunk_count == second.chunk_count
    assert first.embedding_count == second.embedding_count
    assert first.vector_count == second.vector_count
