import pytest
from pathlib import Path
from unittest.mock import MagicMock

from src.ingestion.document_registry import DocumentRegistry, RegistryEntry
from src.ingestion.document_sync_service import DocumentSyncService
from src.ingestion.models import IngestionResult, IngestionMetrics, StageMetrics


class MockVectorStore:
    def __init__(self):
        self.deleted_ids = []

    def delete_document(self, document_id: str) -> None:
        self.deleted_ids.append(document_id)


def test_sync_new_document(tmp_path):
    # Setup temporary file
    test_file = tmp_path / "test.pdf"
    test_file.write_text("dummy PDF content")

    # Mocks
    mock_ingestion = MagicMock()
    mock_ingestion.ingest.return_value = IngestionResult(
        document_id="doc-123",
        source_file=str(test_file),
        page_count=1,
        chunk_count=1,
        embedding_count=1,
        vector_count=1,
        metrics=IngestionMetrics(
            load=StageMetrics.from_duration(0.1, 1),
            chunking=StageMetrics.from_duration(0.1, 1),
            embedding=StageMetrics.from_duration(0.1, 1),
            upsert=StageMetrics.from_duration(0.1, 1),
            total_duration_seconds=0.4,
        ),
    )

    vector_store = MockVectorStore()
    
    registry_file = tmp_path / "registry.json"
    registry = DocumentRegistry(registry_file)

    sync_service = DocumentSyncService(
        ingestion_service=mock_ingestion,
        vector_store=vector_store,
        registry=registry,
    )

    # Act
    sync_service.sync_document(test_file)

    # Assert
    mock_ingestion.ingest.assert_called_once_with(test_file)
    entry = registry.get(str(test_file.resolve()))
    assert entry is not None
    assert entry.document_id == "doc-123"


def test_sync_unchanged_document(tmp_path):
    # Setup temporary file
    test_file = tmp_path / "test.pdf"
    test_file.write_text("dummy PDF content")

    mock_ingestion = MagicMock()
    vector_store = MockVectorStore()
    
    registry_file = tmp_path / "registry.json"
    registry = DocumentRegistry(registry_file)
    
    # Pre-populate registry with the correct hash
    from src.ingestion.hash_utils import compute_file_hash
    file_hash = compute_file_hash(test_file)
    registry.set(
        str(test_file.resolve()),
        RegistryEntry(document_id="doc-123", file_hash=file_hash)
    )

    sync_service = DocumentSyncService(
        ingestion_service=mock_ingestion,
        vector_store=vector_store,
        registry=registry,
    )

    # Act
    sync_service.sync_document(test_file)

    # Assert
    mock_ingestion.ingest.assert_not_called()


def test_sync_modified_document(tmp_path):
    # Setup temporary file
    test_file = tmp_path / "test.pdf"
    test_file.write_text("original PDF content")

    mock_ingestion = MagicMock()
    mock_ingestion.ingest.return_value = IngestionResult(
        document_id="doc-new-123",
        source_file=str(test_file),
        page_count=1,
        chunk_count=1,
        embedding_count=1,
        vector_count=1,
        metrics=IngestionMetrics(
            load=StageMetrics.from_duration(0.1, 1),
            chunking=StageMetrics.from_duration(0.1, 1),
            embedding=StageMetrics.from_duration(0.1, 1),
            upsert=StageMetrics.from_duration(0.1, 1),
            total_duration_seconds=0.4,
        ),
    )

    vector_store = MockVectorStore()
    
    registry_file = tmp_path / "registry.json"
    registry = DocumentRegistry(registry_file)
    
    # Pre-populate registry with an old/different hash
    registry.set(
        str(test_file.resolve()),
        RegistryEntry(document_id="doc-old-123", file_hash="different-hash")
    )

    # Modify the file so we check new content hash
    test_file.write_text("modified PDF content")

    sync_service = DocumentSyncService(
        ingestion_service=mock_ingestion,
        vector_store=vector_store,
        registry=registry,
    )

    # Act
    sync_service.sync_document(test_file)

    # Assert
    assert "doc-old-123" in vector_store.deleted_ids
    mock_ingestion.ingest.assert_called_once_with(test_file)
    entry = registry.get(str(test_file.resolve()))
    assert entry is not None
    assert entry.document_id == "doc-new-123"


def test_delete_document(tmp_path):
    test_file = tmp_path / "test.pdf"

    mock_ingestion = MagicMock()
    vector_store = MockVectorStore()
    
    registry_file = tmp_path / "registry.json"
    registry = DocumentRegistry(registry_file)
    
    # Pre-populate registry
    registry.set(
        str(test_file.resolve()),
        RegistryEntry(document_id="doc-123", file_hash="some-hash")
    )

    sync_service = DocumentSyncService(
        ingestion_service=mock_ingestion,
        vector_store=vector_store,
        registry=registry,
    )

    # Act
    sync_service.delete_document(test_file)

    # Assert
    assert "doc-123" in vector_store.deleted_ids
    assert registry.get(str(test_file.resolve())) is None
