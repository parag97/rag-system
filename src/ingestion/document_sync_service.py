from pathlib import Path

from src.ingestion.document_registry import (
    DocumentRegistry,
    RegistryEntry,
)
from src.ingestion.hash_utils import (
    compute_file_hash,
)
from src.ingestion.document_ingestion_service import (
    DocumentIngestionService,
)
from src.vectordb.base import VectorStore


class DocumentSyncService:
    """
    Synchronizes filesystem documents with Qdrant.

    Handles:
    - new files
    - modified files
    - deleted files
    """

    def __init__(
        self,
        ingestion_service: DocumentIngestionService,
        vector_store: VectorStore,
        registry: DocumentRegistry,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._vector_store = vector_store
        self._registry = registry

    def sync_document(
        self,
        file_path: str | Path,
    ) -> None:

        file_path = Path(file_path)

        source_path = str(
            file_path.resolve()
        )

        file_hash = compute_file_hash(
            file_path
        )

        existing = self._registry.get(
            source_path
        )

        #
        # NEW FILE
        #
        if existing is None:

            result = (
                self._ingestion_service.ingest(
                    file_path
                )
            )

            self._registry.set(
                source_path,
                RegistryEntry(
                    document_id=result.document_id,
                    file_hash=file_hash,
                ),
            )

            print(
                f"Indexed new document: {source_path}"
            )

            return

        #
        # UNCHANGED FILE
        #
        if existing.file_hash == file_hash:

            print(
                f"Skipping unchanged file: {source_path}"
            )

            return

        #
        # MODIFIED FILE
        #
        self._vector_store.delete_document(
            existing.document_id
        )

        result = (
            self._ingestion_service.ingest(
                file_path
            )
        )

        self._registry.set(
            source_path,
            RegistryEntry(
                document_id=result.document_id,
                file_hash=file_hash,
            ),
        )

        print(
            f"Reindexed modified file: {source_path}"
        )

    def delete_document(
        self,
        file_path: str | Path,
    ) -> None:

        file_path = Path(file_path)

        source_path = str(
            file_path.resolve()
        )

        existing = self._registry.get(
            source_path
        )

        if existing is None:
            return

        self._vector_store.delete_document(
            existing.document_id
        )

        self._registry.remove(
            source_path
        )

        print(
            f"Removed document: {source_path}"
        )