from pathlib import Path
import logging
import time

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

logger = logging.getLogger(__name__)


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
        file_ready_max_retries: int = 30,
        file_ready_retry_delay_seconds: int = 1,
    ) -> None:
        self._ingestion_service = ingestion_service
        self._vector_store = vector_store
        self._registry = registry
        self._file_ready_max_retries = file_ready_max_retries
        self._file_ready_retry_delay_seconds = file_ready_retry_delay_seconds

    def _wait_until_ready(
        self,
        file_path: Path,
    ) -> None:
        """
        Wait until a file becomes readable.

        Windows may emit filesystem events before
        file writes have completed.
        """
        for attempt in range(1, self._file_ready_max_retries + 1):
            try:
                with open(file_path, "rb"):
                    logger.info(
                        "File ready: %s",
                        file_path,
                    )
                    return
            except PermissionError:
                logger.warning(
                    "File locked. Retry %d/%d for %s",
                    attempt,
                    self._file_ready_max_retries,
                    file_path,
                )
                if attempt == self._file_ready_max_retries:
                    raise RuntimeError(
                        f"Timed out waiting for file: {file_path}"
                    )
                time.sleep(
                    self._file_ready_retry_delay_seconds
                )
            except FileNotFoundError:
                raise

    def sync_document(
        self,
        file_path: str | Path,
    ) -> None:
        file_path = Path(file_path)
        
        # Wait until file is fully written and accessible
        self._wait_until_ready(file_path)

        source_path = str(
            file_path.resolve()
        )
        file_hash = compute_file_hash(
            file_path
        )
        existing = self._registry.get(
            source_path
        )

        # Handle new file
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
            logger.info(
                "Indexed new document: %s",
                source_path,
            )
            return

        # Handle unchanged file
        if existing.file_hash == file_hash:
            logger.info(
                "Skipping unchanged file: %s",
                source_path,
            )
            return

        # Handle modified file
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
        logger.info(
            "Reindexed modified file: %s",
            source_path,
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
        logger.info(
            "Removed document: %s",
            source_path,
        )