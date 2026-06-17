import logging
from pathlib import Path

from watchdog.events import FileSystemEventHandler

from src.ingestion.document_sync_service import (
    DocumentSyncService,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf",
}


class DocumentEventHandler(
    FileSystemEventHandler
):

    def __init__(
        self,
        sync_service: DocumentSyncService,
    ):
        self._sync_service = sync_service

    def _is_supported(
        self,
        path: Path,
    ) -> bool:
        return (
            path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )

    def on_created(self, event):

        if event.is_directory:
            return

        path = Path(event.src_path)

        if not self._is_supported(path):
            return

        logger.info(
            "New file detected: %s",
            path,
        )

        self._sync_service.sync_document(
            path
        )

    def on_modified(self, event):

        if event.is_directory:
            return

        path = Path(event.src_path)

        if not self._is_supported(path):
            return

        logger.info(
            "Modified file detected: %s",
            path,
        )

        self._sync_service.sync_document(
            path
        )

    def on_deleted(self, event):

        if event.is_directory:
            return

        path = Path(event.src_path)

        if not self._is_supported(path):
            return

        logger.info(
            "Deleted file detected: %s",
            path,
        )

        self._sync_service.delete_document(
            path
        )