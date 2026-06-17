import logging
from pathlib import Path

from watchdog.observers import Observer

from src.ingestion.document_sync_service import (
    DocumentSyncService,
)
from src.ingestion.file_watcher import (
    DocumentEventHandler,
)

logger = logging.getLogger(__name__)


class WatcherService:
    """
    Watches a folder and keeps Qdrant synchronized with files on disk.
    """

    def __init__(
        self,
        watch_folder: str,
        sync_service: DocumentSyncService,
    ) -> None:
        self._watch_folder = watch_folder
        self._sync_service = sync_service
        self._observer = Observer()

    def start(self) -> None:
        """
        Initial sync + start filesystem watcher.
        """

        logger.info(
            "Running initial sync for %s",
            self._watch_folder,
        )

        for pdf_file in Path(
            self._watch_folder
        ).rglob("*.pdf"):
            self._sync_service.sync_document(
                pdf_file
            )

        handler = DocumentEventHandler(
            self._sync_service
        )

        self._observer.schedule(
            handler,
            self._watch_folder,
            recursive=True,
        )

        self._observer.start()

        logger.info(
            "Started watcher for %s",
            self._watch_folder,
        )

    def stop(self) -> None:
        """
        Gracefully stop watcher.
        """

        self._observer.stop()
        self._observer.join()

        logger.info(
            "Stopped watcher."
        )