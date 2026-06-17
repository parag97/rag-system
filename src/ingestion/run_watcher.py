import logging
import time

from watchdog.observers import Observer

from src.container.application_container import (
    ApplicationContainer,
)

from src.ingestion.file_watcher import (
    DocumentEventHandler,
)

logging.basicConfig(
    level=logging.INFO,
)

WATCH_FOLDER = "data/raw"


def main():

    container = ApplicationContainer()

    handler = DocumentEventHandler(
        container.document_sync_service
    )

    observer = Observer()

    observer.schedule(
        handler,
        WATCH_FOLDER,
        recursive=True,
    )

    observer.start()

    logging.info(
        "Watching folder: %s",
        WATCH_FOLDER,
    )

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        observer.stop()

    observer.join()


if __name__ == "__main__":
    main()