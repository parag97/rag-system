"""Abstract interface for loading source documents."""

from abc import ABC, abstractmethod
from pathlib import Path

from src.ingestion.document import Document


class DocumentLoader(ABC):
    """Protocol implemented by format-specific document loaders."""

    @abstractmethod
    def load(self, source_path: str | Path) -> Document:
        """Load a source file into the internal document representation."""
        pass
