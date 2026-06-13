"""Abstract interface for loading source documents."""

from abc import ABC, abstractmethod
from pathlib import Path

from src.ingestion.document import Document


class DocumentLoader(ABC):
    """Protocol implemented by format-specific document loaders.

    Subclasses should handle file parsing and produce `Document` instances
    with consistent page and text extraction across different formats.
    """

    @abstractmethod
    def load(self, source_path: str | Path) -> Document:
        """Load a source file into the internal document representation.

        Args:
            source_path: Path to the source file (PDF, text, etc.).

        Returns:
            A `Document` with extracted pages and metadata.
        """
        pass
