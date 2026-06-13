"""Abstract chunker interface for splitting documents into chunks.

This module defines the `Chunker` protocol used by the ingestion
pipeline to break `Document` objects into indexable `Chunk` instances.
"""

from abc import ABC, abstractmethod

from src.core.chunk import Chunk
from src.ingestion.document import Document


class Chunker(ABC):
    @abstractmethod
    def create_chunks(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            document: Source document to split.

        Returns:
            A list of `Chunk` objects extracted from the document.
        """
        pass